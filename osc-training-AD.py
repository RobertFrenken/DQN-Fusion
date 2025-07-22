import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import torch
import sys
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from models.models import GATWithJK, GraphAutoencoder, GraphAutoencoderNeighborhood
from preprocessing import graph_creation, build_id_mapping_from_normal
from training_utils import PyTorchTrainer, PyTorchDistillationTrainer, DistillationTrainer
from torch_geometric.data import Batch
from sklearn.metrics import confusion_matrix
import random
import matplotlib.pyplot as plt
import numpy as np

from plotting_utils import (
    plot_feature_histograms,
    plot_node_recon_errors,
    plot_graph_reconstruction,
    plot_latent_space,
    plot_recon_error_hist,
    plot_edge_error_hist,
    plot_canid_recon_hist, 
    plot_neighborhood_error_hist,
    plot_neighborhood_composite_error_hist,
    plot_error_components_analysis,
    plot_composite_error_hist,
    plot_structural_error_hist,
    plot_connectivity_error_hist,
    plot_raw_weighted_composite_error_hist
    
)
def extract_latent_vectors(pipeline, loader):
    """Extract latent vectors (graph embeddings) and labels from a data loader."""
    pipeline.autoencoder.eval()
    zs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(pipeline.device)
            # FIX: Use correct 5-output format
            _, _, _, z, _ = pipeline.autoencoder(batch.x, batch.edge_index, batch.batch)
            graphs = Batch.to_data_list(batch)
            start = 0
            for graph in graphs:
                n = graph.x.size(0)
                # Pooling: mean of node embeddings for each graph
                z_graph = z[start:start+n].mean(dim=0).cpu().numpy()
                zs.append(z_graph)
                labels.append(int(graph.y.flatten()[0]))
                start += n
    return np.array(zs), np.array(labels)

def find_optimal_weights(errors_normal, errors_attack, 
                        neighbor_errors_normal, neighbor_errors_attack,
                        canid_errors_normal, canid_errors_attack):
    """Find weights that maximize separation between normal and attack."""
    from scipy.optimize import minimize
    
    def separation_metric(weights):
        w1, w2, w3 = weights
        comp_n = w1*np.array(errors_normal) + w2*np.array(neighbor_errors_normal) + w3*np.array(canid_errors_normal)
        comp_a = w1*np.array(errors_attack) + w2*np.array(neighbor_errors_attack) + w3*np.array(canid_errors_attack)
        
        # Maximize difference in means, minimize overlap
        mean_diff = np.mean(comp_a) - np.mean(comp_n)
        std_sum = np.std(comp_n) + np.std(comp_a)
        return -(mean_diff / (std_sum + 1e-8))  # Negative for minimization
    
    result = minimize(separation_metric, [1.0, 1.0, 1.0], 
                     bounds=[(0.1, 10), (0.1, 10), (0.1, 10)])
    return result.x
class GATPipeline:
    """Pipeline for training and evaluating GAT-based autoencoder and classifier."""
    
    def __init__(self, num_ids, embedding_dim=8, device='cpu'):
        """Initialize the GATPipeline.

        Args:
            num_ids (int): Number of unique CAN IDs.
            embedding_dim (int, optional): Dimension of latent embeddings. Defaults to 8.
            device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.device = device
        self.autoencoder = GraphAutoencoderNeighborhood(num_ids=num_ids, in_channels=11, embedding_dim=embedding_dim).to(device)
        self.classifier = GATWithJK(num_ids=num_ids, in_channels=11, hidden_channels=32, out_channels=1, num_layers=3, heads=8, embedding_dim=embedding_dim).to(device)
        self.threshold = 0.0000

    def _compute_neighborhood_loss(self, neighbor_logits, x, edge_index):
        """Compute neighborhood reconstruction loss."""
        neighbor_targets = self.autoencoder.create_neighborhood_targets(x, edge_index, None)
        return nn.BCEWithLogitsLoss()(neighbor_logits, neighbor_targets)

    def _compute_neighborhood_reconstruction_errors(self, loader):
        """Compute neighborhood reconstruction errors for graphs."""
        errors_normal, errors_attack = [], []
        neighbor_errors_normal, neighbor_errors_attack = [], []
        id_errors_normal, id_errors_attack = [], []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                cont_out, canid_logits, neighbor_logits, z, _ = self.autoencoder(
                    batch.x, batch.edge_index, batch.batch)
                
                # Node reconstruction errors
                node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                
                # Neighborhood reconstruction errors
                neighbor_targets = self.autoencoder.create_neighborhood_targets(
                    batch.x, batch.edge_index, batch.batch)
                neighbor_recon_errors = nn.BCEWithLogitsLoss(reduction='none')(
                    neighbor_logits, neighbor_targets).mean(dim=1)
                
                # CAN ID errors
                canid_pred = canid_logits.argmax(dim=1)
                
                graphs = Batch.to_data_list(batch)
                start = 0
                for graph in graphs:
                    num_nodes = graph.x.size(0)
                    is_attack = int(graph.y.flatten()[0]) == 1
                    
                    # Node error
                    graph_node_errors = node_errors[start:start+num_nodes]
                    if graph_node_errors.numel() > 0:
                        node_error = graph_node_errors.max().item()
                        (errors_attack if is_attack else errors_normal).append(node_error)
                    
                    # Neighborhood error
                    graph_neighbor_errors = neighbor_recon_errors[start:start+num_nodes]
                    if graph_neighbor_errors.numel() > 0:
                        neighbor_error = graph_neighbor_errors.max().item()
                        (neighbor_errors_attack if is_attack else neighbor_errors_normal).append(neighbor_error)
                    
                    # CAN ID error
                    true_canids = graph.x[:, 0].long().cpu()
                    pred_canids = canid_pred[start:start+num_nodes].cpu()
                    id_error = 1.0 - (pred_canids == true_canids).float().mean().item()
                    (id_errors_attack if is_attack else id_errors_normal).append(id_error)
                    
                    start += num_nodes
        
        return (errors_normal, errors_attack, neighbor_errors_normal, 
                neighbor_errors_attack, id_errors_normal, id_errors_attack)
    
    def _set_threshold(self, train_loader, percentile=50):
        """Set the anomaly detection threshold based on training data."""
        errors = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                # FIX: Use correct 5-output format
                cont_out, _, _, _, _ = self.autoencoder(batch.x, batch.edge_index, batch.batch)
                errors.append((cont_out - batch.x[:, 1:]).pow(2).mean(dim=1))
        self.threshold = torch.cat(errors).quantile(percentile / 100.0).item()

    def _create_balanced_dataset(self, filtered_graphs):
        """Create a balanced dataset by downsampling the majority class.

        Args:
            filtered_graphs (list): List of filtered graphs.

        Returns:
            list: List of balanced graphs.
        """
        attack_graphs = [g.cpu() for g in filtered_graphs if g.y.item() == 1]
        normal_graphs = [g.cpu() for g in filtered_graphs if g.y.item() == 0]
        
        # Downsample to match minority class
        min_count = min(len(attack_graphs), len(normal_graphs))
        if len(normal_graphs) > min_count:
            random.seed(42)
            normal_graphs = random.sample(normal_graphs, min_count)
        if len(attack_graphs) > min_count:
            random.seed(42)
            attack_graphs = random.sample(attack_graphs, min_count)
        
        # Combine and shuffle
        balanced_graphs = attack_graphs + normal_graphs
        np.random.shuffle(balanced_graphs)
        return balanced_graphs
    
    def _print_neighborhood_statistics(self, errors_normal, errors_attack, 
                             neighbor_errors_normal, neighbor_errors_attack,
                             id_errors_normal, id_errors_attack):
        """Print neighborhood reconstruction statistics."""
        print(f"Processed {len(errors_normal)} normal graphs, {len(errors_attack)} attack graphs")
        print(f"Mean node reconstruction error (normal): {np.mean(errors_normal) if errors_normal else 'N/A'}")
        print(f"Mean node reconstruction error (attack): {np.mean(errors_attack) if errors_attack else 'N/A'}")
        print(f"Mean neighborhood error (normal): {np.mean(neighbor_errors_normal) if neighbor_errors_normal else 'N/A'}")
        print(f"Mean neighborhood error (attack): {np.mean(neighbor_errors_attack) if neighbor_errors_attack else 'N/A'}")
        
        if errors_normal and errors_attack:
            print(f"Node reconstruction - Normal: {np.mean(errors_normal):.4f}±{np.std(errors_normal):.4f}")
            print(f"Node reconstruction - Attack: {np.mean(errors_attack):.4f}±{np.std(errors_attack):.4f}")
            print(f"Neighborhood reconstruction - Normal: {np.mean(neighbor_errors_normal):.4f}±{np.std(neighbor_errors_normal):.4f}")
            print(f"Neighborhood reconstruction - Attack: {np.mean(neighbor_errors_attack):.4f}±{np.std(neighbor_errors_attack):.4f}")
            print(f"CAN ID reconstruction - Normal: {np.mean(id_errors_normal):.4f}±{np.std(id_errors_normal):.4f}")
            print(f"CAN ID reconstruction - Attack: {np.mean(id_errors_attack):.4f}±{np.std(id_errors_attack):.4f}")

        # Create plots
        neighbor_threshold = np.percentile(neighbor_errors_normal, 95) if neighbor_errors_normal else 0
        print(f"Neighborhood error threshold: {neighbor_threshold}")
        print(f"Node reconstruction threshold: {self.threshold}")

        plot_recon_error_hist(errors_normal, errors_attack, self.threshold, save_path="images/recon_error_hist.png")
        # FIX: Use correct function for neighborhood errors
        plot_neighborhood_error_hist(neighbor_errors_normal, neighbor_errors_attack, neighbor_threshold, save_path="images/neighborhood_error_hist.png")
        plot_canid_recon_hist(id_errors_normal, id_errors_attack, save_path="images/canid_recon_hist.png")

        # ADD: Plot composite error combining all three error types
        if (errors_normal and errors_attack and neighbor_errors_normal and neighbor_errors_attack and
            id_errors_normal and id_errors_attack):
            plot_neighborhood_composite_error_hist(
                errors_normal, errors_attack,
                neighbor_errors_normal, neighbor_errors_attack,
                id_errors_normal, id_errors_attack,
                save_path="images/neighborhood_composite_error_hist.png"
            )

        # Add this to _print_neighborhood_statistics method
        plot_error_components_analysis(
            errors_normal, errors_attack,
            neighbor_errors_normal, neighbor_errors_attack,
            id_errors_normal, id_errors_attack,
            save_path="images/error_components_analysis.png"
        )

    def _print_error_statistics(self, errors_normal, errors_attack, edge_errors_normal, 
                               edge_errors_attack, id_errors_normal, id_errors_attack,
                               structural_errors_normal=None, structural_errors_attack=None,
                               connectivity_errors_normal=None, connectivity_errors_attack=None):
        """Print error statistics and generate plots for various error types.

        Args:
            errors_normal (list): Node errors for normal graphs.
            errors_attack (list): Node errors for attack graphs.
            edge_errors_normal (list): Edge errors for normal graphs.
            edge_errors_attack (list): Edge errors for attack graphs.
            id_errors_normal (list): CAN ID errors for normal graphs.
            id_errors_attack (list): CAN ID errors for attack graphs.
            structural_errors_normal (list, optional): Structural feature scores for normal graphs. Defaults to None.
            structural_errors_attack (list, optional): Structural feature scores for attack graphs. Defaults to None.
            connectivity_errors_normal (list, optional): Connectivity anomaly scores for normal graphs. Defaults to None.
            connectivity_errors_attack (list, optional): Connectivity anomaly scores for attack graphs. Defaults to None.
        """
        print(f"Processed {len(errors_normal)} normal graphs, {len(errors_attack)} attack graphs")
        print(f"Mean reconstruction error (normal): {np.mean(errors_normal) if errors_normal else 'N/A'}")
        print(f"Mean reconstruction error (attack): {np.mean(errors_attack) if errors_attack else 'N/A'}")
        
        # Print detailed statistics for each error type
        if errors_normal and errors_attack:
            print(f"Node reconstruction - Normal: {np.mean(errors_normal):.4f}±{np.std(errors_normal):.4f}")
            print(f"Node reconstruction - Attack: {np.mean(errors_attack):.4f}±{np.std(errors_attack):.4f}")
            print(f"Edge prediction - Normal: {np.mean(edge_errors_normal):.4f}±{np.std(edge_errors_normal):.4f}")
            print(f"Edge prediction - Attack: {np.mean(edge_errors_attack):.4f}±{np.std(edge_errors_attack):.4f}")
            print(f"CAN ID reconstruction - Normal: {np.mean(id_errors_normal):.4f}±{np.std(id_errors_normal):.4f}")
            print(f"CAN ID reconstruction - Attack: {np.mean(id_errors_attack):.4f}±{np.std(id_errors_attack):.4f}")
            
            if structural_errors_normal and structural_errors_attack:
                print(f"Structural features - Normal: {np.mean(structural_errors_normal):.4f}±{np.std(structural_errors_normal):.4f}")
                print(f"Structural features - Attack: {np.mean(structural_errors_attack):.4f}±{np.std(structural_errors_attack):.4f}")
            
            if connectivity_errors_normal and connectivity_errors_attack:
                print(f"Connectivity anomalies - Normal: {np.mean(connectivity_errors_normal):.4f}±{np.std(connectivity_errors_normal):.4f}")
                print(f"Connectivity anomalies - Attack: {np.mean(connectivity_errors_attack):.4f}±{np.std(connectivity_errors_attack):.4f}")
        
        # Create plots
        edge_threshold = np.percentile(edge_errors_normal, 95) if edge_errors_normal else 0
        print(f"Edge error threshold: {edge_threshold}")
        print(f"Node reconstruction threshold: {self.threshold}")

        plot_recon_error_hist(errors_normal, errors_attack, self.threshold, save_path="images/recon_error_hist.png")
        plot_edge_error_hist(edge_errors_normal, edge_errors_attack, edge_threshold, save_path="images/edge_error_hist.png")
        plot_canid_recon_hist(id_errors_normal, id_errors_attack, save_path="images/canid_recon_hist.png")

        # Plot composite error if all components are available
        if (errors_normal and errors_attack and edge_errors_normal and edge_errors_attack and
            id_errors_normal and id_errors_attack):
            plot_composite_error_hist(
                errors_normal, errors_attack,
                edge_errors_normal, edge_errors_attack,
                id_errors_normal, id_errors_attack,
                save_path="images/composite_error_hist.png"
            )
        # plot raw weighted composite error histogram
        if (errors_normal and errors_attack and edge_errors_normal and edge_errors_attack and
            id_errors_normal and id_errors_attack):
            plot_raw_weighted_composite_error_hist(errors_normal, errors_attack,
                                          edge_errors_normal, edge_errors_attack,
                                          id_errors_normal, id_errors_attack,
                                          save_path="images/raw_weighted_composite_error_hist.png")

        # Plot structural differences if available
        if structural_errors_normal and structural_errors_attack:
            plot_structural_error_hist(
                structural_errors_normal, structural_errors_attack,
                save_path="images/structural_error_hist.png"
            )

        # Plot connectivity anomalies if available
        if connectivity_errors_normal and connectivity_errors_attack:
            plot_connectivity_error_hist(
                connectivity_errors_normal, connectivity_errors_attack,
                save_path="images/connectivity_error_hist.png"
            )
        

    def train_stage1(self, train_loader, epochs=100):
        """Train the autoencoder for anomaly detection (Stage 1).

        Args:
            train_loader (torch_geometric.loader.DataLoader): DataLoader for normal graphs.
            epochs (int, optional): Number of training epochs. Defaults to 100.
        """
        self.autoencoder.train()
        opt = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-2, weight_decay=1e-4)
        
        for _ in range(epochs):
            for batch in train_loader:
                batch = batch.to(self.device)
                
                cont_out, canid_logits, neighbor_logits, z, kl_loss = self.autoencoder(
                    batch.x, batch.edge_index, batch.batch)
                
                # Compute losses with edge attributes
                cont_loss = (cont_out - batch.x[:, 1:]).pow(2).mean()
                canid_loss = nn.CrossEntropyLoss()(canid_logits, batch.x[:, 0].long())
                neighbor_loss = self._compute_neighborhood_loss(neighbor_logits, batch.x, batch.edge_index)

                # Use edge attributes if available
                # edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
                # edge_loss = self._compute_edge_loss(z, batch.edge_index, edge_attr)
                
                beta = 0.1
                loss = cont_loss + canid_loss + neighbor_loss + beta * kl_loss
                
                opt.zero_grad()
                loss.backward()
                opt.step()
        
        self._set_threshold(train_loader, percentile=50)

    def train_stage2(self, full_loader, epochs=50):
        """Train the classifier on filtered graphs (Stage 2)."""
        # Print label distribution
        all_labels = [batch.y.cpu().numpy() for batch in full_loader]
        all_labels = np.concatenate(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)
        print("Label distribution in full training set (Stage 2):")
        for u, c in zip(unique, counts):
            print(f"Label {u}: {c} samples")

        # Use neighborhood reconstruction instead of edge reconstruction
        result = self._compute_neighborhood_reconstruction_errors(full_loader)
        (errors_normal, errors_attack, neighbor_errors_normal, 
        neighbor_errors_attack, id_errors_normal, id_errors_attack) = result
        
        # Print statistics
        self._print_neighborhood_statistics(errors_normal, errors_attack, 
                                        neighbor_errors_normal, neighbor_errors_attack,
                                        id_errors_normal, id_errors_attack)

        # Filter and train classifier (rest remains the same)
        filtered = self._filter_anomalous_graphs(full_loader)
        if not filtered:
            print("No graphs exceeded the anomaly threshold. Skipping classifier training.")
            return
        
        filtered = self._create_balanced_dataset(filtered)
        self._train_classifier(filtered, epochs)

    def _filter_anomalous_graphs(self, loader):
        """Filter graphs that exceed the anomaly threshold.

        Args:
            loader (torch_geometric.loader.DataLoader): DataLoader yielding batches of graphs.

        Returns:
            list: List of graphs exceeding the anomaly threshold.
        """
        filtered = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                # FIX: Use correct 5-output format
                cont_out, _, _, _, _ = self.autoencoder(batch.x, batch.edge_index, batch.batch)
                error = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                
                graphs = Batch.to_data_list(batch)
                start = 0
                for graph in graphs:
                    num_nodes = graph.x.size(0)
                    node_errors = error[start:start+num_nodes]
                    if node_errors.numel() > 0 and (node_errors > self.threshold).any():
                        filtered.append(graph)
                    start += num_nodes
        
        print(f"Anomaly threshold: {self.threshold}")
        print(f"Number of graphs exceeding threshold: {len(filtered)}")
        return filtered

    def _train_classifier(self, filtered_graphs, epochs):
        """Train the classifier on filtered graphs.

        Args:
            filtered_graphs (list): List of graphs for classifier training.
            epochs (int): Number of training epochs.
        """
        labels = [int(graph.y.flatten()[0]) for graph in filtered_graphs]
        num_pos = sum(labels)
        num_neg = len(labels) - num_pos
        
        pos_weight = torch.tensor(1.0 if num_pos == 0 else num_neg / num_pos, device=self.device)
        print(f"Using pos_weight={pos_weight.item():.4f} for BCEWithLogitsLoss")
        
        self.classifier.train()
        opt = torch.optim.Adam(self.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in DataLoader(filtered_graphs, batch_size=32, shuffle=True):
                batch = batch.to(self.device)
                preds = self.classifier(batch)
                loss = criterion(preds.squeeze(), batch.y.float())
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Debug info for first batch of first epoch
                if epoch == 0 and num_batches == 1:
                    print("Batch labels (y):", batch.y[:5])
                    print("Classifier raw outputs:", preds[:5].detach().cpu().numpy())
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Print accuracy
            if True:  # You can add condition like (epoch + 1) % 10 == 0
                acc = self._evaluate_classifier(filtered_graphs)
                print(f"Classifier accuracy at epoch {epoch+1}: {acc:.4f}")

    def _evaluate_classifier(self, graphs):
        """Evaluate classifier accuracy on a set of graphs.

        Args:
            graphs (list): List of graphs to evaluate.

        Returns:
            float: Classification accuracy.
        """
        self.classifier.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in DataLoader(graphs, batch_size=32):
                batch = batch.to(self.device)
                out = self.classifier(batch)
                pred_labels = (out.squeeze() > 0.5).long()
                all_preds.append(pred_labels.cpu())
                all_labels.append(batch.y.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        accuracy = (all_preds == all_labels).float().mean().item()
        
        self.classifier.train()
        return accuracy

    def predict(self, data):
        """Predict anomalies and classify graphs using the two-stage pipeline.

        Args:
            data (torch_geometric.data.Batch): Batch of graphs.

        Returns:
            torch.Tensor: Tensor of predicted labels (0 for normal, 1 for attack).
        """
        data = data.to(self.device)
        
        # Stage 1: Anomaly detection
        with torch.no_grad():
            # FIX: Use new 5-output format
            cont_out, canid_logits, neighbor_logits, z, kl_loss = self.autoencoder(
                data.x, data.edge_index, data.batch)
            error = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            # Process each graph in Batch
            preds = []
            graphs = Batch.to_data_list(data)  # FIX: Get graphs from data, not undefined variable
            start = 0
            for graph in graphs:
                num_nodes = graph.x.size(0)
                node_errors = error[start:start+num_nodes]
                if node_errors.numel() > 0 and (node_errors > self.threshold).any():
                    # Stage 2: Classification
                    graph_batch = graph.to(self.device)
                    prob = self.classifier(graph_batch).item()
                    preds.append(1 if prob > 0.5 else 0)
                else:
                    preds.append(0)
                start += num_nodes
            
            return torch.tensor(preds, device=self.device)


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    """Main function for training and evaluating the anomaly detection pipeline.

    Args:
        config (omegaconf.DictConfig): Hydra configuration object.
    """
    config_dict = OmegaConf.to_container(config, resolve=True)
    print(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Model is using device: {device}')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    root_folders = {'hcrl_ch': r"datasets/can-train-and-test-v1.5/hcrl-ch",
                    'hcrl_sa': r"datasets/can-train-and-test-v1.5/hcrl-sa",
                    'set_01' : r"datasets/can-train-and-test-v1.5/set_01",
                    'set_02' : r"datasets/can-train-and-test-v1.5/set_02",
                    'set_03' : r"datasets/can-train-and-test-v1.5/set_03",
                    'set_04' : r"datasets/can-train-and-test-v1.5/set_04",
    }
    KEY = config_dict['root_folder']
    root_folder = root_folders[KEY]
    print(f"Root folder: {root_folder}")

    # Step 1: Build ID mapping from only normal graphs
    id_mapping = build_id_mapping_from_normal(root_folder)

    dataset = graph_creation(root_folder, id_mapping=id_mapping, window_size=100)
    num_ids = len(id_mapping)
    embedding_dim = 8  # or your choice
    print(f"Number of graphs: {len(dataset)}")

    for data in dataset:
        assert not torch.isnan(data.x).any(), "Dataset contains NaN values!"
        assert not torch.isinf(data.x).any(), "Dataset contains Inf values!"

    DATASIZE = config_dict['datasize']
    EPOCHS = config_dict['epochs']
    LR = config_dict['lr']
    BATCH_SIZE = config_dict['batch_size']
    TRAIN_RATIO = config_dict['train_ratio']
    USE_FOCAL_LOSS = config_dict['use_focal_loss']

    print("Size of the total dataset: ", len(dataset))

    # plot the features in a histogram
    feature_names = ["CAN ID", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "data8", "count", "position"]
    plot_feature_histograms([data for data in dataset], feature_names=feature_names, save_path="images/feature_histograms.png")

    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator1)
    print('Size of DATASIZE: ', DATASIZE)
    print('Size of Training dataset: ', len(train_dataset))
    print('Size of Testing dataset: ', len(test_dataset))

    # --------- FILTER NORMAL GRAPHS FOR AUTOENCODER TRAINING ----------
    # Only keep graphs with label == 0 for autoencoder training
    normal_indices = [i for i, data in enumerate(train_dataset) if int(data.y.flatten()[0]) == 0]
    if DATASIZE < 1.0:
        subset_size = int(len(normal_indices) * DATASIZE)
        indices = np.random.choice(normal_indices, subset_size, replace=False)
    else:
        indices = normal_indices
    normal_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(normal_subset, batch_size=BATCH_SIZE, shuffle=True)
    # ---------------------------------------------------------------

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('Size of Training dataloader: ', len(train_loader))
    print('Size of Testing dataloader: ', len(test_loader))
    print('Size of Training dataloader (samples): ', len(train_loader.dataset))
    print('Size of Testing dataloader (samples): ', len(test_loader.dataset))

    pipeline = GATPipeline(num_ids=num_ids, embedding_dim=embedding_dim, device=device)

    print("Stage 1: Training autoencoder for anomaly detection...")
    pipeline.train_stage1(train_loader, epochs=25)

    # Visualize input vs. reconstructed features for a few graphs
    # For classifier, use all graphs (normal + attack)
    full_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    plot_graph_reconstruction(pipeline, full_train_loader, num_graphs=4, save_path="images/graph_recon_examples.png")

    N = 10000  # or any number you like
    indices = np.random.choice(len(train_dataset), size=N, replace=False)
    subsample = [train_dataset[i] for i in indices]
    subsample_loader = DataLoader(subsample, batch_size=BATCH_SIZE, shuffle=False)

    zs, labels = extract_latent_vectors(pipeline, subsample_loader)
    plot_latent_space(zs, labels, save_path="images/latent_space.png")

    # Visualize node-level reconstruction errors
    plot_node_recon_errors(pipeline, full_train_loader, num_graphs=5, save_path="images/node_recon_subplot.png")
    
    print("Stage 2: Training classifier on filtered graphs...")
    pipeline.train_stage2(full_train_loader, epochs=3)

    

    # # Save models
    save_folder = "saved_models"
    os.makedirs(save_folder, exist_ok=True)
    torch.save(pipeline.autoencoder.state_dict(), os.path.join(save_folder, f'autoencoder_{KEY}.pth'))
    torch.save(pipeline.classifier.state_dict(), os.path.join(save_folder, f'classifier_{KEY}.pth'))
    print(f"Models saved in '{save_folder}'.")

    # Optionally: Evaluate on test set
    print("Evaluating on test set...")
    test_labels = [data.y.item() for data in test_dataset]
    unique, counts = np.unique(test_labels, return_counts=True)
    print("Test set label distribution:")
    for u, c in zip(unique, counts):
        print(f"Label {u}: {c} samples")
    preds = []
    labels = []
    for batch in test_loader:
        batch = batch.to(device)
        pred = pipeline.predict(batch)
        preds.append(pred.cpu())
        labels.append(batch.y.cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    accuracy = (preds == labels).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")

    # Print confusion matrix
    cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
    print("Confusion Matrix:")
    print(cm)

    # # Save metrics
    # metrics = {"test_accuracy": accuracy}
    # with open('ad_metrics.json', 'w') as f:
    #     import json
    #     json.dump(metrics, f, indent=4)
    # print("Metrics saved as 'ad_metrics.json'.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")