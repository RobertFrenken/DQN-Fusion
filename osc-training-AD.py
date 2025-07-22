import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
import random

from models.models import GATWithJK, GraphAutoencoderNeighborhood
from preprocessing import graph_creation, build_id_mapping_from_normal, build_all_id_mapping
from torch_geometric.data import Batch

from plotting_utils import (
    plot_feature_histograms,
    plot_node_recon_errors, 
    plot_graph_reconstruction,
    plot_latent_space,
    plot_recon_error_hist,
    plot_neighborhood_error_hist,
    plot_neighborhood_composite_error_hist,
    plot_error_components_analysis,
    plot_raw_weighted_composite_error_hist,
    plot_raw_error_components_with_composite 
)

def extract_latent_vectors(pipeline, loader):
    """Extract latent vectors (graph embeddings) and labels from a data loader."""
    pipeline.autoencoder.eval()
    zs, labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(pipeline.device)
            _, _, _, z, _ = pipeline.autoencoder(batch.x, batch.edge_index, batch.batch)
            
            graphs = Batch.to_data_list(batch)
            start = 0
            for graph in graphs:
                n = graph.x.size(0)
                z_graph = z[start:start+n].mean(dim=0).cpu().numpy()
                zs.append(z_graph)
                labels.append(int(graph.y.flatten()[0]))
                start += n
                
    return np.array(zs), np.array(labels)


class GATPipeline:
    """Two-stage pipeline for CAN bus anomaly detection using GAD-NR neighborhood reconstruction."""
    
    def __init__(self, num_ids, embedding_dim=8, device='cpu'):
        """Initialize the pipeline with autoencoder and classifier."""
        self.device = device
        self.autoencoder = GraphAutoencoderNeighborhood(
            num_ids=num_ids, in_channels=11, embedding_dim=embedding_dim
        ).to(device)
        self.classifier = GATWithJK(
            num_ids=num_ids, in_channels=11, hidden_channels=32, 
            out_channels=1, num_layers=3, heads=8, embedding_dim=embedding_dim
        ).to(device)
        self.threshold = 0.0

    def _compute_neighborhood_loss(self, neighbor_logits, x, edge_index):
        """Compute neighborhood reconstruction loss using BCEWithLogitsLoss."""
        neighbor_targets = self.autoencoder.create_neighborhood_targets(x, edge_index, None)
        return nn.BCEWithLogitsLoss()(neighbor_logits, neighbor_targets)

    def _set_threshold(self, train_loader, percentile=50):
        """Set anomaly detection threshold based on training data reconstruction errors."""
        errors = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                cont_out, _, _, _, _ = self.autoencoder(batch.x, batch.edge_index, batch.batch)
                errors.append((cont_out - batch.x[:, 1:]).pow(2).mean(dim=1))
        self.threshold = torch.cat(errors).quantile(percentile / 100.0).item()

    def train_stage1(self, train_loader, epochs=100):
        """Stage 1: Train autoencoder on normal graphs for anomaly detection."""
        print(f"Training autoencoder for {epochs} epochs...")
        self.autoencoder.train()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-2, weight_decay=1e-4)
        
        for epoch in range(epochs):
            for batch in train_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                cont_out, canid_logits, neighbor_logits, z, kl_loss = self.autoencoder(
                    batch.x, batch.edge_index, batch.batch)
                
                # Compute losses
                cont_loss = (cont_out - batch.x[:, 1:]).pow(2).mean()
                canid_loss = nn.CrossEntropyLoss()(canid_logits, batch.x[:, 0].long())
                neighbor_loss = self._compute_neighborhood_loss(neighbor_logits, batch.x, batch.edge_index)
                
                total_loss = cont_loss + canid_loss + neighbor_loss + 0.1 * kl_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        
        self._set_threshold(train_loader, percentile=50)
        print(f"Set anomaly threshold: {self.threshold:.4f}")

    def _compute_reconstruction_errors(self, loader):
        """Compute reconstruction errors for all graphs in loader."""
        errors_normal, errors_attack = [], []
        neighbor_errors_normal, neighbor_errors_attack = [], []
        id_errors_normal, id_errors_attack = [], []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                    batch.x, batch.edge_index, batch.batch)
                
                # Node reconstruction errors
                node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                
                # Neighborhood reconstruction errors
                neighbor_targets = self.autoencoder.create_neighborhood_targets(
                    batch.x, batch.edge_index, batch.batch)
                neighbor_recon_errors = nn.BCEWithLogitsLoss(reduction='none')(
                    neighbor_logits, neighbor_targets).mean(dim=1)
                
                # CAN ID prediction errors
                canid_pred = canid_logits.argmax(dim=1)
                
                # Process each graph in batch
                graphs = Batch.to_data_list(batch)
                start = 0
                for graph in graphs:
                    num_nodes = graph.x.size(0)
                    is_attack = int(graph.y.flatten()[0]) == 1
                    
                    # Extract errors for this graph
                    graph_node_error = node_errors[start:start+num_nodes].max().item()
                    graph_neighbor_error = neighbor_recon_errors[start:start+num_nodes].max().item()
                    
                    true_canids = graph.x[:, 0].long().cpu()
                    pred_canids = canid_pred[start:start+num_nodes].cpu()
                    canid_error = 1.0 - (pred_canids == true_canids).float().mean().item()
                    
                    # Store in appropriate lists
                    target_lists = (errors_attack, neighbor_errors_attack, id_errors_attack) if is_attack else \
                                 (errors_normal, neighbor_errors_normal, id_errors_normal)
                    target_lists[0].append(graph_node_error)
                    target_lists[1].append(graph_neighbor_error)
                    target_lists[2].append(canid_error)
                    
                    start += num_nodes
        
        return (errors_normal, errors_attack, neighbor_errors_normal, 
                neighbor_errors_attack, id_errors_normal, id_errors_attack)

    def _print_statistics_and_plots(self, errors_normal, errors_attack, 
                                   neighbor_errors_normal, neighbor_errors_attack,
                                   id_errors_normal, id_errors_attack):
        """Print statistics and generate plots for all error types."""
        print(f"\nReconstruction Error Statistics:")
        print(f"Processed {len(errors_normal)} normal, {len(errors_attack)} attack graphs")
        
        if errors_normal and errors_attack:
            print(f"Node reconstruction - Normal: {np.mean(errors_normal):.4f}±{np.std(errors_normal):.4f}")
            print(f"Node reconstruction - Attack: {np.mean(errors_attack):.4f}±{np.std(errors_attack):.4f}")
            print(f"Neighborhood - Normal: {np.mean(neighbor_errors_normal):.4f}±{np.std(neighbor_errors_normal):.4f}")
            print(f"Neighborhood - Attack: {np.mean(neighbor_errors_attack):.4f}±{np.std(neighbor_errors_attack):.4f}")
            print(f"CAN ID - Normal: {np.mean(id_errors_normal):.4f}±{np.std(id_errors_normal):.4f}")
            print(f"CAN ID - Attack: {np.mean(id_errors_attack):.4f}±{np.std(id_errors_attack):.4f}")

            # Generate plots
            neighbor_threshold = np.percentile(neighbor_errors_normal, 95)
            
            plot_recon_error_hist(errors_normal, errors_attack, self.threshold, 
                                save_path="images/recon_error_hist.png")
            plot_neighborhood_error_hist(neighbor_errors_normal, neighbor_errors_attack, 
                                        neighbor_threshold, save_path="images/neighborhood_error_hist.png")
            plot_neighborhood_composite_error_hist(
                errors_normal, errors_attack, neighbor_errors_normal, neighbor_errors_attack,
                id_errors_normal, id_errors_attack, save_path="images/neighborhood_composite_error_hist.png")
            plot_error_components_analysis(
                errors_normal, errors_attack, neighbor_errors_normal, neighbor_errors_attack,
                id_errors_normal, id_errors_attack, save_path="images/error_components_analysis.png")

            plot_raw_error_components_with_composite(
                errors_normal, errors_attack, neighbor_errors_normal, neighbor_errors_attack,
                id_errors_normal, id_errors_attack, save_path="images/raw_error_components_with_composite.png")

    def train_stage2(self, full_loader, epochs=50):
        """Stage 2: Train classifier with all attacks + filtered normal graphs."""
        print(f"\nStage 2: Analyzing reconstruction errors and training classifier...")
        
        # Compute reconstruction errors for all graphs
        result = self._compute_reconstruction_errors(full_loader)
        errors_normal, errors_attack, neighbor_errors_normal, neighbor_errors_attack, id_errors_normal, id_errors_attack = result
        
        # Print statistics and generate plots
        self._print_statistics_and_plots(errors_normal, errors_attack, neighbor_errors_normal, 
                                        neighbor_errors_attack, id_errors_normal, id_errors_attack)

        # Create balanced dataset with new strategy
        balanced_graphs = self._create_balanced_dataset_with_composite_filtering(full_loader)
        if not balanced_graphs:
            print("No graphs available for classifier training.")
            return
            
        self._train_classifier(balanced_graphs, epochs)

    def _compute_composite_reconstruction_errors(self, loader):
        """Compute composite reconstruction errors for filtering normal graphs."""
        graph_data = []  # Store (graph, composite_error, is_attack)
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                    batch.x, batch.edge_index, batch.batch)
                
                # Node reconstruction errors
                node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                
                # Neighborhood reconstruction errors
                neighbor_targets = self.autoencoder.create_neighborhood_targets(
                    batch.x, batch.edge_index, batch.batch)
                neighbor_recon_errors = nn.BCEWithLogitsLoss(reduction='none')(
                    neighbor_logits, neighbor_targets).mean(dim=1)
                
                # CAN ID prediction errors
                canid_pred = canid_logits.argmax(dim=1)
                
                # Process each graph in batch
                graphs = Batch.to_data_list(batch)
                start = 0
                for graph in graphs:
                    num_nodes = graph.x.size(0)
                    is_attack = int(graph.y.flatten()[0]) == 1
                    
                    # Extract errors for this graph
                    graph_node_error = node_errors[start:start+num_nodes].max().item()
                    graph_neighbor_error = neighbor_recon_errors[start:start+num_nodes].max().item()
                    
                    true_canids = graph.x[:, 0].long().cpu()
                    pred_canids = canid_pred[start:start+num_nodes].cpu()
                    canid_error = 1.0 - (pred_canids == true_canids).float().mean().item()
                    
                    # Compute composite error (rescaled raw values)
                    weight_node = 1.0       # Base scale
                    weight_neighbor = 20.0  # Scale up neighborhood errors
                    weight_canid = 0.3      # Scale down CAN ID errors
                    
                    composite_error = (weight_node * graph_node_error + 
                                    weight_neighbor * graph_neighbor_error + 
                                    weight_canid * canid_error)
                    
                    # Store graph with its composite error
                    graph_data.append((graph.cpu(), composite_error, is_attack))
                    start += num_nodes
        
        return graph_data

    def _create_balanced_dataset_with_composite_filtering(self, loader):
        """Create balanced dataset using all attacks + filtered normal graphs."""
        print("Computing composite errors for graph filtering...")
        graph_data = self._compute_composite_reconstruction_errors(loader)
        
        # Separate attack and normal graphs with their composite errors
        attack_graphs = [(graph, error) for graph, error, is_attack in graph_data if is_attack]
        normal_graphs = [(graph, error) for graph, error, is_attack in graph_data if not is_attack]
        
        print(f"Found {len(attack_graphs)} attack graphs and {len(normal_graphs)} normal graphs")
        
        # Use ALL attack graphs
        selected_attack_graphs = [graph for graph, _ in attack_graphs]
        num_attacks = len(selected_attack_graphs)
        
        if num_attacks == 0:
            print("No attack graphs found! Cannot train classifier.")
            return []
        
        # Calculate maximum normal graphs to maintain 4:1 ratio
        max_normal_graphs = num_attacks * 4
        
        if len(normal_graphs) <= max_normal_graphs:
            # Use all normal graphs if we don't exceed 4:1 ratio
            selected_normal_graphs = [graph for graph, _ in normal_graphs]
            print(f"Using all {len(selected_normal_graphs)} normal graphs (ratio: {len(selected_normal_graphs)}:{num_attacks})")
        else:
            # Filter out the "easiest" (lowest composite error) normal graphs
            # Sort by composite error (ascending) and take the hardest examples
            normal_graphs_sorted = sorted(normal_graphs, key=lambda x: x[1])
            selected_normal_graphs = [graph for graph, _ in normal_graphs_sorted[-max_normal_graphs:]]  # Take highest errors
            
            print(f"Filtered normal graphs from {len(normal_graphs)} to {len(selected_normal_graphs)}")
            print(f"Composite error range - Filtered out: [{normal_graphs_sorted[0][1]:.4f}, {normal_graphs_sorted[max_normal_graphs-1][1]:.4f}]")
            print(f"Composite error range - Kept: [{normal_graphs_sorted[-max_normal_graphs][1]:.4f}, {normal_graphs_sorted[-1][1]:.4f}]")
            print(f"Final ratio: {len(selected_normal_graphs)}:{num_attacks} (4:1 max maintained)")
        
        # Combine and shuffle
        balanced_graphs = selected_attack_graphs + selected_normal_graphs
        random.seed(42)
        random.shuffle(balanced_graphs)
        
        print(f"Created dataset for GAT training: {len(selected_normal_graphs)} normal, {num_attacks} attack")
        print(f"Final ratio: {len(selected_normal_graphs)/num_attacks:.1f}:1")
        
        return balanced_graphs

    def _train_classifier(self, filtered_graphs, epochs):
        """Train binary classifier on filtered graphs."""
        print(f"Training classifier for {epochs} epochs...")
        
        labels = [int(graph.y.flatten()[0]) for graph in filtered_graphs]
        num_pos, num_neg = sum(labels), len(labels) - sum(labels)
        pos_weight = torch.tensor(1.0 if num_pos == 0 else num_neg / num_pos, device=self.device)
        
        self.classifier.train()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in DataLoader(filtered_graphs, batch_size=32, shuffle=True):
                batch = batch.to(self.device)
                preds = self.classifier(batch)
                loss = criterion(preds.squeeze(), batch.y.float())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc = self._evaluate_classifier(filtered_graphs)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    def _evaluate_classifier(self, graphs):
        """Evaluate classifier accuracy."""
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
        """Two-stage prediction: anomaly detection + classification."""
        data = data.to(self.device)
        
        with torch.no_grad():
            cont_out, _, _, _, _ = self.autoencoder(data.x, data.edge_index, data.batch)
            error = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            preds = []
            graphs = Batch.to_data_list(data)
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
    """Main training and evaluation pipeline."""
    # Setup
    config_dict = OmegaConf.to_container(config, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Dataset paths
    root_folders = {
        'hcrl_ch': r"datasets/can-train-and-test-v1.5/hcrl-ch",
        'hcrl_sa': r"datasets/can-train-and-test-v1.5/hcrl-sa",
        'set_01': r"datasets/can-train-and-test-v1.5/set_01",
        'set_02': r"datasets/can-train-and-test-v1.5/set_02",
        'set_03': r"datasets/can-train-and-test-v1.5/set_03",
        'set_04': r"datasets/can-train-and-test-v1.5/set_04",
    }
    
    # Load and prepare data
    KEY = config_dict['root_folder']
    root_folder = root_folders[KEY]
    id_mapping = build_id_mapping_from_normal(root_folder)
    dataset = graph_creation(root_folder, id_mapping=id_mapping, window_size=100)
    
    print(f"Dataset: {len(dataset)} graphs, {len(id_mapping)} unique CAN IDs")
    
    # Validate dataset
    for data in dataset:
        assert not torch.isnan(data.x).any(), "Dataset contains NaN values!"
        assert not torch.isinf(data.x).any(), "Dataset contains Inf values!"

    # Configuration
    DATASIZE = config_dict['datasize']
    TRAIN_RATIO = config_dict['train_ratio']
    BATCH_SIZE = config_dict['batch_size']
    
    # Generate feature histograms
    feature_names = ["CAN ID", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "data8", "count", "position"]
    plot_feature_histograms([data for data in dataset], feature_names=feature_names, save_path="images/feature_histograms.png")

    # Train/test split
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    print(f'Train: {len(train_dataset)}, Test: {len(test_dataset)}')

    # Create normal-only training subset for autoencoder
    normal_indices = [i for i, data in enumerate(train_dataset) if int(data.y.flatten()[0]) == 0]
    if DATASIZE < 1.0:
        subset_size = int(len(normal_indices) * DATASIZE)
        indices = np.random.choice(normal_indices, subset_size, replace=False)
    else:
        indices = normal_indices
    
    normal_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(normal_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    full_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f'Normal training samples: {len(train_loader.dataset)}')

    # Initialize pipeline
    pipeline = GATPipeline(num_ids=len(id_mapping), embedding_dim=8, device=device)

    # Training
    print("\n=== Stage 1: Autoencoder Training ===")
    pipeline.train_stage1(train_loader, epochs=25)

    # Visualization
    plot_graph_reconstruction(pipeline, full_train_loader, num_graphs=4, save_path="images/graph_recon_examples.png")
    
    # Latent space visualization
    N = min(10000, len(train_dataset))
    indices = np.random.choice(len(train_dataset), size=N, replace=False)
    subsample = [train_dataset[i] for i in indices]
    subsample_loader = DataLoader(subsample, batch_size=BATCH_SIZE, shuffle=False)
    zs, labels = extract_latent_vectors(pipeline, subsample_loader)
    plot_latent_space(zs, labels, save_path="images/latent_space.png")
    plot_node_recon_errors(pipeline, full_train_loader, num_graphs=5, save_path="images/node_recon_subplot.png")
    
    print("\n=== Stage 2: Classifier Training ===")
    pipeline.train_stage2(full_train_loader, epochs=50)

    # Save models
    save_folder = "saved_models"
    os.makedirs(save_folder, exist_ok=True)
    torch.save(pipeline.autoencoder.state_dict(), os.path.join(save_folder, f'autoencoder_{KEY}.pth'))
    torch.save(pipeline.classifier.state_dict(), os.path.join(save_folder, f'classifier_{KEY}.pth'))
    print(f"Models saved to '{save_folder}'")

    # Evaluation
    print("\n=== Test Set Evaluation ===")
    test_labels = [data.y.item() for data in test_dataset]
    unique, counts = np.unique(test_labels, return_counts=True)
    print("Test set distribution:")
    for u, c in zip(unique, counts):
        print(f"  Label {u}: {c} samples")
    
    preds, labels = [], []
    for batch in test_loader:
        batch = batch.to(device)
        pred = pipeline.predict(batch)
        preds.append(pred.cpu())
        labels.append(batch.y.cpu())
    
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    accuracy = (preds == labels).float().mean().item()
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy()))


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")