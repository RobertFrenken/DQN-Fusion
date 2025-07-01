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
from models.models import GATWithJK, GraphAutoencoder
from preprocessing import graph_creation, build_id_mapping_from_normal
from training_utils import PyTorchTrainer, PyTorchDistillationTrainer, DistillationTrainer
from torch_geometric.data import Batch
from sklearn.metrics import confusion_matrix
import random
import matplotlib.pyplot as plt


from plotting_utils import (
    plot_feature_histograms,
    print_graph_stats,
    print_graph_structure,
    plot_node_recon_errors,
    plot_graph_reconstruction,
    plot_latent_space,
    plot_recon_error_hist,
    plot_edge_error_hist,
    plot_canid_recon_hist, 
    plot_composite_error_hist,
    plot_structural_error_hist,      # <-- add
    plot_connectivity_error_hist     # <-- add
)
def extract_latent_vectors(pipeline, loader):
    """Extract latent vectors (graph embeddings) and labels from a data loader.

    Args:
        pipeline: The GATPipeline object containing the autoencoder.
        loader: DataLoader yielding batches of graphs.

    Returns:
        Tuple of (latent_vectors, labels) as numpy arrays.
    """
    pipeline.autoencoder.eval()
    zs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(pipeline.device)
            # Only need z (latent), ignore other outputs
            _, _, z, _ = pipeline.autoencoder(batch.x, batch.edge_index, batch.batch)
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

def compute_edge_recon_error(autoencoder, batch):
    """Compute edge reconstruction error for each graph in a batch.

    Args:
        autoencoder: The trained autoencoder model.
        batch: A batch of graphs.

    Returns:
        List of edge reconstruction errors, one per graph.
    """
    with torch.no_grad():
        _, _, z, _ = autoencoder(batch.x, batch.edge_index, batch.batch)
        graphs = Batch.to_data_list(batch)
        start = 0
        edge_errors = []
        for graph in graphs:
            num_nodes = graph.x.size(0)
            pos_edge_index = graph.edge_index
            
            # Skip graphs with no edges or very few nodes
            if pos_edge_index.size(1) == 0 or num_nodes < 2:
                edge_errors.append(0.0)
                start += num_nodes
                continue
            
            # Get latent representations for this graph
            z_graph = z[start:start+num_nodes]
            
            # Positive edge predictions
            pos_edge_preds = autoencoder.decode_edge(z_graph, pos_edge_index)
            pos_edge_labels = torch.ones(pos_edge_preds.size(0), device=pos_edge_preds.device)
            
            # Generate better negative edges - ensure they don't exist in positive set
            pos_edge_set = set()
            for i in range(pos_edge_index.size(1)):
                src, dst = pos_edge_index[0, i].item(), pos_edge_index[1, i].item()
                pos_edge_set.add((src, dst))
                pos_edge_set.add((dst, src))  # undirected
            
            # Sample negative edges more carefully
            neg_edges = []
            max_attempts = min(pos_edge_index.size(1) * 2, 100)  # Limit attempts
            attempts = 0
            
            while len(neg_edges) < pos_edge_index.size(1) and attempts < max_attempts:
                src = torch.randint(0, num_nodes, (1,)).item()
                dst = torch.randint(0, num_nodes, (1,)).item()
                if src != dst and (src, dst) not in pos_edge_set:
                    neg_edges.append([src, dst])
                attempts += 1
            
            if len(neg_edges) == 0:
                # Fallback: just use positive edge loss
                edge_loss = nn.BCELoss(reduction='mean')(pos_edge_preds, pos_edge_labels)
            else:
                neg_edge_index = torch.tensor(neg_edges, device=pos_edge_index.device).T
                neg_edge_preds = autoencoder.decode_edge(z_graph, neg_edge_index)
                neg_edge_labels = torch.zeros(neg_edge_preds.size(0), device=neg_edge_preds.device)
                
                edge_preds = torch.cat([pos_edge_preds, neg_edge_preds])
                edge_labels = torch.cat([pos_edge_labels, neg_edge_labels])
                edge_loss = nn.BCELoss(reduction='mean')(edge_preds, edge_labels)
            
            edge_errors.append(edge_loss.item())
            start += num_nodes
    return edge_errors

def compute_graph_structural_features(graphs):
    """Compute structural features for each graph to distinguish normal and attack graphs.

    Args:
        graphs: List of graph objects.

    Returns:
        List of structural feature scores, one per graph.
    """
    structural_errors = []
    
    for graph in graphs:
        num_nodes = graph.x.size(0)
        num_edges = graph.edge_index.size(1)
        
        if num_nodes == 0:
            structural_errors.append(0.0)
            continue
        
        # Basic structural metrics
        edge_density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        
        # Node degree distribution
        if num_edges > 0:
            edge_index = graph.edge_index
            degrees = torch.zeros(num_nodes, device=edge_index.device)
            degrees = degrees.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
            degrees = degrees.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=edge_index.device))
            degree_std = degrees.std().item()
        else:
            degree_std = 0.0
        
        # Combine structural features
        structural_score = edge_density * 10 + avg_degree * 0.1 + degree_std * 0.01
        structural_errors.append(structural_score)
    
    return structural_errors

def compute_edge_prediction_accuracy(autoencoder, batch):
    """Compute edge prediction accuracy (as error) for each graph in a batch.

    Args:
        autoencoder: The trained autoencoder model.
        batch: A batch of graphs.

    Returns:
        List of edge prediction errors (1 - accuracy), one per graph.
    """
    with torch.no_grad():
        _, _, z, _ = autoencoder(batch.x, batch.edge_index, batch.batch)
        graphs = Batch.to_data_list(batch)
        start = 0
        edge_accuracies = []
        
        for graph in graphs:
            num_nodes = graph.x.size(0)
            pos_edge_index = graph.edge_index
            
            if pos_edge_index.size(1) == 0 or num_nodes < 2:
                edge_accuracies.append(1.0)  # Perfect "accuracy" for trivial cases
                start += num_nodes
                continue
            
            z_graph = z[start:start+num_nodes]
            
            # Positive edges
            pos_edge_preds = autoencoder.decode_edge(z_graph, pos_edge_index)
            pos_predictions = (pos_edge_preds > 0.5).float()
            pos_accuracy = pos_predictions.mean().item()
            
            # Negative edges (sample systematically)
            neg_edges = []
            pos_edge_set = set()
            for i in range(pos_edge_index.size(1)):
                src, dst = pos_edge_index[0, i].item(), pos_edge_index[1, i].item()
                pos_edge_set.add((min(src, dst), max(src, dst)))
            
            # Sample negative edges more systematically
            for src in range(num_nodes):
                for dst in range(src + 1, num_nodes):
                    if (src, dst) not in pos_edge_set:
                        neg_edges.append([src, dst])
                        if len(neg_edges) >= pos_edge_index.size(1):
                            break
                if len(neg_edges) >= pos_edge_index.size(1):
                    break
            
            if neg_edges:
                neg_edge_index = torch.tensor(neg_edges, device=pos_edge_index.device).T
                neg_edge_preds = autoencoder.decode_edge(z_graph, neg_edge_index)
                neg_predictions = (neg_edge_preds < 0.5).float()  # Should predict 0
                neg_accuracy = neg_predictions.mean().item()
                
                # Combined accuracy
                total_accuracy = (pos_accuracy + neg_accuracy) / 2
            else:
                total_accuracy = pos_accuracy
            
            # Convert accuracy to error (1 - accuracy gives higher values for worse performance)
            edge_error = 1.0 - total_accuracy
            edge_accuracies.append(edge_error)
            start += num_nodes
    
    return edge_accuracies

def compute_graph_connectivity_anomalies(graphs):
    """Detect anomalies in graph connectivity patterns.

    Args:
        graphs: List of graph objects.

    Returns:
        List of connectivity anomaly scores, one per graph.
    """
    connectivity_scores = []
    
    for graph in graphs:
        num_nodes = graph.x.size(0)
        edge_index = graph.edge_index
        
        if num_nodes < 2:
            connectivity_scores.append(0.0)
            continue
        
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        if edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1
            adj[edge_index[1], edge_index[0]] = 1  # Make symmetric
        
        # Compute connectivity metrics
        num_edges = edge_index.size(1)
        expected_edges = num_nodes * (num_nodes - 1) / 2  # Complete graph
        
        # Measure deviation from expected patterns
        # For CAN networks, we might expect certain connectivity patterns
        connectivity_ratio = num_edges / expected_edges if expected_edges > 0 else 0
        
        # Check for isolated nodes
        degrees = adj.sum(dim=1)
        isolated_nodes = (degrees == 0).sum().item()
        isolation_penalty = isolated_nodes / num_nodes
        
        # Unusual connectivity score
        connectivity_anomaly = abs(connectivity_ratio - 0.1) + isolation_penalty  # Assume normal graphs have ~10% connectivity
        connectivity_scores.append(connectivity_anomaly)
    
    return connectivity_scores

class GATPipeline:
    """Pipeline for training and evaluating GAT-based autoencoder and classifier."""
    def __init__(self, num_ids, embedding_dim=8, device='cpu'):
        """Initialize the GATPipeline.

        Args:
            num_ids: Number of unique CAN IDs.
            embedding_dim: Dimension of latent embeddings.
            device: Device to use ('cpu' or 'cuda').
        """
        self.device = device
        self.autoencoder = GraphAutoencoder(num_ids=num_ids, in_channels=11, embedding_dim=embedding_dim).to(device)
        self.classifier = GATWithJK(num_ids=num_ids, in_channels=11, hidden_channels=32, out_channels=1, num_layers=3, heads=8, embedding_dim=embedding_dim).to(device)
        self.threshold = 0.0000

    def _compute_edge_loss(self, z, edge_index):
        """Compute edge reconstruction loss for a graph.

        Args:
            z: Latent node embeddings.
            edge_index: Edge indices.

        Returns:
            Edge reconstruction loss (torch.Tensor).
        """
        # Skip if no edges
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        pos_edge_preds = self.autoencoder.decode_edge(z, edge_index)
        pos_edge_labels = torch.ones(pos_edge_preds.size(0), device=pos_edge_preds.device)
        
        num_nodes = z.size(0)
        if num_nodes < 2:
            return nn.BCELoss()(pos_edge_preds, pos_edge_labels)
        
        # Better negative edge sampling
        pos_edge_set = set()
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            pos_edge_set.add((src, dst))
            pos_edge_set.add((dst, src))
        
        # Generate negative edges more systematically
        neg_edges = []
        for _ in range(min(edge_index.size(1), num_nodes * (num_nodes - 1) // 4)):
            attempts = 0
            while attempts < 10:  # Limit attempts per negative edge
                src = torch.randint(0, num_nodes, (1,), device=edge_index.device).item()
                dst = torch.randint(0, num_nodes, (1,), device=edge_index.device).item()
                if src != dst and (src, dst) not in pos_edge_set:
                    neg_edges.append([src, dst])
                    break
                attempts += 1
        
        if not neg_edges:
            return nn.BCELoss()(pos_edge_preds, pos_edge_labels)
        
        neg_edge_index = torch.tensor(neg_edges, device=edge_index.device).T
        neg_edge_preds = self.autoencoder.decode_edge(z, neg_edge_index)
        neg_edge_labels = torch.zeros(neg_edge_preds.size(0), device=neg_edge_preds.device)
        
        edge_preds = torch.cat([pos_edge_preds, neg_edge_preds])
        edge_labels = torch.cat([pos_edge_labels, neg_edge_labels])
        
        return nn.BCELoss()(edge_preds, edge_labels)

    def _compute_reconstruction_errors(self, loader, use_max=True):
        """Compute various reconstruction errors for normal and attack graphs.

        Args:
            loader: DataLoader yielding batches of graphs.
            use_max: If True, use max node error per graph; else use mean.

        Returns:
            Tuple of lists containing errors for normal and attack graphs.
        """
        errors_normal, errors_attack = [], []
        edge_errors_normal, edge_errors_attack = [], []
        id_errors_normal, id_errors_attack = [], []
        structural_errors_normal, structural_errors_attack = [], []
        connectivity_errors_normal, connectivity_errors_attack = [], []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                cont_out, canid_logits, z, _ = self.autoencoder(batch.x, batch.edge_index, batch.batch)
                
                # Node reconstruction errors
                node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                
                # Multiple edge-based approaches
                graphs = Batch.to_data_list(batch)
                
                # Traditional edge reconstruction errors
                edge_errors = compute_edge_prediction_accuracy(self.autoencoder, batch)
                
                # Structural feature errors
                structural_errors = compute_graph_structural_features(graphs)
                
                # Connectivity anomaly scores
                connectivity_errors = compute_graph_connectivity_anomalies(graphs)
                
                # CAN ID errors
                canid_pred = canid_logits.argmax(dim=1)
                
                start = 0
                for i, graph in enumerate(graphs):
                    num_nodes = graph.x.size(0)
                    is_attack = int(graph.y.flatten()[0]) == 1
                    
                    # Node error (max or mean)
                    graph_node_errors = node_errors[start:start+num_nodes]
                    if graph_node_errors.numel() > 0:
                        node_error = graph_node_errors.max().item() if use_max else graph_node_errors.mean().item()
                        (errors_attack if is_attack else errors_normal).append(node_error)
                    
                    # Various edge-based errors
                    (edge_errors_attack if is_attack else edge_errors_normal).append(edge_errors[i])
                    (structural_errors_attack if is_attack else structural_errors_normal).append(structural_errors[i])
                    (connectivity_errors_attack if is_attack else connectivity_errors_normal).append(connectivity_errors[i])
                    
                    # CAN ID error
                    true_canids = graph.x[:, 0].long().cpu()
                    pred_canids = canid_pred[start:start+num_nodes].cpu()
                    id_error = 1.0 - (pred_canids == true_canids).float().mean().item()
                    (id_errors_attack if is_attack else id_errors_normal).append(id_error)
                    
                    start += num_nodes
        
        return (errors_normal, errors_attack, edge_errors_normal, 
                edge_errors_attack, id_errors_normal, id_errors_attack,
                structural_errors_normal, structural_errors_attack,
                connectivity_errors_normal, connectivity_errors_attack)

    def _set_threshold(self, train_loader, percentile=50):
        """Set the anomaly detection threshold based on training data.

        Args:
            train_loader: DataLoader for training data.
            percentile: Percentile to use for threshold.
        """
        errors = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                cont_out, _, _, _ = self.autoencoder(batch.x, batch.edge_index, batch.batch)
                errors.append((cont_out - batch.x[:, 1:]).pow(2).mean(dim=1))
        self.threshold = torch.cat(errors).quantile(percentile / 100.0).item()

    def _create_balanced_dataset(self, filtered_graphs):
        """Create a balanced dataset by downsampling the majority class.

        Args:
            filtered_graphs: List of filtered graphs.

        Returns:
            List of balanced graphs.
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

    def _print_error_statistics(self, errors_normal, errors_attack, edge_errors_normal, 
                               edge_errors_attack, id_errors_normal, id_errors_attack,
                               structural_errors_normal=None, structural_errors_attack=None,
                               connectivity_errors_normal=None, connectivity_errors_attack=None):
        """Print error statistics and generate plots for various error types.

        Args:
            errors_normal: Node errors for normal graphs.
            errors_attack: Node errors for attack graphs.
            edge_errors_normal: Edge errors for normal graphs.
            edge_errors_attack: Edge errors for attack graphs.
            id_errors_normal: CAN ID errors for normal graphs.
            id_errors_attack: CAN ID errors for attack graphs.
            structural_errors_normal: Structural feature scores for normal graphs.
            structural_errors_attack: Structural feature scores for attack graphs.
            connectivity_errors_normal: Connectivity anomaly scores for normal graphs.
            connectivity_errors_attack: Connectivity anomaly scores for attack graphs.
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
            train_loader: DataLoader for normal graphs.
            epochs: Number of training epochs.
        """
        self.autoencoder.train()
        opt = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-2, weight_decay=1e-4)
        
        for _ in range(epochs):
            for batch in train_loader:
                batch = batch.to(self.device)
                cont_out, canid_logits, z, kl_loss = self.autoencoder(batch.x, batch.edge_index, batch.batch)
                
                # Compute losses
                cont_loss = (cont_out - batch.x[:, 1:]).pow(2).mean()
                canid_loss = nn.CrossEntropyLoss()(canid_logits, batch.x[:, 0].long())
                edge_loss = self._compute_edge_loss(z, batch.edge_index)
                
                # Combine losses
                beta = 0.1
                loss = cont_loss + canid_loss + edge_loss + beta * kl_loss
                
                opt.zero_grad()
                loss.backward()
                opt.step()
        
        # Set threshold
        self._set_threshold(train_loader, percentile=50)

    def train_stage2(self, full_loader, epochs=50):
        """Train the classifier on filtered graphs (Stage 2).

        Args:
            full_loader: DataLoader for all graphs (normal + attack).
            epochs: Number of training epochs.
        """
        # Print label distribution
        all_labels = [batch.y.cpu().numpy() for batch in full_loader]
        all_labels = np.concatenate(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)
        print("Label distribution in full training set (Stage 2):")
        for u, c in zip(unique, counts):
            print(f"Label {u}: {c} samples")

        # Compute and analyze reconstruction errors with new metrics
        result = self._compute_reconstruction_errors(full_loader)
        if len(result) == 10:  # New version with structural metrics
            (errors_normal, errors_attack, edge_errors_normal, 
             edge_errors_attack, id_errors_normal, id_errors_attack,
             structural_errors_normal, structural_errors_attack,
             connectivity_errors_normal, connectivity_errors_attack) = result
            
            self._print_error_statistics(errors_normal, errors_attack, edge_errors_normal,
                                       edge_errors_attack, id_errors_normal, id_errors_attack,
                                       structural_errors_normal, structural_errors_attack,
                                       connectivity_errors_normal, connectivity_errors_attack)
        else:  # Fallback to old version
            (errors_normal, errors_attack, edge_errors_normal, 
             edge_errors_attack, id_errors_normal, id_errors_attack) = result
            
            self._print_error_statistics(errors_normal, errors_attack, edge_errors_normal,
                                       edge_errors_attack, id_errors_normal, id_errors_attack)

        # Filter graphs based on threshold
        filtered = self._filter_anomalous_graphs(full_loader)
        
        if not filtered:
            print("No graphs exceeded the anomaly threshold. Skipping classifier training.")
            return

        # Create balanced dataset
        filtered = self._create_balanced_dataset(filtered)
        
        # Print filtered dataset statistics
        labels = [int(graph.y.flatten()[0]) for graph in filtered]
        unique, counts = np.unique(labels, return_counts=True)
        print("Label distribution in filtered graphs (for classifier):")
        for u, c in zip(unique, counts):
            print(f"Label {u}: {c} samples")

        # Train classifier
        self._train_classifier(filtered, epochs)

    def _filter_anomalous_graphs(self, loader):
        """Filter graphs that exceed the anomaly threshold.

        Args:
            loader: DataLoader yielding batches of graphs.

        Returns:
            List of graphs exceeding the anomaly threshold.
        """
        filtered = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                cont_out, _, _, _ = self.autoencoder(batch.x, batch.edge_index, batch.batch)
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
            filtered_graphs: List of graphs for classifier training.
            epochs: Number of training epochs.
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
            graphs: List of graphs to evaluate.

        Returns:
            Classification accuracy (float).
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
            data: Batch of graphs.

        Returns:
            Tensor of predicted labels (0 for normal, 1 for attack).
        """
        data = data.to(self.device)
        
        # Stage 1: Anomaly detection
        with torch.no_grad():
            cont_out, canid_logits, z, kl_loss = self.autoencoder(data.x, data.edge_index, data.batch)
            error = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            # Process each graph in Batch
            preds = []
            for graph in Batch.to_data_list(data):
                node_mask = (data.batch == graph.batch)
                if (error[node_mask] > self.threshold).any():
                    # Stage 2: Classification
                    prob = self.classifier(graph.x, graph.edge_index, 
                                         graph.batch).item()
                    preds.append(1 if prob > 0.5 else 0)
                else:
                    preds.append(0)
            
            return torch.tensor(preds, device=self.device)


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    """Main function for training and evaluating the anomaly detection pipeline.

    Args:
        config: Hydra configuration object.
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
    pipeline.train_stage1(train_loader, epochs=50)

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