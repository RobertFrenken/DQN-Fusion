import torch
import torch.nn as nn
import numpy as np
import os
import gc
import random
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import confusion_matrix, classification_report
from models.adaptive_fusion import QFusionAgent
from models.models import GATWithJK, GraphAutoencoderNeighborhood
from utils.plotting_utils import (
    plot_recon_error_hist,
    plot_neighborhood_error_hist,
    plot_error_components_analysis,
    plot_graph_reconstruction,
    plot_latent_space,
    plot_node_recon_errors,
)

FUSION_WEIGHTS = {
    'node_reconstruction': 1.0,
    'neighborhood_prediction': 20.0,
    'can_id_prediction': 0.3
}

class GATPipeline:
    """
    Two-stage GAT pipeline for CAN bus anomaly detection.
    
    Stage 1: Graph autoencoder training on normal data for anomaly detection
    Stage 2: GAT classifier training on filtered data for attack classification
    """
    
    def __init__(self, num_ids: int, embedding_dim: int = 8, device: str = 'cpu'):
        """
        Initialize the two-stage pipeline.
        
        Args:
            num_ids: Number of unique CAN IDs in the dataset
            embedding_dim: Dimensionality of node embeddings
            device: Target device for computation
        """
        self.device = torch.device(device)
        self.is_cuda = torch.cuda.is_available() and self.device.type == 'cuda'
        
        # Initialize models
        self.autoencoder = GraphAutoencoderNeighborhood(
            num_ids=num_ids, 
            in_channels=11, 
            embedding_dim=embedding_dim
        ).to(self.device)
        
        self.classifier = GATWithJK(
            num_ids=num_ids, 
            in_channels=11, 
            hidden_channels=32,
            out_channels=1, 
            num_layers=5, 
            heads=8, 
            embedding_dim=embedding_dim
        ).to(self.device)

        # Initialize Q-learning fusion agent
        self.fusion_agent = QFusionAgent(
            alpha_steps=21,     # 0.0, 0.05, 0.10, ..., 1.0
            state_bins=10,      # 10x10 state grid
            lr=0.1,             # Learning rate
            gamma=0.9,          # Discount factor
            epsilon=0.2,        # Initial exploration
            epsilon_decay=0.995,
            min_epsilon=0.01
        )
        
        self.threshold = 0.0
        print(f"✓ GAT Pipeline initialized on {self.device}")

    def train_stage1(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
                    epochs: int = 20) -> None:
        """
        Stage 1: Train autoencoder on normal graphs for anomaly detection.
        
        Args:
            train_loader: DataLoader with normal graphs only
            val_loader: Optional validation DataLoader
            epochs: Number of training epochs
        """
        print(f"\n=== Stage 1: Autoencoder Training ({epochs} epochs) ===")
        self.autoencoder.train()
        
        # Setup optimization
        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(), 
            lr=2e-3, 
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3, verbose=True
        )
        scaler = torch.cuda.amp.GradScaler() if self.is_cuda else None
        
        # Training state
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.autoencoder.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                if self.is_cuda:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        loss = self._compute_autoencoder_loss(batch)
                else:
                    loss = self._compute_autoencoder_loss(batch)
                
                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                    optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Memory cleanup
                if batch_idx % 20 == 0:
                    self._cleanup_memory()
            
            train_avg_loss = epoch_loss / max(num_batches, 1)
            
            # Validation phase
            val_avg_loss = None
            if val_loader is not None:
                val_avg_loss = self._validate_autoencoder(val_loader)
                current_loss = val_avg_loss
                print(f"Epoch {epoch+1}/{epochs} - Train: {train_avg_loss:.4f}, Val: {val_avg_loss:.4f}")
            else:
                current_loss = train_avg_loss
                print(f"Epoch {epoch+1}/{epochs} - Train: {train_avg_loss:.4f}")
            
            # Model checkpointing
            if current_loss < best_loss * 0.999:
                best_loss = current_loss
                patience_counter = 0
                best_model_state = {
                    'state_dict': self.autoencoder.state_dict().copy(),
                    'epoch': epoch + 1,
                    'train_loss': train_avg_loss,
                    'val_loss': val_avg_loss
                }
                print(f"  → New best model saved (loss: {current_loss:.4f})")
            else:
                patience_counter += 1
            
            scheduler.step(current_loss)
            
            # Early stopping
            if patience_counter >= 5:
                print(f"Early stopping: No improvement for {patience_counter} epochs")
                break
            
            # Periodic memory logging
            if (epoch + 1) % 5 == 0:
                log_memory_usage(f"Autoencoder Epoch {epoch+1}")
        
        # Restore best model
        if best_model_state is not None:
            self.autoencoder.load_state_dict(best_model_state['state_dict'])
            print(f"✓ Restored best autoencoder from epoch {best_model_state['epoch']}")
        
        # Set anomaly detection threshold
        self._set_threshold(train_loader)
        print(f"✓ Anomaly threshold set: {self.threshold:.4f}")

    def train_stage2(self, full_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
                    epochs: int = 10, key_suffix: str = "") -> None:
        """
        Stage 2: Train classifier on filtered graphs.
        
        Args:
            full_loader: DataLoader with all training graphs
            val_loader: Optional validation DataLoader
            epochs: Number of training epochs
            key_suffix: Suffix for saved plots and logs
        """
        print(f"\n=== Stage 2: Classifier Training ({epochs} epochs) ===")
        
        # Analyze reconstruction errors
        self._analyze_reconstruction_errors(full_loader, key_suffix)
        
        # Create balanced dataset for classifier training
        balanced_graphs = self._create_balanced_dataset(full_loader)
        if not balanced_graphs:
            print("❌ No graphs available for classifier training")
            return
        
        # Prepare validation graphs if available
        val_graphs = None
        if val_loader is not None:
            val_graphs = self._prepare_validation_graphs(val_loader)
        
        # Train classifier
        self._train_classifier(balanced_graphs, val_graphs, epochs)

    def _compute_autoencoder_loss(self, batch) -> torch.Tensor:
        """Compute multi-component autoencoder loss."""
        cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
            batch.x, batch.edge_index, batch.batch)
        
        # Component losses
        recon_loss = nn.MSELoss()(cont_out, batch.x[:, 1:])
        canid_loss = nn.CrossEntropyLoss()(canid_logits, batch.x[:, 0].long())
        neighbor_loss = self._compute_neighborhood_loss(neighbor_logits, batch.x, batch.edge_index)
        
        # Weighted combination
        total_loss = recon_loss + 0.1 * canid_loss + 0.5 * neighbor_loss
        return total_loss

    def _compute_neighborhood_loss(self, neighbor_logits: torch.Tensor, 
                                 x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute neighborhood reconstruction loss."""
        neighbor_targets = self.autoencoder.create_neighborhood_targets(x, edge_index, None)
        return nn.BCEWithLogitsLoss()(neighbor_logits, neighbor_targets)

    def _validate_autoencoder(self, val_loader: DataLoader) -> float:
        """Validate autoencoder and return average loss."""
        self.autoencoder.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device, non_blocking=True)
                
                if self.is_cuda:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        loss = self._compute_autoencoder_loss(batch)
                else:
                    loss = self._compute_autoencoder_loss(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)

    def _set_threshold(self, train_loader: DataLoader, percentile: int = 50) -> None:
        """Set anomaly detection threshold based on training data."""
        errors = []
        self.autoencoder.eval()
        
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                cont_out, _, _, _, _ = self.autoencoder(batch.x, batch.edge_index, batch.batch)
                node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                errors.append(node_errors)
        
        all_errors = torch.cat(errors)
        self.threshold = all_errors.quantile(percentile / 100.0).item()

    def _analyze_reconstruction_errors(self, loader: DataLoader, key_suffix: str) -> None:
        """Analyze and visualize reconstruction errors."""
        print("Analyzing reconstruction errors...")
        
        # Compute detailed error statistics
        error_stats = self._compute_detailed_errors(loader)
        
        # Print statistics
        self._print_error_statistics(error_stats)
        
        # Generate visualizations
        self._generate_error_plots(error_stats, key_suffix)

    def _compute_detailed_errors(self, loader: DataLoader) -> Dict[str, List[float]]:
        """Compute detailed reconstruction errors for all graphs."""
        error_stats = {
            'normal_node': [], 'attack_node': [],
            'normal_neighbor': [], 'attack_neighbor': [],
            'normal_canid': [], 'attack_canid': []
        }
        
        self.autoencoder.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                    batch.x, batch.edge_index, batch.batch)
                
                # Compute component errors
                node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                neighbor_targets = self.autoencoder.create_neighborhood_targets(
                    batch.x, batch.edge_index, batch.batch)
                neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                    neighbor_logits, neighbor_targets).mean(dim=1)
                canid_pred = canid_logits.argmax(dim=1)
                
                # Process each graph
                graphs = Batch.to_data_list(batch)
                start = 0
                for graph in graphs:
                    num_nodes = graph.x.size(0)
                    is_attack = int(graph.y.flatten()[0]) == 1
                    prefix = 'attack' if is_attack else 'normal'
                    
                    # Extract graph-level errors
                    graph_node_error = node_errors[start:start+num_nodes].max().item()
                    graph_neighbor_error = neighbor_errors[start:start+num_nodes].max().item()
                    
                    true_canids = graph.x[:, 0].long().cpu()
                    pred_canids = canid_pred[start:start+num_nodes].cpu()
                    canid_error = 1.0 - (pred_canids == true_canids).float().mean().item()
                    
                    # Store errors
                    error_stats[f'{prefix}_node'].append(graph_node_error)
                    error_stats[f'{prefix}_neighbor'].append(graph_neighbor_error)
                    error_stats[f'{prefix}_canid'].append(canid_error)
                    
                    start += num_nodes
        
        return error_stats

    def _print_error_statistics(self, error_stats: Dict[str, List[float]]) -> None:
        """Print detailed error statistics."""
        print(f"\n📊 Reconstruction Error Analysis")
        print(f"{'='*60}")
        
        for component in ['node', 'neighbor', 'canid']:
            normal_errors = error_stats[f'normal_{component}']
            attack_errors = error_stats[f'attack_{component}']
            
            if normal_errors and attack_errors:
                print(f"\n{component.title()} Errors:")
                print(f"  Normal: {np.mean(normal_errors):.4f} ± {np.std(normal_errors):.4f}")
                print(f"  Attack: {np.mean(attack_errors):.4f} ± {np.std(attack_errors):.4f}")

    def _generate_error_plots(self, error_stats: Dict[str, List[float]], key_suffix: str) -> None:
        """Generate error analysis plots."""
        # Extract data for plotting
        normal_node = error_stats['normal_node']
        attack_node = error_stats['attack_node']
        normal_neighbor = error_stats['normal_neighbor']
        attack_neighbor = error_stats['attack_neighbor']
        normal_canid = error_stats['normal_canid']
        attack_canid = error_stats['attack_canid']
        
        # Generate plots
        if normal_node and attack_node:
            plot_recon_error_hist(normal_node, attack_node, self.threshold,
                                save_path=f"images/recon_error_hist_{key_suffix}.png")
            
            neighbor_threshold = np.percentile(normal_neighbor, 95) if normal_neighbor else 0.5
            plot_neighborhood_error_hist(normal_neighbor, attack_neighbor, neighbor_threshold,
                                        save_path=f"images/neighborhood_error_hist_{key_suffix}.png")
            
            plot_error_components_analysis(
                normal_node, attack_node, normal_neighbor, attack_neighbor,
                normal_canid, attack_canid,
                save_path=f"images/error_components_analysis_{key_suffix}.png")

    def _create_balanced_dataset(self, loader: DataLoader) -> List:
        """Create balanced dataset for classifier training using composite filtering."""
        print("Creating balanced dataset with composite filtering...")
        
        # Compute composite errors for all graphs
        graph_data = self._compute_composite_errors(loader)
        
        # Separate by class
        attack_graphs = [(graph, error) for graph, error, is_attack in graph_data if is_attack]
        normal_graphs = [(graph, error) for graph, error, is_attack in graph_data if not is_attack]
        
        print(f"Found {len(attack_graphs)} attack, {len(normal_graphs)} normal graphs")
        
        # Use all attacks
        selected_attacks = [graph for graph, _ in attack_graphs]
        num_attacks = len(selected_attacks)
        
        if num_attacks == 0:
            print("❌ No attack graphs found")
            return []
        
        # Balance with hardest normal examples (4:1 ratio max)
        max_normal = num_attacks * 4
        
        if len(normal_graphs) <= max_normal:
            selected_normal = [graph for graph, _ in normal_graphs]
        else:
            # Sort by composite error and take hardest examples
            normal_graphs_sorted = sorted(normal_graphs, key=lambda x: x[1])
            selected_normal = [graph for graph, _ in normal_graphs_sorted[-max_normal:]]
            print(f"Filtered normal graphs: {len(normal_graphs)} → {len(selected_normal)}")
        
        # Combine and shuffle
        balanced_graphs = selected_attacks + selected_normal
        random.seed(42)
        random.shuffle(balanced_graphs)
        
        print(f"✓ Balanced dataset: {len(selected_normal)} normal, {num_attacks} attack")
        return balanced_graphs

    def _compute_composite_errors(self, loader: DataLoader) -> List[Tuple]:
        """Compute composite reconstruction errors for graph filtering."""
        all_data = []
        
        self.autoencoder.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                    batch.x, batch.edge_index, batch.batch)
                
                # Compute component errors
                node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                neighbor_targets = self.autoencoder.create_neighborhood_targets(
                    batch.x, batch.edge_index, batch.batch)
                neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                    neighbor_logits, neighbor_targets).mean(dim=1)
                canid_pred = canid_logits.argmax(dim=1)
                
                # Process each graph
                graphs = Batch.to_data_list(batch)
                start = 0
                for graph in graphs:
                    num_nodes = graph.x.size(0)
                    is_attack = int(graph.y.flatten()[0]) == 1
                    
                    # Graph-level errors
                    node_error = node_errors[start:start+num_nodes].max().item()
                    neighbor_error = neighbor_errors[start:start+num_nodes].max().item()
                    
                    true_canids = graph.x[:, 0].long()
                    pred_canids = canid_pred[start:start+num_nodes]
                    canid_error = 1.0 - (pred_canids == true_canids).float().mean().item()
                    
                    # Composite error with learned weights
                    composite_error = (
                        FUSION_WEIGHTS['node_reconstruction'] * node_error +
                        FUSION_WEIGHTS['neighborhood_prediction'] * neighbor_error +
                        FUSION_WEIGHTS['can_id_prediction'] * canid_error
                    )
                    
                    all_data.append((graph.cpu(), composite_error, is_attack))
                    start += num_nodes
        
        return all_data

    def _prepare_validation_graphs(self, val_loader: DataLoader) -> List:
        """Prepare validation graphs for classifier training."""
        print("Preparing validation graphs...")
        val_data = self._compute_composite_errors(val_loader)
        val_graphs = [graph for graph, _, _ in val_data]
        print(f"✓ Prepared {len(val_graphs)} validation graphs")
        return val_graphs

    def _train_classifier(self, balanced_graphs: List, val_graphs: Optional[List] = None, 
                         epochs: int = 20) -> None:
        """Train binary GAT classifier."""
        print(f"Training classifier on {len(balanced_graphs)} graphs...")
        
        # Setup training
        labels = [int(graph.y.flatten()[0]) for graph in balanced_graphs]
        num_pos, num_neg = sum(labels), len(labels) - sum(labels)
        pos_weight = torch.tensor(1.0 if num_pos == 0 else num_neg / num_pos, device=self.device)
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=3, verbose=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler = torch.cuda.amp.GradScaler() if self.is_cuda else None
        
        # Training state
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        batch_size = 1024 if torch.cuda.is_available() else 256
        
        for epoch in range(epochs):
            # Training phase
            self.classifier.train()
            epoch_loss = 0.0
            num_batches = 0
            
            train_loader = DataLoader(balanced_graphs, batch_size=batch_size, shuffle=True)
            
            for batch in train_loader:
                batch = batch.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.is_cuda:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        preds = self.classifier(batch)
                        loss = criterion(preds.squeeze(), batch.y.float())
                else:
                    preds = self.classifier(batch)
                    loss = criterion(preds.squeeze(), batch.y.float())
                
                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                    optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            train_avg_loss = epoch_loss / max(num_batches, 1)
            
            # Validation phase
            val_avg_loss = None
            if val_graphs is not None:
                val_avg_loss = self._validate_classifier(val_graphs, criterion, batch_size)
                current_loss = val_avg_loss
                print(f"Epoch {epoch+1}/{epochs} - Train: {train_avg_loss:.4f}, Val: {val_avg_loss:.4f}")
            else:
                current_loss = train_avg_loss
                print(f"Epoch {epoch+1}/{epochs} - Train: {train_avg_loss:.4f}")
            
            # Model checkpointing
            if current_loss < best_loss * 0.999:
                best_loss = current_loss
                patience_counter = 0
                best_model_state = self.classifier.state_dict().copy()
                print(f"  → New best classifier saved (loss: {current_loss:.4f})")
            else:
                patience_counter += 1
            
            scheduler.step(current_loss)
            
            # Early stopping
            if patience_counter >= 5:
                print(f"Early stopping: No improvement for {patience_counter} epochs")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.classifier.load_state_dict(best_model_state)
            print(f"✓ Restored best classifier")

    def _validate_classifier(self, val_graphs: List, criterion, batch_size: int) -> float:
        """Validate classifier and return average loss."""
        self.classifier.eval()
        total_loss = 0.0
        num_batches = 0
        
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device, non_blocking=True)
                
                if self.is_cuda:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        preds = self.classifier(batch)
                        loss = criterion(preds.squeeze(), batch.y.float())
                else:
                    preds = self.classifier(batch)
                    loss = criterion(preds.squeeze(), batch.y.float())
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)

    def predict(self, data) -> torch.Tensor:
        """Two-stage prediction: anomaly detection followed by classification."""
        data = data.to(self.device)
        
        with torch.no_grad():
            # Stage 1: Anomaly detection
            cont_out, _, _, _, _ = self.autoencoder(data.x, data.edge_index, data.batch)
            node_errors = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            # Stage 2: Classification for flagged graphs
            predictions = []
            graphs = Batch.to_data_list(data)
            start = 0
            
            for graph in graphs:
                num_nodes = graph.x.size(0)
                graph_errors = node_errors[start:start+num_nodes]
                
                if graph_errors.numel() > 0 and (graph_errors > self.threshold).any():
                    # Classify as potentially malicious
                    graph_batch = graph.to(self.device)
                    logit = self.classifier(graph_batch).item()
                    pred = 1 if logit > 0.0 else 0
                else:
                    # Normal by anomaly detection
                    pred = 0
                
                predictions.append(pred)
                start += num_nodes
        
        return torch.tensor(predictions, device=self.device)

    def predict_with_fusion(self, data, alpha: float = 0.85) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction with fusion of anomaly detection and classification scores.
        
        Args:
            data: Input batch data
            alpha: Weight for GAT classifier (0.0-1.0), remaining weight for anomaly detection
        
        Returns:
            Tuple of (final_predictions, anomaly_scores, gat_probabilities)
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            # Get autoencoder outputs
            cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                data.x, data.edge_index, data.batch)
            
            # Compute composite anomaly scores
            node_errors = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            neighbor_targets = self.autoencoder.create_neighborhood_targets(
                data.x, data.edge_index, data.batch)
            neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                neighbor_logits, neighbor_targets).mean(dim=1)
            canid_pred = canid_logits.argmax(dim=1)
            
            final_preds = []
            anomaly_scores = []
            gat_probs = []
            
            graphs = Batch.to_data_list(data)
            start = 0
            
            for graph in graphs:
                num_nodes = graph.x.size(0)
                
                # Compute composite anomaly score
                node_error = node_errors[start:start+num_nodes].max().item()
                neighbor_error = neighbor_errors[start:start+num_nodes].max().item()
                
                true_canids = graph.x[:, 0].long().cpu()
                pred_canids = canid_pred[start:start+num_nodes].cpu()
                canid_error = 1.0 - (pred_canids == true_canids).float().mean().item()
                
                # Composite score with learned weights
                raw_anomaly_score = (
                    FUSION_WEIGHTS['node_reconstruction'] * node_error +
                    FUSION_WEIGHTS['neighborhood_prediction'] * neighbor_error +
                    FUSION_WEIGHTS['can_id_prediction'] * canid_error
                )
                
                # Normalize to [0,1]
                norm_anomaly_score = torch.sigmoid(torch.tensor(raw_anomaly_score * 10 - 5)).item()
                
                # Get GAT probability
                graph_batch = graph.to(self.device)
                gat_logit = self.classifier(graph_batch).item()
                gat_prob = torch.sigmoid(torch.tensor(gat_logit)).item()
                
                # Fusion (GAT-dominant strategy)
                fused_score = (1 - alpha) * norm_anomaly_score + alpha * gat_prob
                
                final_preds.append(1 if fused_score > 0.5 else 0)
                anomaly_scores.append(norm_anomaly_score)
                gat_probs.append(gat_prob)
                
                start += num_nodes
        
        return (torch.tensor(final_preds, device=self.device),
                torch.tensor(anomaly_scores),
                torch.tensor(gat_probs))
    
    def train_adaptive_fusion(self, train_loader: DataLoader, 
                            epochs: int = 5, validation_interval: int = 100) -> Dict:
        """
        Train the Q-learning fusion agent using online learning.
        
        Args:
            train_loader: Training data for fusion learning
            epochs: Number of training epochs
            validation_interval: Interval for validation and epsilon decay
            
        Returns:
            Dictionary with training statistics
        """
        print(f"\n=== Training Adaptive Fusion Agent ({epochs} epochs) ===")
        
        if not self.fusion_enabled:
            print("Enabling fusion mode...")
            self.fusion_enabled = True
        
        self.autoencoder.eval()
        self.classifier.eval()
        
        training_stats = {
            'epoch_accuracies': [],
            'epoch_rewards': [],
            'epsilon_values': [],
            'fusion_weights_used': []
        }
        
        for epoch in range(epochs):
            epoch_correct = 0
            epoch_total = 0
            epoch_rewards = []
            epoch_alphas = []
            
            print(f"\nEpoch {epoch + 1}/{epochs} (ε={self.fusion_agent.epsilon:.3f})")
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                
                # Get model outputs
                with torch.no_grad():
                    # Autoencoder outputs for anomaly detection
                    cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                        batch.x, batch.edge_index, batch.batch)
                    
                    # Compute anomaly scores
                    anomaly_scores = self._compute_normalized_anomaly_scores(
                        batch, cont_out, canid_logits, neighbor_logits)
                    
                    # GAT classifier outputs
                    gat_logits = self.classifier(batch)
                    gat_probs = torch.sigmoid(gat_logits.squeeze()).cpu().numpy()
                
                # Process each graph in the batch
                graphs = Batch.to_data_list(batch)
                true_labels = batch.y.cpu().numpy()
                
                for i, (graph, true_label, anomaly_score, gat_prob) in enumerate(
                    zip(graphs, true_labels, anomaly_scores, gat_probs)):
                    
                    # Select fusion weight
                    alpha, action_idx, state = self.fusion_agent.select_action(
                        anomaly_score, gat_prob, training=True)
                    
                    # Make fused prediction
                    fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
                    prediction = 1 if fused_score > 0.5 else 0
                    
                    # Compute reward
                    reward = self.fusion_agent.compute_reward(
                        prediction, int(true_label), anomaly_score, gat_prob)
                    
                    # Update Q-table
                    self.fusion_agent.update_q_table(state, action_idx, reward)
                    
                    # Track statistics
                    epoch_correct += (prediction == int(true_label))
                    epoch_total += 1
                    epoch_rewards.append(reward)
                    epoch_alphas.append(alpha)
                
                # Periodic validation and epsilon decay
                if batch_idx % validation_interval == 0 and batch_idx > 0:
                    self.fusion_agent.decay_epsilon()
                    
                    if batch_idx % (validation_interval * 2) == 0:
                        current_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
                        avg_reward = np.mean(epoch_rewards[-validation_interval:])
                        print(f"  Batch {batch_idx}: Acc={current_acc:.3f}, "
                              f"Reward={avg_reward:.3f}, α_avg={np.mean(epoch_alphas[-validation_interval:]):.3f}")
            
            # Epoch statistics
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            epoch_avg_reward = np.mean(epoch_rewards)
            epoch_avg_alpha = np.mean(epoch_alphas)
            
            training_stats['epoch_accuracies'].append(epoch_accuracy)
            training_stats['epoch_rewards'].append(epoch_avg_reward)
            training_stats['epsilon_values'].append(self.fusion_agent.epsilon)
            training_stats['fusion_weights_used'].append(epoch_avg_alpha)
            
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Accuracy: {epoch_accuracy:.4f}")
            print(f"  Avg Reward: {epoch_avg_reward:.4f}")
            print(f"  Avg Fusion Weight: {epoch_avg_alpha:.4f}")
            print(f"  Epsilon: {self.fusion_agent.epsilon:.4f}")
        
        print(f"\n✓ Adaptive fusion training complete!")
        self._plot_fusion_training_stats(training_stats)
        
        return training_stats
    
    def predict_with_adaptive_fusion(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction using learned adaptive fusion weights.
        
        Returns:
            Tuple of (predictions, anomaly_scores, gat_probs, fusion_weights)
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            # Get model outputs
            cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                data.x, data.edge_index, data.batch)
            gat_logits = self.classifier(data)
            gat_probs = torch.sigmoid(gat_logits.squeeze())
            
            # Compute anomaly scores
            anomaly_scores = self._compute_normalized_anomaly_scores(
                data, cont_out, canid_logits, neighbor_logits)
            
            predictions = []
            fusion_weights = []
            
            graphs = Batch.to_data_list(data)
            
            for i, (graph, anomaly_score, gat_prob) in enumerate(
                zip(graphs, anomaly_scores, gat_probs.cpu().numpy())):
                
                # Get learned fusion weight (no exploration during inference)
                alpha, _, _ = self.fusion_agent.select_action(
                    anomaly_score, gat_prob, training=False)
                
                # Fused prediction
                fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
                prediction = 1 if fused_score > 0.5 else 0
                
                predictions.append(prediction)
                fusion_weights.append(alpha)
        
        return (
            torch.tensor(predictions, device=self.device),
            torch.tensor(anomaly_scores),
            gat_probs.cpu(),
            torch.tensor(fusion_weights)
        )

    def evaluate_adaptive_fusion(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model using adaptive fusion and compare with fixed fusion.
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        print(f"\n=== Adaptive Fusion Evaluation ===")
        
        # Results storage
        all_labels = []
        all_adaptive_preds = []
        all_fixed_preds = []
        all_anomaly_scores = []
        all_gat_probs = []
        all_fusion_weights = []
        
        for batch in test_loader:
            batch = batch.to(self.device)
            labels = batch.y.cpu().numpy()
            
            # Adaptive fusion predictions
            adaptive_preds, anomaly_scores, gat_probs, fusion_weights = \
                self.predict_with_adaptive_fusion(batch)
            
            # Fixed fusion predictions (α=0.85)
            fixed_preds, _, _ = self.predict_with_fusion(batch, alpha=0.85)
            
            # Store results
            all_labels.extend(labels)
            all_adaptive_preds.extend(adaptive_preds.cpu().numpy())
            all_fixed_preds.extend(fixed_preds.cpu().numpy())
            all_anomaly_scores.extend(anomaly_scores.cpu().numpy())
            all_gat_probs.extend(gat_probs.cpu().numpy())
            all_fusion_weights.extend(fusion_weights.cpu().numpy())
        
        # Calculate metrics
        adaptive_accuracy = np.mean(np.array(all_adaptive_preds) == np.array(all_labels))
        fixed_accuracy = np.mean(np.array(all_fixed_preds) == np.array(all_labels))
        
        results = {
            'adaptive': {
                'accuracy': adaptive_accuracy,
                'confusion_matrix': confusion_matrix(all_labels, all_adaptive_preds),
                'report': classification_report(all_labels, all_adaptive_preds, output_dict=True)
            },
            'fixed': {
                'accuracy': fixed_accuracy,
                'confusion_matrix': confusion_matrix(all_labels, all_fixed_preds),
                'report': classification_report(all_labels, all_fixed_preds, output_dict=True)
            },
            'fusion_analysis': {
                'anomaly_scores': all_anomaly_scores,
                'gat_probs': all_gat_probs,
                'fusion_weights': all_fusion_weights,
                'labels': all_labels
            }
        }
        
        # Print comparison
        print(f"Adaptive Fusion Accuracy: {adaptive_accuracy:.4f}")
        print(f"Fixed Fusion Accuracy:    {fixed_accuracy:.4f}")
        print(f"Improvement: {(adaptive_accuracy - fixed_accuracy)*100:.2f}%")
        
        # Analyze fusion weight distribution
        avg_weight = np.mean(all_fusion_weights)
        print(f"Average Fusion Weight: {avg_weight:.3f} (vs. fixed 0.85)")
        
        # Generate analysis plots
        self._plot_fusion_analysis(results['fusion_analysis'])
        
        return results

    def save_fusion_agent(self, save_folder: str, dataset_key: str):
        """Save the trained fusion agent."""
        agent_path = os.path.join(save_folder, f'fusion_agent_{dataset_key}.pkl')
        self.fusion_agent.save_agent(agent_path)
    
    def load_fusion_agent(self, save_folder: str, dataset_key: str):
        """Load a trained fusion agent."""
        agent_path = os.path.join(save_folder, f'fusion_agent_{dataset_key}.pkl')
        self.fusion_agent.load_agent(agent_path)
        self.fusion_enabled = True

    def evaluate(self, test_loader: DataLoader, method: str = 'fusion') -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            method: Evaluation method ('standard', 'fusion', or 'both')
        
        Returns:
            Dictionary containing evaluation results
        """
        print(f"\n=== Model Evaluation ({method}) ===")
        
        all_labels = []
        all_standard_preds = []
        all_fusion_preds = []
        all_anomaly_scores = []
        all_gat_probs = []
        
        for batch in test_loader:
            batch = batch.to(self.device)
            labels = batch.y.cpu().numpy()
            
            if method in ['standard', 'both']:
                standard_preds = self.predict(batch).cpu().numpy()
                all_standard_preds.extend(standard_preds)
            
            if method in ['fusion', 'both']:
                fusion_preds, anomaly_scores, gat_probs = self.predict_with_fusion(batch)
                all_fusion_preds.extend(fusion_preds.cpu().numpy())
                all_anomaly_scores.extend(anomaly_scores.cpu().numpy())
                all_gat_probs.extend(gat_probs.cpu().numpy())
            
            all_labels.extend(labels)
        
        # Compile results
        results = {}
        
        if method in ['standard', 'both']:
            std_accuracy = np.mean(np.array(all_standard_preds) == np.array(all_labels))
            results['standard'] = {
                'accuracy': std_accuracy,
                'confusion_matrix': confusion_matrix(all_labels, all_standard_preds),
                'report': classification_report(all_labels, all_standard_preds, output_dict=True)
            }
            print(f"Standard Two-Stage Accuracy: {std_accuracy:.4f}")
        
        if method in ['fusion', 'both']:
            fusion_accuracy = np.mean(np.array(all_fusion_preds) == np.array(all_labels))
            results['fusion'] = {
                'accuracy': fusion_accuracy,
                'confusion_matrix': confusion_matrix(all_labels, all_fusion_preds),
                'report': classification_report(all_labels, all_fusion_preds, output_dict=True),
                'anomaly_scores': all_anomaly_scores,
                'gat_probs': all_gat_probs
            }
            print(f"Fusion Strategy Accuracy: {fusion_accuracy:.4f}")
        
        return results

    def save_models(self, save_folder: str, dataset_key: str, 
                   epochs: int, embedding_dim: int, num_ids: int) -> None:
        """Save trained models with metadata."""
        os.makedirs(save_folder, exist_ok=True)
        
        # Autoencoder save data
        autoencoder_data = {
            'state_dict': self.autoencoder.state_dict(),
            'threshold': self.threshold,
            'epochs': epochs,
            'embedding_dim': embedding_dim,
            'num_ids': num_ids,
            'fusion_weights': FUSION_WEIGHTS
        }
        
        # Classifier save data
        classifier_data = {
            'state_dict': self.classifier.state_dict(),
            'epochs': epochs,
            'embedding_dim': embedding_dim,
            'num_ids': num_ids
        }
        
        # Save models
        autoencoder_path = os.path.join(save_folder, f'autoencoder_best_{dataset_key}.pth')
        classifier_path = os.path.join(save_folder, f'classifier_{dataset_key}.pth')
        
        torch.save(autoencoder_data, autoencoder_path)
        torch.save(classifier_data, classifier_path)
        
        print(f"✓ Models saved to '{save_folder}':")
        print(f"  - Autoencoder: {autoencoder_path}")
        print(f"  - Classifier: {classifier_path}")

    def _cleanup_memory(self) -> None:
        """Clean up GPU/CPU memory."""
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()