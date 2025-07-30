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
from preprocessing import graph_creation, build_id_mapping_from_normal
from training_utils import DistillationTrainer, distillation_loss_fn, FocalLoss
from torch_geometric.data import Batch, Data

def create_teacher_student_models(num_ids, embedding_dim, device):
    """Create teacher (large) and student (small) models."""
    
    # Teacher models (large, complex)
    teacher_autoencoder = GraphAutoencoderNeighborhood(
        num_ids=num_ids, 
        in_channels=11, 
        embedding_dim=embedding_dim,
        hidden_dim=32,           # Larger hidden dimension
        latent_dim=32,           # Larger latent dimension
        num_encoder_layers=3,    # More layers
        num_decoder_layers=3,
        encoder_heads=4,         # More attention heads
        decoder_heads=4
    ).to(device)
    
    teacher_classifier = GATWithJK(
        num_ids=num_ids, 
        in_channels=11, 
        hidden_channels=32,      # Larger hidden channels
        out_channels=1, 
        num_layers=5,            # More layers
        heads=8,                # More attention heads
        embedding_dim=embedding_dim
    ).to(device)
    
    # Student models (small, efficient)
    student_autoencoder = GraphAutoencoderNeighborhood(
        num_ids=num_ids, 
        in_channels=11, 
        embedding_dim=embedding_dim,
        hidden_dim=16,           # Smaller hidden dimension
        latent_dim=16,           # Smaller latent dimension
        num_encoder_layers=2,    # Fewer layers
        num_decoder_layers=2,
        encoder_heads=2,         # Fewer attention heads
        decoder_heads=2
    ).to(device)
    
    student_classifier = GATWithJK(
        num_ids=num_ids, 
        in_channels=11, 
        hidden_channels=16,      # Smaller hidden channels
        out_channels=1, 
        num_layers=2,            # Fewer layers
        heads=4,                 # Fewer attention heads
        embedding_dim=embedding_dim
    ).to(device)
    
    return teacher_autoencoder, teacher_classifier, student_autoencoder, student_classifier

def create_data_loaders(train_subset, test_dataset, full_train_dataset, batch_size, device):
    """Create optimized data loaders based on device type."""
    
    # Adaptive batch size based on dataset size
    dataset_size = len(train_subset) + len(test_dataset)
    
    if torch.cuda.is_available() and device.type == 'cuda':
        # GPU optimizations with dataset-aware batch sizing
        if dataset_size > 300000:  # Very large dataset (like hcrl_ch/set_03)
            batch_size = min(batch_size, 64)   # Very small batches for huge datasets
            pin_memory = True
            num_workers = 1
            persistent_workers = False
            prefetch_factor = 1
            print(f"Very large dataset detected: Using GPU batch_size={batch_size}, workers={num_workers}")
        elif dataset_size > 100000:  # Large dataset
            batch_size = min(batch_size, 128)  # Small batches for large datasets
            pin_memory = True
            num_workers = 1
            persistent_workers = False
            prefetch_factor = 2
            print(f"Large dataset detected: Using GPU batch_size={batch_size}, workers={num_workers}")
        else:  # Standard dataset
            batch_size = min(batch_size, 512)  # Normal batches for smaller datasets
            pin_memory = True
            num_workers = 2
            persistent_workers = True
            prefetch_factor = 2
            print(f"Standard dataset: Using GPU batch_size={batch_size}, workers={num_workers}")
    else:
        # CPU optimizations
        pin_memory = False
        num_workers = 0
        persistent_workers = False
        prefetch_factor = None
        print("Using CPU-optimized data loaders")
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    full_train_loader = DataLoader(
        full_train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    return train_loader, test_loader, full_train_loader

class KnowledgeDistillationPipeline:
    """Adaptive knowledge distillation pipeline for CAN bus anomaly detection."""
    
    def __init__(self, teacher_autoencoder, teacher_classifier, 
                 student_autoencoder, student_classifier, device='cpu'):
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'
        
        # Teacher models (pre-trained)
        self.teacher_autoencoder = teacher_autoencoder.to(device)
        self.teacher_classifier = teacher_classifier.to(device)
        
        # Student models (to be trained)
        self.student_autoencoder = student_autoencoder.to(device)
        self.student_classifier = student_classifier.to(device)
        
        # Set teachers to eval mode
        self.teacher_autoencoder.eval()
        self.teacher_classifier.eval()
        
        # Freeze teacher parameters
        for param in self.teacher_autoencoder.parameters():
            param.requires_grad = False
        for param in self.teacher_classifier.parameters():
            param.requires_grad = False
            
        self.threshold = 0.0
        
        print(f"Initialized KD Pipeline on {device} (CUDA: {self.is_cuda})")

    def load_teacher_models(self, autoencoder_path, classifier_path):
        """Load pre-trained teacher models with proper checkpoint handling."""
        print(f"Loading teacher autoencoder from: {autoencoder_path}")
        
        # Load autoencoder checkpoint
        autoencoder_checkpoint = torch.load(autoencoder_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(autoencoder_checkpoint, dict):
            if 'state_dict' in autoencoder_checkpoint:
                # Checkpoint contains metadata
                autoencoder_state_dict = autoencoder_checkpoint['state_dict']
                if 'threshold' in autoencoder_checkpoint:
                    self.threshold = autoencoder_checkpoint['threshold']
            else:
                # Dictionary but direct state dict
                autoencoder_state_dict = autoencoder_checkpoint
        else:
            # Direct state dict
            autoencoder_state_dict = autoencoder_checkpoint
        
        # Load classifier checkpoint
        print(f"Loading teacher classifier from: {classifier_path}")
        classifier_checkpoint = torch.load(classifier_path, map_location=self.device)
        
        if isinstance(classifier_checkpoint, dict):
            if 'state_dict' in classifier_checkpoint:
                classifier_state_dict = classifier_checkpoint['state_dict']
            else:
                classifier_state_dict = classifier_checkpoint
        else:
            classifier_state_dict = classifier_checkpoint
        
        # Load state dicts
        try:
            self.teacher_autoencoder.load_state_dict(autoencoder_state_dict)
            print("Teacher autoencoder loaded successfully!")
            
            self.teacher_classifier.load_state_dict(classifier_state_dict)
            print("Teacher classifier loaded successfully!")
            
        except RuntimeError as e:
            print(f"Architecture mismatch: {str(e)[:200]}...")
            print("The saved model has a different architecture than expected.")
            print("Please check if the saved models were created with different parameters.")
            raise e
        
        # Set to eval mode
        self.teacher_autoencoder.eval()
        self.teacher_classifier.eval()
        
        print("Teacher models loaded and set to evaluation mode!")

    def distill_autoencoder(self, train_loader, epochs=20, alpha=0.5, temperature=5.0):
        """Adaptive autoencoder distillation for CPU/GPU."""
        print(f"Distilling autoencoder for {epochs} epochs on {self.device}...")
        
        self.student_autoencoder.train()
        optimizer = torch.optim.Adam(self.student_autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Projection layer setup
        teacher_latent_dim = self.teacher_autoencoder.latent_dim
        student_latent_dim = self.student_autoencoder.latent_dim
        
        if teacher_latent_dim != student_latent_dim:
            projection_layer = nn.Linear(student_latent_dim, teacher_latent_dim).to(self.device)
            proj_optimizer = torch.optim.Adam(projection_layer.parameters(), lr=1e-3)
            print(f"Added projection layer: {student_latent_dim} -> {teacher_latent_dim}")
        else:
            projection_layer = None
        
        # Mixed precision only for CUDA
        scaler = torch.cuda.amp.GradScaler() if self.is_cuda else None
        if scaler:
            print("Using mixed precision training")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Adaptive data transfer
                if self.is_cuda:
                    batch = batch.to(self.device, non_blocking=True)
                else:
                    batch = batch.to(self.device)
                
                # Device-specific forward pass
                if self.is_cuda:
                    # GPU path with mixed precision
                    with torch.cuda.amp.autocast():
                        total_loss = self._compute_autoencoder_loss(
                            batch, projection_layer, alpha, temperature)
                else:
                    # CPU path without autocast
                    total_loss = self._compute_autoencoder_loss(
                        batch, projection_layer, alpha, temperature)
                
                # Device-specific backward pass
                optimizer.zero_grad()
                if projection_layer is not None:
                    proj_optimizer.zero_grad()
                
                if scaler is not None:
                    # Mixed precision backward pass
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student_autoencoder.parameters(), max_norm=1.0)
                    if projection_layer is not None:
                        torch.nn.utils.clip_grad_norm_(projection_layer.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    if projection_layer is not None:
                        scaler.step(proj_optimizer)
                    scaler.update()
                else:
                    # Standard backward pass
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student_autoencoder.parameters(), max_norm=1.0)
                    if projection_layer is not None:
                        torch.nn.utils.clip_grad_norm_(projection_layer.parameters(), max_norm=1.0)
                    optimizer.step()
                    if projection_layer is not None:
                        proj_optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Device-specific memory management
                if self.is_cuda and batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                elif not self.is_cuda and batch_idx % 100 == 0:
                    # Less frequent cleanup on CPU
                    pass
            
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                print(f"Autoencoder Distillation Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Clear cache between epochs
            if self.is_cuda:
                torch.cuda.empty_cache()
        
        self._set_threshold_student(train_loader)
        print(f"Set student anomaly threshold: {self.threshold:.4f}")

    def _compute_autoencoder_loss(self, batch, projection_layer, alpha, temperature):
        """Compute autoencoder distillation loss."""
        # Teacher forward pass
        with torch.no_grad():
            teacher_cont_out, teacher_canid_logits, teacher_neighbor_logits, teacher_z, _ = \
                self.teacher_autoencoder(batch.x, batch.edge_index, batch.batch)
        
        # Student forward pass
        student_cont_out, student_canid_logits, student_neighbor_logits, student_z, student_kl_loss = \
            self.student_autoencoder(batch.x, batch.edge_index, batch.batch)
        
        # Reconstruction losses
        cont_loss = nn.MSELoss()(student_cont_out, batch.x[:, 1:])
        canid_loss = nn.CrossEntropyLoss()(student_canid_logits, batch.x[:, 0].long())
        
        neighbor_targets = self.student_autoencoder.create_neighborhood_targets(
            batch.x, batch.edge_index, batch.batch)
        neighbor_loss = nn.BCEWithLogitsLoss()(student_neighbor_logits, neighbor_targets)
        
        # Knowledge distillation losses
        if projection_layer is not None:
            student_z_projected = projection_layer(student_z)
            feature_distill_loss = nn.MSELoss()(student_z_projected, teacher_z.detach())
        else:
            feature_distill_loss = nn.MSELoss()(student_z, teacher_z.detach())
        
        cont_distill_loss = nn.MSELoss()(student_cont_out, teacher_cont_out.detach())
        canid_distill_loss = distillation_loss_fn(student_canid_logits, teacher_canid_logits.detach(), T=temperature)
        neighbor_distill_loss = distillation_loss_fn(student_neighbor_logits, teacher_neighbor_logits.detach(), T=temperature)
        
        # Combine losses properly
        reconstruction_loss = cont_loss + canid_loss + neighbor_loss + 0.1 * student_kl_loss
        distillation_loss = feature_distill_loss + cont_distill_loss + canid_distill_loss + neighbor_distill_loss
        
        total_loss = (1 - alpha) * reconstruction_loss + alpha * distillation_loss
        return total_loss

    def distill_classifier(self, full_loader, epochs=20, alpha=0.7, temperature=5.0):
        """Adaptive classifier distillation using filtered graphs."""
        print(f"Distilling classifier for {epochs} epochs on {self.device}...")
        
        # Create balanced dataset using student autoencoder
        balanced_graphs = self._create_balanced_dataset_student(full_loader)
        if not balanced_graphs:
            print("No graphs available for classifier distillation.")
            return
        
        self.student_classifier.train()
        
        # Use DistillationTrainer for classifier
        trainer = DistillationTrainer(
            teacher=self.teacher_classifier,
            student=self.student_classifier,
            device=self.device,
            teacher_epochs=0,  # Teacher already trained
            student_epochs=epochs,
            distill_alpha=alpha,
            warmup_epochs=5,
            lr=1e-3,
            use_focal_loss=True
        )
        
        # Create device-specific data loaders
        if self.is_cuda:
            pin_memory = True
            num_workers = 2
            persistent_workers = True
            prefetch_factor = 2
        else:
            pin_memory = False
            num_workers = 0
            persistent_workers = False
            prefetch_factor = None
        
        balanced_loader = DataLoader(
            balanced_graphs, 
            batch_size=32, 
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        test_loader = DataLoader(
            balanced_graphs[:min(100, len(balanced_graphs))], 
            batch_size=32, 
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        
        # Train student classifier with knowledge distillation
        trainer.train_student(balanced_loader, test_loader)

    def _set_threshold_student(self, train_loader, percentile=50):
        """Set anomaly detection threshold using student autoencoder."""
        errors = []
        with torch.no_grad():
            for batch in train_loader:
                if self.is_cuda:
                    batch = batch.to(self.device, non_blocking=True)
                else:
                    batch = batch.to(self.device)
                cont_out, _, _, _, _ = self.student_autoencoder(batch.x, batch.edge_index, batch.batch)
                errors.append((cont_out - batch.x[:, 1:]).pow(2).mean(dim=1))
        self.threshold = torch.cat(errors).quantile(percentile / 100.0).item()

    def _create_balanced_dataset_student(self, loader):
        """Adaptive dataset creation to prevent memory spikes."""
        print("Computing composite errors using student model...")
        
        all_graphs = []
        all_composite_errors = []
        all_is_attack = []
        
        # Process in smaller chunks to prevent memory spikes
        chunk_size = 0
        total_batches = len(loader)
        cache_frequency = 20 if self.is_cuda else 100
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # Adaptive data transfer
                if self.is_cuda:
                    batch = batch.to(self.device, non_blocking=True)
                else:
                    batch = batch.to(self.device)
                
                # Print progress to monitor
                if batch_idx % 100 == 0:
                    print(f"Processing batch {batch_idx}/{total_batches}")
                
                # VECTORIZED: Single forward pass for entire batch
                cont_out, canid_logits, neighbor_logits, _, _ = self.student_autoencoder(
                    batch.x, batch.edge_index, batch.batch)
                
                # VECTORIZED: Compute all errors at once
                node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                neighbor_targets = self.student_autoencoder.create_neighborhood_targets(
                    batch.x, batch.edge_index, batch.batch)
                neighbor_recon_errors = nn.BCEWithLogitsLoss(reduction='none')(
                    neighbor_logits, neighbor_targets).mean(dim=1)
                canid_pred = canid_logits.argmax(dim=1)
                
                # VECTORIZED: Graph-level aggregation using scatter operations
                batch_size = batch.batch.max().item() + 1
                graph_node_errors = torch.zeros(batch_size, device=self.device)
                graph_neighbor_errors = torch.zeros(batch_size, device=self.device)
                
                # Efficient scatter operations instead of loops
                graph_node_errors.scatter_reduce_(0, batch.batch, node_errors, reduce='amax')
                graph_neighbor_errors.scatter_reduce_(0, batch.batch, neighbor_recon_errors, reduce='amax')
                
                # VECTORIZED: Process all graphs in batch simultaneously
                graphs = Batch.to_data_list(batch)
                batch_composite_errors = []
                batch_is_attack = []
                
                # Compute CAN ID errors vectorized per graph
                for i, graph in enumerate(graphs):
                    if i < batch_size:
                        is_attack = int(graph.y.flatten()[0]) == 1
                        
                        # Find node range for this graph
                        graph_mask = batch.batch == i
                        if graph_mask.sum() > 0:
                            graph_start = graph_mask.nonzero()[0].item()
                            graph_end = graph_start + graph.x.size(0)
                            
                            # Vectorized CAN ID accuracy
                            true_canids = batch.x[graph_start:graph_end, 0].long()
                            pred_canids = canid_pred[graph_start:graph_end]
                            canid_accuracy = (pred_canids == true_canids).float().mean().item()
                            canid_error = 1.0 - canid_accuracy
                            
                            # Composite error computation
                            composite_error = (1.0 * graph_node_errors[i].item() + 
                                             20.0 * graph_neighbor_errors[i].item() + 
                                             0.3 * canid_error)
                            
                            batch_composite_errors.append(composite_error)
                            batch_is_attack.append(is_attack)
                            
                            # Move to CPU to save GPU memory
                            all_graphs.append(graph.cpu())
                
                # Extend lists
                all_composite_errors.extend(batch_composite_errors)
                all_is_attack.extend(batch_is_attack)
                
                # Device-specific memory management
                chunk_size += 1
                if chunk_size % cache_frequency == 0:
                    if self.is_cuda:
                        torch.cuda.empty_cache()
                    print(f"Processed {chunk_size} batches, cleared cache")
                
                # DELETE tensors explicitly to prevent accumulation
                del cont_out, canid_logits, neighbor_logits, node_errors
                del neighbor_targets, neighbor_recon_errors, canid_pred
                del graph_node_errors, graph_neighbor_errors, batch
        
        # Rest of filtering logic
        print("Filtering and balancing dataset...")
        graph_data = list(zip(all_graphs, all_composite_errors, all_is_attack))
        attack_graphs = [(graph, error) for graph, error, is_attack in graph_data if is_attack]
        normal_graphs = [(graph, error) for graph, error, is_attack in graph_data if not is_attack]
        
        selected_attack_graphs = [graph for graph, _ in attack_graphs]
        num_attacks = len(selected_attack_graphs)
        
        if num_attacks == 0:
            return []
        
        max_normal_graphs = num_attacks * 4
        
        if len(normal_graphs) <= max_normal_graphs:
            selected_normal_graphs = [graph for graph, _ in normal_graphs]
        else:
            normal_graphs_sorted = sorted(normal_graphs, key=lambda x: x[1])
            selected_normal_graphs = [graph for graph, _ in normal_graphs_sorted[-max_normal_graphs:]]
        
        balanced_graphs = selected_attack_graphs + selected_normal_graphs
        random.shuffle(balanced_graphs)
        
        print(f"Created balanced dataset: {len(selected_normal_graphs)} normal, {num_attacks} attack")
        
        # Final cleanup
        if self.is_cuda:
            torch.cuda.empty_cache()
        
        return balanced_graphs

    def predict_student(self, data):
        """Adaptive prediction to maintain consistent utilization."""
        if self.is_cuda:
            data = data.to(self.device, non_blocking=True)
        else:
            data = data.to(self.device)
        
        with torch.no_grad():
            # Single forward pass for entire batch
            cont_out, _, _, _, _ = self.student_autoencoder(data.x, data.edge_index, data.batch)
            node_errors = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            # Vectorized graph-level error computation
            batch_size = data.batch.max().item() + 1
            graph_errors = torch.zeros(batch_size, device=self.device)
            graph_errors.scatter_reduce_(0, data.batch, node_errors, reduce='amax')
            
            # Vectorized threshold check
            anomaly_mask = graph_errors > self.threshold
            preds = torch.zeros(batch_size, device=self.device)
            
            # Process anomalous graphs
            if anomaly_mask.any():
                anomaly_indices = torch.nonzero(anomaly_mask).flatten()
                
                # Extract all anomalous subgraphs efficiently
                subgraph_data_list = []
                valid_anomaly_indices = []
                
                for idx in anomaly_indices:
                    # Vectorized subgraph extraction
                    node_mask = data.batch == idx
                    edge_mask = (data.batch[data.edge_index[0]] == idx) & (data.batch[data.edge_index[1]] == idx)
                    
                    if node_mask.sum() > 0:
                        # Extract subgraph
                        subgraph_x = data.x[node_mask]
                        
                        if edge_mask.sum() > 0:
                            subgraph_edge_index = data.edge_index[:, edge_mask]
                            # Remap node indices efficiently
                            node_mapping = torch.full((data.x.size(0),), -1, dtype=torch.long, device=self.device)
                            node_mapping[node_mask] = torch.arange(node_mask.sum(), device=self.device)
                            subgraph_edge_index = node_mapping[subgraph_edge_index]
                        else:
                            subgraph_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                        
                        # Create subgraph data
                        subgraph = Data(x=subgraph_x, edge_index=subgraph_edge_index)
                        subgraph_data_list.append(subgraph)
                        valid_anomaly_indices.append(idx)
                
                # BATCH CLASSIFY all anomalous graphs at once
                if subgraph_data_list:
                    try:
                        # Create single batch for all anomalous subgraphs
                        batched_anomalous = Batch.from_data_list(subgraph_data_list)
                        classifier_probs = self.student_classifier(batched_anomalous)
                        
                        # Assign predictions vectorized
                        if classifier_probs.dim() > 1:
                            classifier_probs = classifier_probs.flatten()
                        
                        for i, idx in enumerate(valid_anomaly_indices):
                            if i < classifier_probs.size(0):
                                preds[idx] = (classifier_probs[i] > 0.5).float()
                    
                    except Exception as e:
                        # Fallback only if absolutely necessary
                        print(f"Batch classification failed, using individual processing: {e}")
                        for i, (subgraph, idx) in enumerate(zip(subgraph_data_list, valid_anomaly_indices)):
                            try:
                                prob = self.student_classifier(subgraph)
                                preds[idx] = (prob.item() > 0.5)
                            except:
                                preds[idx] = 0
            
            return preds.long()

class GATPipeline:
    """Original GATPipeline for teacher model evaluation comparison."""
    
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
                    graph_batch = graph.to(self.device)
                    prob = self.classifier(graph_batch).item()
                    preds.append(1 if prob > 0.5 else 0)
                else:
                    preds.append(0)
                start += num_nodes
            
            return torch.tensor(preds, device=self.device)

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    """Adaptive knowledge distillation pipeline for CAN bus anomaly detection."""
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
    
    # Load data
    KEY = config_dict['root_folder']
    root_folder = root_folders[KEY]
    id_mapping = build_id_mapping_from_normal(root_folder)
    dataset = graph_creation(root_folder, id_mapping=id_mapping, window_size=100)
    
    # Fix the random seed for consistent batching
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print(f"Dataset: {len(dataset)} graphs, {len(id_mapping)} unique CAN IDs")
    
    # Adaptive configuration based on device
    DATASIZE = config_dict['datasize']
    TRAIN_RATIO = config_dict['train_ratio']
    
    # Adaptive batch size based on device and dataset size
    if torch.cuda.is_available():
        BATCH_SIZE = min(config_dict['batch_size'], 512)  # Cap for GPU memory efficiency
        print(f"GPU detected: Using BATCH_SIZE: {BATCH_SIZE}")
    else:
        BATCH_SIZE = min(config_dict['batch_size'], 128)  # Smaller for CPU efficiency
        print(f"CPU detected: Using BATCH_SIZE: {BATCH_SIZE}")
    
    EPOCHS = config_dict['epochs']
    print(f"Using EPOCHS: {EPOCHS}")
    
    # Train/test split
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    # Create normal-only training subset
    normal_indices = [i for i, data in enumerate(train_dataset) if int(data.y.flatten()[0]) == 0]
    if DATASIZE < 1.0:
        subset_size = int(len(normal_indices) * DATASIZE)
        indices = np.random.choice(normal_indices, subset_size, replace=False)
    else:
        indices = normal_indices
    
    normal_subset = Subset(train_dataset, indices)
    
    print(f"Using {len(normal_subset)} training samples, {len(test_dataset)} test samples")
    
    # Create adaptive data loaders
    train_loader, test_loader, full_train_loader = create_data_loaders(
        normal_subset, test_dataset, train_dataset, BATCH_SIZE, device
    )
    
    # Create teacher and student models
    teacher_ae, teacher_clf, student_ae, student_clf = create_teacher_student_models(
        num_ids=len(id_mapping), embedding_dim=8, device=device
    )
    
    # Initialize knowledge distillation pipeline
    kd_pipeline = KnowledgeDistillationPipeline(
        teacher_autoencoder=teacher_ae,
        teacher_classifier=teacher_clf,
        student_autoencoder=student_ae,
        student_classifier=student_clf,
        device=device
    )
    
    # Load pre-trained teacher models
    teacher_ae_path = f"output_model_1/autoencoder_best_{KEY}.pth"
    teacher_clf_path = f"output_model_1/classifier_{KEY}.pth"

    try:
        kd_pipeline.load_teacher_models(teacher_ae_path, teacher_clf_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run osc-training-AD.py first to train teacher models!")
        return
    
    # Knowledge distillation training with memory management
    print("\n=== Stage 1: Autoencoder Knowledge Distillation ===")
    kd_pipeline.distill_autoencoder(train_loader, epochs=EPOCHS, alpha=0.5, temperature=5.0)
    
    # Clear memory before stage 2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n=== Stage 2: Classifier Knowledge Distillation ===")
    kd_pipeline.distill_classifier(full_train_loader, epochs=EPOCHS, alpha=0.7, temperature=5.0)
    
    # Save student models
    save_folder = "saved_models"
    os.makedirs(save_folder, exist_ok=True)
    torch.save(kd_pipeline.student_autoencoder.state_dict(), 
               os.path.join(save_folder, f'student_autoencoder_{KEY}.pth'))
    torch.save(kd_pipeline.student_classifier.state_dict(), 
               os.path.join(save_folder, f'student_classifier_{KEY}.pth'))
    print(f"Student models saved to '{save_folder}'")
    
    # Adaptive evaluation
    print("\n=== Evaluation: Student Model Performance ===")
    
    all_student_preds = []
    all_labels = []
    
    # Adaptive batch processing
    cache_frequency = 20 if torch.cuda.is_available() else 50
    
    # Process in optimized batches with memory clearing
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Adaptive data transfer
            if torch.cuda.is_available():
                batch = batch.to(device, non_blocking=True)
            else:
                batch = batch.to(device)
            
            # Batch prediction
            student_preds = kd_pipeline.predict_student(batch)
            labels = batch.y
            
            all_student_preds.append(student_preds.cpu())
            all_labels.append(labels.cpu())
            
            # Device-specific memory clearing
            if (i + 1) % cache_frequency == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Concatenate results
    student_preds = torch.cat(all_student_preds)
    labels = torch.cat(all_labels)
    
    student_accuracy = (student_preds == labels).float().mean().item()

    print(f"Student Model Accuracy: {student_accuracy:.4f}")
    print("\nStudent Confusion Matrix:")
    print(confusion_matrix(labels.numpy(), student_preds.numpy()))

    # Model size comparison
    teacher_params = sum(p.numel() for p in teacher_ae.parameters()) + sum(p.numel() for p in teacher_clf.parameters())
    student_params = sum(p.numel() for p in student_ae.parameters()) + sum(p.numel() for p in student_clf.parameters())

    print(f"\nModel Size Comparison:")
    print(f"Teacher Parameters: {teacher_params:,}")
    print(f"Student Parameters: {student_params:,}")
    print(f"Compression Ratio: {teacher_params/student_params:.1f}x")

    print(f"\nKnowledge Distillation Complete on {device}!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")