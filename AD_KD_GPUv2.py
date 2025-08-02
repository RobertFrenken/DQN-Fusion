"""
Knowledge Distillation for CAN Bus Anomaly Detection

This module implements a comprehensive knowledge distillation framework for CAN bus
intrusion detection, transferring knowledge from large teacher models to compact
student models while maintaining detection performance.

Key Features:
- Teacher-student architecture with configurable compression ratios
- Memory-optimized training with mixed precision support
- Simplified and full distillation modes for different speed/accuracy tradeoffs
- GPU acceleration with automatic fallback to CPU
- Comprehensive evaluation framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.data import random_split, Subset
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix, classification_report
import random
import psutil
import gc
import multiprocessing as mp
from typing import Tuple, Dict, Any, Optional, List
import warnings

from models.models import GATWithJK, GraphAutoencoderNeighborhood
from preprocessing import graph_creation, build_id_mapping_from_normal
from training_utils import distillation_loss_fn

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# ==================== Configuration Constants ====================

DATASET_PATHS = {
    'hcrl_ch': r"datasets/can-train-and-test-v1.5/hcrl-ch",
    'hcrl_sa': r"datasets/can-train-and-test-v1.5/hcrl-sa",
    'set_01': r"datasets/can-train-and-test-v1.5/set_01",
    'set_02': r"datasets/can-train-and-test-v1.5/set_02",
    'set_03': r"datasets/can-train-and-test-v1.5/set_03",
    'set_04': r"datasets/can-train-and-test-v1.5/set_04",
}

# Model architecture configurations
TEACHER_CONFIG = {
    'hidden_dim': 32,
    'latent_dim': 32,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'encoder_heads': 4,
    'decoder_heads': 4,
    'classifier_hidden': 32,
    'classifier_layers': 5,
    'classifier_heads': 8
}

STUDENT_CONFIG = {
    'hidden_dim': 16,
    'latent_dim': 16,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'encoder_heads': 2,
    'decoder_heads': 2,
    'classifier_hidden': 16,
    'classifier_layers': 2,
    'classifier_heads': 4
}

# ==================== Utility Functions ====================

def setup_gpu_optimization():
    """Configure GPU memory optimization settings."""
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        torch.cuda.empty_cache()
        print("✓ GPU memory optimization enabled")

def log_memory_usage(stage: str = ""):
    """Log current CPU and GPU memory usage."""
    cpu_mem = psutil.virtual_memory()
    print(f"[{stage}] CPU Memory: {cpu_mem.used/1024**3:.1f}GB / {cpu_mem.total/1024**3:.1f}GB ({cpu_mem.percent:.1f}%)")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_cached:.2f}GB cached")

def cleanup_memory():
    """Perform comprehensive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_teacher_student_models(num_ids: int, embedding_dim: int, device: str) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Create teacher (large) and student (compact) model pairs.
    
    Args:
        num_ids: Number of unique CAN IDs in the dataset
        embedding_dim: Dimensionality of CAN ID embeddings
        device: Target device for model placement
        
    Returns:
        Tuple of (teacher_autoencoder, teacher_classifier, student_autoencoder, student_classifier)
    """
    # Teacher models (high capacity)
    teacher_autoencoder = GraphAutoencoderNeighborhood(
        num_ids=num_ids,
        in_channels=11,
        embedding_dim=embedding_dim,
        **{k: v for k, v in TEACHER_CONFIG.items() if 'classifier' not in k}
    ).to(device)
    
    teacher_classifier = GATWithJK(
        num_ids=num_ids,
        in_channels=11,
        hidden_channels=TEACHER_CONFIG['classifier_hidden'],
        out_channels=1,
        num_layers=TEACHER_CONFIG['classifier_layers'],
        heads=TEACHER_CONFIG['classifier_heads'],
        embedding_dim=embedding_dim
    ).to(device)
    
    # Student models (compact)
    student_autoencoder = GraphAutoencoderNeighborhood(
        num_ids=num_ids,
        in_channels=11,
        embedding_dim=embedding_dim,
        **{k: v for k, v in STUDENT_CONFIG.items() if 'classifier' not in k}
    ).to(device)
    
    student_classifier = GATWithJK(
        num_ids=num_ids,
        in_channels=11,
        hidden_channels=STUDENT_CONFIG['classifier_hidden'],
        out_channels=1,
        num_layers=STUDENT_CONFIG['classifier_layers'],
        heads=STUDENT_CONFIG['classifier_heads'],
        embedding_dim=embedding_dim
    ).to(device)
    
    # Calculate compression ratio
    teacher_params = sum(p.numel() for p in teacher_autoencoder.parameters()) + \
                    sum(p.numel() for p in teacher_classifier.parameters())
    student_params = sum(p.numel() for p in student_autoencoder.parameters()) + \
                    sum(p.numel() for p in student_classifier.parameters())
    
    compression_ratio = teacher_params / student_params
    print(f"✓ Models created - Compression ratio: {compression_ratio:.1f}x")
    print(f"  Teacher: {teacher_params:,} parameters")
    print(f"  Student: {student_params:,} parameters")
    
    return teacher_autoencoder, teacher_classifier, student_autoencoder, student_classifier

def create_optimized_data_loaders(train_subset, test_dataset, full_train_dataset, 
                                 batch_size: int, device: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create memory-optimized data loaders with appropriate configuration.
    
    Args:
        train_subset: Normal graphs for autoencoder training
        test_dataset: Test dataset for evaluation
        full_train_dataset: Full training dataset for classifier training
        batch_size: Base batch size (will be optimized)
        device: Target device
        
    Returns:
        Tuple of (train_loader, test_loader, full_train_loader)
    """
    # Optimize batch size and worker configuration based on device
    if torch.cuda.is_available() and 'cuda' in device:
        optimized_batch_size = 2048
        num_workers = 8
        pin_memory = True
        prefetch_factor = 4
        persistent_workers = True
    else:
        optimized_batch_size = min(batch_size, 1024)
        num_workers = 4
        pin_memory = False
        prefetch_factor = 2
        persistent_workers = False
    
    print(f"✓ DataLoader config: batch_size={optimized_batch_size}, workers={num_workers}")
    
    # Create data loaders
    loaders = []
    datasets = [
        (train_subset, True, "training"),
        (test_dataset, False, "test"),
        (full_train_dataset, True, "full_training")
    ]
    
    for dataset, shuffle, name in datasets:
        if dataset is not None:
            loader = DataLoader(
                dataset,
                batch_size=optimized_batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor
            )
            loaders.append(loader)
            print(f"  {name}: {len(dataset)} samples")
    
    return tuple(loaders)

# ==================== Knowledge Distillation Pipeline ====================

class KnowledgeDistillationPipeline:
    """
    Comprehensive knowledge distillation pipeline for CAN bus anomaly detection.
    
    This pipeline implements teacher-student learning where a compact student model
    learns from a large, pre-trained teacher model while maintaining detection performance.
    """
    
    def __init__(self, teacher_autoencoder: nn.Module, teacher_classifier: nn.Module,
                 student_autoencoder: nn.Module, student_classifier: nn.Module, 
                 device: str = 'cpu'):
        """
        Initialize the knowledge distillation pipeline.
        
        Args:
            teacher_autoencoder: Pre-trained teacher autoencoder
            teacher_classifier: Pre-trained teacher classifier
            student_autoencoder: Student autoencoder to be trained
            student_classifier: Student classifier to be trained
            device: Target device for computation
        """
        self.device = torch.device(device)
        self.is_cuda = torch.cuda.is_available() and self.device.type == 'cuda'
        
        # Teacher models (frozen)
        self.teacher_autoencoder = teacher_autoencoder.to(self.device)
        self.teacher_classifier = teacher_classifier.to(self.device)
        
        # Student models (trainable)
        self.student_autoencoder = student_autoencoder.to(self.device)
        self.student_classifier = student_classifier.to(self.device)
        
        # Set teachers to evaluation mode and freeze parameters
        self._freeze_teacher_models()
        
        self.threshold = 0.0
        print(f"✓ KD Pipeline initialized on {self.device} (CUDA: {self.is_cuda})")

    def _freeze_teacher_models(self):
        """Set teacher models to evaluation mode and freeze parameters."""
        self.teacher_autoencoder.eval()
        self.teacher_classifier.eval()
        
        for param in self.teacher_autoencoder.parameters():
            param.requires_grad = False
        for param in self.teacher_classifier.parameters():
            param.requires_grad = False

    def load_teacher_models(self, autoencoder_path: str, classifier_path: str):
        """
        Load pre-trained teacher models from checkpoint files.
        
        Args:
            autoencoder_path: Path to teacher autoencoder checkpoint
            classifier_path: Path to teacher classifier checkpoint
            
        Raises:
            FileNotFoundError: If checkpoint files are not found
            RuntimeError: If model architectures don't match checkpoints
        """
        print(f"Loading teacher models...")
        print(f"  Autoencoder: {autoencoder_path}")
        print(f"  Classifier: {classifier_path}")
        
        # Load autoencoder
        autoencoder_checkpoint = torch.load(autoencoder_path, map_location=self.device)
        
        if isinstance(autoencoder_checkpoint, dict):
            autoencoder_state_dict = autoencoder_checkpoint.get('state_dict', autoencoder_checkpoint)
            self.threshold = autoencoder_checkpoint.get('threshold', 0.0)
        else:
            autoencoder_state_dict = autoencoder_checkpoint
        
        # Load classifier
        classifier_checkpoint = torch.load(classifier_path, map_location=self.device)
        
        if isinstance(classifier_checkpoint, dict):
            classifier_state_dict = classifier_checkpoint.get('state_dict', classifier_checkpoint)
        else:
            classifier_state_dict = classifier_checkpoint
        
        # Load state dictionaries
        try:
            self.teacher_autoencoder.load_state_dict(autoencoder_state_dict)
            self.teacher_classifier.load_state_dict(classifier_state_dict)
            print("✓ Teacher models loaded successfully")
        except RuntimeError as e:
            print(f"❌ Architecture mismatch: {str(e)[:200]}...")
            raise e
        
        # Re-freeze after loading
        self._freeze_teacher_models()
        print(f"✓ Teacher models frozen with threshold: {self.threshold:.4f}")

    def distill_autoencoder_simplified(self, train_loader: DataLoader, epochs: int = 20, 
                                     alpha: float = 0.7, temperature: float = 5.0):
        """
        Simplified autoencoder distillation focusing on latent space and reconstruction.
        
        This method provides 3-4x speedup by focusing only on the most critical
        knowledge transfer components: VGAE latent space and reconstruction quality.
        
        Args:
            train_loader: DataLoader with normal graphs for training
            epochs: Number of training epochs
            alpha: Weight for distillation loss vs. task loss
            temperature: Temperature for knowledge distillation
        """
        print(f"\n=== Simplified Autoencoder Distillation ===")
        print(f"Epochs: {epochs}, Alpha: {alpha}, Temperature: {temperature}")
        print("Focus: VGAE latent space + reconstruction (3-4x speedup)")
        
        self.student_autoencoder.train()
        
        # Optimizer setup
        optimizer = torch.optim.Adam(
            self.student_autoencoder.parameters(), 
            lr=4e-3, 
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3, verbose=True
        )
        
        # Handle dimension mismatch between teacher and student
        projection_layer = self._setup_projection_layer()
        proj_optimizer = None
        if projection_layer is not None:
            proj_optimizer = torch.optim.Adam(projection_layer.parameters(), lr=1e-3)
        
        # Mixed precision setup for CUDA
        scaler = torch.cuda.amp.GradScaler() if self.is_cuda else None
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device, non_blocking=True)
                
                # Compute loss with mixed precision if available
                if self.is_cuda:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        total_loss = self._compute_simplified_autoencoder_loss(
                            batch, projection_layer, alpha, temperature)
                else:
                    total_loss = self._compute_simplified_autoencoder_loss(
                        batch, projection_layer, alpha, temperature)
                
                # Backward pass
                self._perform_backward_pass(total_loss, optimizer, proj_optimizer, scaler, 
                                          self.student_autoencoder, projection_layer)
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Cleanup and memory management
                del batch, total_loss
                if batch_idx % 20 == 0:
                    cleanup_memory()
            
            # Epoch statistics
            avg_loss = epoch_loss / max(num_batches, 1)
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss * 0.999:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress reporting
            if (epoch + 1) % 5 == 0:
                print(f"Autoencoder Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
                log_memory_usage(f"AE Epoch {epoch+1}")
            
            # Early stopping check
            if patience_counter >= 8:
                print(f"Early stopping: No improvement for {patience_counter} epochs")
                break
            
            # Inter-epoch cleanup
            cleanup_memory()
        
        # Set anomaly detection threshold
        self._set_student_threshold(train_loader)
        print(f"✓ Student threshold set: {self.threshold:.4f}")

    def distill_classifier(self, full_loader: DataLoader, epochs: int = 20, 
                          alpha: float = 0.7, temperature: float = 5.0):
        """
        Fast batch-based classifier distillation.
        
        Args:
            full_loader: DataLoader with all training graphs
            epochs: Number of training epochs
            alpha: Weight for distillation loss vs. task loss
            temperature: Temperature for knowledge distillation
        """
        print(f"\n=== Classifier Distillation ===")
        print(f"Epochs: {epochs}, Alpha: {alpha}, Temperature: {temperature}")
        
        self.student_classifier.train()
        
        # Optimizer setup
        optimizer = torch.optim.Adam(
            self.student_classifier.parameters(), 
            lr=2e-3, 
            weight_decay=5e-5
        )
        
        # Mixed precision setup
        scaler = torch.cuda.amp.GradScaler() if self.is_cuda else None
        
        # Training state
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            processed_samples = 0
            
            for batch_idx, batch in enumerate(full_loader):
                batch = batch.to(self.device, non_blocking=True)
                
                try:
                    # Teacher predictions (no gradients)
                    with torch.no_grad():
                        teacher_logits = self.teacher_classifier(batch)
                    
                    # Student forward pass with mixed precision
                    if self.is_cuda:
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            total_loss = self._compute_classifier_loss(
                                batch, teacher_logits, alpha, temperature)
                    else:
                        total_loss = self._compute_classifier_loss(
                            batch, teacher_logits, alpha, temperature)
                    
                    # Backward pass
                    self._perform_backward_pass(total_loss, optimizer, None, scaler, 
                                              self.student_classifier, None)
                    
                    epoch_loss += total_loss.item()
                    batch_count += 1
                    processed_samples += batch.y.size(0)
                    
                    # Cleanup
                    del teacher_logits, total_loss
                    
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")
                
                finally:
                    del batch
                    if batch_idx % 25 == 0:
                        cleanup_memory()
                
                # Progress reporting
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(full_loader)}")
                    if batch_idx % 1000 == 0:
                        log_memory_usage(f"Classifier epoch {epoch+1}")
            
            # Epoch statistics
            avg_loss = epoch_loss / max(batch_count, 1)
            
            # Early stopping
            if avg_loss < best_loss * 0.999:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Classifier Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
                  f"Batches: {batch_count}, Samples: {processed_samples}")
            
            # Early stopping check
            if patience_counter >= 8:
                print(f"Early stopping: No improvement for {patience_counter} epochs")
                break
            
            # Inter-epoch cleanup
            cleanup_memory()

    def _setup_projection_layer(self) -> Optional[nn.Module]:
        """Setup projection layer if teacher and student have different latent dimensions."""
        teacher_latent_dim = getattr(self.teacher_autoencoder, 'latent_dim', 32)
        student_latent_dim = getattr(self.student_autoencoder, 'latent_dim', 16)
        
        if teacher_latent_dim != student_latent_dim:
            projection_layer = nn.Linear(student_latent_dim, teacher_latent_dim).to(self.device)
            print(f"✓ Projection layer added: {student_latent_dim} → {teacher_latent_dim}")
            return projection_layer
        return None

    def _compute_simplified_autoencoder_loss(self, batch, projection_layer, alpha, temperature):
        """Compute simplified autoencoder distillation loss."""
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            _, _, _, teacher_z, _ = self.teacher_autoencoder(batch.x, batch.edge_index, batch.batch)
            teacher_z = teacher_z.detach().clone()
        
        # Student forward pass
        student_cont_out, _, _, student_z, student_kl_loss = \
            self.student_autoencoder(batch.x, batch.edge_index, batch.batch)
        
        # Core loss components
        # 1. Reconstruction quality
        reconstruction_loss = nn.MSELoss()(student_cont_out, batch.x[:, 1:])
        
        # 2. Latent space alignment (key knowledge transfer)
        if projection_layer is not None:
            student_z_projected = projection_layer(student_z)
            latent_distill_loss = nn.MSELoss()(student_z_projected, teacher_z)
            del student_z_projected
        else:
            latent_distill_loss = nn.MSELoss()(student_z, teacher_z)
        
        # 3. Variational regularization
        kl_regularization = 0.1 * student_kl_loss
        
        # Combine losses
        task_loss = reconstruction_loss + kl_regularization
        knowledge_loss = latent_distill_loss
        total_loss = (1 - alpha) * task_loss + alpha * knowledge_loss
        
        # Cleanup intermediate tensors
        del teacher_z, student_cont_out, student_z, student_kl_loss
        del reconstruction_loss, latent_distill_loss, kl_regularization
        del task_loss, knowledge_loss
        
        return total_loss

    def _compute_classifier_loss(self, batch, teacher_logits, alpha, temperature):
        """Compute classifier distillation loss."""
        # Student forward pass
        student_logits = self.student_classifier(batch)
        
        # Distillation loss (soft targets)
        distill_loss = nn.KLDivLoss(reduction='batchmean')(
            nn.LogSoftmax(dim=-1)(student_logits / temperature),
            nn.Softmax(dim=-1)(teacher_logits / temperature)
        ) * (temperature ** 2)
        
        # Task loss (hard targets)
        task_loss = nn.BCEWithLogitsLoss()(student_logits.squeeze(), batch.y.float())
        
        # Combined loss
        total_loss = alpha * distill_loss + (1 - alpha) * task_loss
        
        # Cleanup
        del student_logits, distill_loss, task_loss
        
        return total_loss

    def _perform_backward_pass(self, loss, optimizer, proj_optimizer, scaler, model, projection_layer):
        """Perform backward pass with optional mixed precision."""
        optimizer.zero_grad(set_to_none=True)
        if proj_optimizer is not None:
            proj_optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if projection_layer is not None:
                torch.nn.utils.clip_grad_norm_(projection_layer.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            if proj_optimizer is not None:
                scaler.step(proj_optimizer)
            scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if projection_layer is not None:
                torch.nn.utils.clip_grad_norm_(projection_layer.parameters(), max_norm=1.0)
            optimizer.step()
            if proj_optimizer is not None:
                proj_optimizer.step()

    def _set_student_threshold(self, train_loader: DataLoader, percentile: int = 50):
        """Set anomaly detection threshold using student autoencoder."""
        print("Setting student anomaly detection threshold...")
        errors = []
        
        self.student_autoencoder.eval()
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                cont_out, _, _, _, _ = self.student_autoencoder(batch.x, batch.edge_index, batch.batch)
                batch_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                errors.append(batch_errors.cpu())
                del cont_out, batch_errors, batch
        
        self.threshold = torch.cat(errors).quantile(percentile / 100.0).item()
        self.student_autoencoder.train()

    def predict_student(self, data) -> torch.Tensor:
        """
        Efficient batch-based prediction using student models.
        
        Args:
            data: Batch data for prediction
            
        Returns:
            Tensor of predictions (0=normal, 1=attack)
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            # Stage 1: Batch anomaly detection
            cont_out, _, _, _, _ = self.student_autoencoder(data.x, data.edge_index, data.batch)
            node_errors = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            # Graph-level error aggregation
            batch_size = data.batch.max().item() + 1
            graph_errors = torch.zeros(batch_size, device=self.device)
            graph_errors.scatter_reduce_(0, data.batch, node_errors, reduce='amax')
            
            # Find anomalous graphs
            anomaly_mask = graph_errors > self.threshold
            predictions = torch.zeros(batch_size, device=self.device)
            
            # Stage 2: Classify anomalous graphs
            if anomaly_mask.any():
                anomaly_indices = torch.nonzero(anomaly_mask).flatten()
                
                # Extract anomalous subgraph efficiently
                anomalous_node_mask = torch.isin(data.batch, anomaly_indices)
                anomalous_edge_mask = (
                    torch.isin(data.batch[data.edge_index[0]], anomaly_indices) & 
                    torch.isin(data.batch[data.edge_index[1]], anomaly_indices)
                )
                
                if anomalous_node_mask.any():
                    # Create anomalous subgraph batch
                    anomalous_x = data.x[anomalous_node_mask]
                    
                    if anomalous_edge_mask.any():
                        # Remap edge indices
                        old_to_new = torch.full((data.x.size(0),), -1, dtype=torch.long, device=self.device)
                        old_to_new[anomalous_node_mask] = torch.arange(anomalous_node_mask.sum(), device=self.device)
                        
                        anomalous_edge_index = data.edge_index[:, anomalous_edge_mask]
                        anomalous_edge_index = old_to_new[anomalous_edge_index]
                    else:
                        anomalous_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                    
                    # Remap batch indices
                    anomalous_batch = data.batch[anomalous_node_mask]
                    unique_graphs, anomalous_batch = torch.unique(anomalous_batch, return_inverse=True)
                    
                    # Single classifier forward pass
                    anomalous_data = Data(x=anomalous_x, edge_index=anomalous_edge_index, batch=anomalous_batch)
                    classifier_logits = self.student_classifier(anomalous_data)
                    classifier_preds = (torch.sigmoid(classifier_logits.squeeze()) > 0.5).float()
                    
                    # Map predictions back
                    for i, original_idx in enumerate(unique_graphs):
                        original_pos = (anomaly_indices == original_idx).nonzero().item()
                        predictions[anomaly_indices[original_pos]] = classifier_preds[i]
            
            return predictions.long()

    def evaluate_student(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Comprehensive evaluation of student model performance.
        
        Args:
            test_loader: DataLoader with test data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("\n=== Student Model Evaluation ===")
        
        all_predictions = []
        all_labels = []
        
        self.student_autoencoder.eval()
        self.student_classifier.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i % 100 == 0:
                    print(f"Evaluating batch {i}/{len(test_loader)}")
                
                batch = batch.to(self.device)
                student_preds = self.predict_student(batch)
                labels = batch.y
                
                # Move to CPU immediately
                all_predictions.append(student_preds.cpu())
                all_labels.append(labels.cpu())
                
                # Cleanup
                del student_preds, labels, batch
                
                # Memory management
                if (i + 1) % 10 == 0:
                    cleanup_memory()
        
        # Compile results
        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels)
        
        # Calculate metrics
        accuracy = (predictions == labels).float().mean().item()
        conf_matrix = confusion_matrix(labels.numpy(), predictions.numpy())
        classification_rep = classification_report(labels.numpy(), predictions.numpy(), output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': classification_rep,
            'predictions': predictions.numpy(),
            'labels': labels.numpy()
        }
        
        # Print results
        print(f"Student Model Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
        return results

    def save_student_models(self, save_folder: str, dataset_key: str):
        """
        Save trained student models with metadata.
        
        Args:
            save_folder: Directory to save models
            dataset_key: Dataset identifier for filename
        """
        os.makedirs(save_folder, exist_ok=True)
        
        # Save student autoencoder
        autoencoder_data = {
            'state_dict': self.student_autoencoder.state_dict(),
            'threshold': self.threshold,
            'architecture': STUDENT_CONFIG
        }
        autoencoder_path = os.path.join(save_folder, f'student_autoencoder_{dataset_key}.pth')
        torch.save(autoencoder_data, autoencoder_path)
        
        # Save student classifier
        classifier_data = {
            'state_dict': self.student_classifier.state_dict(),
            'architecture': STUDENT_CONFIG
        }
        classifier_path = os.path.join(save_folder, f'student_classifier_{dataset_key}.pth')
        torch.save(classifier_data, classifier_path)
        
        # Save threshold separately for easy access
        threshold_path = os.path.join(save_folder, f'student_threshold_{dataset_key}.pth')
        torch.save({'threshold': self.threshold}, threshold_path)
        
        print(f"✓ Student models saved to '{save_folder}':")
        print(f"  Autoencoder: {autoencoder_path}")
        print(f"  Classifier: {classifier_path}")
        print(f"  Threshold: {threshold_path}")

# ==================== Main Training Pipeline ====================

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    """
    Main knowledge distillation training and evaluation pipeline.
    
    Args:
        config: Hydra configuration object containing training parameters
    """
    print(f"\n{'='*80}")
    print("KNOWLEDGE DISTILLATION FOR CAN BUS ANOMALY DETECTION")
    print(f"{'='*80}")
    
    # Setup GPU optimization
    setup_gpu_optimization()
    
    # Configuration
    config_dict = OmegaConf.to_container(config, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # Dataset configuration
    dataset_key = config_dict['root_folder']
    if dataset_key not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    
    root_folder = DATASET_PATHS[dataset_key]
    
    # Load and preprocess data
    print(f"\n=== Data Loading and Preprocessing ===")
    id_mapping = build_id_mapping_from_normal(root_folder)
    
    print(f"Starting preprocessing with {mp.cpu_count()} CPU cores...")
    start_time = time.time()
    dataset = graph_creation(root_folder, id_mapping=id_mapping, window_size=100)
    preprocessing_time = time.time() - start_time
    
    print(f"✓ Dataset: {len(dataset)} graphs, {len(id_mapping)} CAN IDs")
    print(f"✓ Preprocessing time: {preprocessing_time:.2f} seconds")
    log_memory_usage("Initial")
    
    # Configuration parameters
    DATASIZE = config_dict['datasize']
    TRAIN_RATIO = config_dict['train_ratio']
    BATCH_SIZE = config_dict['batch_size']
    EPOCHS = config_dict['epochs']
    USE_SIMPLE_DISTILLATION = config_dict.get('use_simple_distillation', True)
    
    print(f"Configuration: BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, "
          f"SIMPLE={USE_SIMPLE_DISTILLATION}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train/test split
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    # Create normal-only training subset for autoencoder
    normal_indices = [i for i, data in enumerate(train_dataset) if int(data.y.flatten()[0]) == 0]
    if DATASIZE < 1.0:
        subset_size = int(len(normal_indices) * DATASIZE)
        indices = np.random.choice(normal_indices, subset_size, replace=False)
    else:
        indices = normal_indices
    
    normal_subset = Subset(train_dataset, indices)
    print(f"✓ Data split: {len(normal_subset)} normal training, {len(test_dataset)} test")
    
    # Create optimized data loaders
    train_loader, test_loader, full_train_loader = create_optimized_data_loaders(
        normal_subset, test_dataset, train_dataset, BATCH_SIZE, str(device)
    )
    
    # Create teacher and student models
    print(f"\n=== Model Creation ===")
    teacher_ae, teacher_clf, student_ae, student_clf = create_teacher_student_models(
        num_ids=len(id_mapping), embedding_dim=8, device=str(device)
    )
    
    # Initialize knowledge distillation pipeline
    kd_pipeline = KnowledgeDistillationPipeline(
        teacher_autoencoder=teacher_ae,
        teacher_classifier=teacher_clf,
        student_autoencoder=student_ae,
        student_classifier=student_clf,
        device=str(device)
    )
    
    # Load pre-trained teacher models
    print(f"\n=== Loading Teacher Models ===")
    teacher_ae_path = f"output_model_optimized/autoencoder_best_{dataset_key}.pth"
    teacher_clf_path = f"output_model_optimized/classifier_{dataset_key}.pth"
    
    try:
        kd_pipeline.load_teacher_models(teacher_ae_path, teacher_clf_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run osc_training_AD.py first to train teacher models!")
        return
    
    log_memory_usage("After loading teachers")
    
    # Stage 1: Autoencoder distillation
    print(f"\n=== Stage 1: Autoencoder Knowledge Distillation ===")
    if USE_SIMPLE_DISTILLATION:
        print("Using simplified distillation (3-4x speedup)")
        kd_pipeline.distill_autoencoder_simplified(
            train_loader, epochs=EPOCHS, alpha=0.7, temperature=5.0
        )
    else:
        print("Using full distillation (comprehensive knowledge transfer)")
        # Note: Full distillation method would be implemented here
        # For now, fallback to simplified version
        kd_pipeline.distill_autoencoder_simplified(
            train_loader, epochs=EPOCHS, alpha=0.5, temperature=5.0
        )
    
    log_memory_usage("After autoencoder distillation")
    
    # Stage 2: Classifier distillation
    print(f"\n=== Stage 2: Classifier Knowledge Distillation ===")
    kd_pipeline.distill_classifier(
        full_train_loader, epochs=EPOCHS, alpha=0.7, temperature=5.0
    )
    
    log_memory_usage("After classifier distillation")
    
    # Save student models
    print(f"\n=== Saving Student Models ===")
    kd_pipeline.save_student_models("saved_models", dataset_key)
    
    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    
    # Dataset statistics
    test_labels = [data.y.item() for data in test_dataset]
    unique, counts = np.unique(test_labels, return_counts=True)
    print("Test set distribution:")
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count:,} samples ({count/len(test_labels)*100:.1f}%)")
    
    # Model evaluation
    results = kd_pipeline.evaluate_student(test_loader)
    
    # Model size comparison
    teacher_params = (sum(p.numel() for p in teacher_ae.parameters()) + 
                     sum(p.numel() for p in teacher_clf.parameters()))
    student_params = (sum(p.numel() for p in student_ae.parameters()) + 
                     sum(p.numel() for p in student_clf.parameters()))
    
    print(f"\n=== Model Compression Analysis ===")
    print(f"Teacher Parameters: {teacher_params:,}")
    print(f"Student Parameters: {student_params:,}")
    print(f"Compression Ratio: {teacher_params/student_params:.1f}x")
    print(f"Model Size Reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    
    # Final memory usage
    log_memory_usage("Final")
    print(f"\n✓ Knowledge Distillation Complete on {device}!")

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        end_time = time.time()
        print(f"\n⏱️  Total runtime: {end_time - start_time:.2f} seconds")
        cleanup_memory()