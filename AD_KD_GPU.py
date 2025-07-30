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
import psutil
import gc

def log_memory_usage(stage=""):
    """Log current memory usage."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"[{stage}] GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_cached:.2f}GB cached")
    
    ram_usage = psutil.virtual_memory().percent
    print(f"[{stage}] RAM Usage: {ram_usage:.1f}%")

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
    """Create memory-safe data loaders - NO MULTIPROCESSING."""
    
    # Determine dataset size for batch sizing
    dataset_size = len(train_subset) + len(test_dataset)
    
    # ALWAYS use single-threaded loading to prevent OOM
    pin_memory = False
    num_workers = 0  # ALWAYS 0 - no multiprocessing
    persistent_workers = False
    prefetch_factor = None
    
    # Adaptive batch sizing for memory safety
    if dataset_size > 300000:  # Very large dataset (like hcrl_ch)
        batch_size = min(batch_size, 1024)   # Very small batches
        print(f"Very large dataset detected: Using batch_size={batch_size}")
    elif dataset_size > 100000:  # Large dataset
        batch_size = min(batch_size, 2048)   # Small batches
        print(f"Large dataset detected: Using batch_size={batch_size}")
    else:  # Standard dataset
        batch_size = min(batch_size, 4096)  # Medium batches
        print(f"Standard dataset: Using batch_size={batch_size}")
    
    print("Using SINGLE-THREADED data loaders (no multiprocessing)")
    
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
    """Memory-optimized knowledge distillation pipeline."""
    
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
        """Load pre-trained teacher models."""
        print(f"Loading teacher autoencoder from: {autoencoder_path}")
        
        autoencoder_checkpoint = torch.load(autoencoder_path, map_location=self.device)
        
        if isinstance(autoencoder_checkpoint, dict):
            if 'state_dict' in autoencoder_checkpoint:
                autoencoder_state_dict = autoencoder_checkpoint['state_dict']
                if 'threshold' in autoencoder_checkpoint:
                    self.threshold = autoencoder_checkpoint['threshold']
            else:
                autoencoder_state_dict = autoencoder_checkpoint
        else:
            autoencoder_state_dict = autoencoder_checkpoint
        
        print(f"Loading teacher classifier from: {classifier_path}")
        classifier_checkpoint = torch.load(classifier_path, map_location=self.device)
        
        if isinstance(classifier_checkpoint, dict):
            if 'state_dict' in classifier_checkpoint:
                classifier_state_dict = classifier_checkpoint['state_dict']
            else:
                classifier_state_dict = classifier_checkpoint
        else:
            classifier_state_dict = classifier_checkpoint
        
        try:
            self.teacher_autoencoder.load_state_dict(autoencoder_state_dict)
            print("Teacher autoencoder loaded successfully!")
            
            self.teacher_classifier.load_state_dict(classifier_state_dict)
            print("Teacher classifier loaded successfully!")
            
        except RuntimeError as e:
            print(f"Architecture mismatch: {str(e)[:200]}...")
            raise e
        
        self.teacher_autoencoder.eval()
        self.teacher_classifier.eval()
        print("Teacher models loaded and set to evaluation mode!")

    def distill_autoencoder(self, train_loader, epochs=20, alpha=0.5, temperature=5.0):
        """Memory-optimized autoencoder distillation."""
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
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                
                if self.is_cuda:
                    with torch.cuda.amp.autocast():
                        total_loss = self._compute_autoencoder_loss(
                            batch, projection_layer, alpha, temperature)
                else:
                    total_loss = self._compute_autoencoder_loss(
                        batch, projection_layer, alpha, temperature)
                
                optimizer.zero_grad()
                if projection_layer is not None:
                    proj_optimizer.zero_grad()
                
                if scaler is not None:
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
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student_autoencoder.parameters(), max_norm=1.0)
                    if projection_layer is not None:
                        torch.nn.utils.clip_grad_norm_(projection_layer.parameters(), max_norm=1.0)
                    optimizer.step()
                    if projection_layer is not None:
                        proj_optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Cleanup batch
                del batch, total_loss
                
                # Frequent memory cleanup
                if batch_idx % 20 == 0:
                    gc.collect()
                    if self.is_cuda:
                        torch.cuda.empty_cache()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                print(f"Autoencoder Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
                log_memory_usage(f"Epoch {epoch+1}")
            
            # Major cleanup between epochs
            gc.collect()
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
        
        # Combine losses
        reconstruction_loss = cont_loss + canid_loss + neighbor_loss + 0.1 * student_kl_loss
        distillation_loss = feature_distill_loss + cont_distill_loss + canid_distill_loss + neighbor_distill_loss
        
        total_loss = (1 - alpha) * reconstruction_loss + alpha * distillation_loss
        return total_loss

    def distill_classifier(self, full_loader, epochs=20, alpha=0.7, temperature=5.0):
        """FAST batch-based classifier distillation - NO individual graph processing."""
        print(f"Distilling classifier for {epochs} epochs on {self.device}...")
        
        self.student_classifier.train()
        optimizer = torch.optim.Adam(self.student_classifier.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            processed_samples = 0
            
            # Process FULL BATCHES for speed
            for batch_idx, batch in enumerate(full_loader):
                batch = batch.to(self.device, non_blocking=True)
                
                try:
                    # Teacher predictions on FULL BATCH
                    with torch.no_grad():
                        teacher_logits = self.teacher_classifier(batch)
                    
                    # Student predictions on FULL BATCH
                    student_logits = self.student_classifier(batch)
                    
                    # Distillation loss
                    distill_loss = nn.KLDivLoss(reduction='batchmean')(
                        nn.LogSoftmax(dim=-1)(student_logits / temperature),
                        nn.Softmax(dim=-1)(teacher_logits / temperature)
                    ) * (temperature ** 2)
                    
                    # Task loss
                    task_loss = nn.BCEWithLogitsLoss()(
                        student_logits.squeeze(), batch.y.float()
                    )
                    
                    # Combined loss
                    total_loss = alpha * distill_loss + (1 - alpha) * task_loss
                    
                    # Backward pass
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student_classifier.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    batch_count += 1
                    processed_samples += batch.y.size(0)  # Number of graphs in batch
                    
                    # Cleanup
                    del teacher_logits, student_logits, distill_loss, task_loss, total_loss
                    
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")
                
                finally:
                    del batch
                    # Less frequent cleanup for speed
                    if batch_idx % 100 == 0:
                        gc.collect()
                        if self.is_cuda:
                            torch.cuda.empty_cache()
                
                # Progress update
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(full_loader)}, Processed: {processed_samples}")
                    if batch_idx % 500 == 0:
                        log_memory_usage(f"Classifier epoch {epoch+1}")
            
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"Classifier Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Batches: {batch_count}, Samples: {processed_samples}")
            
            # Major cleanup between epochs
            gc.collect()
            if self.is_cuda:
                torch.cuda.empty_cache()
            
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"Classifier Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Batches: {batch_count}")
            
            # Major cleanup between epochs
            gc.collect()
            if self.is_cuda:
                torch.cuda.empty_cache()

    def _set_threshold_student(self, train_loader, percentile=50):
        """Set anomaly detection threshold using student autoencoder."""
        errors = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                cont_out, _, _, _, _ = self.student_autoencoder(batch.x, batch.edge_index, batch.batch)
                batch_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                errors.append(batch_errors.cpu())  # Move to CPU immediately
                del cont_out, batch_errors, batch
        
        self.threshold = torch.cat(errors).quantile(percentile / 100.0).item()

    def predict_student(self, data):
        """Memory-efficient prediction."""
        data = data.to(self.device)
        
        with torch.no_grad():
            # Autoencoder pass
            cont_out, _, _, _, _ = self.student_autoencoder(data.x, data.edge_index, data.batch)
            node_errors = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            del cont_out
            
            # Graph-level error computation
            batch_size = data.batch.max().item() + 1
            graph_errors = torch.zeros(batch_size, device=self.device)
            graph_errors.scatter_reduce_(0, data.batch, node_errors, reduce='amax')
            
            del node_errors
            
            # Find anomalous graphs
            anomaly_mask = graph_errors > self.threshold
            preds = torch.zeros(batch_size, device=self.device)
            
            del graph_errors
            
            # Process anomalous graphs in small chunks
            if anomaly_mask.any():
                anomaly_indices = torch.nonzero(anomaly_mask).flatten()
                chunk_size = 16  # Very small chunks
                
                for chunk_start in range(0, len(anomaly_indices), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(anomaly_indices))
                    chunk_indices = anomaly_indices[chunk_start:chunk_end]
                    
                    subgraph_data_list = []
                    valid_indices = []
                    
                    for idx in chunk_indices:
                        node_mask = data.batch == idx
                        edge_mask = (data.batch[data.edge_index[0]] == idx) & (data.batch[data.edge_index[1]] == idx)
                        
                        if node_mask.sum() > 0:
                            subgraph_x = data.x[node_mask].clone()
                            
                            if edge_mask.sum() > 0:
                                subgraph_edge_index = data.edge_index[:, edge_mask].clone()
                                node_mapping = torch.full((data.x.size(0),), -1, dtype=torch.long, device=self.device)
                                node_mapping[node_mask] = torch.arange(node_mask.sum(), device=self.device)
                                subgraph_edge_index = node_mapping[subgraph_edge_index]
                            else:
                                subgraph_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                            
                            subgraph = Data(x=subgraph_x, edge_index=subgraph_edge_index)
                            subgraph_data_list.append(subgraph)
                            valid_indices.append(idx)
                    
                    # Batch classify this chunk
                    if subgraph_data_list:
                        try:
                            batched_anomalous = Batch.from_data_list(subgraph_data_list)
                            classifier_probs = self.student_classifier(batched_anomalous)
                            
                            if classifier_probs.dim() > 1:
                                classifier_probs = classifier_probs.flatten()
                            
                            for i, idx in enumerate(valid_indices):
                                if i < classifier_probs.size(0):
                                    preds[idx] = (classifier_probs[i] > 0.5).float()
                            
                            del batched_anomalous, classifier_probs
                            
                        except Exception as e:
                            print(f"Chunk classification failed: {e}")
                            for subgraph, idx in zip(subgraph_data_list, valid_indices):
                                try:
                                    prob = self.student_classifier(subgraph)
                                    preds[idx] = (prob.item() > 0.5)
                                except:
                                    preds[idx] = 0
                        
                        finally:
                            del subgraph_data_list, valid_indices
                            gc.collect()
                            if self.is_cuda:
                                torch.cuda.empty_cache()
            
            del anomaly_mask
            return preds.long()

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    """Memory-optimized knowledge distillation pipeline."""
    
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
    
    # Fix random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print(f"Dataset: {len(dataset)} graphs, {len(id_mapping)} unique CAN IDs")
    log_memory_usage("Initial")
    
    # Configuration
    DATASIZE = config_dict['datasize']
    TRAIN_RATIO = config_dict['train_ratio']
    BATCH_SIZE = config_dict['batch_size']
    EPOCHS = config_dict['epochs']
    
    print(f"Config - BATCH_SIZE: {BATCH_SIZE}, EPOCHS: {EPOCHS}")
    
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
    
    # Create memory-safe data loaders
    train_loader, test_loader, full_train_loader = create_data_loaders(
        normal_subset, test_dataset, train_dataset, BATCH_SIZE, device
    )
    
    # Create models
    teacher_ae, teacher_clf, student_ae, student_clf = create_teacher_student_models(
        num_ids=len(id_mapping), embedding_dim=8, device=device
    )
    
    # Initialize KD pipeline
    kd_pipeline = KnowledgeDistillationPipeline(
        teacher_autoencoder=teacher_ae,
        teacher_classifier=teacher_clf,
        student_autoencoder=student_ae,
        student_classifier=student_clf,
        device=device
    )
    
    # Load teacher models
    teacher_ae_path = f"output_model_1/autoencoder_best_{KEY}.pth"
    teacher_clf_path = f"output_model_1/classifier_{KEY}.pth"

    try:
        kd_pipeline.load_teacher_models(teacher_ae_path, teacher_clf_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run osc-training-AD.py first to train teacher models!")
        return
    
    log_memory_usage("After loading teachers")
    
    # Stage 1: Autoencoder distillation
    print("\n=== Stage 1: Autoencoder Knowledge Distillation ===")
    kd_pipeline.distill_autoencoder(train_loader, epochs=EPOCHS, alpha=0.5, temperature=5.0)
    
    log_memory_usage("After autoencoder distillation")
    
    # Stage 2: Classifier distillation
    print("\n=== Stage 2: Classifier Knowledge Distillation ===")
    kd_pipeline.distill_classifier(full_train_loader, epochs=EPOCHS, alpha=0.7, temperature=5.0)
    
    log_memory_usage("After classifier distillation")
    
    # Save student models
    save_folder = "saved_models"
    os.makedirs(save_folder, exist_ok=True)
    torch.save(kd_pipeline.student_autoencoder.state_dict(), 
               os.path.join(save_folder, f'student_autoencoder_{KEY}.pth'))
    torch.save(kd_pipeline.student_classifier.state_dict(), 
               os.path.join(save_folder, f'student_classifier_{KEY}.pth'))
    print(f"Student models saved to '{save_folder}'")
    
    # Evaluation
    print("\n=== Evaluation: Student Model Performance ===")
    
    all_student_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 100 == 0:
                print(f"Evaluating batch {i}/{len(test_loader)}")
                log_memory_usage(f"Eval batch {i}")
            
            batch = batch.to(device)
            student_preds = kd_pipeline.predict_student(batch)
            labels = batch.y
            
            # Move to CPU immediately
            all_student_preds.append(student_preds.cpu())
            all_labels.append(labels.cpu())
            
            # Cleanup
            del student_preds, labels, batch
            
            # Frequent cleanup
            if (i + 1) % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Results
    student_preds = torch.cat(all_student_preds)
    labels = torch.cat(all_labels)
    
    student_accuracy = (student_preds == labels).float().mean().item()

    print(f"\nStudent Model Accuracy: {student_accuracy:.4f}")
    print("Student Confusion Matrix:")
    print(confusion_matrix(labels.numpy(), student_preds.numpy()))

    # Model size comparison
    teacher_params = sum(p.numel() for p in teacher_ae.parameters()) + sum(p.numel() for p in teacher_clf.parameters())
    student_params = sum(p.numel() for p in student_ae.parameters()) + sum(p.numel() for p in student_clf.parameters())

    print(f"\nModel Size Comparison:")
    print(f"Teacher Parameters: {teacher_params:,}")
    print(f"Student Parameters: {student_params:,}")
    print(f"Compression Ratio: {teacher_params/student_params:.1f}x")

    log_memory_usage("Final")
    print(f"\nKnowledge Distillation Complete on {device}!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")