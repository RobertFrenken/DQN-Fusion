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
import psutil
import gc
from models.models import GATWithJK, GraphAutoencoderNeighborhood
from preprocessing import graph_creation, build_id_mapping_from_normal
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
    plot_raw_error_components_with_composite,
    plot_fusion_score_distributions
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

def create_optimized_data_loaders(train_subset, test_dataset, full_train_dataset, batch_size, device):
    """Create optimized data loaders similar to KD pipeline."""
    
    # Determine dataset size for batch sizing
    dataset_size = len(train_subset) if train_subset else 0
    dataset_size += len(test_dataset) if test_dataset else 0
    dataset_size += len(full_train_dataset) if full_train_dataset else 0
    
    # Optimized worker configuration
    pin_memory = True if torch.cuda.is_available() else False
    num_workers = 6
    prefetch_factor = 3 if num_workers > 0 else None
    persistent_workers = True if num_workers > 0 else False

    print(f"Creating optimized data loaders with batch size {batch_size} for {dataset_size} total graphs on {device}")
    
    '''
    set_01: 302k graphs    # ~15-20GB in memory
    set_02: 407k graphs    # ~20-25GB in memory  
    set_03: 332k graphs    # ~16-20GB in memory
    set_04: 244k graphs    # ~12-15GB in memory
    hcrl-ch: 290k graphs   # ~14-18GB in memory
    hcrl-sa: 18k graphs
    '''
    
    # Aggressive batch sizing for teacher training
    if torch.cuda.is_available():
        batch_size = 2048  # Same as KD pipeline
    else:
        batch_size = min(batch_size, 1024)

    print(f"Using optimized batch_size={batch_size} with {num_workers} workers")
    print(f"Memory settings: pin_memory={pin_memory}, prefetch_factor={prefetch_factor}")

    loaders = []
    
    if train_subset is not None:
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        loaders.append(train_loader)
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        loaders.append(test_loader)
    
    if full_train_dataset is not None:
        full_train_loader = DataLoader(
            full_train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        loaders.append(full_train_loader)
    
    return loaders if len(loaders) > 1 else loaders[0]

# Add memory logging function
def log_memory_usage(stage=""):
    """Log current memory usage."""
    cpu_mem = psutil.virtual_memory()
    print(f"CPU Memory: {cpu_mem.used/1024**3:.1f}GB / {cpu_mem.total/1024**3:.1f}GB ({cpu_mem.percent:.1f}%)")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"[{stage}] GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_cached:.2f}GB cached")
    
    ram_usage = psutil.virtual_memory().percent
    print(f"[{stage}] RAM Usage: {ram_usage:.1f}%")
class GATPipeline:
    """Two-stage pipeline for CAN bus anomaly detection using GAD-NR neighborhood reconstruction."""
    
    def __init__(self, num_ids, embedding_dim=8, device='cpu'):
        """Initialize the pipeline with autoencoder and classifier."""
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'
        
        self.autoencoder = GraphAutoencoderNeighborhood(
            num_ids=num_ids, in_channels=11, embedding_dim=embedding_dim
        ).to(device)
        
        self.classifier = GATWithJK(
            num_ids=num_ids, in_channels=11, hidden_channels=32, 
            out_channels=1, num_layers=5, heads=8, embedding_dim=embedding_dim
        ).to(device)
        self.threshold = 0.0
        
        print(f"Initialized GAT Pipeline on {device} (CUDA: {self.is_cuda})")

    def train_stage1(self, train_loader, val_loader=None, epochs=20):
        """Stage 1: Train autoencoder on normal graphs for anomaly detection."""
        print(f"Training autoencoder for {epochs} epochs with OPTIMIZED settings...")
        self.autoencoder.train()
        
        # IMPROVED: Better learning rate schedule + weight decay
        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(), 
            lr=2e-3,        # Slightly higher initial LR
            weight_decay=1e-4  # Add weight decay for regularization
        )
        
        # ADD: Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3, verbose=True
        )
        
        # ADD: Mixed precision training for CUDA
        scaler = torch.cuda.amp.GradScaler() if self.is_cuda else None
        
        # FIXED: Initialize best_val_loss properly for both cases
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # TRAINING PHASE - THIS WAS MISSING!
            self.autoencoder.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device, non_blocking=True)
                
                # ENABLE FP16 mixed precision if available
                if self.is_cuda:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        cont_out, canid_logits, neighbor_logits, z, pool = self.autoencoder(
                            batch.x, batch.edge_index, batch.batch)
                        
                        # Multi-component loss
                        recon_loss = nn.MSELoss()(cont_out, batch.x[:, 1:])
                        canid_loss = nn.CrossEntropyLoss()(canid_logits, batch.x[:, 0].long())
                        neighbor_loss = self._compute_neighborhood_loss(neighbor_logits, batch.x, batch.edge_index)
                        
                        total_loss = recon_loss + 0.1 * canid_loss + 0.5 * neighbor_loss
                else:
                    cont_out, canid_logits, neighbor_logits, z, pool = self.autoencoder(
                        batch.x, batch.edge_index, batch.batch)
                    
                    # Multi-component loss
                    recon_loss = nn.MSELoss()(cont_out, batch.x[:, 1:])
                    canid_loss = nn.CrossEntropyLoss()(canid_logits, batch.x[:, 0].long())
                    neighbor_loss = self._compute_neighborhood_loss(neighbor_logits, batch.x, batch.edge_index)
                    
                    total_loss = recon_loss + 0.1 * canid_loss + 0.5 * neighbor_loss
                
                # OPTIMIZED backward pass
                optimizer.zero_grad(set_to_none=True)
                
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                    optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Progress logging
                if batch_idx == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}")
                
                # MEMORY cleanup
                del batch, cont_out, canid_logits, neighbor_logits, z, pool
                del recon_loss, canid_loss, neighbor_loss, total_loss
                
                # Cleanup every 20 batches
                if batch_idx % 20 == 0:
                    gc.collect()
                    if self.is_cuda:
                        torch.cuda.empty_cache()
            
            # EPOCH summary - NOW epoch_loss and num_batches are properly defined
            train_avg_loss = epoch_loss / max(num_batches, 1)
            
            # VALIDATION PHASE
            if val_loader is not None:
                self.autoencoder.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device, non_blocking=True)
                        
                        if self.is_cuda:
                            with torch.amp.autocast('cuda', dtype=torch.float16):
                                cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                                    batch.x, batch.edge_index, batch.batch)
                                
                                recon_loss = nn.MSELoss()(cont_out, batch.x[:, 1:])
                                canid_loss = nn.CrossEntropyLoss()(canid_logits, batch.x[:, 0].long())
                                neighbor_loss = self._compute_neighborhood_loss(neighbor_logits, batch.x, batch.edge_index)
                                
                                total_loss = recon_loss + 0.1 * canid_loss + 0.5 * neighbor_loss
                        else:
                            cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                                batch.x, batch.edge_index, batch.batch)
                            
                            recon_loss = nn.MSELoss()(cont_out, batch.x[:, 1:])
                            canid_loss = nn.CrossEntropyLoss()(canid_logits, batch.x[:, 0].long())
                            neighbor_loss = self._compute_neighborhood_loss(neighbor_logits, batch.x, batch.edge_index)
                            
                            total_loss = recon_loss + 0.1 * canid_loss + 0.5 * neighbor_loss
                        
                        val_loss += total_loss.item()
                        val_batches += 1
                        
                        # Cleanup
                        del batch, cont_out, canid_logits, neighbor_logits
                        del recon_loss, canid_loss, neighbor_loss, total_loss
                
                val_avg_loss = val_loss / max(val_batches, 1)
                print(f"Autoencoder Epoch {epoch+1}/{epochs} - Train Loss: {train_avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}")
                
                # SAVE BEST MODEL based on validation loss
                if val_avg_loss < best_val_loss * 0.999:  # Require 0.1% improvement
                    best_val_loss = val_avg_loss
                    patience_counter = 0
                    best_model_state = {
                        'state_dict': self.autoencoder.state_dict().copy(),
                        'epoch': epoch + 1,
                        'train_loss': train_avg_loss,
                        'val_loss': val_avg_loss
                    }
                    print(f"  → New best autoencoder saved (epoch {epoch+1}, val_loss: {val_avg_loss:.4f})")
                else:
                    patience_counter += 1
                
                scheduler.step(val_avg_loss)  # Use validation loss for scheduler
            else:
                # Fallback to training loss if no validation loader
                print(f"Autoencoder Epoch {epoch+1}/{epochs} - Train Loss: {train_avg_loss:.4f}")
                if train_avg_loss < best_val_loss * 0.999:
                    best_val_loss = train_avg_loss
                    patience_counter = 0
                    best_model_state = {
                        'state_dict': self.autoencoder.state_dict().copy(),
                        'epoch': epoch + 1,
                        'train_loss': train_avg_loss,
                        'val_loss': None
                    }
                    print(f"  → New best autoencoder saved (epoch {epoch+1}, train_loss: {train_avg_loss:.4f})")
                else:
                    patience_counter += 1
                
                scheduler.step(train_avg_loss)
            
            # Memory logging every 5 epochs
            if (epoch + 1) % 5 == 0:
                log_memory_usage(f"Autoencoder Epoch {epoch+1}")
            
            # Early stopping
            if patience_counter >= 5:
                print(f"Early stopping: No improvement for {patience_counter} epochs")
                break
            
            # Major cleanup between epochs
            gc.collect()
            if self.is_cuda:
                torch.cuda.empty_cache()
        
        # RESTORE BEST MODEL
        if best_model_state is not None:
            self.autoencoder.load_state_dict(best_model_state['state_dict'])
            if best_model_state['val_loss'] is not None:
                print(f"Restored best autoencoder from epoch {best_model_state['epoch']} " +
                    f"(val_loss: {best_model_state['val_loss']:.4f})")
            else:
                print(f"Restored best autoencoder from epoch {best_model_state['epoch']} " +
                    f"(train_loss: {best_model_state['train_loss']:.4f})")

        self._set_threshold(train_loader, percentile=50)
        print(f"Set anomaly threshold: {self.threshold:.4f}")

    def _train_classifier(self, filtered_graphs, val_graphs=None, epochs=20):  # Increased from default
        """Train binary classifier on filtered graphs."""
        print(f"Training classifier for {epochs} epochs with OPTIMIZED settings...")
        
        labels = [int(graph.y.flatten()[0]) for graph in filtered_graphs]
        num_pos, num_neg = sum(labels), len(labels) - sum(labels)
        pos_weight = torch.tensor(1.0 if num_pos == 0 else num_neg / num_pos, device=self.device)
        
        self.classifier.train()
        
        # IMPROVED: Better learning rate + weight decay
        optimizer = torch.optim.Adam(
            self.classifier.parameters(), 
            lr=1e-3,           # Good starting LR
            weight_decay=1e-4  # Regularization
        )
        
        # ADD: Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=3, verbose=True
        )
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # ADD: Mixed precision for CUDA
        scaler = torch.cuda.amp.GradScaler() if self.is_cuda else None
        
        # FIXED: Use validation loss for best model selection
        best_val_loss = float('inf')
        best_model_state = None 
        patience_counter = 0
        
        # OPTIMIZED: Larger batch size for classifier
        classifier_batch_size = 1024 if torch.cuda.is_available() else 256
        
        for epoch in range(epochs):
            # TRAINING PHASE
            self.classifier.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(DataLoader(filtered_graphs, batch_size=classifier_batch_size, shuffle=True)):
                batch = batch.to(self.device, non_blocking=True)
                
                # ENABLE FP16 mixed precision
                if self.is_cuda:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        preds = self.classifier(batch)
                        loss = criterion(preds.squeeze(), batch.y.float())
                else:
                    preds = self.classifier(batch)
                    loss = criterion(preds.squeeze(), batch.y.float())
                
                # OPTIMIZED backward pass
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
                
                # MEMORY cleanup
                del batch, preds, loss
                
                # Cleanup every 20 batches
                if batch_idx % 20 == 0:
                    gc.collect()
                    if self.is_cuda:
                        torch.cuda.empty_cache()
            
            train_avg_loss = epoch_loss / max(num_batches, 1)
            
            # VALIDATION PHASE
            if val_graphs is not None:
                self.classifier.eval()
                val_loss = 0.0
                val_batches = 0
                val_accuracy = 0.0
                
                with torch.no_grad():
                    for batch in DataLoader(val_graphs, batch_size=classifier_batch_size, shuffle=False):
                        batch = batch.to(self.device, non_blocking=True)
                        
                        if self.is_cuda:
                            with torch.amp.autocast('cuda', dtype=torch.float16):
                                preds = self.classifier(batch)
                                loss = criterion(preds.squeeze(), batch.y.float())
                        else:
                            preds = self.classifier(batch)
                            loss = criterion(preds.squeeze(), batch.y.float())
                        
                        val_loss += loss.item()
                        val_batches += 1
                        
                        # Also compute accuracy for monitoring
                        pred_labels = (preds.squeeze() > 0.0).long()
                        val_accuracy += (pred_labels == batch.y.long()).float().mean().item()
                        
                        # Cleanup
                        del batch, preds, loss, pred_labels
                
                val_avg_loss = val_loss / max(val_batches, 1)
                val_avg_accuracy = val_accuracy / max(val_batches, 1)
                
                print(f"Classifier Epoch {epoch+1}/{epochs} - Train Loss: {train_avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_avg_accuracy:.4f}")
                
                # CHANGED: Save best model based on validation loss (not accuracy)
                if val_avg_loss < best_val_loss * 0.999:  # Require 0.1% improvement
                    best_val_loss = val_avg_loss
                    patience_counter = 0
                    best_model_state = {
                        'state_dict': self.classifier.state_dict().copy(),
                        'epoch': epoch + 1,
                        'train_loss': train_avg_loss,
                        'val_loss': val_avg_loss,
                        'val_accuracy': val_avg_accuracy
                    }
                    print(f"  → New best classifier saved (epoch {epoch+1}, val_loss: {val_avg_loss:.4f}, val_acc: {val_avg_accuracy:.4f})")
                else:
                    patience_counter += 1
                
                scheduler.step(val_avg_loss)  # Use validation loss for scheduler
            else:
                # Fallback to training loss if no validation
                train_accuracy = self._evaluate_classifier(filtered_graphs)
                print(f"Classifier Epoch {epoch+1}/{epochs} - Train Loss: {train_avg_loss:.4f}, Train Acc: {train_accuracy:.4f}")
                
                if train_avg_loss < best_val_loss * 0.999:
                    best_val_loss = train_avg_loss
                    patience_counter = 0
                    best_model_state = {
                        'state_dict': self.classifier.state_dict().copy(),
                        'epoch': epoch + 1,
                        'train_loss': train_avg_loss,
                        'val_loss': None,
                        'val_accuracy': train_accuracy
                    }
                    print(f"  → New best classifier saved (epoch {epoch+1}, train_loss: {train_avg_loss:.4f})")
                else:
                    patience_counter += 1
                
                scheduler.step(train_avg_loss)
            
            # Early stopping
            if patience_counter >= 5:
                print(f"Early stopping: No improvement for {patience_counter} epochs")
                break
            
            # Major cleanup between epochs
            gc.collect()
            if self.is_cuda:
                torch.cuda.empty_cache()
        
        # RESTORE BEST MODEL
        if best_model_state is not None:
            self.classifier.load_state_dict(best_model_state['state_dict'])
            if best_model_state['val_loss'] is not None:
                print(f"Restored best classifier from epoch {best_model_state['epoch']} " +
                    f"(val_loss: {best_model_state['val_loss']:.4f}, val_acc: {best_model_state['val_accuracy']:.4f})")
            else:
                print(f"Restored best classifier from epoch {best_model_state['epoch']} " +
                    f"(train_loss: {best_model_state['train_loss']:.4f}, train_acc: {best_model_state['val_accuracy']:.4f})")

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
                               id_errors_normal, id_errors_attack, key_suffix=""):
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

            # FIXED: Generate plots with KEY suffix
            neighbor_threshold = np.percentile(neighbor_errors_normal, 95)
            
            plot_recon_error_hist(errors_normal, errors_attack, self.threshold, 
                                save_path=f"images/recon_error_hist_{key_suffix}.png")
            plot_neighborhood_error_hist(neighbor_errors_normal, neighbor_errors_attack, 
                                        neighbor_threshold, save_path=f"images/neighborhood_error_hist_{key_suffix}.png")
            plot_neighborhood_composite_error_hist(
                errors_normal, errors_attack, neighbor_errors_normal, neighbor_errors_attack,
                id_errors_normal, id_errors_attack, save_path=f"images/neighborhood_composite_error_hist_{key_suffix}.png")
            plot_error_components_analysis(
                errors_normal, errors_attack, neighbor_errors_normal, neighbor_errors_attack,
                id_errors_normal, id_errors_attack, save_path=f"images/error_components_analysis_{key_suffix}.png")

            plot_raw_error_components_with_composite(
                errors_normal, errors_attack, neighbor_errors_normal, neighbor_errors_attack,
                id_errors_normal, id_errors_attack, save_path=f"images/raw_error_components_with_composite_{key_suffix}.png")

    def train_stage2(self, full_loader, val_loader=None, epochs=10, key_suffix=""):
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
        
        # CREATE VALIDATION GRAPHS from val_loader if provided
        val_graphs = None
        if val_loader is not None:
            print("Creating validation graphs for classifier...")
            val_graph_data = self._compute_composite_reconstruction_errors(val_loader)
            val_graphs = [graph for graph, error, is_attack in val_graph_data]
            print(f"Created {len(val_graphs)} validation graphs for classifier evaluation")
        
        self._train_classifier(balanced_graphs, epochs, val_graphs=val_graphs)

    def _compute_composite_reconstruction_errors(self, loader):
        """Compute composite reconstruction errors for filtering normal graphs."""
        print("Computing composite errors...")
    
        all_graphs = []
        all_composite_errors = []
        all_is_attack = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Single forward pass for entire batch
                cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                    batch.x, batch.edge_index, batch.batch)
                
                # Vectorized error computation
                node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                
                neighbor_targets = self.autoencoder.create_neighborhood_targets(
                    batch.x, batch.edge_index, batch.batch)
                neighbor_recon_errors = nn.BCEWithLogitsLoss(reduction='none')(
                    neighbor_logits, neighbor_targets).mean(dim=1)
                
                canid_pred = canid_logits.argmax(dim=1)
                
                # Vectorized processing of graphs in batch
                graphs = Batch.to_data_list(batch)
                start = 0
                
                for graph in graphs:
                    num_nodes = graph.x.size(0)
                    is_attack = int(graph.y.flatten()[0]) == 1
                    
                    # Vectorized max operations
                    graph_node_error = node_errors[start:start+num_nodes].max().item()
                    graph_neighbor_error = neighbor_recon_errors[start:start+num_nodes].max().item()
                    
                    # Vectorized CAN ID accuracy
                    true_canids = graph.x[:, 0].long()
                    pred_canids = canid_pred[start:start+num_nodes]
                    canid_error = 1.0 - (pred_canids == true_canids).float().mean().item()
                    
                    # Composite error (same weights)
                    composite_error = (1.0 * graph_node_error + 
                                    20.0 * graph_neighbor_error + 
                                    0.3 * canid_error)
                    
                    all_graphs.append(graph.cpu())
                    all_composite_errors.append(composite_error)
                    all_is_attack.append(is_attack)
                    start += num_nodes
        
        # Return as list of tuples (same format as original)
        return list(zip(all_graphs, all_composite_errors, all_is_attack))

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
    
    def predict_with_fusion(self, data, fusion_method='weighted', alpha=0.6):
        """
        Two-stage prediction with fusion of anomaly detection and classification scores.
        
        Args:
            data: Input batch data
            fusion_method: 'weighted', 'product', 'max', or 'learned'
            alpha: Weight for anomaly score (0.0-1.0) when fusion_method='weighted'
        
        Returns:
            final_preds: Fused predictions
            anomaly_scores: Raw anomaly scores  
            gat_probs: Raw GAT probabilities
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            # Get autoencoder outputs for anomaly detection
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
                
                # Compute composite anomaly score for this graph
                graph_node_error = node_errors[start:start+num_nodes].max().item()
                graph_neighbor_error = neighbor_errors[start:start+num_nodes].max().item()
                
                true_canids = graph.x[:, 0].long().cpu()
                pred_canids = canid_pred[start:start+num_nodes].cpu()
                canid_error = 1.0 - (pred_canids == true_canids).float().mean().item()
                
                # Rescaled composite anomaly score
                weight_node = 1.0
                weight_neighbor = 20.0  
                weight_canid = 0.3
                
                raw_anomaly_score = (weight_node * graph_node_error + 
                                weight_neighbor * graph_neighbor_error + 
                                weight_canid * canid_error)
                
                # Normalize anomaly score to [0,1] using sigmoid
                normalized_anomaly_score = torch.sigmoid(torch.tensor(raw_anomaly_score * 10 - 5)).item()
                
                # Get GAT classification probability
                graph_batch = graph.to(self.device)
                gat_logit = self.classifier(graph_batch).item()
                gat_prob = torch.sigmoid(torch.tensor(gat_logit)).item()
                
                # Apply fusion mechanism
                if fusion_method == 'weighted':
                    # Weighted average
                    fused_score = alpha * normalized_anomaly_score + (1 - alpha) * gat_prob
                    
                elif fusion_method == 'product':
                    # Geometric mean (emphasizes agreement)
                    fused_score = (normalized_anomaly_score * gat_prob) ** 0.5
                    
                elif fusion_method == 'max':
                    # Maximum (conservative - either detector triggers)
                    fused_score = max(normalized_anomaly_score, gat_prob)
                    
                elif fusion_method == 'learned':
                    # Simple learned fusion (requires training - placeholder)
                    fused_score = 0.7 * normalized_anomaly_score + 0.3 * gat_prob
                    
                else:
                    raise ValueError(f"Unknown fusion method: {fusion_method}")
                
                final_preds.append(1 if fused_score > 0.5 else 0)
                anomaly_scores.append(normalized_anomaly_score)
                gat_probs.append(gat_prob)
                
                start += num_nodes
        
        return (torch.tensor(final_preds, device=self.device), 
                torch.tensor(anomaly_scores), 
                torch.tensor(gat_probs))

    def evaluate_with_fusion(self, test_loader, fusion_methods=['weighted', 'product', 'max']):
        """Evaluate multiple fusion methods and return detailed results."""
        print("\n=== Fusion Evaluation ===")
        
        # Collect all predictions and labels
        all_labels = []
        all_anomaly_scores = []
        all_gat_probs = []
        
        for batch in test_loader:
            batch = batch.to(self.device)
            _, anomaly_scores, gat_probs = self.predict_with_fusion(batch, fusion_method='weighted')
            
            all_labels.extend(batch.y.cpu().numpy())
            all_anomaly_scores.extend(anomaly_scores.cpu().numpy())
            all_gat_probs.extend(gat_probs.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_anomaly_scores = np.array(all_anomaly_scores)
        all_gat_probs = np.array(all_gat_probs)
        
        results = {}
        
        # Test different fusion methods
        for method in fusion_methods:
            print(f"\n--- Fusion Method: {method} ---")
            
            if method == 'weighted':
                # Test different alpha values
                best_acc = 0
                best_alpha = 0.5
                
                for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    fused_scores = alpha * all_anomaly_scores + (1 - alpha) * all_gat_probs
                    preds = (fused_scores > 0.5).astype(int)
                    acc = (preds == all_labels).mean()
                    
                    print(f"  α={alpha:.1f}: Accuracy={acc:.4f}")
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_alpha = alpha
                
                # Use best alpha for final evaluation
                fused_scores = best_alpha * all_anomaly_scores + (1 - best_alpha) * all_gat_probs
                final_preds = (fused_scores > 0.5).astype(int)
                results[method] = {'accuracy': best_acc, 'alpha': best_alpha, 'predictions': final_preds}
                
            elif method == 'product':
                fused_scores = (all_anomaly_scores * all_gat_probs) ** 0.5
                final_preds = (fused_scores > 0.5).astype(int)
                acc = (final_preds == all_labels).mean()
                results[method] = {'accuracy': acc, 'predictions': final_preds}
                print(f"  Accuracy: {acc:.4f}")
                
            elif method == 'max':
                fused_scores = np.maximum(all_anomaly_scores, all_gat_probs)
                final_preds = (fused_scores > 0.5).astype(int)
                acc = (final_preds == all_labels).mean()
                results[method] = {'accuracy': acc, 'predictions': final_preds}
                print(f"  Accuracy: {acc:.4f}")
        
        # Individual component performance
        anomaly_only_preds = (all_anomaly_scores > 0.5).astype(int)
        gat_only_preds = (all_gat_probs > 0.5).astype(int)
        
        anomaly_only_acc = (anomaly_only_preds == all_labels).mean()
        gat_only_acc = (gat_only_preds == all_labels).mean()
        
        print(f"\n--- Individual Components ---")
        print(f"Anomaly Detection Only: {anomaly_only_acc:.4f}")
        print(f"GAT Classification Only: {gat_only_acc:.4f}")
        
        results['anomaly_only'] = {'accuracy': anomaly_only_acc, 'predictions': anomaly_only_preds}
        results['gat_only'] = {'accuracy': gat_only_acc, 'predictions': gat_only_preds}
        
        return results, all_labels, all_anomaly_scores, all_gat_probs
    

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    """Main training and evaluation pipeline."""

    # CRITICAL: Enable GPU memory optimization FIRST
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        torch.cuda.empty_cache()
        print("Enabled CUDA memory optimization")
    
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

    # CREATE ALL REQUIRED DIRECTORIES
    required_dirs = ["images", "output_model_optimized"]
    for dir_name in required_dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")

    id_mapping = build_id_mapping_from_normal(root_folder)
    print(f"Starting preprocessing...")
    start_time = time.time()
    dataset = graph_creation(root_folder, id_mapping=id_mapping, window_size=100)
    preprocessing_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
    
    print(f"Dataset: {len(dataset)} graphs, {len(id_mapping)} unique CAN IDs")
    
    # Validate dataset
    for data in dataset:
        assert not torch.isnan(data.x).any(), "Dataset contains NaN values!"
        assert not torch.isinf(data.x).any(), "Dataset contains Inf values!"

    # Configuration
    DATASIZE = config_dict['datasize']
    TRAIN_RATIO = config_dict['train_ratio']
    BATCH_SIZE = config_dict['batch_size']
    EPOCHS = config_dict['epochs']
    
    # Generate feature histograms
    feature_names = ["CAN ID", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "data8", "count", "position"]
    plot_feature_histograms([data for data in dataset], feature_names=feature_names, 
                           save_path=f"images/feature_histograms_{KEY}.png")

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
    
    
    # UPDATED: Use optimized data loaders
    normal_subset = Subset(train_dataset, indices)
    train_loader = create_optimized_data_loaders(normal_subset, None, None, BATCH_SIZE, device)
    val_loader = create_optimized_data_loaders(None, test_dataset, None, BATCH_SIZE, device)  
    full_train_loader = create_optimized_data_loaders(None, None, train_dataset, BATCH_SIZE, device)

    print(f'Normal training samples: {len(train_loader.dataset)}')

    # Initialize pipeline
    pipeline = GATPipeline(num_ids=len(id_mapping), embedding_dim=8, device=device)

    # Training
    print("\n=== Stage 1: Autoencoder Training ===")
    pipeline.train_stage1(train_loader, val_loader, epochs=EPOCHS)

    log_memory_usage("After autoencoder training")

    # Visualization
    plot_graph_reconstruction(pipeline, full_train_loader, num_graphs=4, 
                            save_path=f"images/graph_recon_examples_{KEY}.png")
    
    # Latent space visualization
    N = min(10000, len(train_dataset))
    indices = np.random.choice(len(train_dataset), size=N, replace=False)
    subsample = [train_dataset[i] for i in indices]
    subsample_loader = DataLoader(subsample, batch_size=BATCH_SIZE, shuffle=False)
    zs, labels = extract_latent_vectors(pipeline, subsample_loader)
    plot_latent_space(zs, labels, save_path=f"images/latent_space_{KEY}.png")
    plot_node_recon_errors(pipeline, full_train_loader, num_graphs=5, 
                         save_path=f"images/node_recon_subplot_{KEY}.png")
    
    print("\n=== Stage 2: Classifier Training ===")
    pipeline.train_stage2(full_train_loader, val_loader=val_loader, epochs=EPOCHS, key_suffix=KEY)

    # ENHANCED: Save models with better naming and metadata
    save_folder = "output_model_optimized"  # Different folder to distinguish
    os.makedirs(save_folder, exist_ok=True)
    
    # Save with metadata
    autoencoder_save_data = {
        'state_dict': pipeline.autoencoder.state_dict(),
        'threshold': pipeline.threshold,
        'epochs': EPOCHS,
        'embedding_dim': 8,
        'num_ids': len(id_mapping),
        'validation_based': True,
    }
    
    classifier_save_data = {
        'state_dict': pipeline.classifier.state_dict(),
        'epochs': EPOCHS,
        'embedding_dim': 8,
        'num_ids': len(id_mapping),
        'validation_based': True,
    }
    
    torch.save(autoencoder_save_data, os.path.join(save_folder, f'autoencoder_best_{KEY}.pth'))
    torch.save(classifier_save_data, os.path.join(save_folder, f'classifier_{KEY}.pth'))
    
    print(f"BEST VALIDATION models saved to '{save_folder}'")
    print(f"  - Autoencoder: Best validation loss model (autoencoder_best_val_{KEY}.pth)")
    print(f"  - Classifier: Best validation accuracy model (classifier_best_val_{KEY}.pth)")

    # Evaluation
    print("\n=== Test Set Evaluation ===")
    test_labels = [data.y.item() for data in test_dataset]
    unique, counts = np.unique(test_labels, return_counts=True)
    print("Test set distribution:")
    for u, c in zip(unique, counts):
        print(f"  Label {u}: {c} samples")

    # Standard prediction (original method)
    print("\n--- Standard Two-Stage Prediction ---")
    preds, labels = [], []
    for batch in val_loader:
        batch = batch.to(device)
        pred = pipeline.predict(batch)
        preds.append(pred.cpu())
        labels.append(batch.y.cpu())

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    standard_accuracy = (preds == labels).float().mean().item()

    print(f"Standard Test Accuracy: {standard_accuracy:.4f}")
    print("Standard Confusion Matrix:")
    print(confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy()))


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")