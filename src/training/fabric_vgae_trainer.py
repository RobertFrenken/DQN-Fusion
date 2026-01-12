"""
PyTorch Lightning Fabric trainer for VGAE (Variational Graph Autoencoder) models.

This trainer provides:
- Efficient VGAE model training with Fabric
- Variational loss handling (reconstruction + KL divergence)
- Dynamic batch sizing
- Mixed precision training
- Advanced logging and checkpointing
- Support for both node and neighborhood reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import time
from tqdm import tqdm

from src.utils.fabric_utils import FabricTrainerBase, DynamicBatchSizer
from src.models.models import GraphAutoencoderNeighborhood
from src.utils.losses import distillation_loss_fn

logger = logging.getLogger(__name__)


class FabricVGAETrainer(FabricTrainerBase):
    """
    PyTorch Lightning Fabric trainer for VGAE models with advanced optimizations.
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 fabric_config: Dict[str, Any] = None,
                 use_dynamic_batching: bool = True):
        """
        Initialize Fabric VGAE trainer.
        
        Args:
            model_config: Configuration for VGAE model
            training_config: Training hyperparameters
            fabric_config: Fabric-specific configuration
            use_dynamic_batching: Whether to use dynamic batch sizing
        """
        super().__init__(fabric_config)
        
        self.model_config = model_config
        self.training_config = training_config
        self.use_dynamic_batching = use_dynamic_batching
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup with Fabric
        self.model, self.optimizer, self.scheduler = self.setup_model_and_optimizer(
            self.model, self.optimizer, self.scheduler
        )
        
        # Loss configuration
        self.loss_weights = self._setup_loss_weights()
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        # Dynamic batch sizing
        self.optimal_batch_size = None
        if self.use_dynamic_batching:
            logger.info("Dynamic batch sizing will be determined during training setup")
    
    def _create_model(self) -> GraphAutoencoderNeighborhood:
        """Create VGAE model from configuration."""
        return GraphAutoencoderNeighborhood(**self.model_config)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.training_config.get('optimizer', 'adamw').lower()
        lr = self.training_config.get('learning_rate', 1e-3)
        weight_decay = self.training_config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adamw':
            return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.training_config.get('momentum', 0.9)
            return SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_config = self.training_config.get('scheduler')
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config.get('type', 'cosine').lower()
        
        if scheduler_type == 'onecycle':
            max_lr = scheduler_config.get('max_lr', self.training_config.get('learning_rate', 1e-3))
            total_steps = scheduler_config.get('total_steps', 1000)
            return OneCycleLR(self.optimizer, max_lr=max_lr, total_steps=total_steps)
        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 100)
            return CosineAnnealingLR(self.optimizer, T_max=T_max)
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}")
            return None
    
    def _setup_loss_weights(self) -> Dict[str, float]:
        """Setup loss weights for different components."""
        loss_config = self.training_config.get('loss', {})
        return {
            'reconstruction': loss_config.get('reconstruction_weight', 1.0),
            'kl_divergence': loss_config.get('kl_weight', 0.1),
            'canid_classification': loss_config.get('canid_weight', 1.0),
            'neighborhood': loss_config.get('neighborhood_weight', 0.5)
        }
    
    def determine_optimal_batch_size(self, sample_data) -> int:
        """
        Determine optimal batch size using dynamic sizing.
        
        Args:
            sample_data: Sample data for batch size estimation
            
        Returns:
            Optimal batch size
        """
        if not self.use_dynamic_batching:
            return self.training_config.get('batch_size', 32)
        
        try:
            batch_sizer = DynamicBatchSizer(
                model=self.model,
                sample_input=sample_data,
                target_memory_usage=self.training_config.get('target_memory_usage', 0.85),
                min_batch_size=self.training_config.get('min_batch_size', 1),
                max_batch_size=self.training_config.get('max_batch_size', 4096)  # Lower for VGAE
            )
            
            optimal_size = batch_sizer.estimate_optimal_batch_size()
            logger.info(f"Determined optimal batch size: {optimal_size}")
            return optimal_size
            
        except Exception as e:
            logger.warning(f"Dynamic batch sizing failed: {e}")
            fallback_size = self.training_config.get('batch_size', 32)
            logger.info(f"Using fallback batch size: {fallback_size}")
            return fallback_size
    
    def compute_vgae_losses(self, 
                           batch,
                           cont_out, 
                           canid_logits, 
                           neighbor_logits, 
                           kl_loss) -> Dict[str, torch.Tensor]:
        """
        Compute VGAE losses including reconstruction, KL divergence, and classification.
        
        Args:
            batch: Input batch
            cont_out: Continuous feature reconstruction
            canid_logits: CAN ID classification logits
            neighbor_logits: Neighborhood prediction logits
            kl_loss: KL divergence loss
            
        Returns:
            Dictionary of computed losses
        """
        # Node feature reconstruction loss (continuous features only)
        continuous_features = batch.x[:, 1:]  # Exclude CAN ID
        reconstruction_loss = F.mse_loss(cont_out, continuous_features)
        
        # CAN ID classification loss
        canid_targets = batch.x[:, 0].long()  # CAN IDs
        canid_loss = F.cross_entropy(canid_logits, canid_targets)
        
        # Neighborhood reconstruction loss
        neighbor_targets = self.model.create_neighborhood_targets(
            batch.x, batch.edge_index, batch.batch
        )
        neighborhood_loss = F.binary_cross_entropy_with_logits(
            neighbor_logits, neighbor_targets
        )
        
        return {
            'reconstruction': reconstruction_loss,
            'kl_divergence': kl_loss,
            'canid_classification': canid_loss,
            'neighborhood': neighborhood_loss
        }
    
    def train_epoch(self, 
                   train_loader, 
                   epoch: int,
                   accumulation_steps: int = 1) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            accumulation_steps: Gradient accumulation steps
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl_divergence': 0.0,
            'canid_classification': 0.0,
            'neighborhood': 0.0
        }
        
        all_canid_predictions = []
        all_canid_targets = []
        num_batches = len(train_loader)
        
        # Setup progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device (handled by Fabric)
            batch = self.fabric.to_device(batch)
            
            # Forward pass with autocast
            with self.fabric.autocast():
                cont_out, canid_logits, neighbor_logits, z, kl_loss = self.model(
                    batch.x, batch.edge_index, batch.batch
                )
                
                # Compute individual losses
                losses = self.compute_vgae_losses(
                    batch, cont_out, canid_logits, neighbor_logits, kl_loss
                )
                
                # Weighted total loss
                total_loss = (
                    self.loss_weights['reconstruction'] * losses['reconstruction'] +
                    self.loss_weights['kl_divergence'] * losses['kl_divergence'] +
                    self.loss_weights['canid_classification'] * losses['canid_classification'] +
                    self.loss_weights['neighborhood'] * losses['neighborhood']
                ) / accumulation_steps
            
            # Backward pass
            self.fabric.backward(total_loss)
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == num_batches - 1:
                # Gradient clipping
                if self.training_config.get('gradient_clip_val'):
                    self.fabric.clip_gradients(
                        self.model, 
                        self.optimizer,
                        max_norm=self.training_config['gradient_clip_val']
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.scheduler:
                    self.scheduler.step()
            
            # Collect metrics
            total_losses['total'] += total_loss.item() * accumulation_steps
            for loss_name, loss_value in losses.items():
                total_losses[loss_name] += loss_value.item()
            
            # CAN ID classification metrics
            canid_predictions = torch.argmax(canid_logits, dim=1)
            canid_targets = batch.x[:, 0].long()
            all_canid_predictions.extend(canid_predictions.cpu().numpy())
            all_canid_targets.extend(canid_targets.cpu().numpy())
            
            # Update progress bar
            current_loss = total_losses['total'] / (batch_idx + 1)
            pbar.set_postfix({
                'total_loss': f'{current_loss:.4f}',
                'recon': f'{total_losses["reconstruction"]/(batch_idx+1):.4f}',
                'kl': f'{total_losses["kl_divergence"]/(batch_idx+1):.4f}'
            })
            
            self.global_step += 1
        
        # Calculate epoch metrics
        epoch_metrics = {}
        for loss_name, total_loss in total_losses.items():
            epoch_metrics[f'{loss_name}_loss'] = total_loss / num_batches
        
        # CAN ID classification metrics
        canid_accuracy = accuracy_score(all_canid_targets, all_canid_predictions)
        canid_f1 = f1_score(all_canid_targets, all_canid_predictions, average='weighted', zero_division=0)
        
        epoch_metrics.update({
            'canid_accuracy': canid_accuracy,
            'canid_f1': canid_f1,
            'epoch': epoch,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })
        
        # Log metrics
        self.log_metrics({f'train_{k}': v for k, v in epoch_metrics.items()})
        self.train_metrics.append(epoch_metrics)
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl_divergence': 0.0,
            'canid_classification': 0.0,
            'neighborhood': 0.0
        }
        
        all_canid_predictions = []
        all_canid_targets = []
        num_batches = len(val_loader)
        
        # Setup progress bar
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                batch = self.fabric.to_device(batch)
                
                # Forward pass
                with self.fabric.autocast():
                    cont_out, canid_logits, neighbor_logits, z, kl_loss = self.model(
                        batch.x, batch.edge_index, batch.batch
                    )
                    
                    # Compute individual losses
                    losses = self.compute_vgae_losses(
                        batch, cont_out, canid_logits, neighbor_logits, kl_loss
                    )
                    
                    # Weighted total loss
                    total_loss = (
                        self.loss_weights['reconstruction'] * losses['reconstruction'] +
                        self.loss_weights['kl_divergence'] * losses['kl_divergence'] +
                        self.loss_weights['canid_classification'] * losses['canid_classification'] +
                        self.loss_weights['neighborhood'] * losses['neighborhood']
                    )
                
                # Collect metrics
                total_losses['total'] += total_loss.item()
                for loss_name, loss_value in losses.items():
                    total_losses[loss_name] += loss_value.item()
                
                # CAN ID classification metrics
                canid_predictions = torch.argmax(canid_logits, dim=1)
                canid_targets = batch.x[:, 0].long()
                all_canid_predictions.extend(canid_predictions.cpu().numpy())
                all_canid_targets.extend(canid_targets.cpu().numpy())
                
                # Update progress bar
                current_loss = total_losses['total'] / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Calculate epoch metrics
        epoch_metrics = {}
        for loss_name, total_loss in total_losses.items():
            epoch_metrics[f'{loss_name}_loss'] = total_loss / num_batches
        
        # CAN ID classification metrics
        canid_accuracy = accuracy_score(all_canid_targets, all_canid_predictions)
        canid_f1 = f1_score(all_canid_targets, all_canid_predictions, average='weighted', zero_division=0)
        
        epoch_metrics.update({
            'canid_accuracy': canid_accuracy,
            'canid_f1': canid_f1,
            'epoch': epoch
        })
        
        # Log metrics
        self.log_metrics({f'val_{k}': v for k, v in epoch_metrics.items()})
        self.val_metrics.append(epoch_metrics)
        
        return epoch_metrics
    
    def fit(self, 
            train_loader, 
            val_loader=None,
            epochs: int = None,
            save_best: bool = True,
            checkpoint_dir: str = "checkpoints") -> Dict[str, List[Dict[str, float]]]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            save_best: Whether to save the best model
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.training_config.get('epochs', 100)
        
        # Setup data loaders with Fabric
        train_loader = self.setup_dataloader(train_loader)
        if val_loader:
            val_loader = self.setup_dataloader(val_loader)
        
        # Determine optimal batch size if needed
        if self.use_dynamic_batching and self.optimal_batch_size is None:
            sample_batch = next(iter(train_loader))
            self.optimal_batch_size = self.determine_optimal_batch_size(sample_batch)
        
        # Training configuration
        accumulation_steps = self.training_config.get('accumulation_steps', 1)
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting VGAE training for {epochs} epochs")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Loss weights: {self.loss_weights}")
        
        best_val_loss = float('inf')
        best_canid_f1 = 0.0
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch, accumulation_steps)
            
            # Validation phase
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate_epoch(val_loader, epoch)
                
                # Check for best model
                val_loss = val_metrics['total_loss']
                canid_f1 = val_metrics['canid_f1']
                
                if save_best and (val_loss < best_val_loss or canid_f1 > best_canid_f1):
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.best_metric = val_loss
                    if canid_f1 > best_canid_f1:
                        best_canid_f1 = canid_f1
                    
                    # Save best model
                    best_model_path = checkpoint_dir / "best_vgae_model.ckpt"
                    self.save_checkpoint(
                        self.model, 
                        self.optimizer,
                        self.scheduler,
                        str(best_model_path),
                        extra_state={'best_val_loss': best_val_loss, 'best_canid_f1': best_canid_f1}
                    )
            
            # Periodic checkpointing
            checkpoint_interval = self.training_config.get('checkpoint_interval', 10)
            if epoch % checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"vgae_epoch_{epoch}.ckpt"
                self.save_checkpoint(self.model, self.optimizer, self.scheduler, str(checkpoint_path))
            
            # Log epoch summary
            log_msg = (f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_metrics['total_loss']:.4f}, "
                      f"Train CAN-ID F1: {train_metrics['canid_f1']:.4f}")
            if val_metrics:
                log_msg += (f", Val Loss: {val_metrics['total_loss']:.4f}, "
                           f"Val CAN-ID F1: {val_metrics['canid_f1']:.4f}")
            logger.info(log_msg)
        
        # Save final model
        final_model_path = checkpoint_dir / "final_vgae_model.ckpt"
        self.save_checkpoint(self.model, self.optimizer, self.scheduler, str(final_model_path))
        
        training_history = {
            'train': self.train_metrics,
            'val': self.val_metrics
        }
        
        logger.info("VGAE training completed successfully")
        return training_history
    
    def encode(self, data_loader):
        """
        Encode data using the trained VGAE encoder.
        
        Args:
            data_loader: Data loader for encoding
            
        Returns:
            Tuple of (latent_embeddings, kl_losses)
        """
        self.model.eval()
        embeddings = []
        kl_losses = []
        
        data_loader = self.setup_dataloader(data_loader)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Encoding"):
                batch = self.fabric.to_device(batch)
                
                with self.fabric.autocast():
                    z, kl_loss = self.model.encode(batch.x, batch.edge_index)
                
                embeddings.append(z.cpu().numpy())
                kl_losses.append(kl_loss.item())
        
        return np.vstack(embeddings), np.array(kl_losses)
    
    def reconstruct(self, data_loader):
        """
        Reconstruct data using the trained VGAE.
        
        Args:
            data_loader: Data loader for reconstruction
            
        Returns:
            Dictionary with reconstructed outputs
        """
        self.model.eval()
        reconstructions = {
            'continuous': [],
            'canid_logits': [],
            'neighbor_logits': []
        }
        
        data_loader = self.setup_dataloader(data_loader)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Reconstructing"):
                batch = self.fabric.to_device(batch)
                
                with self.fabric.autocast():
                    cont_out, canid_logits, neighbor_logits, z, kl_loss = self.model(
                        batch.x, batch.edge_index, batch.batch
                    )
                
                reconstructions['continuous'].append(cont_out.cpu().numpy())
                reconstructions['canid_logits'].append(canid_logits.cpu().numpy())
                reconstructions['neighbor_logits'].append(neighbor_logits.cpu().numpy())
        
        # Concatenate all reconstructions
        for key in reconstructions:
            reconstructions[key] = np.vstack(reconstructions[key])
        
        return reconstructions


class FabricVGAEKnowledgeDistillationTrainer(FabricVGAETrainer):
    """
    Extended VGAE trainer with knowledge distillation support.
    """
    
    def __init__(self, 
                 student_config: Dict[str, Any],
                 teacher_model_path: str,
                 training_config: Dict[str, Any],
                 distillation_config: Dict[str, Any],
                 fabric_config: Dict[str, Any] = None):
        """
        Initialize VGAE knowledge distillation trainer.
        
        Args:
            student_config: Configuration for student VGAE model
            teacher_model_path: Path to trained teacher model
            training_config: Training hyperparameters
            distillation_config: Knowledge distillation configuration
            fabric_config: Fabric-specific configuration
        """
        # Initialize with student configuration
        super().__init__(student_config, training_config, fabric_config)
        
        self.distillation_config = distillation_config
        
        # Load teacher model
        self.teacher_model = self._load_teacher_model(teacher_model_path)
        self.teacher_model.eval()
        
        # Update loss weights to include distillation
        self.loss_weights.update({
            'latent_distillation': distillation_config.get('latent_weight', 0.5),
            'output_distillation': distillation_config.get('output_weight', 0.3)
        })
    
    def _load_teacher_model(self, model_path: str):
        """Load pre-trained teacher VGAE model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Determine teacher model configuration
        if 'model_config' in checkpoint:
            teacher_config = checkpoint['model_config']
        else:
            # Fallback: assume larger architecture
            teacher_config = self.model_config.copy()
            teacher_config.update({
                'hidden_dim': self.model_config['hidden_dim'] * 2,
                'latent_dim': self.model_config['latent_dim'] * 2,
                'num_encoder_layers': self.model_config['num_encoder_layers'] + 1,
                'num_decoder_layers': self.model_config['num_decoder_layers'] + 1
            })
        
        teacher_model = GraphAutoencoderNeighborhood(**teacher_config)
        
        # Load state dict
        if 'model' in checkpoint:
            teacher_model.load_state_dict(checkpoint['model'])
        else:
            teacher_model.load_state_dict(checkpoint)
        
        logger.info(f"Teacher VGAE model loaded from {model_path}")
        return teacher_model
    
    def compute_distillation_losses(self, 
                                   student_z, 
                                   teacher_z,
                                   student_outputs,
                                   teacher_outputs) -> Dict[str, torch.Tensor]:
        """
        Compute knowledge distillation losses.
        
        Args:
            student_z: Student latent embeddings
            teacher_z: Teacher latent embeddings  
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            
        Returns:
            Dictionary of distillation losses
        """
        temperature = self.distillation_config.get('temperature', 3.0)
        
        # Latent space distillation (MSE between embeddings)
        if student_z.size() != teacher_z.size():
            # Project to common dimension if needed
            if student_z.size(-1) < teacher_z.size(-1):
                teacher_z = teacher_z[:, :student_z.size(-1)]
            else:
                # Pad student if it's larger (unusual case)
                padding = teacher_z.size(-1) - student_z.size(-1)
                student_z = F.pad(student_z, (0, padding))
        
        latent_distill_loss = F.mse_loss(student_z, teacher_z.detach())
        
        # Output distillation for CAN ID predictions
        student_canid, teacher_canid = student_outputs[1], teacher_outputs[1]
        output_distill_loss = F.kl_div(
            F.log_softmax(student_canid / temperature, dim=1),
            F.softmax(teacher_canid.detach() / temperature, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return {
            'latent_distillation': latent_distill_loss,
            'output_distillation': output_distill_loss
        }
    
    def train_epoch(self, train_loader, epoch: int, accumulation_steps: int = 1) -> Dict[str, float]:
        """
        Train epoch with knowledge distillation.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            accumulation_steps: Gradient accumulation steps
            
        Returns:
            Training metrics dictionary
        """
        self.model.train()
        self.teacher_model.eval()
        
        total_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl_divergence': 0.0,
            'canid_classification': 0.0,
            'neighborhood': 0.0,
            'latent_distillation': 0.0,
            'output_distillation': 0.0
        }
        
        all_canid_predictions = []
        all_canid_targets = []
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} VGAE Distillation Training")
        
        for batch_idx, batch in enumerate(pbar):
            batch = self.fabric.to_device(batch)
            
            # Forward pass with autocast
            with self.fabric.autocast():
                # Student predictions
                student_outputs = self.model(batch.x, batch.edge_index, batch.batch)
                cont_out, canid_logits, neighbor_logits, student_z, kl_loss = student_outputs
                
                # Teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(batch.x, batch.edge_index, batch.batch)
                    teacher_z = teacher_outputs[3]  # Extract teacher latent embeddings
                
                # Standard VGAE losses
                vgae_losses = self.compute_vgae_losses(
                    batch, cont_out, canid_logits, neighbor_logits, kl_loss
                )
                
                # Distillation losses
                distill_losses = self.compute_distillation_losses(
                    student_z, teacher_z, student_outputs, teacher_outputs
                )
                
                # Combined losses
                all_losses = {**vgae_losses, **distill_losses}
                
                # Weighted total loss
                total_batch_loss = sum(
                    self.loss_weights[loss_name] * loss_value 
                    for loss_name, loss_value in all_losses.items()
                    if loss_name in self.loss_weights
                ) / accumulation_steps
            
            # Backward pass
            self.fabric.backward(total_batch_loss)
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == num_batches - 1:
                if self.training_config.get('gradient_clip_val'):
                    self.fabric.clip_gradients(
                        self.model, self.optimizer,
                        max_norm=self.training_config['gradient_clip_val']
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.scheduler:
                    self.scheduler.step()
            
            # Collect metrics
            total_losses['total'] += total_batch_loss.item() * accumulation_steps
            for loss_name, loss_value in all_losses.items():
                if loss_name in total_losses:
                    total_losses[loss_name] += loss_value.item()
            
            # CAN ID classification metrics
            canid_predictions = torch.argmax(canid_logits, dim=1)
            canid_targets = batch.x[:, 0].long()
            all_canid_predictions.extend(canid_predictions.cpu().numpy())
            all_canid_targets.extend(canid_targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'total_loss': f'{total_losses["total"]/(batch_idx+1):.4f}',
                'recon': f'{total_losses["reconstruction"]/(batch_idx+1):.4f}',
                'distill': f'{total_losses["latent_distillation"]/(batch_idx+1):.4f}'
            })
            
            self.global_step += 1
        
        # Calculate metrics
        epoch_metrics = {}
        for loss_name, total_loss in total_losses.items():
            epoch_metrics[f'{loss_name}_loss'] = total_loss / num_batches
        
        # CAN ID classification metrics
        canid_accuracy = accuracy_score(all_canid_targets, all_canid_predictions)
        canid_f1 = f1_score(all_canid_targets, all_canid_predictions, average='weighted', zero_division=0)
        
        epoch_metrics.update({
            'canid_accuracy': canid_accuracy,
            'canid_f1': canid_f1,
            'epoch': epoch,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })
        
        # Log metrics
        self.log_metrics({f'train_{k}': v for k, v in epoch_metrics.items()})
        self.train_metrics.append(epoch_metrics)
        
        return epoch_metrics