"""
PyTorch Lightning Fabric trainer for GAT (Graph Attention Network) models.

This trainer provides:
- Efficient GAT model training with Fabric
- Dynamic batch sizing
- Mixed precision training
- Advanced logging and checkpointing
- Memory optimization strategies
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
from src.models.models import GATWithJK
from src.utils.losses import distillation_loss_fn, FocalLoss

logger = logging.getLogger(__name__)


class FabricGATTrainer(FabricTrainerBase):
    """
    PyTorch Lightning Fabric trainer for GAT models with advanced optimizations.
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 fabric_config: Dict[str, Any] = None,
                 use_dynamic_batching: bool = True):
        """
        Initialize Fabric GAT trainer.
        
        Args:
            model_config: Configuration for GAT model
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
        
        # Loss function setup
        self.loss_fn = self._create_loss_function()
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        # Dynamic batch sizing
        self.optimal_batch_size = None
        if self.use_dynamic_batching:
            logger.info("Dynamic batch sizing will be determined during training setup")
    
    def _create_model(self) -> GATWithJK:
        """Create GAT model from configuration."""
        return GATWithJK(**self.model_config)
    
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
    
    def _create_loss_function(self):
        """Create loss function based on configuration."""
        loss_config = self.training_config.get('loss', {})
        loss_type = loss_config.get('type', 'bce').lower()
        
        if loss_type == 'focal':
            alpha = loss_config.get('alpha', 1.0)
            gamma = loss_config.get('gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == 'bce':
            pos_weight = loss_config.get('pos_weight')
            if pos_weight:
                pos_weight = torch.tensor(pos_weight)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.BCEWithLogitsLoss()
    
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
                max_batch_size=self.training_config.get('max_batch_size', 8192)
            )
            
            optimal_size = batch_sizer.estimate_optimal_batch_size()
            logger.info(f"Determined optimal batch size: {optimal_size}")
            return optimal_size
            
        except Exception as e:
            logger.warning(f"Dynamic batch sizing failed: {e}")
            fallback_size = self.training_config.get('batch_size', 32)
            logger.info(f"Using fallback batch size: {fallback_size}")
            return fallback_size
    
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
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = len(train_loader)
        
        # Setup progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device (handled by Fabric)
            batch = self.fabric.to_device(batch)
            targets = batch.y.float()
            
            # Forward pass with autocast
            with self.fabric.autocast():
                outputs = self.model(batch).squeeze()
                loss = self.loss_fn(outputs, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass
            self.fabric.backward(loss)
            
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
            total_loss += loss.item() * accumulation_steps
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
            
            self.global_step += 1
        
        # Calculate epoch metrics
        epoch_metrics = self._calculate_metrics(all_predictions, all_targets, total_loss / num_batches)
        epoch_metrics['epoch'] = epoch
        epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
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
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        num_batches = len(val_loader)
        
        # Setup progress bar
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                batch = self.fabric.to_device(batch)
                targets = batch.y.float()
                
                # Forward pass
                with self.fabric.autocast():
                    outputs = self.model(batch).squeeze()
                    loss = self.loss_fn(outputs, targets)
                
                # Collect metrics
                total_loss += loss.item()
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).long()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                current_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Calculate epoch metrics
        epoch_metrics = self._calculate_metrics(
            all_predictions, all_targets, total_loss / num_batches, all_probabilities
        )
        epoch_metrics['epoch'] = epoch
        
        # Log metrics
        self.log_metrics({f'val_{k}': v for k, v in epoch_metrics.items()})
        self.val_metrics.append(epoch_metrics)
        
        return epoch_metrics
    
    def _calculate_metrics(self, 
                          predictions: List[int], 
                          targets: List[int], 
                          loss: float,
                          probabilities: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics.
        
        Args:
            predictions: Predicted labels
            targets: True labels
            loss: Average loss
            probabilities: Predicted probabilities (for AUC calculation)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'loss': loss,
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, zero_division=0),
            'recall': recall_score(targets, predictions, zero_division=0),
            'f1': f1_score(targets, predictions, zero_division=0)
        }
        
        # Add AUC if probabilities are available
        if probabilities is not None:
            try:
                metrics['auc'] = roc_auc_score(targets, probabilities)
            except ValueError:
                # Handle case where only one class is present
                metrics['auc'] = 0.0
        
        return metrics
    
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
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch, accumulation_steps)
            
            # Validation phase
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate_epoch(val_loader, epoch)
                
                # Check for best model
                val_loss = val_metrics['loss']
                val_f1 = val_metrics['f1']
                
                if save_best and (val_loss < best_val_loss or val_f1 > best_val_f1):
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.best_metric = val_loss
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                    
                    # Save best model
                    best_model_path = checkpoint_dir / "best_model.ckpt"
                    self.save_checkpoint(
                        self.model, 
                        self.optimizer,
                        self.scheduler,
                        str(best_model_path),
                        extra_state={'best_val_loss': best_val_loss, 'best_val_f1': best_val_f1}
                    )
            
            # Periodic checkpointing
            checkpoint_interval = self.training_config.get('checkpoint_interval', 10)
            if epoch % checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"epoch_{epoch}.ckpt"
                self.save_checkpoint(self.model, self.optimizer, self.scheduler, str(checkpoint_path))
            
            # Log epoch summary
            log_msg = f"Epoch {epoch}/{epochs} - Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f}"
            if val_metrics:
                log_msg += f", Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}"
            logger.info(log_msg)
        
        # Save final model
        final_model_path = checkpoint_dir / "final_model.ckpt"
        self.save_checkpoint(self.model, self.optimizer, self.scheduler, str(final_model_path))
        
        training_history = {
            'train': self.train_metrics,
            'val': self.val_metrics
        }
        
        logger.info("Training completed successfully")
        return training_history
    
    def predict(self, data_loader, return_probabilities: bool = False):
        """
        Make predictions on a dataset.
        
        Args:
            data_loader: Data loader for predictions
            return_probabilities: Whether to return probabilities instead of labels
            
        Returns:
            Predictions array
        """
        self.model.eval()
        predictions = []
        
        data_loader = self.setup_dataloader(data_loader)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                batch = self.fabric.to_device(batch)
                
                with self.fabric.autocast():
                    outputs = self.model(batch).squeeze()
                    probabilities = torch.sigmoid(outputs)
                
                if return_probabilities:
                    predictions.extend(probabilities.cpu().numpy())
                else:
                    labels = (probabilities > 0.5).long()
                    predictions.extend(labels.cpu().numpy())
        
        return np.array(predictions)


class FabricGATKnowledgeDistillationTrainer(FabricGATTrainer):
    """
    Extended GAT trainer with knowledge distillation support.
    """
    
    def __init__(self, 
                 student_config: Dict[str, Any],
                 teacher_model_path: str,
                 training_config: Dict[str, Any],
                 distillation_config: Dict[str, Any],
                 fabric_config: Dict[str, Any] = None):
        """
        Initialize knowledge distillation trainer.
        
        Args:
            student_config: Configuration for student GAT model
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
        
        # Update loss function for distillation
        self.distillation_loss_fn = distillation_loss_fn
    
    def _load_teacher_model(self, model_path: str):
        """Load pre-trained teacher model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Determine teacher model configuration from checkpoint
        if 'model_config' in checkpoint:
            teacher_config = checkpoint['model_config']
        else:
            # Fallback: assume same architecture but larger
            teacher_config = self.model_config.copy()
            teacher_config.update({
                'hidden_channels': self.model_config['hidden_channels'] * 2,
                'num_layers': self.model_config['num_layers'] + 2,
                'heads': self.model_config['heads'] * 2
            })
        
        teacher_model = GATWithJK(**teacher_config)
        
        # Load state dict
        if 'model' in checkpoint:
            teacher_model.load_state_dict(checkpoint['model'])
        else:
            teacher_model.load_state_dict(checkpoint)
        
        logger.info(f"Teacher model loaded from {model_path}")
        return teacher_model
    
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
        
        total_loss = 0.0
        total_student_loss = 0.0
        total_distill_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = len(train_loader)
        
        # Distillation parameters
        alpha = self.distillation_config.get('alpha', 0.7)
        temperature = self.distillation_config.get('temperature', 3.0)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Distillation Training")
        
        for batch_idx, batch in enumerate(pbar):
            batch = self.fabric.to_device(batch)
            targets = batch.y.float()
            
            # Forward pass with autocast
            with self.fabric.autocast():
                # Student predictions
                student_outputs = self.model(batch).squeeze()
                
                # Teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(batch).squeeze()
                
                # Calculate losses
                student_loss = self.loss_fn(student_outputs, targets)
                distill_loss = self.distillation_loss_fn(
                    student_outputs, teacher_outputs, temperature
                )
                
                # Combined loss
                total_batch_loss = (
                    alpha * distill_loss + (1 - alpha) * student_loss
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
            total_loss += total_batch_loss.item() * accumulation_steps
            total_student_loss += student_loss.item()
            total_distill_loss += distill_loss.item()
            
            predictions = (torch.sigmoid(student_outputs) > 0.5).long()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'total_loss': f'{total_loss/(batch_idx+1):.4f}',
                'student_loss': f'{total_student_loss/(batch_idx+1):.4f}',
                'distill_loss': f'{total_distill_loss/(batch_idx+1):.4f}'
            })
            
            self.global_step += 1
        
        # Calculate metrics
        epoch_metrics = self._calculate_metrics(all_predictions, all_targets, total_loss / num_batches)
        epoch_metrics.update({
            'student_loss': total_student_loss / num_batches,
            'distillation_loss': total_distill_loss / num_batches,
            'epoch': epoch,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        })
        
        # Log metrics
        self.log_metrics({f'train_{k}': v for k, v in epoch_metrics.items()})
        self.train_metrics.append(epoch_metrics)
        
        return epoch_metrics