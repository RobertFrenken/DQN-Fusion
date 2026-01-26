"""  
Unified Lightning Modules for CAN-Graph Training

Consolidates all PyTorch Lightning modules into a single coherent file.

Components:
- BaseKDGATModule: Base class with common functionality
- VAELightningModule: VGAE training (unsupervised)
- GATLightningModule: Graph Attention Network (classification)
- DQNLightningModule: DQN training for fusion
- FusionLightningModule: DQN-based fusion with prediction caching

Replaces:
- src/training/can_graph_module.py (310 lines)
- src/training/lightning_modules_OLD.py (404 lines)
- src/training/fusion_lightning.py (295 lines)
"""

import logging
from typing import Any, Optional, Dict, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as pl
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.models.models import GATWithJK, create_dqn_teacher, create_dqn_student
from src.models.vgae import GraphAutoencoderNeighborhood
from src.models.dqn import EnhancedDQNFusionAgent
from src.training.knowledge_distillation import KDHelper, cleanup_memory

logger = logging.getLogger(__name__)


# ============================================================================
# BASE LIGHTNING MODULE
# ============================================================================

class BaseKDGATModule(pl.LightningModule):
    """
    Base Lightning module for KD-GAT experiments.
    
    Provides:
    - Common optimizer configuration
    - Metrics tracking
    - Checkpoint management
    - Memory management (cleanup between epochs)
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        train_loader: Any = None,
        val_loader: Any = None,
    ):
        super().__init__()
        
        self.cfg = cfg
        self.train_loader_ref = train_loader
        self.val_loader_ref = val_loader
        
        # Save batch size for backward compatibility
        self.batch_size = cfg.training.batch_size if hasattr(cfg.training, 'batch_size') else 32
        
        # Save hyperparameters for checkpointing
        # Convert to dict to avoid OmegaConf Union serialization issues
        from dataclasses import asdict
        try:
            cfg_dict = asdict(cfg)
            self.save_hyperparameters({'cfg': cfg_dict})
        except Exception:
            # If conversion fails, save minimal info
            logger.warning("Could not serialize full config, saving minimal hyperparameters")
            self.save_hyperparameters({
                'model_type': getattr(cfg.model, 'type', 'unknown'),
                'dataset': cfg.dataset.name,
                'batch_size': self.batch_size
            })
        
        # Metrics tracking (limited size to prevent memory leaks)
        self._max_loss_history = 1000  # Keep last N losses only
        self.train_losses = []
        self.val_losses = []
    
    def on_train_epoch_end(self):
        """Clean up memory at end of training epoch."""
        # Trim loss history to prevent unbounded growth
        if len(self.train_losses) > self._max_loss_history:
            self.train_losses = self.train_losses[-self._max_loss_history:]
        
        # Release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_validation_epoch_end(self):
        """Clean up memory at end of validation epoch."""
        # Trim loss history to prevent unbounded growth
        if len(self.val_losses) > self._max_loss_history:
            self.val_losses = self.val_losses[-self._max_loss_history:]
        
        # Release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        
        # Select optimizer
        optimizer_name = self.cfg.training.optimizer.name.lower()
        lr = self.cfg.training.learning_rate
        wd = self.cfg.training.weight_decay
        
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=lr, 
                weight_decay=wd,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Configure scheduler if specified
        if not self.cfg.training.scheduler.use_scheduler:
            return optimizer
        
        scheduler_name = self.cfg.training.scheduler.scheduler_type.lower()
        
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.training.max_epochs,
            )
        elif scheduler_name == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.cfg.training.max_epochs,
            )
        elif scheduler_name == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95,
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    def on_train_start(self):
        """Called when training starts."""
        logger.info("🚀 Starting training run")
    
    def on_train_end(self):
        """Called when training ends."""
        logger.info("✅ Training complete")


# ============================================================================
# VGAE LIGHTNING MODULE (Unsupervised)
# ============================================================================

class VAELightningModule(BaseKDGATModule):
    """
    Lightning module for Variational Graph AutoEncoder (VGAE).

    Used for unsupervised pretraining and anomaly detection.
    Reconstructs graph structure and node features.

    Supports knowledge distillation when cfg.training.use_knowledge_distillation=True.
    KD signals: latent space + reconstruction (dual-signal approach).
    """

    def __init__(
        self,
        cfg: DictConfig,

        train_loader: Any = None,
        val_loader: Any = None,
        num_ids: int = 1000
    ):
        super().__init__(cfg, train_loader, val_loader)

        self.num_ids = num_ids

        # Build VGAE model
        self.model = self._build_vgae()

        # Loss weights
        self.reconstruction_loss_fn = nn.MSELoss()
        self.kl_weight = getattr(cfg.training, 'kl_weight', 0.01)

        # Initialize KD helper (no-op if disabled)
        self.kd_helper = KDHelper(cfg, self.model, model_type="vgae")

        # Track KD losses for logging
        self.kd_latent_losses = []
        self.kd_recon_losses = []

        logger.info(
            f"✅ Initialized VGAE with {sum(p.numel() for p in self.model.parameters())} parameters"
        )
        if self.kd_helper.enabled:
            logger.info(f"   Knowledge Distillation: ENABLED (alpha={self.kd_helper.alpha})")
    
    def _build_vgae(self) -> nn.Module:
        """Build VGAE model from config."""
        # Build params from config and prefer explicit progressive `hidden_dims` if present
        if hasattr(self.cfg.model, 'hidden_dims') and self.cfg.model.hidden_dims:
            hidden_dims = list(self.cfg.model.hidden_dims)
        elif hasattr(self.cfg.model, 'encoder_dims') and self.cfg.model.encoder_dims:
            hidden_dims = list(self.cfg.model.encoder_dims)
        else:
            hidden_dims = None

        # Get latent_dim - ensure it's explicitly from config, not falling back incorrectly
        latent_dim = getattr(self.cfg.model, 'latent_dim', None)
        if latent_dim is None:
            latent_dim = hidden_dims[-1] if hidden_dims else 32
            logger.warning(f"latent_dim not in config, using fallback: {latent_dim}")

        # Log the configuration for debugging
        logger.info(f"🔧 VGAE Build Config:")
        logger.info(f"   hidden_dims: {hidden_dims}")
        logger.info(f"   latent_dim: {latent_dim}")
        logger.info(f"   input_dim: {self.cfg.model.input_dim}")
        logger.info(f"   num_ids: {self.num_ids}")

        vgae_params = {
            'num_ids': self.num_ids,
            'in_channels': self.cfg.model.input_dim,
            'hidden_dims': hidden_dims,
            'latent_dim': latent_dim,
            'encoder_heads': getattr(self.cfg.model, 'attention_heads', 4),
            'decoder_heads': getattr(self.cfg.model, 'attention_heads', 4),
            'embedding_dim': getattr(self.cfg.model, 'embedding_dim', 32),
            'dropout': getattr(self.cfg.model, 'dropout', 0.15),
            'batch_norm': getattr(self.cfg.model, 'batch_norm', True)
        }

        logger.info(f"   embedding_dim: {vgae_params['embedding_dim']}")
        logger.info(f"   encoder_heads: {vgae_params['encoder_heads']}")

        # Add gradient checkpointing if enabled in config
        if hasattr(self.cfg.training, 'memory_optimization'):
            use_checkpointing = getattr(self.cfg.training.memory_optimization, 'gradient_checkpointing', False)
            vgae_params['use_checkpointing'] = use_checkpointing
            if use_checkpointing:
                logger.info("✅ Gradient checkpointing ENABLED for VGAE (memory-efficient training)")
            else:
                logger.warning("⚠️  Gradient checkpointing DISABLED - may cause OOM on large datasets")

        return GraphAutoencoderNeighborhood(**vgae_params)
    
    def forward(self, batch):
        """Forward pass through VGAE."""
        x = batch.x
        edge_index = batch.edge_index
        b = batch.batch
        return self.model(x, edge_index, b)
    
    def training_step(self, batch, batch_idx):
        """Training step with reconstruction + KL loss, with optional KD."""
        # Forward pass
        cont_out, canid_logits, neighbor_logits, z, kl_loss = self.forward(batch)

        # Reconstruction losses
        continuous_features = batch.x[:, 1:]
        reconstruction_loss = self.reconstruction_loss_fn(cont_out, continuous_features)

        canid_targets = batch.x[:, 0].long()
        canid_loss = F.cross_entropy(canid_logits, canid_targets)

        # Task loss (base loss without KD)
        task_loss = reconstruction_loss + 0.1 * canid_loss + self.kl_weight * kl_loss

        # Knowledge distillation (if enabled)
        if self.kd_helper.enabled:
            # Get teacher outputs (latent z and reconstructions)
            teacher_outputs = self.kd_helper.get_teacher_outputs(batch)

            # Package student reconstruction outputs
            student_recon = {
                'cont_out': cont_out,
                'canid_logits': canid_logits,
                'neighbor_logits': neighbor_logits
            }

            # Compute VGAE-specific KD loss (latent + reconstruction)
            kd_loss = self.kd_helper.compute_vgae_kd_loss(
                student_z=z,
                student_recon=student_recon,
                teacher_outputs=teacher_outputs
            )

            # Combine task loss and KD loss
            total_loss = self.kd_helper.combine_losses(task_loss, kd_loss)

            # Log KD-specific metrics
            self.log('kd_loss', kd_loss, on_epoch=True)
        else:
            total_loss = task_loss

        self.train_losses.append(total_loss.item())

        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', reconstruction_loss, on_epoch=True)
        self.log('train_canid_loss', canid_loss, on_epoch=True)
        self.log('train_kl_loss', kl_loss, on_epoch=True)

        # Memory cleanup every 20 batches
        if batch_idx % 20 == 0:
            cleanup_memory()

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Forward pass
        cont_out, canid_logits, neighbor_logits, z, kl_loss = self.forward(batch)
        
        # Reconstruction losses
        continuous_features = batch.x[:, 1:]
        reconstruction_loss = self.reconstruction_loss_fn(cont_out, continuous_features)
        
        canid_targets = batch.x[:, 0].long()
        canid_loss = F.cross_entropy(canid_logits, canid_targets)
        
        # Total loss
        total_loss = reconstruction_loss + 0.1 * canid_loss + self.kl_weight * kl_loss
        
        self.val_losses.append(total_loss.item())
        
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', reconstruction_loss, on_epoch=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        cont_out, canid_logits, neighbor_logits, z, kl_loss = self.forward(batch)

        continuous_features = batch.x[:, 1:]
        reconstruction_loss = self.reconstruction_loss_fn(cont_out, continuous_features)

        self.log('test_recon_loss', reconstruction_loss)
        return {'test_loss': reconstruction_loss}

    def configure_optimizers(self):
        """Configure optimizer, including projection layer if KD enabled."""
        # Get base optimizer config from parent
        optimizer_config = super().configure_optimizers()

        # Add projection layer parameters if KD enabled
        if self.kd_helper.enabled and self.kd_helper.projection_layer is not None:
            # Get the optimizer from the config
            if isinstance(optimizer_config, dict):
                optimizer = optimizer_config['optimizer']
            else:
                optimizer = optimizer_config

            # Add projection layer params with separate learning rate
            proj_params = list(self.kd_helper.projection_layer.parameters())
            optimizer.add_param_group({
                'params': proj_params,
                'lr': 1e-3,  # Separate LR for projection layer
                'weight_decay': 1e-5
            })
            logger.info(f"   Added projection layer params to optimizer (lr=1e-3)")

        return optimizer_config


# ============================================================================
# GAT LIGHTNING MODULE (Classifier)
# ============================================================================

class GATLightningModule(BaseKDGATModule):
    """
    Lightning module for Graph Attention Network (GAT).

    Used for supervised classification of normal vs attack traffic.

    Supports knowledge distillation when cfg.training.use_knowledge_distillation=True.
    KD signal: soft label distillation with temperature scaling.
    """

    def __init__(
        self,
        cfg: DictConfig,
        train_loader: Any = None,
        val_loader: Any = None,
        num_ids: int = 1000
    ):
        super().__init__(cfg, train_loader, val_loader)

        self.num_ids = num_ids

        # Build GAT model
        self.model = self._build_gat()

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize KD helper (no-op if disabled)
        self.kd_helper = KDHelper(cfg, self.model, model_type="gat")

        logger.info(
            f"✅ Initialized GAT with {sum(p.numel() for p in self.model.parameters())} parameters"
        )
        if self.kd_helper.enabled:
            logger.info(f"   Knowledge Distillation: ENABLED (alpha={self.kd_helper.alpha})")
    
    def _build_gat(self) -> nn.Module:
        """Build GAT model from config."""
        if hasattr(self.cfg.model, 'gat'):
            gat_params = dict(self.cfg.model.gat)
        else:
            gat_params = {
                'input_dim': self.cfg.model.input_dim,
                'hidden_channels': self.cfg.model.hidden_channels,
                'output_dim': self.cfg.model.output_dim,
                'num_layers': self.cfg.model.num_layers,
                'heads': self.cfg.model.heads,
                'dropout': self.cfg.model.dropout,
                'num_fc_layers': self.cfg.model.num_fc_layers,
                'embedding_dim': self.cfg.model.embedding_dim,
            }

        # Normalize parameter names
        gat_params['in_channels'] = gat_params.pop('input_dim')
        gat_params['out_channels'] = gat_params.pop('output_dim')

        # Remove unused params
        for unused_param in ['use_jumping_knowledge', 'jk_mode', 'use_residual',
                            'use_batch_norm', 'activation']:
            gat_params.pop(unused_param, None)

        gat_params['num_ids'] = self.num_ids

        # Add gradient checkpointing if enabled in config
        if hasattr(self.cfg.training, 'memory_optimization'):
            use_checkpointing = getattr(self.cfg.training.memory_optimization, 'gradient_checkpointing', False)
            gat_params['use_checkpointing'] = use_checkpointing
            if use_checkpointing:
                logger.info("✅ Gradient checkpointing ENABLED for GAT (memory-efficient training)")
            else:
                logger.warning("⚠️  Gradient checkpointing DISABLED - may cause OOM on large datasets")

        return GATWithJK(**gat_params)
    
    def forward(self, batch):
        """Forward pass through GAT."""
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        """Training step with classification loss, with optional KD."""
        logits = self.forward(batch)
        task_loss = self.loss_fn(logits, batch.y)

        # Knowledge distillation (if enabled)
        if self.kd_helper.enabled:
            # Get teacher outputs (soft labels)
            teacher_outputs = self.kd_helper.get_teacher_outputs(batch)

            # Compute GAT-specific KD loss (soft label distillation)
            kd_loss = self.kd_helper.compute_gat_kd_loss(
                student_logits=logits,
                teacher_logits=teacher_outputs['logits']
            )

            # Combine task loss and KD loss
            loss = self.kd_helper.combine_losses(task_loss, kd_loss)

            # Log KD-specific metrics
            self.log('kd_loss', kd_loss, on_epoch=True)
        else:
            loss = task_loss

        self.train_losses.append(loss.item())

        # Compute accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == batch.y).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True)

        # Memory cleanup every 20 batches
        if batch_idx % 20 == 0:
            cleanup_memory()

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch.y)
        
        self.val_losses.append(loss.item())
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == batch.y).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch.y)
        
        preds = logits.argmax(dim=1)
        acc = (preds == batch.y).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return {'test_loss': loss, 'test_acc': acc}


# ============================================================================
# DQN LIGHTNING MODULE (RL / Fusion)
# ============================================================================

class DQNLightningModule(BaseKDGATModule):
    """
    Lightning module for Deep Q-Network (DQN).
    
    Used for reinforcement learning-based fusion of model outputs.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        train_loader: Any = None,
        val_loader: Any = None,
    ):
        super().__init__(cfg, train_loader, val_loader)
        
        # Build DQN model
        self.model = self._build_dqn()
        
        # Target network for stable learning
        self.target_network = self._build_dqn()
        self.target_network.load_state_dict(self.model.state_dict())
        
        # Hyperparameters
        self.gamma = getattr(cfg.training, 'gamma', 0.99)
        self.target_update_freq = getattr(cfg.training, 'target_update_freq', 100)
        self.update_counter = 0
        
        logger.info(
            f"✅ Initialized DQN with {sum(p.numel() for p in self.model.parameters())} parameters"
        )
    
    def _build_dqn(self) -> nn.Module:
        """Build DQN model from config."""
        model_type = getattr(self.cfg.model, 'dqn_type', 'teacher')
        
        if model_type == 'teacher':
            return create_dqn_teacher()
        else:
            return create_dqn_student()
    
    def forward(self, x):
        """Forward pass through DQN."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step with Q-learning."""
        states, actions, rewards, next_states, dones = batch
        
        # Current Q-values
        q_values = self.forward(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        self.train_losses.append(loss.item())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.model.state_dict())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        states, actions, rewards, next_states, dones = batch
        
        with torch.no_grad():
            q_values = self.forward(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
            loss = F.smooth_l1_loss(q_values, target_q_values)
        
        self.val_losses.append(loss.item())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        states, actions, rewards, next_states, dones = batch
        
        with torch.no_grad():
            q_values = self.forward(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
            loss = F.smooth_l1_loss(q_values, target_q_values)
        
        self.log('test_loss', loss)
        return {'test_loss': loss}


# ============================================================================
# FUSION LIGHTNING MODULE (DQN-based Fusion with Prediction Caching)
# ============================================================================

class FusionPredictionCache(Dataset):
    """
    Pre-computed cache of VGAE and GAT predictions for efficient fusion training.
    
    Stores:
    - Anomaly scores from VGAE
    - Classification probabilities from GAT
    - Ground truth labels
    - Sample indices for tracking
    """
    
    def __init__(
        self, 
        anomaly_scores: np.ndarray, 
        gat_probs: np.ndarray, 
        labels: np.ndarray, 
        indices: np.ndarray = None
    ):
        """
        Initialize prediction cache.
        
        Args:
            anomaly_scores: [N] array of anomaly scores in [0, 1]
            gat_probs: [N] array of GAT probabilities in [0, 1]
            labels: [N] array of binary labels (0=normal, 1=attack)
            indices: [N] array of original sample indices (optional)
        """
        self.anomaly_scores = torch.tensor(anomaly_scores, dtype=torch.float32)
        self.gat_probs = torch.tensor(gat_probs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        if indices is None:
            self.indices = torch.arange(len(anomaly_scores))
        else:
            self.indices = torch.tensor(indices, dtype=torch.long)
        
        if not (len(self.anomaly_scores) == len(self.gat_probs) == len(self.labels)):
            raise ValueError("All inputs must have same length")
    
    def __len__(self):
        return len(self.anomaly_scores)
    
    def __getitem__(self, idx):
        return {
            'anomaly_score': self.anomaly_scores[idx],
            'gat_prob': self.gat_probs[idx],
            'label': self.labels[idx],
            'index': self.indices[idx]
        }


class FusionLightningModule(pl.LightningModule):
    """
    Lightning module for DQN-based fusion training.
    
    Training loop:
    1. Sample minibatch from cached predictions
    2. Use DQN to select fusion weights
    3. Compute fused predictions
    4. Calculate rewards based on correctness
    5. Update Q-network
    6. Periodically update target network
    """
    
    def __init__(self, fusion_config: Dict[str, Any], num_ids: int = None):
        """
        Initialize fusion module.
        
        Args:
            fusion_config: Hydra-Zen config with fusion parameters
            num_ids: Number of unique CAN IDs (unused but for compatibility)
        """
        super().__init__()
        # Save hyperparameters (avoid complex nested objects)
        self.save_hyperparameters({
            'alpha_steps': fusion_config.get('alpha_steps', 21),
            'fusion_lr': fusion_config.get('fusion_lr', 0.001),
            'gamma': fusion_config.get('gamma', 0.9)
        })
        
        self.fusion_config = fusion_config
        self.num_ids = num_ids
        
        # Initialize DQN fusion agent
        # Determine device - use cuda if available, else cpu
        import torch
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.fusion_agent = EnhancedDQNFusionAgent(
            alpha_steps=fusion_config.get('alpha_steps', 21),
            state_dim=2,  # anomaly_score, gat_prob
            lr=fusion_config.get('fusion_lr', 0.001),
            gamma=fusion_config.get('gamma', 0.9),
            epsilon=fusion_config.get('fusion_epsilon', 0.9),
            epsilon_decay=fusion_config.get('fusion_epsilon_decay', 0.995),
            min_epsilon=fusion_config.get('fusion_min_epsilon', 0.2),
            buffer_size=fusion_config.get('fusion_buffer_size', 100000),
            batch_size=fusion_config.get('fusion_batch_size', 256),
            device=device_str
        )
        
        # Training tracking
        self.episode_rewards = []
        self.episode_accuracies = []
        self.episode_q_values = []
        self.validation_accuracies = []

        # Disable Lightning's automatic optimization (agent handles it internally)
        self.automatic_optimization = False
        
        # Target network update counter
        self.target_update_counter = 0
        self.target_update_freq = fusion_config.get('target_update_freq', 100)
        
        # Batch tracking
        self.batch_count = 0
        
        # Memory optimization settings
        self.memory_optimization = fusion_config.get('memory_optimization', {})
        self.empty_cache_every_n_steps = self.memory_optimization.get('empty_cache_every_n_steps', 100)
    
    def configure_optimizers(self):
        """Configure optimizer for fusion agent Q-network."""
        return torch.optim.Adam(
            self.fusion_agent.q_network.parameters(),
            lr=self.fusion_config.get('fusion_lr', 0.001)
        )
    
    def training_step(self, batch, batch_idx):
        """
        Single training step on a minibatch of cached predictions.
        
        Args:
            batch: Dict with keys 'anomaly_score', 'gat_prob', 'label', 'index'
            batch_idx: Index of batch in epoch
        
        Returns:
            None (manual optimization)
        """
        # Extract batch data
        anomaly_scores = batch['anomaly_score']  # [batch_size]
        gat_probs = batch['gat_prob']            # [batch_size]
        labels = batch['label']                  # [batch_size]
        
        batch_size = len(anomaly_scores)
        
        # Normalize and stack states
        states = torch.stack([anomaly_scores, gat_probs], dim=1)  # [batch_size, 2]
        states = states.to(self.device)  # Move to device
        
        # Forward pass: get Q-values for all actions
        q_values = self.fusion_agent.q_network(states)  # [batch_size, num_actions]
        
        # Action selection (epsilon-greedy during training)
        if np.random.rand() < self.fusion_agent.epsilon:
            actions = torch.randint(0, self.fusion_agent.action_dim, (batch_size,))
        else:
            actions = q_values.argmax(dim=1)
        
        # Get fusion weights and compute fused predictions
        alphas = self.fusion_agent.alpha_values[actions.cpu().numpy()]
        fused_scores = (1 - alphas) * anomaly_scores.cpu().numpy() + alphas * gat_probs.cpu().numpy()
        predictions = (fused_scores > 0.5).astype(int)
        
        # Compute rewards
        correct = (predictions == labels.cpu().numpy())
        rewards = torch.tensor(
            np.where(correct, 1.0, -1.0),
            dtype=torch.float32,
            device=self.device
        )
        
        # Add confidence bonus
        confidence = np.maximum(anomaly_scores.cpu().numpy(), gat_probs.cpu().numpy())
        confidence_bonus = torch.tensor(
            np.where(correct, 0.5 * confidence, -0.3 * confidence),
            dtype=torch.float32,
            device=self.device
        )
        rewards = rewards + confidence_bonus
        
        # Normalize rewards
        rewards = torch.clamp(rewards, -1.0, 1.0)
        
        # Store in replay buffer
        next_states = states.clone()  # Simplified: use current state as next state
        dones = torch.zeros(batch_size, dtype=torch.float32)
        
        for i in range(batch_size):
            self.fusion_agent.store_experience(
                states[i].cpu().numpy(),
                actions[i].item(),
                rewards[i].item(),
                next_states[i].cpu().numpy(),
                dones[i].item()
            )
        
        # Train DQN step - the agent handles optimization internally
        if len(self.fusion_agent.replay_buffer) >= self.fusion_agent.batch_size:
            loss = self.fusion_agent.train_step()  # returns a Python float or None
            if loss is not None:
                self.log('train_loss', float(loss), prog_bar=True)
        else:
            loss = None

        # Update target network periodically
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self.fusion_agent.update_target_network()

        # Logging
        accuracy = correct.mean()
        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_reward', rewards.mean(), prog_bar=True)
        self.log('epsilon', self.fusion_agent.epsilon)

        self.episode_accuracies.append(accuracy.item())
        self.episode_rewards.append(rewards.mean().item())

        # Decay epsilon
        if batch_idx % 100 == 0:
            self.fusion_agent.decay_epsilon()

        # With manual optimization we don't return a tensor
        return None
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step: evaluate fusion agent on validation set.
        
        Uses greedy policy (epsilon=0) for evaluation.
        """
        # Extract batch data
        anomaly_scores = batch['anomaly_score']
        gat_probs = batch['gat_prob']
        labels = batch['label']
        
        batch_size = len(anomaly_scores)
        
        # Normalize and stack states
        states = torch.stack([anomaly_scores, gat_probs], dim=1)
        states = states.to(self.device)  # Move to device
        
        # Get Q-values and select greedy actions
        with torch.no_grad():
            q_values = self.fusion_agent.q_network(states)
            actions = q_values.argmax(dim=1)
        
        # Get fusion weights and compute fused predictions
        alphas = self.fusion_agent.alpha_values[actions.cpu().numpy()]
        fused_scores = (1 - alphas) * anomaly_scores.cpu().numpy() + alphas * gat_probs.cpu().numpy()
        predictions = (fused_scores > 0.5).astype(int)
        
        # Compute metrics
        correct = (predictions == labels.cpu().numpy())
        accuracy = correct.mean()
        
        # Log validation metrics
        self.log('val_accuracy', accuracy, prog_bar=True)
        
        self.validation_accuracies.append(accuracy)
        
        return accuracy
    
    def on_train_epoch_end(self):
        """Called at end of training epoch."""
        if self.episode_accuracies:
            avg_accuracy = np.mean(self.episode_accuracies[-100:])
            avg_reward = np.mean(self.episode_rewards[-100:])
            logger.info(
                f"Epoch {self.current_epoch}: "
                f"Accuracy={avg_accuracy:.4f}, Reward={avg_reward:.3f}"
            )
    
    def get_policy(self) -> np.ndarray:
        """Get learned policy as heatmap of best fusion weights."""
        n_points = 50
        anomaly_range = np.linspace(0, 1, n_points)
        gat_range = np.linspace(0, 1, n_points)
        
        policy = np.zeros((n_points, n_points))
        
        with torch.no_grad():
            for i, anomaly in enumerate(anomaly_range):
                for j, gat_prob in enumerate(gat_range):
                    state = torch.tensor(
                        [anomaly, gat_prob], 
                        dtype=torch.float32, 
                        device=self.device
                    )
                    q_values = self.fusion_agent.q_network(state.unsqueeze(0))
                    best_action = q_values.argmax(dim=1).item()
                    policy[i, j] = self.fusion_agent.alpha_values[best_action]
        
        return policy


# Note: Legacy unified LightningModule has been removed.
# Use the specialized modules: VAELightningModule, GATLightningModule, 
# DQNLightningModule, FusionLightningModule instead.
