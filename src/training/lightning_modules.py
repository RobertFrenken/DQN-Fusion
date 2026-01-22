# ============================================================================
# PyTorch Lightning Modules for KD-GAT
# Clean, reusable modules for VGAE, GAT, and DQN architectures
# ============================================================================

import logging
from typing import Any, Optional, Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra_zen import instantiate

from src.utils.experiment_paths import ExperimentPathManager

logger = logging.getLogger(__name__)

# ============================================================================
# BASE LIGHTNING MODULE
# ============================================================================

class BaseKDGATModule(pl.LightningModule):
    """Base Lightning module for KD-GAT experiments"""
    
    def __init__(
        self,
        cfg: DictConfig,
        path_manager: ExperimentPathManager,
        train_loader: Any = None,
        val_loader: Any = None,
    ):
        super().__init__()
        
        self.cfg = cfg
        self.path_manager = path_manager
        self.train_loader_ref = train_loader
        self.val_loader_ref = val_loader
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters('cfg')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        
        # Select optimizer
        optimizer_name = self.cfg.training_config.optimizer.lower()
        
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.training_config.learning_rate,
                weight_decay=self.cfg.training_config.weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.cfg.training_config.learning_rate,
                weight_decay=self.cfg.training_config.weight_decay,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.training_config.learning_rate,
                weight_decay=self.cfg.training_config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Configure scheduler if specified
        if self.cfg.training_config.scheduler is None:
            return optimizer
        
        scheduler_name = self.cfg.training_config.scheduler.lower()
        
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.training_config.epochs,
            )
        elif scheduler_name == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.cfg.training_config.epochs,
            )
        elif scheduler_name == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95,
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
        """Called when training starts"""
        logger.info(f"🚀 Starting training run from {self.path_manager.get_run_dir_safe()}")
    
    def on_train_end(self):
        """Called when training ends - save final metrics"""
        import json
        
        metrics = {
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'best_val_loss': float(min(self.val_losses)) if self.val_losses else None,
        }
        
        metrics_path = self.path_manager.get_training_metrics_path()
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✅ Training metrics saved to {metrics_path}")


# ============================================================================
# VGAE LIGHTNING MODULE (Unsupervised)
# ============================================================================

class VAELightningModule(BaseKDGATModule):
    """Lightning module for Variational Graph AutoEncoder (VGAE)"""
    
    def __init__(
        self,
        cfg: DictConfig,
        path_manager: ExperimentPathManager,
        train_loader: Any = None,
        val_loader: Any = None,
    ):
        super().__init__(cfg, path_manager, train_loader, val_loader)
        
        # Build VGAE model
        self.model = self._build_vgae()
        
        # Loss function
        self.reconstruction_loss = nn.MSELoss()
        self.kl_weight = cfg.learning_config.kl_weight
        
        logger.info(f"✅ Initialized VGAE with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _build_vgae(self) -> nn.Module:
        """Build VGAE model from config"""
        # Implement your VGAE architecture
        # This is a placeholder - replace with your actual VGAE
        from src.models.vgae import VGAE
        return VGAE(
            hidden_dim=self.cfg.model_size_config.hidden_dim,
            latent_dim=self.cfg.model_config.latent_dim,
            num_layers=self.cfg.model_size_config.num_layers,
            dropout=self.cfg.model_size_config.dropout,
        )
    
    def forward(self, x, edge_index):
        """Forward pass through VGAE"""
        return self.model(x, edge_index)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        x, edge_index = batch.x, batch.edge_index
        
        # Forward pass
        recon_x, mu, logvar = self.forward(x, edge_index)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(recon_x, x)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss * self.kl_weight
        
        # Total loss
        loss = recon_loss + kl_loss
        
        self.train_losses.append(loss.item())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('train_kl_loss', kl_loss, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, edge_index = batch.x, batch.edge_index
        
        with torch.no_grad():
            recon_x, mu, logvar = self.forward(x, edge_index)
            
            recon_loss = self.reconstruction_loss(recon_x, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss * self.kl_weight
            
            loss = recon_loss + kl_loss
        
        self.val_losses.append(loss.item())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        x, edge_index = batch.x, batch.edge_index
        
        with torch.no_grad():
            recon_x, mu, logvar = self.forward(x, edge_index)
            recon_loss = self.reconstruction_loss(recon_x, x)
        
        self.log('test_recon_loss', recon_loss)
        return {'test_loss': recon_loss}


# ============================================================================
# GAT LIGHTNING MODULE (Classifier / Fusion)
# ============================================================================

class GATLightningModule(BaseKDGATModule):
    """Lightning module for Graph Attention Network (GAT)"""
    
    def __init__(
        self,
        cfg: DictConfig,
        path_manager: ExperimentPathManager,
        train_loader: Any = None,
        val_loader: Any = None,
    ):
        super().__init__(cfg, path_manager, train_loader, val_loader)
        
        # Build GAT model
        self.model = self._build_gat()
        
        # Loss function
        if cfg.learning_type == "classifier":
            self.loss_fn = nn.CrossEntropyLoss()
        elif cfg.learning_type == "fusion":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown learning type for GAT: {cfg.learning_type}")
        
        logger.info(f"✅ Initialized GAT with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _build_gat(self) -> nn.Module:
        """Build GAT model from config"""
        from src.models.gat import GAT
        return GAT(
            input_dim=self.cfg.model_config.input_dim,
            hidden_dim=self.cfg.model_size_config.hidden_dim,
            output_dim=self.cfg.model_config.output_dim,
            num_heads=self.cfg.model_config.num_heads,
            num_layers=self.cfg.model_size_config.num_layers,
            dropout=self.cfg.model_size_config.dropout,
        )
    
    def forward(self, x, edge_index):
        """Forward pass through GAT"""
        return self.model(x, edge_index)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        
        logits = self.forward(x, edge_index)
        loss = self.loss_fn(logits, y)
        
        self.train_losses.append(loss.item())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            loss = self.loss_fn(logits, y)
        
        self.val_losses.append(loss.item())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            loss = self.loss_fn(logits, y)
        
        self.log('test_loss', loss)
        return {'test_loss': loss}


# ============================================================================
# DQN LIGHTNING MODULE (RL / Fusion)
# ============================================================================

class DQNLightningModule(BaseKDGATModule):
    """Lightning module for Deep Q-Network (DQN)"""
    
    def __init__(
        self,
        cfg: DictConfig,
        path_manager: ExperimentPathManager,
        train_loader: Any = None,
        val_loader: Any = None,
    ):
        super().__init__(cfg, path_manager, train_loader, val_loader)
        
        # Build DQN model
        self.model = self._build_dqn()
        
        logger.info(f"✅ Initialized DQN with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _build_dqn(self) -> nn.Module:
        """Build DQN model from config"""
        from src.models.dqn import DQN
        return DQN(
            input_dim=self.cfg.model_config.input_dim,
            hidden_dim=self.cfg.model_size_config.hidden_dim,
            action_dim=self.cfg.model_config.action_dim,
            num_layers=self.cfg.model_size_config.num_layers,
            dropout=self.cfg.model_size_config.dropout,
            dueling=self.cfg.model_config.dueling,
        )
    
    def forward(self, x):
        """Forward pass through DQN"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step (DQN-specific)"""
        states, actions, rewards, next_states, dones = batch
        
        # Current Q-values
        q_values = self.forward(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.forward(next_states).max(1)[0]
            target_q_values = rewards + 0.99 * next_q_values * (1 - dones)
        
        # Loss
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        self.train_losses.append(loss.item())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        states, actions, rewards, next_states, dones = batch
        
        with torch.no_grad():
            q_values = self.forward(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            next_q_values = self.forward(next_states).max(1)[0]
            target_q_values = rewards + 0.99 * next_q_values * (1 - dones)
            
            loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        self.val_losses.append(loss.item())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        states, actions, rewards, next_states, dones = batch
        
        with torch.no_grad():
            q_values = self.forward(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            next_q_values = self.forward(next_states).max(1)[0]
            target_q_values = rewards + 0.99 * next_q_values * (1 - dones)
            
            loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        self.log('test_loss', loss)
        return {'test_loss': loss}
