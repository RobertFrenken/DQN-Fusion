"""
Unified Lightning Modules for CAN-Graph Training

Consolidates all PyTorch Lightning modules into a single coherent file.

Components:
- BaseKDGATModule: Base class with common functionality
- VAELightningModule: VGAE training (unsupervised)
- GATLightningModule: Graph Attention Network (classification)
- DQNLightningModule: DQN training for fusion
- FusionLightningModule: DQN-based fusion with prediction caching
- CANGraphLightningModule: Legacy unified module (backward compatibility)

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
from src.utils.experiment_paths import ExperimentPathManager

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
    """
    
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
        """Configure optimizer and learning rate scheduler."""
        
        # Select optimizer
        optimizer_name = self.cfg.training_config.optimizer.lower()
        lr = self.cfg.training_config.learning_rate
        wd = self.cfg.training_config.weight_decay
        
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
        logger.info(f"🚀 Starting training run from {self.path_manager.get_run_dir_safe()}")
    
    def on_train_end(self):
        """Called when training ends - save final metrics."""
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
    """
    Lightning module for Variational Graph AutoEncoder (VGAE).
    
    Used for unsupervised pretraining and anomaly detection.
    Reconstructs graph structure and node features.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        path_manager: ExperimentPathManager,
        train_loader: Any = None,
        val_loader: Any = None,
        num_ids: int = 1000
    ):
        super().__init__(cfg, path_manager, train_loader, val_loader)
        
        self.num_ids = num_ids
        
        # Build VGAE model
        self.model = self._build_vgae()
        
        # Loss weights
        self.reconstruction_loss_fn = nn.MSELoss()
        self.kl_weight = getattr(cfg.learning_config, 'kl_weight', 0.01)
        
        logger.info(
            f"✅ Initialized VGAE with {sum(p.numel() for p in self.model.parameters())} parameters"
        )
    
    def _build_vgae(self) -> nn.Module:
        """Build VGAE model from config."""
        # Build params from config and prefer explicit progressive `hidden_dims` if present
        if hasattr(self.cfg.model_config, 'hidden_dims'):
            hidden_dims = list(self.cfg.model_config.hidden_dims)
        elif hasattr(self.cfg.model_config, 'encoder_dims'):
            hidden_dims = list(self.cfg.model_config.encoder_dims)
        else:
            hidden_dims = None

        vgae_params = {
            'num_ids': self.num_ids,
            'in_channels': self.cfg.model_config.input_dim,
            'hidden_dims': hidden_dims,
            'latent_dim': getattr(
                self.cfg.model_config, 
                'latent_dim', 
                (hidden_dims[-1] if hidden_dims else 32)
            ),
            'encoder_heads': getattr(self.cfg.model_config, 'attention_heads', 4),
            'decoder_heads': getattr(self.cfg.model_config, 'attention_heads', 4),
            'embedding_dim': getattr(self.cfg.model_config, 'embedding_dim', 32),
            'dropout': getattr(self.cfg.model_config, 'dropout', 0.15),
            'batch_norm': getattr(self.cfg.model_config, 'batch_norm', True)
        }
        
        return GraphAutoencoderNeighborhood(**vgae_params)
    
    def forward(self, batch):
        """Forward pass through VGAE."""
        x = batch.x
        edge_index = batch.edge_index
        b = batch.batch
        return self.model(x, edge_index, b)
    
    def training_step(self, batch, batch_idx):
        """Training step with reconstruction + KL loss."""
        # Forward pass
        cont_out, canid_logits, neighbor_logits, z, kl_loss = self.forward(batch)
        
        # Reconstruction losses
        continuous_features = batch.x[:, 1:]
        reconstruction_loss = self.reconstruction_loss_fn(cont_out, continuous_features)
        
        canid_targets = batch.x[:, 0].long()
        canid_loss = F.cross_entropy(canid_logits, canid_targets)
        
        # Total loss
        total_loss = reconstruction_loss + 0.1 * canid_loss + self.kl_weight * kl_loss
        
        self.train_losses.append(total_loss.item())
        
        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', reconstruction_loss, on_epoch=True)
        self.log('train_canid_loss', canid_loss, on_epoch=True)
        self.log('train_kl_loss', kl_loss, on_epoch=True)
        
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


# ============================================================================
# GAT LIGHTNING MODULE (Classifier)
# ============================================================================

class GATLightningModule(BaseKDGATModule):
    """
    Lightning module for Graph Attention Network (GAT).
    
    Used for supervised classification of normal vs attack traffic.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        path_manager: ExperimentPathManager,
        train_loader: Any = None,
        val_loader: Any = None,
        num_ids: int = 1000
    ):
        super().__init__(cfg, path_manager, train_loader, val_loader)
        
        self.num_ids = num_ids
        
        # Build GAT model
        self.model = self._build_gat()
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        logger.info(
            f"✅ Initialized GAT with {sum(p.numel() for p in self.model.parameters())} parameters"
        )
    
    def _build_gat(self) -> nn.Module:
        """Build GAT model from config."""
        if hasattr(self.cfg.model_config, 'gat'):
            gat_params = dict(self.cfg.model_config.gat)
        else:
            gat_params = {
                'input_dim': self.cfg.model_config.input_dim,
                'hidden_channels': self.cfg.model_config.hidden_channels,
                'output_dim': self.cfg.model_config.output_dim,
                'num_layers': self.cfg.model_config.num_layers,
                'heads': self.cfg.model_config.heads,
                'dropout': self.cfg.model_config.dropout,
                'num_fc_layers': self.cfg.model_config.num_fc_layers,
                'embedding_dim': self.cfg.model_config.embedding_dim,
            }
        
        # Normalize parameter names
        gat_params['in_channels'] = gat_params.pop('input_dim')
        gat_params['out_channels'] = gat_params.pop('output_dim')
        
        # Remove unused params
        for unused_param in ['use_jumping_knowledge', 'jk_mode', 'use_residual', 
                            'use_batch_norm', 'activation']:
            gat_params.pop(unused_param, None)
        
        gat_params['num_ids'] = self.num_ids
        
        return GATWithJK(**gat_params)
    
    def forward(self, batch):
        """Forward pass through GAT."""
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        """Training step with classification loss."""
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch.y)
        
        self.train_losses.append(loss.item())
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == batch.y).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True)
        
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
        path_manager: ExperimentPathManager,
        train_loader: Any = None,
        val_loader: Any = None,
    ):
        super().__init__(cfg, path_manager, train_loader, val_loader)
        
        # Build DQN model
        self.model = self._build_dqn()
        
        # Target network for stable learning
        self.target_network = self._build_dqn()
        self.target_network.load_state_dict(self.model.state_dict())
        
        # Hyperparameters
        self.gamma = cfg.training_config.get('gamma', 0.99)
        self.target_update_freq = cfg.training_config.get('target_update_freq', 100)
        self.update_counter = 0
        
        logger.info(
            f"✅ Initialized DQN with {sum(p.numel() for p in self.model.parameters())} parameters"
        )
    
    def _build_dqn(self) -> nn.Module:
        """Build DQN model from config."""
        model_type = getattr(self.cfg.model_config, 'dqn_type', 'teacher')
        
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
        self.save_hyperparameters(ignore=['num_ids'])
        
        self.fusion_config = fusion_config
        self.num_ids = num_ids
        
        # Initialize DQN fusion agent
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
            device=self.device if self.device.type == 'cuda' else 'cpu'
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


# ============================================================================
# LEGACY UNIFIED MODULE (Backward Compatibility)
# ============================================================================

class CANGraphLightningModule(pl.LightningModule):
    """
    Legacy unified Lightning Module for CAN intrusion detection.
    
    Handles GAT, VGAE, autoencoder, knowledge distillation, and fusion modes.
    
    DEPRECATED: Use specialized modules (VAELightningModule, GATLightningModule, etc.)
    Kept for backward compatibility with existing code.
    """
    
    def __init__(
        self, 
        model_config, 
        training_config, 
        model_type="gat", 
        training_mode="normal", 
        num_ids=1000
    ):
        super().__init__()
        
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters()
        
        self.model_config = model_config
        self.training_config = training_config
        self.model_type = model_type
        self.training_mode = training_mode
        self.num_ids = num_ids
        self.batch_size = training_config.batch_size
        
        self.model = self._create_model()
        self.teacher_model = None
        
        if training_mode == "knowledge_distillation":
            self.setup_knowledge_distillation()
    
    def _create_model(self):
        """Create model based on model_type."""
        if self.model_type in ["gat", "gat_student"]:
            if hasattr(self.model_config, 'gat'):
                gat_params = dict(self.model_config.gat)
            else:
                gat_params = {
                    'input_dim': self.model_config.input_dim,
                    'hidden_channels': self.model_config.hidden_channels,
                    'output_dim': self.model_config.output_dim,
                    'num_layers': self.model_config.num_layers,
                    'heads': self.model_config.heads,
                    'dropout': self.model_config.dropout,
                    'num_fc_layers': self.model_config.num_fc_layers,
                    'embedding_dim': self.model_config.embedding_dim,
                }
            
            gat_params['in_channels'] = gat_params.pop('input_dim')
            gat_params['out_channels'] = gat_params.pop('output_dim')
            
            for unused in ['use_jumping_knowledge', 'jk_mode', 'use_residual', 
                          'use_batch_norm', 'activation']:
                gat_params.pop(unused, None)
            
            gat_params['num_ids'] = self.num_ids
            return GATWithJK(**gat_params)
        
        elif self.model_type in ["vgae", "vgae_student"]:
            if hasattr(self.model_config, 'hidden_dims'):
                hidden_dims = list(self.model_config.hidden_dims)
            elif hasattr(self.model_config, 'encoder_dims'):
                hidden_dims = list(self.model_config.encoder_dims)
            else:
                hidden_dims = None

            vgae_params = {
                'num_ids': self.num_ids,
                'in_channels': self.model_config.input_dim,
                'hidden_dims': hidden_dims,
                'latent_dim': getattr(
                    self.model_config, 
                    'latent_dim', 
                    (hidden_dims[-1] if hidden_dims else 32)
                ),
                'encoder_heads': getattr(self.model_config, 'attention_heads', 4),
                'decoder_heads': getattr(self.model_config, 'attention_heads', 4),
                'embedding_dim': getattr(self.model_config, 'embedding_dim', 32),
                'dropout': getattr(self.model_config, 'dropout', 0.15),
                'batch_norm': getattr(self.model_config, 'batch_norm', True)
            }
            return GraphAutoencoderNeighborhood(**vgae_params)
        
        elif self.model_type in ["dqn", "dqn_student"]:
            if self.model_type == "dqn":
                return create_dqn_teacher()
            else:
                return create_dqn_student()
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def training_step(self, batch, batch_idx):
        """Dispatch to the appropriate training step based on training_mode."""
        if self.training_mode == "autoencoder":
            return self._autoencoder_training_step(batch, batch_idx)
        elif self.training_mode == "knowledge_distillation":
            return self._knowledge_distillation_step(batch, batch_idx)
        elif self.training_mode == "fusion":
            return self._fusion_training_step(batch, batch_idx)
        else:
            return self._normal_training_step(batch, batch_idx)
    
    def _normal_training_step(self, batch, batch_idx):
        """Standard training step."""
        if self.model_type == "gat":
            output = self.model(batch)
        else:
            output = self.forward(batch)

        base_loss = self._compute_loss(output, batch)
        
        try:
            batch_size = batch.y.size(0)
        except Exception:
            batch_size = None
        
        self.log('train_loss', base_loss, prog_bar=True, batch_size=batch_size)
        return base_loss
    
    def _autoencoder_training_step(self, batch, batch_idx):
        """Autoencoder training step (normal traffic only)."""
        if hasattr(batch, 'y'):
            normal_mask = batch.y == 0
            if normal_mask.sum() == 0:
                return None
            filtered_batch = self._filter_batch_by_mask(batch, normal_mask)
            output = self.forward(filtered_batch)
            loss = self._compute_autoencoder_loss(output, filtered_batch)
        else:
            output = self.forward(batch)
            loss = self._compute_autoencoder_loss(output, batch)
        
        self.log('train_autoencoder_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss
    
    def _knowledge_distillation_step(self, batch, batch_idx):
        """Knowledge distillation training step."""
        student_output = self.forward(batch)
        teacher_output = self._get_teacher_output_cached(batch)
        
        distillation_loss = self._compute_distillation_loss(
            student_output, teacher_output, batch
        )
        
        # Log teacher-student comparison
        if self.training_config.get('log_teacher_student_comparison', True):
            with torch.no_grad():
                if hasattr(batch, 'y'):
                    if isinstance(teacher_output, tuple):
                        teacher_logits = teacher_output[1]
                        student_logits = (student_output[1] if isinstance(student_output, tuple) 
                                        else student_output)
                    else:
                        teacher_logits = teacher_output
                        student_logits = (student_output[1] if isinstance(student_output, tuple) 
                                        else student_output)
                    
                    teacher_acc = (teacher_logits.argmax(dim=-1) == batch.y).float().mean()
                    student_acc = (student_logits.argmax(dim=-1) == batch.y).float().mean()
                    
                    self.log('teacher_accuracy', teacher_acc, prog_bar=False, 
                           batch_size=batch.y.size(0))
                    self.log('student_accuracy', student_acc, prog_bar=False, 
                           batch_size=batch.y.size(0))
                    self.log('accuracy_gap', teacher_acc - student_acc, prog_bar=False, 
                           batch_size=batch.y.size(0))
                
                teacher_flat = (teacher_output[0].flatten() if isinstance(teacher_output, tuple) 
                              else teacher_output.flatten())
                student_flat = (student_output[0].flatten() if isinstance(student_output, tuple) 
                              else student_output.flatten())
                similarity = F.cosine_similarity(
                    teacher_flat.unsqueeze(0), 
                    student_flat.unsqueeze(0)
                )
                self.log('teacher_student_similarity', similarity, prog_bar=False, 
                       batch_size=batch.y.size(0))
        
        self.log('train_distillation_loss', distillation_loss, prog_bar=True, 
               batch_size=batch.y.size(0))
        return distillation_loss
    
    def _fusion_training_step(self, batch, batch_idx):
        """Fusion training step."""
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)
        self.log('train_fusion_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)
        self.log('val_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)
        self.log('test_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss
    
    def forward(self, batch):
        """Forward dispatcher based on model_type."""
        if self.model_type in ["gat", "gat_student"]:
            return self.model(batch)
        
        if self.model_type in ["vgae", "vgae_student"]:
            x = getattr(batch, 'x', None)
            edge_index = getattr(batch, 'edge_index', None)
            b = getattr(batch, 'batch', None)
            if x is None or edge_index is None or b is None:
                raise ValueError(
                    "Batch missing required attributes for VGAE: x, edge_index, batch"
                )
            return self.model(x, edge_index, b)
        
        if self.model_type in ["dqn", "dqn_student"]:
            x = getattr(batch, 'x', None)
            if x is None:
                raise ValueError("Batch missing 'x' required for DQN forward")
            return self.model(x)
        
        raise ValueError(f"Unsupported model_type in forward: {self.model_type}")
    
    def _compute_loss(self, output, batch):
        """Compute loss based on model type."""
        if self.model_type in ["vgae", "vgae_student"]:
            cont_out, canid_logits, neighbor_logits, z, kl_loss = output
            reconstruction_loss = F.mse_loss(cont_out, batch.x[:, 1:])
            canid_loss = F.cross_entropy(canid_logits, batch.x[:, 0].long())
            return reconstruction_loss + 0.1 * canid_loss + 0.01 * kl_loss
        else:
            if hasattr(batch, 'y'):
                return F.cross_entropy(output, batch.y)
            else:
                return F.mse_loss(output, batch.x)
    
    def _compute_autoencoder_loss(self, output, batch):
        """Compute autoencoder loss."""
        if self.model_type in ["vgae", "vgae_student"]:
            cont_out, canid_logits, neighbor_logits, z, kl_loss = output
            continuous_features = batch.x[:, 1:]
            reconstruction_loss = F.mse_loss(cont_out, continuous_features)
            canid_targets = batch.x[:, 0].long()
            canid_loss = F.cross_entropy(canid_logits, canid_targets)
            return reconstruction_loss + 0.1 * canid_loss + 0.01 * kl_loss
        else:
            return F.mse_loss(output, batch.x)
    
    def _compute_distillation_loss(self, student_output, teacher_output, batch):
        """Compute knowledge distillation loss."""
        temperature = self.training_config.get('distillation_temperature', 4.0)
        alpha = self.training_config.get('distillation_alpha', 0.7)
        
        if hasattr(batch, 'y'):
            hard_loss = self._compute_loss(student_output, batch)
            
            if student_output.dim() > 1 and student_output.size(-1) > 1:
                soft_targets = torch.softmax(teacher_output / temperature, dim=-1)
                soft_prob = torch.log_softmax(student_output / temperature, dim=-1)
                soft_loss = F.kl_div(
                    soft_prob, soft_targets, reduction='batchmean'
                ) * (temperature ** 2)
            else:
                soft_loss = F.mse_loss(student_output, teacher_output)
            
            total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            self.log('hard_loss', hard_loss, prog_bar=False, batch_size=student_output.size(0))
            self.log('soft_loss', soft_loss, prog_bar=False, batch_size=student_output.size(0))
            
            return total_loss
    
    def _filter_batch_by_mask(self, batch, mask):
        """Filter batch by mask (simplified implementation)."""
        return batch
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        def get_config_value(key, default=None):
            if hasattr(self.training_config, 'get'):
                return self.training_config.get(key, default)
            else:
                return getattr(self.training_config, key, default)
        
        # Get optimizer parameters
        if hasattr(self.training_config, 'optimizer'):
            optimizer_config = self.training_config.optimizer
            optimizer_name = optimizer_config.name.lower()
            learning_rate = optimizer_config.lr
            weight_decay = optimizer_config.weight_decay
        else:
            optimizer_name = get_config_value('optimizer', 'adam').lower()
            learning_rate = get_config_value('learning_rate', 0.001)
            weight_decay = get_config_value('weight_decay', 0.0001)
        
        # Create optimizer
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = get_config_value('momentum', 0.9)
            optimizer = torch.optim.SGD(
                self.parameters(), lr=learning_rate, 
                weight_decay=weight_decay, momentum=momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Check if scheduler is needed
        use_scheduler = get_config_value('use_scheduler', False)
        if hasattr(self.training_config, 'scheduler') and self.training_config.scheduler:
            use_scheduler = self.training_config.scheduler.use_scheduler
        
        if not use_scheduler:
            return optimizer
        
        # Create scheduler
        if hasattr(self.training_config, 'scheduler'):
            scheduler_config = self.training_config.scheduler
            scheduler_type = scheduler_config.scheduler_type.lower()
            scheduler_params = scheduler_config.params
            
            if scheduler_type == 'cosine':
                T_max = scheduler_params.get('T_max', self.training_config.max_epochs)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
            elif scheduler_type == 'step':
                step_size = scheduler_params.get('step_size', 30)
                gamma = scheduler_params.get('gamma', 0.1)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=step_size, gamma=gamma
                )
            elif scheduler_type == 'exponential':
                gamma = scheduler_params.get('gamma', 0.95)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        else:
            scheduler_type = get_config_value('scheduler_type', 'cosine').lower()
            scheduler_params = get_config_value('scheduler_params', {})
            
            if scheduler_type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=scheduler_params.get('T_max', self.training_config.max_epochs)
                )
            elif scheduler_type == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=scheduler_params.get('step_size', 30), 
                    gamma=scheduler_params.get('gamma', 0.1)
                )
            elif scheduler_type == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, 
                    gamma=scheduler_params.get('gamma', 0.95)
                )
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        
        return [optimizer], [scheduler]
    
    def setup_knowledge_distillation(self):
        """Set up knowledge distillation (placeholder)."""
        logger.warning("setup_knowledge_distillation() called but not fully implemented")
    
    def _get_teacher_output_cached(self, batch):
        """Get teacher output (placeholder)."""
        raise NotImplementedError("Teacher output caching not implemented")
