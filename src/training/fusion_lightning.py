"""
PyTorch Lightning Module for Fusion Training

Implements DQN-based fusion of VGAE and GAT outputs using PyTorch Lightning.
Pre-caches predictions from both models to eliminate redundant computation.

Key Benefits:
- Lightning handles training loop, checkpointing, logging
- Pre-cached predictions reduce complexity
- Vectorized GPU operations for fusion agent training
- Seamless integration with Hydra-Zen configs
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging

from src.models.adaptive_fusion import EnhancedDQNFusionAgent

logger = logging.getLogger(__name__)


class FusionPredictionCache(Dataset):
    """
    Pre-computed cache of VGAE and GAT predictions for efficient fusion training.
    
    Stores:
    - Anomaly scores from VGAE
    - Classification probabilities from GAT
    - Ground truth labels
    - Sample indices for tracking
    """
    
    def __init__(self, anomaly_scores: np.ndarray, gat_probs: np.ndarray, 
                 labels: np.ndarray, indices: np.ndarray = None):
        """
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
        
        assert len(self.anomaly_scores) == len(self.gat_probs) == len(self.labels), \
            "All inputs must have same length"
    
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
        
        # Target network update counter
        self.target_update_counter = 0
        self.target_update_freq = fusion_config.get('target_update_freq', 100)
        
        # Batch tracking for effective episodes
        self.batch_count = 0
        
        # Memory optimization settings
        self.memory_optimization = fusion_config.get('memory_optimization', {})
        self.empty_cache_every_n_steps = self.memory_optimization.get('empty_cache_every_n_steps', 100)
        self.use_gradient_checkpointing = self.memory_optimization.get('gradient_checkpointing', False)
    
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
            Loss value for logging
        """
        # Extract batch data (already on device)
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
        
        # Train DQN step
        if len(self.fusion_agent.replay_buffer) >= self.fusion_agent.batch_size:
            loss = self.fusion_agent.train_step()
        else:
            loss = 0.0
        
        # Update target network periodically
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self.fusion_agent.update_target_network()
        
        # Logging
        accuracy = correct.mean()
        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_reward', rewards.mean(), prog_bar=True)
        if loss > 0:
            self.log('train_loss', loss, prog_bar=True)
        self.log('epsilon', self.fusion_agent.epsilon)
        
        self.episode_accuracies.append(accuracy.item())
        self.episode_rewards.append(rewards.mean().item())
        
        # Decay epsilon
        if batch_idx % 100 == 0:
            self.fusion_agent.decay_epsilon()
        
        return loss if loss > 0 else torch.tensor(0.0, device=self.device)
    
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
            avg_accuracy = np.mean(self.episode_accuracies[-100:])  # Last 100 batches
            avg_reward = np.mean(self.episode_rewards[-100:])
            logger.info(f"Epoch {self.current_epoch}: Accuracy={avg_accuracy:.4f}, Reward={avg_reward:.3f}")
    
    def get_policy(self) -> np.ndarray:
        """Get learned policy as heatmap of best fusion weights."""
        n_points = 50
        anomaly_range = np.linspace(0, 1, n_points)
        gat_range = np.linspace(0, 1, n_points)
        
        policy = np.zeros((n_points, n_points))
        
        with torch.no_grad():
            for i, anomaly in enumerate(anomaly_range):
                for j, gat_prob in enumerate(gat_range):
                    state = torch.tensor([anomaly, gat_prob], dtype=torch.float32, device=self.device)
                    q_values = self.fusion_agent.q_network(state.unsqueeze(0))
                    best_action = q_values.argmax(dim=1).item()
                    policy[i, j] = self.fusion_agent.alpha_values[best_action]
        
        return policy
