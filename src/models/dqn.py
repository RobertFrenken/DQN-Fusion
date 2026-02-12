import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Q-network with configurable depth and width."""

    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class EnhancedDQNFusionAgent:
    """
    Enhanced DQN agent for dynamic fusion of GAT and VGAE outputs.
    Includes Double DQN, proper target updates, and validation.
    """
    
    def __init__(self, alpha_steps=21, lr=1e-3, gamma=0.9, epsilon=0.2,
                 epsilon_decay=0.995, min_epsilon=0.01, buffer_size=50000,
                 batch_size=128, target_update_freq=100, device='cpu', state_dim=15,
                 hidden_dim=128, num_layers=3):

        # Action and state space
        self.alpha_values = np.linspace(0, 1, alpha_steps)
        self.action_dim = alpha_steps
        self.state_dim = state_dim

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.buffer_size = buffer_size

        # Networks
        self.q_network = QNetwork(state_dim, self.action_dim, hidden_dim, num_layers).to(self.device)
        self.target_network = QNetwork(state_dim, self.action_dim, hidden_dim, num_layers).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1000, factor=0.8)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability
        
        # Experience replay
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # Training tracking
        self.training_step = 0
        self.update_counter = 0
        self.reward_history = []
        self.accuracy_history = []
        self.loss_history = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Validation tracking
        self.validation_scores = []
        self.best_validation_score = -float('inf')
        self.patience_counter = 0
        self.max_patience = 5000
        
        log.info("DQN Agent initialized: %d actions, state_dim=%d", alpha_steps, self.state_dim)

    def normalize_state(self, state_features: np.ndarray) -> np.ndarray:
        """
        Normalize 15D state representation.

        Args:
            state_features: [15] array with:
                [0:3] - VGAE errors (node, neighbor, canid)
                [3:7] - VGAE latent stats (mean, std, max, min)
                [7] - VGAE confidence
                [8:10] - GAT logits (class 0, class 1)
                [10:14] - GAT embedding stats (mean, std, max, min)
                [14] - GAT confidence

        Returns:
            Normalized 15D state as float32 array
        """
        # Ensure input is numpy array
        if not isinstance(state_features, np.ndarray):
            state_features = np.array(state_features, dtype=np.float32)

        # Validate dimensions
        if len(state_features) != 15:
            raise ValueError(f"Expected 15D state, got {len(state_features)}D")

        # Clip confidence values to [0, 1]
        state_features[7] = np.clip(state_features[7], 0.0, 1.0)  # VGAE confidence
        state_features[14] = np.clip(state_features[14], 0.0, 1.0)  # GAT confidence

        return state_features.astype(np.float32)
        

    def select_action(self, state_features: np.ndarray, training: bool = True) -> Tuple[float, int, np.ndarray]:
        """
        Select action using epsilon-greedy policy with 15D state.

        Args:
            state_features: [15] array with all VGAE and GAT features
            training: Whether in training mode (use epsilon-greedy)

        Returns:
            Tuple of (alpha_value, action_idx, normalized_state)
        """
        state = self.normalize_state(state_features)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if training and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = torch.argmax(q_values).item()

        alpha_value = self.alpha_values[action_idx]
        return alpha_value, action_idx, state

    def compute_fusion_reward(self, prediction: int, true_label: int,
                             state_features: np.ndarray,
                             alpha: float) -> float:
        """
        Enhanced reward function with 15D state features.

        Args:
            prediction: Fused prediction (0 or 1)
            true_label: Ground truth label (0 or 1)
            state_features: [15] array with all features
            alpha: Fusion weight selected by DQN

        Returns:
            Reward scalar
        """
        # Extract derived scores from 15D state
        # VGAE errors: [0:3] - use weighted combination
        vgae_errors = state_features[0:3]
        vgae_weights = np.array([0.4, 0.35, 0.25])
        anomaly_score = float(np.clip(np.sum(vgae_errors * vgae_weights), 0.0, 1.0))

        # GAT logits: [8:10] - convert to probability
        gat_logits = state_features[8:10]
        gat_probs = np.exp(gat_logits) / np.sum(np.exp(gat_logits))  # Softmax
        gat_prob = float(gat_probs[1])  # Probability of attack class

        # Confidence scores
        vgae_confidence = float(state_features[7])
        gat_confidence = float(state_features[14])
        combined_confidence = max(vgae_confidence, gat_confidence)

        # Base accuracy reward (higher magnitude for importance)
        base_reward = 3.0 if prediction == true_label else -3.0

        # Model agreement bonus/penalty
        model_agreement = 1.0 - abs(anomaly_score - gat_prob)

        if prediction == true_label:
            # Reward model agreement for correct predictions
            agreement_bonus = 1.0 * model_agreement

            # Confidence bonus based on prediction type
            if true_label == 1:  # Attack case
                confidence = max(anomaly_score, gat_prob)
            else:  # Normal case
                confidence = 1.0 - max(anomaly_score, gat_prob)

            confidence_bonus = 0.5 * confidence + 0.3 * combined_confidence
            total_reward = base_reward + agreement_bonus + confidence_bonus

        else:
            # Penalize disagreement for wrong predictions
            disagreement_penalty = -1.0 * (1.0 - model_agreement)

            # Penalize overconfidence in wrong predictions
            fused_confidence = alpha * gat_prob + (1 - alpha) * anomaly_score
            if prediction == 1:  # False positive
                overconf_penalty = -1.5 * fused_confidence
            else:  # False negative
                overconf_penalty = -1.5 * (1.0 - fused_confidence)

            total_reward = base_reward + disagreement_penalty + overconf_penalty

        # Encourage balanced fusion (slight preference for learning)
        balance_bonus = 0.3 * (1.0 - abs(alpha - 0.5) * 2)

        return total_reward + balance_bonus

    def store_experience(self, state: np.ndarray, action_idx: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        state = state.astype(np.float32) if state.dtype != np.float32 else state
        next_state = next_state.astype(np.float32) if next_state.dtype != np.float32 else next_state
        
        experience = (state, action_idx, reward, next_state, done)
        self.replay_buffer.append(experience)
        self.current_episode_reward += reward

    def train_step(self) -> Optional[float]:
        """Enhanced training step with Double DQN."""
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # Sample random batch
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[idx] for idx in batch_indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use main network for action selection, target for evaluation
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(dim=1)[1]
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss and optimize
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()
        
        # Track training metrics
        self.training_step += 1
        self.loss_history.append(loss.item())
        self.reward_history.append(rewards.mean().item())
        
        return loss.item()

    def update_target_network(self):
        """Update target network parameters."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def validate_agent(self, validation_data: List[Tuple], num_samples: int = 1000) -> Dict:
        """
        Validate agent performance with 15D states.

        Args:
            validation_data: List of (state_features[15], true_label) tuples
            num_samples: Number of samples to validate on

        Returns:
            Dict with validation metrics
        """
        self.q_network.eval()

        correct = 0
        total_reward = 0
        alpha_values_used = []

        sample_data = validation_data[:num_samples] if len(validation_data) >= num_samples else validation_data

        for state_features, true_label in sample_data:
            # Select action with 15D state
            alpha, _, _ = self.select_action(state_features, training=False)
            alpha_values_used.append(alpha)

            # Derive scalar scores for fusion from 15D state
            # VGAE errors: weighted combination
            vgae_errors = state_features[0:3]
            vgae_weights = np.array([0.4, 0.35, 0.25])
            anomaly_score = float(np.clip(np.sum(vgae_errors * vgae_weights), 0.0, 1.0))

            # GAT logits: softmax to probability
            gat_logits = state_features[8:10]
            gat_probs = np.exp(gat_logits) / np.sum(np.exp(gat_logits))
            gat_prob = float(gat_probs[1])

            # Make fusion prediction
            fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
            prediction = 1 if fused_score > 0.5 else 0

            # Calculate metrics
            correct += (prediction == true_label)
            reward = self.compute_fusion_reward(prediction, true_label, state_features, alpha)
            total_reward += reward

        self.q_network.train()

        validation_results = {
            'accuracy': correct / len(sample_data),
            'avg_reward': total_reward / len(sample_data),
            'avg_alpha': np.mean(alpha_values_used),
            'alpha_std': np.std(alpha_values_used)
        }

        # Update learning rate based on validation
        self.scheduler.step(validation_results['avg_reward'])

        # Early stopping check
        current_score = validation_results['accuracy'] + 0.1 * validation_results['avg_reward']
        if current_score > self.best_validation_score:
            self.best_validation_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        self.validation_scores.append(validation_results)
        return validation_results