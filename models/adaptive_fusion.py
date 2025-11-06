import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Tuple, Dict, List, Optional
import pickle
import os
import sys



class QNetwork(nn.Module):
    """Q-network with batch normalization and dropout."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class EnhancedDQNFusionAgent:
    """
    Enhanced DQN agent for dynamic fusion of GAT and VGAE outputs.
    Includes Double DQN, proper target updates, and validation.
    """
    
    def __init__(self, alpha_steps=21, lr=1e-3, gamma=0.9, epsilon=0.2,
                 epsilon_decay=0.995, min_epsilon=0.01, buffer_size=50000,
                 batch_size=128, target_update_freq=100, device='cpu'):
        
        # Action and state space
        self.alpha_values = np.linspace(0, 1, alpha_steps)
        self.action_dim = alpha_steps
        self.state_dim = 4  # anomaly_score, gat_prob, confidence_diff, avg_confidence
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1000, factor=0.8)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability
        
        # Experience replay
        self.replay_buffer = deque(maxlen=buffer_size)
        
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
        
        print(f"✓ Enhanced DQN Agent initialized with {alpha_steps} actions, state_dim={self.state_dim}")

    def normalize_state(self, anomaly_score: float, gat_prob: float) -> np.ndarray:
        """Enhanced state representation with additional features."""
        # Clip to valid ranges
        anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
        gat_prob = np.clip(gat_prob, 0.0, 1.0)
        
        # Additional features
        confidence_diff = abs(anomaly_score - gat_prob)
        avg_confidence = (anomaly_score + gat_prob) / 2.0
        
        return np.array([anomaly_score, gat_prob, confidence_diff, avg_confidence], dtype=np.float32)

    def select_action(self, anomaly_score: float, gat_prob: float, training: bool = True) -> Tuple[float, int, np.ndarray]:
        """Select action using epsilon-greedy policy with enhanced state."""
        state = self.normalize_state(anomaly_score, gat_prob)
        state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
        
        if training and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = torch.argmax(q_values).item()
                
        alpha_value = self.alpha_values[action_idx]
        return alpha_value, action_idx, state

    def compute_fusion_reward(self, prediction: int, true_label: int, 
                             anomaly_score: float, gat_prob: float, 
                             alpha: float) -> float:
        """Enhanced reward function for better fusion learning."""
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
            
            confidence_bonus = 0.5 * confidence
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
        self.replay_buffer.append((state, action_idx, reward, next_state, done))
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

    def end_episode(self):
        """Call at the end of each episode."""
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def validate_agent(self, validation_data: List[Tuple], num_samples: int = 1000) -> Dict:
        """Validate agent performance during training."""
        self.q_network.eval()
        
        correct = 0
        total_reward = 0
        alpha_values_used = []
        
        sample_data = validation_data[:num_samples] if len(validation_data) >= num_samples else validation_data
        
        for anomaly_score, gat_prob, true_label in sample_data:
            alpha, _, _ = self.select_action(anomaly_score, gat_prob, training=False)
            alpha_values_used.append(alpha)
            
            # Make fusion prediction
            fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
            prediction = 1 if fused_score > 0.5 else 0
            
            # Calculate metrics
            correct += (prediction == true_label)
            reward = self.compute_fusion_reward(prediction, true_label, anomaly_score, gat_prob, alpha)
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

    def should_stop_training(self) -> bool:
        """Check if training should stop due to convergence."""
        return self.patience_counter >= self.max_patience

    def get_training_stats(self) -> Dict:
        """Get comprehensive training statistics."""
        return {
            'training_steps': self.training_step,
            'episodes': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_loss': np.mean(self.loss_history[-1000:]) if self.loss_history else 0,
            'current_epsilon': self.epsilon,
            'best_validation_score': self.best_validation_score,
            'patience_counter': self.patience_counter,
            'buffer_size': len(self.replay_buffer)
        }

    def save_agent(self, filepath: str):
        """Save complete agent state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'hyperparameters': {
                'alpha_values': self.alpha_values,
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'best_validation_score': self.best_validation_score
            },
            'training_history': {
                'reward_history': self.reward_history,
                'loss_history': self.loss_history,
                'episode_rewards': self.episode_rewards,
                'validation_scores': self.validation_scores
            }
        }, filepath)
        
        print(f"✓ Enhanced DQN agent saved to {filepath}")

    def load_agent(self, filepath: str):
        """Load complete agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore hyperparameters
        hyper = checkpoint['hyperparameters']
        self.alpha_values = hyper['alpha_values']
        self.epsilon = hyper['epsilon']
        self.training_step = hyper['training_step']
        self.best_validation_score = hyper['best_validation_score']
        
        # Restore training history
        history = checkpoint['training_history']
        self.reward_history = history['reward_history']
        self.loss_history = history['loss_history']
        self.episode_rewards = history['episode_rewards']
        self.validation_scores = history['validation_scores']
        
        print(f"✓ Enhanced DQN agent loaded from {filepath}")