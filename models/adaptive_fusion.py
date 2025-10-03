import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import pickle
import os

class QFusionAgent:
    """
    Q-learning agent for dynamic fusion of GAT and VGAE outputs.
    
    This agent learns optimal fusion weights based on the confidence levels
    of both the anomaly detection (VGAE) and classification (GAT) components.
    """
    
    def __init__(self, alpha_steps: int = 21, state_bins: int = 10, 
                 lr: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        """
        Initialize Q-learning fusion agent.
        
        Args:
            alpha_steps: Number of discrete fusion weights (0.0, 0.05, ..., 1.0)
            state_bins: Number of bins for discretizing state features
            lr: Learning rate for Q-learning updates
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon value
        """
        self.alpha_values = np.linspace(0, 1, alpha_steps)
        self.state_bins = state_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: [anomaly_score_bin, gat_prob_bin, alpha_action]
        self.Q = np.zeros((state_bins, state_bins, alpha_steps))
        
        # Experience tracking
        self.experience_buffer = []
        self.reward_history = []
        self.accuracy_history = []
        
    def discretize_state(self, anomaly_score: float, gat_prob: float) -> Tuple[int, int]:
        """Discretize continuous scores into bins."""
        # Clip to [0, 1] range and discretize
        anomaly_score = np.clip(anomaly_score, 0, 1)
        gat_prob = np.clip(gat_prob, 0, 1)
        
        a_bin = min(int(anomaly_score * self.state_bins), self.state_bins - 1)
        g_bin = min(int(gat_prob * self.state_bins), self.state_bins - 1)
        
        return a_bin, g_bin
    
    def select_action(self, anomaly_score: float, gat_prob: float, 
                     training: bool = True) -> Tuple[float, int, Tuple[int, int]]:
        """
        Select fusion weight using epsilon-greedy policy.
        
        Args:
            anomaly_score: Normalized anomaly score [0, 1]
            gat_prob: GAT probability [0, 1]
            training: Whether in training mode (affects exploration)
            
        Returns:
            Tuple of (alpha_value, action_index, state_bins)
        """
        a_bin, g_bin = self.discretize_state(anomaly_score, gat_prob)
        
        if training and np.random.rand() < self.epsilon:
            # Exploration: random action
            action_idx = np.random.randint(len(self.alpha_values))
        else:
            # Exploitation: best known action
            action_idx = np.argmax(self.Q[a_bin, g_bin])
        
        alpha_value = self.alpha_values[action_idx]
        return alpha_value, action_idx, (a_bin, g_bin)
    
    def compute_reward(self, prediction: int, true_label: int, 
                      anomaly_score: float, gat_prob: float, 
                      confidence_bonus: bool = True) -> float:
        """
        Compute reward based on prediction correctness and confidence.
        
        Args:
            prediction: Model prediction (0 or 1)
            true_label: Ground truth label (0 or 1)
            anomaly_score: Anomaly detection score
            gat_prob: GAT probability
            confidence_bonus: Whether to add confidence-based bonus
            
        Returns:
            Reward value
        """
        # Base reward for correctness
        base_reward = 1.0 if prediction == true_label else -1.0
        
        if not confidence_bonus:
            return base_reward
        
        # Confidence bonus: reward high-confidence correct predictions more
        if prediction == true_label:
            # For correct predictions, reward confidence
            if prediction == 1:  # Attack correctly identified
                confidence = max(anomaly_score, gat_prob)
            else:  # Normal correctly identified  
                confidence = 1.0 - max(anomaly_score, gat_prob)
            
            confidence_reward = 0.5 * confidence
            return base_reward + confidence_reward
        else:
            # For incorrect predictions, penalize overconfidence
            if prediction == 1:  # False positive
                overconfidence = max(anomaly_score, gat_prob)
            else:  # False negative
                overconfidence = 1.0 - min(anomaly_score, gat_prob)
            
            confidence_penalty = -0.5 * overconfidence
            return base_reward + confidence_penalty
    
    def update_q_table(self, state: Tuple[int, int], action_idx: int, 
                      reward: float, next_state: Optional[Tuple[int, int]] = None):
        """Update Q-table using Q-learning rule."""
        a_bin, g_bin = state
        
        if next_state is not None:
            next_a_bin, next_g_bin = next_state
            best_next_q = np.max(self.Q[next_a_bin, next_g_bin])
        else:
            best_next_q = 0.0  # Terminal state
        
        # Q-learning update
        current_q = self.Q[a_bin, g_bin, action_idx]
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - current_q
        
        self.Q[a_bin, g_bin, action_idx] += self.lr * td_error
        
        # Track experience
        self.reward_history.append(reward)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_policy_summary(self) -> Dict:
        """Get summary of learned policy."""
        policy_matrix = np.zeros((self.state_bins, self.state_bins))
        
        for i in range(self.state_bins):
            for j in range(self.state_bins):
                best_action = np.argmax(self.Q[i, j])
                policy_matrix[i, j] = self.alpha_values[best_action]
        
        return {
            'policy_matrix': policy_matrix,
            'q_table': self.Q.copy(),
            'avg_recent_reward': np.mean(self.reward_history[-100:]) if self.reward_history else 0.0,
            'total_experiences': len(self.reward_history),
            'current_epsilon': self.epsilon
        }
    
    def save_agent(self, filepath: str):
        """Save agent state to file."""
        agent_state = {
            'Q': self.Q,
            'alpha_values': self.alpha_values,
            'state_bins': self.state_bins,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'reward_history': self.reward_history,
            'accuracy_history': self.accuracy_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(agent_state, f)
        print(f"✓ Q-learning agent saved to {filepath}")
    
    def load_agent(self, filepath: str):
        """Load agent state from file."""
        with open(filepath, 'rb') as f:
            agent_state = pickle.load(f)
        
        self.Q = agent_state['Q']
        self.alpha_values = agent_state['alpha_values']
        self.state_bins = agent_state['state_bins']
        self.lr = agent_state['lr']
        self.gamma = agent_state['gamma']
        self.epsilon = agent_state['epsilon']
        self.reward_history = agent_state['reward_history']
        self.accuracy_history = agent_state.get('accuracy_history', [])
        
        print(f"✓ Q-learning agent loaded from {filepath}")