import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Tuple, Dict, Optional
import pickle
import os

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNFusionAgent:
    """
    DQN agent for dynamic fusion of GAT and VGAE outputs.
    Uses a neural network to approximate Q-values for discrete fusion weights.
    """
    def __init__(self, alpha_steps=21, lr=1e-3, gamma=0.9, epsilon=0.1,
                 epsilon_decay=0.995, min_epsilon=0.01, buffer_size=10000, batch_size=64, device='cpu'):
        self.alpha_values = np.linspace(0, 1, alpha_steps)
        self.action_dim = alpha_steps
        self.state_dim = 2  # anomaly_score, gat_prob
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device

        self.q_network = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.reward_history = []
        self.accuracy_history = []

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, anomaly_score: float, gat_prob: float, training: bool = True) -> Tuple[float, int, np.ndarray]:
        state = np.array([anomaly_score, gat_prob], dtype=np.float32)
        state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
        if training and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = torch.argmax(q_values).item()
        alpha_value = self.alpha_values[action_idx]
        return alpha_value, action_idx, state

    def compute_reward(self, prediction: int, true_label: int, anomaly_score: float, gat_prob: float, confidence_bonus: bool = True) -> float:
        base_reward = 1.0 if prediction == true_label else -1.0
        if not confidence_bonus:
            return base_reward
        if prediction == true_label:
            confidence = max(anomaly_score, gat_prob) if prediction == 1 else 1.0 - max(anomaly_score, gat_prob)
            confidence_reward = 0.5 * confidence
            return base_reward + confidence_reward
        else:
            overconfidence = max(anomaly_score, gat_prob) if prediction == 1 else 1.0 - min(anomaly_score, gat_prob)
            confidence_penalty = -0.5 * overconfidence
            return base_reward + confidence_penalty

    def store_experience(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.replay_buffer[idx] for idx in batch))

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reward_history.append(rewards.mean().item())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_policy_summary(self, n_samples=100):
        # Sample states uniformly and show best alpha for each
        policy = []
        for a in np.linspace(0, 1, n_samples):
            for g in np.linspace(0, 1, n_samples):
                state = torch.tensor([a, g], dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_network(state)
                    best_action = torch.argmax(q_values).item()
                policy.append((a, g, self.alpha_values[best_action]))
        return policy

    def save_agent(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'alpha_values': self.alpha_values,
            'epsilon': self.epsilon,
            'reward_history': self.reward_history,
            'accuracy_history': self.accuracy_history
        }, filepath)
        print(f"✓ DQN agent saved to {filepath}")

    def load_agent(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.alpha_values = checkpoint['alpha_values']
        self.epsilon = checkpoint['epsilon']
        self.reward_history = checkpoint['reward_history']
        self.accuracy_history = checkpoint['accuracy_history']
        print(f"✓ DQN agent loaded from {filepath}")