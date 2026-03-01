import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fusion Agent ABC
# ---------------------------------------------------------------------------


class FusionAgent(ABC):
    """Abstract base class for fusion agents.

    All fusion agents operate on the same N-D state vector produced by
    the model registry's extractors (VGAE 8-D + GAT 7-D = 15-D).
    """

    @abstractmethod
    def train_on_cache(
        self,
        train_states: torch.Tensor,
        train_labels: torch.Tensor,
        val_states: torch.Tensor,
        val_labels: torch.Tensor,
        cfg,
    ) -> float:
        """Train the agent on cached predictions. Returns best validation accuracy."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Return serializable state dict for checkpointing."""

    @abstractmethod
    def fuse(self, state_features: np.ndarray) -> int:
        """Given a state vector, return a fused binary prediction (0 or 1)."""


class QNetwork(nn.Module):
    """Q-network with configurable depth and width."""

    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, cfg) -> "QNetwork":
        """Construct from a PipelineConfig."""
        from .registry import fusion_state_dim

        return cls(
            state_dim=fusion_state_dim(),
            action_dim=cfg.fusion.alpha_steps,
            hidden_dim=cfg.dqn.hidden,
            num_layers=cfg.dqn.layers,
        )

    def forward(self, x):
        return self.net(x)


class EnhancedDQNFusionAgent:
    """
    Enhanced DQN agent for dynamic fusion of GAT and VGAE outputs.
    Includes Double DQN, proper target updates, and validation.
    """

    def __init__(
        self,
        alpha_steps=21,
        lr=1e-3,
        gamma=0.9,
        epsilon=0.2,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        buffer_size=50000,
        batch_size=128,
        target_update_freq=100,
        device="cpu",
        *,
        state_dim,
        hidden_dim=128,
        num_layers=3,
    ):

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
        self.q_network = QNetwork(state_dim, self.action_dim, hidden_dim, num_layers).to(
            self.device
        )
        self.target_network = QNetwork(state_dim, self.action_dim, hidden_dim, num_layers).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer and loss
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=1000, factor=0.8
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability

        # Experience replay
        self.replay_buffer = deque(maxlen=self.buffer_size)

        # Training tracking
        self.training_step = 0
        self.update_counter = 0
        self.reward_history = []
        self.loss_history = []

        # Validation tracking
        self.validation_scores = []
        self.best_validation_score = -float("inf")
        self.patience_counter = 0
        self.max_patience = 5000

        # Derive feature indices from registry (no hardcoded offsets)
        from .registry import feature_layout

        layout = feature_layout()
        vgae_start, vgae_dim, _ = layout["vgae"]
        gat_start, gat_dim, _ = layout["gat"]
        self._confidence_indices = [layout[n][2] for n in layout]
        self._vgae_error_slice = slice(vgae_start, vgae_start + 3)
        self._gat_logit_slice = slice(gat_start, gat_start + 2)
        self._vgae_conf_idx = layout["vgae"][2]
        self._gat_conf_idx = layout["gat"][2]

        log.info("DQN Agent initialized: %d actions, state_dim=%d", alpha_steps, self.state_dim)

    def normalize_state(self, state_features: np.ndarray) -> np.ndarray:
        """Normalize state representation (dimension from registry).

        Returns:
            Normalized state as float32 array
        """
        if not isinstance(state_features, np.ndarray):
            state_features = np.array(state_features, dtype=np.float32)

        if len(state_features) != self.state_dim:
            raise ValueError(f"Expected {self.state_dim}D state, got {len(state_features)}D")

        state_features = state_features.copy()

        # Clip confidence values to [0, 1]
        for idx in self._confidence_indices:
            state_features[idx] = np.clip(state_features[idx], 0.0, 1.0)

        return state_features.astype(np.float32)

    def select_action(
        self, state_features: np.ndarray, training: bool = True
    ) -> Tuple[float, int, np.ndarray]:
        """Select action using epsilon-greedy policy.

        Args:
            state_features: N-D state (dimension from registry)
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

    def _derive_scores(self, state_features: np.ndarray) -> Tuple[float, float]:
        """Derive anomaly_score and gat_prob from state features.

        Returns:
            (anomaly_score, gat_prob) tuple
        """
        # VGAE errors - use weighted combination
        vgae_errors = state_features[self._vgae_error_slice]
        vgae_weights = np.array([0.4, 0.35, 0.25])
        anomaly_score = float(np.clip(np.sum(vgae_errors * vgae_weights), 0.0, 1.0))

        # GAT logits - numerically stable softmax
        gat_logits = state_features[self._gat_logit_slice]
        shifted = gat_logits - np.max(gat_logits)
        gat_probs = np.exp(shifted) / np.sum(np.exp(shifted))
        gat_prob = float(gat_probs[1])  # Probability of attack class

        return anomaly_score, gat_prob

    def compute_fusion_reward(
        self, prediction: int, true_label: int, state_features: np.ndarray, alpha: float
    ) -> float:
        """Enhanced reward function.

        Args:
            prediction: Fused prediction (0 or 1)
            true_label: Ground truth label (0 or 1)
            state_features: N-D state (dimension from registry)
            alpha: Fusion weight selected by DQN

        Returns:
            Reward scalar
        """
        anomaly_score, gat_prob = self._derive_scores(state_features)

        # Confidence scores
        vgae_confidence = float(state_features[self._vgae_conf_idx])
        gat_confidence = float(state_features[self._gat_conf_idx])
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

    def store_experience(
        self, state: np.ndarray, action_idx: int, reward: float, next_state: np.ndarray, done: bool
    ):
        """Store experience in replay buffer."""
        state = state.astype(np.float32) if state.dtype != np.float32 else state
        next_state = next_state.astype(np.float32) if next_state.dtype != np.float32 else next_state

        experience = (state, action_idx, reward, next_state, done)
        self.replay_buffer.append(experience)

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
            next_q_values = (
                self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            )
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
        """Validate agent performance.

        Args:
            validation_data: List of (state_features, true_label) tuples
            num_samples: Number of samples to validate on

        Returns:
            Dict with validation metrics
        """
        self.q_network.eval()

        correct = 0
        total_reward = 0
        alpha_values_used = []

        sample_data = (
            validation_data[:num_samples]
            if len(validation_data) >= num_samples
            else validation_data
        )

        if not sample_data:
            self.q_network.train()
            return {"accuracy": 0.0, "avg_reward": 0.0, "avg_alpha": 0.0, "alpha_std": 0.0}

        for state_features, true_label in sample_data:
            # Select action
            alpha, _, _ = self.select_action(state_features, training=False)
            alpha_values_used.append(alpha)

            # Derive scalar scores for fusion
            anomaly_score, gat_prob = self._derive_scores(state_features)

            # Make fusion prediction
            fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
            prediction = 1 if fused_score > 0.5 else 0

            # Calculate metrics
            correct += prediction == true_label
            reward = self.compute_fusion_reward(prediction, true_label, state_features, alpha)
            total_reward += reward

        self.q_network.train()

        validation_results = {
            "accuracy": correct / len(sample_data),
            "avg_reward": total_reward / len(sample_data),
            "avg_alpha": np.mean(alpha_values_used),
            "alpha_std": np.std(alpha_values_used),
        }

        # Update learning rate based on validation
        self.scheduler.step(validation_results["avg_reward"])

        # Early stopping check
        current_score = validation_results["accuracy"] + 0.1 * validation_results["avg_reward"]
        if current_score > self.best_validation_score:
            self.best_validation_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        self.validation_scores.append(validation_results)
        return validation_results


# ---------------------------------------------------------------------------
# MLP Fusion Agent
# ---------------------------------------------------------------------------


class MLPFusionNetwork(nn.Module):
    """Simple MLP for binary classification from fusion state vectors."""

    def __init__(self, state_dim: int, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPFusionAgent(FusionAgent):
    """Supervised MLP baseline: learns binary classification directly from state vectors.

    Same 15-D state as DQN, but trained with BCE loss instead of RL episodes.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        lr: float = 0.001,
        device: str = "cpu",
    ):
        self.device = device
        self.model = MLPFusionNetwork(state_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train_on_cache(self, train_states, train_labels, val_states, val_labels, cfg) -> float:
        max_epochs = cfg.fusion.mlp_max_epochs
        batch_size = cfg.dqn.batch_size
        best_acc = 0.0

        for epoch in range(max_epochs):
            self.model.train()
            idx = torch.randperm(len(train_states))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(train_states), batch_size):
                batch_idx = idx[start : start + batch_size]
                states = train_states[batch_idx].to(self.device)
                labels = train_labels[batch_idx].float().to(self.device)

                logits = self.model(states)
                loss = self.loss_fn(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            if (epoch + 1) % 10 == 0:
                acc = self._evaluate(val_states, val_labels)
                log.info(
                    "MLP epoch %d/%d  loss=%.4f  val_acc=%.4f",
                    epoch + 1,
                    max_epochs,
                    epoch_loss / max(n_batches, 1),
                    acc,
                )
                if acc > best_acc:
                    best_acc = acc

        return best_acc

    def _evaluate(self, states: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(states.to(self.device))
            preds = (logits > 0).long()
            correct = (preds == labels.to(self.device)).sum().item()
        return correct / len(labels)

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict()}

    def fuse(self, state_features: np.ndarray) -> int:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            logit = self.model(t)
            return 1 if logit.item() > 0 else 0


# ---------------------------------------------------------------------------
# Weighted Average Fusion Agent
# ---------------------------------------------------------------------------


class WeightedAvgFusionAgent(FusionAgent):
    """Simplest baseline: learns a single scalar alpha per model.

    If this matches DQN's F1, the RL approach is unjustified.
    Fusion: score = (1 - sigmoid(w)) * vgae_conf + sigmoid(w) * gat_conf
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.weight = nn.Parameter(torch.zeros(1, device=device))
        self.optimizer = optim.Adam([self.weight], lr=0.01)
        self.loss_fn = nn.BCELoss()

        from .registry import feature_layout

        layout = feature_layout()
        self._vgae_conf_idx = layout["vgae"][2]
        self._gat_conf_idx = layout["gat"][2]

    def train_on_cache(self, train_states, train_labels, val_states, val_labels, cfg) -> float:
        max_epochs = cfg.fusion.mlp_max_epochs
        best_acc = 0.0

        for epoch in range(max_epochs):
            alpha = torch.sigmoid(self.weight)
            vgae_conf = train_states[:, self._vgae_conf_idx].to(self.device)
            gat_conf = train_states[:, self._gat_conf_idx].to(self.device)
            scores = (1 - alpha) * vgae_conf + alpha * gat_conf
            scores = torch.clamp(scores, 1e-7, 1 - 1e-7)
            labels = train_labels.float().to(self.device)

            loss = self.loss_fn(scores, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                acc = self._evaluate(val_states, val_labels)
                a = torch.sigmoid(self.weight).item()
                log.info(
                    "WeightedAvg epoch %d/%d  loss=%.4f  val_acc=%.4f  alpha=%.3f",
                    epoch + 1,
                    max_epochs,
                    loss.item(),
                    acc,
                    a,
                )
                if acc > best_acc:
                    best_acc = acc

        return best_acc

    def _evaluate(self, states: torch.Tensor, labels: torch.Tensor) -> float:
        with torch.no_grad():
            alpha = torch.sigmoid(self.weight)
            vgae_conf = states[:, self._vgae_conf_idx].to(self.device)
            gat_conf = states[:, self._gat_conf_idx].to(self.device)
            scores = (1 - alpha) * vgae_conf + alpha * gat_conf
            preds = (scores > 0.5).long()
            correct = (preds == labels.to(self.device)).sum().item()
        return correct / len(labels)

    def state_dict(self) -> dict:
        return {"weight": self.weight.detach().cpu(), "alpha": torch.sigmoid(self.weight).item()}

    def fuse(self, state_features: np.ndarray) -> int:
        with torch.no_grad():
            alpha = torch.sigmoid(self.weight).item()
            vgae_conf = state_features[self._vgae_conf_idx]
            gat_conf = state_features[self._gat_conf_idx]
            score = (1 - alpha) * vgae_conf + alpha * gat_conf
            return 1 if score > 0.5 else 0
