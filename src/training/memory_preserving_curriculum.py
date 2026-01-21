"""
Memory-Preserving Curriculum Learning
Prevents catastrophic forgetting during curriculum transitions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import lightning.pytorch as pl
from collections import deque


class ElasticWeightConsolidation:
    """
    Protect important weights learned during balanced phase.
    Based on Kirkpatrick et al. 2017 "Overcoming catastrophic forgetting"
    """
    
    def __init__(self, model: torch.nn.Module, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_information = {}
        self.optimal_params = {}
        
    def compute_fisher_information(self, dataloader, device='cuda'):
        """Compute Fisher Information Matrix after balanced training phase."""
        self.model.eval()
        
        # Store current parameters as optimal
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()
        
        # Compute Fisher Information
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param.data)
        
        for batch in dataloader:
            batch = batch.to(device)
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(batch)
            
            # Handle different output shapes (GAT outputs [batch, 2], targets are [batch])
            if output.dim() == 2 and output.size(1) == 2:
                # Multi-class output: use cross_entropy
                loss = F.cross_entropy(output, batch.y.long())
            else:
                # Single output: use binary_cross_entropy_with_logits
                loss = F.binary_cross_entropy_with_logits(output.squeeze(), batch.y.float())
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients (Fisher approximation)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Average over dataset
        for name in fisher:
            fisher[name] /= len(dataloader)
            
        self.fisher_information = fisher
        print(f"✅ Fisher Information computed for {len(fisher)} parameter groups")
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC penalty to preserve important weights."""
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                # Penalty = Fisher * (current_param - optimal_param)^2
                penalty = self.fisher_information[name] * (param - self.optimal_params[name]) ** 2
                ewc_loss += penalty.sum()
        
        return self.ewc_lambda * ewc_loss


class ExperienceReplayBuffer:
    """
    Maintain a buffer of balanced examples throughout training.
    """
    
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add_samples(self, samples: List):
        """Add samples to replay buffer."""
        for sample in samples:
            self.buffer.append(sample)
    
    def sample_balanced_batch(self, batch_size: int = 64) -> List:
        """Sample a balanced batch from replay buffer."""
        if len(self.buffer) == 0:
            return []
        
        # Separate normal and attack samples in buffer
        normal_samples = [s for s in self.buffer if s.y.item() == 0]
        attack_samples = [s for s in self.buffer if s.y.item() == 1]
        
        if len(attack_samples) == 0 or len(normal_samples) == 0:
            # Not enough diversity, return random sample
            return np.random.choice(list(self.buffer), min(batch_size, len(self.buffer)), replace=False).tolist()
        
        # Sample balanced batch
        n_per_class = batch_size // 2
        
        sampled_normal = np.random.choice(normal_samples, min(n_per_class, len(normal_samples)), replace=False).tolist()
        sampled_attack = np.random.choice(attack_samples, min(n_per_class, len(attack_samples)), replace=False).tolist()
        
        return sampled_normal + sampled_attack


class MemoryPreservingCurriculumLoss(torch.nn.Module):
    """
    Loss function that combines curriculum learning with memory preservation.
    """
    
    def __init__(self, ewc_weight: float = 1000.0, replay_weight: float = 0.5):
        super().__init__()
        self.ewc = None
        self.replay_buffer = ExperienceReplayBuffer()
        self.ewc_weight = ewc_weight
        self.replay_weight = replay_weight
        
    def set_ewc(self, ewc: ElasticWeightConsolidation):
        """Set EWC component after balanced training phase."""
        self.ewc = ewc
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                epoch: int, total_epochs: int,
                replay_batch: List = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute memory-preserving curriculum loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels  
            epoch: Current epoch
            total_epochs: Total training epochs
            replay_batch: Optional replay samples for memory preservation
            
        Returns:
            Total loss and loss components dict
        """
        losses = {}
        
        # Base classification loss - handle different output shapes
        if predictions.dim() == 2 and predictions.size(1) == 2:
            # Multi-class output: use cross_entropy
            base_loss = F.cross_entropy(predictions, targets.long())
        else:
            # Single output: use binary_cross_entropy_with_logits
            base_loss = F.binary_cross_entropy_with_logits(predictions.squeeze(), targets.float())
            
        losses['base_loss'] = base_loss
        
        total_loss = base_loss
        
        # EWC loss (after balanced training phase)
        if self.ewc is not None and epoch > total_epochs * 0.2:  # After 20% of training
            ewc_loss = self.ewc.compute_ewc_loss()
            losses['ewc_loss'] = ewc_loss
            total_loss += ewc_loss
            
        # Replay loss (maintain performance on balanced examples)
        if replay_batch is not None and len(replay_batch) > 0:
            # This would need to be computed in the training step
            # replay_loss = self._compute_replay_loss(replay_batch)
            # losses['replay_loss'] = replay_loss  
            # total_loss += self.replay_weight * replay_loss
            pass
            
        return total_loss, losses


class MemoryPreservingCurriculumModule(pl.LightningModule):
    """
    Lightning module with memory-preserving curriculum learning.
    """
    
    def __init__(self, model, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.curriculum_loss = MemoryPreservingCurriculumLoss()
        self.ewc = None
        
        # Track training phases
        self.balanced_phase_complete = False
        self.transition_epoch = None
        
    def training_step(self, batch, batch_idx):
        # Forward pass
        predictions = self.model(batch)
        
        # Check if we're transitioning from balanced to imbalanced
        current_epoch = self.current_epoch
        total_epochs = self.trainer.max_epochs
        
        # Initialize EWC after balanced phase (first 20% of training)
        if not self.balanced_phase_complete and current_epoch > total_epochs * 0.2:
            self.balanced_phase_complete = True
            self.transition_epoch = current_epoch
            self._initialize_ewc()
            
        # Compute loss with memory preservation
        loss, loss_components = self.curriculum_loss(
            predictions, batch.y, current_epoch, total_epochs
        )
        
        # Log components
        for name, value in loss_components.items():
            self.log(f'train_{name}', value)
            
        return loss
    
    def _initialize_ewc(self):
        """Initialize EWC after balanced training phase."""
        print(f"🧠 Initializing EWC at epoch {self.current_epoch} to preserve balanced learning")
        
        # Get current training dataloader for Fisher computation
        train_dataloader = self.trainer.datamodule.train_dataloader()
        
        # Initialize EWC
        self.ewc = ElasticWeightConsolidation(self.model)
        self.ewc.compute_fisher_information(train_dataloader, self.device)
        
        # Set EWC in loss function
        self.curriculum_loss.set_ewc(self.ewc)
        
        print("✅ EWC initialized - important weights are now protected")
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)