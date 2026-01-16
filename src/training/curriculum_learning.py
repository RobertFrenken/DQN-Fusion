"""
Curriculum Learning for Extreme Class Imbalance

Start with balanced data, gradually increase difficulty.
"""

import torch
import numpy as np
from typing import Dict, List
import lightning.pytorch as pl


class CurriculumSampler:
    """
    Gradually increases normal:attack ratio during training.
    Starts balanced, ends at realistic distribution.
    """
    
    def __init__(self, start_ratio: float = 1.0, end_ratio: float = 100.0, 
                 total_epochs: int = 200):
        """
        Args:
            start_ratio: Initial normal:attack ratio (1:1)
            end_ratio: Final normal:attack ratio (100:1)
            total_epochs: Total training epochs
        """
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.total_epochs = total_epochs
        
    def get_current_ratio(self, epoch: int) -> float:
        """Get current sampling ratio for this epoch."""
        progress = min(epoch / self.total_epochs, 1.0)
        
        # Exponential curriculum (slow start, rapid increase)
        ratio = self.start_ratio * (self.end_ratio / self.start_ratio) ** progress
        
        return ratio
    
    def sample_batch(self, normal_graphs: List, attack_graphs: List, 
                    epoch: int, batch_size: int = 1024) -> List:
        """Sample batch according to curriculum."""
        current_ratio = self.get_current_ratio(epoch)
        
        n_attacks = min(len(attack_graphs), batch_size // (1 + int(current_ratio)))
        n_normals = int(n_attacks * current_ratio)
        
        # Sample attacks and normals
        sampled_attacks = np.random.choice(attack_graphs, n_attacks, replace=False)
        sampled_normals = np.random.choice(normal_graphs, n_normals, replace=False)
        
        return list(sampled_attacks) + list(sampled_normals)


class NoveltyAwareLoss(torch.nn.Module):
    """
    Loss that emphasizes learning from novel/difficult examples.
    """
    
    def __init__(self, memory_size: int = 1000, novelty_weight: float = 2.0):
        super().__init__()
        self.memory_size = memory_size
        self.novelty_weight = novelty_weight
        self.feature_memory = []
        
    def compute_novelty_scores(self, features: torch.Tensor) -> torch.Tensor:
        """Compute how novel each example is compared to seen examples."""
        if len(self.feature_memory) == 0:
            return torch.ones(features.size(0))
            
        # Compare to stored features
        memory_features = torch.stack(self.feature_memory)
        similarities = torch.cosine_similarity(
            features.unsqueeze(1), memory_features.unsqueeze(0), dim=2
        )
        
        # Novelty = 1 - max_similarity
        max_similarities = similarities.max(dim=1)[0]
        novelty_scores = 1.0 - max_similarities
        
        return novelty_scores
    
    def update_memory(self, features: torch.Tensor):
        """Update feature memory with new examples."""
        self.feature_memory.extend(features.detach().cpu())
        
        # Keep memory size bounded
        if len(self.feature_memory) > self.memory_size:
            self.feature_memory = self.feature_memory[-self.memory_size:]
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                features: torch.Tensor) -> torch.Tensor:
        """Compute novelty-aware loss."""
        # Base loss
        base_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        # Novelty weighting
        novelty_scores = self.compute_novelty_scores(features)
        novelty_weights = 1.0 + self.novelty_weight * novelty_scores
        
        # Update memory
        self.update_memory(features)
        
        # Weighted loss
        weighted_loss = base_loss * novelty_weights
        
        return weighted_loss.mean()