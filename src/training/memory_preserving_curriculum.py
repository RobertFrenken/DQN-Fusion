"""
Memory-Preserving Curriculum Learning with Elastic Weight Consolidation

Implements EWC (Elastic Weight Consolidation) to prevent catastrophic forgetting
during curriculum learning as the training distribution shifts over time.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting
    during curriculum learning.
    
    EWC adds a regularization term that penalizes changes to parameters that
    were important for previous tasks (earlier curriculum phases).
    
    Args:
        model: The neural network model to protect
        fisher_samples: Number of samples to use for Fisher matrix estimation
        ewc_lambda: Strength of the EWC penalty (higher = more preservation)
    """
    
    def __init__(
        self, 
        model: nn.Module,
        fisher_samples: int = 200,
        ewc_lambda: float = 0.4
    ):
        self.model = model
        self.fisher_samples = fisher_samples
        self.ewc_lambda = ewc_lambda
        
        # Storage for Fisher information and optimal parameters
        self.fisher: Dict[str, Tensor] = {}
        self.optimal_params: Dict[str, Tensor] = {}
        self.consolidated = False
        
    def consolidate(self, dataloader) -> None:
        """
        Compute Fisher information matrix and store optimal parameters.
        
        Call this at the end of each curriculum phase to "consolidate"
        what the model has learned.
        
        Args:
            dataloader: DataLoader for computing Fisher information
        """
        self.model.eval()
        
        # Store current parameters as optimal
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        # Initialize Fisher information
        self.fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        # Compute Fisher information using gradient of log-likelihood
        sample_count = 0
        for batch in dataloader:
            if sample_count >= self.fisher_samples:
                break
                
            self.model.zero_grad()
            
            # Forward pass
            try:
                if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                    # PyG batch - pass the entire batch object (not separate tensors)
                    # Models like GATWithJK expect the full Data/Batch object
                    output = self.model(batch)
                    if isinstance(output, tuple):
                        output = output[0]  # Get logits
                    
                    # Use log probability of predicted class
                    if output.dim() > 1:
                        log_probs = torch.log_softmax(output, dim=-1)
                        predicted = output.argmax(dim=-1)
                        loss = -log_probs.gather(1, predicted.unsqueeze(1)).mean()
                    else:
                        loss = output.mean()
                else:
                    # Standard batch
                    output = self.model(batch)
                    if isinstance(output, tuple):
                        output = output[0]
                    loss = output.mean()
                    
                loss.backward()
                
                # Accumulate squared gradients (Fisher approximation)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.fisher[name] += param.grad.data.clone().pow(2)
                        
                sample_count += batch.y.size(0) if hasattr(batch, 'y') else 1
                
            except Exception as e:
                logger.warning(f"EWC consolidation failed on batch: {e}")
                continue
        
        # Normalize Fisher information
        for name in self.fisher:
            self.fisher[name] /= max(1, sample_count)
        
        self.consolidated = True
        logger.info(f"EWC consolidated over {sample_count} samples")
    
    def penalty(self) -> Tensor:
        """
        Compute EWC penalty for current parameters.
        
        Returns:
            EWC regularization loss (scalar tensor)
        """
        if not self.consolidated:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.optimal_params:
                # Penalize deviation from optimal parameters weighted by Fisher
                diff = param - self.optimal_params[name]
                penalty += (self.fisher[name] * diff.pow(2)).sum()
        
        return self.ewc_lambda * penalty
    
    def get_state(self) -> Dict:
        """Get EWC state for checkpointing."""
        return {
            'fisher': {k: v.cpu() for k, v in self.fisher.items()},
            'optimal_params': {k: v.cpu() for k, v in self.optimal_params.items()},
            'consolidated': self.consolidated,
        }
    
    def load_state(self, state: Dict, device: Optional[torch.device] = None) -> None:
        """Load EWC state from checkpoint."""
        device = device or next(self.model.parameters()).device
        self.fisher = {k: v.to(device) for k, v in state.get('fisher', {}).items()}
        self.optimal_params = {k: v.to(device) for k, v in state.get('optimal_params', {}).items()}
        self.consolidated = state.get('consolidated', False)
