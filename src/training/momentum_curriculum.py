"""
Momentum-Based Curriculum Scheduler

Implements a momentum-based curriculum learning schedule that adjusts
the difficulty ratio based on model confidence and training progress.
"""

import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class MomentumCurriculumScheduler:
    """
    Momentum-based curriculum scheduler that smoothly adjusts difficulty ratios
    based on model confidence and training progress.
    
    The scheduler uses exponential moving average (momentum) to smooth ratio changes
    and can adapt based on the model's classification confidence.
    
    Args:
        total_epochs: Total number of training epochs
        initial_ratio: Starting normal:attack ratio (e.g., 1.0 for 1:1)
        target_ratio: Target normal:attack ratio at end of training
        momentum: Momentum coefficient for smoothing (0.0 to 1.0)
        confidence_threshold: Model confidence threshold for ratio adjustment
        warmup_epochs: Number of warmup epochs with fixed initial ratio
    """
    
    def __init__(
        self,
        total_epochs: int,
        initial_ratio: float = 1.0,
        target_ratio: float = 4.0,
        momentum: float = 0.9,
        confidence_threshold: float = 0.75,
        warmup_epochs: int = 10
    ):
        self.total_epochs = total_epochs
        self.initial_ratio = initial_ratio
        self.target_ratio = target_ratio
        self.momentum = momentum
        self.confidence_threshold = confidence_threshold
        self.warmup_epochs = warmup_epochs
        
        # Internal state
        self.current_ratio = initial_ratio
        self.running_avg_ratio = initial_ratio
        self.history = []
        
    def update_ratio(
        self, 
        epoch: int, 
        model_confidence: float = 0.5
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Update curriculum ratio based on epoch and model confidence.
        
        Args:
            epoch: Current training epoch (0-indexed)
            model_confidence: Model's classification confidence (0.0 to 1.0)
            
        Returns:
            Tuple of (new_ratio, metrics_dict)
        """
        # During warmup, use initial ratio
        if epoch < self.warmup_epochs:
            metrics = {
                'phase': 'warmup',
                'progress': epoch / self.warmup_epochs,
                'target_ratio': self.initial_ratio,
                'actual_ratio': self.initial_ratio,
                'momentum_ratio': self.initial_ratio,
            }
            self.history.append(metrics)
            return self.initial_ratio, metrics
        
        # Compute linear progress after warmup
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs
        progress = min(1.0, adjusted_epoch / max(1, adjusted_total))
        
        # Compute base target ratio (linear interpolation)
        base_ratio = self.initial_ratio + progress * (self.target_ratio - self.initial_ratio)
        
        # Adjust based on model confidence
        # If confidence is high, we can increase difficulty faster
        # If confidence is low, slow down the difficulty increase
        confidence_factor = 1.0
        if model_confidence > self.confidence_threshold:
            # Model is confident - slightly accelerate
            confidence_factor = 1.0 + 0.1 * (model_confidence - self.confidence_threshold)
        elif model_confidence < self.confidence_threshold * 0.8:
            # Model is struggling - slow down
            confidence_factor = 0.9
        
        target_ratio = base_ratio * confidence_factor
        target_ratio = max(self.initial_ratio, min(self.target_ratio, target_ratio))
        
        # Apply momentum smoothing
        self.running_avg_ratio = (
            self.momentum * self.running_avg_ratio + 
            (1 - self.momentum) * target_ratio
        )
        
        # Update current ratio
        self.current_ratio = self.running_avg_ratio
        
        metrics = {
            'phase': 'curriculum',
            'progress': progress,
            'target_ratio': target_ratio,
            'actual_ratio': self.current_ratio,
            'momentum_ratio': self.running_avg_ratio,
            'confidence_factor': confidence_factor,
            'model_confidence': model_confidence,
        }
        self.history.append(metrics)
        
        return self.current_ratio, metrics
    
    def get_state(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_ratio': self.current_ratio,
            'running_avg_ratio': self.running_avg_ratio,
            'history': self.history,
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_ratio = state.get('current_ratio', self.initial_ratio)
        self.running_avg_ratio = state.get('running_avg_ratio', self.initial_ratio)
        self.history = state.get('history', [])
