"""
Momentum-Based Curriculum Learning Implementation
Provides smooth, adaptive curriculum progression instead of hard phase transitions.

DIFFERENCE FROM CURRENT SYSTEM:
- Current: Hard transitions (1:1 → 5:1 → 100:1 at fixed epochs)  
- Momentum: Smooth exponential decay with adaptive momentum based on learning progress

BENEFITS:
1. No jarring distribution shifts that could cause forgetting
2. Adaptive pacing based on model performance
3. Momentum prevents premature transitions
4. More stable learning curves

RESEARCH BASIS:
- "Momentum-Based Curriculum Learning" (2023)
- Reduces training instability from hard curriculum switches
- Better final performance on imbalanced datasets
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
import logging


class MomentumCurriculumScheduler:
    """
    Smooth momentum-based curriculum progression.
    
    Key Differences from Hard Phase Transitions:
    1. Exponential decay instead of step functions
    2. Momentum-smoothed progress based on model performance
    3. Adaptive pacing prevents premature transitions
    """
    
    def __init__(self, 
                 total_epochs: int,
                 initial_ratio: float = 1.0,      # Start balanced (1:1)
                 target_ratio: float = 0.01,     # End imbalanced (100:1)  
                 momentum: float = 0.9,          # Momentum for smoothing
                 confidence_threshold: float = 0.8,  # Minimum confidence to progress
                 warmup_epochs: int = 50):       # Minimum epochs before progression
        
        self.total_epochs = total_epochs
        self.initial_ratio = initial_ratio
        self.target_ratio = target_ratio
        self.momentum = momentum
        self.confidence_threshold = confidence_threshold
        self.warmup_epochs = warmup_epochs
        
        # State tracking
        self.current_ratio = initial_ratio
        self.momentum_accumulator = 0.0
        self.confidence_history = []
        
        # Compute base decay rate
        self.base_decay_rate = np.log(target_ratio / initial_ratio) / total_epochs
        
        logging.info(f"📈 Momentum Curriculum initialized:")
        logging.info(f"   Ratio progression: {initial_ratio:.2f} → {target_ratio:.4f}")
        logging.info(f"   Momentum: {momentum}, Threshold: {confidence_threshold}")
    
    def update_ratio(self, epoch: int, model_confidence: float) -> Tuple[float, Dict]:
        """
        Update curriculum ratio using momentum-based progression.
        
        Args:
            epoch: Current training epoch
            model_confidence: Model's confidence on normal class (0-1)
            
        Returns:
            Updated normal:attack ratio and progress metrics
        """
        metrics = {}
        
        # Store confidence history
        self.confidence_history.append(model_confidence)
        if len(self.confidence_history) > 10:
            self.confidence_history.pop(0)
        
        # Compute progress signal
        progress_signal = self._compute_progress_signal(epoch, model_confidence)
        metrics['progress_signal'] = progress_signal
        
        # Update momentum accumulator
        self.momentum_accumulator = (
            self.momentum * self.momentum_accumulator + 
            (1 - self.momentum) * progress_signal
        )
        metrics['momentum_accumulator'] = self.momentum_accumulator
        
        # Compute new ratio using momentum-smoothed progress
        if epoch >= self.warmup_epochs:
            # Exponential decay modulated by momentum
            effective_epoch = epoch * (1.0 + self.momentum_accumulator)
            decay_factor = np.exp(self.base_decay_rate * effective_epoch)
            new_ratio = self.initial_ratio * decay_factor
            
            # Clamp to target ratio
            new_ratio = max(new_ratio, self.target_ratio)
            
            # Smooth transition (no sudden jumps)
            ratio_change = new_ratio - self.current_ratio
            max_change = 0.1 * self.current_ratio  # Max 10% change per epoch
            
            if abs(ratio_change) > max_change:
                new_ratio = self.current_ratio + np.sign(ratio_change) * max_change
                
            self.current_ratio = new_ratio
        
        metrics['current_ratio'] = self.current_ratio
        metrics['expected_normal_samples'] = self.current_ratio / (1 + self.current_ratio)
        
        return self.current_ratio, metrics
    
    def _compute_progress_signal(self, epoch: int, confidence: float) -> float:
        """
        Compute how much to accelerate/decelerate curriculum progression.
        
        Positive signal = accelerate (model is learning well)
        Negative signal = decelerate (model is struggling)
        """
        if len(self.confidence_history) < 3:
            return 0.0
        
        # Recent confidence trend
        recent_confidence = np.mean(self.confidence_history[-3:])
        confidence_trend = recent_confidence - np.mean(self.confidence_history[:-3])
        
        # Base progress signal
        if recent_confidence > self.confidence_threshold and confidence_trend >= 0:
            # Model is confident and improving → accelerate
            progress_signal = min(confidence_trend * 2.0, 0.5)
        elif recent_confidence < self.confidence_threshold * 0.7:
            # Model is struggling → decelerate  
            progress_signal = max(-0.3, confidence_trend * 1.5)
        else:
            # Neutral → slight acceleration based on confidence
            progress_signal = (recent_confidence - self.confidence_threshold) * 0.2
            
        return progress_signal


class MomentumCurriculumDataModule:
    """
    Data module using momentum-based curriculum scheduling.
    """
    
    def __init__(self,
                 normal_samples: List,
                 attack_samples: List,
                 batch_size: int = 64,
                 total_epochs: int = 200):
        
        self.normal_samples = normal_samples
        self.attack_samples = attack_samples
        self.batch_size = batch_size
        
        # Initialize momentum scheduler
        self.scheduler = MomentumCurriculumScheduler(
            total_epochs=total_epochs,
            initial_ratio=1.0,    # Start 1:1 balanced
            target_ratio=0.01,    # End 100:1 imbalanced
            momentum=0.9,
            confidence_threshold=0.75
        )
        
        logging.info(f"🔄 Momentum Curriculum Data Module initialized")
        logging.info(f"   {len(normal_samples)} normal, {len(attack_samples)} attack samples")
    
    def get_epoch_batch_composition(self, epoch: int, confidence: float) -> Dict:
        """
        Get batch composition for current epoch using momentum curriculum.
        
        COMPARISON TO HARD TRANSITIONS:
        - Hard: 1:1 → 5:1 → 100:1 (sudden jumps)
        - Momentum: Smooth exponential 1:1 → ... → 100:1 (gradual)
        """
        # Update curriculum ratio
        current_ratio, metrics = self.scheduler.update_ratio(epoch, confidence)
        
        # Calculate samples per batch  
        total_attack_budget = self.batch_size // 2  # Half batch for attacks
        total_normal_budget = self.batch_size - total_attack_budget
        
        # Adjust based on curriculum ratio
        normal_fraction = current_ratio / (1 + current_ratio)
        
        # Smooth allocation
        n_normals = int(total_normal_budget * normal_fraction)
        n_attacks = self.batch_size - n_normals
        
        # Ensure minimum representation
        n_attacks = max(n_attacks, 1)  # Always at least 1 attack sample
        n_normals = self.batch_size - n_attacks
        
        return {
            'n_normal': n_normals,
            'n_attack': n_attacks, 
            'ratio': current_ratio,
            'normal_percentage': (n_normals / self.batch_size) * 100,
            **metrics
        }


def compare_curriculum_approaches():
    """Compare hard transitions vs momentum curriculum."""
    
    print("📊 CURRICULUM APPROACH COMPARISON")
    print("=" * 60)
    
    # Simulate both approaches
    epochs = list(range(0, 200, 10))
    confidence_trajectory = [0.5 + 0.4 * (1 - np.exp(-e/50)) for e in epochs]
    
    # Hard transitions (current system)
    print("🔥 HARD TRANSITIONS (Current System):")
    hard_ratios = []
    for epoch in epochs:
        if epoch < 60:      # 30% of 200 epochs
            ratio = 1.0
        elif epoch < 140:   # 70% of 200 epochs  
            ratio = 0.2
        else:
            ratio = 0.01
        hard_ratios.append(ratio)
        if epoch % 40 == 0:
            print(f"   Epoch {epoch:3d}: {ratio:.3f} ratio ({ratio/(1+ratio)*100:.1f}% normal)")
    
    print("\\n🌊 MOMENTUM CURRICULUM (Proposed):")
    momentum_scheduler = MomentumCurriculumScheduler(total_epochs=200)
    momentum_ratios = []
    
    for i, epoch in enumerate(epochs):
        confidence = confidence_trajectory[i]
        ratio, metrics = momentum_scheduler.update_ratio(epoch, confidence)
        momentum_ratios.append(ratio)
        
        if epoch % 40 == 0:
            print(f"   Epoch {epoch:3d}: {ratio:.3f} ratio ({ratio/(1+ratio)*100:.1f}% normal) "
                  f"[momentum: {metrics['momentum_accumulator']:.2f}]")
    
    # Summary comparison
    print(f"\\n📈 COMPARISON SUMMARY:")
    print(f"   Hard Transitions: {len([r for r in hard_ratios if r == 1.0])} balanced epochs")
    print(f"   Momentum: {len([r for r in momentum_ratios if r > 0.5])} balanced epochs")
    print(f"   Momentum provides {len([r for r in momentum_ratios if 0.1 < r < 0.9])} transition epochs")
    print(f"   vs. Hard's {len([r for r in hard_ratios if 0.1 < r < 0.9])} transition epochs")


if __name__ == "__main__":
    compare_curriculum_approaches()