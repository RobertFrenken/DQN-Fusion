"""
Training modes for CAN-Graph knowledge distillation.

This package contains specialized training logic for:
- Normal supervised training
- Autoencoder training (VGAE)
- Knowledge distillation
- Curriculum learning
- Fusion agent training (DQN)
"""

from .fusion import FusionTrainer
from .curriculum import CurriculumTrainer

__all__ = ['FusionTrainer', 'CurriculumTrainer']
