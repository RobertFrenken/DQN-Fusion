"""Models module for CAN-Graph project.

Public API re-exported from submodules:

    from src.models import get, register, fusion_state_dim, extractors
    from src.models import FusionFeatureExtractor, VGAEFusionExtractor, GATFusionExtractor
"""
from .registry import ModelEntry, register, get, fusion_state_dim, extractors
from .fusion_features import FusionFeatureExtractor, VGAEFusionExtractor, GATFusionExtractor
