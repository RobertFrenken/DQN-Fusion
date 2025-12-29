"""
CAN-Graph: Multi-Stage Knowledge-Distilled VGAE and GAT for CAN Intrusion Detection

A comprehensive framework for Controller Area Network (CAN) bus intrusion detection
using Graph Neural Networks with adaptive fusion and knowledge distillation.
"""

__version__ = "1.0.0"
__author__ = "CAN-Graph Team"

# Make key classes easily importable
from .models.models import GATWithJK, GraphAutoencoderNeighborhood
from .models.adaptive_fusion import EnhancedDQNFusionAgent

__all__ = [
    "GATWithJK", 
    "GraphAutoencoderNeighborhood", 
    "EnhancedDQNFusionAgent"
]