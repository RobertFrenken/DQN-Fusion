"""
Utility functions and helpers for CAN-Graph.

Includes:
- Lightning-native GPU utilities
- Cache management
- Other modern utility functions
"""

# Import only Lightning-native utilities (legacy functions removed)
try:
    from .lightning_gpu_utils import LightningGPUOptimizer, LightningDataLoader
    __all__ = ["LightningGPUOptimizer", "LightningDataLoader"]
except ImportError:
    __all__ = []