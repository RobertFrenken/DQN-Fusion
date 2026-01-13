"""
Utility functions and helpers for CAN-Graph.

Includes:
- Plotting and visualization functions  
- Logging and performance monitoring
- Cache management
"""

# Import only existing utilities (removed legacy GPU optimization imports)
try:
    from .utils_logging import log_memory_usage
    from .legacy_compatibility import detect_gpu_capabilities_unified, create_optimized_data_loaders
    __all__ = ["log_memory_usage", "detect_gpu_capabilities_unified", "create_optimized_data_loaders"]
except ImportError:
    __all__ = []