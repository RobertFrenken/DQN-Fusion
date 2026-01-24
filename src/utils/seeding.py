"""Seeding utilities for reproducibility."""
from typing import Optional
import random
import numpy as np


def set_global_seeds(seed: Optional[int], deterministic: bool = True, cudnn_benchmark: bool = False):
    """Set seeds for random, numpy, and torch. If `seed` is None, do nothing.

    deterministic: when True, set torch.backends.cudnn.deterministic = True and benchmark False
    cudnn_benchmark: when True, sets torch.backends.cudnn.benchmark = True (only if a real torch is available)
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic/benchmark settings
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    except Exception:
        # Torch not available (test shims) — ignore
        pass

    return seed
