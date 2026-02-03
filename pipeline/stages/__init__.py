"""Training stages for KD-GAT pipeline.

Public API:
    STAGE_FNS: dict mapping stage name -> function(cfg) -> result
"""
from .training import train_autoencoder, train_curriculum, train_normal
from .fusion import train_fusion
from .evaluation import evaluate

STAGE_FNS = {
    "autoencoder": train_autoencoder,
    "curriculum":  train_curriculum,
    "normal":      train_normal,
    "fusion":      train_fusion,
    "evaluation":  evaluate,
}

__all__ = ["STAGE_FNS"]
