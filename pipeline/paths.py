"""All paths derived from PipelineConfig. One function, one truth.

Every file location in the entire system comes from stage_dir().
The Snakefile, the CLI, the stages -- they all call these functions.
No second implementation. No disagreement possible.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PipelineConfig

# stage_name -> (learning_type, model_arch, training_mode)
STAGES = {
    "autoencoder": ("unsupervised", "vgae", "autoencoder"),
    "curriculum":  ("supervised",   "gat",  "curriculum"),
    "normal":      ("supervised",   "gat",  "normal"),
    "fusion":      ("rl_fusion",    "dqn",  "fusion"),
    "evaluation":  ("evaluation",   "eval", "evaluation"),
}

DATASETS = ["hcrl_ch", "hcrl_sa", "set_01", "set_02", "set_03", "set_04"]


def stage_dir(cfg: PipelineConfig, stage: str) -> Path:
    """Canonical experiment directory.

    Layout: {root}/{modality}/{dataset}/{size}/{learning_type}/{model}/{distill}/{mode}
    """
    learning_type, model_arch, mode = STAGES[stage]
    distillation = "distilled" if cfg.use_kd else "no_distillation"
    return (
        Path(cfg.experiment_root)
        / cfg.modality
        / cfg.dataset
        / cfg.model_size
        / learning_type
        / model_arch
        / distillation
        / mode
    )


def checkpoint_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the best model checkpoint is saved."""
    return stage_dir(cfg, stage) / "best_model.pt"


def config_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the frozen config JSON is saved alongside the model."""
    return stage_dir(cfg, stage) / "config.json"


def log_dir(cfg: PipelineConfig, stage: str) -> Path:
    """Lightning / CSV log directory for a stage."""
    return stage_dir(cfg, stage) / "logs"


def data_dir(cfg: PipelineConfig) -> Path:
    """Raw data directory for a dataset."""
    return Path("data") / cfg.modality / cfg.dataset


def cache_dir(cfg: PipelineConfig) -> Path:
    """Processed-graph cache directory."""
    return Path(cfg.experiment_root) / cfg.modality / cfg.dataset / "cache"
