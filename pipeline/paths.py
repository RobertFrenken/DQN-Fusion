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


def run_id(cfg: PipelineConfig, stage: str) -> str:
    """Deterministic run ID from config and stage.

    Format: {dataset}/{model_size}_{stage}[_kd]
    Examples:
        - "hcrl_sa/teacher_autoencoder"
        - "hcrl_sa/student_curriculum_kd"
        - "set_01/teacher_fusion"

    This ID is used for:
    - Filesystem directory names (via stage_dir)
    - MLflow run names (for tracking)
    - Snakemake target paths (deterministic at DAG construction time)
    """
    kd_suffix = "_kd" if cfg.use_kd else ""
    return f"{cfg.dataset}/{cfg.model_size}_{stage}{kd_suffix}"


def stage_dir(cfg: PipelineConfig, stage: str) -> Path:
    """Canonical experiment directory.

    Layout (simplified): {root}/{dataset}/{size}_{stage}[_kd]

    Before: 8 levels (modality/dataset/size/learning/model/distill/mode)
    After:  2 levels (dataset/run_name)
    """
    return Path(cfg.experiment_root) / run_id(cfg, stage)


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
    return Path("data") / "cache" / cfg.dataset
