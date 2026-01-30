"""Config validation. Catches mistakes before they become 6-hour SLURM failures.

No Pydantic. No triple-validator stack. Just plain checks that raise ValueError.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PipelineConfig

from .paths import STAGES, DATASETS, checkpoint_path, data_dir

_log = logging.getLogger(__name__)


def validate(cfg: PipelineConfig, stage: str) -> None:
    """Raise ValueError if config + stage combination is invalid."""
    errors: list[str] = []

    # --- basic checks ---
    if stage not in STAGES:
        errors.append(f"Unknown stage '{stage}'. Choose from: {list(STAGES.keys())}")

    if cfg.dataset not in DATASETS:
        errors.append(f"Unknown dataset '{cfg.dataset}'. Choose from: {DATASETS}")

    if not data_dir(cfg).exists():
        errors.append(f"Data directory not found: {data_dir(cfg)}")

    if cfg.model_size not in ("teacher", "student"):
        errors.append(f"model_size must be 'teacher' or 'student', got '{cfg.model_size}'")

    # --- KD consistency ---
    if cfg.model_size == "student" and not cfg.use_kd:
        _log.warning("Student without KD — running as ablation baseline")

    if cfg.use_kd and not cfg.teacher_path:
        errors.append("use_kd=True but teacher_path is empty")

    # --- prerequisite checkpoints ---
    if stage == "curriculum":
        vgae_ckpt = checkpoint_path(cfg, "autoencoder")
        if not vgae_ckpt.exists():
            errors.append(f"Curriculum needs VGAE first: {vgae_ckpt}")

    if stage == "fusion":
        vgae_ckpt = checkpoint_path(cfg, "autoencoder")
        gat_ckpt = checkpoint_path(cfg, "curriculum")
        if not vgae_ckpt.exists():
            errors.append(f"Fusion needs VGAE first: {vgae_ckpt}")
        if not gat_ckpt.exists():
            errors.append(f"Fusion needs GAT first: {gat_ckpt}")

    if stage == "evaluation":
        # Evaluation can run with any available checkpoint (GAT, VGAE, or fusion)
        gat_ckpt = checkpoint_path(cfg, "curriculum")
        vgae_ckpt = checkpoint_path(cfg, "autoencoder")
        if not gat_ckpt.exists() and not vgae_ckpt.exists():
            errors.append(f"Evaluation needs at least one checkpoint (GAT or VGAE)")

    # --- numeric sanity ---
    if cfg.lr <= 0:
        errors.append("lr must be positive")
    if cfg.max_epochs < 1:
        errors.append("max_epochs must be >= 1")
    if cfg.batch_size < 1:
        errors.append("batch_size must be >= 1")
    if not 0 < cfg.dqn_gamma <= 1:
        errors.append("dqn_gamma must be in (0, 1]")
    if not 0 < cfg.safety_factor <= 1:
        errors.append("safety_factor must be in (0, 1]")

    if errors:
        raise ValueError(
            "Config validation failed:\n  " + "\n  ".join(errors)
        )
