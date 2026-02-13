"""Config validation. Catches mistakes before they become 6-hour SLURM failures.

Most field-level checks are now handled by Pydantic Field() constraints.
This module handles filesystem checks and cross-stage prerequisite checks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import PipelineConfig

from config import STAGES, get_datasets, checkpoint_path, config_path, data_dir

_log = logging.getLogger(__name__)


def validate(cfg: PipelineConfig, stage: str) -> None:
    """Raise ValueError if config + stage combination is invalid."""
    errors: list[str] = []

    # --- basic checks ---
    if stage not in STAGES:
        errors.append(f"Unknown stage '{stage}'. Choose from: {list(STAGES.keys())}")

    if cfg.dataset not in get_datasets():
        errors.append(f"Unknown dataset '{cfg.dataset}'. Choose from: {get_datasets()}")

    if not data_dir(cfg).exists():
        errors.append(f"Data directory not found: {data_dir(cfg)}")

    # --- KD consistency ---
    if cfg.scale == "small" and not cfg.has_kd:
        _log.warning("Small model without KD -- running as ablation baseline")

    if cfg.has_kd and not cfg.kd.model_path and stage != "evaluation":
        errors.append("KD auxiliary enabled but model_path is empty")

    if cfg.has_kd and cfg.kd.model_path:
        tp = Path(cfg.kd.model_path)
        if not tp.exists():
            errors.append(f"Teacher checkpoint not found: {tp}")
        teacher_cfg = tp.parent / "config.json"
        if not teacher_cfg.exists():
            errors.append(f"Teacher config not found: {teacher_cfg}")

    # --- prerequisite checkpoints + frozen configs ---
    def _check_prereq(prereq_stage: str, needed_by: str) -> None:
        ckpt = checkpoint_path(cfg, prereq_stage)
        cfg_file = config_path(cfg, prereq_stage)
        if not ckpt.exists():
            errors.append(f"{needed_by} needs {prereq_stage} checkpoint: {ckpt}")
        if not cfg_file.exists():
            errors.append(f"{needed_by} needs {prereq_stage} config: {cfg_file}")

    if stage == "curriculum":
        _check_prereq("autoencoder", "Curriculum")

    if stage == "fusion":
        _check_prereq("autoencoder", "Fusion")
        _check_prereq("curriculum", "Fusion")

    if stage == "evaluation":
        gat_ckpt = checkpoint_path(cfg, "curriculum")
        vgae_ckpt = checkpoint_path(cfg, "autoencoder")
        if not gat_ckpt.exists() and not vgae_ckpt.exists():
            errors.append("Evaluation needs at least one checkpoint (GAT or VGAE)")

    if errors:
        raise ValueError(
            "Config validation failed:\n  " + "\n  ".join(errors)
        )
