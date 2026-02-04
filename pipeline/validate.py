"""Config validation. Catches mistakes before they become 6-hour SLURM failures.

No Pydantic. No triple-validator stack. Just plain checks that raise ValueError.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PipelineConfig

from .paths import STAGES, DATASETS, checkpoint_path, config_path, data_dir

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

    if cfg.use_kd and not cfg.teacher_path and stage != "evaluation":
        errors.append("use_kd=True but teacher_path is empty")

    if cfg.use_kd and cfg.teacher_path:
        tp = Path(cfg.teacher_path)
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
