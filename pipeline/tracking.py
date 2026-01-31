"""MLflow experiment tracking integration.

Replaces the custom SQLite registry with MLflow tracking.
Uses deterministic run IDs for compatibility with Snakemake's DAG.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow

if TYPE_CHECKING:
    from .config import PipelineConfig

# MLflow tracking URI - must be on GPFS scratch for concurrent writes
# Set via environment variable or default to scratch directory
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db"
)

# Experiment name for all KD-GAT runs
EXPERIMENT_NAME = "kd-gat-pipeline"


def setup_tracking() -> None:
    """Initialize MLflow tracking URI and experiment.

    Must be called before any tracking operations.
    Creates the experiment if it doesn't exist.
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    # Ensure experiment exists
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)


def start_run(cfg: PipelineConfig, stage: str, run_name: str) -> None:
    """Start an MLflow run with deterministic run name.

    Args:
        cfg: Pipeline configuration
        stage: Training stage (autoencoder, curriculum, etc.)
        run_name: Deterministic run ID (dataset/model_size_stage[_kd])
    """
    # Ensure tracking is set up
    setup_tracking()

    # Start the run with deterministic run name
    mlflow.start_run(
        experiment_id=mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id,
        run_name=run_name,
    )

    # Log all configuration parameters
    config_dict = asdict(cfg)
    mlflow.log_params(config_dict)

    # Set metadata tags
    mlflow.set_tag("stage", stage)
    mlflow.set_tag("dataset", cfg.dataset)
    mlflow.set_tag("model_size", cfg.model_size)
    mlflow.set_tag("use_kd", cfg.use_kd)
    mlflow.set_tag("model_arch", _get_model_arch(stage))

    # Set teacher relationship for KD runs
    if cfg.use_kd and cfg.teacher_path:
        # Extract teacher run ID from path if possible
        teacher_id = _extract_teacher_id(cfg.teacher_path)
        if teacher_id:
            mlflow.set_tag("teacher_run_id", teacher_id)

    # Log start time
    mlflow.set_tag("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))


def end_run(result: dict[str, Any] | None = None, success: bool = True) -> None:
    """End the current MLflow run and log results.

    Args:
        result: Dictionary of metrics to log (e.g., {"val_loss": 0.123, "f1": 0.95})
        success: Whether the run completed successfully
    """
    if result:
        # Log all metrics
        mlflow.log_metrics(result)

    # Log completion status
    mlflow.set_tag("status", "complete" if success else "failed")
    mlflow.set_tag("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))

    # End the run
    mlflow.end_run()


def log_failure(error_msg: str) -> None:
    """Log a failed run with error message.

    Args:
        error_msg: Error message or exception string
    """
    mlflow.set_tag("status", "failed")
    mlflow.set_tag("error_msg", error_msg)
    mlflow.set_tag("end_time", time.strftime("%Y-%m-%d %H:%M:%S"))
    mlflow.end_run()


def _get_model_arch(stage: str) -> str:
    """Get model architecture from stage name."""
    from .paths import STAGES
    _, model_arch, _ = STAGES.get(stage, ("unknown", "unknown", "unknown"))
    return model_arch


def _extract_teacher_id(teacher_path: str) -> str | None:
    """Extract teacher run ID from checkpoint path.

    Example:
        experimentruns/hcrl_sa/teacher_autoencoder/best_model.pt
        -> "hcrl_sa/teacher_autoencoder"
    """
    try:
        path = Path(teacher_path)
        # Assume path is: {root}/{dataset}/{run_name}/best_model.pt
        if len(path.parts) >= 3:
            dataset = path.parts[-3]
            run_name = path.parts[-2]
            return f"{dataset}/{run_name}"
    except Exception:
        pass
    return None
