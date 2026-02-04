"""MLflow experiment tracking integration.

Replaces the custom SQLite registry with MLflow tracking.
Uses deterministic run IDs for compatibility with Snakemake's DAG.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import psutil
import torch

if TYPE_CHECKING:
    from .config import PipelineConfig

log = logging.getLogger(__name__)

# MLflow tracking URI - must be on GPFS scratch for concurrent writes
# Set via environment variable or default to scratch directory
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db"
)

# Experiment name for all KD-GAT runs
EXPERIMENT_NAME = "kd-gat-pipeline"

_tracking_initialized = False


def setup_tracking() -> None:
    """Initialize MLflow tracking URI and experiment.

    Must be called before any tracking operations.
    Creates the experiment if it doesn't exist.

    Handles concurrent initialization from multiple SLURM jobs
    by retrying on SQLite "table already exists" errors.
    """
    global _tracking_initialized
    if _tracking_initialized:
        return

    mlflow.set_tracking_uri(TRACKING_URI)

    # Concurrent SLURM jobs may race to create the DB schema.
    # MLflow's _initialize_tables uses CREATE TABLE (not IF NOT EXISTS),
    # so a second process can crash. Retry after a short sleep.
    max_retries = 3
    for attempt in range(max_retries):
        try:
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if experiment is None:
                mlflow.create_experiment(EXPERIMENT_NAME)
            _tracking_initialized = True
            return
        except Exception as e:
            if "already exists" in str(e) and attempt < max_retries - 1:
                log.warning("MLflow DB init race (attempt %d/%d), retrying: %s",
                            attempt + 1, max_retries, e)
                time.sleep(2 * (attempt + 1))
            else:
                raise


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
        result: Dictionary of metrics to log. Must be a flat {str: numeric} dict.
                Nested dicts (e.g. from evaluation) are skipped with a warning.
        success: Whether the run completed successfully
    """
    if result:
        # Only log flat numeric metrics; skip nested dicts (e.g. evaluation output)
        flat = {k: v for k, v in result.items() if isinstance(v, (int, float))}
        if flat:
            mlflow.log_metrics(flat)
        if len(flat) < len(result):
            log.info("Skipped %d non-numeric metric entries in end_run",
                     len(result) - len(flat))

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


def log_memory_metrics(step: int | None = None) -> dict[str, float]:
    """Log current CPU and GPU memory usage to MLflow.

    Args:
        step: Optional step number for metric logging

    Returns:
        Dictionary of memory metrics (useful for logging elsewhere)
    """
    metrics = {}

    # CPU memory
    mem = psutil.virtual_memory()
    metrics["cpu_mem_percent"] = mem.percent
    metrics["cpu_mem_used_gb"] = mem.used / (1024 ** 3)
    metrics["cpu_mem_available_gb"] = mem.available / (1024 ** 3)

    # GPU memory (if available)
    if torch.cuda.is_available():
        metrics["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        metrics["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
        metrics["gpu_mem_max_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)

        # Per-GPU stats if multiple GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            for i in range(num_gpus):
                metrics[f"gpu{i}_mem_allocated_gb"] = torch.cuda.memory_allocated(i) / (1024 ** 3)

    # Log to MLflow
    try:
        if step is not None:
            mlflow.log_metrics(metrics, step=step)
        else:
            mlflow.log_metrics(metrics)
    except Exception as e:
        log.warning("Failed to log memory metrics to MLflow: %s", e)

    return metrics


def get_memory_summary() -> str:
    """Get a human-readable memory summary string.

    Returns:
        Formatted string with current memory usage
    """
    mem = psutil.virtual_memory()
    summary = f"CPU: {mem.percent:.1f}% ({mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB)"

    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        summary += f" | GPU: {gpu_alloc:.2f}/{gpu_reserved:.2f} GB (alloc/reserved)"

    return summary
