"""Main training flow: replicates the Snakemake DAG as Prefect tasks.

DAG structure (per dataset):
    preprocess ──┬──► large pipeline (vgae → gat → dqn → eval)
                 │         │ teacher checkpoints
                 │         ▼
                 ├──► small_kd pipeline (vgae → gat → dqn → eval)
                 │
                 └──► small_nokd pipeline (vgae → gat → dqn → eval)

Usage:
    # Full pipeline for one dataset
    python -m pipeline.cli flow --dataset hcrl_sa

    # Single scale
    python -m pipeline.cli flow --dataset hcrl_sa --scale large

    # All datasets (reads from config/datasets.yaml)
    python -m pipeline.cli flow
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from prefect import flow, task
from prefect.futures import wait

log = logging.getLogger(__name__)

# Python executable (same as Snakefile's PY)
_PY = sys.executable


def _run_stage(
    stage: str,
    model: str,
    scale: str,
    dataset: str,
    auxiliaries: str = "none",
    teacher_path: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a pipeline stage as a subprocess via the CLI.

    Using subprocess ensures each stage gets a clean CUDA context
    (critical for spawn multiprocessing) and matches how Snakemake
    dispatches stages.
    """
    cmd = [
        _PY, "-m", "pipeline.cli", stage,
        "--model", model,
        "--scale", scale,
        "--dataset", dataset,
        "--auxiliaries", auxiliaries,
    ]
    if teacher_path:
        cmd.extend(["--teacher-path", teacher_path])

    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, capture_output=False)
    return result


# ---------------------------------------------------------------------------
# Prefect tasks — thin wrappers around _run_stage
# ---------------------------------------------------------------------------

@task(name="preprocess", retries=0, log_prints=True)
def task_preprocess(dataset: str) -> None:
    """Ensure preprocessed graph cache exists for a dataset."""
    from src.training.datamodules import load_dataset
    from config.resolver import resolve

    cfg = resolve("vgae", "large", dataset=dataset)
    from config import data_dir, cache_dir
    load_dataset(dataset, data_dir(cfg), cache_dir(cfg), seed=cfg.seed)
    log.info("Preprocessed cache ready for %s", dataset)


@task(name="vgae-{scale}", retries=2, retry_delay_seconds=30, log_prints=True)
def task_vgae(
    dataset: str,
    scale: str,
    auxiliaries: str = "none",
    teacher_path: str | None = None,
) -> str:
    """Train VGAE. Returns checkpoint path."""
    _run_stage("autoencoder", "vgae", scale, dataset, auxiliaries, teacher_path)
    from config.resolver import resolve
    from config import checkpoint_path

    cfg = resolve("vgae", scale, auxiliaries=auxiliaries, dataset=dataset)
    ckpt = checkpoint_path(cfg, "autoencoder")
    return str(ckpt)


@task(name="gat-{scale}", retries=2, retry_delay_seconds=30, log_prints=True)
def task_gat(
    dataset: str,
    scale: str,
    auxiliaries: str = "none",
    teacher_path: str | None = None,
) -> str:
    """Train GAT with curriculum learning. Returns checkpoint path."""
    _run_stage("curriculum", "gat", scale, dataset, auxiliaries, teacher_path)
    from config.resolver import resolve
    from config import checkpoint_path

    cfg = resolve("gat", scale, auxiliaries=auxiliaries, dataset=dataset)
    ckpt = checkpoint_path(cfg, "curriculum")
    return str(ckpt)


@task(name="dqn-{scale}", retries=2, retry_delay_seconds=30, log_prints=True)
def task_dqn(
    dataset: str,
    scale: str,
    auxiliaries: str = "none",
    teacher_path: str | None = None,
) -> str:
    """Train DQN fusion agent. Returns checkpoint path."""
    _run_stage("fusion", "dqn", scale, dataset, auxiliaries, teacher_path)
    from config.resolver import resolve
    from config import checkpoint_path

    cfg = resolve("dqn", scale, auxiliaries=auxiliaries, dataset=dataset)
    ckpt = checkpoint_path(cfg, "fusion")
    return str(ckpt)


@task(name="eval-{scale}", retries=1, log_prints=True)
def task_eval(
    dataset: str,
    scale: str,
    auxiliaries: str = "none",
) -> None:
    """Run evaluation on all trained models for a variant."""
    _run_stage("evaluation", "vgae", scale, dataset, auxiliaries)


# ---------------------------------------------------------------------------
# Sub-flows for each pipeline variant
# ---------------------------------------------------------------------------

@flow(name="large-pipeline")
def large_pipeline(dataset: str) -> dict[str, str]:
    """Large-scale teacher pipeline (no KD)."""
    vgae_ckpt = task_vgae(dataset, "large")
    gat_ckpt = task_gat(dataset, "large")
    dqn_ckpt = task_dqn(dataset, "large")
    task_eval(dataset, "large")
    return {"vgae": vgae_ckpt, "gat": gat_ckpt, "dqn": dqn_ckpt}


@flow(name="small-kd-pipeline")
def small_kd_pipeline(
    dataset: str,
    teacher_ckpts: dict[str, str],
) -> None:
    """Small-scale KD pipeline (distilled from large teacher)."""
    vgae_ckpt = task_vgae(
        dataset, "small",
        auxiliaries="kd_standard",
        teacher_path=teacher_ckpts["vgae"],
    )
    gat_ckpt = task_gat(
        dataset, "small",
        auxiliaries="kd_standard",
        teacher_path=teacher_ckpts["gat"],
    )
    dqn_ckpt = task_dqn(
        dataset, "small",
        auxiliaries="kd_standard",
        teacher_path=teacher_ckpts["dqn"],
    )
    task_eval(dataset, "small", auxiliaries="kd_standard")


@flow(name="small-nokd-pipeline")
def small_nokd_pipeline(dataset: str) -> None:
    """Small-scale ablation pipeline (no KD, no teacher)."""
    vgae_ckpt = task_vgae(dataset, "small")
    gat_ckpt = task_gat(dataset, "small")
    dqn_ckpt = task_dqn(dataset, "small")
    task_eval(dataset, "small")


# ---------------------------------------------------------------------------
# Top-level flow
# ---------------------------------------------------------------------------

@flow(name="dataset-pipeline")
def _dataset_pipeline(dataset: str, scale: str | None = None) -> None:
    """All variants for a single dataset (fan-out target)."""
    log.info("=== Pipeline for dataset: %s ===", dataset)

    # Preprocess (shared by all variants)
    task_preprocess(dataset)

    if scale is None or scale == "large":
        large_pipeline(dataset)

        # Small KD depends on large teacher checkpoints
        if scale is None or scale == "small_kd":
            from config.resolver import resolve
            from config import checkpoint_path

            teacher_paths = {}
            for model, stage in [("vgae", "autoencoder"), ("gat", "curriculum"), ("dqn", "fusion")]:
                cfg = resolve(model, "large", dataset=dataset)
                teacher_paths[model] = str(checkpoint_path(cfg, stage))

            small_kd_pipeline(dataset, teacher_paths)

    elif scale == "small_kd":
        # Running small_kd alone — teacher must already exist
        from config.resolver import resolve
        from config import checkpoint_path

        teacher_paths = {}
        for model, stage in [("vgae", "autoencoder"), ("gat", "curriculum"), ("dqn", "fusion")]:
            cfg = resolve(model, "large", dataset=dataset)
            tp = checkpoint_path(cfg, stage)
            if not tp.exists():
                raise FileNotFoundError(
                    f"Teacher checkpoint not found: {tp}. "
                    f"Run with --scale large first."
                )
            teacher_paths[model] = str(tp)
        small_kd_pipeline(dataset, teacher_paths)

    if scale is None or scale == "small_nokd":
        small_nokd_pipeline(dataset)


@flow(name="kd-gat-pipeline", log_prints=True)
def train_pipeline(
    datasets: list[str] | None = None,
    scale: str | None = None,
) -> None:
    """Full KD-GAT training pipeline.

    Parameters
    ----------
    datasets : list[str] | None
        Datasets to train on.  None = all from catalog.
    scale : str | None
        If set, only run the specified scale variant
        ("large", "small_kd", "small_nokd").  None = all.
    """
    if datasets is None:
        from config.paths import get_datasets
        datasets = get_datasets()

    # Fan out per-dataset work — each dataset is independent
    futures = []
    for ds in datasets:
        futures.append(_dataset_pipeline.submit(ds, scale))
    wait(futures)

    log.info("=== Pipeline complete for %d dataset(s) ===", len(datasets))
