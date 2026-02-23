"""Ray-based pipeline orchestration for KD-GAT.

Faithfully replicates the Prefect DAG from pipeline/flows/train_flow.py:

    preprocess ──┬──► large pipeline (vgae → gat → dqn → eval)
                 │         │ teacher checkpoints
                 │         ▼
                 ├──► small_kd pipeline (vgae → gat → dqn → eval)
                 │
                 └──► small_nokd pipeline (vgae → gat → dqn → eval)

Each stage runs as a subprocess for clean CUDA context, matching the
original design. Ray handles DAG scheduling and per-dataset fan-out.

Usage:
    python -m pipeline.cli flow --dataset hcrl_sa
    python -m pipeline.cli flow --dataset hcrl_sa --scale large
    python -m pipeline.cli flow --eval-only --dataset hcrl_sa
    python -m pipeline.cli flow --local  # Ray local mode
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import ray

log = logging.getLogger(__name__)

_PY = sys.executable


# ---------------------------------------------------------------------------
# Subprocess dispatch (same as Prefect version)
# ---------------------------------------------------------------------------

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
    (critical for spawn multiprocessing).
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
# Ray remote tasks
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=1)
def task_preprocess(dataset: str) -> None:
    """Ensure preprocessed graph cache exists for a dataset."""
    from src.training.datamodules import load_dataset
    from config.resolver import resolve
    from config import data_dir, cache_dir

    cfg = resolve("vgae", "large", dataset=dataset)
    load_dataset(dataset, data_dir(cfg), cache_dir(cfg), seed=cfg.seed)
    log.info("Preprocessed cache ready for %s", dataset)


@ray.remote(num_gpus=1)
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
    return str(checkpoint_path(cfg, "autoencoder"))


@ray.remote(num_gpus=1)
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
    return str(checkpoint_path(cfg, "curriculum"))


@ray.remote(num_gpus=1)
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
    return str(checkpoint_path(cfg, "fusion"))


@ray.remote(num_gpus=1)
def task_eval(
    dataset: str,
    scale: str,
    auxiliaries: str = "none",
) -> None:
    """Run evaluation on all trained models for a variant."""
    _run_stage("evaluation", "vgae", scale, dataset, auxiliaries)


# ---------------------------------------------------------------------------
# Pipeline variants (sequential chains via ObjectRef dependencies)
# ---------------------------------------------------------------------------

def large_pipeline(dataset: str) -> dict[str, str]:
    """Large-scale teacher pipeline (no KD). Returns checkpoint paths."""
    vgae_ref = task_vgae.remote(dataset, "large")
    vgae_ckpt = ray.get(vgae_ref)

    gat_ref = task_gat.remote(dataset, "large")
    gat_ckpt = ray.get(gat_ref)

    dqn_ref = task_dqn.remote(dataset, "large")
    dqn_ckpt = ray.get(dqn_ref)

    ray.get(task_eval.remote(dataset, "large"))

    return {"vgae": vgae_ckpt, "gat": gat_ckpt, "dqn": dqn_ckpt}


def small_kd_pipeline(dataset: str, teacher_ckpts: dict[str, str]) -> None:
    """Small-scale KD pipeline (distilled from large teacher)."""
    vgae_ref = task_vgae.remote(
        dataset, "small",
        auxiliaries="kd_standard",
        teacher_path=teacher_ckpts["vgae"],
    )
    ray.get(vgae_ref)

    gat_ref = task_gat.remote(
        dataset, "small",
        auxiliaries="kd_standard",
        teacher_path=teacher_ckpts["gat"],
    )
    ray.get(gat_ref)

    dqn_ref = task_dqn.remote(
        dataset, "small",
        auxiliaries="kd_standard",
        teacher_path=teacher_ckpts["dqn"],
    )
    ray.get(dqn_ref)

    ray.get(task_eval.remote(dataset, "small", auxiliaries="kd_standard"))


def small_nokd_pipeline(dataset: str) -> None:
    """Small-scale ablation pipeline (no KD, no teacher)."""
    ray.get(task_vgae.remote(dataset, "small"))
    ray.get(task_gat.remote(dataset, "small"))
    ray.get(task_dqn.remote(dataset, "small"))
    ray.get(task_eval.remote(dataset, "small"))


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------

@ray.remote
def dataset_pipeline(dataset: str, scale: str | None = None) -> None:
    """All variants for a single dataset."""
    log.info("=== Pipeline for dataset: %s ===", dataset)

    # Preprocess (shared by all variants)
    ray.get(task_preprocess.remote(dataset))

    if scale is None or scale == "large":
        teacher_ckpts = large_pipeline(dataset)

        # Small KD depends on large teacher checkpoints
        if scale is None or scale == "small_kd":
            small_kd_pipeline(dataset, teacher_ckpts)

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


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

def train_pipeline(
    datasets: list[str] | None = None,
    scale: str | None = None,
    local: bool = False,
) -> None:
    """Full KD-GAT training pipeline.

    Parameters
    ----------
    datasets : list[str] | None
        Datasets to train on.  None = all from catalog.
    scale : str | None
        If set, only run the specified scale variant
        ("large", "small_kd", "small_nokd").  None = all.
    local : bool
        If True, use Ray local mode (no cluster).
    """
    from .ray_slurm import ray_init_kwargs

    if datasets is None:
        from config.paths import get_datasets
        datasets = get_datasets()

    # Initialize Ray
    if not ray.is_initialized():
        kwargs = ray_init_kwargs()
        if local:
            kwargs["num_gpus"] = 0
        ray.init(**kwargs)

    # Fan out per-dataset work — each dataset is independent
    refs = [dataset_pipeline.remote(ds, scale) for ds in datasets]
    ray.get(refs)

    log.info("=== Pipeline complete for %d dataset(s) ===", len(datasets))


def eval_pipeline(
    datasets: list[str] | None = None,
    scale: str | None = None,
    local: bool = False,
) -> None:
    """Re-run evaluation for existing trained models."""
    from .ray_slurm import ray_init_kwargs

    if datasets is None:
        from config.paths import get_datasets
        datasets = get_datasets()

    if not ray.is_initialized():
        kwargs = ray_init_kwargs()
        if local:
            kwargs["num_gpus"] = 0
        ray.init(**kwargs)

    refs = []
    for ds in datasets:
        log.info("=== Evaluation for dataset: %s ===", ds)

        if scale is None or scale == "large":
            refs.append(task_eval.remote(ds, "large"))

        if scale is None or scale == "small_kd":
            refs.append(task_eval.remote(ds, "small", auxiliaries="kd_standard"))

        if scale is None or scale == "small_nokd":
            refs.append(task_eval.remote(ds, "small"))

    ray.get(refs)
    log.info("=== Evaluation complete for %d dataset(s) ===", len(datasets))
