"""Ray-based pipeline orchestration for KD-GAT.

    preprocess ──┬──► large pipeline (vgae → gat → dqn → eval)
                 │         │ teacher checkpoints       ┌──► small_nokd pipeline
                 │         ▼                           │    (concurrent, no dependency
                 ├──► small_kd pipeline                │     on teacher checkpoints)
                 │    (vgae → gat → dqn → eval)       │
                 └────────────────────────────────────join

Each stage runs as a subprocess for clean CUDA context.
Ray handles DAG scheduling and per-dataset fan-out.

Usage:
    python -m pipeline.cli flow --dataset hcrl_sa
    python -m pipeline.cli flow --dataset hcrl_sa --scale large
    python -m pipeline.cli flow --eval-only --dataset hcrl_sa
    python -m pipeline.cli flow --local  # Ray local mode
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time

import ray

log = logging.getLogger(__name__)

_PY = sys.executable

# Set KD_GAT_BENCHMARK=1 to enable detailed orchestration timing.
# Output written to KD_GAT_BENCHMARK_LOG (default: benchmark_timing.jsonl).
_BENCHMARK = os.environ.get("KD_GAT_BENCHMARK", "") == "1"
_BENCHMARK_LOG = os.environ.get("KD_GAT_BENCHMARK_LOG", "benchmark_timing.jsonl")

# Track when the last stage ended so we can measure inter-stage gaps.
_last_stage_end: float | None = None


def _query_gpu_utilization() -> dict[str, float | None]:
    """Sample GPU utilization and memory via nvidia-smi. Returns {} on failure."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return {}
        # Take the first GPU line (single-GPU jobs).
        parts = out.stdout.strip().split("\n")[0].split(",")
        return {
            "gpu_util_pct": float(parts[0].strip()),
            "gpu_mem_used_mib": float(parts[1].strip()),
            "gpu_mem_total_mib": float(parts[2].strip()),
        }
    except Exception:
        return {}


def _write_benchmark_record(record: dict) -> None:
    """Append a JSONL timing record to the benchmark log."""
    with open(_BENCHMARK_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Subprocess dispatch
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
    (critical for spawn multiprocessing). Logs wall-clock timing
    for benchmarking subprocess overhead vs training time.

    When KD_GAT_BENCHMARK=1, writes detailed timing to a JSONL log:
    - spawn_overhead_s: time for subprocess to start (Popen → first poll)
    - execution_s: wall-clock time of the subprocess itself
    - total_s: full wall time including spawn + teardown
    - inter_stage_gap_s: idle time since the previous stage ended
    - gpu_pre/gpu_post: nvidia-smi snapshots before/after
    """
    global _last_stage_end

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

    if not _BENCHMARK:
        # Fast path: original behavior, no extra overhead.
        t0 = time.monotonic()
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.monotonic() - t0
        log.info(
            "Stage %s/%s/%s completed in %.1fs (dataset=%s)",
            model, scale, stage, elapsed, dataset,
        )
        return result

    # --- Benchmark path: detailed timing instrumentation ---
    gpu_pre = _query_gpu_utilization()
    inter_stage_gap = None
    if _last_stage_end is not None:
        inter_stage_gap = time.monotonic() - _last_stage_end

    t_call = time.monotonic()

    # Use Popen to measure spawn overhead separately from execution.
    proc = subprocess.Popen(cmd)
    t_spawned = time.monotonic()
    spawn_overhead = t_spawned - t_call

    proc.wait()
    t_done = time.monotonic()

    _last_stage_end = t_done

    execution_time = t_done - t_spawned
    total_time = t_done - t_call

    gpu_post = _query_gpu_utilization()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset,
        "model": model,
        "scale": scale,
        "stage": stage,
        "auxiliaries": auxiliaries,
        "spawn_overhead_s": round(spawn_overhead, 3),
        "execution_s": round(execution_time, 3),
        "total_s": round(total_time, 3),
        "inter_stage_gap_s": round(inter_stage_gap, 3) if inter_stage_gap is not None else None,
        "gpu_pre": gpu_pre,
        "gpu_post": gpu_post,
    }
    _write_benchmark_record(record)

    log.info(
        "Stage %s/%s/%s completed in %.1fs "
        "(spawn=%.3fs, exec=%.1fs, gap=%.3fs, dataset=%s)",
        model, scale, stage, total_time,
        spawn_overhead, execution_time,
        inter_stage_gap if inter_stage_gap is not None else 0.0,
        dataset,
    )
    return subprocess.CompletedProcess(cmd, proc.returncode)


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


def _make_stage_task(stage: str, model: str):
    """Factory for Ray remote tasks that train a model and return its checkpoint path."""
    @ray.remote(num_gpus=1)
    def task(dataset: str, scale: str, auxiliaries: str = "none", teacher_path: str | None = None) -> str:
        _run_stage(stage, model, scale, dataset, auxiliaries, teacher_path)
        from config.resolver import resolve
        from config import checkpoint_path
        cfg = resolve(model, scale, auxiliaries=auxiliaries, dataset=dataset)
        return str(checkpoint_path(cfg, stage))
    task.__name__ = f"task_{model}"
    return task


task_vgae = _make_stage_task("autoencoder", "vgae")
task_gat = _make_stage_task("curriculum", "gat")
task_dqn = _make_stage_task("fusion", "dqn")


@ray.remote(num_gpus=1)
def task_eval(dataset: str, scale: str, auxiliaries: str = "none") -> None:
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


@ray.remote
def _small_nokd_pipeline_remote(dataset: str) -> None:
    """Remote wrapper for small_nokd_pipeline.

    Allows dataset_pipeline to launch small_nokd concurrently with
    large_pipeline + small_kd_pipeline since small_nokd has no dependency
    on teacher checkpoints. On single-GPU, Ray still serializes the GPU
    tasks; on multi-GPU clusters, this enables true parallelism.
    """
    small_nokd_pipeline(dataset)


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------

@ray.remote
def dataset_pipeline(dataset: str, scale: str | None = None) -> None:
    """All variants for a single dataset.

    When running all variants (scale=None), small_nokd_pipeline launches
    concurrently with large_pipeline. It has no dependency on teacher
    checkpoints, so it can overlap with large + small_kd on multi-GPU
    clusters. On single-GPU, Ray serializes the GPU tasks automatically.
    """
    log.info("=== Pipeline for dataset: %s ===", dataset)

    # Preprocess (shared by all variants)
    ray.get(task_preprocess.remote(dataset))

    # Launch small_nokd early when running all variants — it has no
    # dependency on large pipeline outputs (no teacher checkpoints needed).
    nokd_ref = None
    if scale is None:
        nokd_ref = _small_nokd_pipeline_remote.remote(dataset)
        log.info("Launched small_nokd concurrently for %s", dataset)

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

    # Join the concurrent small_nokd task if we launched it early.
    if nokd_ref is not None:
        ray.get(nokd_ref)
    elif scale == "small_nokd":
        # Running small_nokd alone (explicit --scale small_nokd)
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
