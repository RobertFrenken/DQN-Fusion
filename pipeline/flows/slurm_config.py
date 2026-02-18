"""SLURMCluster configuration for Prefect + dask-jobqueue.

Provides a DaskTaskRunner factory that submits tasks to SLURM via
dask-jobqueue's SLURMCluster.  SLURM parameters (account, partition,
GPU type) are read from config/constants.py (env-overridable).

Usage:
    from pipeline.flows.slurm_config import make_dask_runner
    runner = make_dask_runner(gpu=True)
"""
from __future__ import annotations

import os

from config.constants import SLURM_ACCOUNT, SLURM_PARTITION, SLURM_GPU_TYPE

# Prefect/dask state on GPFS (not NFS) to avoid .nfs ghost files
SCRATCH_ROOT = os.getenv(
    "KD_GAT_SCRATCH", "/fs/scratch/PAS1266"
)
PREFECT_SCRATCH = f"{SCRATCH_ROOT}/.prefect"
DASK_LOG_DIR = f"{SCRATCH_ROOT}/.dask-logs"


def _ensure_dirs() -> None:
    """Create scratch directories if they don't exist."""
    for d in (PREFECT_SCRATCH, DASK_LOG_DIR):
        os.makedirs(d, exist_ok=True)


def make_dask_runner(
    gpu: bool = True,
    cores_per_worker: int = 16,
    memory: str = "128GB",
    walltime: str = "06:00:00",
    n_workers: int = 1,
    adapt_max: int | None = None,
):
    """Create a DaskTaskRunner backed by a SLURMCluster.

    Parameters
    ----------
    gpu : bool
        Request GPU resources (1 GPU per worker).
    cores_per_worker : int
        CPU cores per SLURM job.
    memory : str
        Memory per SLURM job (e.g. "128GB").
    walltime : str
        Max walltime per job.
    n_workers : int
        Fixed number of workers (ignored if adapt_max is set).
    adapt_max : int | None
        If set, use adaptive scaling up to this many workers.
    """
    from prefect_dask import DaskTaskRunner

    _ensure_dirs()

    job_extra_directives = [
        f"--account={SLURM_ACCOUNT}",
    ]
    if gpu:
        job_extra_directives.append(f"--gpus-per-node=1")

    cluster_kwargs = {
        "cores": cores_per_worker,
        "memory": memory,
        "queue": SLURM_PARTITION,
        "walltime": walltime,
        "log_directory": DASK_LOG_DIR,
        "job_extra_directives": job_extra_directives,
        "python": os.environ.get("KD_GAT_PYTHON", "python"),
    }

    adapt_kwargs = None
    if adapt_max is not None:
        adapt_kwargs = {"minimum": 1, "maximum": adapt_max}

    return DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs=cluster_kwargs,
        adapt_kwargs=adapt_kwargs,
    )


def make_local_runner():
    """Create a DaskTaskRunner using a local cluster (for testing)."""
    from prefect_dask import DaskTaskRunner

    return DaskTaskRunner()
