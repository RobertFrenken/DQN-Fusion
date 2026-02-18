"""Prefect flow orchestration for KD-GAT pipeline.

Uses Prefect flows + dask-jobqueue SLURMCluster for distributed execution.

Usage:
    python -m pipeline.cli flow --dataset hcrl_sa --scale large
    python -m pipeline.cli flow --dataset hcrl_sa               # all scales
    python -m pipeline.cli flow --eval-only --dataset hcrl_sa   # re-run evals
"""
