"""Prefect flow orchestration for KD-GAT pipeline.

Replaces Snakemake DAG with Prefect flows + dask-jobqueue SLURMCluster.
Snakemake still works as a fallback (pipeline/Snakefile).

Usage:
    python -m pipeline.cli flow --dataset hcrl_sa --scale large
    python -m pipeline.cli flow --dataset hcrl_sa               # all scales
    python -m pipeline.cli flow --eval-only --dataset hcrl_sa   # re-run evals
"""
