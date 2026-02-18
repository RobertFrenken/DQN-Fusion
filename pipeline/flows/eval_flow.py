"""Standalone evaluation flow: re-run evaluations without re-training.

Usage:
    python -m pipeline.cli flow --eval-only --dataset hcrl_sa
    python -m pipeline.cli flow --eval-only  # all datasets
"""
from __future__ import annotations

import logging

from prefect import flow

from .train_flow import task_eval

log = logging.getLogger(__name__)


@flow(name="kd-gat-evaluation", log_prints=True)
def eval_pipeline(
    datasets: list[str] | None = None,
    scale: str | None = None,
) -> None:
    """Re-run evaluation for existing trained models.

    Parameters
    ----------
    datasets : list[str] | None
        Datasets to evaluate.  None = all from catalog.
    scale : str | None
        If set, only evaluate the specified variant.  None = all.
    """
    if datasets is None:
        from config.paths import get_datasets
        datasets = get_datasets()

    for ds in datasets:
        log.info("=== Evaluation for dataset: %s ===", ds)

        if scale is None or scale == "large":
            task_eval(ds, "large")

        if scale is None or scale == "small_kd":
            task_eval(ds, "small", auxiliaries="kd_standard")

        if scale is None or scale == "small_nokd":
            task_eval(ds, "small")

    log.info("=== Evaluation complete for %d dataset(s) ===", len(datasets))
