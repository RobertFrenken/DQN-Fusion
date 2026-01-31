"""CLI tool for querying MLflow experiment data.

Usage:
    python -m pipeline.query --all
    python -m pipeline.query --dataset hcrl_sa --stage curriculum
    python -m pipeline.query --leaderboard --top 5
    python -m pipeline.query --running
    python -m pipeline.query --compare teacher student_kd
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import mlflow
import pandas as pd

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db"
)
EXPERIMENT_NAME = "kd-gat-pipeline"


def setup_mlflow() -> str:
    """Set up MLflow and return experiment ID."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found at {TRACKING_URI}")
    return experiment.experiment_id


def query_all(experiment_id: str, limit: int = 20) -> pd.DataFrame:
    """Get all runs."""
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=limit,
    )
    return runs


def query_filtered(
    experiment_id: str,
    dataset: Optional[str] = None,
    stage: Optional[str] = None,
    model_size: Optional[str] = None,
    use_kd: Optional[bool] = None,
    status: Optional[str] = None,
) -> pd.DataFrame:
    """Query runs with filters."""
    filters = []
    if dataset:
        filters.append(f"tags.dataset = '{dataset}'")
    if stage:
        filters.append(f"tags.stage = '{stage}'")
    if model_size:
        filters.append(f"tags.model_size = '{model_size}'")
    if use_kd is not None:
        filters.append(f"tags.use_kd = '{use_kd}'")
    if status:
        filters.append(f"tags.status = '{status}'")

    filter_string = " AND ".join(filters) if filters else None
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
    )
    return runs


def leaderboard(experiment_id: str, top: int = 10, metric: str = "f1") -> pd.DataFrame:
    """Get top runs by metric."""
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.stage = 'evaluation' AND tags.status = 'complete'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=top,
    )
    return runs


def running_jobs(experiment_id: str) -> pd.DataFrame:
    """Get currently running jobs."""
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.status != 'complete' AND tags.status != 'failed'",
    )
    return runs


def compare(experiment_id: str, model_a: str, model_b: str) -> pd.DataFrame:
    """Compare two model variants (e.g., teacher vs student_kd)."""
    # Parse model variants
    use_kd_a = "kd" in model_a.lower()
    use_kd_b = "kd" in model_b.lower()
    size_a = "student" if "student" in model_a else "teacher"
    size_b = "student" if "student" in model_b else "teacher"

    # Get runs for each variant
    filter_a = f"tags.model_size = '{size_a}' AND tags.use_kd = '{use_kd_a}' AND tags.status = 'complete' AND tags.stage = 'evaluation'"
    filter_b = f"tags.model_size = '{size_b}' AND tags.use_kd = '{use_kd_b}' AND tags.status = 'complete' AND tags.stage = 'evaluation'"

    runs_a = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_a)
    runs_b = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_b)

    # Merge on dataset
    comparison = pd.merge(
        runs_a[["tags.dataset", "metrics.f1", "metrics.accuracy"]],
        runs_b[["tags.dataset", "metrics.f1", "metrics.accuracy"]],
        on="tags.dataset",
        suffixes=(f"_{model_a}", f"_{model_b}"),
    )

    # Calculate deltas
    comparison["f1_delta"] = (
        comparison[f"metrics.f1_{model_b}"] - comparison[f"metrics.f1_{model_a}"]
    )
    comparison["acc_delta"] = (
        comparison[f"metrics.accuracy_{model_b}"]
        - comparison[f"metrics.accuracy_{model_a}"]
    )

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Query MLflow experiment data")

    parser.add_argument("--all", action="store_true", help="Show all runs")
    parser.add_argument("--dataset", type=str, help="Filter by dataset")
    parser.add_argument("--stage", type=str, help="Filter by stage")
    parser.add_argument("--model-size", type=str, help="Filter by model size")
    parser.add_argument("--use-kd", action="store_true", help="Filter by KD usage")
    parser.add_argument("--status", type=str, help="Filter by status")
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard")
    parser.add_argument("--top", type=int, default=10, help="Top N results")
    parser.add_argument("--metric", type=str, default="f1", help="Metric for leaderboard")
    parser.add_argument("--running", action="store_true", help="Show running jobs")
    parser.add_argument(
        "--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"), help="Compare two models"
    )
    parser.add_argument("--limit", type=int, default=20, help="Result limit")
    parser.add_argument("--columns", type=str, help="Comma-separated columns to display")

    args = parser.parse_args()

    # Setup
    experiment_id = setup_mlflow()
    print(f"MLflow Tracking URI: {TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}\n")

    # Query
    if args.compare:
        df = compare(experiment_id, args.compare[0], args.compare[1])
        print(f"Comparison: {args.compare[0]} vs {args.compare[1]}")
    elif args.leaderboard:
        df = leaderboard(experiment_id, args.top, args.metric)
        print(f"Top {args.top} by {args.metric}:")
    elif args.running:
        df = running_jobs(experiment_id)
        print("Currently running jobs:")
    elif args.all:
        df = query_all(experiment_id, args.limit)
        print(f"All runs (limit {args.limit}):")
    else:
        df = query_filtered(
            experiment_id,
            dataset=args.dataset,
            stage=args.stage,
            model_size=args.model_size,
            use_kd=args.use_kd,
            status=args.status,
        )
        print("Filtered runs:")

    # Display
    if args.columns:
        cols = [c.strip() for c in args.columns.split(",")]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

    if len(df) == 0:
        print("No results found.")
    else:
        print(f"Found {len(df)} runs\n")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
