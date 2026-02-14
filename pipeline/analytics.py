"""Analytical query layer over the project database.

Complements MLflow UI — use MLflow for visual metric comparison and artifact
browsing; use this module for hyperparameter sweeps, arbitrary SQL joins on
config_json, and headless/scriptable analysis. Datasette provides interactive
point-and-click DB exploration.

Provides functions for post-run experiment analysis using SQLite json_extract()
on the config_json column in the runs table. All queries run against the local
project DB (data/project.db) — no MLflow or network dependency.

Usage:
    python -m pipeline.analytics sweep --param lr --metric f1
    python -m pipeline.analytics leaderboard --metric f1 --top 10
    python -m pipeline.analytics compare <run_a> <run_b>
    python -m pipeline.analytics diff <run_a> <run_b>
    python -m pipeline.analytics dataset <name>
    python -m pipeline.analytics query "SELECT json_extract(...) FROM ..."
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from .db import DB_PATH, get_connection

_SAFE_PARAM = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_param(name: str) -> str:
    """Validate parameter name to prevent SQL injection in json_extract paths."""
    if not _SAFE_PARAM.match(name):
        raise ValueError(
            f"Invalid parameter name {name!r}: must be alphanumeric + underscore"
        )
    return name


def sweep(
    param: str,
    metric: str,
    *,
    dataset: str | None = None,
    stage: str | None = None,
    db_path: Path | None = None,
) -> list[dict]:
    """Group runs by a hyperparameter value and show metric statistics.

    Returns list of dicts with keys: param_value, count, min, max, mean.
    """
    param = _validate_param(param)
    metric = _validate_param(metric)

    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row

    where_parts = ["r.config_json IS NOT NULL"]
    params: list = []

    if dataset:
        where_parts.append("r.dataset = ?")
        params.append(dataset)
    if stage:
        where_parts.append("r.stage = ?")
        params.append(stage)

    where = " AND ".join(where_parts)

    sql = f"""
        SELECT
            json_extract(r.config_json, '$.{param}') AS param_value,
            COUNT(*)                                  AS count,
            MIN(m.value)                              AS min,
            MAX(m.value)                              AS max,
            ROUND(AVG(m.value), 6)                    AS mean
        FROM runs r
        JOIN metrics m ON m.run_id = r.run_id
        WHERE {where}
          AND m.metric_name = ?
          AND json_extract(r.config_json, '$.{param}') IS NOT NULL
        GROUP BY param_value
        ORDER BY mean DESC
    """
    params.append(metric)

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def leaderboard(
    metric: str,
    *,
    top: int = 10,
    scenario: str | None = None,
    model: str | None = None,
    dataset: str | None = None,
    stage: str | None = None,
    db_path: Path | None = None,
) -> list[dict]:
    """Rank runs by a metric value (descending).

    Returns list of dicts with keys: run_id, dataset, stage, model, scenario, value.
    """
    metric = _validate_param(metric)
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row

    where_parts = ["m.metric_name = ?"]
    params: list = [metric]

    if scenario:
        where_parts.append("m.scenario = ?")
        params.append(scenario)
    if model:
        where_parts.append("m.model = ?")
        params.append(model)
    if dataset:
        where_parts.append("r.dataset = ?")
        params.append(dataset)
    if stage:
        where_parts.append("r.stage = ?")
        params.append(stage)

    where = " AND ".join(where_parts)

    sql = f"""
        SELECT
            r.run_id, r.dataset, r.stage,
            m.model, m.scenario,
            ROUND(m.value, 6) AS value
        FROM metrics m
        JOIN runs r ON r.run_id = m.run_id
        WHERE {where}
        ORDER BY m.value DESC
        LIMIT ?
    """
    params.append(top)

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def compare(
    run_a: str,
    run_b: str,
    *,
    db_path: Path | None = None,
) -> list[dict]:
    """Side-by-side metric comparison between two runs.

    Returns list of dicts with keys: model, scenario, metric_name, value_a, value_b, delta.
    """
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row

    # Verify both runs exist
    for rid in (run_a, run_b):
        if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (rid,)).fetchone() is None:
            conn.close()
            raise KeyError(f"Run not found: {rid!r}")

    # Emulate FULL OUTER JOIN using UNION of LEFT JOINs (compatible with all SQLite versions)
    sql = """
        SELECT
            COALESCE(a.model, b.model)             AS model,
            COALESCE(a.scenario, b.scenario)       AS scenario,
            COALESCE(a.metric_name, b.metric_name) AS metric_name,
            a.value                                AS value_a,
            b.value                                AS value_b,
            ROUND(b.value - a.value, 6)            AS delta
        FROM
            (SELECT * FROM metrics WHERE run_id = ?) a
        LEFT JOIN
            (SELECT * FROM metrics WHERE run_id = ?) b
        ON a.model = b.model AND a.scenario = b.scenario AND a.metric_name = b.metric_name
        UNION
        SELECT
            COALESCE(a2.model, b2.model)             AS model,
            COALESCE(a2.scenario, b2.scenario)       AS scenario,
            COALESCE(a2.metric_name, b2.metric_name) AS metric_name,
            a2.value                                 AS value_a,
            b2.value                                 AS value_b,
            ROUND(b2.value - a2.value, 6)            AS delta
        FROM
            (SELECT * FROM metrics WHERE run_id = ?) b2
        LEFT JOIN
            (SELECT * FROM metrics WHERE run_id = ?) a2
        ON b2.model = a2.model AND b2.scenario = a2.scenario AND b2.metric_name = a2.metric_name
        WHERE a2.run_id IS NULL
        ORDER BY model, scenario, metric_name
    """
    rows = conn.execute(sql, (run_a, run_b, run_b, run_a)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def config_diff(
    run_a: str,
    run_b: str,
    *,
    db_path: Path | None = None,
) -> list[dict]:
    """Show differing hyperparameters between two runs.

    Returns list of dicts with keys: param, value_a, value_b.
    """
    conn = get_connection(db_path)

    results = {}
    for label, rid in [("a", run_a), ("b", run_b)]:
        row = conn.execute(
            "SELECT config_json FROM runs WHERE run_id = ?", (rid,)
        ).fetchone()
        if row is None:
            conn.close()
            raise KeyError(f"Run not found: {rid!r}")
        results[label] = json.loads(row[0]) if row[0] else {}

    conn.close()
    cfg_a, cfg_b = results["a"], results["b"]

    all_keys = sorted(set(cfg_a) | set(cfg_b))
    diffs = []
    for key in all_keys:
        va = cfg_a.get(key)
        vb = cfg_b.get(key)
        if va != vb:
            diffs.append({"param": key, "value_a": va, "value_b": vb})

    return diffs


def dataset_summary(
    dataset: str,
    *,
    db_path: Path | None = None,
) -> dict:
    """All runs and best metrics for a dataset.

    Returns dict with keys: dataset, runs (list), best_metrics (dict of metric→best row).
    """
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row

    runs = conn.execute(
        "SELECT run_id, model_type, scale, stage, has_kd, status FROM runs WHERE dataset = ? ORDER BY run_id",
        (dataset,),
    ).fetchall()

    best = conn.execute(
        """
        SELECT m.metric_name, m.run_id, m.model, m.scenario,
               ROUND(MAX(m.value), 6) AS best_value
        FROM metrics m
        JOIN runs r ON r.run_id = m.run_id
        WHERE r.dataset = ?
          AND m.metric_name IN ('f1', 'accuracy', 'auc', 'mcc')
        GROUP BY m.metric_name
        ORDER BY m.metric_name
        """,
        (dataset,),
    ).fetchall()

    conn.close()
    return {
        "dataset": dataset,
        "runs": [dict(r) for r in runs],
        "best_metrics": {row["metric_name"]: dict(row) for row in best},
    }


def memory_prediction_summary(
    *,
    model_type: str | None = None,
    dataset: str | None = None,
    db_path: Path | None = None,
) -> dict:
    """Summarize memory prediction accuracy across runs.

    Queries ``memory/prediction_ratio`` from MLflow-logged metrics.
    Interpretation: <0.9 = over-allocating, >1.1 = OOM risk, else well-calibrated.

    Returns dict with keys: count, mean, min, max, std, interpretation.
    """
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row

    where_parts = ["m.metric_name = 'memory/prediction_ratio'"]
    params: list = []

    if model_type:
        where_parts.append("r.model_type = ?")
        params.append(model_type)
    if dataset:
        where_parts.append("r.dataset = ?")
        params.append(dataset)

    where = " AND ".join(where_parts)

    sql = f"""
        SELECT
            COUNT(*)           AS count,
            ROUND(AVG(m.value), 4) AS mean,
            ROUND(MIN(m.value), 4) AS min,
            ROUND(MAX(m.value), 4) AS max,
            ROUND(
                SQRT(AVG(m.value * m.value) - AVG(m.value) * AVG(m.value)),
                4
            ) AS std
        FROM metrics m
        JOIN runs r ON r.run_id = m.run_id
        WHERE {where}
    """
    row = conn.execute(sql, params).fetchone()
    conn.close()

    if row is None or row["count"] == 0:
        return {"count": 0, "message": "No memory prediction data found"}

    mean = row["mean"]
    if mean < 0.9:
        interp = "over-allocating (predicted > actual)"
    elif mean > 1.1:
        interp = "OOM risk (actual > predicted)"
    else:
        interp = "well-calibrated"

    return {
        "count": row["count"],
        "mean": mean,
        "min": row["min"],
        "max": row["max"],
        "std": row["std"],
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict], title: str | None = None) -> None:
    """Pretty-print a list of dicts as an aligned table."""
    if not rows:
        print("(no results)")
        return
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    keys = list(rows[0].keys())
    widths = {k: max(len(str(k)), *(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = "  ".join(f"{k:>{widths[k]}}" for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f"{str(row.get(k, '')):>{widths[k]}}" for k in keys))
    print(f"\n({len(rows)} rows)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline.analytics",
        description="KD-GAT experiment analytics",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # sweep
    p = sub.add_parser("sweep", help="Group by hyperparameter, show metric stats")
    p.add_argument("--param", required=True, help="Hyperparameter name (from config.json)")
    p.add_argument("--metric", required=True, help="Metric name to aggregate")
    p.add_argument("--dataset", help="Filter by dataset")
    p.add_argument("--stage", help="Filter by stage")

    # leaderboard
    p = sub.add_parser("leaderboard", help="Rank runs by metric")
    p.add_argument("--metric", required=True, help="Metric name to rank by")
    p.add_argument("--top", type=int, default=10, help="Number of results (default: 10)")
    p.add_argument("--scenario", help="Filter by test scenario")
    p.add_argument("--model", help="Filter by model (gat, vgae, fusion)")
    p.add_argument("--dataset", help="Filter by dataset")
    p.add_argument("--stage", help="Filter by stage")

    # compare
    p = sub.add_parser("compare", help="Side-by-side metric comparison")
    p.add_argument("run_a", help="First run ID (e.g. hcrl_sa/teacher_evaluation)")
    p.add_argument("run_b", help="Second run ID")

    # diff
    p = sub.add_parser("diff", help="Show differing hyperparameters between two runs")
    p.add_argument("run_a", help="First run ID")
    p.add_argument("run_b", help="Second run ID")

    # dataset
    p = sub.add_parser("dataset", help="All runs + best metrics for a dataset")
    p.add_argument("name", help="Dataset name")

    # query
    p = sub.add_parser("query", help="Run arbitrary SQL")
    p.add_argument("sql", help="SQL query string")

    # memory
    p = sub.add_parser("memory", help="Memory prediction accuracy summary")
    p.add_argument("--model", help="Filter by model type (vgae, gat, dqn)")
    p.add_argument("--dataset", help="Filter by dataset")

    args = parser.parse_args(argv)

    if args.command == "sweep":
        rows = sweep(args.param, args.metric, dataset=args.dataset, stage=args.stage)
        _print_table(rows, f"Sweep: {args.param} → {args.metric}")

    elif args.command == "leaderboard":
        rows = leaderboard(
            args.metric, top=args.top, scenario=args.scenario,
            model=args.model, dataset=args.dataset, stage=args.stage,
        )
        _print_table(rows, f"Leaderboard: {args.metric} (top {args.top})")

    elif args.command == "compare":
        rows = compare(args.run_a, args.run_b)
        _print_table(rows, f"Compare: {args.run_a} vs {args.run_b}")

    elif args.command == "diff":
        rows = config_diff(args.run_a, args.run_b)
        _print_table(rows, f"Config diff: {args.run_a} vs {args.run_b}")

    elif args.command == "dataset":
        result = dataset_summary(args.name)
        print(f"\nDataset: {result['dataset']}")
        print(f"Runs: {len(result['runs'])}")
        if result["runs"]:
            _print_table(result["runs"], "Runs")
        if result["best_metrics"]:
            best_rows = list(result["best_metrics"].values())
            _print_table(best_rows, "Best metrics")

    elif args.command == "memory":
        result = memory_prediction_summary(
            model_type=args.model, dataset=args.dataset,
        )
        print(f"\nMemory Prediction Summary")
        print("=" * 30)
        if result.get("count", 0) == 0:
            print(result.get("message", "No data"))
        else:
            print(f"  Runs:           {result['count']}")
            print(f"  Mean ratio:     {result['mean']}")
            print(f"  Min:            {result['min']}")
            print(f"  Max:            {result['max']}")
            print(f"  Std:            {result['std']}")
            print(f"  Assessment:     {result['interpretation']}")

    elif args.command == "query":
        from .db import query
        columns, rows = query(args.sql)
        if columns:
            header = "  ".join(f"{c:>15}" for c in columns)
            print(header)
            print("-" * len(header))
            for row in rows:
                print("  ".join(f"{str(v):>15}" for v in row))
        print(f"\n({len(rows)} rows)")


if __name__ == "__main__":
    main()
