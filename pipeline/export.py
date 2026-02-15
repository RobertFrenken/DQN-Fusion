"""Export project DB to static JSON for the GitHub Pages dashboard.

Usage:
    python -m pipeline.export [--output-dir docs/dashboard/data]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .db import get_connection

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("docs/dashboard/data")


def export_leaderboard(output_dir: Path) -> Path:
    """Best F1/accuracy per model x dataset x scale."""
    conn = get_connection()
    conn.row_factory = _dict_factory
    rows = conn.execute("""
        SELECT
            r.dataset, r.model_type, r.scale, r.has_kd,
            m.model, m.metric_name,
            ROUND(MAX(m.value), 6) AS best_value
        FROM metrics m
        JOIN runs r ON r.run_id = m.run_id
        WHERE m.scenario = 'val'
          AND m.metric_name IN ('f1', 'accuracy', 'precision', 'recall', 'auc', 'mcc')
          AND r.status = 'complete'
        GROUP BY r.dataset, r.model_type, r.scale, r.has_kd, m.model, m.metric_name
        ORDER BY r.dataset, m.model, m.metric_name
    """).fetchall()
    conn.close()

    out = output_dir / "leaderboard.json"
    out.write_text(json.dumps(rows, indent=2))
    log.info("Exported %d leaderboard entries → %s", len(rows), out)
    return out


def export_runs(output_dir: Path) -> Path:
    """All completed runs with config and status."""
    conn = get_connection()
    conn.row_factory = _dict_factory
    rows = conn.execute("""
        SELECT
            run_id, dataset, model_type, scale, stage,
            has_kd, status, teacher_run, started_at, completed_at
        FROM runs
        ORDER BY started_at DESC
    """).fetchall()
    conn.close()

    out = output_dir / "runs.json"
    out.write_text(json.dumps(rows, indent=2))
    log.info("Exported %d runs → %s", len(rows), out)
    return out


def export_metrics(output_dir: Path) -> Path:
    """Per-run flattened metrics."""
    conn = get_connection()
    conn.row_factory = _dict_factory

    run_ids = [r["run_id"] for r in conn.execute(
        "SELECT DISTINCT run_id FROM metrics"
    ).fetchall()]

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for rid in run_ids:
        rows = conn.execute(
            "SELECT model, scenario, metric_name, value FROM metrics WHERE run_id = ?",
            (rid,),
        ).fetchall()
        # Use sanitized filename
        fname = rid.replace("/", "_") + ".json"
        (metrics_dir / fname).write_text(json.dumps(rows, indent=2))

    conn.close()
    log.info("Exported metrics for %d runs → %s", len(run_ids), metrics_dir)
    return metrics_dir


def export_datasets(output_dir: Path) -> Path:
    """Dataset metadata."""
    conn = get_connection()
    conn.row_factory = _dict_factory
    rows = conn.execute("""
        SELECT name, domain, protocol, source, description,
               num_files, num_samples, num_graphs, num_unique_ids,
               attack_types
        FROM datasets
        ORDER BY name
    """).fetchall()
    conn.close()

    out = output_dir / "datasets.json"
    out.write_text(json.dumps(rows, indent=2))
    log.info("Exported %d datasets → %s", len(rows), out)
    return out


def export_kd_transfer(output_dir: Path) -> Path:
    """Teacher vs student metric pairs for KD analysis."""
    conn = get_connection()
    conn.row_factory = _dict_factory

    rows = conn.execute("""
        SELECT
            student.run_id   AS student_run,
            student.dataset,
            student.model_type,
            student.scale     AS student_scale,
            student.teacher_run,
            sm.metric_name,
            ROUND(sm.value, 6) AS student_value,
            ROUND(tm.value, 6) AS teacher_value
        FROM runs student
        JOIN runs teacher ON teacher.run_id = student.teacher_run
        JOIN metrics sm ON sm.run_id = student.run_id AND sm.scenario = 'val'
        JOIN metrics tm ON tm.run_id = teacher.run_id
                       AND tm.scenario = 'val'
                       AND tm.metric_name = sm.metric_name
                       AND tm.model = sm.model
        WHERE student.has_kd = 1
          AND sm.metric_name IN ('f1', 'accuracy', 'auc')
        ORDER BY student.dataset, sm.metric_name
    """).fetchall()
    conn.close()

    out = output_dir / "kd_transfer.json"
    out.write_text(json.dumps(rows, indent=2))
    log.info("Exported %d KD transfer pairs → %s", len(rows), out)
    return out


def export_training_curves(output_dir: Path) -> Path:
    """Per-run training curves from epoch_metrics table."""
    conn = get_connection()
    conn.row_factory = _dict_factory

    run_ids = [r["run_id"] for r in conn.execute(
        "SELECT DISTINCT run_id FROM epoch_metrics"
    ).fetchall()]

    curves_dir = output_dir / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    for rid in run_ids:
        rows = conn.execute(
            "SELECT epoch, metric_name, value FROM epoch_metrics WHERE run_id = ? ORDER BY epoch, metric_name",
            (rid,),
        ).fetchall()
        fname = rid.replace("/", "_") + ".json"
        (curves_dir / fname).write_text(json.dumps(rows, indent=2))

    conn.close()
    log.info("Exported training curves for %d runs → %s", len(run_ids), curves_dir)
    return curves_dir


def export_all(output_dir: Path) -> None:
    """Run all exports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    export_leaderboard(output_dir)
    export_runs(output_dir)
    export_metrics(output_dir)
    export_datasets(output_dir)
    export_kd_transfer(output_dir)
    export_training_curves(output_dir)
    log.info("All exports complete → %s", output_dir)


def _dict_factory(cursor, row):
    """SQLite row factory that returns dicts."""
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline.export",
        description="Export project DB to static JSON for dashboard",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )

    export_all(args.output_dir)


if __name__ == "__main__":
    main()
