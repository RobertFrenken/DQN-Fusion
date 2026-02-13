"""SQLite project database for structured experiment tracking.

Provides a queryable store for dataset metadata, training runs, and evaluation
metrics — complementing MLflow (which tracks live params/metrics) and the
filesystem (which Snakemake uses for DAG triggers).

Tables:
    datasets — one row per dataset (stats from cache + catalog)
    runs     — one row per training/evaluation run
    metrics  — flattened evaluation metrics (model × scenario × metric)

Usage:
    python -m pipeline.db init          # Create schema
    python -m pipeline.db populate      # Populate from existing outputs
    python -m pipeline.db query "SQL"   # Run arbitrary SQL
    python -m pipeline.db summary       # Print dataset + run counts
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path("data/project.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    name           TEXT PRIMARY KEY,
    domain         TEXT NOT NULL,
    protocol       TEXT,
    source         TEXT,
    description    TEXT,
    num_files      INTEGER,
    num_samples    INTEGER,
    num_graphs     INTEGER,
    num_unique_ids INTEGER,
    attack_types   TEXT,
    added_by       TEXT,
    added_date     TEXT,
    cache_valid    INTEGER DEFAULT 0,
    parquet_path   TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    dataset        TEXT REFERENCES datasets(name),
    model_size     TEXT,
    stage          TEXT,
    use_kd         INTEGER,
    status         TEXT,
    teacher_run    TEXT,
    started_at     TEXT,
    completed_at   TEXT,
    mlflow_run_id  TEXT,
    config_json    TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id       TEXT REFERENCES runs(run_id),
    model        TEXT,
    scenario     TEXT,
    metric_name  TEXT,
    value        REAL,
    PRIMARY KEY (run_id, model, scenario, metric_name)
);
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns missing from older databases."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "config_json" not in cols:
        conn.execute("ALTER TABLE runs ADD COLUMN config_json TEXT")
        log.info("Migrated: added config_json column to runs table")


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection to the project database, creating schema if needed."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    _migrate_schema(conn)
    return conn


def register_datasets(stats_list: list[dict], db_path: Path | None = None) -> None:
    """Insert or update dataset entries from ingestion stats."""
    conn = get_connection(db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        for stats in stats_list:
            attack_types = json.dumps(stats.get("attack_types", []))
            conn.execute(
                """INSERT OR REPLACE INTO datasets
                   (name, domain, protocol, source, description,
                    num_files, num_samples, num_graphs, num_unique_ids,
                    attack_types, added_by, added_date, parquet_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stats["name"],
                    stats.get("domain", ""),
                    stats.get("protocol", ""),
                    stats.get("source", ""),
                    stats.get("description", ""),
                    stats.get("num_files"),
                    stats.get("num_samples"),
                    stats.get("num_graphs"),
                    stats.get("num_unique_ids"),
                    attack_types,
                    stats.get("added_by", ""),
                    now,
                    stats.get("parquet_path", ""),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def record_run_start(
    run_id: str,
    dataset: str,
    model_size: str,
    stage: str,
    use_kd: bool,
    config_json: str,
    teacher_run: str = "",
    db_path: Path | None = None,
) -> None:
    """Record the start of a run. Called from cli.py before stage dispatch."""
    conn = get_connection(db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, dataset, model_size, stage, use_kd, status,
                teacher_run, started_at, completed_at, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
            (run_id, dataset, model_size, stage,
             1 if use_kd else 0, "running", teacher_run, now, config_json),
        )
        conn.commit()
    finally:
        conn.close()


def record_run_end(
    run_id: str,
    success: bool,
    metrics: dict | None = None,
    db_path: Path | None = None,
) -> None:
    """Record run completion. Called from cli.py after stage dispatch."""
    conn = get_connection(db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        status = "complete" if success else "failed"
        conn.execute(
            "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
            (status, now, run_id),
        )

        if metrics and success:
            _insert_metrics_from_dict(conn, run_id, metrics)

        conn.commit()
    finally:
        conn.close()


def _insert_metrics_from_dict(
    conn: sqlite3.Connection, run_id: str, metrics: dict
) -> None:
    """Insert metrics from a result dict (flat or nested eval format)."""
    for model_name, model_data in metrics.items():
        if model_name == "test":
            # Test scenario metrics
            for sub_model, scenarios in model_data.items():
                if not isinstance(scenarios, dict):
                    continue
                for scenario, scenario_data in scenarios.items():
                    for section in ("core", "additional"):
                        for name, value in scenario_data.get(section, {}).items():
                            if isinstance(value, (int, float)):
                                conn.execute(
                                    """INSERT OR REPLACE INTO metrics
                                       (run_id, model, scenario, metric_name, value)
                                       VALUES (?, ?, ?, ?, ?)""",
                                    (run_id, sub_model, scenario, name, value),
                                )
        elif isinstance(model_data, dict):
            for section in ("core", "additional"):
                for name, value in model_data.get(section, {}).items():
                    if isinstance(value, (int, float)):
                        conn.execute(
                            """INSERT OR REPLACE INTO metrics
                               (run_id, model, scenario, metric_name, value)
                               VALUES (?, ?, ?, ?, ?)""",
                            (run_id, model_name, "val", name, value),
                        )


def _populate_datasets_from_cache(conn: sqlite3.Connection) -> int:
    """Populate datasets table from existing cache_metadata.json files."""
    from .ingest import load_catalog

    catalog = load_catalog()
    cache_root = Path("data/cache")
    count = 0

    for name, entry in catalog.items():
        meta_path = cache_root / name / "cache_metadata.json"
        if not meta_path.exists():
            log.warning("No cache metadata for %s, skipping", name)
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        test_subdirs = entry.get("test_subdirs", [])
        attack_types = json.dumps([
            s.replace("test_", "").lstrip("0123456789_")
            for s in test_subdirs
        ])

        parquet_dir = Path("data/parquet") / entry["domain"] / name
        parquet_path = str(parquet_dir) if parquet_dir.exists() else ""

        conn.execute(
            """INSERT OR REPLACE INTO datasets
               (name, domain, protocol, source, description,
                num_files, num_samples, num_graphs, num_unique_ids,
                attack_types, added_by, added_date, cache_valid, parquet_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                entry.get("domain", ""),
                entry.get("protocol", ""),
                entry.get("source", ""),
                entry.get("description", ""),
                meta.get("source_csv_count"),
                None,  # num_samples unknown without reading CSVs
                meta.get("num_graphs"),
                meta.get("num_unique_ids"),
                attack_types,
                entry.get("added_by", ""),
                meta.get("created_at", ""),
                1,
                parquet_path,
            ),
        )
        count += 1
        log.info("Registered dataset: %s (%d graphs, %d unique IDs)",
                 name, meta.get("num_graphs", 0), meta.get("num_unique_ids", 0))

    return count


def _populate_runs(conn: sqlite3.Connection) -> int:
    """Populate runs table from existing experimentruns/ config.json files."""
    exp_root = Path("experimentruns")
    if not exp_root.exists():
        return 0

    count = 0
    for ds_dir in sorted(exp_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            cfg_path = run_dir / "config.json"
            if not cfg_path.exists():
                continue

            config_text = cfg_path.read_text()
            cfg = json.loads(config_text)

            run_name = run_dir.name  # e.g. "teacher_autoencoder"
            dataset = ds_dir.name
            run_id = f"{dataset}/{run_name}"

            # Parse stage and model_size from directory name
            parts = run_name.split("_")
            model_size = parts[0]  # teacher or student
            use_kd = run_name.endswith("_kd")

            # Determine stage from the run name
            stage = "unknown"
            for s in ["autoencoder", "curriculum", "fusion", "evaluation"]:
                if s in run_name:
                    stage = s
                    break

            has_model = (run_dir / "best_model.pt").exists()
            has_metrics = (run_dir / "metrics.json").exists()
            status = "complete" if (has_model or has_metrics) else "unknown"

            teacher_run = ""
            if cfg.get("teacher_path"):
                tp = Path(cfg["teacher_path"])
                if len(tp.parts) >= 3:
                    teacher_run = f"{tp.parts[-3]}/{tp.parts[-2]}"

            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, dataset, model_size, stage, use_kd, status,
                    teacher_run, started_at, completed_at, config_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, dataset, model_size, stage,
                    1 if use_kd else 0, status, teacher_run,
                    None, None, config_text,
                ),
            )
            count += 1

    log.info("Registered %d runs", count)
    return count


def _populate_metrics(conn: sqlite3.Connection) -> int:
    """Populate metrics table from existing metrics.json files."""
    exp_root = Path("experimentruns")
    if not exp_root.exists():
        return 0

    count = 0
    for ds_dir in sorted(exp_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue

            run_id = f"{ds_dir.name}/{run_dir.name}"

            with open(metrics_path) as f:
                all_metrics = json.load(f)

            # Validation metrics (top-level gat, vgae, fusion)
            for model_name, model_data in all_metrics.items():
                if model_name == "test":
                    continue
                if not isinstance(model_data, dict):
                    continue
                for section in ["core", "additional"]:
                    section_data = model_data.get(section, {})
                    for metric_name, value in section_data.items():
                        if isinstance(value, (int, float)):
                            conn.execute(
                                """INSERT OR REPLACE INTO metrics
                                   (run_id, model, scenario, metric_name, value)
                                   VALUES (?, ?, ?, ?, ?)""",
                                (run_id, model_name, "val", metric_name, value),
                            )
                            count += 1

            # Test scenario metrics
            test_data = all_metrics.get("test", {})
            for model_name, scenarios in test_data.items():
                if not isinstance(scenarios, dict):
                    continue
                for scenario, scenario_data in scenarios.items():
                    for section in ["core", "additional"]:
                        section_data = scenario_data.get(section, {})
                        for metric_name, value in section_data.items():
                            if isinstance(value, (int, float)):
                                conn.execute(
                                    """INSERT OR REPLACE INTO metrics
                                       (run_id, model, scenario, metric_name, value)
                                       VALUES (?, ?, ?, ?, ?)""",
                                    (run_id, model_name, scenario,
                                     metric_name, value),
                                )
                                count += 1

    log.info("Registered %d metric entries", count)
    return count


def populate(db_path: Path | None = None) -> dict[str, int]:
    """Populate the project DB from existing filesystem outputs."""
    conn = get_connection(db_path)
    try:
        counts = {
            "datasets": _populate_datasets_from_cache(conn),
            "runs": _populate_runs(conn),
            "metrics": _populate_metrics(conn),
        }
        conn.commit()
    finally:
        conn.close()
    return counts


def query(sql: str, db_path: Path | None = None) -> tuple[list[str], list[tuple]]:
    """Execute a SQL query and return results."""
    conn = get_connection(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return columns, [tuple(r) for r in rows]
    finally:
        conn.close()


def summary(db_path: Path | None = None) -> str:
    """Print a summary of the project database."""
    conn = get_connection(db_path)
    try:
        ds_count = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        metric_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]

        lines = [
            f"Project DB: {db_path or DB_PATH}",
            f"  Datasets: {ds_count}",
            f"  Runs:     {run_count}",
            f"  Metrics:  {metric_count}",
        ]

        if ds_count > 0:
            lines.append("\n  Datasets:")
            for row in conn.execute(
                "SELECT name, domain, num_graphs, num_unique_ids FROM datasets ORDER BY name"
            ):
                lines.append(
                    f"    {row[0]:12s}  {row[1]:12s}  "
                    f"graphs={row[2] or '?':>8}  ids={row[3] or '?':>6}"
                )

        if run_count > 0:
            lines.append("\n  Runs by stage:")
            for row in conn.execute(
                "SELECT stage, COUNT(*) FROM runs GROUP BY stage ORDER BY stage"
            ):
                lines.append(f"    {row[0]:15s}  {row[1]}")

        return "\n".join(lines)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline.db",
        description="KD-GAT project database management",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create database schema")
    sub.add_parser("populate", help="Populate from existing outputs")
    sub.add_parser("summary", help="Print database summary")

    q = sub.add_parser("query", help="Run a SQL query")
    q.add_argument("sql", help="SQL query string")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )

    if args.command == "init":
        conn = get_connection()
        conn.close()
        log.info("Database initialized at %s", DB_PATH)

    elif args.command == "populate":
        counts = populate()
        log.info("Populated: %s", counts)

    elif args.command == "summary":
        print(summary())

    elif args.command == "query":
        columns, rows = query(args.sql)
        if columns:
            # Print header
            header = "  ".join(f"{c:>15}" for c in columns)
            print(header)
            print("-" * len(header))
            for row in rows:
                print("  ".join(f"{str(v):>15}" for v in row))
        print(f"\n({len(rows)} rows)")


if __name__ == "__main__":
    main()
