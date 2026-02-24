"""Build analytics DuckDB from lakehouse JSON + experiment run metadata.

Reads:
  - data/lakehouse/runs/*.json  (fire-and-forget lakehouse records)
  - experimentruns/*/config.json + metrics.json  (filesystem experiment outputs)

Writes:
  - data/lakehouse/analytics.duckdb

Tables:
  - runs          Per-run metadata (dataset, model, scale, stage, kd, status, timestamps)
  - metrics       Flattened core metrics per run × model (gat, vgae, fusion)
  - datasets      Dataset catalog from config/datasets.yaml
  - configs       Frozen PipelineConfig per run (key hyperparameters only)

Usage:
    python -m pipeline.build_analytics              # Full rebuild
    python -m pipeline.build_analytics --dry-run    # Show what would be built
    duckdb data/lakehouse/analytics.duckdb          # Interactive queries

Example queries:
    SELECT dataset, model_type, scale, has_kd,
           MAX(gat_f1) AS best_f1
    FROM runs JOIN metrics USING (run_id)
    WHERE stage = 'evaluation' AND success
    GROUP BY ALL ORDER BY best_f1 DESC;
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

EXPERIMENT_ROOT = Path("experimentruns")
_DATA_ROOT: str | None = os.environ.get("KD_GAT_DATA_ROOT")

# Core metrics to extract from nested metrics.json per model type.
# Maps: flat column name -> (model_key, section, metric_name)
CORE_METRIC_COLS = [
    "accuracy", "precision", "recall", "f1", "specificity",
    "balanced_accuracy", "mcc", "fpr", "fnr", "auc", "n_samples",
]


def _analytics_db_path() -> Path:
    """Path to the analytics DuckDB file."""
    if _DATA_ROOT:
        return Path(_DATA_ROOT) / "lakehouse" / "analytics.duckdb"
    return EXPERIMENT_ROOT / "analytics.duckdb"


def _scan_experiment_runs() -> list[dict]:
    """Scan experimentruns/ for completed runs, return structured metadata."""
    runs = []
    if not EXPERIMENT_ROOT.is_dir():
        return runs

    for ds_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            cfg_path = run_dir / "config.json"
            if not cfg_path.exists():
                continue
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                continue

            parts = run_dir.name.split("_")
            model_type = cfg.get("model_type") or (parts[0] if parts else "unknown")
            scale = cfg.get("scale") or (parts[1] if len(parts) > 1 else "unknown")

            _AUX_SUFFIXES = {"kd", "nokd"}
            if cfg.get("stage"):
                stage = cfg["stage"]
            else:
                remaining = parts[2:]
                if remaining and remaining[-1] in _AUX_SUFFIXES:
                    remaining = remaining[:-1]
                stage = "_".join(remaining)

            has_kd = bool(cfg.get("auxiliaries")) or (
                "_kd" in run_dir.name and "nokd" not in run_dir.name
            )

            run_id = f"{ds_dir.name}/{run_dir.name}"
            metrics_path = run_dir / "metrics.json"
            has_metrics = metrics_path.exists()
            has_checkpoint = (run_dir / "best_model.pt").exists()
            has_done = (run_dir / ".done").exists()

            # Get timestamps from filesystem
            mtime = run_dir.stat().st_mtime
            completed_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

            run_record = {
                "run_id": run_id,
                "dataset": ds_dir.name,
                "model_type": model_type,
                "scale": scale,
                "stage": stage,
                "has_kd": has_kd,
                "success": has_metrics or has_done,
                "has_checkpoint": has_checkpoint,
                "has_metrics": has_metrics,
                "completed_at": completed_at,
                "source": "filesystem",
            }

            # Extract core metrics from metrics.json
            metrics_records = []
            if has_metrics:
                try:
                    metrics = json.loads(metrics_path.read_text())
                    for model_key, model_data in metrics.items():
                        if model_key == "test":
                            continue
                        if isinstance(model_data, dict) and "core" in model_data:
                            core = model_data["core"]
                            record = {
                                "run_id": run_id,
                                "model": model_key,
                            }
                            for col in CORE_METRIC_COLS:
                                val = core.get(col)
                                if isinstance(val, (int, float)):
                                    record[col] = float(val)
                            metrics_records.append(record)
                except Exception as e:
                    log.warning("Failed to parse metrics for %s: %s", run_id, e)

            # Extract key hyperparameters
            config_record = {
                "run_id": run_id,
                "lr": cfg.get("training", {}).get("lr"),
                "batch_size": cfg.get("training", {}).get("batch_size"),
                "max_epochs": cfg.get("training", {}).get("max_epochs"),
                "patience": cfg.get("training", {}).get("patience"),
                "precision": cfg.get("training", {}).get("precision"),
                "dynamic_batching": cfg.get("training", {}).get("dynamic_batching"),
                "vgae_latent_dim": cfg.get("vgae", {}).get("latent_dim"),
                "gat_hidden": cfg.get("gat", {}).get("hidden"),
                "gat_layers": cfg.get("gat", {}).get("layers"),
                "gat_heads": cfg.get("gat", {}).get("heads"),
                "kd_alpha": None,
                "kd_temperature": None,
            }
            if has_kd and cfg.get("auxiliaries"):
                aux = cfg["auxiliaries"][0]
                config_record["kd_alpha"] = aux.get("alpha")
                config_record["kd_temperature"] = aux.get("temperature")

            runs.append({
                "run": run_record,
                "metrics": metrics_records,
                "config": config_record,
            })

    return runs


def _scan_lakehouse_json() -> list[dict]:
    """Read lakehouse JSON files for any runs not captured by filesystem scan."""
    lakehouse_dir = Path(_DATA_ROOT) / "lakehouse" / "runs" if _DATA_ROOT else EXPERIMENT_ROOT / "lakehouse"
    if not lakehouse_dir.is_dir():
        return []

    records = []
    for f in sorted(lakehouse_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            records.append(data)
        except Exception as e:
            log.warning("Failed to parse lakehouse JSON %s: %s", f.name, e)
    return records


def _load_datasets_catalog() -> list[dict]:
    """Load dataset catalog from config/datasets.yaml."""
    catalog_path = Path("config/datasets.yaml")
    if not catalog_path.exists():
        return []

    import yaml
    raw = yaml.safe_load(catalog_path.read_text())
    datasets = []
    for name, info in raw.items():
        datasets.append({
            "name": name,
            "domain": info.get("domain", ""),
            "protocol": info.get("protocol", ""),
            "source": info.get("source", ""),
        })
    return datasets


def build(dry_run: bool = False) -> Path:
    """Build (or rebuild) analytics.duckdb from all available data sources."""
    import duckdb

    db_path = _analytics_db_path()

    # Scan data sources
    fs_runs = _scan_experiment_runs()
    lakehouse_records = _scan_lakehouse_json()
    datasets_catalog = _load_datasets_catalog()

    log.info(
        "Found %d filesystem runs, %d lakehouse records, %d datasets",
        len(fs_runs), len(lakehouse_records), len(datasets_catalog),
    )

    if dry_run:
        print(f"Would write to: {db_path}")
        print(f"  Filesystem runs: {len(fs_runs)}")
        print(f"  Lakehouse records: {len(lakehouse_records)}")
        print(f"  Datasets: {len(datasets_catalog)}")
        for r in fs_runs:
            run = r["run"]
            n_metrics = len(r["metrics"])
            print(f"  {run['run_id']:50s} stage={run['stage']:12s} metrics={n_metrics}")
        return db_path

    # Remove old DB and create fresh
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        _create_schema(con)
        _insert_datasets(con, datasets_catalog)
        _insert_runs(con, fs_runs, lakehouse_records)
        _insert_metrics(con, fs_runs)
        _insert_configs(con, fs_runs)

        # Summary
        counts = {}
        for table in ["runs", "metrics", "datasets", "configs"]:
            counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        log.info("Analytics DB built: %s", counts)
        print(f"Built {db_path} ({db_path.stat().st_size / 1024:.0f} KB)")
        for table, count in counts.items():
            print(f"  {table}: {count} rows")
    finally:
        con.close()

    return db_path


def _create_schema(con) -> None:
    """Create analytics tables."""
    con.execute("""
        CREATE TABLE datasets (
            name VARCHAR PRIMARY KEY,
            domain VARCHAR,
            protocol VARCHAR,
            source VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE runs (
            run_id VARCHAR PRIMARY KEY,
            dataset VARCHAR,
            model_type VARCHAR,
            scale VARCHAR,
            stage VARCHAR,
            has_kd BOOLEAN,
            success BOOLEAN,
            has_checkpoint BOOLEAN,
            has_metrics BOOLEAN,
            completed_at TIMESTAMP WITH TIME ZONE,
            source VARCHAR,
            timestamp TIMESTAMP WITH TIME ZONE,
            failure_reason VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE metrics (
            run_id VARCHAR,
            model VARCHAR,
            accuracy DOUBLE,
            "precision" DOUBLE,
            recall DOUBLE,
            f1 DOUBLE,
            specificity DOUBLE,
            balanced_accuracy DOUBLE,
            mcc DOUBLE,
            fpr DOUBLE,
            fnr DOUBLE,
            auc DOUBLE,
            n_samples INTEGER,
            PRIMARY KEY (run_id, model)
        )
    """)

    con.execute("""
        CREATE TABLE configs (
            run_id VARCHAR PRIMARY KEY,
            lr DOUBLE,
            batch_size INTEGER,
            max_epochs INTEGER,
            patience INTEGER,
            "precision" VARCHAR,
            dynamic_batching BOOLEAN,
            vgae_latent_dim INTEGER,
            gat_hidden INTEGER,
            gat_layers INTEGER,
            gat_heads INTEGER,
            kd_alpha DOUBLE,
            kd_temperature DOUBLE
        )
    """)


def _insert_datasets(con, datasets: list[dict]) -> None:
    """Insert dataset catalog rows."""
    if not datasets:
        return
    con.executemany(
        "INSERT INTO datasets VALUES (?, ?, ?, ?)",
        [(d["name"], d["domain"], d["protocol"], d["source"]) for d in datasets],
    )


def _insert_runs(con, fs_runs: list[dict], lakehouse_records: list[dict]) -> None:
    """Insert run records from filesystem scan + lakehouse JSON (deduplicated)."""
    seen = set()

    # Filesystem runs first (canonical)
    for r in fs_runs:
        run = r["run"]
        run_id = run["run_id"]
        if run_id in seen:
            continue
        seen.add(run_id)
        con.execute(
            """INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                run["dataset"],
                run["model_type"],
                run["scale"],
                run["stage"],
                run["has_kd"],
                run["success"],
                run.get("has_checkpoint", False),
                run.get("has_metrics", False),
                run.get("completed_at"),
                run["source"],
                None,  # timestamp (from lakehouse)
                None,  # failure_reason
            ),
        )

    # Lakehouse records (fill in any missing runs + add timestamp/failure info)
    for rec in lakehouse_records:
        run_id = rec.get("run_id", "")
        if run_id in seen:
            # Update timestamp if we have it from lakehouse
            if rec.get("timestamp"):
                con.execute(
                    "UPDATE runs SET timestamp = ? WHERE run_id = ?",
                    (rec["timestamp"], run_id),
                )
            continue
        seen.add(run_id)
        con.execute(
            """INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                rec.get("dataset", ""),
                rec.get("model_type", ""),
                rec.get("scale", ""),
                rec.get("stage", ""),
                rec.get("has_kd", False),
                rec.get("success", False),
                False,
                False,
                None,
                "lakehouse",
                rec.get("timestamp"),
                rec.get("failure_reason"),
            ),
        )


def _insert_metrics(con, fs_runs: list[dict]) -> None:
    """Insert flattened core metrics per run x model."""
    for r in fs_runs:
        for m in r["metrics"]:
            con.execute(
                """INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    m["run_id"],
                    m["model"],
                    m.get("accuracy"),
                    m.get("precision"),
                    m.get("recall"),
                    m.get("f1"),
                    m.get("specificity"),
                    m.get("balanced_accuracy"),
                    m.get("mcc"),
                    m.get("fpr"),
                    m.get("fnr"),
                    m.get("auc"),
                    int(m["n_samples"]) if m.get("n_samples") is not None else None,
                ),
            )


def _insert_configs(con, fs_runs: list[dict]) -> None:
    """Insert key hyperparameters per run."""
    for r in fs_runs:
        c = r["config"]
        con.execute(
            """INSERT INTO configs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                c["run_id"],
                c.get("lr"),
                c.get("batch_size"),
                c.get("max_epochs"),
                c.get("patience"),
                c.get("precision"),
                c.get("dynamic_batching"),
                c.get("vgae_latent_dim"),
                c.get("gat_hidden"),
                c.get("gat_layers"),
                c.get("gat_heads"),
                c.get("kd_alpha"),
                c.get("kd_temperature"),
            ),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build analytics DuckDB from experiment data")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be built")
    args = parser.parse_args()
    build(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
