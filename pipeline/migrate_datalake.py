"""One-time migration: build Parquet datalake from existing experiment runs.

Reads:
  - experimentruns/{dataset}/{run}/config.json + metrics.json
  - data/cache/{dataset}/cache_metadata.json
  - config/datasets.yaml

Writes:
  - data/datalake/runs.parquet
  - data/datalake/metrics.parquet
  - data/datalake/configs.parquet
  - data/datalake/datasets.parquet
  - data/datalake/artifacts.parquet
  - data/datalake/training_curves/{run_id}.parquet
  - data/datalake/analytics.duckdb  (views over Parquet)

Idempotent — safe to re-run. Overwrites existing Parquet files.

Usage:
    python -m pipeline.migrate_datalake              # Full migration
    python -m pipeline.migrate_datalake --dry-run    # Show what would be built
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

EXPERIMENT_ROOT = Path("experimentruns")
CACHE_ROOT = Path("data/cache")
DATALAKE_ROOT = Path("data/datalake")

# Core metrics extracted from nested metrics.json per model type.
CORE_METRIC_COLS = [
    "accuracy", "precision", "recall", "f1", "specificity",
    "balanced_accuracy", "mcc", "fpr", "fnr", "auc", "n_samples",
]


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def _scan_runs() -> list[dict]:
    """Scan experimentruns/ for all completed runs."""
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

            # Timestamps from filesystem
            mtime = run_dir.stat().st_mtime
            completed_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            started_at = None
            if cfg_path.exists():
                started_at = datetime.fromtimestamp(
                    cfg_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()

            # Duration estimate from config.json mtime → metrics.json/best_model.pt mtime
            duration_seconds = None
            for end_file in ("metrics.json", "best_model.pt"):
                p = run_dir / end_file
                if p.exists():
                    duration_seconds = p.stat().st_mtime - cfg_path.stat().st_mtime
                    break

            metrics_path = run_dir / "metrics.json"
            has_metrics = metrics_path.exists()
            has_checkpoint = (run_dir / "best_model.pt").exists()

            # Auxiliaries list to string for KD info
            auxiliaries = ""
            if cfg.get("auxiliaries"):
                aux_types = [a.get("type", "") for a in cfg["auxiliaries"]]
                auxiliaries = ",".join(aux_types)

            runs.append({
                "run_id": run_id,
                "dataset": ds_dir.name,
                "model_type": model_type,
                "scale": scale,
                "stage": stage,
                "has_kd": has_kd,
                "auxiliaries": auxiliaries,
                "success": has_metrics or has_checkpoint,
                "completed_at": completed_at,
                "started_at": started_at,
                "duration_seconds": duration_seconds,
                "data_version": None,  # TODO: read from DVC if available
                "wandb_run_id": None,  # TODO: parse from wandb/ dir
                "source": "filesystem",
                "run_dir": run_dir,
                "config": cfg,
                "has_metrics": has_metrics,
                "metrics_path": metrics_path if has_metrics else None,
            })
    return runs


def _extract_metrics(runs: list[dict]) -> list[dict]:
    """Extract per-run per-model core metrics from metrics.json files."""
    records = []
    for run in runs:
        if not run["has_metrics"]:
            continue
        try:
            metrics = json.loads(run["metrics_path"].read_text())
        except Exception as e:
            log.warning("Failed to parse metrics for %s: %s", run["run_id"], e)
            continue

        for model_key, model_data in metrics.items():
            if model_key == "test":
                continue
            if isinstance(model_data, dict) and "core" in model_data:
                core = model_data["core"]
                record = {"run_id": run["run_id"], "model": model_key}
                for col in CORE_METRIC_COLS:
                    val = core.get(col)
                    record[col] = float(val) if isinstance(val, (int, float)) else None
                records.append(record)
    return records


def _extract_configs(runs: list[dict]) -> list[dict]:
    """Extract key hyperparameters from config.json files."""
    records = []
    for run in runs:
        cfg = run["config"]
        has_kd = run["has_kd"]

        kd_alpha = None
        kd_temperature = None
        if has_kd and cfg.get("auxiliaries"):
            aux = cfg["auxiliaries"][0]
            kd_alpha = aux.get("alpha")
            kd_temperature = aux.get("temperature")

        records.append({
            "run_id": run["run_id"],
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
            "kd_alpha": kd_alpha,
            "kd_temperature": kd_temperature,
            "full_config_json": json.dumps(cfg, default=str),
        })
    return records


def _extract_artifacts(runs: list[dict]) -> list[dict]:
    """Build artifact manifest from run directories."""
    ARTIFACT_TYPES = {
        "best_model.pt": "checkpoint",
        "embeddings.npz": "embeddings",
        "attention_weights.npz": "attention",
        "dqn_policy.json": "policy",
        "metrics.json": "metrics",
        "config.json": "config",
        "explanations.npz": "explanations",
        "cka_matrix.json": "cka",
    }
    records = []
    for run in runs:
        run_dir = run["run_dir"]
        for filename, artifact_type in ARTIFACT_TYPES.items():
            fpath = run_dir / filename
            if fpath.exists():
                records.append({
                    "run_id": run["run_id"],
                    "artifact_type": artifact_type,
                    "file_path": str(fpath),
                    "file_size_bytes": fpath.stat().st_size,
                })
    return records


def _load_datasets_catalog() -> list[dict]:
    """Load dataset catalog from config/datasets.yaml + cache metadata."""
    import yaml

    catalog_path = Path("config/datasets.yaml")
    if not catalog_path.exists():
        return []

    raw = yaml.safe_load(catalog_path.read_text())
    datasets = []
    for name, info in raw.items():
        # Read cache metadata for graph stats
        cache_meta_path = CACHE_ROOT / name / "cache_metadata.json"
        n_train_graphs = None
        cache_size_bytes = None
        if cache_meta_path.exists():
            try:
                meta = json.loads(cache_meta_path.read_text())
                n_train_graphs = meta.get("num_graphs")
            except Exception:
                pass
            # Calculate cache size
            cache_dir = CACHE_ROOT / name
            if cache_dir.is_dir():
                cache_size_bytes = sum(
                    f.stat().st_size for f in cache_dir.iterdir() if f.is_file()
                )

        n_test_scenarios = len(info.get("test_subdirs", []))

        datasets.append({
            "name": name,
            "domain": info.get("domain", ""),
            "protocol": info.get("protocol", ""),
            "source": info.get("source", ""),
            "description": info.get("description", ""),
            "n_train_graphs": n_train_graphs,
            "n_test_scenarios": n_test_scenarios,
            "cache_size_bytes": cache_size_bytes,
        })
    return datasets


def _extract_training_curves(runs: list[dict]) -> dict[str, list[dict]]:
    """Extract per-epoch training curves from Lightning CSV logs.

    Returns {run_id: [{"epoch": int, "metric_name": str, "value": float}, ...]}.
    """
    curves = {}
    for run in runs:
        run_dir = run["run_dir"]
        csv_logs = list(run_dir.glob("csv_logs/*/metrics.csv")) + \
                   list(run_dir.glob("lightning_logs/*/metrics.csv"))
        if not csv_logs:
            continue

        # Use the latest version (highest version number)
        csv_log = sorted(csv_logs)[-1]
        try:
            rows = []
            with open(csv_log) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epoch = row.get("epoch")
                    if epoch is None:
                        continue
                    for key, val in row.items():
                        if key in ("epoch", "step") or val == "":
                            continue
                        try:
                            rows.append({
                                "run_id": run["run_id"],
                                "epoch": int(float(epoch)),
                                "metric_name": key,
                                "value": float(val),
                            })
                        except (ValueError, TypeError):
                            continue
            if rows:
                curves[run["run_id"]] = rows
        except Exception as e:
            log.warning("Failed to parse CSV log for %s: %s", run["run_id"], e)
    return curves


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------

def _write_runs_parquet(runs: list[dict], dest: Path) -> None:
    """Write runs.parquet."""
    records = []
    for r in runs:
        records.append({
            "run_id": r["run_id"],
            "dataset": r["dataset"],
            "model_type": r["model_type"],
            "scale": r["scale"],
            "stage": r["stage"],
            "has_kd": r["has_kd"],
            "auxiliaries": r["auxiliaries"],
            "success": r["success"],
            "completed_at": r["completed_at"],
            "started_at": r["started_at"],
            "duration_seconds": r["duration_seconds"],
            "data_version": r["data_version"],
            "wandb_run_id": r["wandb_run_id"],
            "source": r["source"],
        })

    table = pa.Table.from_pylist(records)
    pq.write_table(table, dest / "runs.parquet")
    log.info("Wrote %d runs → runs.parquet", len(records))


def _write_metrics_parquet(metrics: list[dict], dest: Path) -> None:
    """Write metrics.parquet."""
    table = pa.Table.from_pylist(metrics)
    pq.write_table(table, dest / "metrics.parquet")
    log.info("Wrote %d metric records → metrics.parquet", len(metrics))


def _write_configs_parquet(configs: list[dict], dest: Path) -> None:
    """Write configs.parquet."""
    table = pa.Table.from_pylist(configs)
    pq.write_table(table, dest / "configs.parquet")
    log.info("Wrote %d config records → configs.parquet", len(configs))


def _write_artifacts_parquet(artifacts: list[dict], dest: Path) -> None:
    """Write artifacts.parquet."""
    table = pa.Table.from_pylist(artifacts)
    pq.write_table(table, dest / "artifacts.parquet")
    log.info("Wrote %d artifact records → artifacts.parquet", len(artifacts))


def _write_datasets_parquet(datasets: list[dict], dest: Path) -> None:
    """Write datasets.parquet."""
    table = pa.Table.from_pylist(datasets)
    pq.write_table(table, dest / "datasets.parquet")
    log.info("Wrote %d dataset records → datasets.parquet", len(datasets))


def _write_training_curves(curves: dict[str, list[dict]], dest: Path) -> None:
    """Write per-run training curve Parquet files."""
    curves_dir = dest / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for run_id, rows in curves.items():
        table = pa.Table.from_pylist(rows)
        fname = run_id.replace("/", "_") + ".parquet"
        pq.write_table(table, curves_dir / fname)
        count += 1
    log.info("Wrote %d training curve files → training_curves/", count)


def _build_analytics_duckdb(dest: Path) -> None:
    """Create analytics.duckdb with views over Parquet files."""
    import duckdb

    db_path = dest / "analytics.duckdb"
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        datalake = str(dest)

        con.execute(f"""
            CREATE VIEW runs AS
            SELECT * FROM read_parquet('{datalake}/runs.parquet')
        """)
        con.execute(f"""
            CREATE VIEW metrics AS
            SELECT * FROM read_parquet('{datalake}/metrics.parquet')
        """)
        con.execute(f"""
            CREATE VIEW configs AS
            SELECT * FROM read_parquet('{datalake}/configs.parquet')
        """)
        con.execute(f"""
            CREATE VIEW datasets AS
            SELECT * FROM read_parquet('{datalake}/datasets.parquet')
        """)
        con.execute(f"""
            CREATE VIEW artifacts AS
            SELECT * FROM read_parquet('{datalake}/artifacts.parquet')
        """)

        # Convenience views
        con.execute("""
            CREATE VIEW v_leaderboard AS
            SELECT r.dataset, r.model_type, r.scale, r.has_kd,
                   m.model, MAX(m.f1) AS best_f1, MAX(m.accuracy) AS best_accuracy,
                   MAX(m.auc) AS best_auc, MAX(m.mcc) AS best_mcc
            FROM runs r
            JOIN metrics m USING (run_id)
            WHERE r.stage = 'evaluation' AND r.success
            GROUP BY r.dataset, r.model_type, r.scale, r.has_kd, m.model
        """)

        con.execute("""
            CREATE VIEW v_kd_impact AS
            SELECT
                kd.dataset, kd.model AS model,
                kd.best_f1 AS kd_f1, nokd.best_f1 AS nokd_f1,
                kd.best_f1 - nokd.best_f1 AS f1_delta,
                teacher.best_f1 AS teacher_f1
            FROM v_leaderboard kd
            JOIN v_leaderboard nokd
                ON kd.dataset = nokd.dataset
                AND kd.model = nokd.model
                AND kd.scale = 'small' AND kd.has_kd = true
                AND nokd.scale = 'small' AND nokd.has_kd = false
            LEFT JOIN v_leaderboard teacher
                ON teacher.dataset = kd.dataset
                AND teacher.model = kd.model
                AND teacher.scale = 'large'
        """)

        # Summary
        counts = {}
        for view in ["runs", "metrics", "datasets", "configs", "artifacts"]:
            counts[view] = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        log.info("Analytics DuckDB built: %s", counts)
        print(f"Built {db_path}")
        for view, count in counts.items():
            print(f"  {view}: {count} rows")
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def migrate(dry_run: bool = False) -> None:
    """Run the full migration."""
    log.info("Scanning experiment runs...")
    runs = _scan_runs()
    metrics = _extract_metrics(runs)
    configs = _extract_configs(runs)
    artifacts = _extract_artifacts(runs)
    datasets = _load_datasets_catalog()
    curves = _extract_training_curves(runs)

    print(f"Found {len(runs)} runs across {len(set(r['dataset'] for r in runs))} datasets")
    print(f"  Metrics records: {len(metrics)}")
    print(f"  Config records: {len(configs)}")
    print(f"  Artifact records: {len(artifacts)}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Training curves: {len(curves)} runs with CSV logs")

    if dry_run:
        print(f"\nWould write to: {DATALAKE_ROOT}/")
        for r in runs:
            print(f"  {r['run_id']:50s} stage={r['stage']:15s} kd={r['has_kd']}")
        return

    # Create output directory
    DATALAKE_ROOT.mkdir(parents=True, exist_ok=True)

    # Write Parquet files
    _write_runs_parquet(runs, DATALAKE_ROOT)
    _write_metrics_parquet(metrics, DATALAKE_ROOT)
    _write_configs_parquet(configs, DATALAKE_ROOT)
    _write_artifacts_parquet(artifacts, DATALAKE_ROOT)
    _write_datasets_parquet(datasets, DATALAKE_ROOT)
    _write_training_curves(curves, DATALAKE_ROOT)

    # Build DuckDB views
    _build_analytics_duckdb(DATALAKE_ROOT)

    print(f"\nMigration complete → {DATALAKE_ROOT}/")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Migrate experiment data to Parquet datalake")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be built")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
