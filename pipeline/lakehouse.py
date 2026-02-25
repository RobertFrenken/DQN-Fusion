"""Sync structured experiment results to datalake (Parquet) + S3 (JSON backup).

Primary write: append to data/datalake/*.parquet via DuckDB INSERT INTO.
Secondary write: JSON to S3 (fire-and-forget backup, will be deprecated).

Failures are logged but never crash the training pipeline.

Configuration:
    KD_GAT_DATA_ROOT — local data root (see config.paths)
    KD_GAT_S3_BUCKET — S3 bucket name (default: "kd-gat")
    AWS credentials via ~/.aws/credentials (standard boto3 chain)

Downstream consumption:
    duckdb -c "SELECT * FROM 'data/datalake/runs.parquet'"
    duckdb data/datalake/analytics.duckdb  # pre-built views
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from config.paths import lakehouse_dir

log = logging.getLogger(__name__)

_S3_BUCKET = os.environ.get("KD_GAT_S3_BUCKET", "kd-gat")
_LAKEHOUSE_PREFIX = "lakehouse/runs"
_DATALAKE_ROOT = Path("data/datalake")

# Core metrics to extract from nested metrics.json
_CORE_METRIC_COLS = [
    "accuracy", "precision", "recall", "f1", "specificity",
    "balanced_accuracy", "mcc", "fpr", "fnr", "auc", "n_samples",
]


# ---------------------------------------------------------------------------
# Parquet datalake writes (primary)
# ---------------------------------------------------------------------------

def _append_to_datalake(
    run_id: str,
    dataset: str,
    model_type: str,
    scale: str,
    stage: str,
    has_kd: bool,
    metrics: dict | None,
    success: bool,
    failure_reason: str | None,
) -> bool:
    """Append run + metrics to datalake Parquet files via DuckDB."""
    try:
        import duckdb
    except ImportError:
        log.debug("duckdb not installed — datalake append skipped")
        return False

    if not (_DATALAKE_ROOT / "runs.parquet").exists():
        log.debug("Datalake not initialized — run `python -m pipeline.migrate_datalake` first")
        return False

    try:
        now = datetime.now(timezone.utc).isoformat()
        datalake = str(_DATALAKE_ROOT)
        con = duckdb.connect()

        # Upsert run record
        con.execute(f"""
            INSERT INTO '{datalake}/runs.parquet'
            BY NAME (SELECT
                ? AS run_id, ? AS dataset, ? AS model_type, ? AS scale,
                ? AS stage, ? AS has_kd, '' AS auxiliaries, ? AS success,
                ? AS completed_at, ? AS started_at, NULL AS duration_seconds,
                NULL AS data_version, NULL AS wandb_run_id, 'pipeline' AS source
            )
        """, [run_id, dataset, model_type, scale, stage, has_kd, success, now, now])

        # Append metrics if evaluation stage
        if metrics:
            for model_key, model_data in metrics.items():
                if model_key == "test":
                    continue
                if not isinstance(model_data, dict) or "core" not in model_data:
                    continue
                core = model_data["core"]
                values = [run_id, model_key]
                for col in _CORE_METRIC_COLS:
                    val = core.get(col)
                    values.append(float(val) if isinstance(val, (int, float)) else None)
                placeholders = ", ".join(["?"] * len(values))
                cols = "run_id, model, " + ", ".join(
                    f'"{c}"' if c == "precision" else c for c in _CORE_METRIC_COLS
                )
                con.execute(
                    f"INSERT INTO '{datalake}/metrics.parquet' ({cols}) VALUES ({placeholders})",
                    values,
                )

        con.close()
        log.info("Datalake append OK: %s", run_id)
        return True
    except Exception as e:
        log.warning("Datalake append failed for %s: %s", run_id, e)
        return False


def register_artifacts(run_id: str, run_dir: Path) -> bool:
    """Scan a run directory and append artifact records to datalake."""
    try:
        import duckdb
    except ImportError:
        return False

    if not (_DATALAKE_ROOT / "artifacts.parquet").exists():
        return False

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

    try:
        records = []
        for filename, artifact_type in ARTIFACT_TYPES.items():
            fpath = run_dir / filename
            if fpath.exists():
                records.append((run_id, artifact_type, str(fpath), fpath.stat().st_size))

        if not records:
            return True

        datalake = str(_DATALAKE_ROOT)
        con = duckdb.connect()
        con.executemany(
            f"INSERT INTO '{datalake}/artifacts.parquet' VALUES (?, ?, ?, ?)",
            records,
        )
        con.close()
        log.info("Registered %d artifacts for %s", len(records), run_id)
        return True
    except Exception as e:
        log.warning("Artifact registration failed for %s: %s", run_id, e)
        return False


# ---------------------------------------------------------------------------
# Legacy JSON writes (S3 backup — will be deprecated)
# ---------------------------------------------------------------------------

def _write_local_json(payload: dict, run_id: str) -> bool:
    """Write payload JSON to local lakehouse directory (fire-and-forget)."""
    try:
        dest = lakehouse_dir()
        dest.mkdir(parents=True, exist_ok=True)
        filename = run_id.replace("/", "_") + ".json"
        (dest / filename).write_text(json.dumps(payload, default=str), encoding="utf-8")
        log.info("Lakehouse local JSON write OK: %s", dest / filename)
        return True
    except Exception as e:
        log.warning("Lakehouse local JSON write failed for %s: %s", run_id, e)
        return False


def _write_s3(payload: dict, run_id: str) -> bool:
    """Write payload JSON to S3 lakehouse (fire-and-forget)."""
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        log.debug("boto3 not installed — S3 lakehouse sync skipped")
        return False

    try:
        s3 = boto3.client("s3")
        key = f"{_LAKEHOUSE_PREFIX}/{run_id}.json"
        s3.put_object(
            Bucket=_S3_BUCKET,
            Key=key,
            Body=json.dumps(payload, default=str),
            ContentType="application/json",
        )
        log.info("Lakehouse S3 sync OK: s3://%s/%s", _S3_BUCKET, key)
        return True
    except (BotoCoreError, ClientError) as e:
        log.warning("Lakehouse S3 sync failed for %s: %s", run_id, e)
    except Exception as e:
        log.warning("Lakehouse S3 sync error for %s: %s", run_id, e)

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sync_to_lakehouse(
    run_id: str,
    dataset: str,
    model_type: str,
    scale: str,
    stage: str,
    has_kd: bool,
    metrics: dict | None = None,
    success: bool = True,
    failure_reason: str | None = None,
) -> bool:
    """Send run results to datalake (Parquet) + S3 (JSON). Returns True if any succeeded.

    This function is intentionally fire-and-forget: it catches all
    exceptions and logs warnings instead of raising.  Training must
    never fail because of a lakehouse sync issue.
    """
    # Flatten core metrics for legacy JSON payload
    flat_metrics: dict[str, float] = {}
    if metrics:
        for model_key, model_data in metrics.items():
            if model_key == "test":
                continue
            if isinstance(model_data, dict) and "core" in model_data:
                for k, v in model_data["core"].items():
                    if isinstance(v, (int, float)):
                        flat_metrics[f"{model_key}_{k}"] = v
            elif isinstance(model_data, (int, float)):
                flat_metrics[model_key] = model_data

    payload = {
        "run_id": run_id,
        "dataset": dataset,
        "model_type": model_type,
        "scale": scale,
        "stage": stage,
        "has_kd": has_kd,
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **flat_metrics,
    }
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason

    # Primary: Parquet datalake
    parquet_ok = _append_to_datalake(
        run_id, dataset, model_type, scale, stage, has_kd,
        metrics, success, failure_reason,
    )

    # Secondary: local JSON (legacy)
    local_ok = _write_local_json(payload, run_id)

    # Tertiary: S3 JSON (fire-and-forget backup)
    s3_ok = _write_s3(payload, run_id)

    return parquet_ok or local_ok or s3_ok
