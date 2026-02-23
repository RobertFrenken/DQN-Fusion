"""Sync structured experiment results to local lakehouse + S3.

Writes run metadata + metrics as JSON, one file per run.
Local write is primary (always succeeds on NFS); S3 is secondary
(fire-and-forget). Failures are logged but never crash the training pipeline.

Configuration:
    KD_GAT_DATA_ROOT — local lakehouse root (see config.paths.lakehouse_dir)
    KD_GAT_S3_BUCKET — S3 bucket name (default: "kd-gat")
    AWS credentials via ~/.aws/credentials (standard boto3 chain)

Downstream consumption:
    # Local (no credentials needed):
    duckdb -c "SELECT * FROM read_json('$KD_GAT_DATA_ROOT/lakehouse/runs/*.json')"

    # S3:
    duckdb -c "SELECT * FROM read_json('s3://kd-gat/lakehouse/runs/*.json')"
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from config.paths import lakehouse_dir

log = logging.getLogger(__name__)

_S3_BUCKET = os.environ.get("KD_GAT_S3_BUCKET", "kd-gat")
_LAKEHOUSE_PREFIX = "lakehouse/runs"


def _write_local(payload: dict, run_id: str) -> bool:
    """Write payload JSON to local lakehouse directory (fire-and-forget)."""
    try:
        dest = lakehouse_dir()
        dest.mkdir(parents=True, exist_ok=True)
        # Flatten run_id (may contain '/') to a safe filename
        filename = run_id.replace("/", "_") + ".json"
        (dest / filename).write_text(json.dumps(payload, default=str), encoding="utf-8")
        log.info("Lakehouse local write OK: %s", dest / filename)
        return True
    except Exception as e:
        log.warning("Lakehouse local write failed for %s: %s", run_id, e)
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
    """Send run results to local lakehouse + S3. Returns True if either succeeded.

    This function is intentionally fire-and-forget: it catches all
    exceptions and logs warnings instead of raising.  Training must
    never fail because of a lakehouse sync issue.
    """
    # Flatten core metrics from the nested evaluation structure
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

    # Local write first (always succeeds on NFS)
    local_ok = _write_local(payload, run_id)

    # S3 write second (fire-and-forget)
    s3_ok = _write_s3(payload, run_id)

    return local_ok or s3_ok
