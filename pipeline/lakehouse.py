"""Sync structured experiment results to S3 lakehouse.

Writes run metadata + metrics as JSON to S3, one file per run.
Non-blocking, fire-and-forget — failures are logged but never crash
the training pipeline.

Configuration:
    KD_GAT_S3_BUCKET — S3 bucket name (default: "kd-gat")
    AWS credentials via ~/.aws/credentials (standard boto3 chain)

Downstream consumption:
    duckdb -c "SELECT * FROM read_json('s3://kd-gat/lakehouse/runs/*.json')"
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_S3_BUCKET = os.environ.get("KD_GAT_S3_BUCKET", "kd-gat")
_LAKEHOUSE_PREFIX = "lakehouse/runs"


def sync_to_lakehouse(
    run_id: str,
    dataset: str,
    model_type: str,
    scale: str,
    stage: str,
    has_kd: bool,
    metrics: dict | None = None,
    success: bool = True,
) -> bool:
    """Send run results to S3 lakehouse. Returns True on success.

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

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        log.debug("boto3 not installed — lakehouse sync skipped")
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
        log.info("Lakehouse sync OK: s3://%s/%s", _S3_BUCKET, key)
        return True
    except (BotoCoreError, ClientError) as e:
        log.warning("Lakehouse sync failed for %s: %s", run_id, e)
    except Exception as e:
        log.warning("Lakehouse sync error for %s: %s", run_id, e)

    return False
