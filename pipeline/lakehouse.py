"""Sync structured experiment results to Cloudflare R2 lakehouse.

Sends run metadata + metrics as JSON to a Cloudflare Pipeline endpoint,
which auto-batches into Parquet on R2.  Non-blocking, fire-and-forget
with retry — failures are logged but never crash the training pipeline.

Configuration via environment variables:
    KD_GAT_LAKEHOUSE_URL   — Cloudflare Pipeline HTTP endpoint
    KD_GAT_LAKEHOUSE_TOKEN — Bearer token for authentication

If neither is set, sync is silently skipped (graceful no-op).

Downstream consumption:
    duckdb -c "SELECT * FROM read_parquet('s3://kd-gat-data/runs/*.parquet')"
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_LAKEHOUSE_URL = os.environ.get("KD_GAT_LAKEHOUSE_URL", "")
_LAKEHOUSE_TOKEN = os.environ.get("KD_GAT_LAKEHOUSE_TOKEN", "")

# Max retries for transient network failures
_MAX_RETRIES = 2
_TIMEOUT_SECONDS = 10.0


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
    """Send run results to R2 lakehouse. Returns True on success.

    This function is intentionally fire-and-forget: it catches all
    exceptions and logs warnings instead of raising.  Training must
    never fail because of a lakehouse sync issue.
    """
    if not _LAKEHOUSE_URL:
        log.debug("KD_GAT_LAKEHOUSE_URL not set — lakehouse sync skipped")
        return False

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
        import httpx
    except ImportError:
        log.debug("httpx not installed — lakehouse sync skipped")
        return False

    headers = {"Content-Type": "application/json"}
    if _LAKEHOUSE_TOKEN:
        headers["Authorization"] = f"Bearer {_LAKEHOUSE_TOKEN}"

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = httpx.post(
                _LAKEHOUSE_URL,
                json=payload,
                headers=headers,
                timeout=_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            log.info("Lakehouse sync OK: %s (attempt %d)", run_id, attempt)
            return True
        except httpx.TimeoutException:
            log.warning("Lakehouse sync timeout (attempt %d/%d)", attempt, _MAX_RETRIES)
        except httpx.HTTPStatusError as e:
            log.warning("Lakehouse sync HTTP %d (attempt %d/%d): %s",
                        e.response.status_code, attempt, _MAX_RETRIES, e)
        except Exception as e:
            log.warning("Lakehouse sync error (attempt %d/%d): %s", attempt, _MAX_RETRIES, e)

    log.warning("Lakehouse sync failed after %d attempts for %s", _MAX_RETRIES, run_id)
    return False
