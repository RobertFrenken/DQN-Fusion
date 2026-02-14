"""Migrate legacy experiment directories to the new naming convention.

Old format: ``{model_size}_{stage}[_kd]``  (e.g. ``teacher_autoencoder``)
New format: ``{model_type}_{scale}_{stage}[_{aux}]``  (e.g. ``vgae_large_autoencoder``)

Usage:
    python -m pipeline.migrate_paths --dry-run              # Preview renames
    python -m pipeline.migrate_paths --execute              # Execute renames
    python -m pipeline.migrate_paths --dry-run --dataset hcrl_sa  # Single dataset
    python -m pipeline.migrate_paths --execute --no-db-update     # Skip DB updates
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from config.paths import EXPERIMENT_ROOT, STAGES

log = logging.getLogger(__name__)

# teacher→large, student→small
_SCALE_MAP = {"teacher": "large", "student": "small"}

# stage keyword → model_type (from STAGES dict)
_STAGE_MODEL_TYPE = {stage: info[1] for stage, info in STAGES.items()}
# evaluation stage maps to "eval" in STAGES, but we keep it as-is in the dir name
_STAGE_MODEL_TYPE["evaluation"] = "eval"


def _detect_legacy_dir(name: str) -> dict | None:
    """Parse a legacy directory name into (scale, stage, has_kd).

    Returns None if the name is not in legacy format.
    """
    parts = name.split("_")
    if len(parts) < 2:
        return None

    # Legacy dirs start with "teacher" or "student"
    if parts[0] not in _SCALE_MAP:
        return None

    scale = _SCALE_MAP[parts[0]]
    has_kd = name.endswith("_kd")

    # Find the stage keyword
    stage = None
    for s in ["autoencoder", "curriculum", "normal", "fusion", "evaluation"]:
        if s in parts:
            stage = s
            break

    if stage is None:
        return None

    model_type = _STAGE_MODEL_TYPE.get(stage, "unknown")
    aux_suffix = "_kd" if has_kd else ""

    return {
        "scale": scale,
        "stage": stage,
        "model_type": model_type,
        "has_kd": has_kd,
        "new_name": f"{model_type}_{scale}_{stage}{aux_suffix}",
    }


def plan_migrations(
    experiment_root: str = EXPERIMENT_ROOT,
    dataset: str | None = None,
) -> list[dict]:
    """Scan for legacy directories and plan renames.

    Returns list of dicts with keys: dataset, old_name, new_name, old_path, new_path.
    """
    root = Path(experiment_root)
    if not root.exists():
        return []

    plans = []
    ds_dirs = [root / dataset] if dataset else sorted(root.iterdir())

    for ds_dir in ds_dirs:
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            info = _detect_legacy_dir(run_dir.name)
            if info is None:
                continue

            new_path = ds_dir / info["new_name"]
            if new_path.exists():
                log.warning("Target already exists, skipping: %s → %s", run_dir, new_path)
                continue

            plans.append({
                "dataset": ds_dir.name,
                "old_name": run_dir.name,
                "new_name": info["new_name"],
                "old_path": run_dir,
                "new_path": new_path,
                "old_run_id": f"{ds_dir.name}/{run_dir.name}",
                "new_run_id": f"{ds_dir.name}/{info['new_name']}",
            })

    return plans


def execute_migrations(
    plans: list[dict],
    update_db: bool = True,
) -> int:
    """Execute planned renames.

    Returns count of successful renames.
    """
    count = 0
    for plan in plans:
        old_path = plan["old_path"]
        new_path = plan["new_path"]

        if new_path.exists():
            log.warning("Skipping (target exists): %s", new_path)
            continue

        try:
            shutil.move(str(old_path), str(new_path))
            # NFS needs a brief pause to propagate metadata
            time.sleep(0.1)
            log.info("Renamed: %s → %s", old_path, new_path)
            count += 1
        except Exception as e:
            log.error("Failed to rename %s: %s", old_path, e)
            continue

    if update_db and count > 0:
        _update_db_run_ids(plans)

    return count


def _update_db_run_ids(plans: list[dict]) -> None:
    """Update run_id values in the project DB after renames."""
    try:
        from .db import get_connection
        conn = get_connection()
        for plan in plans:
            # Only update if the new_path actually exists (rename succeeded)
            if not plan["new_path"].exists():
                continue
            conn.execute(
                "UPDATE runs SET run_id = ? WHERE run_id = ?",
                (plan["new_run_id"], plan["old_run_id"]),
            )
            conn.execute(
                "UPDATE metrics SET run_id = ? WHERE run_id = ?",
                (plan["new_run_id"], plan["old_run_id"]),
            )
        conn.commit()
        conn.close()
        log.info("Updated project DB run IDs")
    except Exception as e:
        log.warning("Failed to update project DB: %s", e)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline.migrate_paths",
        description="Migrate legacy experiment directories to new naming convention",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Preview renames without executing")
    mode.add_argument("--execute", action="store_true", help="Execute renames")

    parser.add_argument("--dataset", help="Migrate only a specific dataset")
    parser.add_argument("--no-db-update", action="store_true", help="Skip project DB updates")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )

    plans = plan_migrations(dataset=args.dataset)

    if not plans:
        print("No legacy directories found to migrate.")
        return

    print(f"\n{'DRY RUN' if args.dry_run else 'EXECUTING'}: {len(plans)} renames planned\n")
    for p in plans:
        print(f"  {p['old_run_id']}  →  {p['new_run_id']}")

    if args.execute:
        print()
        count = execute_migrations(plans, update_db=not args.no_db_update)
        print(f"\nCompleted: {count}/{len(plans)} renames")
    else:
        print(f"\nRun with --execute to apply these renames.")


if __name__ == "__main__":
    main()
