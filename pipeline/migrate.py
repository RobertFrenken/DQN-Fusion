"""Migrate existing experiments from 8-level to 2-level directory structure.

Usage:
    python -m pipeline.migrate --dry-run          # Preview changes
    python -m pipeline.migrate --execute          # Perform migration
    python -m pipeline.migrate --backfill-only    # Only populate MLflow (no file moves)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .paths import STAGES, run_id, stage_dir
from .tracking import setup_tracking, start_run, end_run

# Old path structure (8 levels)
# {root}/{modality}/{dataset}/{size}/{learning_type}/{model_arch}/{distill}/{mode}
OLD_PATTERN = "experimentruns/automotive/*/*/*/*/*/*/*"


def parse_old_path(path: Path) -> Optional[tuple[str, str, str, bool]]:
    """Parse old 8-level path to extract (dataset, model_size, stage, use_kd).

    Example:
        experimentruns/automotive/hcrl_sa/teacher/unsupervised/vgae/no_distillation/autoencoder
        -> ("hcrl_sa", "teacher", "autoencoder", False)
    """
    parts = path.parts
    if len(parts) < 8:
        return None

    # Extract components
    modality = parts[1]
    dataset = parts[2]
    model_size = parts[3]
    learning_type = parts[4]
    model_arch = parts[5]
    distillation = parts[6]
    mode = parts[7]

    # Map mode to stage name
    stage = None
    for stage_name, (lt, ma, m) in STAGES.items():
        if learning_type == lt and model_arch == ma and mode == m:
            stage = stage_name
            break

    if stage is None:
        print(f"  Warning: Could not map {path} to a known stage")
        return None

    use_kd = distillation == "distilled"

    return dataset, model_size, stage, use_kd


def find_old_experiments(root: Path = Path("experimentruns")) -> list[Path]:
    """Find all old-style experiment directories."""
    # Look for directories with config.json or best_model.pt at the 8th level
    old_dirs = []
    if not root.exists():
        return old_dirs

    # Check if we have the old structure
    automotive_dir = root / "automotive"
    if automotive_dir.exists():
        # Recursively find all directories with config.json or best_model.pt
        for marker_file in automotive_dir.rglob("best_model.pt"):
            exp_dir = marker_file.parent
            # Only include if it's an 8-level path (has 8+ parts)
            if len(exp_dir.parts) >= 8 and exp_dir not in old_dirs:
                old_dirs.append(exp_dir)

        # Also check for config.json
        for config_file in automotive_dir.rglob("config.json"):
            exp_dir = config_file.parent
            if exp_dir not in old_dirs:
                old_dirs.append(exp_dir)

    return sorted(old_dirs)


def compute_new_path(old_path: Path) -> Optional[Path]:
    """Compute new 2-level path from old 8-level path."""
    parsed = parse_old_path(old_path)
    if parsed is None:
        return None

    dataset, model_size, stage, use_kd = parsed

    # Build new path
    kd_suffix = "_kd" if use_kd else ""
    new_name = f"{model_size}_{stage}{kd_suffix}"
    new_path = Path("experimentruns") / dataset / new_name

    return new_path


def migrate_experiment(old_path: Path, new_path: Path, dry_run: bool = True) -> bool:
    """Move experiment from old path to new path.

    Returns True if successful, False otherwise.
    """
    if not old_path.exists():
        print(f"  ✗ Source does not exist: {old_path}")
        return False

    if new_path.exists():
        print(f"  ! Target already exists: {new_path}")
        print(f"    Skipping to avoid overwrite")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would move:")
        print(f"    {old_path}")
        print(f"    -> {new_path}")
        return True
    else:
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            print(f"  ✓ Moved: {old_path.name} -> {new_path}")
            return True
        except Exception as e:
            print(f"  ✗ Failed to move {old_path}: {e}")
            return False


def backfill_mlflow(experiment_path: Path, skip_if_no_config: bool = True) -> bool:
    """Backfill MLflow tracking data from existing experiment directory.

    Reads config.json and metrics.json (if exists) and creates MLflow run entry.
    If config.json doesn't exist, skips MLflow backfill (can't recreate config).
    """
    config_file = experiment_path / "config.json"
    metrics_file = experiment_path / "metrics.json"

    if not config_file.exists():
        if skip_if_no_config:
            print(f"  ⊘ No config.json - skipping MLflow backfill")
            print(f"    (Files migrated but MLflow tracking unavailable for old runs)")
            return True
        else:
            print(f"  ✗ No config.json in {experiment_path}")
            return False

    try:
        # Load config
        cfg = PipelineConfig.load(config_file)

        # Infer stage from path
        # New path format: experimentruns/{dataset}/{model_size}_{stage}[_kd]
        dir_name = experiment_path.name
        stage = None
        for stage_name in STAGES.keys():
            if stage_name in dir_name:
                stage = stage_name
                break

        if stage is None:
            print(f"  ✗ Could not infer stage from {dir_name}")
            return False

        # Generate run name
        run_name = run_id(cfg, stage)

        # Load metrics if available
        metrics = None
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)

        # Create MLflow run
        setup_tracking()
        start_run(cfg, stage, run_name)

        # Log completion if we have metrics
        if metrics:
            end_run(metrics, success=True)
        else:
            end_run(success=True)

        print(f"  ✓ Backfilled MLflow run: {run_name}")
        return True

    except Exception as e:
        print(f"  ✗ Failed to backfill {experiment_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate experiments from 8-level to 2-level structure"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the migration",
    )
    parser.add_argument(
        "--backfill-only",
        action="store_true",
        help="Only populate MLflow (no file moves)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experimentruns"),
        help="Experiment root directory",
    )

    args = parser.parse_args()

    if not (args.dry_run or args.execute or args.backfill_only):
        parser.error("Must specify --dry-run, --execute, or --backfill-only")

    print("=" * 70)
    print("Experiment Migration Tool")
    print("=" * 70)

    # Find old experiments
    old_experiments = find_old_experiments(args.root)
    print(f"\nFound {len(old_experiments)} old-style experiment directories")

    if len(old_experiments) == 0:
        print("\nNo old experiments found. Migration not needed.")
        return

    # Process each experiment
    migrated = 0
    skipped = 0
    failed = 0

    for old_path in old_experiments:
        print(f"\nProcessing: {old_path}")

        new_path = compute_new_path(old_path)
        if new_path is None:
            print("  ✗ Could not compute new path")
            failed += 1
            continue

        if args.backfill_only:
            # Use current path (assuming already migrated or new)
            if backfill_mlflow(old_path):
                migrated += 1
            else:
                failed += 1
        else:
            # Migrate files
            if migrate_experiment(old_path, new_path, dry_run=args.dry_run):
                migrated += 1

                # Backfill MLflow if executing
                if args.execute:
                    backfill_mlflow(new_path)
            else:
                skipped += 1

    # Summary
    print("\n" + "=" * 70)
    print("Migration Summary")
    print("=" * 70)
    print(f"Total experiments: {len(old_experiments)}")
    print(f"Migrated: {migrated}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made.")
        print("Run with --execute to perform the migration.")
    elif args.execute:
        print("\n✓ Migration complete!")
        print("\nNext steps:")
        print("1. Verify new experiment paths in experimentruns/")
        print("2. Check MLflow tracking data:")
        print("   python -m pipeline.query --all")
        print("3. Update Snakemake workflows if needed")
        print("4. Clean up old 'automotive' directory if satisfied:")
        print("   rm -rf experimentruns/automotive")


if __name__ == "__main__":
    main()
