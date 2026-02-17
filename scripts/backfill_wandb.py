"""One-time backfill of existing experiment runs into W&B.

Scans experimentruns/ for config.json and metrics.json files,
creates a W&B run for each with appropriate tags and config.

Usage:
    python scripts/backfill_wandb.py              # Backfill all runs
    python scripts/backfill_wandb.py --dry-run    # Preview without creating runs
"""

import json
import sys
from pathlib import Path

import wandb

PROJECT = "kd-gat"
EXPERIMENT_ROOT = Path("experimentruns")


def parse_run_dir(run_dir: Path) -> dict:
    """Parse dataset and run identity from directory structure.

    Expected: experimentruns/{dataset}/{model}_{scale}_{stage}[_{aux}]
    """
    dataset = run_dir.parent.name
    parts = run_dir.name.split("_")

    # eval_large_evaluation, eval_small_evaluation_kd
    # vgae_large_autoencoder, gat_small_curriculum_kd
    model_type = parts[0]
    scale = parts[1]
    stage = parts[2]
    aux = "_".join(parts[3:]) if len(parts) > 3 else "none"

    return {
        "dataset": dataset,
        "model_type": model_type,
        "scale": scale,
        "stage": stage,
        "aux": aux,
        "run_name": f"{dataset}/{run_dir.name}",
    }


def backfill_run(run_dir: Path, dry_run: bool = False) -> bool:
    """Create a W&B run from an existing experiment directory."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return False

    identity = parse_run_dir(run_dir)
    config = json.loads(config_path.read_text())

    # Read metrics if available (evaluation runs)
    metrics_path = run_dir / "metrics.json"
    metrics = None
    if metrics_path.exists():
        try:
            raw = json.loads(metrics_path.read_text())
            # Flatten top-level metrics for W&B summary
            metrics = {}
            for model_key, model_metrics in raw.items():
                if isinstance(model_metrics, dict) and "core" in model_metrics:
                    for k, v in model_metrics["core"].items():
                        if isinstance(v, (int, float)):
                            metrics[f"{model_key}/{k}"] = v
        except (json.JSONDecodeError, KeyError):
            pass

    tags = [
        "backfill",
        identity["dataset"],
        identity["model_type"],
        identity["scale"],
        identity["stage"],
    ]
    if identity["aux"] != "none":
        tags.append(identity["aux"])

    if dry_run:
        print(f"  [DRY RUN] {identity['run_name']} | tags={tags}")
        if metrics:
            print(f"            metrics: {list(metrics.keys())[:5]}...")
        return True

    run = wandb.init(
        project=PROJECT,
        name=identity["run_name"],
        config={**config, **identity},
        tags=tags,
        notes=f"Backfilled from {run_dir}",
    )

    if metrics:
        wandb.log(metrics)

    wandb.finish()
    return True


def main():
    dry_run = "--dry-run" in sys.argv

    if not EXPERIMENT_ROOT.exists():
        print(f"ERROR: {EXPERIMENT_ROOT} not found")
        sys.exit(1)

    # Collect all run directories with config.json
    run_dirs = sorted(EXPERIMENT_ROOT.rglob("config.json"))
    print(f"Found {len(run_dirs)} experiment runs")

    if dry_run:
        print("DRY RUN — no W&B runs will be created\n")

    success = 0
    for config_path in run_dirs:
        run_dir = config_path.parent
        try:
            if backfill_run(run_dir, dry_run=dry_run):
                success += 1
        except Exception as e:
            print(f"  FAILED: {run_dir} — {e}")

    print(f"\nBackfilled {success}/{len(run_dirs)} runs")


if __name__ == "__main__":
    main()
