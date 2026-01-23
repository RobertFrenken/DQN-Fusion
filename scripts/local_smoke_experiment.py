"""
Local smoke experiment helper

Creates a minimal CANGraphConfig, prints the canonical experiment directories,
optionally runs a single-epoch training (disabled by default) to verify end-to-end
paths and MLflow logging.

Usage:
    python scripts/local_smoke_experiment.py --model vgae_student --dataset hcrl_ch --training autoencoder --epochs 1
    python scripts/local_smoke_experiment.py --model gat --dataset hcrl_sa --training normal --run

Notes:
- Defaults are CPU-only and short to be safe on development machines.
- If your dataset is not present under the default dataset paths, set --data-path to point to local data.
"""

import argparse
from pathlib import Path
import logging
import sys
import random
import csv

# Bring project root into path so imports resolve as in normal execution
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Defer heavy imports to runtime so this module (and helpers) can be imported in environments
# without hydra_zen during quick unit tests. The imports are performed inside main().
# from src.config.hydra_zen_configs import CANGraphConfigStore
# from train_with_hydra_zen import HydraZenTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local_smoke_experiment")

# Trainer factory hook - allows tests to inject a fake trainer object without importing heavy dependencies
def _default_trainer_factory(cfg):
    from train_with_hydra_zen import HydraZenTrainer
    return HydraZenTrainer(cfg)

TRAINER_FACTORY = _default_trainer_factory


from typing import Optional

def create_synthetic_dataset(root: Path, num_rows: int = 200, seed: Optional[int] = None):
    """Create a tiny synthetic dataset structure with CSV files.

    The format matches expected CSV layout: Timestamp,arbitration_id,data_field,attack
    """
    if seed is not None:
        random.seed(seed)

    train_dir = root / "train_01_attack_free"
    train_dir.mkdir(parents=True, exist_ok=True)
    csv_path = train_dir / "small.csv"

    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "arbitration_id", "data_field", "attack"])
        for i in range(num_rows):
            can_id = format(0x100 + (i % 256), 'X')
            payload = ''.join(format(random.randint(0, 255), '02X') for _ in range(8))
            writer.writerow([i, can_id, payload, 0])

    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vgae_student", help="Model type (vgae, vgae_student, gat, gat_student, dqn, dqn_student)")
    parser.add_argument("--dataset", default="hcrl_ch", help="Dataset (hcrl_ch, hcrl_sa, set_01, ...)")
    parser.add_argument("--training", default="autoencoder", help="Training mode (autoencoder, normal, knowledge_distillation)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for a smoke run")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--run", action="store_true", help="Actually run training (disabled by default)")
    parser.add_argument("--experiment-root", default=str(project_root / "experimentruns_test"), help="Where to place the experiment outputs for this smoke run")
    parser.add_argument("--data-path", default=None, help="Override dataset data_path if your local dataset lives elsewhere")
    parser.add_argument("--use-synthetic-data", action="store_true", help="Generate a tiny synthetic dataset for quick smoke runs")
    parser.add_argument("--synthetic-size", type=int, default=200, help="Number of rows in synthetic CSV(s)")
    parser.add_argument("--synthetic-seed", type=int, default=None, help="Optional RNG seed for synthetic dataset generation")
    parser.add_argument("--write-summary", action="store_true", help="Write a short summary.json into the experiment run directory when --run is used")
    args = parser.parse_args()

    # Import heavy components here to avoid module import-time dependency in tests
    from src.config.hydra_zen_configs import CANGraphConfigStore
    from train_with_hydra_zen import HydraZenTrainer as _RealHydraZenTrainer

    store = CANGraphConfigStore()
    cfg = store.create_config(model_type=args.model, dataset_name=args.dataset, training_mode=args.training, max_epochs=args.epochs, batch_size=args.batch_size)

    # Make experiment_root explicit and canonical for this smoke run
    cfg.experiment_root = str(Path(args.experiment_root).resolve())
    cfg.trainer.devices = 1 if args.device == "cpu" else 1
    cfg.training.batch_size = args.batch_size
    cfg.training.max_epochs = args.epochs

    if args.data_path:
        cfg.dataset.data_path = args.data_path

    logger.info("Created config:")
    logger.info(f"  model: {cfg.model.type}")
    logger.info(f"  dataset: {cfg.dataset.name} -> {cfg.dataset.data_path}")
    logger.info(f"  training mode: {cfg.training.mode}")
    logger.info(f"  experiment_root: {cfg.experiment_root}")

    # Use the factory to get trainer - testable hook
    trainer = TRAINER_FACTORY(cfg)

    # If requested, create a tiny synthetic dataset in the provided data_path
    if args.use_synthetic_data:
        if not args.data_path:
            synthetic_dir = Path(cfg.experiment_root) / 'synthetic_dataset'
        else:
            synthetic_dir = Path(args.data_path)
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        create_synthetic_dataset(synthetic_dir, num_rows=args.synthetic_size, seed=args.synthetic_seed)
        cfg.dataset.data_path = str(synthetic_dir)
        logger.info(f"✅ Synthetic dataset created at {synthetic_dir} (rows={args.synthetic_size}, seed={args.synthetic_seed})")

    # Create canonical directories and show them
    paths = trainer.get_hierarchical_paths()
    logger.info("Canonical experiment paths created:")
    for k, p in paths.items():
        logger.info(f"  {k}: {p}")

    if args.run:
        try:
            logger.info("Running a short training run (this may require your dataset to be present)...")
            trainer.train()
            logger.info("Smoke run finished successfully")

            # Optionally write a short summary file under the run directory for traceability
            if args.write_summary:
                try:
                    from src.utils.experiment_paths import ExperimentPathManager
                    pm = ExperimentPathManager(cfg)
                    run_dir = pm.get_run_dir_safe()
                    summary = {
                        'model': cfg.model.type,
                        'dataset': cfg.dataset.name,
                        'data_path': cfg.dataset.data_path,
                        'training_mode': cfg.training.mode,
                        'experiment_dir': str(pm.get_experiment_dir()),
                    }
                    import json
                    with open(run_dir / 'summary.json', 'w') as f:
                        json.dump(summary, f, indent=2)
                    logger.info(f"Wrote summary to {run_dir / 'summary.json'}")
                except Exception as e:
                    logger.warning(f"Could not write summary file: {e}")

        except Exception as e:
            logger.error("Smoke run failed")
            logger.exception(e)
            logger.error("Common causes: dataset not found at dataset.data_path, missing required artifacts for distillation/fusion. Set --data-path or pre-run required teacher models before distillation experiments.")
    else:
        logger.info("Run flag not set; no training executed. Use --run to attempt a one-epoch smoke training run.")


if __name__ == "__main__":
    main()
