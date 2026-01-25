"""
Quick smoke training script for hcrl_sa dataset.
Creates a tiny synthetic dataset (3 small CSVs), builds a config using CANGraphConfigStore,
sets training to 1 epoch and small batch size, and runs training via HydraZenTrainer.

Designed to run in resource-constrained environments (login node) and exit quickly.
"""
import os
import random
import sys
import logging
from pathlib import Path
# Ensure project root is on sys.path so imports like `src.*` resolve
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.hydra_zen_configs import CANGraphConfigStore
from train_with_hydra_zen import HydraZenTrainer

logger = logging.getLogger(__name__)

TMP_ROOT = Path('/tmp/hcrl_sa_smoke')
DATASET_DIR = TMP_ROOT / 'hcrl_sa'

# Create small dataset directory with a few CSVs
def create_small_dataset(dataset_dir: Path):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Typical layout includes train_01_attack_free and train_02_with_attacks
    for folder in ['train_01_attack_free', 'train_02_with_attacks']:
        folder_path = dataset_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        for i in range(2):  # two small CSVs each
            file_path = folder_path / f'smoke_{i+1}.csv'
            with open(file_path, 'w') as f:
                # write header (will be overwritten by parsing but okay)
                f.write('Timestamp,arbitration_id,data_field,attack\n')
                # create 120 rows so sliding window (100) yields one graph
                for t in range(120):
                    # use hex digits *without* 0x prefix so safe_hex_to_int treats it as hex
                    arbitration_id = f"{0x100 + (t % 5):x}"
                    # make varying payload lengths
                    payload = ''.join(random.choice('0123456789abcdef') for _ in range(8))
                    attack = '1' if (t % 50 == 0) else '0'
                    f.write(f"{t},{arbitration_id},{payload},{attack}\n")


def run_smoke():
    create_small_dataset(DATASET_DIR)

    store = CANGraphConfigStore()
    # Build config manually to avoid relying on store helper methods that may be missing
    model_cfg = store.get_model_config('gat')
    dataset_cfg = store.get_dataset_config('hcrl_sa')
    from src.config.hydra_zen_configs import NormalTrainingConfig, TrainerConfig, CANGraphConfig

    training_cfg = NormalTrainingConfig()
    trainer_cfg = TrainerConfig()

    cfg = CANGraphConfig(model=model_cfg, dataset=dataset_cfg, training=training_cfg, trainer=trainer_cfg)

    # Apply smoke overrides
    cfg.dataset.data_path = str(DATASET_DIR)
    cfg.experiment_root = str(TMP_ROOT / 'experiment_runs')
    cfg.trainer.max_epochs = 1
    cfg.training.max_epochs = 1
    cfg.training.batch_size = 4
    cfg.training.optimize_batch_size = False
    cfg.training.run_test = False

    # Avoid Lightning validation sanity checks which call model.forward
    cfg.trainer.num_sanity_val_steps = 0
    # Keep checkpointing enabled so callbacks are consistent
    cfg.trainer.enable_checkpointing = True

    trainer_manager = HydraZenTrainer(cfg)

    # Manually run minimal training without validation to avoid missing forward in validation hooks
    from src.training.datamodules import load_dataset, create_dataloaders
    train_dataset, val_dataset, num_ids = load_dataset(cfg.dataset.name, cfg, force_rebuild_cache=True)
    
    # Ensure num_ids is at least the number we found
    logger.info(f"Dataset reports {num_ids} unique CAN IDs")
    
    model = trainer_manager.setup_model(num_ids)
    train_loader, _ = create_dataloaders(train_dataset, val_dataset, model.batch_size)

    lit_trainer = trainer_manager.setup_trainer()

    # Disable EarlyStopping callback to avoid requiring validation metrics in this smoke run
    try:
        from lightning.pytorch.callbacks import EarlyStopping
        lit_trainer.callbacks = [c for c in lit_trainer.callbacks if not isinstance(c, EarlyStopping)]
    except Exception:
        # If callbacks are not available or structure differs, ignore
        pass

    # Fit only on training loader to avoid validation forward calls
    lit_trainer.fit(model, train_loader)

    print('Smoke training finished. Model:', type(model), 'Trainer:', type(lit_trainer))


if __name__ == '__main__':
    run_smoke()
