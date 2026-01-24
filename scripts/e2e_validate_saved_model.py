"""End-to-end test: save a model state-dict via HydraZenTrainer and validate it with the artifact validator.

Usage: conda run -n gnn-experiments python scripts/e2e_validate_saved_model.py
"""
import sys
from pathlib import Path
import logging

# Ensure project root is importable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config.hydra_zen_configs import config_store
from train_with_hydra_zen import HydraZenTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    # Create a small config (GAT normal) and point experiment_root to experimentruns_test/tmp_e2e
    store = config_store
    cfg = store.create_config(model_type='gat', dataset_name='hcrl_sa', training_mode='normal')
    cfg.experiment_root = 'experimentruns_test'
    cfg.dataset.name = 'hcrl_sa'
    cfg.training.max_epochs = 1
    cfg.training.seed = 123

    # Create trainer and paths
    trainer = HydraZenTrainer(cfg)
    paths = trainer.get_hierarchical_paths()

    # Make a small model and save its state-dict
    _, _, num_ids =  [], [], 1
    model = trainer.setup_model(num_ids=1)

    ckpt_path = trainer._save_state_dict(model, paths['checkpoint_dir'], 'e2e_smoke.pth')
    print('Saved checkpoint at', ckpt_path)

    # Validate artifact using script
    from scripts.validate_artifact import validate_artifact
    sanitized = validate_artifact(ckpt_path, resave_sanitized=True)
    print('Validator returned:', sanitized)

    # Attempt torch.load on sanitized file using weights_only=True
    try:
        import torch
        load_path = sanitized
        if not load_path.exists():
            # Fall back to the original if sanitized not returned
            load_path = ckpt_path
        print('Attempting to load via torch.load(weights_only=True):', load_path)
        ck = torch.load(str(load_path), map_location='cpu', weights_only=True)
        print('Loaded type:', type(ck))
        if isinstance(ck, dict):
            print('Keys:', list(ck.keys())[:10])
    except Exception as e:
        print('torch.load(weights_only=True) failed:', e)


if __name__ == '__main__':
    run()
