import json
import os
import sys
import importlib.util
from pathlib import Path
import pytest

# Load config module by file path (avoid importing heavy package tree during collection)
_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'config', 'hydra_zen_configs.py'))
_spec = importlib.util.spec_from_file_location('hydra_cfg_mod', _cfg_path)
_cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg_mod)

# Load dependency manifest util by file path
_dm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'utils', 'dependency_manifest.py'))
_spec2 = importlib.util.spec_from_file_location('dependency_manifest', _dm_path)
_dm_mod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_dm_mod)

CANGraphConfig = _cfg_mod.CANGraphConfig
CANGraphConfigStore = _cfg_mod.CANGraphConfigStore
FusionTrainingConfig = _cfg_mod.FusionTrainingConfig
TrainerConfig = _cfg_mod.TrainerConfig

load_manifest = _dm_mod.load_manifest
validate_manifest_for_config = _dm_mod.validate_manifest_for_config
ManifestValidationError = _dm_mod.ManifestValidationError


def _write_dummy(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('dummy')


def test_manifest_validation_success(tmp_path):
    store = CANGraphConfigStore()
    cfg = CANGraphConfig(model=store.get_model_config('gat'), dataset=store.get_dataset_config('hcrl_sa'), training=FusionTrainingConfig(), trainer=TrainerConfig())
    cfg.experiment_root = str(tmp_path / 'experiment_runs')

    # canonical artifact locations (models saved in models/ subdir)
    ae = Path(cfg.experiment_root) / cfg.modality / cfg.dataset.name / 'unsupervised' / 'vgae' / 'teacher' / cfg.distillation / 'autoencoder' / 'models' / 'vgae_autoencoder.pth'
    clf = Path(cfg.experiment_root) / cfg.modality / cfg.dataset.name / 'supervised' / 'gat' / 'teacher' / cfg.distillation / 'normal' / 'models' / f'gat_{cfg.dataset.name}_normal.pth'

    _write_dummy(ae)
    _write_dummy(clf)

    manifest = {
        'autoencoder': {'path': str(ae), 'model_size': 'teacher', 'training_mode': 'autoencoder', 'distillation': cfg.distillation},
        'classifier': {'path': str(clf), 'model_size': 'teacher', 'training_mode': 'normal', 'distillation': cfg.distillation}
    }

    manifest_path = tmp_path / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    loaded = load_manifest(str(manifest_path))
    ok, msg = validate_manifest_for_config(loaded, cfg)
    assert ok and msg == 'ok'


def test_manifest_missing_entries_and_errors(tmp_path):
    store = CANGraphConfigStore()
    cfg = CANGraphConfig(model=store.get_model_config('gat'), dataset=store.get_dataset_config('hcrl_sa'), training=FusionTrainingConfig(), trainer=TrainerConfig())
    cfg.experiment_root = str(tmp_path / 'experiment_runs')

    # Create manifest missing classifier
    manifest = {
        'autoencoder': {'path': '/nonexistent/path.pth', 'model_size': 'teacher', 'training_mode': 'autoencoder', 'distillation': 'no_distillation'}
    }

    manifest_path = tmp_path / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    loaded = load_manifest(str(manifest_path))
    with pytest.raises(ManifestValidationError):
        validate_manifest_for_config(loaded, cfg)
