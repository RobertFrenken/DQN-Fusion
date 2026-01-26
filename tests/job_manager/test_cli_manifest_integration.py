import json
import os
import sys
import importlib.util
from pathlib import Path
import pytest

# Load train_with_hydra_zen module by path (avoid heavy import at collection)
# Provide a lightweight shim for hydra_zen if it's not installed in the test environment
_t_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'train_with_hydra_zen.py'))
_spec = importlib.util.spec_from_file_location('train_with_hydra_zen', _t_path)
_train_mod = importlib.util.module_from_spec(_spec)
import types
if 'hydra_zen' not in sys.modules:
    sys.modules['hydra_zen'] = types.SimpleNamespace(make_config=lambda *a, **k: None, store=lambda *a, **k: None, zen=None)
_spec.loader.exec_module(_train_mod)

# Load dependency_manifest util by path (avoid heavy imports in fusion_training)
_dm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'utils', 'dependency_manifest.py'))
_spec2 = importlib.util.spec_from_file_location('dependency_manifest', _dm_path)
_fusion_mod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_fusion_mod)


def _write_dummy(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('dummy')


def test_train_with_hydra_zen_manifest_applies_paths(tmp_path, monkeypatch):
    # Create dummy artifact files
    ae = tmp_path / 'vgae_autoencoder.pth'
    clf = tmp_path / 'gat_classifier.pth'
    _write_dummy(ae)
    _write_dummy(clf)

    manifest = {
        'autoencoder': {'path': str(ae), 'model_size': 'teacher', 'training_mode': 'autoencoder', 'distillation': 'no_distillation'},
        'classifier': {'path': str(clf), 'model_size': 'teacher', 'training_mode': 'normal', 'distillation': 'no_distillation'}
    }

    manifest_path = tmp_path / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    # Unit-test the helper directly (avoids heavy config store integration)
    from types import SimpleNamespace
    cfg = SimpleNamespace(training=SimpleNamespace(mode='fusion'))

    _train_mod.apply_manifest_to_config(cfg, str(manifest_path))

    assert getattr(cfg.training, 'autoencoder_path', None) == str(ae)
    assert getattr(cfg.training, 'classifier_path', None) == str(clf)


def test_dependency_manifest_validate_paths_success(tmp_path):
    # Create dummy artifact files
    ae = tmp_path / 'vgae_autoencoder2.pth'
    clf = tmp_path / 'gat_classifier2.pth'
    _write_dummy(ae)
    _write_dummy(clf)

    manifest = {
        'autoencoder': {'path': str(ae), 'model_size': 'teacher', 'training_mode': 'autoencoder', 'distillation': 'no_distillation'},
        'classifier': {'path': str(clf), 'model_size': 'teacher', 'training_mode': 'normal', 'distillation': 'no_distillation'}
    }

    manifest_path = tmp_path / 'manifest2.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    loaded = _fusion_mod.load_manifest(str(manifest_path))
    ok, msg = _fusion_mod.validate_manifest_for_config(loaded, type('cfg', (), {'training': type('t', (), {'mode':'fusion'})()}))
    assert ok and msg == 'ok'

def test_train_with_hydra_zen_manifest_invalid_raises(tmp_path, monkeypatch):
    # Create manifest missing classifier
    manifest = {
        'autoencoder': {'path': '/nonexistent/path.pth', 'model_size': 'teacher', 'training_mode': 'autoencoder', 'distillation': 'no_distillation'}
    }

    manifest_path = tmp_path / 'bad_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    # Monkeypatch train to avoid heavy training calls
    monkeypatch.setattr(_train_mod.HydraZenTrainer, 'train', lambda self: (None, None), raising=True)

    argv = ['train_with_hydra_zen.py', '--model', 'gat', '--dataset', 'hcrl_sa', '--training', 'fusion', '--dependency-manifest', str(manifest_path)]
    monkeypatch.setattr(sys, 'argv', argv)

    with pytest.raises(Exception):
        _train_mod.main()