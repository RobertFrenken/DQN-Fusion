import sys
from pathlib import Path
import shutil
import tempfile
import logging

# Ensure project imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ensure project root is on sys.path for test imports
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Skip if hydra_zen not installed to keep local dev environments flexible
try:
    import hydra_zen  # noqa: F401
except Exception:
    import pytest
    pytest.skip("hydra_zen not available in test environment", allow_module_level=True)

from scripts import local_smoke_experiment as smoke_mod


def test_synthetic_data_flag_creates_dataset(tmp_path, monkeypatch):
    args = [
        'local_smoke_experiment.py',
        '--model', 'vgae_student',
        '--dataset', 'hcrl_ch',
        '--training', 'autoencoder',
        '--epochs', '1',
        '--use-synthetic-data',
        '--experiment-root', str(tmp_path / 'experimentruns_test')
    ]
    monkeypatch.setattr(sys, 'argv', args)
    # Run main (it should create the synthetic dataset and not crash)
    smoke_mod.main()
    # Check synthetic dataset path created under experiment root
    synth = Path(str(tmp_path / 'experimentruns_test')) / 'synthetic_dataset'
    assert synth.exists()
    # Basic sanity: contains a train folder and csv
    assert any(synth.glob('**/*.csv'))
