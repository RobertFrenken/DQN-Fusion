import logging
import sys
from pathlib import Path
import os
import pytest

# Ensure project root is on sys.path for test imports (helps running this test file directly)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Skip tests if hydra_zen is not installed in this environment (keeps CI/dev robust)
try:
    import hydra_zen  # noqa: F401
except Exception:
    import pytest
    pytest.skip("hydra_zen not available in test environment", allow_module_level=True)

from src.config.hydra_zen_configs import CANGraphConfigStore, validate_config


def test_validate_config_warns_when_dataset_missing(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    store = CANGraphConfigStore()
    cfg = store.create_config(model_type="vgae", dataset_name="hcrl_ch", training_mode="autoencoder")

    # Point to a non-existent dataset path
    cfg.dataset.data_path = str(tmp_path / "no_such_dataset")

    with caplog.at_level(logging.WARNING):
        # Should not raise, only warn
        assert validate_config(cfg) is True

    msgs = [r.message for r in caplog.records]
    assert any("Dataset path does not exist" in m for m in msgs), "Expected a dataset-missing warning"


from typing import Optional

def create_minimal_csv_dataset(root: Path, num_rows: int = 200, seed: Optional[int] = None):
    """Create a tiny dataset structure with a few CSVs that the preprocessing can discover."""
    from scripts.local_smoke_experiment import create_synthetic_dataset
    return create_synthetic_dataset(root, num_rows=num_rows, seed=seed)


def test_local_smoke_run_invokes_trainer(monkeypatch, tmp_path, caplog):
    # Prepare a tiny dataset and a separate experiment root
    dataset_dir = tmp_path / "dataset"
    create_minimal_csv_dataset(dataset_dir)
    import scripts.local_smoke_experiment as smoke_mod

    # Stub out the actual training to keep the test fast and deterministic
    called = {"run": False}

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg
            self.batch_size = 1
        def train(self):
            called["run"] = True
        def get_hierarchical_paths(self):
            # Create a minimal canonical directory structure for the smoke run
            base = Path(self.cfg.experiment_root)
            exp_dir = base / 'automotive' / self.cfg.dataset.name / 'unsupervised' / self.cfg.model.type / 'student' / 'no_distillation' / self.cfg.training.mode
            exp_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = exp_dir / 'checkpoints'
            checkpoint.mkdir(parents=True, exist_ok=True)
            model_dir = exp_dir / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            logs = exp_dir / 'logs'
            logs.mkdir(parents=True, exist_ok=True)
            mlruns = exp_dir / '.mlruns'
            mlruns.mkdir(parents=True, exist_ok=True)
            return {
                'experiment_dir': exp_dir,
                'checkpoint_dir': checkpoint,
                'model_save_dir': model_dir,
                'log_dir': logs,
                'mlruns_dir': mlruns
            }

    monkeypatch.setattr(smoke_mod, 'TRAINER_FACTORY', lambda cfg: FakeTrainer(cfg))

    # Run main with CLI-like args
    exp_root = tmp_path / "experimentruns_test"
    test_argv = [
        "local_smoke_experiment.py",
        "--model",
        "vgae_student",
        "--dataset",
        "hcrl_ch",
        "--training",
        "autoencoder",
        "--epochs",
        "1",
        "--run",
        "--data-path",
        str(dataset_dir),
        "--experiment-root",
        str(exp_root),
    ]

    monkeypatch.setattr(sys, "argv", test_argv)

    caplog.set_level(logging.INFO)
    # Should not raise and should call our fake_train
    smoke_mod.main()

    assert called["run"] is True, "Expected trainer.train() to be invoked in smoke run"

    # Verify canonical directories were created
    # The smoke script uses a CANGraphConfig.canonical_experiment_dir() internally
    # Check that the experiment root contains the modality/dataset structure
    assert (exp_root).exists()
    # check some subpath exists
    expected_subdir = exp_root / "automotive" / "hcrl_ch"
    assert expected_subdir.exists(), f"Expected experiment subdir created: {expected_subdir}"
