import sys
import types
from pathlib import Path
import json

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.local_smoke_experiment as smoke_mod


def test_integration_smoke_end_to_end(tmp_path, monkeypatch):
    # Create synthetic dataset folder
    dataset_dir = tmp_path / 'dataset'
    dataset_dir.mkdir()
    # create a small csv
    from scripts.local_smoke_experiment import create_synthetic_dataset
    create_synthetic_dataset(dataset_dir, num_rows=50, seed=42)

    # Stub out heavy config imports by inserting a fake config store module
    fake_cfg_mod = types.ModuleType('src.config.hydra_zen_configs')

    class FakeStore:
        def create_config(self, model_type='vgae_student', dataset_name='hcrl_ch', training_mode='autoencoder', **kwargs):
            # Minimal config object expected by local_smoke_experiment
            model = types.SimpleNamespace(type=model_type)
            class FakeDataset:
                def __init__(self, name, data_path):
                    self.name = name
                    self.data_path = data_path
                def __fspath__(self):
                    # Pathlib uses this to coerce to string when joining paths
                    return self.name
                def __str__(self):
                    return self.name

            dataset = FakeDataset(dataset_name, str(dataset_dir))
            training = types.SimpleNamespace(mode=training_mode)
            trainer = types.SimpleNamespace(devices=1)
            # Provide required path attributes expected by ExperimentPathManager
            cfg = types.SimpleNamespace(
                model=model,
                dataset=dataset,
                training=training,
                trainer=trainer,
                experiment_root=None,
                modality='automotive',
                learning_type='unsupervised',
                model_architecture='vgae',
                model_size='student',
                distillation='no',
                training_mode=training_mode
            )
            return cfg

        def get_dataset_config(self, name):
            return types.SimpleNamespace(name=name, data_path=str(dataset_dir))

    fake_cfg_mod.CANGraphConfigStore = FakeStore
    sys.modules['src.config.hydra_zen_configs'] = fake_cfg_mod

    # Stub train_with_hydra_zen module to avoid heavy imports
    fake_train_mod = types.ModuleType('train_with_hydra_zen')
    class DummyTrainer:
        def __init__(self, cfg):
            self.cfg = cfg
        def train(self):
            # create a dummy model file in a canonical run dir derived from cfg
            base = Path(self.cfg.experiment_root)
            exp_dir = base / self.cfg.modality / self.cfg.dataset / self.cfg.learning_type / self.cfg.model_architecture / self.cfg.model_size / self.cfg.distillation / self.cfg.training_mode
            run_dir = exp_dir / 'run_000'
            run_dir.mkdir(parents=True, exist_ok=True)
            model_file = run_dir / 'model.pt'
            model_file.write_bytes(b'dummy')
        def get_hierarchical_paths(self):
            base = Path(self.cfg.experiment_root)
            exp_dir = base / self.cfg.modality / self.cfg.dataset / self.cfg.learning_type / self.cfg.model_architecture / self.cfg.model_size / self.cfg.distillation / self.cfg.training_mode
            return {
                'experiment_dir': exp_dir,
                'checkpoint_dir': exp_dir / 'checkpoints',
                'model_save_dir': exp_dir / 'models',
                'log_dir': exp_dir / 'logs',
                'mlruns_dir': exp_dir / '.mlruns'
            }
    fake_train_mod.HydraZenTrainer = DummyTrainer
    sys.modules['train_with_hydra_zen'] = fake_train_mod

    # Ensure missing optional deps don't break imports during tests
    fake_omegaconf = types.ModuleType('omegaconf')
    fake_omegaconf.DictConfig = dict
    sys.modules['omegaconf'] = fake_omegaconf

    # Provide a lightweight fake implementation for src.utils.experiment_paths so summary writing works
    fake_ep_mod = types.ModuleType('src.utils.experiment_paths')
    class EP:
        def __init__(self, cfg):
            self.cfg = cfg
            self.base = Path(cfg.experiment_root)
            self.exp_dir = self.base / cfg.modality / cfg.dataset / cfg.learning_type / cfg.model_architecture / cfg.model_size / cfg.distillation / cfg.training_mode
        def get_run_dir_safe(self, run_id=None):
            run_dir = self.exp_dir / (f"run_{run_id:03d}" if run_id is not None else 'run_000')
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        def get_experiment_dir(self):
            return self.exp_dir
    fake_ep_mod.ExperimentPathManager = EP
    sys.modules['src.utils.experiment_paths'] = fake_ep_mod

    # Use the fake TRAINER_FACTORY to return our DummyTrainer
    monkeypatch.setattr(smoke_mod, 'TRAINER_FACTORY', lambda cfg: DummyTrainer(cfg))

    # Run smoke main with synthetic data
    monkeypatch.setattr(sys, 'argv', [
        'local_smoke_experiment.py',
        '--model', 'vgae_student',
        '--dataset', 'hcrl_ch',
        '--training', 'autoencoder',
        '--use-synthetic-data',
        '--run',
        '--write-summary',
        '--experiment-root', str(tmp_path / 'experimentruns_test'),
    ])

    # Execute
    smoke_mod.main()

    # Verify summary.json exists in a run directory
    root = tmp_path / 'experimentruns_test'
    # Find summary.json
    summaries = list(root.rglob('summary.json'))
    assert summaries, 'Expected at least one summary.json to be written'
    data = json.loads(summaries[0].read_text())
    assert data.get('model') in ('vgae_student', 'vgae')

    # Check dummy model file exists in run dir(s)
    models = list(root.rglob('model.pt'))
    assert models, 'Expected dummy model file to be created by DummyTrainer'
