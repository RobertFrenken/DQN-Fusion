import sys
import os
import importlib.util
from pathlib import Path

# Lightweight stubs for optional heavy deps (mirror patterns from existing tests)
try:
    import torch  # noqa: F401
except Exception:
    import types, pickle
    torch_mod = types.ModuleType('torch')
    def _torch_save(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    def _torch_load(path, map_location=None):
        with open(path, 'rb') as f:
            return pickle.load(f)
    torch_mod.save = _torch_save
    torch_mod.load = _torch_load
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    nn_mod = types.ModuleType('torch.nn')
    f_mod = types.ModuleType('torch.nn.functional')
    sys.modules['torch'] = torch_mod
    sys.modules['torch.nn'] = nn_mod
    sys.modules['torch.nn.functional'] = f_mod
    torch = torch_mod

# Minimal pl stub
try:
    import lightning.pytorch as pl  # noqa: F401
except Exception:
    import types
    pl = types.SimpleNamespace()
    class TrainerStub:
        def __init__(self, *args, **kwargs):
            self._kwargs = kwargs
            self.logger = kwargs.get('logger')
        def fit(self, *a, **k):
            return True
        def test(self, *a, **k):
            return {'ok': True}
        def save_checkpoint(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text('checkpoint')
    class LightningModuleStub:
        pass
    pl.Trainer = TrainerStub
    pl.LightningModule = LightningModuleStub
    sys.modules['lightning'] = types.ModuleType('lightning')
    sys.modules['lightning.pytorch'] = pl

# Stub loggers/callbacks
try:
    from lightning.pytorch.loggers import MLFlowLogger, CSVLogger
except Exception:
    import types
    class CSVLoggerStub:
        def __init__(self, save_dir, name):
            self.save_dir = save_dir
            self.name = name
    class MLFlowLoggerStub:
        def __init__(self, experiment_name=None, tracking_uri=None, log_model=False):
            self.experiment_name = experiment_name
            self.tracking_uri = tracking_uri
            self._params = {}
        def log_param(self, k, v):
            self._params[k] = v
    CSVLogger = CSVLoggerStub
    MLFlowLogger = MLFlowLoggerStub
    sys.modules['lightning.pytorch.loggers'] = types.SimpleNamespace(CSVLogger=CSVLogger, MLFlowLogger=MLFlowLogger)

# Minimal callbacks
try:
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, DeviceStatsMonitor
except Exception:
    class ModelCheckpointStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    class EarlyStoppingStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    class DeviceStatsMonitorStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    ModelCheckpoint = ModelCheckpointStub
    EarlyStopping = EarlyStoppingStub
    DeviceStatsMonitor = DeviceStatsMonitorStub
    sys.modules['lightning.pytorch.callbacks'] = types.SimpleNamespace(ModelCheckpoint=ModelCheckpoint, EarlyStopping=EarlyStopping, DeviceStatsMonitor=DeviceStatsMonitor)

# Load config module
_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'config', 'hydra_zen_configs.py'))
_spec = importlib.util.spec_from_file_location('hydra_zen_configs', _cfg_path)
_cfg_mod = importlib.util.module_from_spec(_spec)
# Ensure hydra_zen is available (tests run without optional runtime deps)
try:
    import hydra_zen  # noqa: F401
except Exception:
    import types
    sys.modules['hydra_zen'] = types.ModuleType('hydra_zen')
    sys.modules['hydra_zen'].make_config = lambda *a, **k: None
    sys.modules['hydra_zen'].store = lambda *a, **k: None
    sys.modules['hydra_zen'].zen = None

_spec.loader.exec_module(_cfg_mod)

CANGraphConfig = _cfg_mod.CANGraphConfig
CANDatasetConfig = _cfg_mod.CANDatasetConfig
KnowledgeDistillationConfig = _cfg_mod.KnowledgeDistillationConfig
StudentVGAEConfig = _cfg_mod.StudentVGAEConfig
StudentGATConfig = _cfg_mod.StudentGATConfig
StudentDQNConfig = _cfg_mod.StudentDQNConfig

# Provide minimal 'src' package stubs for imports done at module import time
import types
src_mod = types.ModuleType('src')
src_config_mod = types.ModuleType('src.config')
src_config_mod.hydra_zen_configs = _cfg_mod
src_mod.config = src_config_mod
sys.modules['src'] = src_mod
sys.modules['src.config'] = src_config_mod
sys.modules['src.config.hydra_zen_configs'] = _cfg_mod

# Minimal stubs for 'src.training' symbols imported by the training script
training_mod = types.ModuleType('src.training')
training_mod.CANGraphDataModule = object
training_mod.load_dataset = lambda *a, **k: ([], [], 1)
training_mod.create_dataloaders = lambda *a, **k: (None, None)
class DummyLightningModule:
    def __init__(self, *a, **k):
        self.batch_size = getattr(self, 'batch_size', 64)
    @classmethod
    def load_from_checkpoint(cls, *a, **k):
        return cls()
    def state_dict(self):
        # Return a minimal state dict so trainer_manager can save it during tests
        return {'weights': [1, 2, 3]}
training_mod.CANGraphLightningModule = DummyLightningModule
training_mod.FusionLightningModule = DummyLightningModule
training_mod.create_fusion_prediction_cache = lambda **k: ([], [], [], [], [], [])
training_mod.EnhancedCANGraphDataModule = object
training_mod.CurriculumCallback = object

can_graph_data_mod = types.ModuleType('src.training.can_graph_data')
can_graph_data_mod.CANGraphDataModule = object
can_graph_data_mod.load_dataset = training_mod.load_dataset
can_graph_data_mod.create_dataloaders = training_mod.create_dataloaders

can_graph_module_mod = types.ModuleType('src.training.can_graph_module')
can_graph_module_mod.CANGraphLightningModule = DummyLightningModule

fusion_lightning_mod = types.ModuleType('src.training.fusion_lightning')
fusion_lightning_mod.FusionLightningModule = DummyLightningModule
fusion_lightning_mod.FusionPredictionCache = object

prediction_cache_mod = types.ModuleType('src.training.prediction_cache')
prediction_cache_mod.create_fusion_prediction_cache = training_mod.create_fusion_prediction_cache

enhanced_dm_mod = types.ModuleType('src.training.enhanced_datamodule')
enhanced_dm_mod.EnhancedCANGraphDataModule = object
enhanced_dm_mod.CurriculumCallback = object

sys.modules['src.training'] = training_mod
sys.modules['src.training.can_graph_data'] = can_graph_data_mod
sys.modules['src.training.can_graph_module'] = can_graph_module_mod
sys.modules['src.training.fusion_lightning'] = fusion_lightning_mod
sys.modules['src.training.prediction_cache'] = prediction_cache_mod
sys.modules['src.training.enhanced_datamodule'] = enhanced_dm_mod

# Import trainer module by path
_tr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_with_hydra_zen.py'))
_spec2 = importlib.util.spec_from_file_location('train_with_hydra_zen', _tr_path)
_tr_mod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_tr_mod)
HydraZenTrainer = _tr_mod.HydraZenTrainer


def _make_cfg(model_config, tmp_path):
    cfg = CANGraphConfig(model=model_config, dataset=CANDatasetConfig(name='hcrl_sa'), training=KnowledgeDistillationConfig())
    cfg.experiment_root = str(tmp_path / 'experiment_runs')
    # Ensure a dataset path exists to satisfy validation checks if any
    data_dir = tmp_path / 'data' / 'hcrl_sa'
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg.dataset.data_path = str(data_dir)
    # Disable optimizer's batch-size tuning for the smoke test environment
    cfg.training.optimize_batch_size = False
    return cfg


import pytest

@pytest.mark.parametrize('model_cfg', [StudentVGAEConfig(), StudentGATConfig(), StudentDQNConfig()])
def test_knowledge_distillation_smoke_all_models(tmp_path, model_cfg):
    """Smoke test: run knowledge distillation training for each student model and ensure a model state is saved."""
    cfg = _make_cfg(model_cfg, tmp_path)

    # Provide a dummy teacher checkpoint to satisfy validation
    teacher_path = tmp_path / 'teacher.pth'
    teacher_path.write_text('teacher')
    cfg.training.teacher_model_path = str(teacher_path)
    # Ensure trainer precision matches distillation mixed-precision requirement
    cfg.trainer.precision = '16-mixed'

    trainer_manager = HydraZenTrainer(cfg)
    # Ensure the training module will create a Trainer stub exposing fit/test
    _tr_mod.pl.Trainer = pl.Trainer

    # If the training module previously set a fallback Trainer lambda, override the setup_trainer
    # to return a small stub with fit/test/save_checkpoint so training can proceed under tests.
    class LocalTrainer:
        def __init__(self, *a, **k):
            self._kwargs = k
        def fit(self, *a, **k):
            return True
        def test(self, *a, **k):
            return {'ok': True}
        def save_checkpoint(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text('checkpoint')

    _tr_mod.HydraZenTrainer.setup_trainer = lambda self: LocalTrainer()

    model, trainer = trainer_manager.train()

    paths = trainer_manager.get_hierarchical_paths()
    expected_model = paths['model_save_dir'] / f"{cfg.model.type}_{cfg.training.mode}.pth"
    assert expected_model.exists(), f"Expected saved model at {expected_model}"
