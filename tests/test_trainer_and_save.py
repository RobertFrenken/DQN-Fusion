import sys
import os
import importlib.util
from pathlib import Path
import tempfile

# Lightweight stubs for optional heavy deps
# Provide a minimal 'torch' if not present
try:
    import torch  # noqa: F401
except Exception:
    import types
    import pickle
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
    # minimal nn modules to satisfy imports
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
    class TensorBoardLoggerStub:
        def __init__(self, save_dir, name):
            self.save_dir = save_dir
            self.name = name
    CSVLogger = CSVLoggerStub
    MLFlowLogger = MLFlowLoggerStub
    TensorBoardLogger = TensorBoardLoggerStub
    sys.modules['lightning.pytorch.loggers'] = types.SimpleNamespace(CSVLogger=CSVLogger, MLFlowLogger=MLFlowLogger, TensorBoardLogger=TensorBoardLogger)

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
    # Provide a minimal Tuner stub
    class TunerStub:
        def __init__(self, trainer):
            self.trainer = trainer
        def scale_batch_size(self, *a, **k):
            return None
    sys.modules['lightning.pytorch.tuner'] = types.SimpleNamespace(Tuner=TunerStub)

# Ensure hydra_zen is available (tests run without optional runtime deps)
try:
    import hydra_zen  # noqa: F401
except Exception:
    import types
    sys.modules['hydra_zen'] = types.ModuleType('hydra_zen')
    sys.modules['hydra_zen'].make_config = lambda *a, **k: None
    sys.modules['hydra_zen'].store = lambda *a, **k: None
    sys.modules['hydra_zen'].zen = None

# Import config module by path (avoid heavy import of entire package)
_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'config', 'hydra_zen_configs.py'))
_spec = importlib.util.spec_from_file_location('hydra_zen_configs', _config_path)
_cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg_mod)

CANGraphConfig = _cfg_mod.CANGraphConfig
GATConfig = _cfg_mod.GATConfig
CANDatasetConfig = _cfg_mod.CANDatasetConfig
NormalTrainingConfig = _cfg_mod.NormalTrainingConfig

# Provide minimal 'src' package stubs for imports done at module import time
import types
src_mod = types.ModuleType('src')
src_config_mod = types.ModuleType('src.config')
src_config_mod.hydra_zen_configs = _cfg_mod
src_mod.config = src_config_mod
sys.modules['src'] = src_mod
sys.modules['src.config'] = src_config_mod
sys.modules['src.config.hydra_zen_configs'] = _cfg_mod

# Provide missing factory functions expected by the training script (no-op placeholders)
def _create_gat_normal_config(dataset, **_):
    return _cfg_mod.CANGraphConfig(model=_cfg_mod.GATConfig(), dataset=_cfg_mod.CANDatasetConfig(name=dataset), training=_cfg_mod.NormalTrainingConfig())

def _create_distillation_config(dataset, **_):
    return _cfg_mod.CANGraphConfig(model=_cfg_mod.GATConfig(), dataset=_cfg_mod.CANDatasetConfig(name=dataset), training=_cfg_mod.KnowledgeDistillationConfig())

def _create_autoencoder_config(dataset, **_):
    return _cfg_mod.CANGraphConfig(model=_cfg_mod.VGAEConfig(), dataset=_cfg_mod.CANDatasetConfig(name=dataset), training=_cfg_mod.AutoencoderTrainingConfig())

def _create_fusion_config(dataset, **_):
    return _cfg_mod.CANGraphConfig(model=_cfg_mod.GATConfig(), dataset=_cfg_mod.CANDatasetConfig(name=dataset), training=_cfg_mod.FusionTrainingConfig())

setattr(_cfg_mod, 'create_gat_normal_config', _create_gat_normal_config)
setattr(_cfg_mod, 'create_distillation_config', _create_distillation_config)
setattr(_cfg_mod, 'create_autoencoder_config', _create_autoencoder_config)
setattr(_cfg_mod, 'create_fusion_config', _create_fusion_config)

# Minimal stubs for 'src.training' symbols imported by the training script
training_mod = types.ModuleType('src.training')
# Minimal datamodule/dataloader helpers
training_mod.CANGraphDataModule = object
training_mod.load_dataset = lambda *a, **k: ([], [], 1)
training_mod.create_dataloaders = lambda *a, **k: (None, None)
# Minimal Lightning modules used for instantiation
class DummyLightningModule:
    def __init__(self, *a, **k):
        self.batch_size = getattr(self, 'batch_size', 64)
    @classmethod
    def load_from_checkpoint(cls, *a, **k):
        return cls()
training_mod.CANGraphLightningModule = DummyLightningModule
training_mod.FusionLightningModule = DummyLightningModule
training_mod.create_fusion_prediction_cache = lambda **k: ([], [], [], [], [], [])
training_mod.EnhancedCANGraphDataModule = object
training_mod.CurriculumCallback = object
# Provide submodule entries to satisfy 'from src.training.can_graph_data import ...' style imports
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
# Ensure our stubs for lightning are available to the module
_spec2.loader.exec_module(_tr_mod)

HydraZenTrainer = _tr_mod.HydraZenTrainer


def test_setup_trainer_creates_trainer_and_dirs(tmp_path):
    model = GATConfig()
    dataset = CANDatasetConfig(name='hcrl_sa')
    training = NormalTrainingConfig()
    cfg = CANGraphConfig(model=model, dataset=dataset, training=training)
    cfg.experiment_root = str(tmp_path / 'experiment_runs')

    # Create a valid dataset path to satisfy strict validation
    dataset_path = tmp_path / 'data' / dataset.name
    dataset_path.mkdir(parents=True, exist_ok=True)
    cfg.dataset.data_path = str(dataset_path)

    trainer_manager = HydraZenTrainer(cfg)
    trainer = trainer_manager.setup_trainer()

    # Expect trainer stub
    assert hasattr(trainer, '_kwargs') or hasattr(trainer, 'logger')

    # Ensure canonical directories exist
    paths = trainer_manager.get_hierarchical_paths()
    for d in ['experiment_dir', 'checkpoint_dir', 'model_save_dir', 'log_dir', 'mlruns_dir']:
        assert paths[d].exists()


def test_save_state_dict_writes_and_logs(tmp_path, monkeypatch):
    model = GATConfig()
    dataset = CANDatasetConfig(name='hcrl_sa')
    training = NormalTrainingConfig()
    cfg = CANGraphConfig(model=model, dataset=dataset, training=training)
    cfg.experiment_root = str(tmp_path / 'experiment_runs')

    # Create a valid dataset path to satisfy strict validation
    dataset_path = tmp_path / 'data' / dataset.name
    dataset_path.mkdir(parents=True, exist_ok=True)
    cfg.dataset.data_path = str(dataset_path)

    trainer_manager = HydraZenTrainer(cfg)
    paths = trainer_manager.get_hierarchical_paths()

    # Create a dummy model object exposing state_dict
    class DummyModel:
        def __init__(self):
            self._state = {'w': [1,2,3]}
        def state_dict(self):
            return self._state

    dummy = DummyModel()

    saved_path = trainer_manager._save_state_dict(dummy, paths['model_save_dir'], 'dummy.pth')
    assert Path(saved_path).exists()

    # Stub mlflow to capture log_param calls
    import types
    mlflow_stub = types.SimpleNamespace()
    logged = {}
    def set_tracking_uri(uri):
        logged['uri'] = uri
    def log_param(k, v):
        logged[k] = v
    mlflow_stub.set_tracking_uri = set_tracking_uri
    mlflow_stub.log_param = log_param

    monkeypatch.setitem(sys.modules, 'mlflow', mlflow_stub)

    trainer_manager._log_model_path_to_mlflow(saved_path, paths['mlruns_dir'])
    assert 'model_path' in logged
    assert logged['model_path'] == str(saved_path)
