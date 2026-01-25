import importlib.util
import importlib
import os
import sys
import types
import pytest

# Ensure hydra config module can import
try:
    import hydra_zen  # noqa: F401
except Exception:
    sys.modules['hydra_zen'] = types.ModuleType('hydra_zen')
    sys.modules['hydra_zen'].make_config = lambda *a, **k: None
    sys.modules['hydra_zen'].store = lambda *a, **k: None
    sys.modules['hydra_zen'].zen = None

# Provide lightweight torch stub to satisfy imports in config
if 'torch' not in sys.modules:
    torch_stub = types.ModuleType('torch')
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.get_num_threads = lambda: 4
    sys.modules['torch'] = torch_stub

# Import config module
_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'config', 'hydra_zen_configs.py'))
_spec = importlib.util.spec_from_file_location('hydra_cfg', _cfg_path)
_cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg_mod)

CANDatasetConfig = _cfg_mod.CANDatasetConfig
GATConfig = _cfg_mod.GATConfig
NormalTrainingConfig = _cfg_mod.NormalTrainingConfig
CANGraphConfig = _cfg_mod.CANGraphConfig

# Import trainer module
_tr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'train_with_hydra_zen.py'))
_spec2 = importlib.util.spec_from_file_location('train_with_hydra_zen', _tr_path)
_tr_mod = importlib.util.module_from_spec(_spec2)
# Provide a minimal lightning.pytorch.tuner.Tuner with scale_batch_size attribute we can monkeypatch
import types as _types
pl_mod = _types.ModuleType('lightning.pytorch')
tuner_mod = _types.ModuleType('lightning.pytorch.tuner')
class TunerStub:
    def __init__(self, trainer):
        self.trainer = trainer
    def scale_batch_size(self, *a, **k):
        raise RuntimeError('tuner fail')

tuner_mod.Tuner = TunerStub
sys.modules['lightning.pytorch.tuner'] = tuner_mod
sys.modules['lightning.pytorch'] = pl_mod
_spec2.loader.exec_module(_tr_mod)

HydraZenTrainer = _tr_mod.HydraZenTrainer


def test_optimize_batch_size_raises_on_tuner_failure(tmp_path):
    model = GATConfig()
    dataset = CANDatasetConfig(name='hcrl_sa')
    # Provide explicit dataset path to satisfy validation
    dataset.data_path = str(tmp_path / 'data')
    (tmp_path / 'data').mkdir()

    training = NormalTrainingConfig()
    # enable optimize
    training.optimize_batch_size = True

    cfg = CANGraphConfig(model=model, dataset=dataset, training=training)
    cfg.experiment_root = str(tmp_path / 'experiment_runs')

    trainer_manager = HydraZenTrainer(cfg)

    # Create dummy model and datamodule
    class DummyModel:
        batch_size = 32
    class DummyDataModule:
        batch_size = 32

    with pytest.raises(RuntimeError) as exc:
        trainer_manager._optimize_batch_size_with_datamodule(DummyModel(), DummyDataModule())
    assert 'Batch size optimization failed' in str(exc.value)
