import sys
import os
import importlib.util
from pathlib import Path

# Provide lightweight hydra_zen stub if missing to allow import
try:
    import hydra_zen  # noqa: F401
except Exception:
    import types
    sys.modules['hydra_zen'] = types.ModuleType('hydra_zen')
    sys.modules['hydra_zen'].make_config = lambda *a, **k: None
    sys.modules['hydra_zen'].store = lambda *a, **k: None
    sys.modules['hydra_zen'].zen = None

# Load config module by path
_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'config', 'hydra_zen_configs.py'))
_spec = importlib.util.spec_from_file_location('hydra_cfg_mod', _cfg_path)
_cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg_mod)

CANGraphConfig = _cfg_mod.CANGraphConfig
CANDatasetConfig = _cfg_mod.CANDatasetConfig
KnowledgeDistillationConfig = _cfg_mod.KnowledgeDistillationConfig


def test_student_distillation_canonical_paths(tmp_path):
    model = _cfg_mod.StudentGATConfig()  # gat_student
    dataset = CANDatasetConfig(name='hcrl_sa')
    training = KnowledgeDistillationConfig()

    cfg = CANGraphConfig(model=model, dataset=dataset, training=training)
    cfg.experiment_root = str(tmp_path / 'experiment_runs')

    exp_dir = cfg.canonical_experiment_dir()
    model_save_dir = exp_dir / 'models'
    expected_model_file = model_save_dir / f"{cfg.model.type}_{cfg.training.mode}.pth"

    # Ensure the path is the canonical location and uses 'supervised' learning_type for a GAT student
    assert expected_model_file.resolve() == (Path(cfg.experiment_root) / cfg.modality / cfg.dataset.name / 'supervised' / 'gat' / 'student' / cfg.distillation / cfg.training.mode / 'models' / f"{cfg.model.type}_{cfg.training.mode}.pth").resolve()


def test_vgae_student_uses_unsupervised_learning_type(tmp_path):
    model = _cfg_mod.StudentVGAEConfig()
    dataset = _cfg_mod.CANDatasetConfig(name='hcrl_sa')
    training = _cfg_mod.KnowledgeDistillationConfig()

    cfg = _cfg_mod.CANGraphConfig(model=model, dataset=dataset, training=training)
    cfg.experiment_root = str(tmp_path / 'experiment_runs')

    exp_dir = cfg.canonical_experiment_dir()
    # VGAE student should use unsupervised learning_type (inferred from base arch)
    expected = Path(cfg.experiment_root) / cfg.modality / cfg.dataset.name / 'unsupervised' / 'vgae' / 'student' / cfg.distillation / cfg.training.mode
    assert exp_dir.resolve() == expected.resolve()
