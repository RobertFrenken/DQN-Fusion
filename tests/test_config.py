import sys
import os
import pytest
from pathlib import Path

# Ensure project root is on PYTHONPATH so `src` package can be imported in tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Workaround: tests run in environments that may not have heavy deps (e.g., torch). Provide lightweight stubs if missing.
try:
    import torch  # noqa: F401
except Exception:
    import types
    sys.modules['torch'] = types.ModuleType('torch')
    sys.modules['torch.nn'] = types.ModuleType('torch.nn')
    sys.modules['torch.cuda'] = types.ModuleType('torch.cuda')

import importlib.util

# Some optional runtime deps may not be installed in the test environment (e.g., hydra_zen, torch).
# Provide minimal stubs so we can import and test configuration helpers without heavy deps.
try:
    import hydra_zen  # noqa: F401
except Exception:
    import types
    sys.modules['hydra_zen'] = types.ModuleType('hydra_zen')
    sys.modules['hydra_zen'].make_config = lambda *a, **k: None
    # Provide a callable `store` that is a no-op for tests
    sys.modules['hydra_zen'].store = lambda *a, **k: None
    sys.modules['hydra_zen'].zen = None

# Import module directly by path to avoid importing top-level `src` package which pulls heavy deps.
_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'config', 'hydra_zen_configs.py'))
_spec = importlib.util.spec_from_file_location('hydra_zen_configs', _config_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CANGraphConfig = _mod.CANGraphConfig
GATConfig = _mod.GATConfig
CANDatasetConfig = _mod.CANDatasetConfig
NormalTrainingConfig = _mod.NormalTrainingConfig
FusionTrainingConfig = _mod.FusionTrainingConfig
KnowledgeDistillationConfig = _mod.KnowledgeDistillationConfig
validate_config = _mod.validate_config


def test_canonical_dir(tmp_path):
    model = GATConfig()
    dataset = CANDatasetConfig(name="hcrl_sa")
    training = NormalTrainingConfig()

    cfg = CANGraphConfig(model=model, dataset=dataset, training=training)
    cfg.experiment_root = str(tmp_path / "experiment_runs")
    cfg.modality = "automotive"
    cfg.model_size = "teacher"
    cfg.distillation = "no_distillation"

    expected = Path(cfg.experiment_root) / cfg.modality / cfg.dataset.name / "supervised" / cfg.model.type / cfg.model_size / cfg.distillation / cfg.training.mode
    assert cfg.canonical_experiment_dir() == expected.resolve()


def test_validate_config_fusion_missing_artifacts(tmp_path):
    model = GATConfig()
    dataset = CANDatasetConfig(name="hcrl_sa", data_path=str(tmp_path / "data"))
    training = FusionTrainingConfig()

    cfg = CANGraphConfig(model=model, dataset=dataset, training=training)
    cfg.experiment_root = str(tmp_path / "experiment_runs")

    # Ensure dataset path exists to avoid dataset path error
    (tmp_path / "data").mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        validate_config(cfg)
    # Ensure error mentions required artifact name and path
    msg = str(excinfo.value)
    assert "autoencoder" in msg and "classifier" in msg



def test_validate_config_kd_requires_teacher_path(tmp_path):
    model = GATConfig()
    dataset = CANDatasetConfig(name="hcrl_sa", data_path=str(tmp_path / "data"))
    training = KnowledgeDistillationConfig()

    cfg = CANGraphConfig(model=model, dataset=dataset, training=training)
    cfg.experiment_root = str(tmp_path / "experiment_runs")

    (tmp_path / "data").mkdir()

    with pytest.raises(ValueError) as excinfo:
        validate_config(cfg)
    assert "teacher_model_path" in str(excinfo.value)


def test_required_artifacts_kd_paths(tmp_path):
    model = GATConfig()
    dataset = CANDatasetConfig(name="hcrl_sa")
    training = KnowledgeDistillationConfig()

    cfg = CANGraphConfig(model=model, dataset=dataset, training=training)
    cfg.experiment_root = str(tmp_path / "experiment_runs")

    # Default teacher path when not explicitly set
    artifacts = cfg.required_artifacts()
    expected_default = cfg.canonical_experiment_dir() / "teacher" / f"best_teacher_model_{cfg.dataset.name}.pth"
    assert "teacher_model" in artifacts
    assert Path(artifacts["teacher_model"]).resolve() == expected_default.resolve()

    # When teacher_model_path is provided, it should be used verbatim
    provided = tmp_path / "teacher_seen.pth"
    training.teacher_model_path = str(provided)
    artifacts2 = cfg.required_artifacts()
    assert Path(artifacts2["teacher_model"]).resolve() == provided.resolve()


def test_required_artifacts_fusion_and_curriculum(tmp_path):
    model = GATConfig()
    dataset = CANDatasetConfig(name="hcrl_sa")
    training = FusionTrainingConfig()

    cfg = CANGraphConfig(model=model, dataset=dataset, training=training)
    cfg.experiment_root = str(tmp_path / "experiment_runs")

    artifacts = cfg.required_artifacts()
    exp_dir = cfg.canonical_experiment_dir()

    # Expect canonical absolute locations under experiment_root
    expected_ae = Path(cfg.experiment_root) / cfg.modality / cfg.dataset.name / "unsupervised" / "vgae" / "teacher" / cfg.distillation / "autoencoder" / "vgae_autoencoder.pth"
    expected_cl = Path(cfg.experiment_root) / cfg.modality / cfg.dataset.name / "supervised" / "gat" / "teacher" / cfg.distillation / "normal" / f"gat_{cfg.dataset.name}_normal.pth"

    assert Path(artifacts["autoencoder"]).resolve() == expected_ae.resolve()
    assert Path(artifacts["classifier"]).resolve() == expected_cl.resolve()

    # Ensure the paths explicitly include model_size and distillation components
    assert "teacher" in str(artifacts["autoencoder"]) and cfg.distillation in str(artifacts["autoencoder"])
    assert "teacher" in str(artifacts["classifier"]) and cfg.distillation in str(artifacts["classifier"])

    # Curriculum training expects a VGAE artifact at canonical unsupervised path
    CurriculumTrainingConfig = _mod.CurriculumTrainingConfig
    c_training = CurriculumTrainingConfig()
    cfg2 = CANGraphConfig(model=model, dataset=dataset, training=c_training)
    cfg2.experiment_root = str(tmp_path / "experiment_runs")
    artifacts_curr = cfg2.required_artifacts()
    expected_curr = Path(cfg2.experiment_root) / cfg2.modality / cfg2.dataset.name / "unsupervised" / "vgae" / "teacher" / cfg2.distillation / "autoencoder" / "vgae_autoencoder.pth"
    assert Path(artifacts_curr["vgae"]).resolve() == expected_curr.resolve()

    # validate_config should raise a clear FileNotFoundError when VGAE is missing for curriculum
    with pytest.raises(FileNotFoundError) as exc2:
        validate_config(cfg2)
    assert "VGAE model" in str(exc2.value) or "vgae" in str(exc2.value)


def test_canonical_dir_autoencoder(tmp_path):
    # Use _mod to avoid importing the top-level `src` package which pulls heavy deps
    VGAEConfig = _mod.VGAEConfig
    AutoencoderTrainingConfig = _mod.AutoencoderTrainingConfig

    model = VGAEConfig()
    dataset = CANDatasetConfig(name="hcrl_sa")
    training = AutoencoderTrainingConfig()

    cfg = CANGraphConfig(model=model, dataset=dataset, training=training)
    cfg.experiment_root = str(tmp_path / "experiment_runs")

    expected = Path(cfg.experiment_root) / cfg.modality / cfg.dataset.name / "unsupervised" / cfg.model.type / cfg.model_size / cfg.distillation / cfg.training.mode
    assert cfg.canonical_experiment_dir() == expected.resolve()
