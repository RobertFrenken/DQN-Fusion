import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import math
from src.config import hydra_zen_configs as cfg_mod
from src.training.can_graph_module import CANGraphLightningModule


def count_parameters(model):
    return sum(int(p.numel()) for p in model.parameters() if p is not None)


@pytest.mark.parametrize(
    "config_cls, model_type",
    [
        (cfg_mod.GATConfig, 'gat'),
        (cfg_mod.StudentGATConfig, 'gat_student'),
        (cfg_mod.VGAEConfig, 'vgae'),
        (cfg_mod.StudentVGAEConfig, 'vgae_student'),
        (cfg_mod.DQNConfig, 'dqn'),
        (cfg_mod.StudentDQNConfig, 'dqn_student'),
    ],
)
def test_model_parameters_within_budget(config_cls, model_type):
    # instantiate config and ensure target is present
    cfg = config_cls()
    assert hasattr(cfg, 'target_parameters'), f"Config {config_cls.__name__} must define 'target_parameters'"
    target = getattr(cfg, 'target_parameters')

    # build model via lightning module to respect wiring
    training_config = type('T', (), {'batch_size': 32, 'mode': 'normal'})()
    lm = CANGraphLightningModule(model_config=cfg, training_config=training_config, model_type=model_type, training_mode='normal', num_ids=50)
    model = lm.model

    # compute param count (int)
    total_params = count_parameters(model)

    # allow a tolerance band (15%) for architectural implementation differences
    tol = math.ceil(target * 0.15)
    lower = target - tol
    upper = target + tol

    assert lower <= total_params <= upper, (
        f"Model {model_type} params={total_params} not within ±15% of target {target} (allowed {lower}-{upper})"
    )
