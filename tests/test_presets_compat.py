import importlib.util
import os
import sys

# Import modules via test harness used in other tests
_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'config', 'hydra_zen_configs.py'))
_spec = importlib.util.spec_from_file_location('hydra_zen_configs', _config_path)
_cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg_mod)

# Make available under src.config to mimic runtime
import types
src_mod = types.ModuleType('src')
src_config_mod = types.ModuleType('src.config')
src_config_mod.hydra_zen_configs = _cfg_mod
src_mod.config = src_config_mod
sys.modules['src'] = src_mod
sys.modules['src.config'] = src_config_mod
sys.modules['src.config.hydra_zen_configs'] = _cfg_mod

# Import training module
_tr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_with_hydra_zen.py'))
_spec2 = importlib.util.spec_from_file_location('train_with_hydra_zen', _tr_path)
_tr_mod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_tr_mod)


def test_preset_naming_conventions():
    presets = _tr_mod.get_preset_configs()
    # Ensure canonical preset names exist
    assert 'autoencoder_set_04' in presets
    assert 'gat_normal_hcrl_ch' in presets
    # Legacy-style names should NOT be accepted by the CLI directly
    legacy = 'automotive_set_04_unsupervised_vgae_teacher_no_autoencoder'
    assert legacy not in presets
