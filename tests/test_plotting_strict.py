import importlib.util
import sys
import types
import os
import pytest

# Provide minimal plotting_config before importing plotting_utils
plot_mod = types.ModuleType('config.plotting_config')
plot_mod.COLOR_SCHEMES = {}
plot_mod.apply_publication_style = lambda *a, **k: None
plot_mod.save_publication_figure = lambda *a, **k: None
sys.modules['config.plotting_config'] = plot_mod

# Provide lightweight torch stub to satisfy imports in plotting_utils
if 'torch' not in sys.modules:
    torch_stub = types.ModuleType('torch')
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules['torch'] = torch_stub

# Import plotting_utils by path to avoid top-level package imports
p_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'utils', 'plotting_utils.py'))
spec = importlib.util.spec_from_file_location('plotting_utils', p_path)
pu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pu)


def test_plot_fusion_analysis_requires_existing_figure():
    with pytest.raises(ValueError) as exc:
        pu.plot_fusion_analysis([], [], [], [], 'hcrl_sa', current_fig=None, current_axes=None)
    assert 'plot_fusion_analysis requires a pre-existing figure' in str(exc.value)
