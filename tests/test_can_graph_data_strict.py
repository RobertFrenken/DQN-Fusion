import importlib.util
import os
import sys
from pathlib import Path
import types
import pytest

# Provide lightweight torch stub to allow importing modules without heavy runtime deps
try:
    import torch  # noqa: F401
except Exception:
    import types
    torch_mod = types.ModuleType('torch')
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    # minimal utils.data
    utils_mod = types.ModuleType('torch.utils')
    data_mod = types.ModuleType('torch.utils.data')
    data_mod.Subset = list
    data_mod.random_split = lambda dataset, splits: ([], [])
    utils_mod.data = data_mod
    sys.modules['torch'] = torch_mod
    sys.modules['torch.utils'] = utils_mod
    sys.modules['torch.utils.data'] = data_mod

# Stub torch_geometric loader to avoid heavy imports
import types
sys.modules['torch_geometric'] = types.ModuleType('torch_geometric')
loader_mod = types.ModuleType('torch_geometric.loader')
class TGDataLoader:
    pass
loader_mod.DataLoader = TGDataLoader
sys.modules['torch_geometric.loader'] = loader_mod

# Stub Lightning to avoid importing heavy runtime deps
import types
pl_mod = types.ModuleType('lightning')
pl_pyt = types.ModuleType('lightning.pytorch')
pl_pyt.LightningDataModule = object
sys.modules['lightning'] = pl_mod
sys.modules['lightning.pytorch'] = pl_pyt

# Provide minimal `src.preprocessing.preprocessing` stub
import types
pre_mod = types.ModuleType('src.preprocessing.preprocessing')
class GraphDataset:
    def __init__(self, graphs):
        self._g = graphs
    def __len__(self):
        return len(self._g)
pre_mod.GraphDataset = GraphDataset
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.preprocessing'] = types.ModuleType('src.preprocessing')
sys.modules['src.preprocessing.preprocessing'] = pre_mod

# Import load_dataset function by path
_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'training', 'can_graph_data.py'))
_spec = importlib.util.spec_from_file_location('can_graph_data', _cfg_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_dataset = _mod.load_dataset

# Import config types
_cfg_conf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'config', 'hydra_zen_configs.py'))
# Provide a minimal hydra_zen stub to satisfy imports in the config module
try:
    import hydra_zen  # noqa: F401
except Exception:
    import types as _types
    sys.modules['hydra_zen'] = _types.ModuleType('hydra_zen')
    sys.modules['hydra_zen'].make_config = lambda *a, **k: None
    sys.modules['hydra_zen'].store = lambda *a, **k: None
    sys.modules['hydra_zen'].zen = None

_spec2 = importlib.util.spec_from_file_location('hydra_cfg', _cfg_conf_path)
_cfg_mod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_cfg_mod)

CANDatasetConfig = _cfg_mod.CANDatasetConfig
GATConfig = _cfg_mod.GATConfig
NormalTrainingConfig = _cfg_mod.NormalTrainingConfig


def test_load_dataset_requires_explicit_data_path(tmp_path):
    model = GATConfig()
    dataset = CANDatasetConfig(name='hcrl_sa')
    # Simulate a config that explicitly has no data_path set
    dataset.data_path = None

    training = NormalTrainingConfig()
    cfg = types.SimpleNamespace()
    cfg.dataset = dataset

    with pytest.raises(ValueError) as exc:
        load_dataset('hcrl_sa', cfg)
    assert 'Dataset path must be explicitly set' in str(exc.value)


def test_adaptive_graph_dataset_requires_vgae():
    # Import module directly to avoid package import side-effects
    ed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'training', 'enhanced_datamodule.py'))
    # Ensure torch.utils.data.DataLoader and Dataset are available as lightweight stubs
    import types as _types
    if 'torch.utils.data' not in sys.modules:
        td = _types.ModuleType('torch.utils.data')
        class DataLoaderStub:
            def __init__(self, *a, **k):
                pass
        td.DataLoader = DataLoaderStub
        td.Dataset = list
        sys.modules['torch.utils.data'] = td
        # Ensure torch.utils exists and points to data
        utils_mod = _types.ModuleType('torch.utils')
        utils_mod.data = td
        sys.modules['torch.utils'] = utils_mod
        # Also ensure top-level torch has utils attribute
        if 'torch' in sys.modules:
            setattr(sys.modules['torch'], 'utils', utils_mod)

    # Load the AdaptiveGraphDataset class body from the source via AST to avoid import-time heavy deps
    import ast
    source = open(ed_path, 'r').read()
    module_ast = ast.parse(source)
    class_node = None
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == 'AdaptiveGraphDataset':
            class_node = node
            break
    assert class_node is not None, "AdaptiveGraphDataset not found in module"

    # Create a new module AST with just the class and a minimal imports set
    new_module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(new_module)
    code_obj = compile(new_module, filename=ed_path, mode='exec')

    # Minimal namespace with torch and numpy placeholders
    ns = {}
    class TorchStub:
        cuda = types.SimpleNamespace(is_available=lambda: False)
    ns['torch'] = TorchStub()
    ns['np'] = types.SimpleNamespace(random=types.SimpleNamespace(random=lambda n: [0.5]*n), percentile=lambda a, p: 0.5, array=lambda x: x, ndarray=object)
    ns['Dataset'] = list
    # Provide typing aliases used in class annotations
    ns['List'] = list
    ns['Dict'] = dict
    ns['Optional'] = lambda x: x
    ns['Tuple'] = tuple

    exec(code_obj, ns)
    AdaptiveGraphDataset = ns['AdaptiveGraphDataset']

    normal = [types.SimpleNamespace(x=None, edge_index=None) for _ in range(10)]
    attack = [types.SimpleNamespace(x=None, edge_index=None) for _ in range(2)]

    # Construct instance without invoking __init__ (avoids importing other submodules at init)
    ag = AdaptiveGraphDataset.__new__(AdaptiveGraphDataset)
    ag.normal_graphs = normal
    ag.attack_graphs = attack
    ag.vgae_model = None
    ag.current_epoch = 0
    ag.total_epochs = 200
    ag.difficulty_percentile = 50.0
    ag.difficulty_cache = {}

    with pytest.raises(RuntimeError) as exc:
        ag._get_difficulty_scores(normal)
    assert 'VGAE model required' in str(exc.value)
