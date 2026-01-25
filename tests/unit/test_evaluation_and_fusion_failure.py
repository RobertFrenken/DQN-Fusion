import importlib.util
import os
import sys
from pathlib import Path
import types

# Provide lightweight torch stub so module imports succeed in tests
try:
    import torch  # noqa: F401
except Exception:
    torch = types.ModuleType('torch')
    class Device:
        def __init__(self, t):
            self.type = t
        def __str__(self):
            return f"torch.device('{self.type}')"
    torch.device = Device
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _: types.SimpleNamespace(name='FakeGPU', total_memory=8 * 1024**3, multiprocessor_count=1)
    )
    torch.load = lambda *a, **k: {}
    torch.get_num_threads = lambda: 4
    # minimal nn package
    nn_mod = types.ModuleType('torch.nn')
    f_mod = types.ModuleType('torch.nn.functional')
    # utils.data stubs
    utils_mod = types.ModuleType('torch.utils')
    data_mod = types.ModuleType('torch.utils.data')
    data_mod.random_split = lambda dataset, splits: ([], [])
    data_mod.Subset = list
    class DataLoaderStub:
        def __init__(self, *a, **k):
            pass
    data_mod.DataLoader = DataLoaderStub
    utils_mod.data = data_mod
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = nn_mod
    sys.modules['torch.nn.functional'] = f_mod
    sys.modules['torch.utils'] = utils_mod
    sys.modules['torch.utils.data'] = data_mod

# Replace top-level `src` package with a lightweight stub so importing training modules doesn't execute package __init__
src_pkg = types.ModuleType('src')
src_pkg.__path__ = []
sys.modules['src'] = src_pkg
sys.modules['src.models'] = types.ModuleType('src.models')

# Stub models module to avoid heavy dependencies when importing fusion_training
models_mod = types.ModuleType('src.models.models')
class DummyModel:
    def load_state_dict(self, *a, **k):
        pass
models_mod.GATWithJK = DummyModel
models_mod.GraphAutoencoderNeighborhood = DummyModel
sys.modules['src.models.models'] = models_mod
sys.modules['src.models.gat'] = types.ModuleType('src.models.gat')

# Stub dqn module
dqn_mod = types.ModuleType('src.models.dqn')
class DummyAgent:
    def __init__(self, *a, **k):
        self.q_network = None
        self.target_network = None
        self.epsilon = 0.1
        self.alpha_values = [0.0]
        self.action_dim = 2
    def save_agent(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text('agent')
dqn_mod.EnhancedDQNFusionAgent = DummyAgent
sys.modules['src.models.dqn'] = dqn_mod

# Minimal preprocessing stubs
pre_mod = types.ModuleType('src.preprocessing.preprocessing')
def graph_creation(*a, **k):
    return []
def build_id_mapping_from_normal(*a, **k):
    return {}
pre_mod.graph_creation = graph_creation
pre_mod.build_id_mapping_from_normal = build_id_mapping_from_normal
sys.modules['src.preprocessing'] = types.ModuleType('src.preprocessing')
sys.modules['src.preprocessing.preprocessing'] = pre_mod

# Minimal training submodules
sys.modules['src.training'] = types.ModuleType('src.training')
fe_mod = types.ModuleType('src.training.fusion_extractor')
class FusionDataExtractor:
    def __init__(self, *a, **k):
        pass
fe_mod.FusionDataExtractor = FusionDataExtractor
sys.modules['src.training.fusion_extractor'] = fe_mod

# Stub lightning utils
lg_mod = types.ModuleType('src.utils.lightning_gpu_utils')
lg_mod.LightningGPUOptimizer = lambda *a, **k: None
lg_mod.LightningDataLoader = lambda *a, **k: None
sys.modules['src.utils.lightning_gpu_utils'] = lg_mod

# Stub utils modules
cache_mod = types.ModuleType('src.utils.cache_manager')
class CacheManagerStub:
    def __init__(self, *a, **k):
        pass
cache_mod.CacheManager = CacheManagerStub
sys.modules['src.utils'] = types.ModuleType('src.utils')
sys.modules['src.utils.cache_manager'] = cache_mod

# Stub config modules
cfg_mod = types.ModuleType('src.config.fusion_config')
cfg_mod.DATASET_PATHS = {}
cfg_mod.FUSION_WEIGHTS = {}
sys.modules['src.config'] = types.ModuleType('src.config')
sys.modules['src.config.fusion_config'] = cfg_mod

plot_mod = types.ModuleType('src.config.plotting_config')
plot_mod.COLOR_SCHEMES = {}
def apply_publication_style(*a, **k):
    return None
def save_publication_figure(*a, **k):
    return None
plot_mod.apply_publication_style = apply_publication_style
plot_mod.save_publication_figure = save_publication_figure
sys.modules['src.config.plotting_config'] = plot_mod

# Stub plotting utils
pu_mod = types.ModuleType('src.utils.plotting_utils')
def plot_fusion_training_progress(*a, **k):
    return None
def plot_fusion_analysis(*a, **k):
    return None
def plot_enhanced_fusion_training_progress(*a, **k):
    return None
pu_mod.plot_fusion_training_progress = plot_fusion_training_progress
pu_mod.plot_fusion_analysis = plot_fusion_analysis
pu_mod.plot_enhanced_fusion_training_progress = plot_enhanced_fusion_training_progress
sys.modules['src.utils.plotting_utils'] = pu_mod

# Stub torch_geometric to avoid heavy imports
tg_mod = types.ModuleType('torch_geometric')
loader_mod = types.ModuleType('torch_geometric.loader')
class TGDataLoader:
    def __init__(self, *a, **k):
        pass
loader_mod.DataLoader = TGDataLoader
sys.modules['torch_geometric'] = tg_mod
sys.modules['torch_geometric.loader'] = loader_mod

# Provide a simple Batch class for evaluation imports
data_mod_tg = types.ModuleType('torch_geometric.data')
class Batch:
    pass
data_mod_tg.Batch = Batch
sys.modules['torch_geometric.data'] = data_mod_tg
nn_mod_tg = types.ModuleType('torch_geometric.nn')
def global_mean_pool(*a, **k):
    return None
nn_mod_tg.global_mean_pool = global_mean_pool
sys.modules['torch_geometric.nn'] = nn_mod_tg

# Import fusion_training module directly
_ft_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'training', 'fusion_training.py'))
_spec = importlib.util.spec_from_file_location('fusion_training', _ft_path)
_ft_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ft_mod)

FusionTrainingPipeline = _ft_mod.FusionTrainingPipeline

# Import evaluation module directly
_eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'evaluation', 'evaluation.py'))
_spec2 = importlib.util.spec_from_file_location('evaluation', _eval_path)
_eval_mod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_eval_mod)

ComprehensiveEvaluationPipeline = _eval_mod.ComprehensiveEvaluationPipeline


def test_fusion_load_pretrained_models_reports_missing_paths(tmp_path):
    pipeline = FusionTrainingPipeline(num_ids=1, embedding_dim=8, device='cuda')

    # Use clearly non-existent paths
    a = tmp_path / 'missing_ae.pth'
    c = tmp_path / 'missing_clf.pth'

    try:
        pipeline.load_pretrained_models(str(a), str(c))
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError as e:
        msg = str(e)
        assert 'autoencoder' in msg and 'classifier' in msg


def test_evaluation_load_raises_when_missing(tmp_path):
    # Provide minimal model creators to avoid heavy imports
    def fake_create_student_models(num_ids, embedding_dim=8, device='cpu'):
        class A:
            def load_state_dict(self, *a, **k):
                pass
        return A(), A()
    def fake_create_teacher_models(num_ids, embedding_dim=8, device='cpu'):
        class A:
            def load_state_dict(self, *a, **k):
                pass
        return A(), A()

    # Inject stubs into module
    _eval_mod.create_student_models = fake_create_student_models
    _eval_mod.create_teacher_models = fake_create_teacher_models

    pipeline = ComprehensiveEvaluationPipeline(num_ids=1, embedding_dim=8, device=torch.device('cpu'))

    a = tmp_path / 'no_ae.pth'
    c = tmp_path / 'no_clf.pth'

    try:
        pipeline.load_student_models(str(a), str(c))
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError as e:
        assert 'autoencoder' in str(e)
        assert 'classifier' in str(e)

    try:
        pipeline.load_teacher_models(str(a), str(c))
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError as e:
        assert 'autoencoder' in str(e)
        assert 'classifier' in str(e)
