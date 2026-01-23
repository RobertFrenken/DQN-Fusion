# Lightweight global test stubs for heavy dependencies (torch, torch_geometric, pytorch_lightning)
# These allow the unit tests to import project modules without installing heavy packages
# The stubs provide minimal attributes used during import-time checks and tests.
import sys
import types

if 'torch' not in sys.modules:
    torch_stub = types.ModuleType('torch')
    # minimal torch features used by imports
    torch_stub.save = lambda *a, **k: None
    torch_stub.load = lambda *a, **k: {}
    torch_stub.Tensor = object

    # cuda submodule
    cuda_mod = types.ModuleType('torch.cuda')
    # Default to True so CUDA-only pipelines instantiate in test env; tests may override if needed
    cuda_mod.is_available = lambda: True
    cuda_mod.get_device_properties = lambda _: types.SimpleNamespace(name='FakeGPU', total_memory=8 * 1024**3, multiprocessor_count=1)
    sys.modules['torch.cuda'] = cuda_mod
    torch_stub.cuda = cuda_mod

    # device helper
    torch_stub.device = lambda s: types.SimpleNamespace(type=s)
    torch_stub.get_num_threads = lambda: 4

    # nn submodule
    nn_mod = types.ModuleType('torch.nn')
    nn_mod.Module = object
    # functional submodule
    nn_func = types.ModuleType('torch.nn.functional')
    nn_func.relu = lambda x: x
    sys.modules['torch.nn.functional'] = nn_func
    nn_mod.functional = nn_func
    sys.modules['torch.nn'] = nn_mod
    torch_stub.nn = nn_mod

    # optim submodule
    optim_mod = types.ModuleType('torch.optim')
    optim_mod.Optimizer = object
    optim_mod.Adam = lambda params, lr=0.001: object()
    optim_mod.SGD = lambda params, lr=0.01: object()
    sys.modules['torch.optim'] = optim_mod
    torch_stub.optim = optim_mod

    # utils.data submodule with Subset
    utils_mod = types.ModuleType('torch.utils')
    data_mod = types.ModuleType('torch.utils.data')
    data_mod.Subset = lambda *a, **k: list(a)
    data_mod.Dataset = object
    data_mod.DataLoader = lambda dataset, batch_size=32, shuffle=False: list(dataset)
    sys.modules['torch.utils'] = utils_mod
    sys.modules['torch.utils.data'] = data_mod
    # random_split helper for tests
    def _random_split(dataset, lengths):
        if isinstance(lengths, (list, tuple)) and len(lengths) == 2:
            a = int(lengths[0])
            return dataset[:a], dataset[a:]
        # fallback: evenly split
        mid = len(dataset) // 2
        return dataset[:mid], dataset[mid:]
    data_mod.random_split = _random_split
    utils_mod.data = data_mod
    torch_stub.utils = utils_mod

    sys.modules['torch'] = torch_stub

# Ensure any existing lightning.pytorch stub is complete
if 'lightning.pytorch' in sys.modules:
    lp_existing = sys.modules['lightning.pytorch']
    if not hasattr(lp_existing, 'LightningModule'):
        lp_existing.LightningModule = object
    if not hasattr(lp_existing, 'LightningDataModule'):
        lp_existing.LightningDataModule = object
    if not hasattr(lp_existing, 'Trainer'):
        lp_existing.Trainer = lambda *a, **k: types.SimpleNamespace()
    if not hasattr(lp_existing, 'Callback'):
        lp_existing.Callback = object


# minimal pytorch_lightning stub
if 'pytorch_lightning' not in sys.modules:
    pl = types.ModuleType('pytorch_lightning')
    pl.Trainer = object
    # callbacks submodule
    callbacks_mod = types.ModuleType('pytorch_lightning.callbacks')
    callbacks_mod.EarlyStopping = lambda *a, **k: object()
    callbacks_mod.ModelCheckpoint = lambda *a, **k: object()
    callbacks_mod.DeviceStatsMonitor = lambda *a, **k: object()
    sys.modules['pytorch_lightning.callbacks'] = callbacks_mod
    pl.callbacks = callbacks_mod
    sys.modules['pytorch_lightning'] = pl

# minimal lightning.pytorch shim (newer package layout)
if 'lightning' not in sys.modules:
    lightning = types.ModuleType('lightning')
    lp = types.ModuleType('lightning.pytorch')
    def _trainer_stub(*a, **k):
        # Return a simple object that captures kwargs for assertions in tests
        return types.SimpleNamespace(_kwargs=k, logger=k.get('logger', None))
    lp.Trainer = _trainer_stub
    lp.Callback = object
    lp.LightningDataModule = object
    lp.LightningModule = object
    # loggers
    loggers_mod = types.ModuleType('lightning.pytorch.loggers')
    loggers_mod.CSVLogger = lambda *a, **k: object()
    loggers_mod.TensorBoardLogger = lambda *a, **k: object()
    loggers_mod.MLFlowLogger = lambda *a, **k: object()
    sys.modules['lightning.pytorch.loggers'] = loggers_mod
    lp.loggers = loggers_mod
    # callbacks
    callbacks_mod2 = types.ModuleType('lightning.pytorch.callbacks')
    callbacks_mod2.EarlyStopping = lambda *a, **k: object()
    callbacks_mod2.ModelCheckpoint = lambda *a, **k: object()
    callbacks_mod2.DeviceStatsMonitor = lambda *a, **k: object()
    sys.modules['lightning.pytorch.callbacks'] = callbacks_mod2
    lp.callbacks = callbacks_mod2
    # tuner
    tuner_mod = types.ModuleType('lightning.pytorch.tuner')
    tuner_mod.Tuner = lambda trainer: types.SimpleNamespace(scale_batch_size=lambda *a, **k: None)
    sys.modules['lightning.pytorch.tuner'] = tuner_mod
    lp.tuner = tuner_mod

    sys.modules['lightning.pytorch'] = lp
    lightning.pytorch = lp
    sys.modules['lightning'] = lightning

# minimal torch_geometric stub
if 'torch_geometric' not in sys.modules:
    tg = types.ModuleType('torch_geometric')
    # data submodule
    data_mod = types.ModuleType('torch_geometric.data')
    data_mod.Dataset = object
    data_mod.Data = lambda *a, **k: None
    data_mod.Batch = lambda *a, **k: list(a)
    sys.modules['torch_geometric.data'] = data_mod
    tg.data = data_mod
    # nn submodule
    tg_nn = types.ModuleType('torch_geometric.nn')
    tg_nn.global_mean_pool = lambda x, idx: x
    tg_nn.GATConv = lambda *a, **k: object()
    tg_nn.JumpingKnowledge = lambda *a, **k: object()
    sys.modules['torch_geometric.nn'] = tg_nn
    tg.nn = tg_nn
    # loader submodule
    loader_mod = types.ModuleType('torch_geometric.loader')
    loader_mod.DataLoader = lambda dataset, batch_size=32, shuffle=False: list(dataset)
    sys.modules['torch_geometric.loader'] = loader_mod
    tg.loader = loader_mod
    sys.modules['torch_geometric'] = tg

# simple tqdm stub
if 'tqdm' not in sys.modules:
    td = types.ModuleType('tqdm')
    td.tqdm = lambda x, **k: x
    sys.modules['tqdm'] = td
