# Lightweight global test stubs for heavy dependencies (torch, torch_geometric, pytorch_lightning)
# These allow the unit tests to import project modules without installing heavy packages
# The stubs provide minimal attributes used during import-time checks and tests.
import sys
import types

# Require a real torch installation for tests that exercise models and tensor ops
try:
    import torch as _torch  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "The test suite requires a real installation of 'torch'.\n"
        "Please activate the 'gnn-experiments' conda environment and run tests using "
        "`scripts/run_tests_in_conda.sh` which will run pytest inside that environment."
    ) from e

# The rest of the test stubs (lightning, torch_geometric, etc.) remain to help with imports
# but we intentionally avoid stubbing 'torch' to prevent masking the real package.

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


# Require a real pytorch-lightning or lightning.pytorch installation for tests
try:
    import importlib
    _lp_spec = importlib.util.find_spec('lightning.pytorch') or importlib.util.find_spec('pytorch_lightning')
except Exception:
    _lp_spec = None

if _lp_spec is None:
    raise RuntimeError(
        "Tests require 'pytorch_lightning' (or 'lightning.pytorch') to be installed in the 'gnn-experiments' env.\n"
        "Install it with: conda install -n gnn-experiments -c conda-forge pytorch-lightning"
    )
# If present, the real package will be imported normally by test code.

# Require a real torch_geometric installation for tests that use GNN layers
try:
    import importlib
    _tg_spec = importlib.util.find_spec('torch_geometric')
except Exception:
    _tg_spec = None

if _tg_spec is None:
    raise RuntimeError(
        "Tests require 'torch_geometric' to be installed in the 'gnn-experiments' env.\n"
        "Install it with: conda install -n gnn-experiments -c conda-forge pyg"
    )
# If present, the real package will be imported normally by test code.

# Require tqdm for nicer test output (optional but recommended)
try:
    import importlib
    _tqdm_spec = importlib.util.find_spec('tqdm')
except Exception:
    _tqdm_spec = None

if _tqdm_spec is None:
    # Not critical — tests will still run, but log a helpful message when running under the conda env
    import warnings
    warnings.warn("Optional dependency 'tqdm' not found in the environment. Install with 'conda install -n gnn-experiments -c conda-forge tqdm' for nicer test output.")
