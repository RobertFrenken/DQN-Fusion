"""
Check environment for required Python packages and tools and print a short report.
Exit code: 0 (always) - human-readable diagnostics only.
"""
import importlib
import shutil
import sys
from pathlib import Path

REQUIRED_PKG = [
    'torch',
    'lightning',
    'mlflow',
    'hydra_zen',
    'omegaconf',
    'torch_geometric'
]

TOOLS = ['sbatch', 'conda', 'uv', 'mlflow']

missing_pkgs = []
for pkg in REQUIRED_PKG:
    try:
        importlib.import_module(pkg)
    except Exception:
        missing_pkgs.append(pkg)

missing_tools = []
for t in TOOLS:
    if shutil.which(t) is None:
        missing_tools.append(t)

print('\nEnvironment check summary')
print('========================')
if missing_pkgs:
    print('\nMissing Python packages:')
    for p in missing_pkgs:
        print(f'  - {p}')
else:
    print('\nAll required Python packages appear to be installed (quick check).')

if missing_tools:
    print('\nMissing CLI tools:')
    for t in missing_tools:
        print(f'  - {t}')
    print('\nNotes:')
    print('  - If you plan to run on OSC/Slurm, ensure `conda` and `sbatch` are available on submit host.')
    print('  - If you prefer uv for local dev, install `uv` and use `just install-uv`.')
else:
    print('\nAll CLI tools present (quick check).')

# Detect CUDA availability via torch if available
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda}")
    if cuda:
        print(f"CUDA device count: {torch.cuda.device_count()}")
except Exception:
    print('\nCould not import torch to check CUDA (ignore if you use CPU only)')

print('\nRecommended next actions:')
print('  - Install missing packages with your preferred tool (uv or conda).')
print("  - Run 'just smoke-synthetic' to verify end-to-end smoke run on CPU.")

sys.exit(0)
