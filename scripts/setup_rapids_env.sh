#!/usr/bin/env bash
# Create a conda environment with RAPIDS 24.12 + PyTorch + PyG for GPU-accelerated
# preprocessing and dimensionality reduction.
#
# Usage:
#   bash scripts/setup_rapids_env.sh
#
# The existing gnn-experiments env is untouched and remains the CPU fallback.
set -euo pipefail

ENV_NAME="gnn-rapids"
CUDA_VERSION="12.4"
RAPIDS_VERSION="24.12"
PYTHON_VERSION="3.11"

echo "=== Creating conda env: ${ENV_NAME} ==="

# Remove existing env if present
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true

conda create -n "${ENV_NAME}" -y \
    python="${PYTHON_VERSION}" \
    -c rapidsai -c conda-forge -c nvidia \
    cudf="${RAPIDS_VERSION}" \
    cuml="${RAPIDS_VERSION}" \
    cupy \
    cuda-version="${CUDA_VERSION}"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch + PyG ==="
pip install torch>=2.0.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch-geometric>=2.6.1
pip install pytorch-lightning>=2.4.0

echo "=== Installing remaining dependencies ==="
pip install \
    numpy>=1.26.4 \
    pandas>=2.2.3 \
    pyarrow>=14.0.0 \
    wandb>=0.18 \
    pyyaml>=6.0 \
    psutil>=6.0.0 \
    scikit-learn>=1.5.0 \
    pytest>=7.0.0 \
    httpx>=0.27

echo "=== Verifying RAPIDS ==="
python -c "import cuml; import cudf; print('RAPIDS OK')"

echo "=== Done. Activate with: conda activate ${ENV_NAME} ==="
