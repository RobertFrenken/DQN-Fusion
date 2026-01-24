#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="gnn-experiments"
PYTEST_ARGS="${@:-}" 

# Check if conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Please install conda or run tests inside the environment manually."
  exit 1
fi

# Check environment exists
if ! conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  echo "Conda environment '${ENV_NAME}' not found. Create it or change the script to use another env."
  conda env list
  exit 1
fi

echo "Running pytest inside conda env '${ENV_NAME}'..."
# Prefer running pytest with the env's python if pytest is installed there
if conda run -n "${ENV_NAME}" python -c "import pytest" >/dev/null 2>&1; then
  conda run -n "${ENV_NAME}" python -m pytest ${PYTEST_ARGS}
else
  echo "pytest is not installed inside the '${ENV_NAME}' environment."
  echo "Please install test tooling into the environment, for example:"
  echo "  conda install -n ${ENV_NAME} -c conda-forge pytest"
  echo "Or, to use pip inside the env:"
  echo "  conda run -n ${ENV_NAME} python -m pip install pytest"
  echo "After installing pytest, re-run this script. Aborting."
  exit 2
fi
