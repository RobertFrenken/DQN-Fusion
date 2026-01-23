#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-hcrl_sa}
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

echo "🚀 Running GPU smoke suite (dataset=${DATASET})"
cd "${PROJECT_ROOT}"

# Activate existing venv if present, else create a local one
if [ -f .venv/bin/activate ]; then
  echo "Using existing virtualenv .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "Creating virtualenv .venv and installing requirements (may take a minute)"
  python -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  if [ -f requirements.txt ]; then
    pip install -r requirements.txt || true
  fi
fi

# Quick GPU check
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Detected GPU:"; nvidia-smi || true
else
  echo "Warning: nvidia-smi not found. Ensure NVIDIA drivers are installed on the runner." >&2
fi

# Run the smoke-suite. This script exits with non-zero on failure and creates artifacts under experimentruns_test/
python scripts/smoke_suite_experimentruns_test.py
RC=$?

# Print summary of artifact locations
echo "Artifacts located under:"
ls -la experimentruns_test || true

exit ${RC}