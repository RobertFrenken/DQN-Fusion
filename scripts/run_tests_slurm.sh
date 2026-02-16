#!/usr/bin/env bash
# Submit pytest to a SLURM compute node (cpu partition, no GPU needed).
#
# Usage:
#   bash scripts/run_tests_slurm.sh                       # all slurm-marked tests
#   bash scripts/run_tests_slurm.sh -k "test_full_pipeline"  # specific test
#   bash scripts/run_tests_slurm.sh -m slurm              # only slurm-marked
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$PROJECT_DIR/slurm_logs"

sbatch --account=PAS3209 --partition=cpu \
  --time=60 --mem=32G --cpus-per-task=8 \
  --job-name=pytest --output="$PROJECT_DIR/slurm_logs/%j-pytest.out" \
  --error="$PROJECT_DIR/slurm_logs/%j-pytest.err" \
  --wrap="cd $PROJECT_DIR && PYTHONPATH=$PROJECT_DIR python -m pytest tests/ -v --run-slurm $*"

echo "Submitted pytest job. Check slurm_logs/ for output."
