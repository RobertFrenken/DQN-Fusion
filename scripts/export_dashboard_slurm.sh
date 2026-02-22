#!/usr/bin/env bash
# Dashboard export SLURM script.
#
# Usage:
#   sbatch scripts/export_dashboard_slurm.sh              # CPU partition (default)
#   sbatch scripts/export_dashboard_slurm.sh --gpu         # GPU partition (RAPIDS)
#   sbatch scripts/export_dashboard_slurm.sh --gpu --only-heavy
#
# When --gpu is passed, this script self-submits with GPU SBATCH directives.
# Otherwise it uses the CPU partition as before.

# --- Default SBATCH directives (CPU) ---
#SBATCH --job-name=dashboard-export
#SBATCH --account=PAS3209
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/dashboard_export_%j.out
#SBATCH --error=slurm_logs/dashboard_export_%j.err

set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT"
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

# Check if --gpu was requested (before SLURM allocation, i.e. on login node)
if [[ "${1:-}" == "--gpu" ]] && [[ -z "${SLURM_JOB_ID:-}" ]]; then
    shift
    echo "Re-submitting to GPU partition with RAPIDS environment..."
    sbatch \
        --job-name=dashboard-export-gpu \
        --account=PAS3209 \
        --partition=gpu \
        --gres=gpu:1 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=16G \
        --time=01:00:00 \
        --output=slurm_logs/dashboard_export_gpu_%j.out \
        --error=slurm_logs/dashboard_export_gpu_%j.err \
        --export=ALL,KD_GAT_USE_RAPIDS=1 \
        "$0" "$@"
    exit 0
fi

# Activate environment
if [[ "${KD_GAT_USE_RAPIDS:-}" == "1" ]]; then
    module load cuda/12.6.2
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate gnn-rapids
    echo "=== RAPIDS environment active ==="
    python -c "import cuml; import cudf; print('RAPIDS OK')"
else
    source ~/CAN-Graph-Test/KD-GAT/.venv/bin/activate
fi

# Run the export + S3 sync
bash scripts/export_dashboard.sh "$@"
