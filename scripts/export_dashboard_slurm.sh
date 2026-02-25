#!/usr/bin/env bash
# Dashboard export SLURM script.
#
# Usage:
#   sbatch scripts/export_dashboard_slurm.sh
#
# Note: All exports are now lightweight (~2s). SLURM submission is only
# needed if you want S3 sync from a compute node or want it logged.

# --- SBATCH directives ---
#SBATCH --job-name=dashboard-export
#SBATCH --account=PAS3209
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/dashboard_export_%j.out
#SBATCH --error=slurm_logs/dashboard_export_%j.err

set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

source ~/KD-GAT/.venv/bin/activate

bash scripts/export_dashboard.sh "$@"
