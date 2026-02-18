#!/usr/bin/env bash
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

# Activate conda env
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate gnn-experiments

# Run the export + S3 sync
bash scripts/export_dashboard.sh "$@"
