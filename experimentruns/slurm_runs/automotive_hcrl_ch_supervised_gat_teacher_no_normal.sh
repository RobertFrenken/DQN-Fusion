#!/bin/bash
#SBATCH --job-name=automotive_hcrl_ch_supervised_gat_teacher_no_norma
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=PAS3209
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frenken.2@osu.edu
#SBATCH --output=experimentruns/slurm_runs/automotive_hcrl_ch_supervised_gat_teacher_no_normal_20260123_182707.log
#SBATCH --error=experimentruns/slurm_runs/automotive_hcrl_ch_supervised_gat_teacher_no_normal_20260123_182707.err
#SBATCH --chdir=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT

# Minimal KD-GAT Slurm script (simplified for portability)
set -euo pipefail

echo "Starting KD-GAT job on $(hostname)"
module load miniconda3/24.1.2-py310 || true
source activate gnn-gpu || true
module load cuda/11.8.0 || true
# Echo python executable for quick debugging
echo "python -> $(python -c 'import sys; print(sys.executable)')" || true

# Run training (Hydra-Zen based)
echo "Running: python train_with_hydra_zen.py --preset=gat_normal_hcrl_ch"
python train_with_hydra_zen.py --preset gat_normal_hcrl_ch

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Job finished successfully"
  exit 0
else
  echo "Job failed with exit code: $EXIT_CODE"
  exit $EXIT_CODE
fi

# Run training with Hydra-Zen
echo "🚀 Starting training..."
python src/training/train_with_hydra_zen.py --preset gat_normal_hcrl_ch

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ JOB COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "Results saved to: experimentruns"
    echo ""
    exit 0
else
    echo ""
    echo "=========================================="
    echo "❌ JOB FAILED"
    echo "=========================================="
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "Check logs: experimentruns/slurm_runs/gat_normal_hcrl_ch_20260123_182707.err"
    echo ""
    exit 1
fi
