#!/bin/bash
#SBATCH --job-name=curriculum_hcrl_sa
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=PAS3209
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frenken.2@osu.edu
#SBATCH --output=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/slurm_runs/hcrl_sa/curriculum_hcrl_sa_20260125_085729.log
#SBATCH --error=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/slurm_runs/hcrl_sa/curriculum_hcrl_sa_20260125_085729.err
#SBATCH --chdir=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT

# KD-GAT Slurm script
set -euo pipefail

echo "Starting KD-GAT job on $(hostname)"
module load miniconda3/24.1.2-py310 || true
source activate gnn-experiments || true
module load cuda/12.3.0 || module load cuda/11.8.0 || true

# Limit OpenMP/BLAS threads to avoid oversubscription on shared nodes
# GPU optimizations and CUDA debugging
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1  # Enable synchronous CUDA calls for better debugging
export TORCH_USE_CUDA_DSA=1   # Enable device-side assertions
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Enable faulthandler to get Python tracebacks on crashes
export PYTHONFAULTHANDLER=1
# Echo python executable for quick debugging
echo "python -> $(python -c 'import sys; print(sys.executable)')" || true

# Dataset environment exports (optional)
export CAN_DATA_PATH=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/data/automotive/hcrl_sa
export DATA_PATH=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/data/automotive/hcrl_sa


# Run training (Hydra-Zen based)
echo "Running: python train_with_hydra_zen.py --preset=curriculum_hcrl_sa --data-path /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/data/automotive/hcrl_sa"
python train_with_hydra_zen.py --preset curriculum_hcrl_sa --data-path /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/data/automotive/hcrl_sa

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ JOB COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "Results saved to: /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns"
    exit 0
else
    echo ""
    echo "=========================================="
    echo "❌ JOB FAILED"
    echo "=========================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check logs: /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/slurm_runs/hcrl_sa/curriculum_hcrl_sa_20260125_085729.err"
    exit $EXIT_CODE
fi
