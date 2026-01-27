#!/bin/bash
#SBATCH --job-name=dqn_hcrl_ch_fusion
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1
#SBATCH --account=PAS3209
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frugoli.1@osu.edu
#SBATCH --output=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/automotive/hcrl_ch/rl_fusion/dqn/teacher/no_distillation/fusion/slurm_logs/dqn_hcrl_ch_fusion_20260126_235424.out
#SBATCH --error=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/automotive/hcrl_ch/rl_fusion/dqn/teacher/no_distillation/fusion/slurm_logs/dqn_hcrl_ch_fusion_20260126_235424.err
#SBATCH --chdir=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT
#SBATCH --dependency=afterok:43968703

# CAN-Graph Training Job (Frozen Config Pattern)
# Generated: 2026-01-26 23:54:24
# Config: /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/automotive/hcrl_ch/rl_fusion/dqn/teacher/no_distillation/fusion/configs/frozen_config_20260126_235424.json
set -euo pipefail

echo "=================================================================="
echo "CAN-Graph Training Job (Frozen Config)"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Frozen Config: /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/automotive/hcrl_ch/rl_fusion/dqn/teacher/no_distillation/fusion/configs/frozen_config_20260126_235424.json"
echo "=================================================================="

# Load environment
module load miniconda3/24.1.2-py310 || true
source activate gnn-experiments || true
module load cuda/12.3.0 || module load cuda/11.8.0 || true

# Environment configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1

echo "Python: $(which python)"
echo "=================================================================="

# Run training with frozen config (no re-resolution needed)
echo "Running: python train_with_hydra_zen.py --frozen-config /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/automotive/hcrl_ch/rl_fusion/dqn/teacher/no_distillation/fusion/configs/frozen_config_20260126_235424.json"

python train_with_hydra_zen.py --frozen-config /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/automotive/hcrl_ch/rl_fusion/dqn/teacher/no_distillation/fusion/configs/frozen_config_20260126_235424.json

EXIT_CODE=$?

echo "=================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ JOB COMPLETED SUCCESSFULLY"
    echo "Results: /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/automotive/hcrl_ch/rl_fusion/dqn/teacher/no_distillation/fusion"
else
    echo "❌ JOB FAILED (exit code: $EXIT_CODE)"
    echo "Check error log: /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/automotive/hcrl_ch/rl_fusion/dqn/teacher/no_distillation/fusion/slurm_logs/dqn_hcrl_ch_fusion_20260126_235424.err"
fi
echo "End time: $(date)"
echo "=================================================================="

exit $EXIT_CODE
