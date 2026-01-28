#!/bin/bash
#SBATCH --job-name=eval_hcrl_sa_teachers
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --account=PAS3209
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frugoli.1@osu.edu
#SBATCH --output=evaluation_results/hcrl_sa/teacher/slurm_eval_%j.out
#SBATCH --error=evaluation_results/hcrl_sa/teacher/slurm_eval_%j.err
#SBATCH --chdir=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT

set -euo pipefail

echo "=================================================================="
echo "HCRL_SA Teacher Models Evaluation"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=================================================================="

# Load environment
module load miniconda3/24.1.2-py310 || true
source activate gnn-experiments || true
module load cuda/12.3.0 || module load cuda/11.8.0 || true

# Environment configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=================================================================="

# Dataset and output configuration
DATASET="hcrl_sa"
OUTPUT_DIR="evaluation_results/${DATASET}/teacher"
mkdir -p "${OUTPUT_DIR}"

# Model paths
VGAE_MODEL="experimentruns/automotive/hcrl_sa/unsupervised/vgae/teacher/no_distillation/autoencoder/models/vgae_teacher_autoencoder_run_003.pth"
GAT_MODEL="experimentruns/automotive/hcrl_sa/supervised/gat/teacher/no_distillation/curriculum/models/gat_teacher_curriculum_run_002.pth"
DQN_MODEL="experimentruns/automotive/hcrl_sa/rl_fusion/dqn/teacher/no_distillation/fusion/models/fusion_agent_hcrl_sa.pth"

echo ""
echo "=========================================="
echo "Evaluating HCRL_SA Teacher Models"
echo "=========================================="
echo ""

# ============================================================
# 1. VGAE Teacher Evaluation
# ============================================================
echo "1. Evaluating VGAE Teacher (autoencoder)..."
echo "   Model: ${VGAE_MODEL}"
echo ""

python -m src.evaluation.evaluation \
  --dataset "${DATASET}" \
  --model-path "${VGAE_MODEL}" \
  --training-mode autoencoder \
  --mode standard \
  --batch-size 512 \
  --device cuda \
  --csv-output "${OUTPUT_DIR}/vgae_teacher_results.csv" \
  --json-output "${OUTPUT_DIR}/vgae_teacher_results.json" \
  --threshold-optimization true

echo ""
echo "✅ VGAE Teacher evaluation complete"
echo ""

# ============================================================
# 2. GAT Teacher Evaluation
# ============================================================
echo "2. Evaluating GAT Teacher (curriculum)..."
echo "   Model: ${GAT_MODEL}"
echo ""

python -m src.evaluation.evaluation \
  --dataset "${DATASET}" \
  --model-path "${GAT_MODEL}" \
  --training-mode curriculum \
  --mode standard \
  --batch-size 512 \
  --device cuda \
  --csv-output "${OUTPUT_DIR}/gat_teacher_results.csv" \
  --json-output "${OUTPUT_DIR}/gat_teacher_results.json" \
  --threshold-optimization true

echo ""
echo "✅ GAT Teacher evaluation complete"
echo ""

# ============================================================
# 3. DQN Teacher Evaluation (Fusion)
# ============================================================
echo "3. Evaluating DQN Teacher (fusion)..."
echo "   DQN Model: ${DQN_MODEL}"
echo "   VGAE Model: ${VGAE_MODEL}"
echo "   GAT Model: ${GAT_MODEL}"
echo ""

python -m src.evaluation.evaluation \
  --dataset "${DATASET}" \
  --model-path "${DQN_MODEL}" \
  --vgae-path "${VGAE_MODEL}" \
  --gat-path "${GAT_MODEL}" \
  --training-mode fusion \
  --mode standard \
  --batch-size 512 \
  --device cuda \
  --csv-output "${OUTPUT_DIR}/dqn_teacher_results.csv" \
  --json-output "${OUTPUT_DIR}/dqn_teacher_results.json" \
  --threshold-optimization true

echo ""
echo "✅ DQN Teacher evaluation complete"
echo ""

# ============================================================
# Summary
# ============================================================
echo "=========================================="
echo "All Three Teacher Models Evaluated!"
echo "=========================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Files created:"
echo "  - vgae_teacher_results.csv / .json"
echo "  - gat_teacher_results.csv / .json"
echo "  - dqn_teacher_results.csv / .json"
echo ""

echo "=================================================================="
echo "Evaluation Complete!"
echo "End time: $(date)"
echo "=================================================================="
