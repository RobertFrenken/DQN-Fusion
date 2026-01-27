#!/bin/bash
#SBATCH --job-name=test_run_counter_batch
#SBATCH --time=00:25:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1
#SBATCH --account=PAS3209
#SBATCH --partition=gpu
#SBATCH --chdir=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT

# TEST: Run counter + Batch size config implementation
# Fast test with 10 epochs to verify:
# 1. Run counter increments models files
# 2. Batch size logging works
# 3. tuned_batch_size gets updated in frozen config

set -euo pipefail

echo "=================================================================="
echo "CAN-Graph TEST: Run Counter & Batch Size Config"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Test Config: experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder/configs/frozen_config_test.json"
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

# START GPU MONITORING IN BACKGROUND
# Logs every 2 seconds: timestamp, GPU%, Memory%, Used(MiB), Total(MiB)
echo "🔍 Starting GPU monitoring (logs every 2 seconds)..."
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 2 > gpu_monitor_${SLURM_JOB_ID}.csv &

MONITOR_PID=$!
echo "   Monitor PID: $MONITOR_PID"
echo "   Output: gpu_monitor_${SLURM_JOB_ID}.csv"
echo ""

# Run training with test frozen config
FROZEN_CONFIG="/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder/configs/frozen_config_test.json"

echo "TEST RUN 1: Testing run counter, batch size optimization, and logging"
echo "Running: python train_with_hydra_zen.py --frozen-config $FROZEN_CONFIG"
echo ""

python train_with_hydra_zen.py --frozen-config "$FROZEN_CONFIG"

EXIT_CODE=$?

# STOP GPU MONITORING
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo ""
echo "=================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST RUN 1 COMPLETED SUCCESSFULLY"
    echo "Results: experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder"
    echo ""
    echo "📊 GPU MONITORING:"
    if [ -f "gpu_monitor_${SLURM_JOB_ID}.csv" ]; then
        echo "   ✅ GPU log saved: gpu_monitor_${SLURM_JOB_ID}.csv"
        echo "   📈 To analyze, run:"
        echo "      python analyze_gpu_monitor.py gpu_monitor_${SLURM_JOB_ID}.csv"
    else
        echo "   ⚠️  GPU monitoring file not created"
    fi
    echo ""
    echo "Checking run counter and models..."
    RUN_COUNTER_FILE="experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder/run_counter.txt"
    MODELS_DIR="experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder/models"

    if [ -f "$RUN_COUNTER_FILE" ]; then
        NEXT_RUN=$(cat "$RUN_COUNTER_FILE")
        echo "🔢 Run counter: next run will be $NEXT_RUN"
    else
        echo "⚠️  Run counter file not found"
    fi

    if [ -d "$MODELS_DIR" ]; then
        echo "💾 Models saved:"
        find "$MODELS_DIR" -name "*.pth" -type f -exec basename {} \;
    else
        echo "⚠️  Models directory not found"
    fi
else
    echo "❌ TEST RUN 1 FAILED (exit code: $EXIT_CODE)"
    echo "Check error output above for details"
fi
echo "End time: $(date)"
echo "=================================================================="

exit $EXIT_CODE
