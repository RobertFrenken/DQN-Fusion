#!/bin/bash
#SBATCH --job-name=fusion-unified-set_04
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS3209
#SBATCH --output=fusion_training_%j.out
#SBATCH --error=fusion_training_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frenken.2@osu.edu

# Show job details for debugging
scontrol show job $SLURM_JOBID

echo "=== UNIFIED CAN-GRAPH TRAINING ==="
echo "Training Type: Fusion"
echo "Dataset: set_04"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================="

# GPU-optimized environment setup
module load miniconda3/24.1.2-py310
source activate gnn-gpu
module load cuda/11.8.0

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Move to working directory
cd /users/PAS2022/rf15/CAN-Graph

# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "=== GPU Configuration ==="
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "========================="

start_time=$(date +%s)

# UPDATED: Use unified training script instead of old approach
echo "=== TRAINING COMMAND (UNIFIED) ==="
python train_with_hydra_zen.py \
    --config-path conf \
    --config-name hpc_optimized \
    --training fusion \
    --dataset set_04 \
    --autoencoder_path saved_models/autoencoder_set_04.pth \
    --classifier_path saved_models/classifier_set_04.pth

# Alternative: If pre-trained models have different names
# python train_with_hydra_zen.py \
#     --config-path conf \
#     --config-name hpc_optimized \
#     --training fusion \
#     --dataset set_04 \
#     --autoencoder_path saved_models/best_teacher_model_set_04.pth \
#     --classifier_path saved_models/gat_normal_set_04.pth

echo "=================================="

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "=== Job Completed ==="
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 3600))h $(($elapsed % 3600 / 60))m)"
echo "Finished: $(date)"
echo "======================"

# Copy outputs to organized location
mkdir -p /users/PAS2022/rf15/CAN-Graph/osc_jobs/fusion_unified_set_04_$SLURM_JOB_ID
cp -r outputs/lightning_logs/* /users/PAS2022/rf15/CAN-Graph/osc_jobs/fusion_unified_set_04_$SLURM_JOB_ID/ 2>/dev/null || true
cp saved_models/*set_04* /users/PAS2022/rf15/CAN-Graph/osc_jobs/fusion_unified_set_04_$SLURM_JOB_ID/ 2>/dev/null || true

echo "✅ Fusion training completed successfully!"