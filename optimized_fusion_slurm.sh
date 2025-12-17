#!/bin/bash
#SBATCH --job-name=fusion-set_04-opt # job name
#SBATCH --output=%x-%j.out
#SBATCH --time=12:00:00    # Reduced time with optimizations (was 25:00:00)
#SBATCH --nodes=1          # Node count
#SBATCH --ntasks-per-node=1 # total number of tasks per node
#SBATCH --cpus-per-task=8  # Optimized: 8 cores (was 16) - sufficient for GPU job
#SBATCH --mem=48G          # Slightly reduced (was 64G) but still safe
#SBATCH --gpus-per-node=1  # Keep 1 A100 GPU
#SBATCH --account=<PAS3209>
#SBATCH --output=fusion_training_%j.out
#SBATCH --mail-type=END          # send email when job ends
#SBATCH --mail-type=FAIL          # send email if job fails
#SBATCH --mail-user=frenken.2@osu.edu

# Show job details for debugging
scontrol show job $SLURM_JOBID

# GPU-optimized environment setup
module load miniconda3/24.1.2-py310
source activate gnn-gpu
module load cuda/11.8.0

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8          # Match CPU cores
export MKL_NUM_THREADS=8          # Intel MKL threading
export NUMEXPR_NUM_THREADS=8      # NumExpr threading

# Move to working directory
cd /users/PAS2022/rf15/CAN-Graph

# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

start_time=$(date +%s)

echo "=== GPU Job Configuration ==="
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "=========================="

# Run optimized fusion training
python training/fusion_training.py root_folder='set_04' fusion_episodes=800

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "=== Job Completed ==="
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 3600))h $(($elapsed % 3600 / 60))m)"
echo "======================"