#!/usr/bin/env bash
#SBATCH --job-name=preprocess-gpu
#SBATCH --account=PAS3209
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/preprocess_gpu_%j.out
#SBATCH --error=slurm_logs/preprocess_gpu_%j.err

# GPU-accelerated preprocessing using cudf.pandas.
#
# Usage:
#   sbatch scripts/preprocess_gpu_slurm.sh <dataset> [<dataset2> ...]
#   sbatch scripts/preprocess_gpu_slurm.sh set_01
#   sbatch scripts/preprocess_gpu_slurm.sh set_01 set_02 set_03
set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT"
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

# Load CUDA module
module load cuda/12.4.1

# Activate RAPIDS conda env
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate gnn-rapids

echo "=== RAPIDS verification ==="
python -c "import cudf; import cuml; print('RAPIDS OK')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Process each dataset argument
DATASETS=("${@:-}")
if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Usage: sbatch scripts/preprocess_gpu_slurm.sh <dataset> [<dataset2> ...]"
    exit 1
fi

for ds in "${DATASETS[@]}"; do
    echo "=== Preprocessing dataset: ${ds} ==="
    python -m pipeline.cli autoencoder --model vgae --scale large --dataset "${ds}" --preprocess-only 2>&1 || {
        echo "WARNING: Preprocessing for ${ds} failed, continuing..."
    }
done

echo "=== Done ==="
