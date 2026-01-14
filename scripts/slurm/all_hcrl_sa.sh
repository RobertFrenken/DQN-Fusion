#!/bin/bash
#SBATCH --job-name=can-pipeline-hcrl_sa
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --output=logs/pipeline_hcrl_sa_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# Environment setup
module load miniconda3/24.1.2-py310
source activate gnn-gpu
module load cuda/11.8.0

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $SLURM_SUBMIT_DIR

echo "=== Full CAN-Graph Pipeline: hcrl_sa ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Estimated time: 10-12 hours"

# Step 1: Train teacher models
echo "Step 1/4: Training teacher models..."
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal --config-override training.precision=16-mixed
python train_with_hydra_zen.py --training autoencoder --dataset hcrl_sa --config-override training.precision=16-mixed

# Step 2: Knowledge distillation  
echo "Step 2/4: Knowledge distillation..."
python train_with_hydra_zen.py \
    --training knowledge_distillation \
    --dataset hcrl_sa \
    --teacher_path saved_models/best_teacher_model_hcrl_sa.pth \
    --student_scale 0.5 \
    --config-override training.precision=16-mixed

# Step 3: Fusion training
echo "Step 3/4: Fusion training..."
python train_fusion_lightning.py \
    --dataset hcrl_sa \
    --max-epochs 100 \
    --precision 16-mixed

# Step 4: Evaluation
echo "Step 4/4: Model evaluation..."
python scripts/evaluate_models.py --dataset hcrl_sa --all-models

echo "=== Full Pipeline Complete ==="
