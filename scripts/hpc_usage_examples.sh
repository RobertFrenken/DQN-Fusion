#!/bin/bash
# HPC Training Usage Examples - UNIFIED SINGLE FILE APPROACH

# Use optimized config
export HYDRA_FULL_ERROR=1

echo "=============================================="
echo "🎯 UNIFIED TRAINING: All modes use ONE file!"
echo "=============================================="
echo

# 1. Individual GAT training with HPC optimizations
echo "1️⃣ Individual GAT Training"
python train_with_hydra_zen.py \
    --config-path conf \
    --config-name hpc_optimized \
    --model gat \
    --dataset hcrl_sa \
    --training normal

# 2. Individual VGAE training (autoencoder mode)
echo "2️⃣ Individual VGAE Training"  
python train_with_hydra_zen.py \
    --config-path conf \
    --config-name hpc_optimized \
    --model vgae \
    --dataset hcrl_sa \
    --training autoencoder

# 3. Knowledge distillation with HPC optimizations  
echo "3️⃣ Knowledge Distillation Training"
python train_with_hydra_zen.py \
    --config-path conf \
    --config-name hpc_optimized \
    --training knowledge_distillation \
    --dataset hcrl_sa \
    --teacher_path saved_models/best_teacher_model_hcrl_sa.pth

# 4. Fusion training with HPC optimizations
echo "4️⃣ Fusion Training"
python train_with_hydra_zen.py \
    --config-path conf \
    --config-name hpc_optimized \
    --training fusion \
    --dataset hcrl_sa \
    --autoencoder_path saved_models/autoencoder_hcrl_sa.pth \
    --classifier_path saved_models/classifier_hcrl_sa.pth

# 5. Multi-GPU distributed training (any mode)
echo "5️⃣ Multi-GPU Distributed Training"
torchrun --nproc_per_node=4 train_with_hydra_zen.py \
    --config-path conf \
    --config-name hpc_optimized \
    --model gat \
    --dataset hcrl_sa \
    --training normal

echo
echo "✅ All training modes now use: train_with_hydra_zen.py"
echo "🔧 All configurations managed by: conf/hpc_optimized.yaml"
echo "⚡ Benefits: Unified workflow, consistent configs, easier HPC deployment"
