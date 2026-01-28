#!/bin/bash
# Batch size optimization is NOW ENABLED by default
# Use --no-optimize-batch-size to disable if needed
#
# Running 6 pipeline combinations:
# - 6 datasets: hcrl_sa, hcrl_ch, set_01, set_02, set_03, set_04
# Total: 6 × 1 = 6 runs

DATASETS=("hcrl_sa" "hcrl_ch" "set_01" "set_02" "set_03" "set_04")

for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Submitting: dataset=${dataset}, model_size=Student"
    echo "=========================================="

  ./can-train pipeline \
    --modality automotive \
    --model vgae,gat,dqn \
    --learning-type unsupervised,supervised,rl_fusion \
    --training-strategy autoencoder,curriculum,fusion \
    --dataset "${dataset}" \
    --model-size teacher \
    --distillation no-kd \
    --submit

    echo ""
    echo "Completed submission for ${dataset}"
    echo ""
done

echo "=========================================="
echo "All 6 pipeline runs submitted!"
echo "=========================================="