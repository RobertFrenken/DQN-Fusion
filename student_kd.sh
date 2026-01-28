#!/bin/bash
# Batch size optimization is NOW ENABLED by default
# Use --no-optimize-batch-size to disable if needed
#
# Running 6 pipeline combinations with SLURM dependencies:
# - 6 datasets: hcrl_sa, hcrl_ch, set_01, set_02, set_03, set_04
# - Each student KD pipeline waits for corresponding teacher DQN to complete
# Total: 6 × 1 = 6 runs
#
# Teacher DQN job IDs (calculated from run_pipeline.sh submission pattern):
# Based on job IDs 43981560-43981611 (36 jobs = 12 pipelines × 3 stages each)
# Pattern: for each dataset, [teacher: VGAE/GAT/DQN, student: VGAE/GAT/DQN]
# Teacher DQN is the 3rd job in each teacher triplet

DATASETS=("hcrl_sa" "hcrl_ch" "set_01" "set_02" "set_03" "set_04")

# Teacher DQN job IDs for each dataset (in same order as DATASETS array)
# These are the jobs that must complete before student KD can start
TEACHER_DQN_JOBS=(
  "43982250"  # hcrl_sa
  "43982254"  # hcrl_ch
  "43982259"  # set_01
  "43982263"  # set_02
  "43982267"  # set_03
  "43982271"  # set_04
)

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    teacher_job="${TEACHER_DQN_JOBS[$i]}"
    echo "=========================================="
    echo "Submitting: dataset=${dataset}, model_size=Student"
    echo "Waiting for teacher DQN job ${teacher_job} to complete"
    echo "=========================================="

  ./can-train pipeline \
    --modality automotive \
    --model vgae,gat,dqn \
    --learning-type unsupervised,supervised,rl_fusion \
    --training-strategy autoencoder,curriculum,fusion \
    --dataset "${dataset}" \
    --model-size student \
    --distillation with-kd \
    --dependency "${teacher_job}" \
    --submit

    echo ""
    echo "Completed submission for ${dataset} (dependent on job ${teacher_job})"
    echo ""
done

echo "=========================================="
echo "All 6 pipeline runs submitted!"
echo "=========================================="