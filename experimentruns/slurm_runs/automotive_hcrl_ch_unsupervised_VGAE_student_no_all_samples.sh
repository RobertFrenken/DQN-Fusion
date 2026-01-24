#!/bin/bash
#SBATCH --job-name=automotive_hcrl_ch_unsupervised_VGAE_student_no_al
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=PAS3209
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frenken.2@osu.edu
#SBATCH --output=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/slurm_runs/automotive_hcrl_ch_unsupervised_VGAE_student_no_all_samples_20260123_150743.log
#SBATCH --error=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/slurm_runs/automotive_hcrl_ch_unsupervised_VGAE_student_no_all_samples_20260123_150743.err
#SBATCH --chdir=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT

# Minimal KD-GAT Slurm script (simplified for portability)
set -euo pipefail

echo "Starting KD-GAT job on $(hostname)"
# GPU-optimized environment setup
module load miniconda3/24.1.2-py310 || true
source activate gnn-experiments || true
module load cuda/11.8.0 || true

# Move to working directory
WORKDIR="/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT"
if [ -d "$WORKDIR" ]; then
  cd "$WORKDIR"
else
  echo "ERROR: working directory $WORKDIR not found" >&2
  exit 1
fi

# Run training (Hydra-Zen based)
# Parse dataset robustly and select an appropriate preset for a short smoke run
IFS='_' read -r -a TOKS <<< "automotive_hcrl_ch_unsupervised_VGAE_student_no_all_samples"
# Default dataset token is TOKS[1]
if [ "${TOKS[1]}" = "hcrl" ] && ( [ "${TOKS[2]}" = "sa" ] || [ "${TOKS[2]}" = "ch" ] ); then
  DATASET="${TOKS[1]}_${TOKS[2]}"
else
  DATASET="${TOKS[1]}"
fi
if echo "automotive_hcrl_ch_unsupervised_VGAE_student_no_all_samples" | grep -qi "VGAE"; then
  PRESET="autoencoder_${DATASET}"
else
  PRESET="gat_normal_${DATASET}"
fi

# For smoke runs use the local smoke helper which can create synthetic data and run a short training
if echo "automotive_hcrl_ch_unsupervised_VGAE_student_no_all_samples" | grep -qi "VGAE"; then
  echo "Creating two small synthetic partitions and running smoke training pointing to that data"
  python - <<'PY'
from pathlib import Path
import random, csv
root = Path('experimentruns_test/big_synthetic')
root.mkdir(parents=True, exist_ok=True)
for part in ['train_01_attack_free', 'train_02_attack_free']:
    d = root / part
    d.mkdir(parents=True, exist_ok=True)
    csv_path = d / 'small.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "arbitration_id", "data_field", "attack"])
        for i in range(200):
            can_id = format(0x100 + (i % 256), 'X')
            payload = ''.join(format(random.randint(0, 255), '02X') for _ in range(8))
            writer.writerow([i, can_id, payload, 0])
print('Created synthetic data at', root)
PY
  echo "Running: python scripts/local_smoke_experiment.py --model vgae --dataset $DATASET --training autoencoder --epochs 1 --data-path experimentruns_test/big_synthetic --run --write-summary"
  python scripts/local_smoke_experiment.py --model vgae --dataset "$DATASET" --training autoencoder --epochs 1 --data-path experimentruns_test/big_synthetic --run --write-summary
else
  echo "Running: python scripts/local_smoke_experiment.py --model gat --dataset $DATASET --training normal --epochs 1 --use-synthetic-data --run --write-summary"
  python scripts/local_smoke_experiment.py --model gat --dataset "$DATASET" --training normal --epochs 1 --use-synthetic-data --run --write-summary
fi

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Job finished successfully"
  exit 0
else
  echo "Job failed with exit code: $EXIT_CODE"
  exit $EXIT_CODE
fi

# Run training with Hydra-Zen
echo "🚀 Starting training..."
python train_with_hydra_zen.py \
    config_store=automotive_hcrl_ch_unsupervised_VGAE_student_no_all_samples \
    hydra.run.dir=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns \
    training_config.epochs=1

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ JOB COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "Results saved to: /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns"
    echo ""
    exit 0
else
    echo ""
    echo "=========================================="
    echo "❌ JOB FAILED"
    echo "=========================================="
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "Check logs: /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/slurm_runs/automotive_hcrl_ch_unsupervised_VGAE_student_no_all_samples_20260123_150743.err"
    echo ""
    exit 1
fi
