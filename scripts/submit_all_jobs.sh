#!/usr/bin/env bash
set -euo pipefail

# Example script to generate and submit jobs for all datasets for three workflows:
# 1) VGAE autoencoder
# 2) GAT curriculum
# 3) DQN fusion

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python - <<'PY'
from osc_job_manager import OSCJobManager
m = OSCJobManager()
# Datasets to run
datasets = m.training_configurations['datasets']['automotive']

print('Submitting VGAE autoencoder jobs for datasets:', datasets)
m.submit_individual_jobs(datasets=datasets, training_types=['autoencoder'])

print('Submitting GAT curriculum jobs for datasets:', datasets)
m.submit_individual_jobs(datasets=datasets, training_types=['curriculum'])

print('Submitting DQN fusion jobs for datasets:', datasets)
# Fusion jobs assume pre-trained artifacts exist in canonical locations. Optionally pass dependency manifests here.
m.submit_individual_jobs(datasets=datasets, training_types=['fusion'])
PY

echo "All jobs submitted (or scripts generated) - check slurm_jobs/ and slurm_logs/ for output."