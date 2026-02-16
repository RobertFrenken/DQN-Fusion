#!/bin/bash
#SBATCH --account=PAS3209
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=test-cache
#SBATCH --output=slurm_logs/%j-test-cache.out
#SBATCH --error=slurm_logs/%j-test-cache.err

# Build test graph caches for all datasets.
# Usage: sbatch scripts/build_test_cache.sh
#   or:  sbatch scripts/build_test_cache.sh set_02 set_03

set -euo pipefail
cd /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT

module load miniconda3/24.1.2-py310
conda activate gnn-experiments

if [ $# -gt 0 ]; then
    DATASETS="$@"
else
    DATASETS="set_01 set_02 set_03 set_04"
fi

python -c "
from src.training.datamodules import load_test_scenarios
from pathlib import Path
import logging, sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

datasets = '$DATASETS'.split()
for ds in datasets:
    print(f'=== Building test cache for {ds} ===', flush=True)
    scenarios = load_test_scenarios(ds, Path(f'data/automotive/{ds}'), Path(f'data/cache/{ds}'))
    for name, graphs in scenarios.items():
        print(f'  {name}: {len(graphs)} graphs', flush=True)
    print(f'=== Done: {ds} ({len(scenarios)} scenarios) ===', flush=True)
print('All done')
"
