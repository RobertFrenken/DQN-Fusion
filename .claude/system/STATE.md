# Current State

**Date**: 2026-02-03

## What's Working

### Pipeline System
- Pipeline system (`pipeline/`) fully operational with Snakemake + MLflow
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- 2-level experiment paths: `experimentruns/{dataset}/{size}_{stage}[_kd]/`
- MLflow tracking at `sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db`

### Recent Fixes (2026-02-03)
- **Snakefile `--use-kd` bug fixed**: Lines 134, 149, 165 now have `--use-kd true` instead of just `--use-kd`
- **Gradient checkpointing enabled by default**: `config.py:gradient_checkpointing = True`
- **Mixed precision already default**: `config.py:precision = "16-mixed"`
- **Memory monitoring added**: `tracking.py:log_memory_metrics()` + `MemoryMonitorCallback` in trainer

### Configuration
- `gradient_checkpointing`: `True` (default) - 30-50% activation memory savings
- `precision`: `"16-mixed"` (default) - 50% memory reduction
- Both `GATWithJK` and `GraphAutoencoderNeighborhood` have checkpointing wired from config

### hcrl_sa Pipeline Status
- **Student curriculum (KD)**: Completed successfully, val_acc 0.999
- **Student autoencoder (KD)**: Previously failed due to `--use-kd` bug, now fixed
- **Teacher models**: Completed
- **Student without KD**: Completed

## Active `src/` Files

Essential (imported by pipeline):
- `src/models/` -- GATWithJK, VGAE, DQN (gradient checkpointing support added)
- `src/preprocessing/preprocessing.py` -- graph construction
- `src/training/datamodules.py` -- load_dataset(), CANGraphDataModule

Quarantined (for paper/future):
- `src/config/plotting_config.py`
- `src/utils/plotting_utils.py`

## Documentation Structure

Cleaned up 2026-02-03:
```
docs/
├── user_guides/          # For user reference
│   ├── snakemake_guide.md
│   ├── mlflow_usage.md
│   └── terminal_upgrades.md
└── save/                 # Context for Claude
    ├── ARCHITECTURE.md   # Current system architecture
    ├── CODEBASE_AUDIT.md # File categorization
    └── memory_management_research.md  # Memory optimization research
```

## What's Not Working / Incomplete

- **Large dataset jobs failing**: set_01 through set_04 and hcrl_ch have OOM issues on larger graphs
- **Old experiment checkpoints**: Pre-MLflow runs have no `config.json`

## Next Steps

1. **Re-run hcrl_sa with fixed Snakefile** to test `--use-kd true` fix
2. **Tune memory settings** for larger datasets (safety_factor, batch_size)
3. **Run remaining 5 datasets** once hcrl_sa confirmed working
4. **Evaluate and collect results** for thesis

## Architecture Summary

- **Entry point**: `python -m pipeline.cli <stage> --preset <model>,<size> --dataset <name>`
- **Orchestration**: Snakemake (`snakemake -s pipeline/Snakefile --profile profiles/slurm`)
- **Tracking**: MLflow (auto-logged via `mlflow.pytorch.autolog()` + memory callback)
- **Filesystem**: Snakemake owns paths (DAG trigger), MLflow owns metadata
- **SLURM**: Account PAS3209, gpu partition, V100 GPUs

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **MLflow DB**: `/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db`
- **Key packages**: PyTorch, PyG, Lightning, MLflow 3.8.1, psutil (for memory monitoring)
