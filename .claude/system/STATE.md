# Current State

**Date**: 2026-02-07

## What's Working

### Pipeline System
- Pipeline system (`pipeline/`) fully operational with Snakemake + MLflow
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- 2-level experiment paths: `experimentruns/{dataset}/{size}_{stage}[_kd]/`
- MLflow tracking at `sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db`

### Data Management Layer (2026-02-07)
- **Dataset catalog**: `data/datasets.yaml` — 6 automotive datasets registered with metadata, CSV schema, train/test subdirs
- **Ingestion**: `pipeline/ingest.py` — CSV→Parquet conversion with hex ID parsing, validation, statistics
- **Project DB**: `pipeline/db.py` — SQLite at `data/project.db` with datasets/runs/metrics tables
  - Populated: 6 datasets, 70 runs, 3,915 metric entries
  - Queryable via `python -m pipeline.db query "SQL"` or DuckDB ATTACH
- **DuckDB** (1.4.4) installed — can query Parquet files and SQLite project DB in same SQL statement
- **MLflow enhancements**: evaluation.py now logs flattened test-scenario metrics + metrics.json artifact
- **Snakefile hooks**: `onsuccess` backs up MLflow DB to `~/backups/` and populates project DB

### Previous Fixes (2026-02-05)
- **Bug 3.1 FIXED**: `apply_dynamic_id_mapping()` no longer expands ID mapping for unseen CAN IDs. Unseen IDs map to OOV index, preventing `nn.Embedding` out-of-bounds crashes on set_01/set_03.
- **Bug 3.2 FIXED**: `_EVAL_RES` time bumped from 60 to 120 minutes for large dataset evaluation.
- **Bug 3.3 FIXED**: Cache validation in `_load_cached_data()` now accepts `GraphDataset` objects and other list-like containers, not just `list`.
- **Bug 3.4 VERIFIED**: `validate.py:40` already has `stage != "evaluation"` guard — no change needed.
- **Bug 3.5 FIXED**: NFS cache race in `_process_dataset_from_scratch()` fixed with `os.fsync()` + retry-with-backoff on rename.
- **New test**: `test_apply_dynamic_id_mapping_no_expansion` added to `tests/test_preprocessing.py`.

### Previous Fixes (2026-02-03)
- **Snakefile `--use-kd` bug fixed**: Lines 134, 149, 165 now have `--use-kd true` instead of just `--use-kd`
- **Gradient checkpointing enabled by default**: `config.py:gradient_checkpointing = True`
- **Mixed precision already default**: `config.py:precision = "16-mixed"`
- **Memory monitoring added**: `tracking.py:log_memory_metrics()` + `MemoryMonitorCallback` in trainer

### Configuration
- `gradient_checkpointing`: `True` (default) - 30-50% activation memory savings
- `precision`: `"16-mixed"` (default) - 50% memory reduction
- Both `GATWithJK` and `GraphAutoencoderNeighborhood` have checkpointing wired from config

## Active `src/` Files

Essential (imported by pipeline):
- `src/models/` -- GATWithJK, VGAE, DQN (gradient checkpointing support added)
- `src/preprocessing/preprocessing.py` -- graph construction
- `src/training/datamodules.py` -- load_dataset(), CANGraphDataModule

## Documentation Structure

```
docs/
├── user_guides/          # For user reference
│   ├── snakemake_guide.md
│   ├── mlflow_usage.md
│   └── terminal_upgrades.md
└── save/                 # Context for Claude
    ├── ARCHITECTURE.md   # Current system architecture
    └── memory_management_research.md  # Memory optimization research
```

## What's Not Working / Incomplete

- **Old experiment checkpoints**: Pre-MLflow runs have no `config.json`
- **Bug 3.6 (research)**: OOD generalization collapse — not a code bug, requires research
- **Bug 3.7**: GAT teacher 137x larger than student due to JK cat mode + `num_fc_layers=3` — needs retrain
- **Parquet conversion**: Not yet run for existing 6 datasets (ingest module ready, need to execute)

## Next Steps

1. **Run Parquet ingestion** for all 6 datasets: `python -m pipeline.ingest --all`
2. **DVC-track Parquet files**: `dvc add data/parquet/automotive/{dataset}` for each
3. **Re-run evaluation** for all 6 datasets (bugs 3.1-3.5 now fixed)
4. **Investigate OOD threshold calibration** (bug 3.6)
5. **Consider GAT FC bottleneck** to address bug 3.7
6. **Document DuckDB workflow** for undergrad onboarding

## Architecture Summary

- **Entry point**: `python -m pipeline.cli <stage> --preset <model>,<size> --dataset <name>`
- **Orchestration**: Snakemake (`snakemake -s pipeline/Snakefile --profile profiles/slurm`)
- **Tracking**: MLflow (auto-logged via `mlflow.pytorch.autolog()` + memory callback)
- **Data catalog**: `data/datasets.yaml` → `pipeline/ingest.py` → Parquet + project DB
- **Query layer**: DuckDB (reads Parquet + SQLite in one SQL) or `python -m pipeline.db query`
- **Filesystem**: Snakemake owns paths (DAG trigger), MLflow owns metadata, project DB owns structured results
- **SLURM**: Account PAS3209, gpu partition, V100 GPUs

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **MLflow DB**: `/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db` (auto-backed up to `~/backups/`)
- **Project DB**: `data/project.db` (SQLite — datasets, runs, metrics)
- **Key packages**: PyTorch, PyG, Lightning, MLflow 3.8.1, DuckDB 1.4.4, PyArrow 14.0.2, psutil
