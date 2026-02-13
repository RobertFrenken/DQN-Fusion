# Current State

**Date**: 2026-02-13

## What's Working

### Config Redesign (2026-02-13)

Completed full migration from flat frozen dataclass to Pydantic v2 + YAML composition:

- **`config/schema.py`**: Pydantic v2 frozen models — `PipelineConfig`, `VGAEArchitecture`, `GATArchitecture`, `DQNArchitecture`, `AuxiliaryConfig`, `TrainingConfig`, `FusionConfig`, `PreprocessingConfig`
- **`config/resolver.py`**: YAML composition — `resolve(model_type, scale, auxiliaries, **overrides)`, `list_models()`, `list_auxiliaries()`
- **YAML files**: `defaults.yaml`, `models/{vgae,gat,dqn}/{large,small}.yaml`, `auxiliaries/{none,kd_standard}.yaml`
- **Dataset catalog**: Moved from `data/datasets.yaml` to `config/datasets.yaml`
- **Path layout**: `{dataset}/{model_type}_{scale}_{stage}[_{aux}]` (was `{model_size}_{stage}[_kd]`)
- **CLI**: `--model`/`--scale`/`--auxiliaries` flags (was `--preset model,size`)
- **Cross-model path resolution**: `load_vgae(gat_cfg, ...)` correctly finds VGAE paths via `_STAGE_MODEL_TYPE` mapping
- **Legacy compat**: Old flat JSON configs load via `PipelineConfig.load()` with automatic `teacher→large` / `student→small` migration
- **Deleted**: `config/pipeline_config.py` (replaced by schema.py + resolver.py + YAML)

**Tests**: 70 passing (64 non-training + 6 smoke tests)

### Pipeline System
- Pipeline system (`pipeline/`) fully operational with Snakemake + MLflow
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- MLflow tracking at `sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db`

### Data Management Layer
- **Dataset catalog**: `config/datasets.yaml` — 6 automotive datasets registered
- **Ingestion**: `pipeline/ingest.py` — CSV→Parquet conversion
- **Project DB**: `pipeline/db.py` — SQLite with datasets/runs/metrics tables (model_type/scale/has_kd columns)
  - Write-through from cli.py + backfill via `populate()`

### Analytics Layer
- `pipeline/analytics.py`: sweep, leaderboard, compare, config_diff, dataset_summary
- Uses SQLite `json_extract()` for config queries
- FULL OUTER JOIN emulated via UNION of LEFT JOINs (SQLite compat)

## Active `src/` Files

Essential (imported by pipeline):
- `src/models/` -- vgae.py, gat.py, dqn.py (gradient checkpointing support)
- `src/preprocessing/preprocessing.py` -- graph construction
- `src/training/datamodules.py` -- load_dataset()

## What's Not Working / Incomplete

- **Old experiment checkpoints**: Pre-MLflow runs have no `config.json`
- **Bug 3.6 (research)**: OOD generalization collapse — not a code bug, requires research
- **Bug 3.7**: GAT teacher 137x larger than student due to JK cat mode + `num_fc_layers=3` — needs retrain
- **Parquet conversion**: Not yet run for existing 6 datasets (ingest module ready, need to execute)
- **Existing experiment paths**: Old paths use `{model_size}_{stage}[_kd]` format, new paths use `{model_type}_{scale}_{stage}[_{aux}]`. `populate()` handles both formats.

## Next Steps

1. **Run Parquet ingestion** for all 6 datasets: `python -m pipeline.ingest --all`
2. **Re-run pipeline** with new config system on a test dataset
3. **Investigate OOD threshold calibration** (bug 3.6)
4. **Consider GAT FC bottleneck** to address bug 3.7

## Architecture Summary

- **Entry point**: `python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`
- **Config**: Pydantic v2 frozen models + YAML composition via `resolve(model_type, scale, auxiliaries="none", **overrides)`
- **Config resolution**: `defaults.yaml` → `models/{type}/{scale}.yaml` → `auxiliaries/{aux}.yaml` → CLI overrides → Pydantic validation → frozen
- **Orchestration**: Snakemake (`snakemake -s pipeline/Snakefile --profile profiles/slurm`)
- **Tracking**: MLflow (auto-logged) + project DB (write-through from cli.py)
- **Data catalog**: `config/datasets.yaml` → `pipeline/ingest.py` → Parquet + project DB
- **Filesystem**: `experimentruns/{dataset}/{model_type}_{scale}_{stage}[_{aux}]/`
- **SLURM**: Account PAS3209, gpu partition, V100 GPUs

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **MLflow DB**: `/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db` (auto-backed up to `~/backups/`)
- **Project DB**: `data/project.db` (SQLite — datasets, runs, metrics)
- **Key packages**: PyTorch, PyG, Lightning, MLflow, Pydantic v2, PyArrow, Datasette, Pandera
