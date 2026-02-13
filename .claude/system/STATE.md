# Current State

**Date**: 2026-02-13

## What's Working

### Config System
- Pydantic v2 frozen models + YAML composition fully operational
- `resolve(model_type, scale, auxiliaries, **overrides)` → frozen `PipelineConfig`
- YAML files: `defaults.yaml`, `models/{vgae,gat,dqn}/{large,small}.yaml`, `auxiliaries/{none,kd_standard}.yaml`
- Dataset catalog: `config/datasets.yaml` (6 automotive datasets)
- Path layout: `{dataset}/{model_type}_{scale}_{stage}[_{aux}]`
- Legacy flat JSON loading via `PipelineConfig.load()` with automatic migration

**Tests**: 78 passing (preprocessing tests are slow, e2e tests have pre-existing config.json assertion issue)

### Pipeline System
- Pipeline system (`pipeline/`) fully operational with Snakemake + MLflow
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- MLflow tracking at `sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db`
- CLI: `python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`
- End-to-end validated: resolve() → config freeze → MLflow → graph loading → training

### Snakemake Features
- **Retries with resource scaling**: `retries: 2` on all training rules, `mem_mb=128GB * attempt`
- **Preprocessing rule**: Dedicated `preprocess` rule warms graph cache per dataset
- **Between-workflow caching**: `cache: True` on preprocess rule, `SNAKEMAKE_OUTPUT_CACHE` on scratch
- **Group jobs**: `group: "evaluation"` bundles 3 eval rules into single SLURM submissions
- **Benchmarks**: Already in use on all training + eval rules

### Data Management Layer
- **Parquet ingestion**: All 6 datasets converted (`data/parquet/automotive/{dataset}/`)
- **Project DB**: `data/project.db` — 6 datasets, 71 runs, 3915 metrics
  - Write-through from cli.py + backfill via `populate()`
- **Analytics**: sweep, leaderboard, compare, config_diff, dataset_summary

### GAT Architecture (Bug 3.7 Fixed)
- Large GAT: `fc_layers: 1` (343k params) — removed bloated 1.3M-param hidden FC layer
- Small GAT: `fc_layers: 2` (65k params) — removed one redundant hidden FC layer
- Teacher/student ratio: **5.3x** (was 16.3x)

## Active `src/` Files

Essential (imported by pipeline):
- `src/models/` — vgae.py, gat.py, dqn.py (gradient checkpointing support)
- `src/preprocessing/preprocessing.py` — graph construction
- `src/training/datamodules.py` — load_dataset()

## What's Not Working / Incomplete

- **Old experiment checkpoints**: Pre-MLflow runs have no `config.json`
- **Bug 3.6 (research)**: OOD generalization collapse — not a code bug, requires research
- **E2E tests**: Pre-existing failure — `train_autoencoder()` doesn't write config.json (CLI does)
- **Existing experiment paths**: Old paths use `{model_size}_{stage}[_kd]` format, `populate()` handles both

## Next Steps

1. **Full pipeline GPU run**: `snakemake --profile profiles/slurm --config 'datasets=["hcrl_sa"]'`
2. **Phase 3**: Model registry + dynamic fusion (see `.claude/plans/pipeline_redesign.md`)
3. **Investigate OOD threshold calibration** (bug 3.6)

## Filesystem

- **Inode usage**: 386k / 1M (cleaned from 763k on 2026-02-13)
- **Conda envs**: Only `gnn-experiments` remains (removed py310, gnn-gpu, dfl, gpu_practice)

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Snakemake cache**: `/fs/scratch/PAS1266/snakemake-cache/`
- **MLflow DB**: `/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db` (auto-backed up to `~/backups/`)
- **Project DB**: `data/project.db` (SQLite — datasets, runs, metrics)
- **Conda**: `module load miniconda3/24.1.2-py310 && conda activate gnn-experiments`
