# Current State

**Date**: 2026-02-15

## What's Working

### Config System
- Pydantic v2 frozen models + YAML composition fully operational
- `resolve(model_type, scale, auxiliaries, **overrides)` → frozen `PipelineConfig`
- YAML files: `defaults.yaml`, `models/{vgae,gat,dqn}/{large,small}.yaml`, `auxiliaries/{none,kd_standard}.yaml`
- Dataset catalog: `config/datasets.yaml` (6 automotive datasets)
- Path layout: `{dataset}/{model_type}_{scale}_{stage}[_{aux}]`
- Legacy flat JSON loading via `PipelineConfig.load()` with automatic migration

**Tests**: 105+ passing, 4 pre-existing FK failures in `TestWriteThroughDB` (slurm-marked tests auto-skip on login nodes). Always run via SLURM `cpu` partition for speed.

### Pipeline System
- Pipeline system (`pipeline/`) fully operational with Snakemake + project DB
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- CLI: `python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`
- End-to-end validated: resolve() → config freeze → DB recording → graph loading → training
- Snakemake `onsuccess` auto-exports dashboard data after pipeline runs

### Snakemake Features
- **Retries with resource scaling**: `retries: 2` on all training rules, `mem_mb=128GB * attempt`
- **Preprocessing rule**: Dedicated `preprocess` rule warms graph cache per dataset
- **Between-workflow caching**: `cache: True` on preprocess rule, `SNAKEMAKE_OUTPUT_CACHE` on scratch
- **Group jobs**: `group: "evaluation"` bundles 3 eval rules into single SLURM submissions
- **Benchmarks**: Already in use on all training + eval rules

### Data Management Layer
- **Parquet ingestion**: All 6 datasets converted (`data/parquet/automotive/{dataset}/`)
- **Project DB**: `data/project.db` — 6 datasets, 70 runs (legacy naming migrated), 3915 metrics
  - Write-through from cli.py + backfill via `populate()`
  - WAL mode + 15s busy timeout + `_retry_on_locked` decorator for concurrent SLURM jobs
  - Indices on metrics and epoch_metrics tables
  - `populate()` runs: `_migrate_legacy_runs()`, `_backfill_timestamps()`, `_backfill_teacher_run()`
- **Analytics**: sweep, leaderboard, compare, config_diff, dataset_summary

### Dashboard (GitHub Pages)
- **Live**: https://robertfrenken.github.io/DQN-Fusion/
- **Stack**: Static JSON + D3.js, deployed from `docs/` on `main`
- **Working tabs**: Leaderboard (270 entries), Dataset Comparison, KD Transfer (108 pairs), Run Timeline (70 timestamped runs)
- **Training Curves**: Empty (epoch_metrics table has 0 rows — will auto-populate from next training runs)
- **Auto-export**: `scripts/export_dashboard.sh` runs in Snakemake `onsuccess`
- **Export validation**: `export_all()` logs warnings for empty exports

### Test Infrastructure
- `@pytest.mark.slurm` marker auto-skips heavy tests on login nodes
- `scripts/run_tests_slurm.sh` submits pytest to SLURM compute nodes
- `--run-slurm` pytest flag or `SLURM_JOB_ID` env var enables slurm-marked tests

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
- **Training curves tab**: Empty until next round of training runs populates `epoch_metrics` table
- **TestWriteThroughDB**: 4 pre-existing test failures — FK constraint on `runs.dataset → datasets(name)` fires because tests insert runs without first inserting a dataset row. Needs fixture fix.
- **`scripts/run_tests_slurm.sh`**: Uses `--partition=serial` which no longer exists — needs update to `cpu`

## Recently Completed

- **Fragility fixes** (2026-02-15): Implemented 10-item fragility fix plan across 3 phases:
  - **Phase 1 (Prerequisites)**: Schema versioning (`SCHEMA_VERSION` constant, `schema_version` column in runs table, versioned JSON export envelope). DB write retry (`_retry_on_locked` decorator with exponential backoff, `busy_timeout` 5s→15s).
  - **Phase 2 (Tier 1)**: Overwrite protection (archive-on-collision before re-running same config). Cache staleness detection (`PREPROCESSING_VERSION` check + param validation on cache load). Export/dashboard schema coupling (pre-flight DB validation, `metric_catalog.json` export, dynamic metric dropdowns in JS, try/catch + defensive `??` in all chart functions). Push recovery (pull-rebase + 3x retry loop in `export_dashboard.sh`, tolerate failure in Snakefile `onsuccess`).
  - **Phase 3 (Tier 2)**: Strict YAML mode (`_warn_unused_keys` on resolver). Dataset catalog validation (`config/catalog.py` with Pydantic `DatasetEntry` model). Dynamic metrics in analytics (`_get_core_metrics()` replaces hardcoded list). Aux type validation (`Literal["kd"]` on `AuxiliaryConfig.type`).
  - New file: `config/catalog.py`. Modified 16 files total.
- **Dashboard data pipeline fix** (2026-02-15): Fixed 3 broken dashboard tabs. KD transfer query rewritten to match student↔teacher by convention (108 pairs, was 0). Timestamps backfilled from filesystem mtime (70/70 runs). Teacher_run propagated to 8 KD eval runs. Legacy naming migrated (0 `model_type='unknown'` remaining). SQLite hardened with WAL mode, busy timeout, indices. Export validation added.
- **SLURM test dispatch** (2026-02-15): Added `@pytest.mark.slurm` to E2E and smoke tests (auto-skip on login nodes). New `scripts/run_tests_slurm.sh` for compute node submission.
- **Dashboard deployment** (2026-02-15): GitHub Pages dashboard live at https://robertfrenken.github.io/DQN-Fusion/. Auto-export via Snakemake `onsuccess`. Fixed orphaned `data/automotive` submodule ref.
- **Legacy path migration** (2026-02-14): All 70 `teacher_*/student_*` dirs renamed to `{model_type}_{scale}_{stage}[_{aux}]` format across 6 datasets.

## Next Steps

1. **Fix `scripts/run_tests_slurm.sh`**: Update `--partition=serial` → `--partition=cpu` (serial no longer exists on Pitzer)
2. **Fix TestWriteThroughDB**: Add dataset fixture to DB tests so FK constraint passes
3. **Benchmark SLURM test speed**: Compare login node vs `cpu` partition (8 CPUs, 16GB) to quantify speedup
4. **Full pipeline GPU run**: `snakemake --profile profiles/slurm --config 'datasets=["hcrl_sa"]'`
5. **Phase 3**: Model registry + dynamic fusion (see `.claude/plans/pipeline_redesign.md`)
6. **Investigate OOD threshold calibration** (bug 3.6)

## Filesystem

- **Inode usage**: 386k / 1M (cleaned from 763k on 2026-02-13)
- **Conda envs**: Only `gnn-experiments` remains (removed py310, gnn-gpu, dfl, gpu_practice)

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Snakemake cache**: `/fs/scratch/PAS1266/snakemake-cache/`
- **Project DB**: `data/project.db` (SQLite WAL — datasets, runs, metrics, epoch_metrics)
- **Dashboard**: https://robertfrenken.github.io/DQN-Fusion/ (GitHub Pages from `docs/`)
- **Conda**: `module load miniconda3/24.1.2-py310 && conda activate gnn-experiments`
