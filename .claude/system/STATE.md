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

**Tests**: 78 passing (preprocessing tests are slow, e2e tests have pre-existing config.json assertion issue)

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
- **Project DB**: `data/project.db` — 6 datasets, 140 runs, 7830 metrics
  - Write-through from cli.py + backfill via `populate()`
- **Analytics**: sweep, leaderboard, compare, config_diff, dataset_summary

### Dashboard (GitHub Pages)
- **Live**: https://robertfrenken.github.io/DQN-Fusion/
- **Stack**: Static JSON + D3.js, deployed from `docs/` on `main`
- **Data**: 540 leaderboard entries, 140 runs, 6 datasets, 30 per-run metrics
- **Auto-export**: `scripts/export_dashboard.sh` runs in Snakemake `onsuccess`
- **Manual export**: `bash scripts/export_dashboard.sh` (or `--dry-run`, `--no-push`)

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

## Recently Completed

- **Dashboard deployment** (2026-02-15): GitHub Pages dashboard live at https://robertfrenken.github.io/DQN-Fusion/. Auto-export via Snakemake `onsuccess`. Fixed orphaned `data/automotive` submodule ref that was breaking Pages builds and causing constant git noise.
- **Legacy path migration** (2026-02-14): All 70 `teacher_*/student_*` dirs renamed to `{model_type}_{scale}_{stage}[_{aux}]` format across 6 datasets. DB run_ids updated. 18 config.json `teacher_path` values rewritten. No legacy dirs remain.

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
- **Project DB**: `data/project.db` (SQLite — datasets, runs, metrics)
- **Dashboard**: https://robertfrenken.github.io/DQN-Fusion/ (GitHub Pages from `docs/`)
- **Conda**: `module load miniconda3/24.1.2-py310 && conda activate gnn-experiments`
