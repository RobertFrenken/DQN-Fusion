# Current State

**Date**: 2026-02-17

## What's Working

### Config System
- Pydantic v2 frozen models + YAML composition fully operational
- `resolve(model_type, scale, auxiliaries, **overrides)` → frozen `PipelineConfig`
- YAML files: `defaults.yaml`, `models/{vgae,gat,dqn}/{large,small}.yaml`, `auxiliaries/{none,kd_standard}.yaml`
- Dataset catalog: `config/datasets.yaml` (6 automotive datasets)
- Path layout: `{dataset}/{model_type}_{scale}_{stage}[_{aux}]`
- Legacy flat JSON loading via `PipelineConfig.load()` with automatic migration

### Pipeline System
- Pipeline system (`pipeline/`) fully operational with Prefect + W&B
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- CLI: `python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`
- Flow: `python -m pipeline.cli flow --dataset <ds> [--scale <scale>]`
- End-to-end validated: resolve() → config freeze → W&B init → graph loading → training → lakehouse sync
- VGAE validation run completed: 300 epochs, val_loss 0.157, checkpoint + metrics + W&B offline run

### Platform (fully migrated)
- **W&B**: `wandb.init()`/`wandb.finish()` in CLI, WandbLogger in Lightning trainer. Offline mode on compute nodes. Project `kd-gat`.
- **Prefect**: `train_pipeline()` and `eval_pipeline()` flows with dask-jobqueue SLURMCluster. Installed and configured.
- **S3 Lakehouse**: Fire-and-forget sync via `pipeline/lakehouse.py` to `s3://kd-gat/lakehouse/runs/`. boto3/awscli installed. Tested and working.
- **DVC**: S3 remote configured alongside local scratch remote. Tested and working (21 files pushed to `s3://kd-gat/dvc/`).

### Dashboard (GitHub Pages)
- **Live**: https://robertfrenken.github.io/DQN-Fusion/
- **Stack**: Static JSON + D3.js v7 (ES modules), deployed from `docs/` on `main`
- **Architecture**: Config-driven component system (BaseChart → 8 chart types, Registry, PanelManager)
- **Data source**: `pipeline/export.py` scans `experimentruns/` filesystem (no DB dependency)
- **Auto-export**: `scripts/export_dashboard.sh` runs export + commit + push + DVC push
- **All panels populated**: 72 embedding projections, 18 DQN policy, 18 attention, 54 ROC curves, 18 recon errors, 6 CKA matrices

### Test Infrastructure
- **101 tests passing** across 8 test files (parallel SLURM submission via `scripts/run_tests_parallel.sh`)
- `@pytest.mark.slurm` marker auto-skips heavy tests on login nodes
- `scripts/run_tests_slurm.sh` submits pytest to SLURM compute nodes (sequential fallback)
- Test coverage: config, preprocessing, pipeline integration, layer boundaries, GAT return_embedding, DQN fusion reward, sweep generation, CLI archive/restore, FastAPI serve

**Tests**: 101 passing (layer boundary + pipeline integration + new feature tests).

## Recently Completed

- **Validation & cleanup** (2026-02-17):
  - 101 tests passing across 8 files (parallel SLURM submission)
  - VGAE validation run: 300 epochs, val_loss 0.157
  - Dashboard export: all 26 panels populated (72 embeddings, 18 DQN, 18 attention, 54 ROC, 18 recon, 6 CKA)
  - Fixed MemoryMonitorCallback._record_epoch bug
  - Fixed export.py UMAP: PCA pre-reduction + pre-sampling before reduction
  - Cleaned up legacy Snakemake/SQLite references from docstrings and comments
  - Deleted obsolete files: snakemake_guide.md, RESEARCH_PLATFORM_ARCHITECTURE.md, research_osc_software.md
- **Platform migration** (2026-02-17): 4-phase migration from Snakemake/SQLite to W&B/Prefect/S3:
  - Phase 1: W&B instrumentation (WandbLogger, wandb.init/finish lifecycle)
  - Phase 2: Prefect orchestration (train_flow, eval_flow, SLURMCluster)
  - Phase 3: S3 lakehouse sync + DVC remote
  - Phase 4: Deleted ~3,100 lines (Snakefile, db.py, analytics.py, ingest.py, state_sync.py, migrate_paths.py, schemas.py, 9 Snakemake rule files, SLURM profile). Refactored export.py to filesystem scanning.
  - All on branch `platform/wandb`, pushed to remote.

## What's Not Working / Incomplete

- **OOD generalization collapse** (Bug 3.6): Research question, not a code bug
- **W&B online sync**: Compute nodes use offline mode; offline runs need manual `wandb sync`

## Next Steps

1. **Merge to main**: PR from `platform/wandb` → `main`
2. **Sync W&B offline runs**: `wandb sync wandb/offline-run-*`
3. **Push dashboard to GitHub Pages**: `bash scripts/export_dashboard.sh`
4. **Investigate OOD threshold calibration** (Bug 3.6)

## Filesystem

- **Inode usage**: ~390k / 20M
- **Conda envs**: gnn-experiments

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Prefect home**: `/fs/scratch/PAS1266/.prefect/` (GPFS — avoid NFS for Prefect)
- **W&B**: Project `kd-gat` (offline on compute nodes, sync later)
- **Dashboard**: `docs/dashboard/` (GitHub Pages — static JSON + D3.js)
- **Conda**: Auto-loaded via `~/.bashrc` (`module load miniconda3` + `conda activate gnn-experiments`)
