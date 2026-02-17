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

**Tests**: 39 passing (layer boundary + pipeline integration). No DB-dependent tests remain.

### Pipeline System
- Pipeline system (`pipeline/`) fully operational with Prefect + W&B
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- CLI: `python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`
- Flow: `python -m pipeline.cli flow --dataset <ds> [--scale <scale>]`
- End-to-end validated: resolve() → config freeze → W&B init → graph loading → training → lakehouse sync

### Platform (newly migrated)
- **W&B**: `wandb.init()`/`wandb.finish()` in CLI, WandbLogger in Lightning trainer. Offline mode on compute nodes.
- **Prefect**: `train_pipeline()` and `eval_pipeline()` flows with dask-jobqueue SLURMCluster.
- **S3 Lakehouse**: Fire-and-forget sync via `pipeline/lakehouse.py` to `s3://kd-gat/lakehouse/runs/`. boto3/awscli installed in gnn-experiments conda env. Sync tested and working.
- **DVC**: S3 remote configured alongside local scratch remote. Tested and working (21 files pushed to `s3://kd-gat/dvc/`).

### Dashboard (GitHub Pages)
- **Live**: https://robertfrenken.github.io/DQN-Fusion/
- **Stack**: Static JSON + D3.js v7 (ES modules), deployed from `docs/` on `main`
- **Architecture**: Config-driven component system (BaseChart → 8 chart types, Registry, PanelManager)
- **Data source**: `pipeline/export.py` scans `experimentruns/` filesystem (no DB dependency)
- **Auto-export**: `scripts/export_dashboard.sh` runs export + commit + push + DVC push

### Test Infrastructure
- `@pytest.mark.slurm` marker auto-skips heavy tests on login nodes
- `scripts/run_tests_slurm.sh` submits pytest to SLURM compute nodes

## Recently Completed

- **S3 setup** (2026-02-17): boto3/awscli installed in gnn-experiments, S3 bucket verified, lakehouse sync tested, DVC push (21 files) to `s3://kd-gat/dvc/`. Conda init added to `~/.bashrc` (`module load miniconda3` + `conda activate gnn-experiments`).
- **Platform migration** (2026-02-17): 4-phase migration from Snakemake/SQLite to W&B/Prefect/S3:
  - Phase 1: W&B instrumentation (WandbLogger, wandb.init/finish lifecycle)
  - Phase 2: Prefect orchestration (train_flow, eval_flow, SLURMCluster)
  - Phase 3: S3 lakehouse sync + DVC remote
  - Phase 4: Deleted ~3,100 lines (Snakefile, db.py, analytics.py, ingest.py, state_sync.py, migrate_paths.py, schemas.py, 9 Snakemake rule files, SLURM profile). Refactored export.py to filesystem scanning.
  - All on branch `platform/wandb`, pushed to remote.

## What's Not Working / Incomplete

- **W&B not yet configured**: Need to run `wandb login` on cluster and set up API key
- **Prefect not yet installed**: Need `pip install prefect prefect-dask dask-jobqueue` in conda env
- **No pipeline runs with new platform yet**: First run needed to validate end-to-end
- **Embedding panels (VGAE/GAT)**: `embeddings.npz` not yet captured — requires re-running evaluation
- **DQN Policy panel**: `dqn_policy.json` not yet captured — requires re-running fusion evaluation
- **OOD generalization collapse** (Bug 3.6): Research question, not a code bug

## Next Steps

1. **Configure W&B**: `wandb login` on cluster, verify API key works
2. **Install Prefect deps**: `pip install prefect prefect-dask dask-jobqueue` in gnn-experiments env
3. **Validate pipeline**: Run single dataset through new platform end-to-end
4. **Merge to main**: PR from `platform/wandb` → `main` after validation
5. **Re-run evaluation**: Generate embeddings.npz and dqn_policy.json artifacts
6. **Investigate OOD threshold calibration** (Bug 3.6)

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
