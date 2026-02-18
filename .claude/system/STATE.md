# Current State

**Date**: 2026-02-18
**Branch**: `main` (merged from `platform/wandb` via PR #2)

## What's Working

### Config System
- Pydantic v2 frozen models + YAML composition fully operational
- `resolve(model_type, scale, auxiliaries, **overrides)` → frozen `PipelineConfig`
- YAML files: `defaults.yaml`, `models/{vgae,gat,dqn}/{large,small}.yaml`, `auxiliaries/{none,kd_standard}.yaml`
- Dataset catalog: `config/datasets.yaml` (6 automotive datasets)
- Path layout: `{dataset}/{model_type}_{scale}_{stage}[_{aux}]`

### Pipeline System
- Pipeline system (`pipeline/`) fully operational with Prefect + W&B
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- CLI: `python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`
- Flow: `python -m pipeline.cli flow --dataset <ds> [--scale <scale>]`
- End-to-end validated: resolve() → config freeze → W&B init → graph loading → training → lakehouse sync

### Platform
- **W&B**: `wandb.init()`/`wandb.finish()` in CLI, WandbLogger in Lightning trainer. Offline mode on compute nodes. 77 runs total (3 offline pending sync).
- **Prefect**: `train_pipeline()` and `eval_pipeline()` flows with dask-jobqueue SLURMCluster.
- **S3 Lakehouse**: Fire-and-forget sync via `pipeline/lakehouse.py` to `s3://kd-gat/lakehouse/runs/`.
- **DVC**: S3 remote configured alongside local scratch remote.

### Dashboard (GitHub Pages)
- **Live**: https://robertfrenken.github.io/DQN-Fusion/
- **Stack**: Static JSON + D3.js v7 (ES modules), deployed from `docs/` on `main`
- **All panels populated**: 73 embedding files, 18 DQN policy, 18 attention, 54 ROC curves, 18 recon errors, 6 CKA matrices
- **Auto-export**: `scripts/export_dashboard.sh` runs export + commit + push + DVC push

### Test Infrastructure
- **101 tests passing** across 8 test files
- Parallel SLURM submission: `scripts/run_tests_parallel.sh`
- Sequential fallback: `scripts/run_tests_slurm.sh` (120min, 64GB)
- `@pytest.mark.slurm` marker auto-skips heavy tests on login nodes

## Experiment Status

All 6 datasets have complete pipelines (9 checkpoints + 3 eval runs each = 54 checkpoints + 18 eval runs):

| Dataset | VGAE (L/S/S-KD) | GAT (L/S/S-KD) | DQN (L/S/S-KD) | Eval (L/S/S-KD) |
|---------|-----------------|-----------------|-----------------|------------------|
| hcrl_ch | 1.8M/352K/352K | 6.4M/406K/406K | 5.3M/260K/260K | 234K/221K/224K |
| hcrl_sa | 1.8M/352K/352K | 6.4M/406K/406K | 5.3M/260K/260K | 265K/297K/261K |
| set_01  | 760K/55K/55K   | 6.3M/344K/344K | 5.3M/260K/260K | 293K/289K/292K |
| set_02  | 1.8M/352K/352K | 6.4M/406K/406K | 5.3M/260K/260K | 233K/241K/232K |
| set_03  | 1.7M/313K/313K | 6.4M/398K/398K | 5.3M/260K/260K | 232K/288K/232K |
| set_04  | 1.8M/352K/352K | 6.4M/406K/406K | 5.3M/260K/260K | 290K/289K/287K |

Most recent checkpoint: `hcrl_sa/vgae_large_autoencoder` (2026-02-17 20:01, validation run).

## Recently Completed

- **PR #2 merged to main** (2026-02-18): Full platform migration from Snakemake/SQLite to W&B/Prefect/S3
- **Legacy cleanup**: Removed all Snakemake/SQLite/MLflow references from code and docs
- **Dashboard export**: All 26 panels populated with fresh data
- **Test validation**: 101 tests passing across 8 files on SLURM

## What's Not Working / Incomplete

- **W&B offline runs**: 3 offline runs pending sync (`wandb sync wandb/offline-run-*`)
- **OOD generalization collapse** (Bug 3.6): Research question, not a code bug

## Next Steps

1. **Sync W&B offline runs**: `wandb sync wandb/offline-run-*`
2. **Push dashboard to GitHub Pages**: Verify live at https://robertfrenken.github.io/DQN-Fusion/
3. **Investigate OOD threshold calibration** (Bug 3.6)
4. **Run fresh experiments** with new platform on all datasets

## Filesystem

- **Inode usage**: ~390k / 20M
- **Conda envs**: gnn-experiments

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Prefect home**: `/fs/scratch/PAS1266/.prefect/` (GPFS)
- **W&B**: Project `kd-gat` (offline on compute nodes, sync later)
- **Dashboard**: `docs/dashboard/` (GitHub Pages — static JSON + D3.js)
- **Conda**: Auto-loaded via `~/.bashrc` (`module load miniconda3` + `conda activate gnn-experiments`)
