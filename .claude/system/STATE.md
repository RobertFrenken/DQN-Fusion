# Current State

**Date**: 2026-02-12

## What's Working

### Pipeline Redesign Phase 1 + Phase 4 (2026-02-12)

**Sub-configs (Phase 1)**:
- 5 frozen dataclasses added to `pipeline/config.py`: `VGAEConfig`, `GATConfig`, `DQNConfig`, `KDConfig`, `FusionConfig`
- `@property` views on `PipelineConfig`: `cfg.vgae`, `cfg.gat`, `cfg.dqn`, `cfg.kd`, `cfg.fusion`
- All consumers migrated: `modules.py`, `utils.py`, `fusion.py`, `evaluation.py` use `cfg.vgae.latent_dim` etc.
- Flat fields, JSON serialization, CLI, presets all unchanged — zero breaking changes
- Existing `config.json` files load without migration

**Write-through DB (Phase 4)**:
- `record_run_start()` / `record_run_end()` added to `pipeline/db.py`
- Wired into `pipeline/cli.py` dispatch block — records to project DB before/after every stage
- Handles both flat and nested eval metrics dicts
- `populate()` remains as backfill/recovery tool (idempotent via `INSERT OR REPLACE`)

**Code cleanliness**:
- `bare except:` → `except OSError:` in `src/training/datamodules.py`
- Hardcoded conda path → `sys.executable` + `KD_GAT_PYTHON` env var in `pipeline/Snakefile`
- Deleted stale `warnings.filterwarnings()` in `src/preprocessing/preprocessing.py`
- Removed 3 `getattr` compatibility shims in `modules.py`, `utils.py`, test file

**Tests**: 33/33 passing (9 new: 7 sub-config + 2 write-through DB)

### Tooling Audit & Upgrade (2026-02-12)
- **Snakemake `benchmark:`**: All 12 non-utility rules emit `benchmark.tsv`
- **Snakemake `report()`**: 3 eval rules wrapped for `snakemake --report report.html`
- **MLflow artifact logging**: Centralized `log_run_artifacts()` in `tracking.py`
- **Datasette**: Installed, browsable at `data/project.db`
- **Pandera validation**: `pipeline/schemas.py` validates Parquet output at ingest time
- **Papermill**: `notebook_report` Snakemake rule + parameterized `03_analytics.ipynb`

### Pipeline System
- Pipeline system (`pipeline/`) fully operational with Snakemake + MLflow
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- 2-level experiment paths: `experimentruns/{dataset}/{size}_{stage}[_kd]/`
- MLflow tracking at `sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db`

### Data Management Layer (2026-02-07)
- **Dataset catalog**: `data/datasets.yaml` — 6 automotive datasets registered
- **Ingestion**: `pipeline/ingest.py` — CSV→Parquet conversion
- **Project DB**: `pipeline/db.py` — SQLite with datasets/runs/metrics tables
  - Write-through from cli.py (new) + backfill via `populate()` (existing)
  - Queryable via `python -m pipeline.db query "SQL"` or Datasette

### Analytics Layer (2026-02-11)
- `pipeline/analytics.py`: sweep, leaderboard, compare, config_diff, dataset_summary
- Uses SQLite `json_extract()` for config queries

### Configuration
- `gradient_checkpointing`: `True` (default)
- `precision`: `"16-mixed"` (default)
- Sub-config access: `cfg.vgae.latent_dim`, `cfg.gat.hidden`, `cfg.dqn.gamma`, `cfg.kd.temperature`, `cfg.fusion.episodes`

## Active `src/` Files

Essential (imported by pipeline):
- `src/models/` -- vgae.py, gat.py, dqn.py (gradient checkpointing support)
- `src/preprocessing/preprocessing.py` -- graph construction (no more warning suppression)
- `src/training/datamodules.py` -- load_dataset() (proper `except OSError`)

## Documentation Structure

```
docs/
└── user_guides/          # For user reference
    ├── snakemake_guide.md
    ├── mlflow_usage.md
    ├── memory_optimization.md
    ├── datasette_usage.md
    └── terminal_upgrades.md
```

## What's Not Working / Incomplete

- **Old experiment checkpoints**: Pre-MLflow runs have no `config.json`
- **Bug 3.6 (research)**: OOD generalization collapse — not a code bug, requires research
- **Bug 3.7**: GAT teacher 137x larger than student due to JK cat mode + `num_fc_layers=3` — needs retrain
- **Parquet conversion**: Not yet run for existing 6 datasets (ingest module ready, need to execute)
- **Phase 3 deferred**: Model registry + dynamic fusion (medium-high risk, 15-D state vector works for 2 models)

## Next Steps

1. **Run Parquet ingestion** for all 6 datasets: `python -m pipeline.ingest --all`
2. **DVC-track Parquet files**: `dvc add data/parquet/automotive/{dataset}` for each
3. **Re-run evaluation** for all 6 datasets (bugs 3.1-3.5 now fixed)
4. **Investigate OOD threshold calibration** (bug 3.6)
5. **Consider GAT FC bottleneck** to address bug 3.7
6. **Phase 3**: Model registry + dynamic fusion (when adding 3rd model type)

## Architecture Summary

- **Entry point**: `python -m pipeline.cli <stage> --preset <model>,<size> --dataset <name>`
- **Config**: `PipelineConfig` frozen dataclass with typed sub-config views (`cfg.vgae`, `cfg.gat`, etc.)
- **Orchestration**: Snakemake (`snakemake -s pipeline/Snakefile --profile profiles/slurm`)
- **Tracking**: MLflow (auto-logged) + project DB (write-through from cli.py)
- **Data catalog**: `data/datasets.yaml` → `pipeline/ingest.py` → Parquet + project DB
- **Query layer**: `python -m pipeline.db query` or Datasette
- **Filesystem**: Snakemake owns paths (DAG trigger), MLflow owns metadata, project DB owns structured results
- **SLURM**: Account PAS3209, gpu partition, V100 GPUs

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **MLflow DB**: `/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db` (auto-backed up to `~/backups/`)
- **Project DB**: `data/project.db` (SQLite — datasets, runs, metrics)
- **Key packages**: PyTorch, PyG, Lightning, MLflow 3.8.1, PyArrow 14.0.2, Datasette, Pandera, psutil
