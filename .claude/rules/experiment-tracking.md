# KD-GAT Experiment Tracking

## W&B

- `cli.py` owns `wandb.init()`/`wandb.finish()` lifecycle.
- Lightning's `WandbLogger` attaches to the active run for per-epoch metrics.
- Compute nodes auto-set `WANDB_MODE=offline`; sync offline runs via `wandb sync wandb/run-*`.

## Datalake (Primary)

Parquet-based structured storage in `data/datalake/`:

| File | Contents |
|------|----------|
| `runs.parquet` | Run-level metadata (dataset, model, scale, stage, KD, success, timestamps) |
| `metrics.parquet` | Per-run per-model core metrics (F1, accuracy, AUC, etc.) |
| `configs.parquet` | Key hyperparameters + full frozen config JSON |
| `datasets.parquet` | Dataset catalog with cache stats |
| `artifacts.parquet` | Manifest: run_id → file path, type, size |
| `training_curves/{run_id}.parquet` | Per-epoch metrics from Lightning CSV logs |
| `analytics.duckdb` | Views + convenience queries over Parquet (always rebuildable) |

**Write path**: `lakehouse.py` appends to Parquet on run completion. `cli.py` calls `register_artifacts()` after each stage.

**Read path**: `build_analytics.py` creates DuckDB views over Parquet. `export.py` reads run metadata from datalake.

**Migration**: `python -m pipeline.migrate_datalake` builds initial Parquet from existing 72 runs. Idempotent.

## Legacy JSON Lakehouse (Deprecated)

- `pipeline/lakehouse.py` still writes per-run JSON to `data/lakehouse/runs/` as backup.
- S3 sync is fire-and-forget to `s3://kd-gat/lakehouse/runs/`. Will be removed after datalake is stable.

## Analytics DuckDB

- `pipeline/build_analytics.py` creates `data/datalake/analytics.duckdb` with views over Parquet.
- Views: `runs`, `metrics`, `datasets`, `configs`, `artifacts`, `v_leaderboard`, `v_kd_impact`.
- Rebuild: `python -m pipeline.build_analytics` (sub-second, just creates views).

## Artifacts

Evaluation stage captures:
- `embeddings.npz` — VGAE z-mean + GAT hidden layers
- `dqn_policy.json` — alpha values by class
- `explanations.npz` — GNNExplainer feature importance (when `run_explainer=True`)

Stored in run directories under `experimentruns/`. Indexed in `artifacts.parquet`.

## Dashboard Export

`python -m pipeline.export` reads metadata from datalake Parquet, artifacts from filesystem → static JSON. `scripts/export_dashboard.sh` commits + pushes to GitHub Pages + syncs to S3.
