# KD-GAT Experiment Tracking

## W&B

- `cli.py` owns `wandb.init()`/`wandb.finish()` lifecycle.
- Lightning's `WandbLogger` attaches to the active run for per-epoch metrics.
- Compute nodes auto-set `WANDB_MODE=offline`; sync offline runs via `wandb sync wandb/run-*`.

## Local Lakehouse

- `pipeline/lakehouse.py` writes per-run JSON to `data/lakehouse/runs/` (NFS, canonical).
- S3 sync is fire-and-forget backup to `s3://kd-gat/lakehouse/runs/`. NFS wins if they disagree.

## Analytics DuckDB

- `pipeline/build_analytics.py` materializes `data/lakehouse/analytics.duckdb` from lakehouse JSON + `experimentruns/` filesystem.
- Tables: `runs`, `metrics`, `datasets`, `configs`.

## Artifacts

Evaluation stage captures:
- `embeddings.npz` — VGAE z-mean + GAT hidden layers
- `dqn_policy.json` — alpha values by class
- `explanations.npz` — GNNExplainer feature importance (when `run_explainer=True`)

Stored in run directories under `experimentruns/`.

## Dashboard Export

`python -m pipeline.export` scans `experimentruns/` filesystem → static JSON. `scripts/export_dashboard.sh` commits + pushes to GitHub Pages + syncs to S3.
