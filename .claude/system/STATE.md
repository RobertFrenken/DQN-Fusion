# Current State

**Date**: 2026-02-23
**Branch**: `main`

## Ecosystem Status

### Fully Operational (Green)

| Component | Details |
|-----------|---------|
| **Config system** | Pydantic v2 frozen models + YAML composition. `resolve(model_type, scale, auxiliaries, **overrides)` → frozen `PipelineConfig`. 6 datasets in `config/datasets.yaml`. |
| **Training pipeline** | All 72 runs complete (6 datasets × 12 configs: 3 stages × {large, small, small+KD}). CLI: `python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>` |
| **Ray orchestration** | `train_pipeline()` and `eval_pipeline()` via Ray remote tasks + SLURM. `--local` flag for Ray local mode. |
| **SLURM integration** | GPU (V100, PAS3209) + CPU partitions configured. |
| **Graph caching** | All 6 datasets cached with test scenarios (`processed_graphs.pt` + `test_*.pt`). DynamicBatchSampler for variable-size graphs. |
| **DVC tracking** | Raw data + cache tracked. S3 remote + local scratch remote configured. |
| **Export pipeline** | 16 exporters with `--skip-heavy`/`--only-heavy` split. Light exports ~2s on login node; heavy exports (UMAP/attention/graph samples) via SLURM. Now supports t-SNE method. |
| **S3 lakehouse** | Fire-and-forget per-run JSON sync to `s3://kd-gat/lakehouse/runs/`. |
| **S3 dashboard data** | Public read + CORS configured on `s3://kd-gat/dashboard/`. Dashboard JS fetches from S3 with `data/` fallback for local dev. |
| **Dashboard** | Live at https://robertfrenken.github.io/DQN-Fusion/. D3.js v7 ES modules, 27 panels, config-driven via `panelConfig.js`. |
| **Test suite** | 108 tests (88 passed, 20 skipped). All passing on CPU fallback after RAPIDS integration. |

### Partially Working (Yellow)

| Component | Issue |
|-----------|-------|
| **W&B tracking** | 77 online runs; offline runs may need sync. Run `wandb sync wandb/offline-run-*`. |
| **RAPIDS GPU acceleration** | Phase 1 integrated (cuML PCA/UMAP/TSNE in export, cudf.pandas in preprocessing). Fallback to CPU verified. Needs `gnn-rapids` conda env setup (`bash scripts/setup_rapids_env.sh`) and GPU partition testing. |
| **Dashboard panels** | Most panels verified. Timeline, duration, and training curves panels need browser verification. |
| **Inference server** | `pipeline/serve.py` exists (`/predict`, `/health`). Untested with current 72-run checkpoints. |

### Missing (Gray)

| Component | Impact |
|-----------|--------|
| **CI/CD** | No automated testing or deployment. GitHub Actions needed. |
| **gnn-rapids conda env** | Setup script exists but env not yet created. Run `bash scripts/setup_rapids_env.sh` on a GPU node. |
| **RAPIDS Phase 2** | Vectorized `safe_hex_to_int()` (currently falls back to CPU under cudf.pandas due to `.apply()` with Python control flow). |

## RAPIDS Integration (Phase 1)

Added 2026-02-20 (`a2cdcc2`). Zero-code-change GPU acceleration with CPU fallback:

| File | Change |
|------|--------|
| `config/constants.py` | `RAPIDS_AVAILABLE` flag (try import cuml) |
| `pipeline/export.py` | cuML PCA/UMAP/TSNE with sklearn fallback in `_reduce_embeddings()` |
| `src/preprocessing/preprocessing.py` | `cudf.pandas.install()` at module level for transparent DataFrame acceleration |
| `pyproject.toml` | `[project.optional-dependencies] rapids = [cudf-cu12, cuml-cu12, cupy-cuda12x]` |
| `scripts/setup_rapids_env.sh` | New: creates `gnn-rapids` conda env (RAPIDS 24.12 + CUDA 12.4) |
| `scripts/preprocess_gpu_slurm.sh` | New: GPU preprocessing SLURM script |
| `scripts/export_dashboard_slurm.sh` | `--gpu` flag for GPU partition with RAPIDS env |

## Experiment Runs

All 6 datasets × 12 configs = 72 runs on disk in `experimentruns/`:

**Per dataset (12 runs each):**
- `vgae_{large,small,small_kd}_autoencoder` (3 VGAE)
- `gat_{large,small,small_kd}_curriculum` (3 GAT)
- `dqn_{large,small,small_kd}_fusion` (3 DQN)
- `eval_{large,small,small_kd}_evaluation` (3 Eval)

**Eval artifacts per run:** `metrics.json`, `config.json`, `embeddings.npz`, `dqn_policy.json`, `attention_weights.npz`, `explanations.npz` (when `run_explainer=True`)

| Dataset | Runs | Eval Artifacts |
|---------|------|----------------|
| hcrl_ch | 12 | metrics + embeddings + policy + attention |
| hcrl_sa | 12 | metrics + embeddings + policy + attention |
| set_01  | 12 | metrics + embeddings + policy + attention |
| set_02  | 12 | metrics + embeddings + policy + attention |
| set_03  | 12 | metrics + embeddings + policy + attention |
| set_04  | 12 | metrics + embeddings + policy + attention |

## Recently Completed

- **Phase 5: Advanced Enhancements** (2026-02-23):
  - 5.1 GNNExplainer integration (`src/explain.py`, evaluation wiring, export, dashboard panel)
  - 5.4 Trial-based batch size auto-tuning (binary search in `pipeline/memory.py`)
  - 5.3 cuGraph decision gate Phase A (profiling scripts + `scripts/analyze_profile.py`)
  - 5.2 Temporal graph classification (`TemporalGrouper`, `TemporalGraphClassifier`, `train_temporal` stage)
- **RAPIDS Phase 1** (2026-02-20): GPU-accelerated dimensionality reduction + preprocessing with CPU fallback. 108 tests passing.
- **Dashboard rework** (2026-02-17/18): S3 data source migration, embedding panel fix, timestamp support, color theme update
- **Export bug fixes** (2026-02-18): `_scan_runs` config.json parsing, `training_curves` index.json generation
- **PR #2 merged** (2026-02-18): Full platform migration from Snakemake/SQLite to W&B/Prefect/S3
- **72 training runs complete** across 6 datasets × 12 configurations

## Data Flow

```
Raw CAN CSVs (6 datasets, 10.8 GB, DVC)
  → Graph Cache (processed_graphs.pt + test_*.pt, DVC)
    → Training Pipeline (VGAE → GAT → DQN, large + small + small-KD)
      → Evaluation (metrics + embeddings + attention + policy)
        → W&B (77 online) | S3 Lakehouse (DuckDB) | experimentruns/ (72 on disk)
          → Export Pipeline (16 exporters, --skip-heavy/--only-heavy, GPU via RAPIDS)
            → S3 Dashboard Bucket (s3://kd-gat/dashboard/)
              → GitHub Pages Dashboard (27 panels, D3.js v7)
```

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Prefect home**: `/fs/scratch/PAS1266/.prefect/`
- **W&B**: Project `kd-gat` (offline on compute nodes, sync later)
- **Dashboard**: `docs/dashboard/` (GitHub Pages — static JSON + D3.js)
- **Conda**: `gnn-experiments` (CPU, auto-loaded), `gnn-rapids` (GPU, not yet created)
