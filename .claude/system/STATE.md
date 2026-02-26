# Current State

**Date**: 2026-02-25
**Branch**: `main`

## Ecosystem Status

### Fully Operational (Green)

| Component | Details |
|-----------|---------|
| **Config system** | Pydantic v2 frozen models + YAML composition. `resolve(model_type, scale, auxiliaries, **overrides)` → frozen `PipelineConfig`. 6 datasets in `config/datasets.yaml`. |
| **Training pipeline** | All 72 runs complete (6 datasets × 12 configs: 3 stages × {large, small, small+KD}). CLI: `python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>` |
| **Ray orchestration** | `train_pipeline()` and `eval_pipeline()` via Ray remote tasks + SLURM. `--local` flag for Ray local mode. Subprocess-per-stage dispatch (intentional — CUDA context isolation). `small_nokd` runs concurrently with `large`. Benchmark mode via `KD_GAT_BENCHMARK=1`. |
| **SLURM integration** | Pitzer cluster. GPU (2x V100 per node, 362GB RAM, PAS3209) + CPU partitions. |
| **Graph caching** | All 6 datasets cached with test scenarios (`processed_graphs.pt` + `test_*.pt`). DynamicBatchSampler for variable-size graphs. |
| **DVC tracking** | Raw data + cache tracked. S3 remote + local scratch remote configured. |
| **Export pipeline** | 8 lightweight exporters (~2s, login node safe) → `reports/data/`: leaderboard, runs, metrics, metric catalog, datasets, KD transfer, training curves, model sizes. Heavy analysis (UMAP, attention, CKA, etc.) in notebooks. |
| **Datalake** | Parquet-based structured storage in `data/datalake/` (runs, metrics, configs, artifacts, training curves). DuckDB analytics views. S3 backup via `aws s3 sync` in SLURM epilog. |
| **Quarto site** | Dashboard + paper + slides rendered via Quarto. Auto-deployed to GitHub Pages via GitHub Actions on push to main. |
| **Test suite** | 108 tests (88 passed, 20 skipped). All passing on CPU fallback after RAPIDS integration. |

### Partially Working (Yellow)

| Component | Issue |
|-----------|-------|
| **W&B tracking** | 77 online runs; offline runs may need sync. Run `wandb sync wandb/offline-run-*`. |
| **RAPIDS GPU acceleration** | Phase 1 integrated (cuML PCA/UMAP/TSNE in export, cudf.pandas in preprocessing). Fallback to CPU verified. Needs `gnn-rapids` conda env setup (`bash scripts/setup_rapids_env.sh`) and GPU partition testing. |
| **Paper figures** | Interactive Mosaic figures ported from dashboard to paper chapters. Browser verification needed. |
| **Inference server** | `pipeline/serve.py` exists (`/predict`, `/health`). Untested with current 72-run checkpoints. |

### Missing (Gray)

| Component | Impact |
|-----------|--------|
| **CI/CD** | GitHub Actions CI: lint + test + quarto-build (auto-deploy to gh-pages on main). |
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
| `pipeline/export.py` | RAPIDS cuML PCA/UMAP/TSNE with sklearn fallback in `_reduce_embeddings()` |

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

- **Pipeline evolution plan complete** (2026-02-25):
  - WS1 (Prefect cleanup): All stale Prefect/Dask/Snakemake references removed. `.snakemake/` deleted.
  - WS2 (Orchestration research): Decision document at `~/plans/orchestration-redesign-decision.md`. Concurrent `small_nokd` variant refactored. Benchmark instrumentation added. R1 benchmark submitted (job 44398773). R2 pending R1 results.
  - WS3 (Datalake consolidation): All 6 phases complete — migration script, Parquet writes, analytics views, export integration, CLI registration, documentation.
  - Cleanup: ECOSYSTEM.md Diagram 5 fixed (path + datalake). 5 stale git stashes dropped. `.claude/settings.local.json` cleaned.
- **Quarto migration complete** (2026-02-25):
  - Deleted legacy D3 dashboard (`docs/dashboard/`), old stub chapters, export scripts
  - Ported 9 interactive Mosaic figures from `dashboard.qmd` into paper chapters
  - Updated CI: removed js-syntax + docs-site-build jobs, added quarto-build + gh-pages deploy
  - Navbar now points to `paper/` chapters. Landing page (`index.qmd`) updated.
- **Phase 5: Advanced Enhancements** (2026-02-23):
  - 5.1 GNNExplainer integration (`src/explain.py`, evaluation wiring, export, dashboard panel)
  - 5.4 Trial-based batch size auto-tuning (binary search in `pipeline/memory.py`)
  - 5.3 cuGraph decision gate Phase A (profiling scripts + `scripts/analyze_profile.py`)
  - 5.2 Temporal graph classification (`TemporalGrouper`, `TemporalGraphClassifier`, `train_temporal` stage)
- **RAPIDS Phase 1** (2026-02-20): GPU-accelerated dimensionality reduction + preprocessing with CPU fallback. 108 tests passing.
- **Dashboard rework** (2026-02-17/18): S3 data source migration, embedding panel fix, timestamp support, color theme update
- **Export bug fixes** (2026-02-18): `_scan_runs` config.json parsing, `training_curves` index.json generation
- **PR #2 merged** (2026-02-18): Full platform migration from Snakemake/SQLite to W&B/Ray/S3
- **72 training runs complete** across 6 datasets × 12 configurations

## Data Flow

```
Raw CAN CSVs (6 datasets, 10.8 GB, DVC)
  → Graph Cache (processed_graphs.pt + test_*.pt, DVC)
    → Training Pipeline (VGAE → GAT → DQN, large + small + small-KD)
      → Evaluation (metrics + embeddings + attention + policy)
        → W&B (77 online) | Datalake (Parquet) | experimentruns/ (72 on disk)
          → Export Pipeline (8 lightweight exporters → reports/data/)
            → Quarto Site (dashboard + paper + slides)
              → GitHub Pages (auto-deploy via GitHub Actions on push to main)
```

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Ray temp**: `/fs/scratch/PAS1266/.ray/`
- **W&B**: Project `kd-gat` (offline on compute nodes, sync later)
- **Reports**: `reports/` (Quarto site — auto-deployed to GitHub Pages)
- **Conda**: `gnn-rapids` (GPU, not yet created — for RAPIDS only)
