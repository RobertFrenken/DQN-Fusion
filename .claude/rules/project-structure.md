# KD-GAT Project Structure

## 3-Layer Hierarchy

```
config/             # Layer 1: Inert, declarative (no imports from pipeline/ or src/)
  schema.py         # Pydantic v2 frozen models: PipelineConfig, VGAEArchitecture, etc.
  resolver.py       # YAML composition: defaults → model_def → auxiliaries → CLI
  paths.py          # Path layout: {dataset}/{model_type}_{scale}_{stage}[_{aux}]
  constants.py      # Domain/infrastructure constants (window sizes, feature counts, etc.)
  __init__.py       # Re-exports: from config import PipelineConfig, resolve, checkpoint_path, ...
  defaults.yaml     # Global baseline config values
  datasets.yaml     # Dataset catalog (add entries here for new datasets)
  models/           # Architecture × Scale YAML files
    vgae/large.yaml, small.yaml
    gat/large.yaml, small.yaml
    dqn/large.yaml, small.yaml
  auxiliaries/      # Loss modifier YAML files (composable)
    none.yaml, kd_standard.yaml
pipeline/           # Layer 2: Orchestration (imports config/, lazy imports from src/)
  cli.py            # Entry point + W&B init + lakehouse sync + archive restore on failure
  serve.py          # FastAPI inference server (/predict, /health)
  stages/           # Stage implementations (training, fusion, evaluation, temporal)
    evaluation.py   # Multi-model eval; captures embeddings.npz + dqn_policy.json + explanations.npz
    temporal.py     # Temporal graph classification (GAT encoder + Transformer over time)
  orchestration/    # Ray orchestration (ray_pipeline, ray_slurm, tune_config)
  tracking.py       # Memory monitoring utilities
  export.py         # Datalake/filesystem → static JSON export for dashboard
  memory.py         # GPU memory management (static, measured, trial-based batch sizing)
  lakehouse.py      # Datalake Parquet append + S3 JSON backup (fire-and-forget)
  migrate_datalake.py  # One-time migration: filesystem → Parquet datalake
  build_analytics.py   # DuckDB views over datalake Parquet
src/                # Layer 3: Domain (models, training, preprocessing; imports config/)
  models/           # vgae.py, gat.py, dqn.py, temporal.py
  explain.py        # GNNExplainer integration (feature importance analysis)
  training/         # load_dataset(), load_test_scenarios(), graph caching
  preprocessing/    # Graph construction from CAN CSVs + temporal.py (TemporalGrouper)
data/
  automotive/       # 6 datasets (DVC-tracked): hcrl_ch, hcrl_sa, set_01-04
  cache/            # Preprocessed graph cache (.pt, .pkl, metadata)
  datalake/         # Parquet structured storage (runs, metrics, configs, artifacts, training_curves/)
                    # analytics.duckdb (views over Parquet)
experimentruns/     # Outputs: best_model.pt, config.json, metrics.json, embeddings.npz, dqn_policy.json, explanations.npz
scripts/            # Automation (export_dashboard.sh, run_tests_slurm.sh, build_test_cache.sh, sweep.sh, etc.)
docs/dashboard/     # GitHub Pages D3.js dashboard (ES modules, config-driven panels)
  js/core/          # BaseChart, Registry, Theme
  js/charts/        # 8 chart types (Table, Bar, Scatter, Line, Timeline, Bubble, ForceGraph, Histogram)
  js/panels/        # PanelManager + panelConfig (11 panels, declarative)
  js/app.js         # Slim entry point
  data/             # Static JSON exports from pipeline
docs-site/          # Astro 5 + Svelte 5 interactive research paper site
  src/components/   # D3Chart.svelte, PlotFigure.svelte, FigureIsland.astro
  src/components/figures/  # Interactive figure islands
  src/config/       # figures.ts (paper figure registry), shared.ts
  src/content.config.ts  # Astro Content Collections (Zod schemas)
  src/data/         # Catalog JSON (synced from docs/dashboard/data/)
  src/layouts/      # ArticleLayout.astro (CSS Grid Distill-style)
  src/lib/d3/       # All 11 D3 chart classes + BaseChart + Theme
  src/lib/data.ts   # Typed fetch helpers for per-run data
  src/lib/resource.svelte.ts  # Reactive fetch-on-state-change for Svelte 5
  src/pages/        # index.astro, showcase.astro, test-figure.mdx
  scripts/sync-data.sh  # Sync dashboard data → src/data/
notebooks/          # Deno Jupyter notebooks for prototyping plots
```
