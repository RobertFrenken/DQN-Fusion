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
  lakehouse.py      # Datalake Parquet append (fire-and-forget)
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
docs/dashboard/     # Legacy D3.js dashboard (being retired, panels migrated to Quarto)
  js/core/          # BaseChart, Registry, Theme
  js/charts/        # 8 chart types (Table, Bar, Scatter, Line, Timeline, Bubble, ForceGraph, Histogram)
  js/panels/        # PanelManager + panelConfig (11 panels, declarative)
  js/app.js         # Slim entry point
  data/             # Static JSON exports from pipeline
reports/            # Quarto website — paper chapters + interactive dashboard (PRIMARY)
  _quarto.yml       # Project config (website, HTML + Typst + Revealjs)
  index.qmd         # Introduction
  02-background.qmd through 07-evaluation.qmd  # Paper chapters
  appendix.qmd      # Appendix
  dashboard.qmd     # Multi-page dashboard (Overview, Performance, Training, GAT & DQN, KD, Graph, Datasets, Staging)
  slides.qmd        # Revealjs presentation
  pipeline_report.qmd  # Auto-generated pipeline report
  custom.scss       # Theme overrides
  references.bib    # BibTeX bibliography
  data/             # Report data (Parquet + JSON from export pipeline, incl. graph_samples.json)
  _ojs/             # Observable JS modules (force-graph.js, mosaic-setup.js, theme.js)
  _site/            # Build output (.gitignored)
notebooks/          # Jupyter notebooks for analysis + prototyping
```
