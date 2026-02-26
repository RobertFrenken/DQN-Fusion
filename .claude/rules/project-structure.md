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
  export.py         # Datalake/filesystem → static JSON/Parquet export for Quarto reports
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
scripts/            # Automation (run_tests_slurm.sh, build_test_cache.sh, sweep.sh, etc.)
ECOSYSTEM.md        # Dependency ecosystem documentation
user_guides/        # User guides (memory_optimization.md)
reports/            # Quarto website — paper chapters + interactive dashboard (PRIMARY)
  _quarto.yml       # Project config (website, HTML + Typst + Revealjs)
  index.qmd         # Landing page
  dashboard.qmd     # Multi-page dashboard (Overview, Performance, Training, GAT & DQN, KD, Graph, Datasets, Staging)
  slides.qmd        # Revealjs presentation
  pipeline_report.qmd  # Auto-generated pipeline report
  pipeline_dag.svg  # Pipeline DAG visualization
  custom.scss       # Theme overrides
  references.bib    # BibTeX bibliography
  data/             # Report data (Parquet + JSON from export pipeline, incl. graph_samples.json)
  _ojs/             # Observable JS modules (force-graph.js, mosaic-setup.js, theme.js)
  paper/            # Research paper (10 chapters with interactive Mosaic figures)
    index.qmd       # Paper introduction
    02-background.qmd through 09-conclusion.qmd  # Paper body
    10-appendix.qmd # Appendix with model sizing details
    _metadata.yml   # Paper-specific metadata + shared _setup.qmd include
    _setup.qmd      # Shared Mosaic/vgplot + DuckDB-WASM init for figures
    references.bib  # Paper bibliography
    data/           # Paper-specific CSV data (ablation, datasets, model params)
  _site/            # Build output (.gitignored)
notebooks/          # Jupyter notebooks for analysis + prototyping
```
