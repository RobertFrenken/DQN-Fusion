# Current State

**Date**: 2026-02-16

## What's Working

### Config System
- Pydantic v2 frozen models + YAML composition fully operational
- `resolve(model_type, scale, auxiliaries, **overrides)` → frozen `PipelineConfig`
- YAML files: `defaults.yaml`, `models/{vgae,gat,dqn}/{large,small}.yaml`, `auxiliaries/{none,kd_standard}.yaml`
- Dataset catalog: `config/datasets.yaml` (6 automotive datasets)
- Path layout: `{dataset}/{model_type}_{scale}_{stage}[_{aux}]`
- Legacy flat JSON loading via `PipelineConfig.load()` with automatic migration

**Tests**: 105+ passing, layer boundary tests (4/4) pass. 4 pre-existing FK failures in `TestWriteThroughDB` (slurm-marked tests auto-skip on login nodes).

### Pipeline System
- Pipeline system (`pipeline/`) fully operational with Snakemake + project DB
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- CLI: `python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`
- End-to-end validated: resolve() → config freeze → DB recording → graph loading → training
- Snakemake `onsuccess` auto-exports dashboard data after pipeline runs

### Snakemake Features
- **Retries with resource scaling**: `retries: 2` on all training rules, `mem_mb=128GB * attempt`
- **Preprocessing rule**: Dedicated `preprocess` rule warms graph cache per dataset
- **Between-workflow caching**: `cache: True` on preprocess rule, `SNAKEMAKE_OUTPUT_CACHE` on scratch
- **Group jobs**: `group: "evaluation"` bundles 3 eval rules into single SLURM submissions
- **Benchmarks**: Already in use on all training + eval rules

### Data Management Layer
- **Parquet ingestion**: All 6 datasets converted (`data/parquet/automotive/{dataset}/`)
- **Project DB**: `data/project.db` — 6 datasets, 70 runs, 3915 metrics, 18,290 epoch_metrics
  - Write-through from cli.py + backfill via `populate()`
  - WAL mode + 15s busy timeout + `_retry_on_locked` decorator for concurrent SLURM jobs
  - Indices on metrics and epoch_metrics tables
  - `populate()` runs: `_migrate_legacy_runs()` (with stale entry cleanup), `_backfill_timestamps()` (started_at + completed_at), `_backfill_epoch_metrics()`, `_backfill_teacher_run()`
- **Analytics**: sweep, leaderboard, compare, config_diff, dataset_summary

### Dashboard (GitHub Pages)
- **Live**: https://robertfrenken.github.io/DQN-Fusion/
- **Stack**: Static JSON + D3.js v7 (ES modules), deployed from `docs/` on `main`
- **Architecture**: Config-driven component system (BaseChart → 8 chart types, Registry, PanelManager)
- **Working panels (5 original)**: Leaderboard (270 entries), Dataset Comparison, KD Transfer (108 pairs), Run Timeline (70 timestamped runs), Training Curves (36 JSON files from 18,290 epoch_metrics)
- **New panels (6, data-pending)**: Force Graph (CAN bus visualization), Bubble Chart (multi-metric comparison), VGAE Latent Space, GAT State Space, DQN Policy Histogram, Model Predictions Breakdown
- **Exported data**: `metric_catalog.json` (20 metrics), `graph_samples.json` (18 samples), `model_sizes.json` (6 entries)
- **Auto-export**: `scripts/export_dashboard.sh` runs in Snakemake `onsuccess`
- **Export validation**: `_validate_data()` warns on empty tables before export

### Dashboard Component Architecture
```
docs/dashboard/js/
  core/
    BaseChart.js      # SVG setup, margins, tooltip, responsive, loading/error/no-data states
    Registry.js       # Chart type registry: register(name, class), get(name)
    Theme.js          # Shared palette (8 colors), CSS variables, scales
  charts/
    TableChart.js     # Sortable leaderboard table
    BarChart.js       # Grouped bar (dataset comparison, model predictions)
    ScatterChart.js   # Configurable scatter (KD transfer, embeddings)
    LineChart.js      # Training curves
    TimelineChart.js  # Run timeline
    BubbleChart.js    # Extends ScatterChart (3-metric + size encoding)
    ForceGraph.js     # D3 force simulation (CAN bus graph)
    HistogramChart.js # Stacked histogram (DQN alpha distribution)
  panels/
    panelConfig.js    # Declarative panel definitions (11 panels)
    PanelManager.js   # Config → nav + panels + controls → lazy-load → render
  app.js              # Slim entry: imports + PanelManager.init()
```

Adding a new panel = adding an entry to `panelConfig.js`. BaseChart lifecycle: `init() → update(data) → render(data) → destroy()`.

### Test Infrastructure
- `@pytest.mark.slurm` marker auto-skips heavy tests on login nodes
- `scripts/run_tests_slurm.sh` submits pytest to SLURM compute nodes
- `--run-slurm` pytest flag or `SLURM_JOB_ID` env var enables slurm-marked tests

### GAT Architecture (Bug 3.7 Fixed)
- Large GAT: `fc_layers: 1` (343k params) — removed bloated 1.3M-param hidden FC layer
- Small GAT: `fc_layers: 2` (65k params) — removed one redundant hidden FC layer
- Teacher/student ratio: **5.3x** (was 16.3x)

## Active `src/` Files

Essential (imported by pipeline):
- `src/models/` — vgae.py, gat.py, dqn.py (gradient checkpointing support)
- `src/models/registry.py` — Model construction + fusion feature extraction
- `src/preprocessing/preprocessing.py` — graph construction
- `src/training/datamodules.py` — load_dataset()

## What's Not Working / Incomplete

- **Old experiment checkpoints**: Pre-MLflow runs have no `config.json`
- **Bug 3.6 (research)**: OOD generalization collapse — not a code bug, requires research
- **E2E tests**: Pre-existing failure — `train_autoencoder()` doesn't write config.json (CLI does)
- **TestWriteThroughDB**: 4 pre-existing test failures — FK constraint on `runs.dataset → datasets(name)` fires because tests insert runs without first inserting a dataset row. Needs fixture fix.
- **Embedding panels (VGAE/GAT)**: `embeddings.npz` not yet captured — requires re-running evaluation with artifact capture code (Phase 3.3-3.4 of dashboard plan)
- **DQN Policy panel**: `dqn_policy.json` not yet captured — requires re-running fusion evaluation (Phase 3.5-3.6)
- **Model Predictions panel**: Per-run metric files need additional breakdown by test scenario
- **Dashboard not yet committed/pushed**: New component architecture + data files need commit + push to GitHub Pages

## Recently Completed

- **Dashboard hardening + visualization expansion** (2026-02-16): Major 4-phase implementation:
  - **Phase 1 (Pipeline Fixes)**: Backfilled epoch_metrics from 44 Lightning CSVs → 18,290 rows (was 0). Backfilled `started_at` timestamps → 100% coverage (was 0%). Added stale entry cleanup to `_migrate_legacy_runs()` — removes DB entries for directories that no longer exist on disk (cleaned 70 orphan entries). Added `_validate_data()` pre-export warnings.
  - **Phase 2 (Component Architecture)**: Refactored monolithic `charts.js` (357 lines) into config-driven ES module system: `BaseChart` base class with SVG/tooltip/responsive infrastructure, `Registry` for chart type lookup, `Theme` for shared palette, 8 chart classes (`TableChart`, `BarChart`, `ScatterChart`, `LineChart`, `TimelineChart`, `BubbleChart`, `ForceGraph`, `HistogramChart`), `PanelManager` orchestrator, `panelConfig.js` declarative definitions (11 panels). `index.html` reduced to minimal shell.
  - **Phase 3 (New Data Exports)**: Graph samples from PyG cache → `graph_samples.json` (18 samples). Model parameter counts from registry → `model_sizes.json` (6 entries). Export functions for embeddings and DQN policy added (await re-evaluation runs). Updated `export_all()` with all new exports.
  - **Phase 4 (New Visualizations)**: 6 new panel definitions added to `panelConfig.js`: Force Graph (CAN bus structure), Bubble Chart (multi-metric comparison), VGAE Latent Space, GAT State Space, DQN Policy Histogram, Model Predictions Breakdown.
  - **Evaluation stage updated**: `evaluation.py` now captures `embeddings.npz` (VGAE z-mean + GAT hidden representations) and `dqn_policy.json` (alpha values by class) during inference. Artifacts saved alongside metrics.json.
  - Modified files: `pipeline/db.py`, `pipeline/export.py`, `pipeline/stages/evaluation.py`, `docs/dashboard/index.html`, `docs/dashboard/css/style.css`, `docs/dashboard/js/app.js`. Created 14 new JS files. Deleted `docs/dashboard/js/charts.js`.
- **Fragility fixes** (2026-02-15): Implemented 10-item fragility fix plan across 3 phases.
- **Dashboard data pipeline fix** (2026-02-15): Fixed 3 broken dashboard tabs.
- **SLURM test dispatch** (2026-02-15): Added `@pytest.mark.slurm` to E2E and smoke tests.
- **Dashboard deployment** (2026-02-15): GitHub Pages dashboard live.
- **Legacy path migration** (2026-02-14): All 70 `teacher_*/student_*` dirs renamed.

## Next Steps

1. **Commit + push dashboard**: Commit new component architecture, data files, and pipeline changes; push to GitHub Pages
2. **Re-run evaluation with artifact capture**: Run evaluation stage to generate `embeddings.npz` and `dqn_policy.json` for each dataset (enables VGAE/GAT/DQN visualization panels)
3. **Re-export after evaluation**: `python -m pipeline.export` to populate `embeddings/` and `dqn_policy/` data directories
4. **Update `scripts/export_dashboard.sh`** and Snakefile `onsuccess` to include new export functions
5. **Verify dashboard locally**: `python -m http.server -d docs/dashboard` to test all 11 panels
6. **Fix TestWriteThroughDB**: Add dataset fixture to DB tests so FK constraint passes
7. **Full pipeline GPU run**: `snakemake --profile profiles/slurm --config 'datasets=["hcrl_sa"]'`
8. **Investigate OOD threshold calibration** (bug 3.6)

## Filesystem

- **Inode usage**: 386k / 1M (cleaned from 763k on 2026-02-13)
- **Conda envs**: Only `gnn-experiments` remains (removed py310, gnn-gpu, dfl, gpu_practice)

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Snakemake cache**: `/fs/scratch/PAS1266/snakemake-cache/`
- **Project DB**: `data/project.db` (SQLite WAL — datasets, runs, metrics, epoch_metrics)
- **Dashboard**: https://robertfrenken.github.io/DQN-Fusion/ (GitHub Pages from `docs/`)
- **Conda**: `module load miniconda3/24.1.2-py310 && conda activate gnn-experiments`
