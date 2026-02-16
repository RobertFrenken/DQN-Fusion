# KD-GAT: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection using a 3-stage knowledge distillation pipeline:
VGAE (unsupervised reconstruction) → GAT (supervised classification) → DQN (RL fusion).
Large models are compressed into small models via KD auxiliaries for edge deployment.

## Key Commands

```bash
# Run a single stage
python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>

# Stages: autoencoder, curriculum, fusion, evaluation
# Models: vgae, gat, dqn
# Scales: large, small
# Auxiliaries: none (default), kd_standard

# Examples
python -m pipeline.cli autoencoder --model vgae --scale large --dataset hcrl_sa
python -m pipeline.cli curriculum --model gat --scale small --auxiliaries kd_standard --teacher-path <path> --dataset hcrl_sa
python -m pipeline.cli fusion --model dqn --scale large --dataset hcrl_ch

# Override nested config values
python -m pipeline.cli autoencoder --model vgae --scale large -O training.lr 0.001 -O vgae.latent_dim 16

# Full pipeline via Snakemake + SLURM
snakemake -s pipeline/Snakefile --profile profiles/slurm

# Single dataset
snakemake -s pipeline/Snakefile --profile profiles/slurm --config 'datasets=["hcrl_sa"]'

# Dry run (always do this first)
snakemake -s pipeline/Snakefile -n

# Data management
python -m pipeline.ingest <dataset>     # CSV → Parquet conversion
python -m pipeline.ingest --all         # Convert all datasets
python -m pipeline.ingest --list        # List catalog entries
python -m pipeline.db populate          # Populate project DB from existing outputs
python -m pipeline.db summary           # Show dataset/run/metric counts
python -m pipeline.db query "SQL"       # Run arbitrary SQL on project DB

# Test graph caching (builds test_*.pt per scenario per dataset)
sbatch scripts/build_test_cache.sh                    # All datasets (set_01-04)
sbatch scripts/build_test_cache.sh set_02 set_03      # Specific datasets

# Experiment analytics (queries project DB)
python -m pipeline.analytics sweep --param lr --metric f1
python -m pipeline.analytics leaderboard --metric f1 --top 10
python -m pipeline.analytics compare <run_a> <run_b>
python -m pipeline.analytics diff <run_a> <run_b>
python -m pipeline.analytics dataset <name>
python -m pipeline.analytics query "SELECT json_extract(...) FROM ..."
python -m pipeline.analytics memory [--model vgae] [--dataset hcrl_sa]

# STATE.md automation
python -m pipeline.state_sync preview   # Preview regenerated sections
python -m pipeline.state_sync update    # Write updated STATE.md in-place
python -m pipeline.cli state            # Shorthand for state_sync update

# Export dashboard data (DB → static JSON)
python -m pipeline.export                                    # Default: docs/dashboard/data/
python -m pipeline.export --output-dir docs/dashboard/data   # Explicit output dir
bash scripts/export_dashboard.sh              # Export + commit + push to Pages
bash scripts/export_dashboard.sh --no-push    # Export + commit only
bash scripts/export_dashboard.sh --dry-run    # Export only (no git)

# Datasette (interactive DB browsing, inside tmux on login node)
datasette data/project.db --port 8001
# Local: ssh -L 8001:localhost:8001 rf15@pitzer.osc.edu → http://localhost:8001

# Snakemake report (after eval runs)
snakemake -s pipeline/Snakefile --report report.html

# Run tests (slurm-marked tests auto-skip on login nodes)
python -m pytest tests/ -v

# Run heavy tests on SLURM compute node
bash scripts/run_tests_slurm.sh                         # all tests
bash scripts/run_tests_slurm.sh -k "test_full_pipeline"  # specific test
bash scripts/run_tests_slurm.sh -m slurm                 # only slurm-marked

# Check SLURM jobs
squeue -u $USER
```

## Project Structure (3-layer hierarchy)

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
  cli.py            # Entry point + write-through DB recording
  stages/           # Stage implementations (training, fusion, evaluation)
    evaluation.py   # Multi-model eval; captures embeddings.npz + dqn_policy.json artifacts
  tracking.py       # Memory monitoring utilities
  export.py         # DB → static JSON export for dashboard (+ graph_samples, model_sizes, embeddings, dqn_policy)
  memory.py         # GPU memory management
  ingest.py         # CSV → Parquet conversion + dataset registration
  db.py             # SQLite project DB (WAL mode) + write-through + backfill migrations (epoch_metrics, timestamps, teacher_run)
  analytics.py      # Post-run analysis: sweeps, leaderboards, comparisons
  Snakefile         # All stages + evaluation + preprocessing cache + test cache + retries + group jobs
src/                # Layer 3: Domain (models, training, preprocessing; imports config/)
  models/           # vgae.py, gat.py, dqn.py
  training/         # load_dataset(), load_test_scenarios(), graph caching
  preprocessing/    # Graph construction from CAN CSVs
data/
  project.db        # SQLite DB: queryable datasets, runs, metrics, epoch_metrics
  automotive/       # 6 datasets (DVC-tracked): hcrl_ch, hcrl_sa, set_01-04
  parquet/          # Columnar format (from ingest), queryable via Datasette or SQL
  cache/            # Preprocessed graph cache (.pt, .pkl, metadata)
experimentruns/     # Outputs: best_model.pt, config.json, metrics.json, embeddings.npz, dqn_policy.json
scripts/            # Automation (export_dashboard.sh, run_tests_slurm.sh, build_test_cache.sh)
docs/dashboard/     # GitHub Pages D3.js dashboard (ES modules, config-driven panels)
  js/core/          # BaseChart, Registry, Theme
  js/charts/        # 8 chart types (Table, Bar, Scatter, Line, Timeline, Bubble, ForceGraph, Histogram)
  js/panels/        # PanelManager + panelConfig (11 panels, declarative)
  js/app.js         # Slim entry point
  data/             # Static JSON exports from pipeline
profiles/slurm/     # SLURM submission profile for Snakemake
```

## Config System

Config is defined by four orthogonal concerns: **model_type** (architecture), **scale** (capacity), **auxiliaries** (loss modifiers like KD), and **dataset**. Adding a new value along any axis = adding a YAML file.

**Resolution order**: `defaults.yaml` → `models/{type}/{scale}.yaml` → `auxiliaries/{aux}.yaml` → CLI overrides → Pydantic validation → frozen.

```python
from config import resolve, PipelineConfig
cfg = resolve("vgae", "large", dataset="hcrl_sa")          # No KD
cfg = resolve("gat", "small", auxiliaries="kd_standard")    # With KD
cfg.vgae.latent_dim    # Nested sub-config access
cfg.training.lr        # Training hyperparameters
cfg.has_kd             # Property: any KD auxiliary?
cfg.kd.temperature     # KD auxiliary config (via property)
cfg.active_arch        # Architecture config for active model_type
```

**Path layout**: `experimentruns/{dataset}/{model_type}_{scale}_{stage}[_{aux}]`
  - `experimentruns/hcrl_sa/vgae_large_autoencoder/`
  - `experimentruns/hcrl_sa/gat_small_curriculum_kd/`

**Legacy config loading**: Old flat JSON config files (`model_size`, `use_kd`, `teacher_path`) still load via `PipelineConfig.load()` with automatic migration. All experiment directories now use the new naming convention (legacy `teacher_*/student_*` dirs were migrated 2026-02-14).

## Critical Constraints

These fix real crashes -- do not violate:

- **PyG `Data.to()` is in-place.** Always `.clone().to(device)`, never `.to(device)` on shared data.
- **Use spawn multiprocessing.** Never `fork` with CUDA. Set `mp_start_method='spawn'` and `multiprocessing_context='spawn'` on all DataLoaders.
- **NFS filesystem.** `.nfs*` ghost files appear on delete. Already in `.gitignore`.
- **No GUI on HPC.** Git auth via SSH key, not HTTPS.

## Architecture Decisions

- **3-layer import hierarchy** (enforced by `tests/test_layer_boundaries.py`):
  - `config/` → never imports from `pipeline/` or `src/`
  - `pipeline/` → imports `config/` at top level; imports `src/` only inside functions (lazy)
  - `src/` → imports `config/` (constants); never imports from `pipeline/`
- Config: Pydantic v2 frozen BaseModels + YAML composition + JSON serialization.
- Sub-configs: `cfg.vgae`, `cfg.gat`, `cfg.dqn`, `cfg.training`, `cfg.fusion` — nested Pydantic models. Always use nested access (`cfg.vgae.latent_dim`), never flat.
- Auxiliaries: `cfg.auxiliaries` is a list of `AuxiliaryConfig`. KD is a composable loss modifier, not a model identity. Use `cfg.has_kd` / `cfg.kd` properties.
- Constants: domain/infrastructure constants live in `config/constants.py` (not in PipelineConfig). Hyperparameters live in PipelineConfig.
- Write-through DB: `cli.py` records run start/end directly to project DB (including teacher_run propagation for KD eval runs). `populate()` is a backfill/recovery tool that also runs legacy migrations, timestamp backfill, epoch_metrics backfill from Lightning CSVs, stale entry cleanup, and teacher_run backfill.
- SQLite: WAL mode + 5s busy timeout for concurrent SLURM jobs. Foreign keys enabled. Indices on `metrics(run_id)`, `metrics(run_id, scenario, metric_name)`, `epoch_metrics(run_id, epoch)`.
- Dashboard: Config-driven ES module architecture. Adding a visualization = adding an entry to `panelConfig.js`. `BaseChart` provides SVG/tooltip/responsive infrastructure; 8 chart types inherit from it. `PanelManager` reads config → builds nav + panels + controls → lazy-loads data → renders. All chart types registered in `Registry`.
- Dual storage: Snakemake owns filesystem paths (DAG triggers), project DB owns structured results (write-through from cli.py). Dashboard exports static JSON for GitHub Pages.
- Data layer: Parquet (columnar storage) + SQLite (project DB) + Datasette (interactive browsing). All serverless.
- Dataset catalog: `config/datasets.yaml` — single place to register new datasets.
- Delete unused code completely. No compatibility shims or `# removed` comments.

## Environment

- **Cluster**: OSC (Ohio Supercomputer Center), RHEL 9, SLURM
- **GPU**: V100 (account PAS3209, gpu partition)
- **Python**: conda `gnn-experiments` (PyTorch, PyG, Lightning, Pydantic v2, Datasette)
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Project DB**: `data/project.db` (SQLite — datasets, runs, metrics, epoch_metrics)
- **Dashboard**: https://robertfrenken.github.io/DQN-Fusion/ (GitHub Pages from `docs/`)

## Session Modes

Switch Claude's focus with `/set-mode <mode>`:

| Mode | Focus | Suppressed |
|------|-------|------------|
| `mlops` | Pipeline, Snakemake, SLURM, config, debugging | Research, writing |
| `research` | OOD generalization, JumpReLU, cascading KD, literature | Pipeline ops, config |
| `writing` | Paper drafting, documentation, results | Code changes, pipeline |
| `data` | Ingestion, preprocessing, validation, cache | Research, writing |

Mode context files live in `.claude/system/modes/`. Switching modes loads the relevant context into the conversation.

## Skills

| Skill | Usage | Description |
|-------|-------|-------------|
| `/set-mode` | `/set-mode mlops` | Switch session focus mode |
| `/run-pipeline` | `/run-pipeline hcrl_sa large` | Submit Snakemake jobs to SLURM |
| `/check-status` | `/check-status hcrl_sa` | Check SLURM queue, checkpoints, DB |
| `/run-tests` | `/run-tests` or `/run-tests test_config` | Run pytest suite |
| `/sync-state` | `/sync-state` | Update STATE.md from current outputs |

## Experiment Tracking

All tracking uses the project SQLite DB (`data/project.db`) as the single source of truth:
- **Write-through**: `cli.py` records run start/end + metrics directly to DB
- **Epoch metrics**: `epoch_metrics` table captures per-epoch training curves (18,290 rows backfilled from 44 Lightning CSVs)
- **Backfill**: `python -m pipeline.db populate` scans filesystem + runs migrations (legacy naming, stale entry cleanup, timestamps, epoch_metrics from Lightning CSVs, teacher_run)
- **Artifacts**: Evaluation stage captures `embeddings.npz` (VGAE z-mean + GAT hidden layers) and `dqn_policy.json` (alpha values by class) — requires re-running evaluation
- **Dashboard**: `python -m pipeline.export` generates static JSON (leaderboard, metrics, training curves, KD transfer, datasets, runs, graph_samples, model_sizes, embeddings, dqn_policy); `scripts/export_dashboard.sh` commits + pushes to GitHub Pages. Auto-runs in Snakemake `onsuccess`.
- **Interactive**: `datasette data/project.db` for ad-hoc SQL browsing

## Detailed Documentation

- `.claude/system/PROJECT_OVERVIEW.md` -- Full architecture, models, memory optimization
- `.claude/system/CONVENTIONS.md` -- Code style, iteration hygiene, git rules
- `.claude/system/STATE.md` -- Current session state (updated each session)
- `.claude/system/modes/` -- Mode-specific context files (mlops, research, writing, data)
- `docs/user_guides/` -- Snakemake guide, Datasette usage, terminal setup
- `docs/dashboard/` -- D3.js dashboard (GitHub Pages)
