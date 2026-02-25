# KD-GAT: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection using a 3-stage knowledge distillation pipeline:
VGAE (unsupervised reconstruction) → GAT (supervised classification) → DQN (RL fusion).
Large models are compressed into small models via KD auxiliaries for edge deployment.

## Key Commands

```bash
# Run a single stage
python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>

# Stages: autoencoder, curriculum, normal, fusion, evaluation, temporal
# Models: vgae, gat, dqn
# Scales: large, small
# Auxiliaries: none (default), kd_standard

# Examples
python -m pipeline.cli autoencoder --model vgae --scale large --dataset hcrl_sa
python -m pipeline.cli curriculum --model gat --scale small --auxiliaries kd_standard --teacher-path <path> --dataset hcrl_sa
python -m pipeline.cli fusion --model dqn --scale large --dataset hcrl_ch
python -m pipeline.cli temporal --model gat --scale large --dataset hcrl_sa -O temporal.enabled true

# Override nested config values
python -m pipeline.cli autoencoder --model vgae --scale large -O training.lr 0.001 -O vgae.latent_dim 16

# Advanced: trial-based batch sizing, GNNExplainer
python -m pipeline.cli curriculum --model gat --scale large --dataset hcrl_sa -O training.memory_estimation trial
python -m pipeline.cli evaluation --model gat --scale large --dataset hcrl_sa -O training.run_explainer true

# cuGraph decision gate profiling
sbatch scripts/profile_conv_type.sh hcrl_sa large
python scripts/analyze_profile.py --dataset hcrl_sa --scale large

# Full pipeline via Ray + SLURM
python -m pipeline.cli flow --dataset hcrl_sa
python -m pipeline.cli flow --dataset hcrl_sa --scale large
python -m pipeline.cli flow --eval-only --dataset hcrl_sa
sbatch scripts/ray_slurm.sh flow --dataset hcrl_sa

# Local execution (no SLURM)
python -m pipeline.cli flow --dataset hcrl_sa --local

# Test graph caching (builds test_*.pt per scenario per dataset)
sbatch scripts/build_test_cache.sh                    # All datasets (set_01-04)
sbatch scripts/build_test_cache.sh set_02 set_03      # Specific datasets

# Export dashboard data (filesystem → static JSON)
python -m pipeline.export                                    # Default: docs/dashboard/data/
python -m pipeline.export --skip-heavy                       # Light exports only (~2s, login node OK)
python -m pipeline.export --only-heavy                       # Heavy exports only (UMAP, attention, graphs)
python -m pipeline.export --output-dir docs/dashboard/data   # Explicit output dir
bash scripts/export_dashboard.sh              # Export + commit + push to Pages + DVC push
bash scripts/export_dashboard.sh --no-push    # Export + commit only
bash scripts/export_dashboard.sh --dry-run    # Export only (no git)
sbatch scripts/export_dashboard_slurm.sh      # Heavy exports via SLURM (cpu partition, 16GB)

# Build analytics DuckDB (materialized view over all experiment data)
python -m pipeline.build_analytics              # Full rebuild (~1s)
python -m pipeline.build_analytics --dry-run    # Show what would be built
duckdb data/lakehouse/analytics.duckdb          # Interactive queries

# Run tests — ALWAYS submit to SLURM, never on login node
bash scripts/run_tests_slurm.sh                         # all tests
bash scripts/run_tests_slurm.sh -k "test_full_pipeline"  # specific test
bash scripts/run_tests_slurm.sh -m slurm                 # only slurm-marked

# Hyperparameter sweeps via Ray Tune
# See pipeline/orchestration/tune_config.py for search spaces
# Legacy parallel-command-processor approach still works:
python scripts/generate_sweep.py \
  --stage autoencoder --model vgae --scale large --dataset hcrl_sa \
  --sweep "training.lr=0.001,0.0005" "vgae.latent_dim=8,16,32" \
  --output /tmp/sweep_commands.txt
sbatch scripts/sweep.sh /tmp/sweep_commands.txt

# FastAPI inference server
uvicorn pipeline.serve:app --host 0.0.0.0 --port 8000

# Check SLURM jobs
squeue -u $USER

# Docs site (Astro) — requires: module load node-js/22.12.0
cd docs-site && npm run dev      # Dev server at localhost:4321 (--host 0.0.0.0 for SSH tunnel)
cd docs-site && npm run build    # Static build → docs-site/dist/
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
  cli.py            # Entry point + W&B init + lakehouse sync + archive restore on failure
  serve.py          # FastAPI inference server (/predict, /health)
  stages/           # Stage implementations (training, fusion, evaluation, temporal)
    evaluation.py   # Multi-model eval; captures embeddings.npz + dqn_policy.json + explanations.npz
    temporal.py     # Temporal graph classification (GAT encoder + Transformer over time)
  orchestration/    # Ray orchestration (ray_pipeline, ray_slurm, tune_config)
  tracking.py       # Memory monitoring utilities
  export.py         # Filesystem → static JSON export for dashboard
  memory.py         # GPU memory management (static, measured, trial-based batch sizing)
  lakehouse.py      # Fire-and-forget sync to S3 lakehouse
src/                # Layer 3: Domain (models, training, preprocessing; imports config/)
  models/           # vgae.py, gat.py, dqn.py, temporal.py
  explain.py        # GNNExplainer integration (feature importance analysis)
  training/         # load_dataset(), load_test_scenarios(), graph caching
  preprocessing/    # Graph construction from CAN CSVs + temporal.py (TemporalGrouper)
data/
  automotive/       # 6 datasets (DVC-tracked): hcrl_ch, hcrl_sa, set_01-04
  cache/            # Preprocessed graph cache (.pt, .pkl, metadata)
experimentruns/     # Outputs: best_model.pt, config.json, metrics.json, embeddings.npz, dqn_policy.json, explanations.npz
scripts/            # Automation (export_dashboard.sh, export_dashboard_slurm.sh, run_tests_slurm.sh, build_test_cache.sh, sweep.sh, generate_sweep.py)
docs/dashboard/     # GitHub Pages D3.js dashboard (ES modules, config-driven panels)
  js/core/          # BaseChart, Registry, Theme
  js/charts/        # 8 chart types (Table, Bar, Scatter, Line, Timeline, Bubble, ForceGraph, Histogram)
  js/panels/        # PanelManager + panelConfig (11 panels, declarative)
  js/app.js         # Slim entry point
  data/             # Static JSON exports from pipeline
docs-site/          # Astro 5 + Svelte 5 interactive research paper site
  src/components/   # D3Chart.svelte (generic D3), PlotFigure.svelte (Observable Plot), FigureIsland.astro, D3Scatter.svelte (legacy)
  src/components/figures/  # Interactive figure islands (EmbeddingsFigure, TrainingCurvesFigure, RocCurvesFigure)
  src/config/       # figures.ts (paper figure registry), shared.ts (ChartType, LayoutWidth)
  src/content.config.ts  # Astro Content Collections (Zod schemas for catalog JSON)
  src/data/         # Catalog JSON (synced from docs/dashboard/data/ by scripts/sync-data.sh)
  src/layouts/      # ArticleLayout.astro (CSS Grid Distill-style layout, KaTeX, chart styles)
  src/lib/d3/       # All 11 D3 chart classes + BaseChart + Theme + ThemeLight
  src/lib/data.ts   # Typed fetch helpers for per-run data (envelope unwrapping)
  src/lib/resource.svelte.ts  # Reactive fetch-on-state-change for Svelte 5
  src/pages/        # index.astro, showcase.astro, test-figure.mdx
  scripts/sync-data.sh  # Sync dashboard data → src/data/ + public/data/ symlink
  public/data/      # Symlink → docs/dashboard/data/ (runtime fetch for per-run data)
  astro.config.mjs  # Astro config (svelte + mdx + remark-math + rehype-katex)
  package.json      # Node.js dependencies (astro, svelte, d3, @observablehq/plot, remark-math, rehype-katex)
notebooks/          # Deno Jupyter notebooks for prototyping plots
  deno.json         # Deno config (imports: @observablehq/plot, d3, canvas)
  deno_plot_template.ipynb  # Observable Plot prototyping template
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
- **Never run pytest on login nodes.** Always submit via `bash scripts/run_tests_slurm.sh`. Login nodes have strict resource limits and will crash.
- **Dynamic batching for variable-size graphs.** DynamicBatchSampler (PyG built-in) packs graphs to a node budget instead of a fixed count. Budget = batch_size × p95_nodes (from cache_metadata.json). Keeps GPU ~85% utilized regardless of graph size variance. Disable with `-O training.dynamic_batching false` if needed for reproducibility comparisons.

## Architecture Decisions

- **3-layer import hierarchy** (enforced by `tests/test_layer_boundaries.py`):
  - `config/` → never imports from `pipeline/` or `src/`
  - `pipeline/` → imports `config/` at top level; imports `src/` only inside functions (lazy)
  - `src/` → imports `config/` (constants); never imports from `pipeline/`
- Config: Pydantic v2 frozen BaseModels + YAML composition + JSON serialization.
- Sub-configs: `cfg.vgae`, `cfg.gat`, `cfg.dqn`, `cfg.training`, `cfg.fusion`, `cfg.temporal` — nested Pydantic models. Always use nested access (`cfg.vgae.latent_dim`), never flat.
- Auxiliaries: `cfg.auxiliaries` is a list of `AuxiliaryConfig`. KD is a composable loss modifier, not a model identity. Use `cfg.has_kd` / `cfg.kd` properties.
- Constants: domain/infrastructure constants live in `config/constants.py` (not in PipelineConfig). Hyperparameters live in PipelineConfig.
- Experiment tracking: W&B (online/offline) for live metrics + S3 lakehouse for structured JSON. `cli.py` owns `wandb.init()`/`wandb.finish()` lifecycle; Lightning's `WandbLogger` attaches to the active run. Compute nodes auto-set `WANDB_MODE=offline`.
- Orchestration: Ray (`pipeline/orchestration/`) with `@ray.remote` tasks. `train_pipeline()` fans out per-dataset work concurrently via `dataset_pipeline` remote functions; `eval_pipeline()` re-runs evaluation only. Each stage task dispatches via subprocess for clean CUDA context. `--local` flag uses Ray local mode. HPO via Ray Tune with OptunaSearch + ASHAScheduler.
- Archive restore: `cli.py` archives previous runs before re-running, and restores the archive if the new run fails.
- Inference serving: `pipeline/serve.py` provides FastAPI endpoints (`/predict`, `/health`) loading VGAE+GAT+DQN from `experimentruns/`.
- **Docs site (Astro + Svelte)**: Astro 5 static site with Svelte 5 islands for interactive figures. Dual renderer architecture: `figures.ts` registry (with `renderer: 'plot' | 'd3'` per figure) → `FigureIsland.astro` (layout/caption) → either `PlotFigure.svelte` (Observable Plot) or `D3Chart.svelte` (D3, `import.meta.glob` registry → `BaseChart` → chart class). All 11 D3 chart types adapted with `import * as d3 from 'd3'`. Hybrid data pipeline: Content Collections with Zod schemas for catalog data (build-time), client-side fetch for per-run data (runtime). CSS Grid layout (Distill-inspired: `.l-body`/`.l-wide`/`.l-full`/`.l-margin`). KaTeX for math (server-side via remark-math + rehype-katex). Interactive figures (embeddings, training curves, ROC) use Svelte 5 runes for state + `resource.svelte.ts` for reactive fetch. Deno Jupyter notebooks (`notebooks/deno_plot_template.ipynb`) for prototyping Observable Plot figures. Run `scripts/sync-data.sh` after `pipeline/export.py` to update site data. Deploy target TBD (Cloudflare Pages).
- Dashboard: Config-driven ES module architecture. Adding a visualization = adding an entry to `panelConfig.js`. `BaseChart` provides SVG/tooltip/responsive infrastructure; 8 chart types inherit from it. `PanelManager` reads config → builds nav + panels + controls → lazy-loads data → renders. All chart types registered in `Registry`.
- Dashboard data: `export.py` scans `experimentruns/` filesystem for `config.json` and `metrics.json` files. No database dependency. Artifacts (`embeddings.npz`, `dqn_policy.json`, etc.) are read directly from run directories. `--skip-heavy` runs light exports (~2s, safe on login node); `--only-heavy` runs CPU-intensive exports (UMAP, attention, graph samples) via SLURM. Dashboard JS fetches from `s3://kd-gat/dashboard/` with `data/` fallback for local dev.
- Dataset catalog: `config/datasets.yaml` — single place to register new datasets.
- Delete unused code completely. No compatibility shims or `# removed` comments.

## Environment

- **Cluster**: OSC (Ohio Supercomputer Center), RHEL 9, SLURM
- **GPU**: V100 (account PAS3209, gpu partition)
- **Python**: 3.12 (OSC `module load python/3.12`), uv venv `.venv/`
- **Package manager**: uv 0.10+ (installed at `~/.local/bin/uv`)
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Tracking**: W&B (project `kd-gat`) + S3 lakehouse (JSON on `s3://kd-gat/lakehouse/`)
- **Node.js**: 22.12.0 via `module load node-js/22.12.0` (npm 10.9.0). Used for docs-site only.
- **Dashboard**: https://robertfrenken.github.io/DQN-Fusion/ (GitHub Pages from `docs/`)

## Environment Variables

| Variable | Location | Used By |
|----------|----------|---------|
| `GH_TOKEN` | `~/.env.local` | gh CLI, GitHub API |
| `CLOUDFLARE_ACCOUNT_ID` | `~/.env.local` | DVC R2 remote |
| `CLOUDFLARE_R2_TOKEN` | `~/.env.local` | DVC R2 remote |
| `CLOUDFLARE_R2_ACCESS_KEY_ID` | `~/.env.local` | DVC R2 remote, lakehouse.py |
| `CLOUDFLARE_R2_SECRET_ACCESS_KEY` | `~/.env.local` | DVC R2 remote, lakehouse.py |
| `CLOUDFLARE_R2_ENDPOINT` | `~/.env.local` | DVC R2 remote, lakehouse.py |
| `AWS_ACCESS_KEY_ID` | `~/.env.local` | AWS S3 (if used) |
| `AWS_SECRET_ACCESS_KEY` | `~/.env.local` | AWS S3 (if used) |
| `KD_GAT_S3_BUCKET` | `.env` | lakehouse.py, export scripts |
| `KD_GAT_SLURM_ACCOUNT` | `.env` | SLURM job scripts |
| `KD_GAT_SLURM_PARTITION` | `.env` | SLURM job scripts |
| `KD_GAT_GPU_TYPE` | `.env` | SLURM job scripts |
| `KD_GAT_SCRATCH` | `.env` | Scratch path for temp data |
| `KD_GAT_PYTHON` | `.env` | Python command in scripts |
| `KD_GAT_DATA_ROOT` | `.env` | Root for raw data + caches (default: ~/kd-gat-data) |
| `KD_GAT_CACHE_ROOT` | `.env` | Cache override (default: $KD_GAT_DATA_ROOT/cache) |
| `WANDB_API_KEY` | `~/.env.local` or `wandb login` | W&B experiment tracking |
| `WANDB_MODE` | Auto-set on compute nodes | W&B offline/online toggle |

## uv + PyTorch + PyG on OSC — Version Compatibility

This stack has a **three-way version coupling** that must stay in sync. Getting any axis wrong causes segfaults (not import errors — silent C++ ABI mismatches).

### The constraint triangle

```
PyTorch version ←→ PyG extension wheels (torch-scatter, torch-sparse, torch-cluster)
      ↕
  CUDA version
```

1. **PyTorch from PyPI** bundles NVIDIA libs (cudnn, cublas, etc.) automatically. Do NOT use the `download.pytorch.org/whl/cu*` index — those are "lean" wheels that expect system CUDA, which OSC doesn't fully provide (e.g., no cuDNN 9).

2. **PyG extensions** are compiled against a specific `torch+cu` combo. Wheels live at `https://data.pyg.org/whl/torch-{VERSION}+cu{CUDA}.html`. The `torch-geometric` package itself is on PyPI; only the C++ extensions (scatter, sparse, cluster) need the flat index.

3. **PyTorch version on PyPI ≠ PyG's torch version tag.** PyPI may ship torch 2.10.0 but PyG only has wheels up to torch 2.8.0. Installing mismatched versions compiles fine but **segfaults at runtime**.

### Current pinned versions (2026-02-22)

| Component | Version | Source |
|-----------|---------|--------|
| Python | 3.12.4 | OSC `module load python/3.12` |
| PyTorch | 2.8.0 (bundled cu128) | PyPI (default index) |
| torch-scatter | 2.1.2+pt28cu126 | `data.pyg.org/whl/torch-2.8.0+cu126.html` |
| torch-sparse | 0.6.18+pt28cu126 | same flat index |
| torch-cluster | 1.6.3+pt28cu126 | same flat index |
| torch-geometric | 2.7.0 | PyPI |
| RAPIDS (optional) | 24.12+ | Separate conda env (no pip wheels) |

### How to upgrade torch

1. Check `https://data.pyg.org/whl/` for the new torch version's wheel page
2. Verify the page returns HTTP 200 and has `cp312-linux_x86_64` wheels
3. Update `pyproject.toml`:
   - `torch>=X.Y.0,<X.Z` in `[project.dependencies]`
   - PyG flat index URL in `[[tool.uv.index]]`
4. `rm uv.lock && uv sync --extra dev`
5. Verify: `.venv/bin/python -c "import torch; import torch_scatter; print('OK')"`

### Traps to avoid

- **`torch>=2.6`** without an upper bound: PyPI resolves to latest (2.10+), which has no PyG wheels. Always pin `<next_major`.
- **`requires-python = ">=3.10"`** without upper bound: uv resolves for all supported Pythons including 3.13, where PyG wheels may not exist. Use `<3.13`.
- **uv-managed Python downloads**: Standalone Python builds from uv can segfault on OSC's RHEL 9. Always use OSC's system Python via `uv venv --python /apps/python/3.12/bin/python3`.
- **`[tool.uv] find-links`** for PyG: Use `[[tool.uv.index]]` with `format = "flat"` and `explicit = true` instead. `find-links` doesn't support the `explicit` flag, so non-PyG packages may accidentally resolve from the flat index.
- **OSC CUDA modules** (cuda/12.6, cudnn/8.x): Not needed when torch comes from PyPI (nvidia-* pip packages bundle everything). Only load modules for RAPIDS or custom CUDA code.

## Skills

| Skill | Usage | Description |
|-------|-------|-------------|
| `/run-pipeline` | `/run-pipeline hcrl_sa large` | Submit Ray pipeline to SLURM |
| `/check-status` | `/check-status hcrl_sa` | Check SLURM queue, checkpoints, W&B |
| `/run-tests` | `/run-tests` or `/run-tests test_config` | Run pytest suite |
| `/sync-state` | `/sync-state` | Update STATE.md from current outputs |

## Experiment Tracking

Tracking uses W&B + S3 lakehouse + analytics DuckDB:
- **W&B**: `cli.py` owns `wandb.init()`/`wandb.finish()` lifecycle. Lightning's `WandbLogger` attaches to the active run for per-epoch metrics. Compute nodes auto-set `WANDB_MODE=offline`; sync offline runs via `wandb sync wandb/run-*`.
- **Local Lakehouse**: `pipeline/lakehouse.py` writes per-run JSON to `data/lakehouse/runs/` (NFS, canonical). S3 sync is fire-and-forget backup to `s3://kd-gat/lakehouse/runs/`. NFS wins if they disagree.
- **Analytics DuckDB**: `pipeline/build_analytics.py` materializes `data/lakehouse/analytics.duckdb` from lakehouse JSON + `experimentruns/` filesystem. Tables: `runs`, `metrics`, `datasets`, `configs`. Replaces `data/project.db` (deprecated).
- **Artifacts**: Evaluation stage captures `embeddings.npz` (VGAE z-mean + GAT hidden layers), `dqn_policy.json` (alpha values by class), and optionally `explanations.npz` (GNNExplainer feature importance when `run_explainer=True`) — stored in run directories.
- **Dashboard**: `python -m pipeline.export` scans `experimentruns/` filesystem to generate static JSON (leaderboard, metrics, training curves, KD transfer, datasets, runs, graph_samples, model_sizes, embeddings, dqn_policy, explanations); `scripts/export_dashboard.sh` commits + pushes to GitHub Pages + syncs to S3 + DVC push.

## Cross-Repo Context

This project interacts with two other repos. Changes in one may require updates in the others.

| Repo | Location | Role |
|------|----------|------|
| **KD-GAT** (this) | `~/KD-GAT` | ML research project — source of tested version pins |
| **dotfiles** | `~/dotfiles` | Shell config, aliases, `~/.env.local` secret management |
| **lab-setup-guide** | `~/lab-setup-guide` | MSL lab onboarding docs — canonical OSC procedures |

**Propagation triggers:**
- Module version change (python/X.Y, cuda) → update dotfiles `dot_bashrc.tmpl` + lab guide docs
- New env var added → document in dotfiles CLAUDE.md env.local section + `.env.example`
- PyTorch/PyG version pin change → update lab guide `pytorch-setup.md` + `pyg-setup.md`
- uv workflow change → update lab guide `osc-environment-management.md`

**Read FROM (authoritative sources):**
- Lab guide → canonical OSC procedures, module system docs
- Dotfiles → `~/.env.local` pattern, shared aliases

## Documentation Sources

| Topic | URL |
|-------|-----|
| uv (package manager) | https://docs.astral.sh/uv/ |
| uv + PyTorch guide | https://docs.astral.sh/uv/guides/integration/pytorch/ |
| PyTorch | https://pytorch.org/docs/stable/ |
| PyTorch Geometric | https://pytorch-geometric.readthedocs.io/en/latest/ |
| PyG Installation | https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html |
| PyTorch Lightning | https://lightning.ai/docs/pytorch/stable/ |
| Pydantic v2 | https://docs.pydantic.dev/latest/ |
| W&B (Weights & Biases) | https://docs.wandb.ai/ |
| Ray | https://docs.ray.io/en/latest/ |
| Ray Tune | https://docs.ray.io/en/latest/tune/index.html |
| OSC Documentation | https://www.osc.edu/resources/technical_support/supercomputers/pitzer |
| SLURM | https://slurm.schedmd.com/documentation.html |
| Astro | https://docs.astro.build/ |
| Svelte 5 | https://svelte.dev/docs |
| D3.js | https://d3js.org/ |
| Claude Code | https://docs.anthropic.com/en/docs/claude-code/ |
| DuckDB | https://duckdb.org/docs/ |
| Ruff | https://docs.astral.sh/ruff/ |

## Detailed Documentation

- `.claude/system/PROJECT_OVERVIEW.md` -- Full architecture, models, memory optimization
- `.claude/system/CONVENTIONS.md` -- Code style, iteration hygiene, git rules
- `.claude/system/STATE.md` -- Current session state (updated each session)
- `docs/user_guides/` -- Terminal setup, pipeline guides
- `docs/dashboard/` -- D3.js dashboard (GitHub Pages)
