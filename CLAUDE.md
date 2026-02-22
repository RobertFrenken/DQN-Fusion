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

# Full pipeline via Prefect + SLURM
python -m pipeline.cli flow --dataset hcrl_sa
python -m pipeline.cli flow --dataset hcrl_sa --scale large
python -m pipeline.cli flow --eval-only --dataset hcrl_sa

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

# Run tests — ALWAYS submit to SLURM, never on login node
bash scripts/run_tests_slurm.sh                         # all tests
bash scripts/run_tests_slurm.sh -k "test_full_pipeline"  # specific test
bash scripts/run_tests_slurm.sh -m slurm                 # only slurm-marked

# Hyperparameter sweeps via parallel-command-processor
python scripts/generate_sweep.py \
  --stage autoencoder --model vgae --scale large --dataset hcrl_sa \
  --sweep "training.lr=0.001,0.0005" "vgae.latent_dim=8,16,32" \
  --output /tmp/sweep_commands.txt
sbatch scripts/sweep.sh /tmp/sweep_commands.txt

# FastAPI inference server
uvicorn pipeline.serve:app --host 0.0.0.0 --port 8000

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
  cli.py            # Entry point + W&B init + lakehouse sync + archive restore on failure
  serve.py          # FastAPI inference server (/predict, /health)
  stages/           # Stage implementations (training, fusion, evaluation)
    evaluation.py   # Multi-model eval; captures embeddings.npz + dqn_policy.json artifacts
  flows/            # Prefect orchestration (train_flow, eval_flow, slurm_config)
  tracking.py       # Memory monitoring utilities
  export.py         # Filesystem → static JSON export for dashboard
  memory.py         # GPU memory management
  lakehouse.py      # Fire-and-forget sync to S3 lakehouse
src/                # Layer 3: Domain (models, training, preprocessing; imports config/)
  models/           # vgae.py, gat.py, dqn.py
  training/         # load_dataset(), load_test_scenarios(), graph caching
  preprocessing/    # Graph construction from CAN CSVs
data/
  automotive/       # 6 datasets (DVC-tracked): hcrl_ch, hcrl_sa, set_01-04
  cache/            # Preprocessed graph cache (.pt, .pkl, metadata)
experimentruns/     # Outputs: best_model.pt, config.json, metrics.json, embeddings.npz, dqn_policy.json
scripts/            # Automation (export_dashboard.sh, export_dashboard_slurm.sh, run_tests_slurm.sh, build_test_cache.sh, sweep.sh, generate_sweep.py)
docs/dashboard/     # GitHub Pages D3.js dashboard (ES modules, config-driven panels)
  js/core/          # BaseChart, Registry, Theme
  js/charts/        # 8 chart types (Table, Bar, Scatter, Line, Timeline, Bubble, ForceGraph, Histogram)
  js/panels/        # PanelManager + panelConfig (11 panels, declarative)
  js/app.js         # Slim entry point
  data/             # Static JSON exports from pipeline
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
- Sub-configs: `cfg.vgae`, `cfg.gat`, `cfg.dqn`, `cfg.training`, `cfg.fusion` — nested Pydantic models. Always use nested access (`cfg.vgae.latent_dim`), never flat.
- Auxiliaries: `cfg.auxiliaries` is a list of `AuxiliaryConfig`. KD is a composable loss modifier, not a model identity. Use `cfg.has_kd` / `cfg.kd` properties.
- Constants: domain/infrastructure constants live in `config/constants.py` (not in PipelineConfig). Hyperparameters live in PipelineConfig.
- Experiment tracking: W&B (online/offline) for live metrics + S3 lakehouse for structured JSON. `cli.py` owns `wandb.init()`/`wandb.finish()` lifecycle; Lightning's `WandbLogger` attaches to the active run. Compute nodes auto-set `WANDB_MODE=offline`.
- Orchestration: Prefect flows (`pipeline/flows/`) with `dask-jobqueue` SLURMCluster. `train_pipeline()` fans out per-dataset work concurrently via `_dataset_pipeline` sub-flows; `eval_pipeline()` re-runs evaluation only. Each stage task dispatches via subprocess for clean CUDA context. `--local` flag uses local Dask cluster instead of SLURM.
- Archive restore: `cli.py` archives previous runs before re-running, and restores the archive if the new run fails.
- Inference serving: `pipeline/serve.py` provides FastAPI endpoints (`/predict`, `/health`) loading VGAE+GAT+DQN from `experimentruns/`.
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
- **Dashboard**: https://robertfrenken.github.io/DQN-Fusion/ (GitHub Pages from `docs/`)

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

## Session Modes

Switch Claude's focus with `/set-mode <mode>`:

| Mode | Focus | Suppressed |
|------|-------|------------|
| `mlops` | Pipeline, Prefect, SLURM, W&B, config, debugging | Research, writing |
| `research` | OOD generalization, JumpReLU, cascading KD, literature | Pipeline ops, config |
| `writing` | Paper drafting, documentation, results | Code changes, pipeline |
| `data` | Ingestion, preprocessing, validation, cache | Research, writing |

Mode context files live in `.claude/system/modes/`. Switching modes loads the relevant context into the conversation.

## Skills

| Skill | Usage | Description |
|-------|-------|-------------|
| `/set-mode` | `/set-mode mlops` | Switch session focus mode |
| `/run-pipeline` | `/run-pipeline hcrl_sa large` | Submit Prefect flow to SLURM |
| `/check-status` | `/check-status hcrl_sa` | Check SLURM queue, checkpoints, W&B |
| `/run-tests` | `/run-tests` or `/run-tests test_config` | Run pytest suite |
| `/sync-state` | `/sync-state` | Update STATE.md from current outputs |

## Experiment Tracking

Tracking uses W&B + S3 lakehouse:
- **W&B**: `cli.py` owns `wandb.init()`/`wandb.finish()` lifecycle. Lightning's `WandbLogger` attaches to the active run for per-epoch metrics. Compute nodes auto-set `WANDB_MODE=offline`; sync offline runs via `wandb sync wandb/run-*`.
- **S3 Lakehouse**: `pipeline/lakehouse.py` writes structured metrics as JSON to `s3://kd-gat/lakehouse/runs/{run_id}.json` via boto3 (fire-and-forget). Uses `KD_GAT_S3_BUCKET` env var (default: `kd-gat`). Queryable via DuckDB: `SELECT * FROM read_json('s3://kd-gat/lakehouse/runs/*.json')`.
- **Artifacts**: Evaluation stage captures `embeddings.npz` (VGAE z-mean + GAT hidden layers) and `dqn_policy.json` (alpha values by class) — stored in run directories.
- **Dashboard**: `python -m pipeline.export` scans `experimentruns/` filesystem to generate static JSON (leaderboard, metrics, training curves, KD transfer, datasets, runs, graph_samples, model_sizes, embeddings, dqn_policy); `scripts/export_dashboard.sh` commits + pushes to GitHub Pages + syncs to S3 + DVC push.

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
| Prefect | https://docs.prefect.io/v3/ |
| OSC Documentation | https://www.osc.edu/resources/technical_support/supercomputers/pitzer |
| SLURM | https://slurm.schedmd.com/documentation.html |
| Claude Code | https://docs.anthropic.com/en/docs/claude-code/ |
| DuckDB | https://duckdb.org/docs/ |
| Ruff | https://docs.astral.sh/ruff/ |

## Detailed Documentation

- `.claude/system/PROJECT_OVERVIEW.md` -- Full architecture, models, memory optimization
- `.claude/system/CONVENTIONS.md` -- Code style, iteration hygiene, git rules
- `.claude/system/STATE.md` -- Current session state (updated each session)
- `.claude/system/modes/` -- Mode-specific context files (mlops, research, writing, data)
- `docs/user_guides/` -- Terminal setup, pipeline guides
- `docs/dashboard/` -- D3.js dashboard (GitHub Pages)
