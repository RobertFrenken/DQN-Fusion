# CAN-Graph KD-GAT: Project Context

**Updated**: 2026-02-15

## What This Is

CAN bus intrusion detection via knowledge distillation. Large models (VGAE → GAT → DQN fusion) are compressed into small models via KD auxiliaries. Runs on OSC HPC via Snakemake/SLURM.

## Architecture

```
VGAE (unsupervised reconstruction) → GAT (supervised classification) → DQN (RL fusion of both)
                                          ↑
                                     EVALUATION (all models)
```

**Entry point**: `python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`

## Layered Architecture

Three-layer import hierarchy (enforced by `tests/test_layer_boundaries.py`):

### Layer 1: `config/` (inert, declarative — no pipeline/ or src/ imports)

- `schema.py` — Pydantic v2 frozen models: `PipelineConfig`, `VGAEArchitecture`, `GATArchitecture`, `DQNArchitecture`, `AuxiliaryConfig`, `TrainingConfig`, `FusionConfig`, `PreprocessingConfig`. Legacy flat JSON loading via `_from_legacy_flat()` (for old config.json files — all dirs now use new naming).
- `resolver.py` — YAML composition: `resolve(model_type, scale, auxiliaries="none", **cli_overrides)`, `list_models()`, `list_auxiliaries()`. Merge order: defaults → model_def → auxiliaries → CLI.
- `paths.py` — Path layout: `{dataset}/{model_type}_{scale}_{stage}[_{aux}]`. String-based variants for Snakefile.
- `constants.py` — Domain/infrastructure constants: feature counts, window sizes, DB paths, MLflow URI, memory limits
- `__init__.py` — Re-exports for clean `from config import PipelineConfig, resolve, checkpoint_path` usage
- `defaults.yaml` — Global baseline config values
- `datasets.yaml` — Dataset catalog (6 automotive datasets)
- `models/{vgae,gat,dqn}/{large,small}.yaml` — Architecture × Scale definitions
- `auxiliaries/{none,kd_standard}.yaml` — Loss modifier configs (composable)

### Layer 2: `pipeline/` (orchestration — imports config/ freely, lazy imports from src/)

- `cli.py` — Arg parser (`--model`/`--scale`/`--auxiliaries`), MLflow run lifecycle, write-through DB recording (propagates teacher_run for KD eval runs), `STAGE_FNS` dispatch
- `stages/` — Training logic split into modules:
  - `training.py` — VGAE (autoencoder) and GAT (curriculum) training
  - `fusion.py` — DQN fusion training (uses `cfg.dqn.*`, `cfg.fusion.*`)
  - `evaluation.py` — Multi-model evaluation and metrics
  - `modules.py` — PyTorch Lightning modules (uses `cfg.vgae.*`, `cfg.gat.*`, `cfg.training.*`)
  - `utils.py` — Shared utilities: model loading with cross-model path resolution (`_cross_model_path`, `_STAGE_MODEL_TYPE`), batch size optimization, trainer construction
- `validate.py` — Config validation (simplified — Pydantic handles field constraints)
- `tracking.py` — MLflow integration: `setup_tracking()`, `start_run()`, `end_run()`, `log_failure()`
- `memory.py` — Memory monitoring and GPU/CPU optimization
- `ingest.py` — CSV→Parquet ingestion, validation against `config/datasets.yaml`, dataset registration
- `db.py` — SQLite project DB (`data/project.db`): WAL mode + busy timeout + foreign keys, schema (model_type/scale/has_kd) with indices on metrics/epoch_metrics, write-through `record_run_start()`/`record_run_end()`, backfill `populate()` (includes `_migrate_legacy_runs()`, `_backfill_timestamps()`, `_backfill_teacher_run()`), CLI queries
- `analytics.py` — Post-run analysis: sweep, leaderboard, compare, config_diff, dataset_summary
- `migrate_paths.py` — Legacy path migration tool (`teacher_*/student_*` → new format). Also rewrites `teacher_path` in config.json files. Migration completed 2026-02-14 (70 dirs across 6 datasets).
- `Snakefile` — Snakemake workflow (`--model`/`--scale`/`--auxiliaries` CLI, configurable DATASETS, `sys.executable` for Python path, onstart MLflow init, onsuccess DB populate + MLflow backup, preprocessing cache rule, retries with resource scaling, evaluation group jobs)

### Layer 3: `src/` (domain — imports config.constants, never imports pipeline/)

## Model Registry

`src/models/registry.py` centralizes model construction and fusion feature extraction.

**Registered models** (order matters — determines 15-D DQN state layout):

| Model | Feature Dim | Extractor | Role |
|-------|-------------|-----------|------|
| `vgae` | 8-D | `VGAEFusionExtractor` | Errors + latent stats + confidence |
| `gat` | 7-D | `GATFusionExtractor` | Class probs + embedding stats + confidence |
| `dqn` | — | None | Consumes features (15-D state) |

**Usage**:
```python
from src.models import get, fusion_state_dim, extractors

entry = get("vgae")                  # ModelEntry(model_type, factory, extractor)
model = entry.factory(cfg, num_ids, in_ch)  # Construct model from config
dim = fusion_state_dim()             # 15 (sum of all extractor dims)
pairs = extractors()                 # [("vgae", ext), ("gat", ext)] in registration order
```

**Adding a new model**: Register a `ModelEntry` in `registry.py` with a factory function (lazy import to avoid circular deps) and an optional `FusionFeatureExtractor` implementation.

## Supporting Code: `src/`

`pipeline/stages/` imports from these `src/` modules:
- `src.models.vgae`, `src.models.gat`, `src.models.dqn` — model architectures
- `src.training.datamodules` — `load_dataset()`
- `src.preprocessing.preprocessing` — `GraphDataset`, graph construction

`load_dataset()` accepts direct `Path` arguments from `pipeline/paths.py`. No legacy adapters remain.

## Config System

Config defined by four orthogonal concerns: **model_type** (architecture), **scale** (capacity), **auxiliaries** (loss modifiers), **dataset**.

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

**Resolution order**: `defaults.yaml` → `models/{type}/{scale}.yaml` → `auxiliaries/{aux}.yaml` → CLI overrides → Pydantic validation → frozen.

**Cross-model loading**: `load_vgae(gat_cfg)` resolves to `vgae_*` paths via `_STAGE_MODEL_TYPE` mapping. Each stage has a canonical model owner (autoencoder→vgae, curriculum→gat, fusion→dqn).

## Data Pipeline

```
config/datasets.yaml                # Dataset catalog (source of truth for dataset metadata)
     ↓ (python -m pipeline.ingest)
data/automotive/{dataset}/train_*/  →  data/parquet/{domain}/{dataset}/*.parquet
     (raw CSVs, DVC-tracked)              (columnar, queryable via SQL)
                                    →  data/cache/{dataset}/processed_graphs.pt
                                          (PyG Data objects, DVC-tracked)
                                          + id_mapping.pkl
                                          + cache_metadata.json
                                    →  data/project.db
                                          (SQLite: datasets, runs, metrics tables)
```

- 6 datasets: hcrl_ch, hcrl_sa, set_01-04
- Cache auto-built on first access, validated via metadata on subsequent loads
- All data versioned with DVC (remote: `/fs/scratch/PAS1266/can-graph-dvc`)
- Project DB: write-through from `cli.py` (primary), backfill via `populate()` (recovery)

## Models

| Model | File | Large | Small | Ratio |
|-------|------|-------|-------|-------|
| `GraphAutoencoderNeighborhood` | `src/models/vgae.py` | (480,240,48) latent 48 | (80,40,16) latent 16 | ~4x |
| `GATWithJK` | `src/models/gat.py` | hidden 48, 3 layers, 8 heads, fc_layers 1 (343k) | hidden 24, 2 layers, 4 heads, fc_layers 2 (65k) | 5.3x |
| `EnhancedDQNFusionAgent` | `src/models/dqn.py` | hidden 576, 3 layers | hidden 160, 2 layers | ~13x |

DQN state: 15D vector (VGAE 8D: errors + latent stats + confidence; GAT 7D: logits + embedding stats + confidence).

## Memory Optimization

Default config enables memory-efficient training:
- `gradient_checkpointing: True` — 30-50% activation memory savings (~20% compute overhead)
- `precision: "16-mixed"` — 50% model/activation memory reduction
- Both `GATWithJK` and `GraphAutoencoderNeighborhood` support checkpointing via `use_checkpointing` flag

## Critical Constraints

**Do not violate these — they fix real crashes:**

- **PyG `Data.to()` is in-place.** Always `.clone().to(device)`, never `.to(device)` on shared data.
- **Use spawn multiprocessing.** `mp_start_method: "spawn"` in config, `mp.set_start_method('spawn', force=True)` in CLI. Fork + CUDA = crashes.
- **DataLoader workers**: `multiprocessing_context='spawn'` on all DataLoader instances.
- **NFS filesystem**: `.nfs*` ghost files appear when processes delete open files. Already in `.gitignore`.
- **No GUI on HPC**: Git auth via SSH key (configured), not HTTPS tokens.

## Experiment Management

**Three-layer architecture**: Snakemake owns the filesystem (DAG orchestration), MLflow owns live tracking (params/metrics/UI), project DB owns structured results (queryable SQL).

**Filesystem** (NFS home, permanent — Snakemake-managed):
```
experimentruns/{dataset}/{model_type}_{scale}_{stage}[_{aux}]/
├── best_model.pt       # Snakemake DAG trigger
├── config.json         # Frozen config (Pydantic JSON, also logged to MLflow)
├── metrics.json        # Evaluation stage only (also logged as MLflow artifact)
```

**MLflow** (GPFS scratch, 90-day purge — auto-backed up to `~/backups/`):
```
/fs/scratch/PAS1266/kd_gat_mlflow/
├── mlflow.db           # SQLite tracking DB
```

**Project DB** (NFS home, permanent):
```
data/project.db         # SQLite: datasets, runs, metrics tables
```

## Environment

- **Cluster**: Ohio Supercomputer Center (OSC), RHEL 9, SLURM scheduler
- **Home**: `/users/PAS2022/rf15/` — NFS v4, 1.7TB — permanent, safe for checkpoints
- **Scratch**: `/fs/scratch/PAS1266/` — GPFS (IBM Spectrum Scale), 90-day purge
- **Git remote**: `git@github.com:RobertFrenken/DQN-Fusion.git` (SSH)
- **Python**: conda env `gnn-experiments` (`module load miniconda3/24.1.2-py310 && conda activate gnn-experiments`)
- **Key packages**: SQLite, Pandas, MLflow, PyArrow, Datasette, Pandera, Pydantic v2
- **SLURM account**: PAS3209, gpu partition, V100 GPUs
