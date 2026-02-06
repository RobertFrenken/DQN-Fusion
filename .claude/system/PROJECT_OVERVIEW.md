# CAN-Graph KD-GAT: Project Context

**Updated**: 2026-02-05

## What This Is

CAN bus intrusion detection via knowledge distillation. Teacher models (VGAE → GAT → DQN fusion) are compressed into lightweight students. Runs on OSC HPC via Snakemake/SLURM.

## Architecture

```
VGAE (unsupervised reconstruction) → GAT (supervised classification) → DQN (RL fusion of both)
                                          ↑
                                     EVALUATION (all models)
```

**Entry point**: `python -m pipeline.cli <stage> [options]`

## Active System: `pipeline/`

Clean, self-contained module. Frozen dataclasses + JSON config. No Hydra, no Pydantic.

- `config.py` — `PipelineConfig` frozen dataclass, presets, JSON save/load
- `stages/` — Training logic split into modules:
  - `training.py` — VGAE (autoencoder) and GAT (curriculum) training
  - `fusion.py` — DQN fusion training
  - `evaluation.py` — Multi-model evaluation and metrics
  - `modules.py` — PyTorch Lightning modules
  - `utils.py` — Shared utilities (teacher loading, config loading)
- `paths.py` — Canonical directory layout, checkpoint/config paths
- `validate.py` — Config validation
- `cli.py` — Arg parser, MLflow run lifecycle, `STAGE_FNS` dispatch
- `tracking.py` — MLflow integration: `setup_tracking()`, `start_run()`, `end_run()`, `log_failure()`
- `query.py` — CLI for querying MLflow experiments
- `memory.py` — Memory monitoring and GPU/CPU optimization
- `Snakefile` — Snakemake workflow (19 rules, 2-level paths, configurable DATASETS, onstart MLflow init)
- `snakemake_config.yaml` — Pipeline-level Snakemake config
- `profiles/slurm/config.yaml` — SLURM cluster submission profile for Snakemake
- `profiles/slurm/status.sh` — Job status checker (sacct-based)

## Supporting Code: `src/`

`pipeline/stages/` imports from these `src/` modules:
- `src.models.vgae`, `src.models.models`, `src.models.dqn` — model architectures
- `src.training.datamodules` — `load_dataset()`, `CANGraphDataModule`
- `src.preprocessing.preprocessing` — `GraphDataset`, graph construction

`load_dataset()` accepts direct `Path` arguments from `pipeline/paths.py`. No legacy adapters remain.

## Data Pipeline

```
data/automotive/{dataset}/train_*/  →  data/cache/{dataset}/processed_graphs.pt
     (raw CSVs, DVC-tracked)              (PyG Data objects, DVC-tracked)
                                          + id_mapping.pkl
                                          + cache_metadata.json
```

- 6 datasets: hcrl_ch, hcrl_sa, set_01-04
- Cache auto-built on first access, validated via metadata on subsequent loads
- All data versioned with DVC (remote: `/fs/scratch/PAS1266/can-graph-dvc`)

## Models

| Model | File | Teacher | Student |
|-------|------|---------|---------|
| `GraphAutoencoderNeighborhood` | `src/models/vgae.py` | (1024,512,96) latent 96 | (80,40,16) latent 16 |
| `GATWithJK` | `src/models/models.py` | hidden 64, 5 layers, 8 heads | hidden 24, 2 layers, 4 heads |
| `EnhancedDQNFusionAgent` | `src/models/dqn.py` | hidden 576, 3 layers | hidden 160, 2 layers |

DQN state: 15D vector (VGAE 8D: errors + latent stats + confidence; GAT 7D: logits + embedding stats + confidence).

## Memory Optimization

Default config enables memory-efficient training:
- `gradient_checkpointing: True` — 30-50% activation memory savings (~20% compute overhead)
- `precision: "16-mixed"` — 50% model/activation memory reduction
- Both `GATWithJK` and `GraphAutoencoderNeighborhood` support checkpointing via `use_checkpointing` flag

Memory monitoring logs to MLflow every N epochs:
- CPU: mem_percent, mem_used_gb, mem_available_gb
- GPU: mem_allocated_gb, mem_reserved_gb, mem_max_allocated_gb

## Critical Constraints

**Do not violate these — they fix real crashes:**

- **PyG `Data.to()` is in-place.** Always `.clone().to(device)`, never `.to(device)` on shared data. Mutating training data before DataLoader fork causes CUDA errors.
- **Use spawn multiprocessing.** `mp_start_method: "spawn"` in config, `mp.set_start_method('spawn', force=True)` in CLI. Fork + CUDA = crashes.
- **DataLoader workers**: `multiprocessing_context='spawn'` on all DataLoader instances.
- **NFS filesystem**: `.nfs*` ghost files appear when processes delete open files. Already in `.gitignore`. Git operations (stash, reset) can fail on them.
- **No GUI on HPC**: Git auth via SSH key (configured), not HTTPS tokens. Avoid `gnome-ssh-askpass`.

## Experiment Management

**Dual-system architecture**: Snakemake owns the filesystem (DAG orchestration), MLflow owns the metadata (tracking/UI).

**Filesystem** (NFS home, permanent — Snakemake-managed):
```
experimentruns/{dataset}/{model_size}_{stage}[_kd]/
├── best_model.pt       # Snakemake DAG trigger
├── config.json         # Frozen config (also logged to MLflow)
├── metrics.json        # Evaluation stage only
```

**MLflow** (GPFS scratch, 90-day purge — supplementary):
```
/fs/scratch/PAS1266/kd_gat_mlflow/
├── mlflow.db           # SQLite tracking DB
```

Snakemake needs deterministic file paths at DAG construction time. MLflow artifact paths contain run UUIDs (not deterministic). So models save to filesystem first (Snakemake), then log to MLflow (tracking). If scratch purges, checkpoints/configs survive on NFS.

**MLflow integration**: Fully wired. `cli.py` wraps stage dispatch with `start_run`/`end_run`. `stages.py` uses `mlflow.pytorch.autolog()` for Lightning stages and manual `log_metrics()` for DQN + evaluation. See `docs/registry_plan.md` for full design history.

## Environment

- **Cluster**: Ohio Supercomputer Center (OSC), RHEL 9, SLURM scheduler
- **Home**: `/users/PAS2022/rf15/` — NFS v4, 1.7TB — permanent, safe for checkpoints
- **Scratch**: `/fs/scratch/PAS1266/` — GPFS (IBM Spectrum Scale), 90-day purge — safe for concurrent DB writes
- **Git remote**: `git@github.com:RobertFrenken/DQN-Fusion.git` (SSH)
- **Python**: conda `gnn-experiments`, PyTorch + PyG + Lightning
- **Key packages**: SQLite 3.51.1, Pandas 2.3.3, MLflow 3.8.1
- **SLURM account**: PAS3209, gpu partition, V100 GPUs
- **tmux**: 3.2a available on login nodes -- use for Snakemake orchestration and Claude Code sessions
- **Jupyter**: Available via OSC OnDemand portal
- **MLflow UI**: Available via OSC OnDemand app (`bc_osc_mlflow`)
