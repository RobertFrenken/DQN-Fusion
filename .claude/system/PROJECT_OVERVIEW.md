# CAN-Graph KD-GAT: Project Context

**Updated**: 2026-01-30

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
- `stages.py` — All training logic (VGAE, GAT, DQN, eval). Imports from `src/` conditionally (inside functions)
- `paths.py` — Canonical directory layout, checkpoint/config paths
- `validate.py` — Config validation
- `cli.py` — Arg parser → `STAGE_FNS` dispatch
- `Snakefile` — Snakemake workflow

## Legacy System: `src/`

Still required — `pipeline/stages.py` imports 5 modules from it:
- `src.models.vgae`, `src.models.models`, `src.models.dqn` — model architectures
- `src.training.datamodules` — `load_dataset()`, `CANGraphDataModule`
- `src.preprocessing.preprocessing` — `GraphDataset`, graph construction

**Dependency chain**: `load_dataset()` → `PathResolver` (src/paths.py, ~1200 lines, only ~100 needed). `pipeline/stages.py` uses a `_NS` adapter to bridge PipelineConfig to PathResolver's old interface. This is the main technical debt.

Other `src/` subdirs (`cli/`, `config/`, `evaluation/`, `utils/`) are not imported by pipeline and are candidates for removal (see `docs/CODEBASE_AUDIT.md`).

## Models

| Model | File | Teacher | Student |
|-------|------|---------|---------|
| `GraphAutoencoderNeighborhood` | `src/models/vgae.py` | (1024,512,96) latent 96 | (80,40,16) latent 16 |
| `GATWithJK` | `src/models/models.py` | hidden 64, 5 layers, 8 heads | hidden 24, 2 layers, 4 heads |
| `EnhancedDQNFusionAgent` | `src/models/dqn.py` | hidden 576, 3 layers | hidden 160, 2 layers |

DQN state: 15D vector (VGAE 8D: errors + latent stats + confidence; GAT 7D: logits + embedding stats + confidence).

## Critical Constraints

**Do not violate these — they fix real crashes:**

- **PyG `Data.to()` is in-place.** Always `.clone().to(device)`, never `.to(device)` on shared data. Mutating training data before DataLoader fork causes CUDA errors.
- **Use spawn multiprocessing.** `mp_start_method: "spawn"` in config, `mp.set_start_method('spawn', force=True)` in CLI. Fork + CUDA = crashes.
- **DataLoader workers**: `multiprocessing_context='spawn'` on all DataLoader instances.
- **NFS filesystem**: `.nfs*` ghost files appear when processes delete open files. Already in `.gitignore`. Git operations (stash, reset) can fail on them.
- **No GUI on HPC**: Git auth via SSH key (configured), not HTTPS tokens. Avoid `gnome-ssh-askpass`.

## Environment

- **Cluster**: Ohio Supercomputer Center (OSC), RHEL 9, SLURM scheduler
- **Filesystem**: NFS (shared home across all nodes)
- **Git remote**: `git@github.com:RobertFrenken/DQN-Fusion.git` (SSH)
- **Python**: conda environment, PyTorch + PyG + Lightning

## Output Layout

```
experimentruns/{modality}/{dataset}/{size}/{learning_type}/{model}/{distill}/{mode}/
├── best_model.pt, config.json, logs/, metrics.json
```
