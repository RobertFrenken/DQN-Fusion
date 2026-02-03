# KD-GAT System Architecture

**Last Updated**: 2026-02-03

## Overview

KD-GAT is a three-stage graph neural network pipeline for CAN bus intrusion detection with knowledge distillation support.

## Pipeline Entry Point

```bash
python -m pipeline.cli <stage> --preset <model>,<size> --dataset <name>
```

Orchestration via Snakemake:
```bash
snakemake -s pipeline/Snakefile --profile profiles/slurm
```

## Three Training Stages

| Stage | Model | Purpose | Output |
|-------|-------|---------|--------|
| `autoencoder` | VGAE | Unsupervised feature learning | `best_model.pt` |
| `curriculum` | GAT | Supervised classification with curriculum | `best_model.pt` |
| `fusion` | DQN | RL-based fusion of VGAE+GAT predictions | `best_model.pt` |

Each stage runs for three model tiers: **teacher** (large), **student with KD**, **student without KD** (ablation).

## Directory Structure

```
experimentruns/{dataset}/{model_size}_{stage}[_kd]/
├── best_model.pt       # Trained model checkpoint
├── config.json         # Frozen configuration
├── slurm.out/.err      # Job logs
└── lightning_logs/     # Training metrics
```

Example paths:
- `experimentruns/hcrl_sa/teacher_autoencoder/best_model.pt`
- `experimentruns/hcrl_sa/student_curriculum_kd/best_model.pt`

## Core Files

### Pipeline Module (`pipeline/`)

| File | Purpose |
|------|---------|
| `cli.py` | CLI entry point, argument parsing |
| `config.py` | `PipelineConfig` dataclass + `PRESETS` |
| `stages.py` | All training logic (VGAE, GAT, DQN modules) |
| `paths.py` | Path derivation functions |
| `validate.py` | Pre-flight validation |
| `tracking.py` | MLflow integration |
| `Snakefile` | SLURM workflow DAG |

### Models (`src/models/`)

| File | Classes |
|------|---------|
| `vgae.py` | `GraphAutoencoderNeighborhood` |
| `models.py` | `GATWithJK` |
| `dqn.py` | `QNetwork`, `EnhancedDQNFusionAgent` |

### Data (`src/preprocessing/`, `src/training/`)

| File | Purpose |
|------|---------|
| `preprocessing.py` | `graph_creation()`, `GraphDataset`, CAN ID mapping |
| `datamodules.py` | `load_dataset()`, `CANGraphDataModule` |

## Configuration System

Single frozen dataclass with preset combinations:

```python
from pipeline.config import PipelineConfig

# Create from preset
cfg = PipelineConfig.from_preset("gat", "student", dataset="hcrl_sa")

# Override specific params
cfg = cfg.with_overrides(lr=0.001, max_epochs=500)

# Serialize/deserialize
cfg.save("config.json")
cfg = PipelineConfig.load("config.json")
```

Key config parameters:
- `precision`: `"16-mixed"` (default) for memory efficiency
- `gradient_checkpointing`: `True` (default) to reduce activation memory
- `use_kd`: `True` to enable knowledge distillation
- `teacher_path`: Path to teacher checkpoint for KD

## Knowledge Distillation

When `use_kd=True`:

1. Teacher model loaded from `teacher_path` and frozen
2. Student trained with combined loss:
   - Task loss (classification/reconstruction)
   - KD loss (soft label matching with temperature)
3. Controlled by `kd_alpha` (loss balance) and `kd_temperature`

## MLflow Tracking

All runs tracked at:
```
sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db
```

Tags: `dataset`, `stage`, `model_size`, `use_kd`, `status`
Metrics: `val_loss`, `accuracy`, `f1`, `precision`, `recall`, `auc`

## Datasets

6 datasets across automotive CAN bus data:
- `hcrl_sa`, `hcrl_ch` (HCRL data)
- `set_01`, `set_02`, `set_03`, `set_04` (additional sets)

Data path: `data/automotive/{dataset}/`

## SLURM Profile

Located at `profiles/slurm/config.yaml`:
- Account: PAS3209
- Partition: gpu
- Resources: 1 GPU (V100), 40GB memory, 4h walltime
