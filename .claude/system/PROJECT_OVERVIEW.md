# CAN-Graph KD-GAT: Project Overview

**Last Updated**: 2026-01-28

## Purpose

Transfer learning system for CAN bus intrusion detection. Knowledge distillation compresses large teacher models (VGAE, GAT) into lightweight students, then a DQN fusion agent learns to combine their predictions.

---

## Pipeline Architecture

```
Stage 1: AUTOENCODER (VGAE)     Stage 2: CURRICULUM (GAT)     Stage 3: FUSION (DQN)
Unsupervised reconstruction  →  Supervised classification   →  RL-based prediction fusion
                                     ↑
                              Stage 4: EVALUATION (all models)
```

**Entry point**: `python -m pipeline.cli <stage> [options]`

### Pipeline Module (`pipeline/`)

| File | Purpose |
|------|---------|
| `config.py` | `PipelineConfig` frozen dataclass — every tunable parameter in one place, JSON save/load, preset factory |
| `paths.py` | `stage_dir()` canonical path layout, `checkpoint_path()`, `config_path()`, dataset list |
| `stages.py` | All training logic: `train_autoencoder`, `train_curriculum`, `train_fusion`, `evaluate` |
| `validate.py` | Config validation (dataset existence, KD consistency, numeric sanity) |
| `cli.py` | Argument parser dispatching to `STAGE_FNS` |
| `Snakefile` | Snakemake workflow definition for automated multi-stage runs |

No Hydra, no Pydantic in the pipeline module — plain Python dataclasses and JSON.

---

## Models (`src/models/`)

| Model | File | Role | Teacher / Student |
|-------|------|------|-------------------|
| `GraphAutoencoderNeighborhood` | `vgae.py` | Unsupervised graph reconstruction (continuous features + CAN ID + neighborhood) | Teacher: `(1024,512,96)` latent 96 / Student: `(80,40,16)` latent 16 |
| `GATWithJK` | `models.py` | Supervised binary classification with Jumping Knowledge | Teacher: hidden 64, 5 layers, 8 heads / Student: hidden 24, 2 layers, 4 heads |
| `EnhancedDQNFusionAgent` | `dqn.py` | DQN that selects fusion alpha from 15D state vector (VGAE+GAT features) | Teacher: hidden 576, 3 layers / Student: hidden 160, 2 layers |

### 15D DQN State Space

```
VGAE (8D): 3 error components + 4 latent stats (mean/std/max/min) + 1 confidence
GAT  (7D): 2 logits + 4 embedding stats (mean/std/max/min) + 1 confidence
```

---

## Knowledge Distillation

KD is an orthogonal toggle (`use_kd` flag), independent from the training stage.

| Stage | KD Method | Details |
|-------|-----------|---------|
| VGAE | Dual-signal MSE | `0.5 * MSE(project(z_s), z_t) + 0.5 * MSE(recon_s, recon_t)` |
| GAT | Soft-label KL div | `KL(student/T, teacher/T) * T^2`, T=4.0, alpha=0.7 |
| DQN/Fusion | Not supported | Fusion uses already-distilled models |

---

## Datasets

| Name | Size | OOM Risk |
|------|------|----------|
| `hcrl_ch` | ~1,200 samples | None |
| `hcrl_sa` | ~10,000 samples | None |
| `set_01` | Small-medium | Low |
| `set_02` | Large | High |
| `set_03` | Large | High |
| `set_04` | Large | High |

---

## Directory Layout

```
experimentruns/{modality}/{dataset}/{size}/{learning_type}/{model}/{distill}/{mode}/
├── best_model.pt          # Model checkpoint
├── config.json            # Frozen PipelineConfig for this run
├── logs/                  # Lightning CSV logs
└── metrics.json           # Evaluation results (stage 4 only)
```

---

## Training Flow

```bash
# Teacher pipeline (no KD)
python -m pipeline.cli autoencoder --preset vgae,teacher --dataset hcrl_ch
python -m pipeline.cli curriculum  --preset gat,teacher  --dataset hcrl_ch
python -m pipeline.cli fusion      --preset dqn,teacher  --dataset hcrl_ch
python -m pipeline.cli evaluation  --dataset hcrl_ch

# Student pipeline (with KD)
python -m pipeline.cli autoencoder --preset vgae,student --dataset hcrl_ch --teacher-path /path/to/vgae_teacher.pt
python -m pipeline.cli curriculum  --preset gat,student  --dataset hcrl_ch --teacher-path /path/to/gat_teacher.pt
python -m pipeline.cli fusion      --preset dqn,student  --dataset hcrl_ch
python -m pipeline.cli evaluation  --dataset hcrl_ch --model-size student
```

### Legacy Entry Point

`train_with_hydra_zen.py` still works via `--frozen-config` or manual args, delegates to `HydraZenTrainer` in `src/training/trainer.py`.

---

## Source Tree

```
pipeline/               # Self-contained training pipeline (new, preferred)
src/
├── models/             # VGAE, GAT, DQN architectures
├── training/           # Legacy trainer, Lightning modules, datamodules, KD helpers
├── evaluation/         # Metrics, evaluation, ablation
├── preprocessing/      # Raw CAN data → graph construction
├── config/             # Hydra-Zen configs (legacy), frozen config utils
├── cli/                # Legacy CLI (pydantic validators)
├── visualizations/     # Publication figure generators (UMAP, performance, policy)
├── scripts/            # Utility scripts
└── utils/              # GPU monitoring, plotting, seeding, caching
config/                 # SLURM profile, Snakemake config, conda envs, plot styles
docs/                   # Architecture docs, readmes, migration notes
```

---

## Key Technical Details

- **OOM mitigation**: Frozen teacher (`@torch.no_grad`), memory cleanup every N batches, gradient checkpointing, adaptive batch sizing via safety factors
- **Curriculum learning**: VGAE scores graph reconstruction difficulty → GAT trains on progressively harder normals
- **Frozen configs**: Every run saves its `PipelineConfig` as JSON alongside the checkpoint for full reproducibility
- **Presets**: `PipelineConfig.from_preset("gat", "teacher")` loads architecture-specific defaults
- **Safety factors**: Per-preset batch size scaling (0.45–0.6) to prevent OOM on large datasets
