# CAN-Graph KD-GAT: Project Overview

## Project Name & Purpose
**CAN-Graph Knowledge Distillation GAT** - Transfer learning system for CAN bus anomaly detection using knowledge distillation from teacher models to lightweight student models.

## Core Architecture

### Three-Stage Pipeline
```
Stage 1: VGAE          Stage 2: GAT             Stage 3: DQN
Unsupervised           Supervised              Reinforcement Learning
Autoencoder            Classification          Fusion Agent
```

### Model Sizing Strategy
- **Teacher Models**: Larger, fully-featured versions for pretraining
  - VGAE Teacher: `hidden_dims=[128, 128]`, `latent_dim=32`
  - GAT Teacher: `hidden_channels=128`, 4 attention heads
- **Student Models**: Compressed versions for KD training
  - VGAE Student: `hidden_dims=[64, 64]`, `latent_dim=16`
  - GAT Student: `hidden_channels=64`, 4 attention heads
- Model size ALWAYS specified via `--training-strategyl-size {teacher|student}`

### Dataset Organization
```
Datasets:
- hcrl_ch:  Small dataset (~1,200 samples)
- hcrl_sa:  Medium dataset (~10,000 samples)
- set_01:   Small-medium dataset
- set_02:   Large dataset (OOM risk)
- set_03:   Large dataset (OOM risk)
- set_04:   Large dataset (OOM risk)
```

### Modality (Application Domain)
```
--modality {automotive, industrial, robotics}
```
REQUIRED for all CLI commands. Affects data loading and result paths.

## Knowledge Distillation Strategy

### KD as Orthogonal Toggle (NOT a training mode)
- KD is INDEPENDENT from training mode
- Can combine with: `autoencoder`, `curriculum`, `normal` modes
- CANNOT combine with: `fusion` mode (uses already-distilled models)

### VGAE KD: Dual-Signal Approach
```
Signal 1: Latent Space
  - MSE loss between student z and projected teacher z
  - Projection layer: student_latent_dim → teacher_latent_dim

Signal 2: Reconstruction
  - MSE loss between student and teacher continuous outputs
  - Same features being reconstructed

Combined: 0.5 * latent_loss + 0.5 * recon_loss
```

### GAT KD: Soft Label Distillation
```
- Temperature-scaled softmax on logits
- KL divergence loss with temperature^2 scaling
- Temperature default: 4.0
- Alpha (KD weight): 0.7 (70% KD loss, 30% task loss)
```

## Memory Management

### Safety Factor System
Located: `config/batch_size_factors.json`
- Per-dataset factors (0.3-0.8 range, lower = more conservative)
- KD-specific factors = regular factors × 0.75 (25% extra for teacher)
- Automatically selected by trainer based on `use_knowledge_distillation` flag

### Example Factors
```json
{
  "hcrl_ch": 0.6,      "hcrl_ch_kd": 0.45,
  "hcrl_sa": 0.55,     "hcrl_sa_kd": 0.41,
  "set_02": 0.35,      "set_02_kd": 0.26
}
```

### OOM Mitigation Techniques
1. Teacher frozen with `@torch.no_grad()` (no gradient graph)
2. Memory cleanup every 20 batches
3. Gradient checkpointing for VGAE/GAT
4. Mixed precision training (AMP)
5. Adaptive batch sizing based on safety factors

## Key Files & Modules

### Lightning Modules
- `src/training/lightning_modules.py`: VAELightningModule, GATLightningModule, DQNLightningModule
- Each supports KD via `KDHelper` class

### Knowledge Distillation
- `src/training/knowledge_distillation.py`: KDHelper class (450+ lines)
  - Teacher loading and freezing
  - Projection layer management
  - Model-specific KD loss computation

### Configuration
- `src/config/hydra_zen_configs.py`: Base configs with KD fields
  - `use_knowledge_distillation: bool`
  - `teacher_model_path: Optional[str]`
  - `distillation_temperature: float = 4.0`
  - `distillation_alpha: float = 0.7`

### CLI & Pipeline
- `src/cli/main.py`: Entry point with pipeline command
- `src/cli/job_manager.py`: SLURM job generation and submission
- `src/cli/config_builder.py`: Build CANGraphConfig from CLI args
- `train_with_hydra_zen.py`: Training script entry point

### Batch Size Optimization
- `src/training/trainer.py`: Reads safety factors from JSON
- `src/training/adaptive_batch_size.py`: SafetyFactorDatabase class for persistence
- JSON database at `config/batch_size_factors.json`

## SLURM Job Organization
```
experimentruns/
  slurm_runs/
    hcrl_ch/           (all jobs for dataset hcrl_ch)
      vgae_hcrl_ch_autoencode.sh
      gat_hcrl_ch_curriculum.sh
      dqn_hcrl_ch_fusion.sh
    hcrl_sa/           (organized by dataset)
      ...
```

## Critical Constraints

### Pipeline Rules
1. DQN (fusion mode) CANNOT use KD (explicitly rejected)
2. `--training-strategyl-size` and `--distillation` are INDEPENDENT dimensions (any combo valid except fusion+KD)
3. KD requires valid `--teacher_path` (validated before SLURM submission)
4. Multi-value params (--training-strategyl, --training-strategy, --distillation) must have same length
5. Modality MUST be specified (automotive/industrial/robotics)

### Config Validation
- Happens at TWO levels:
  1. Pydantic validators (`src/cli/pydantic_validators.py`)
  2. Hydra config validation (`src/config/hydra_zen_configs.py`)
- Both must pass or training doesn't start

## Common Pitfalls & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Jobs submitted but no KD happening | `format_training_args()` not passing flags | Fixed: now passes `--use-kd` and `--teacher_path` |
| Model overwrites | Same filename for teacher/student | Fixed: filename now includes model_size (vgae_teacher_autoencoder.pth) |
| Fusion+KD jobs submitted | Validation only at config level, not CLI | Fixed: added pipeline CLI validation |
| OOM on large datasets with KD | Safety factors not accounting for teacher | Fixed: KD-specific factors in JSON (×0.75) |
| Missing modality in examples | Inconsistent CLI documentation | Solution: ALWAYS include --modality in examples |
