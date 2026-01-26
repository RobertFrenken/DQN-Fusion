# Getting Started with KD-GAT

**Quick start guide for training CAN intrusion detection models with knowledge distillation**

---

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- Conda or Mamba
- 16GB+ RAM for full training

---

## 1. Quick Setup (5 minutes)

### Install Environment

```bash
# Clone repository
git clone <your-repo-url>
cd KD-GAT

# Create environment
conda env create -f environment.yml
conda activate gnn-experiments

# Verify installation
python -c "from src.config.hydra_zen_configs import CANGraphConfigStore; print('✓ Setup complete')"
```

### Download Datasets

```bash
# Place CAN datasets in:
data/automotive/
├── hcrl_sa/
├── hcrl_ch/
├── set_01/
├── set_02/
├── set_03/
└── set_04/
```

---

## 2. First Training Run (Normal GAT)

### Train Teacher GAT Model

```bash
python train_with_hydra_zen.py \
  --model gat \
  --dataset hcrl_sa \
  --training normal
```

**Output**: Saved to `experiment_runs/automotive/hcrl_sa/supervised/gat/teacher/no_distillation/normal/`

### Train VGAE Autoencoder

```bash
python train_with_hydra_zen.py \
  --model vgae \
  --dataset hcrl_sa \
  --training autoencoder
```

**Output**: Saved to `experiment_runs/automotive/hcrl_sa/unsupervised/vgae/teacher/no_distillation/autoencoder/`

---

## 3. Configuration Basics

### Using Config Store

```python
from src.config.hydra_zen_configs import CANGraphConfigStore

store = CANGraphConfigStore()

# Create config
config = store.create_config(
    model_type="gat",
    dataset_name="hcrl_sa",
    training_mode="normal"
)

# Override defaults
config = store.create_config(
    model_type="gat",
    dataset_name="hcrl_sa",
    training_mode="normal",
    max_epochs=200,
    batch_size=128
)
```

### Training Modes

| Mode | Model | Purpose |
|------|-------|---------|
| `normal` | GAT | Supervised classification |
| `autoencoder` | VGAE | Unsupervised reconstruction |
| `curriculum` | GAT | Hard sample mining with VGAE guidance |
| `knowledge_distillation` | GAT/VGAE | Teacher→Student compression |
| `fusion` | DQN | Multi-model fusion with RL |

### Model Types

| Type | Parameters | Use Case |
|------|------------|----------|
| `gat` | ~1.1M | Teacher classifier |
| `gat_student` | ~55K | Onboard student |
| `vgae` | ~1.74M | Teacher autoencoder |
| `vgae_student` | ~87K | Onboard autoencoder |
| `dqn` | ~687K | Fusion agent |

---

## 4. Common Workflows

### A. Train Teacher Models

```bash
# 1. Train VGAE autoencoder (unsupervised)
python train_with_hydra_zen.py --model vgae --dataset hcrl_sa --training autoencoder

# 2. Train GAT classifier (supervised)
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal
```

### B. Knowledge Distillation

```bash
# Distill teacher GAT → student GAT
python train_with_hydra_zen.py \
  --model gat_student \
  --dataset hcrl_sa \
  --training knowledge_distillation \
  --teacher-path experiment_runs/.../best_teacher_model.pth
```

### C. Curriculum Learning

```bash
# Train GAT with VGAE-guided hard mining
python train_with_hydra_zen.py \
  --model gat \
  --dataset hcrl_sa \
  --training curriculum \
  --vgae-path experiment_runs/.../vgae_autoencoder.pth
```

### D. Multi-Model Fusion

```bash
# Train DQN fusion agent
python train_with_hydra_zen.py \
  --model dqn \
  --dataset hcrl_sa \
  --training fusion
```

*Note: Fusion automatically finds required models (VGAE + GAT) in canonical paths*

---

## 5. Submitting to OSC (SLURM)

### Create Job Preset

```bash
# Preview job
python oscjobmanager.py preview --preset gat_normal_hcrl_sa

# Dry run (no submission)
python oscjobmanager.py submit --preset gat_normal_hcrl_sa --dry-run

# Submit job
python oscjobmanager.py submit --preset gat_normal_hcrl_sa
```

### Monitor Jobs

```bash
# Check status
squeue -u $USER

# View logs
tail -f slurm_jobs/job_*.out
```

---

## 6. Project Structure

```
KD-GAT/
├── data/                         # CAN datasets
│   └── automotive/
├── experiment_runs/              # All training outputs (canonical paths)
│   └── automotive/
│       └── {dataset}/
│           ├── supervised/       # Classifiers (GAT)
│           ├── unsupervised/     # Autoencoders (VGAE)
│           └── rl_fusion/        # Fusion agents (DQN)
├── src/
│   ├── config/
│   │   └── hydra_zen_configs.py  # ⭐ Single config source
│   ├── models/                   # Model architectures
│   ├── training/
│   │   ├── trainer.py            # Unified trainer
│   │   ├── lightning_modules.py  # Lightning wrappers
│   │   └── modes/                # Training modes
│   └── paths.py                  # PathResolver
├── train_with_hydra_zen.py       # ⭐ Main training script
└── oscjobmanager.py              # ⭐ Job submission
```

---

## 7. Key Configuration Files

### Single Source of Truth

**All configs** are in: `src/config/hydra_zen_configs.py`

- Model configs: `GATConfig`, `VGAEConfig`, `DQNConfig` (+ student variants)
- Dataset configs: `CANDatasetConfig`
- Training configs: `NormalTrainingConfig`, `AutoencoderTrainingConfig`, etc.
- Root config: `CANGraphConfig`
- Store: `CANGraphConfigStore`

### Path Management

**Canonical experiment paths** follow strict hierarchy:

```
{experiment_root}/{modality}/{dataset}/{learning_type}/{model_arch}/{model_size}/{distillation}/{training_mode}/
```

Example:
```
experiment_runs/automotive/hcrl_sa/supervised/gat/teacher/no_distillation/normal/
```

---

## 8. Common Issues

### CUDA Out of Memory

```bash
# Enable batch size optimization
python train_with_hydra_zen.py \
  --model gat \
  --dataset hcrl_sa \
  --training normal \
  --optimize-batch-size
```

### Dataset Not Found

```bash
# Verify path
ls -la data/automotive/hcrl_sa/

# Or set explicitly
python train_with_hydra_zen.py \
  --model gat \
  --dataset hcrl_sa \
  --training normal \
  --data-path /path/to/your/dataset
```

### Missing Teacher Model

For distillation/curriculum/fusion:

```bash
# Check canonical path exists
ls experiment_runs/automotive/hcrl_sa/.../

# Or specify explicitly
python train_with_hydra_zen.py \
  --model gat_student \
  --dataset hcrl_sa \
  --training knowledge_distillation \
  --teacher-path /explicit/path/to/teacher.pth
```

---

## 9. Next Steps

### Advanced Training

- **Curriculum Learning**: See [EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md)
- **Job Templates**: See [JOB_TEMPLATES.md](JOB_TEMPLATES.md)
- **Parameter Budgets**: See [MODEL_SIZE_CALCULATIONS.md](MODEL_SIZE_CALCULATIONS.md)

### Experiment Tracking

- **MLflow Setup**: See [MLflow_SETUP.md](MLflow_SETUP.md)
- **Workflow Guide**: See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)

### Reference

- **Quick Commands**: See [QUICK_REFERENCES.md](QUICK_REFERENCES.md)
- **Code Templates**: See [CODE_TEMPLATES.md](CODE_TEMPLATES.md)
- **Troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## Support

- Architecture overview: [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)
- Job submission: [SUBMITTING_JOBS.md](SUBMITTING_JOBS.md)
- Issues: Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**You're ready to train!** Start with a simple GAT model and work up to advanced techniques.
