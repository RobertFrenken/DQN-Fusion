## Project Overview

**KD-GAT** = Knowledge Distillation applied to Graph Attention Networks

**Goal:** Run reproducible ML experiments with different configurations and track results
**Your Stack:**
- Hydra-Zen (configuration management)
- PyTorch Lightning (training framework)
- MLflow (experiment tracking)
- Slurm (via oscjobmanager, for cluster execution)
---

## What This Project DOES Need

- Hydra-Zen for experiment combinations
- Easy to modify and iterate
- PyTorch Lightning for consistent training
- Automatic logging and checkpointing
- Mixed precision, distributed training support
- MLflow to compare all runs side-by-side
- oscjobmanager for simple Slurm submissions
- Every run saves its exact config
- Deterministic path structure
---

## What the Agent Should Help With

### ✅ Code Integration
- Integrating new model architectures
- Adding new datasets
- Implementing data loaders
- Updating configurations
- Connecting to Lightning modules

---
## Commands to Know

```bash
# Local training (GPU)
python src/training/train_with_hydra_zen.py config_store=name device=cuda training_config.epochs=100

# Submit to cluster
python oscjobmanager.py submit config_name

# Preview Slurm script (don't submit yet)
python oscjobmanager.py submit config_name --dry-run

# Batch submit
python oscjobmanager.py sweep --model-sizes student,teacher --distillations no,standard

# View results
mlflow ui --backend-store-uri experimentruns/.mlruns

# Check job status
squeue | grep your_username
```
---
## Documentation This Project Has

These files describe the integration process:

1. **README_INTEGRATION.md** - Quick start
2. **INTEGRATION_TODO.md** - Step-by-step checklist
3. **INTEGRATION_CODE_TEMPLATES.md** - Copy-paste code
4. **INTEGRATION_DEBUGGING.md** - Fixing common issues
5. **KD-GAT_INTEGRATION_GUIDE.md** - Complete walkthrough
---
