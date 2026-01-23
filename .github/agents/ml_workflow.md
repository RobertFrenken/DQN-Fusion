# Refactor Helper - KD-GAT Research Workflow

This file guides VS Code's AI agent on what YOUR project actually needs.

**NOT CI/CD. NOT PRs. NOT GitHub workflows.**

**YES: Hydra-Zen system. YES: Local/Slurm training. YES: MLflow tracking.**

---

## Project Overview

**KD-GAT** = Knowledge Distillation applied to Graph Attention Networks

**Goal:** Run reproducible ML experiments with different configurations and track results

**Your Role:** Solo AI researcher

**Your Stack:**
- Hydra-Zen (configuration management)
- PyTorch Lightning (training framework)
- MLflow (experiment tracking)
- Slurm (via oscjobmanager, for cluster execution)

---

## What This Project DOES NOT Need

❌ CI/CD workflows
❌ GitHub Actions
❌ Pull Request automation
❌ Docker containerization
❌ Kubernetes
❌ Continuous deployment
❌ Automated testing pipelines
❌ Complex manifests

**These are for teams, not solo researchers.**

---

## What This Project DOES Need

✅ **Configuration Management**
- Hydra-Zen for 100+ experiment combinations
- No hardcoded hyperparameters in code
- Easy to modify and iterate

✅ **Training Framework**
- PyTorch Lightning for consistent training
- Automatic logging and checkpointing
- Mixed precision, distributed training support

✅ **Experiment Tracking**
- MLflow to compare all runs side-by-side
- Automatic loss curve tracking
- Reproducible results

✅ **Cluster Integration**
- oscjobmanager for simple Slurm submissions
- No manual job script creation
- Results go to same location locally and on cluster

✅ **Reproducibility**
- Every run saves its exact config
- Deterministic path structure
- Can replay any experiment

---

## Research Workflow (What Matters)

### Phase 1: Local Development

```bash
# 1. Have an idea
# "What if I increase hidden_dim to 128?"

# 2. Edit configuration
# File: hydra_configs/config_store.py
# Change: StudentModelSize.hidden_dim = 128

# 3. Test locally (quick check)
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    device=cpu \
    training_config.epochs=1

# 4. If it works, submit to cluster
python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_student_no_all_samples

# 5. Check results in MLflow
mlflow ui --backend-store-uri experimentruns/.mlruns
```

### Phase 2: Compare & Iterate

```bash
# MLflow shows all experiments:
# Run 1: hidden_dim=64   → loss=0.25
# Run 2: hidden_dim=128  → loss=0.20 ← Better!
# Run 3: hidden_dim=256  → loss=0.22

# Based on results → next idea
# "What if I also add batch norm?"
```

### Phase 3: Publish

```bash
# Collect best results from MLflow
# Write paper
# Publish
```

**That's the entire workflow. No PRs. No CI/CD. No GitHub Actions.**

---

## Code Organization

```
KD-GAT/
├── hydra_configs/
│   └── config_store.py                    ← All configurations
├── src/
│   ├── data/datasets.py                   ← Dataset loading
│   ├── models/
│   │   ├── vgae.py                        ← Your VGAE model
│   │   ├── gat.py                         ← Your GAT model
│   │   └── dqn.py                         ← Your DQN model
│   ├── training/
│   │   ├── train_with_hydra_zen.py        ← Main entry point
│   │   ├── lightning_modules.py           ← Lightning modules
│   │   └── trainer.py                     ← Keep existing trainer
│   └── utils/
│       └── experiment_paths.py            ← Path management
├── data/
│   ├── automotive/hcrlch/                 ← Your data
│   ├── internet/
│   └── watertreatment/
├── experimentruns/                        ← Results (auto-created)
├── oscjobmanager.py                       ← Slurm job manager
└── [Keep your existing files as-is]
```

---

## What the Agent Should Help With

### ✅ Code Integration
- Integrating new model architectures
- Adding new datasets
- Implementing data loaders
- Updating configurations
- Connecting to Lightning modules

### ✅ Code Improvements (Within Scope)
- Better type hints in models
- Cleaner Lightning module code
- More efficient data loading
- Config parameter organization

### ❌ Code Things to Ignore
- "Add CI/CD workflows"
- "Set up GitHub Actions"
- "Create PR templates"
- "Docker containerization"
- "GitHub Actions testing"

### ❌ Infrastructure Things to Avoid
- Kubernetes setup
- Helm charts
- Complex deployment pipelines
- Distributed training orchestration (handled by Slurm)

---

## Commands to Know (Agent Shouldn't Suggest Alternatives)

```bash
# Local testing (CPU)
python src/training/train_with_hydra_zen.py config_store=name device=cpu training_config.epochs=1

# Local training (GPU)
python src/training/train_with_hydra_zen.py config_store=name device=cuda training_config.epochs=100

# Hyperparameter sweep (local)
python src/training/train_with_hydra_zen.py -m config_store=name model_size_config.hidden_dim=32,64,128

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

**These are the only commands that matter. Don't let the agent suggest Jenkins, Travis CI, or other CI/CD tools.**

---

## Integration Guidelines

### When Integrating Code:

✅ **DO:**
- Keep models' core logic unchanged
- Add configuration parameters to `__init__` methods
- Use Hydra-Zen config values, not hardcoded paths
- Return consistent data formats
- Include `**kwargs` in function signatures for flexibility

❌ **DON'T:**
- Suggest CI/CD for testing
- Recommend Docker
- Suggest GitHub Actions workflows
- Propose Kubernetes
- Recommend complex deployment setups

### When Modifying Code:

✅ **DO:**
- Make changes that improve research efficiency
- Update docstrings and comments
- Improve code clarity
- Optimize for developer experience
- Ensure reproducibility

❌ **DON'T:**
- Add CI/CD complexity
- Suggest infrastructure changes
- Propose team/collaboration features
- Recommend devops-style improvements

---

## Configuration Philosophy

**All hyperparameters go in config, NOT in code.**

### ❌ Wrong (Hardcoded)
```python
def __init__(self):
    self.hidden_dim = 64
    self.num_layers = 2
    self.dropout = 0.2
```

### ✅ Right (Configurable)
```python
def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2, **kwargs):
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.dropout = dropout
```

Then change in config without touching code:
```python
# In hydra_configs/config_store.py
StudentModelSize.hidden_dim = 64, 128, 256  # Easy sweep!
```

---

## What Success Looks Like

### Week 1:
- ✅ All code integrated with Hydra-Zen
- ✅ Local training works: `python train.py config_store=name device=cpu training_config.epochs=1`
- ✅ Results save to `experimentruns/`

### Week 2:
- ✅ Run 5+ configurations locally
- ✅ View results in MLflow
- ✅ Compare performance across configs

### Week 3:
- ✅ Submit jobs to Slurm: `python oscjobmanager.py submit name`
- ✅ Run hyperparameter sweeps
- ✅ Analyze results from cluster

### Week 4+:
- ✅ Iterate on ideas based on MLflow results
- ✅ Publish findings
- ✅ Scale to larger experiments

**Never need CI/CD. Never need GitHub Actions. Never need Docker.**

---

## Important: Tell the Agent

If the agent suggests CI/CD, workflows, or GitHub Actions, respond with:

> "No, I'm using Hydra-Zen for configuration and Slurm for job execution. 
> I don't need CI/CD. I manage experiments through configuration changes and direct Slurm submissions.
> Focus on helping me integrate my models and data loaders with the Hydra-Zen system instead."

---

## Documentation This Project Has

These files describe the integration process:

1. **README_INTEGRATION.md** - Quick start
2. **INTEGRATION_TODO.md** - Step-by-step checklist
3. **INTEGRATION_CODE_TEMPLATES.md** - Copy-paste code
4. **INTEGRATION_DEBUGGING.md** - Fixing common issues
5. **KD-GAT_INTEGRATION_GUIDE.md** - Complete walkthrough
6. **What_You_Actually_Need.md** - Why you don't need CI/CD
7. **CI_CD_Explained.md** - Understanding the concepts

**Agent should reference these, not suggest alternatives.**

---

## Research Timeline

- **Week 1-2:** Local development & testing
- **Week 3+:** Cluster submissions (Slurm)
- **Month 2-3:** Large-scale experiments
- **Month 4+:** Publishing results

**NOT a CI/CD pipeline. NOT automated testing. NOT continuous deployment.**

**Just reproducible ML research.**

---

## Summary for the Agent

### Your Project Is:
🔬 **Solo ML research with Hydra-Zen configuration management**

### Your Workflow Is:
1. Modify config
2. Test locally
3. Submit to Slurm
4. Compare in MLflow
5. Iterate

### You Don't Need:
- CI/CD
- GitHub Actions
- PRs and reviews
- Automated testing
- Docker
- Kubernetes
- Complex deployment

### You DO Need:
- Configuration flexibility
- Easy Slurm submission
- Experiment tracking
- Reproducible code
- Clean integrations

**Keep it simple. Let me focus on research.**

---

**End of refactor_helper.md**

If your agent keeps ignoring this and suggesting CI/CD, you may need to disable that agent or give it explicit instructions to prioritize this file.
