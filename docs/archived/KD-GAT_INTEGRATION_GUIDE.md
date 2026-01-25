# KD-GAT Integration Guide - Complete Walkthrough

This guide walks you through integrating your GitHub repository with the Hydra-Zen system step-by-step.

## Overview of What You're Doing

You're taking your existing KD-GAT code and integrating it with a production-ready Hydra-Zen configuration system. This gives you:

✅ Type-safe configuration management
✅ Reproducible experiments with 100+ pre-generated configs
✅ Deterministic path structure for all results
✅ MLflow experiment tracking
✅ Slurm job submission automation
✅ Zero hardcoded paths in code

## Your Current Setup

Your GitHub repository (lightning branch) has:

```
KD-GAT/
├── src/
│   ├── data/datasets.py           ← Dataset loading logic
│   ├── models/vgae.py             ← VGAE model
│   ├── models/gat.py              ← GAT model
│   ├── models/dqn.py              ← DQN model (optional)
│   └── training/trainer.py        ← Your existing trainer
├── data/
│   ├── automotive/hcrlch/         ← Your actual data
│   ├── automotive/set01/
│   ├── internet/
│   └── watertreatment/
└── requirements.txt
```

## What Gets Added

The Hydra-Zen system adds these files:

```
KD-GAT/
├── hydra_configs/
│   └── config_store.py            ← NEW: 100+ experiment configs
├── src/
│   ├── training/
│   │   ├── train_with_hydra_zen.py    ← NEW: Main training entry
│   │   ├── lightning_modules.py       ← NEW: PyTorch Lightning modules
│   │   └── trainer.py                 ← Keep your existing trainer
│   └── utils/
│       └── experiment_paths.py        ← NEW: Path management
├── oscjobmanager.py               ← NEW: Slurm submission
└── (other docs)                   ← NEW: Guides and references
```

## Step-by-Step Integration

### Step 1: Copy Core Files (5 minutes)

Copy these files to your project from what you received:

```
hydra_configs/config_store.py       → hydra_configs/config_store.py
src/utils/experiment_paths.py       → src/utils/experiment_paths.py
src/training/train_with_hydra_zen.py → src/training/train_with_hydra_zen.py
src/training/lightning_modules.py   → src/training/lightning_modules.py
oscjobmanager.py                    → oscjobmanager.py
```

Verify files are in place:
```bash
ls -la hydra_configs/config_store.py
ls -la src/utils/experiment_paths.py
ls -la src/training/train_with_hydra_zen.py
ls -la src/training/lightning_modules.py
ls -la oscjobmanager.py
```

### Step 2: Update Dataset Classes (30 minutes)

**File to modify:** `src/data/datasets.py`

**What to do:**
1. Add PyTorch Dataset wrapper class (see INTEGRATION_CODE_TEMPLATES.md Template 1)
2. Update HCRLCHDataset to:
   - Accept `data_path`, `split_ratio`, `normalization` parameters
   - Create `.train`, `.val`, `.test` attributes as Dataset objects
3. Create Set01Dataset, Set02Dataset, Set03Dataset, Set04Dataset classes

**Test it:**
```bash
python3 << 'EOF'
from src.data.datasets import HCRLCHDataset
dataset = HCRLCHDataset('./data/automotive/hcrlch', (0.7, 0.15, 0.15), 'zscore')
print(f"✅ Train: {len(dataset.train)}, Val: {len(dataset.val)}, Test: {len(dataset.test)}")
EOF
```

### Step 3: Update Model Classes (30 minutes)

**Files to modify:** `src/models/vgae.py`, `src/models/gat.py`, `src/models/dqn.py`

**What to do for EACH model:**
1. Change `__init__` to accept configurable parameters
2. Add `**kwargs` to catch extra arguments
3. Keep all implementation logic exactly the same

**Before:**
```python
def __init__(self):
    super().__init__()
    self.hidden_dim = 64  # Hardcoded!
```

**After:**
```python
def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2, **kwargs):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.dropout = dropout
```

**Test it:**
```bash
python3 -c "from src.models.vgae import VGAE; m = VGAE(hidden_dim=64, latent_dim=32, num_layers=2, dropout=0.1); print('✅')"
```

### Step 4: Implement Data Loaders (20 minutes)

**File to modify:** `src/training/train_with_hydra_zen.py`

**What to do:**
1. Find the `load_data_loaders()` function (it's a placeholder)
2. Replace with actual implementation (see INTEGRATION_CODE_TEMPLATES.md Template 3)
3. Import your dataset classes
4. Create DATASET_MAP dictionary
5. Return (train_loader, val_loader, test_loader)

**Test it:**
```bash
python3 << 'EOF'
from src.training.train_with_hydra_zen import load_data_loaders
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    'dataset_config': {
        'name': 'hcrlch',
        'data_path': './data/automotive/hcrlch',
        'split_ratio': (0.7, 0.15, 0.15),
        'normalization': 'zscore',
    },
    'training_config': {'batch_size': 32},
    'num_workers': 0,
    'pin_memory': False,
})

train_loader, val_loader, test_loader = load_data_loaders(cfg)
print(f"✅ Loaders: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test")
EOF
```

### Step 5: Update Configuration (20 minutes)

**File to modify:** `hydra_configs/config_store.py`

**What to do:**
1. Find `project_root` (around line 30)
2. Update to your actual path:
   ```python
   project_root: str = "/home/username/KD-GAT"  # ← Change this!
   ```
3. Update dataset configs:
   - Verify `data_path` points to existing directories
   - Check: `ls -la ./data/automotive/hcrlch/` works
4. Update model configs:
   - Update `_target_` paths (e.g., "src.models.vgae.VGAE")
   - Update parameters to match your models

**Test it:**
```bash
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --cfg job | head -30
```

Expected: Config prints with your paths

### Step 6: Link Models to Lightning (20 minutes)

**File to modify:** `src/training/lightning_modules.py`

**What to do:**
1. Find `_build_vgae()` method
2. Import your actual VGAE class
3. Instantiate with config parameters
4. Repeat for `_build_gat()` and `_build_dqn()`

**Example for VGAE:**
```python
def _build_vgae(self) -> nn.Module:
    from src.models.vgae import VGAE
    
    return VGAE(
        input_dim=self.cfg.model_config.input_dim,
        hidden_dim=self.cfg.model_size_config.hidden_dim,
        latent_dim=self.cfg.model_config.latent_dim,
        num_layers=self.cfg.model_size_config.num_layers,
        dropout=self.cfg.model_size_config.dropout,
    )
```

**Test it:**
```bash
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --cfg job | grep "model_architecture:"
```

### Step 7: Run Single-Epoch Training (5-10 minutes)

**First test on CPU to verify everything works:**

```bash
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    device=cpu \
    training_config.epochs=1
```

**Expected output:**
- Data loads successfully
- Model builds successfully
- Training runs 1 epoch
- Results save to: `experimentruns/automotive/hcrlch/unsupervised/VGAE/student/no/all_samples/run_000/`

**Verify results:**
```bash
ls -la experimentruns/automotive/hcrlch/unsupervised/VGAE/student/no/all_samples/run_000/
# Should have: model.pt, config.yaml, checkpoints/, training_metrics.json
```

### Step 8: Run Hyperparameter Sweep (optional)

```bash
python src/training/train_with_hydra_zen.py -m \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    model_size_config.hidden_dim=32,64,128,256 \
    device=cpu \
    training_config.epochs=1
```

Creates multiple runs: run_000, run_001, run_002, run_003

### Step 9: Submit to Slurm (on OSC)

```bash
# Preview the script
python oscjobmanager.py submit \
    automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --dry-run

# Actually submit
python oscjobmanager.py submit \
    automotive_hcrlch_unsupervised_vgae_student_no_all_samples
```

## File Organization After Integration

Your final project structure:

```
KD-GAT/
├── hydra_configs/
│   └── config_store.py                    ← 100+ pre-generated configs
│
├── src/
│   ├── data/
│   │   └── datasets.py                    ← Your dataset classes (updated)
│   │
│   ├── models/
│   │   ├── vgae.py                        ← Your VGAE (updated)
│   │   ├── gat.py                         ← Your GAT (updated)
│   │   └── dqn.py                         ← Your DQN (updated)
│   │
│   ├── training/
│   │   ├── train_with_hydra_zen.py        ← Main entry point (new)
│   │   ├── lightning_modules.py           ← Lightning modules (new)
│   │   └── trainer.py                     ← Keep your original
│   │
│   └── utils/
│       └── experiment_paths.py            ← Path management (new)
│
├── data/
│   ├── automotive/
│   │   ├── hcrlch/                        ← Your data (unchanged)
│   │   ├── set01/
│   │   ├── set02/
│   │   ├── set03/
│   │   └── set04/
│   ├── internet/
│   └── watertreatment/
│
├── experimentruns/                        ← Results directory (created on first run)
│   └── automotive/
│       └── hcrlch/
│           └── unsupervised/
│               └── VGAE/
│                   └── student/
│                       └── no/
│                           └── all_samples/
│                               ├── run_000/
│                               ├── run_001/
│                               └── ...
│
├── oscjobmanager.py                       ← Slurm job manager (new)
├── requirements.txt                       ← Keep your requirements
└── (documentation files)
```

## Quick Reference: Common Commands

**Single experiment (CPU):**
```bash
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    device=cpu \
    training_config.epochs=5
```

**Single experiment (GPU):**
```bash
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    device=cuda \
    training_config.epochs=100
```

**Hyperparameter sweep:**
```bash
python src/training/train_with_hydra_zen.py -m \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    model_size_config.hidden_dim=32,64,128
```

**Different config:**
```bash
python src/training/train_with_hydra_zen.py \
    config_store=automotive_set01_classifier_gat_teacher_standard_all_samples
```

**View config before running:**
```bash
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --cfg job
```

**Submit to Slurm:**
```bash
python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_student_no_all_samples
```

**Preview Slurm script:**
```bash
python oscjobmanager.py submit \
    automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --dry-run
```

## Troubleshooting

**If something breaks:**

1. Check error message carefully (usually tells you what's wrong)
2. See `INTEGRATION_DEBUGGING.md` for common issues and solutions
3. Verify paths: `ls -la src/data/datasets.py` etc.
4. Test individual components:
   ```bash
   # Test imports
   python3 -c "from src.data.datasets import HCRLCHDataset; print('✅')"
   
   # Test config
   python src/training/train_with_hydra_zen.py config_store=name --cfg job
   
   # Test training
   python src/training/train_with_hydra_zen.py config_store=name device=cpu training_config.epochs=1
   ```

## Key Concepts

**Config Hierarchy (8 levels):**
```
modality (automotive, internet, watertreatment)
  → dataset (hcrlch, set01, set02, set03, set04)
    → learning_type (unsupervised, classifier, fusion)
      → model_architecture (VGAE, GAT, DQN)
        → model_size (teacher, student, intermediate, huge, tiny)
          → distillation (no, standard, topology_preserving)
            → training_mode (all_samples, normals_only, curriculum_*)
              → run_000, run_001, ... (auto-incremented)
```

**Config Name Format:**
```
{modality}_{dataset}_{learning_type}_{model_arch}_{model_size}_{distillation}_{training_mode}
```

**Example:**
```
automotive_hcrlch_unsupervised_vgae_student_no_all_samples
automotive_hcrlch_classifier_gat_teacher_standard_curriculum_classifier
internet_set02_fusion_dqn_intermediate_topology_preserving_all_samples
```

## Success Criteria

You're done when you can:

- [ ] Run: `python src/training/train_with_hydra_zen.py config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples device=cpu training_config.epochs=1`
- [ ] See results save to: `experimentruns/automotive/hcrlch/.../run_000/`
- [ ] Config file saved: `experimentruns/.../run_000/config.yaml`
- [ ] Model saved: `experimentruns/.../run_000/model.pt`
- [ ] Run sweep: `python src/training/train_with_hydra_zen.py -m config_store=name model_size_config.hidden_dim=32,64,128`
- [ ] Submit to Slurm: `python oscjobmanager.py submit name` (on OSC)

## Documentation Files

| File | Purpose |
|------|---------|
| **README_INTEGRATION.md** | Quick start (you read this first) |
| **INTEGRATION_SUMMARY.md** | 5-step overview |
| **INTEGRATION_TODO.md** | Checklist to track progress |
| **INTEGRATION_CODE_TEMPLATES.md** | Copy-paste code |
| **INTEGRATION_DEBUGGING.md** | Fixing common errors |
| **ARCHITECTURE_SUMMARY.md** | System design |
| **QUICK_REFERENCE.md** | Command cheat sheet |
| **SETUP_CHECKLIST.md** | Verification steps |

## Timeline

- Reading docs: 1 hour
- Making code changes: 2 hours
- Testing: 1-2 hours
- **Total: 4-5 hours**

Ready to start? Open `INTEGRATION_TODO.md` and check off items as you go! 🚀
