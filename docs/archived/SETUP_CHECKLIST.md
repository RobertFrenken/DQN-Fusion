# KD-GAT Hydra-Zen Setup Checklist

## Pre-Implementation (Read-Only)

- [ ] Read `ARCHITECTURE_SUMMARY.md` (understand design)
- [ ] Read `QUICK_REFERENCE.md` (command patterns)
- [ ] Review `hydra_configs/config_store.py` (configurations)
- [ ] Check GitHub lightning branch for current code structure

## Step 1: File Integration

### Copy Files to Project

```bash
# From this output, copy these files to your KD-GAT project:

# Configuration
cp hydra_configs/config_store.py <your-project>/hydra_configs/

# Utilities
cp src/utils/experiment_paths.py <your-project>/src/utils/

# Training
cp src/training/train_with_hydra_zen.py <your-project>/src/training/
cp src/training/lightning_modules.py <your-project>/src/training/

# Job Management
cp oscjobmanager.py <your-project>/

# Documentation
cp IMPLEMENTATION_GUIDE.md <your-project>/
cp QUICK_REFERENCE.md <your-project>/
```

### Verify Files

```bash
cd /path/to/KD-GAT
ls -la hydra_configs/config_store.py
ls -la src/utils/experiment_paths.py
ls -la src/training/train_with_hydra_zen.py
ls -la src/training/lightning_modules.py
ls -la oscjobmanager.py
```

- [ ] All files present

## Step 2: Configuration Updates

### Update Config Root Paths

Edit `hydra_configs/config_store.py`:

```python
@dataclass
class ExperimentConfig:
    project_root: str = "/actual/path/to/KD-GAT"  # ← CHANGE THIS
    data_root: str = "${project_root}/data"
    experiment_root: str = "${project_root}/experimentruns"
```

```bash
# Get actual path
pwd
# Output: /home/user/KD-GAT
# Use that above
```

- [ ] `project_root` updated
- [ ] `data_root` correct
- [ ] `experiment_root` has write permissions

### Verify Path Configuration

```bash
cd /path/to/KD-GAT

python3 << 'EOF'
from omegaconf import OmegaConf
from hydra_configs.config_store import store

# Get a sample config
sample_cfg = OmegaConf.create({
    'experiment_root': '/path/to/experimentruns',  # Same as config_store
    'modality': 'automotive',
    'dataset': 'hcrlch',
    'learning_type': 'unsupervised',
    'model_architecture': 'VGAE',
    'model_size': 'student',
    'distillation': 'no',
    'training_mode': 'all_samples'
})

from src.utils.experiment_paths import ExperimentPathManager
pm = ExperimentPathManager(sample_cfg)
pm.print_structure()
EOF
```

- [ ] Path structure prints correctly
- [ ] No errors about missing directories

## Step 3: Data Loading Integration

### Check Your Dataset Structure

Document your dataset layout:

```
data/
├── automotive/
│   ├── hcrlch/
│   │   ├── train_data.npy
│   │   ├── val_data.npy
│   │   └── test_data.npy
│   ├── set01/
│   └── ...
└── ...
```

- [ ] Data layout documented
- [ ] All datasets present

### Implement `load_data_loaders()`

Edit `src/training/train_with_hydra_zen.py`:

```python
def load_data_loaders(cfg: DictConfig) -> Tuple:
    """Load data according to config"""
    
    # TODO: Implement based on your dataset classes
    # Should return (train_loader, val_loader, test_loader)
    
    # Example structure:
    # from src.data.datasets import HCRLCHDataset
    # from torch.utils.data import DataLoader
    
    # dataset = HCRLCHDataset(cfg.dataset_config.data_path)
    # train_loader = DataLoader(dataset.train, batch_size=cfg.training_config.batch_size)
    # val_loader = DataLoader(dataset.val, batch_size=cfg.training_config.batch_size)
    # test_loader = DataLoader(dataset.test, batch_size=cfg.training_config.batch_size)
    
    # return train_loader, val_loader, test_loader
    raise NotImplementedError("Implement for your datasets")
```

- [ ] `load_data_loaders()` implemented
- [ ] Tested with sample config

### Test Data Loading

```bash
python3 << 'EOF'
from src.training.train_with_hydra_zen import load_data_loaders
from omegaconf import OmegaConf

# Create test config
cfg = OmegaConf.create({
    'dataset_config': {
        'data_path': '/path/to/data/automotive/hcrlch',
    },
    'training_config': {'batch_size': 32},
    'num_workers': 0,
    'pin_memory': False,
})

try:
    train_loader, val_loader, test_loader = load_data_loaders(cfg)
    print(f"✅ Data loaders created")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
except Exception as e:
    print(f"❌ Error: {e}")
EOF
```

- [ ] Data loaders work without errors

## Step 4: Model Integration

### Update Lightning Modules

Edit `src/training/lightning_modules.py`:

For each module (VGAE, GAT, DQN), update:

```python
def _build_vgae(self) -> nn.Module:
    """Build VGAE model from config"""
    # ❌ PLACEHOLDER - Replace with actual import
    from src.models.vgae import VGAE
    
    # ✅ ACTUAL CODE
    return VGAE(
        hidden_dim=self.cfg.model_size_config.hidden_dim,
        latent_dim=self.cfg.model_config.latent_dim,
        num_layers=self.cfg.model_size_config.num_layers,
        dropout=self.cfg.model_size_config.dropout,
    )
```

- [ ] `_build_vgae()` updated with real VGAE class
- [ ] `_build_gat()` updated with real GAT class
- [ ] `_build_dqn()` updated with real DQN class

### Check Model Signatures

Verify your models accept expected inputs:

```python
# VGAE
model = VGAE(hidden_dim=64, latent_dim=32, num_layers=2, dropout=0.1)
recon_x, mu, logvar = model(x, edge_index)

# GAT
model = GAT(input_dim=128, hidden_dim=128, output_dim=128, num_heads=4, num_layers=2, dropout=0.1)
logits = model(x, edge_index)

# DQN
model = DQN(input_dim=128, hidden_dim=256, action_dim=10, num_layers=3, dropout=0.1)
q_values = model(x)
```

- [ ] Model signatures match Lightning module expectations

### Test Model Creation

```bash
python3 << 'EOF'
import torch
from src.training.lightning_modules import VAELightningModule
from omegaconf import OmegaConf

# Create minimal config
cfg = OmegaConf.create({
    'model_architecture': 'VGAE',
    'model_size': 'student',
    'model_size_config': {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1},
    'model_config': {'latent_dim': 32},
    'learning_config': {'kl_weight': 0.0001},
    'training_config': {'learning_rate': 1e-3, 'weight_decay': 1e-5, 'optimizer': 'adam', 'scheduler': None},
    'seed': 42,
})

from src.utils.experiment_paths import ExperimentPathManager
pm = ExperimentPathManager({'experiment_root': '/tmp'})

try:
    module = VAELightningModule(cfg, pm)
    print("✅ VAE Lightning module created")
except Exception as e:
    print(f"❌ Error: {e}")
EOF
```

- [ ] Model creation works without errors

## Step 5: Requirements & Dependencies

### Check Python Version

```bash
python3 --version
# Should be 3.8+
```

- [ ] Python 3.8 or higher

### Create Requirements File

Create `requirements_hydra.txt`:

```txt
hydra-core>=1.3.0
hydra-zen>=0.13.0
omegaconf>=2.3.0
pytorch-lightning>=2.0.0
torch>=2.0.0
mlflow>=2.0.0
submitit>=1.4.0
tensorboard>=2.10.0
```

### Install Dependencies

```bash
pip install -r requirements_hydra.txt
```

- [ ] All dependencies installed

### Verify Installation

```bash
python3 << 'EOF'
try:
    import hydra_zen
    print(f"✅ hydra-zen {hydra_zen.__version__}")
    
    import pytorch_lightning as pl
    print(f"✅ pytorch-lightning {pl.__version__}")
    
    import torch
    print(f"✅ torch {torch.__version__}")
    
    import mlflow
    print(f"✅ mlflow {mlflow.__version__}")
    
    import submitit
    print(f"✅ submitit {submitit.__version__}")
except ImportError as e:
    print(f"❌ Missing: {e}")
EOF
```

- [ ] All packages importable

## Step 6: Local Testing

### Test 1: Path Generation

```bash
python3 << 'EOF'
from omegaconf import OmegaConf
from src.utils.experiment_paths import ExperimentPathManager

cfg = OmegaConf.create({
    'experiment_root': './test_experimentruns',
    'modality': 'automotive',
    'dataset': 'hcrlch',
    'learning_type': 'unsupervised',
    'model_architecture': 'VGAE',
    'model_size': 'student',
    'distillation': 'no',
    'training_mode': 'all_samples'
})

pm = ExperimentPathManager(cfg)
pm.print_structure()
print("✅ Path generation works")
EOF
```

- [ ] Path generation succeeds
- [ ] Directory structure printed correctly

### Test 2: Configuration Loading

```bash
cd /path/to/KD-GAT

python3 src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --cfg job \
    hydra.run.dir=./test_run
```

Expected output: Full config printed

- [ ] Config loads without errors
- [ ] All fields present
- [ ] No validation errors

### Test 3: Data Loading

```bash
python3 << 'EOF'
from src.training.train_with_hydra_zen import load_data_loaders
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    'dataset_config': {
        'data_path': './data/automotive/hcrlch',
    },
    'training_config': {'batch_size': 32},
    'num_workers': 0,
    'pin_memory': False,
})

train_loader, val_loader, test_loader = load_data_loaders(cfg)
print(f"✅ Data loaders work")
print(f"   Train: {len(train_loader)} batches")
EOF
```

- [ ] Data loads without errors
- [ ] Batch counts reasonable

### Test 4: Model Creation

```bash
python3 << 'EOF'
from src.training.lightning_modules import VAELightningModule
from src.utils.experiment_paths import ExperimentPathManager
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    'experiment_root': './test_experimentruns',
    'modality': 'automotive',
    'dataset': 'hcrlch',
    'learning_type': 'unsupervised',
    'model_architecture': 'VGAE',
    'model_size': 'student',
    'model_size_config': {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1},
    'model_config': {'latent_dim': 32},
    'learning_config': {'kl_weight': 0.0001},
    'training_config': {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'optimizer': 'adam',
        'scheduler': None,
        'batch_size': 32,
    },
    'seed': 42,
    'device': 'cpu',
})

pm = ExperimentPathManager(cfg)
module = VAELightningModule(cfg, pm)
print(f"✅ Model created")
print(f"   Parameters: {sum(p.numel() for p in module.parameters())}")
EOF
```

- [ ] Model creation succeeds
- [ ] Parameter count printed

## Step 7: Single Training Run

### Run Single Local Experiment

```bash
cd /path/to/KD-GAT

python3 src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    device=cpu \
    training_config.epochs=2 \
    hydra.run.dir=./test_run
```

Expected:
- Paths created
- Config saved
- Data loaded
- Model created
- Training runs for 2 epochs
- Results saved

- [ ] Training runs without crashes
- [ ] Output directories created
- [ ] Model saved to expected location

### Verify Output Structure

```bash
ls -la test_run/experimentruns/automotive/hcrlch/unsupervised/VGAE/student/no/all_samples/run_000/
```

Should have:
- `model.pt` ← Model weights
- `config.yaml` ← Config file
- `checkpoints/` ← Saved checkpoints
- `training_metrics.json` ← Loss curves

- [ ] All expected files present

## Step 8: Slurm Setup (OSC)

### Test Slurm Script Generation

```bash
python3 oscjobmanager.py submit \
    automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --dry-run
```

Expected: Script printed to console

- [ ] Script generates without errors
- [ ] Script looks valid (sbatch directives, conda activation, etc)

### Verify OSC Account

Check account is correct:

```bash
# Edit oscjobmanager.py or hydra_configs/config_store.py
# Update:
# account: str = "PAS3209"
# email: str = "frenken.2@osu.edu"
```

- [ ] Account number correct
- [ ] Email address correct

## Step 9: Hyperparameter Sweep (Local)

### Run Sweep Locally

```bash
cd /path/to/KD-GAT

python3 src/training/train_with_hydra_zen.py -m \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    model_size_config.hidden_dim=32,64 \
    training_config.epochs=2 \
    device=cpu \
    hydra.run.dir=./test_sweep
```

This runs 2 experiments (2 hidden dims):

```
multirun output
├── 0/ (hidden_dim=32)
└── 1/ (hidden_dim=64)
```

- [ ] Sweep completes without errors
- [ ] Multiple run directories created

## Step 10: Documentation Check

- [ ] Read IMPLEMENTATION_GUIDE.md completely
- [ ] Read QUICK_REFERENCE.md completely
- [ ] Review all markdown files in project

## Final Verification

### Checklist Summary

- [ ] All files copied to project
- [ ] Paths configured (`project_root`, `experiment_root`)
- [ ] Data loaders implemented
- [ ] Models linked in Lightning modules
- [ ] Dependencies installed
- [ ] Local test passes (2 epoch training)
- [ ] Output directory structure correct
- [ ] Slurm scripts generate correctly
- [ ] Documentation reviewed

### Quick Test Command

```bash
cd /path/to/KD-GAT

# This one command verifies everything:
python3 src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --cfg job
```

If this works → Configuration ✅

```bash
python3 src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    device=cpu \
    training_config.epochs=1
```

If this works → Full pipeline ✅

## Troubleshooting

### Problem: "Config not found"
```
Solution: Ensure config_store.py is in hydra_configs/
         And you're using exact name from config_store
```

### Problem: "Path error: experiment_root not set"
```
Solution: Update project_root in hydra_configs/config_store.py
```

### Problem: "ModuleNotFoundError: No module named 'src'"
```
Solution: Run from project root directory (cd /path/to/KD-GAT)
```

### Problem: "Data loading fails"
```
Solution: Check data_path in config_store.py matches actual data location
         Implement load_data_loaders() correctly
```

### Problem: "CUDA out of memory"
```
Solution: Reduce batch_size or use device=cpu for testing
```

## Next Steps After Setup

1. ✅ Complete this checklist
2. Run many local experiments
3. Analyze results with MLflow
4. Submit sweep to Slurm
5. Monitor jobs with squeue
6. Analyze results from OSC runs
7. Update configs for next iteration

## Support

If stuck:
1. Check error message - it's usually informative
2. Review IMPLEMENTATION_GUIDE.md
3. Check example commands in QUICK_REFERENCE.md
4. Verify all local tests pass
5. Review code comments in source files

Good luck! 🚀
