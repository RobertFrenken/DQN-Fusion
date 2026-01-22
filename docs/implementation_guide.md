# KD-GAT Hydra-Zen Implementation Guide

## Overview

This guide explains how to integrate the new Hydra-Zen configuration system into your KD-GAT project. It replaces your old configuration system with a clean, type-safe, reproducible setup.

## Files Created

### Configuration
- **`hydra_configs/config_store.py`** - Complete Hydra-Zen configuration store with all model/training variations

### Utilities  
- **`src/utils/experiment_paths.py`** - Deterministic path management (NO fallbacks, strict error checking)

### Training
- **`src/training/train_with_hydra_zen.py`** - Main training entry point with Hydra-Zen + PyTorch Lightning
- **`src/training/lightning_modules.py`** - Lightning modules for VGAE, GAT, DQN

### Job Submission
- **`oscjobmanager.py`** - OSC Slurm job submission manager

## Quick Start

### 1. Setup Paths in Config

Update `hydra_configs/config_store.py`:

```python
@dataclass
class ExperimentConfig:
    project_root: str = "/actual/path/to/KD-GAT"  # Change this!
    data_root: str = "${project_root}/data"
    experiment_root: str = "${project_root}/experimentruns"
```

### 2. Run Single Experiment (Local)

```bash
cd /path/to/KD-GAT

# Train VGAE (unsupervised, student, no distillation, all samples)
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples

# Results will be saved to:
# experimentruns/automotive/hcrlch/unsupervised/VGAE/student/no/all_samples/run_000/
```

### 3. Run Hyperparameter Sweep

```bash
# Sweep hidden dimensions
python src/training/train_with_hydra_zen.py -m \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    model_size_config.hidden_dim=64,128,256

# This creates 3 separate runs automatically
```

### 4. Submit to OSC Slurm

```bash
# Single experiment
python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_student_no_all_samples

# Dry run (preview script without submitting)
python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_student_no_all_samples --dry-run

# Submit sweep of experiments
python oscjobmanager.py sweep \
    --model-sizes student,teacher \
    --distillations no,standard \
    --training-modes all_samples,normals_only
```

## Architecture Hierarchy

Your experiments are organized in 8 levels:

```
L1: experimentruns/
L2:   {modality}/              (automotive, internet, watertreatment)
L3:     {dataset}/             (hcrlch, set01, set02, set03, set04)
L4:       {learning_type}/     (unsupervised, classifier, fusion)
L5:         {model_arch}/      (VGAE, GAT, DQN)
L6:           {model_size}/    (teacher, student, intermediate, huge, tiny)
L7:             {distillation}/ (no, standard, topology_preserving)
L8:               {training_mode}/ (all_samples, normals_only, curriculum_*)
L9:                 run_000/
                      ├── model.pt                    # Trained model
                      ├── checkpoints/                # Training checkpoints
                      ├── config.yaml                 # Exact config used
                      ├── training_metrics.json       # Training loss curves
                      ├── validation_metrics.json     # Val metrics
                      └── evaluation/
                          ├── test_results.json
                          ├── test_set/
                          ├── known_unknowns/
                          └── unknown_unknowns/
```

## Configuring Your Datasets

In `hydra_configs/config_store.py`, update dataset configs:

```python
@dataclass
class HCRLCHDataset:
    _target_: str = "src.data.datasets.HCRLCHDataset"
    name: str = "hcrlch"
    modality: str = "automotive"
    data_path: str = "${oc.select:data_root}/automotive/hcrlch"
    split_ratio: tuple = (0.7, 0.15, 0.15)
    normalization: str = "zscore"
```

Make sure your dataset classes:
1. Have `_target_` pointing to actual class
2. Are instantiable with these parameters
3. Return `(train_loader, val_loader, test_loader)` from appropriate function

## Implementing Data Loaders

In `src/training/train_with_hydra_zen.py`, implement `load_data_loaders()`:

```python
def load_data_loaders(cfg: DictConfig) -> Tuple:
    """Load dataset according to config"""
    
    # Instantiate dataset from config
    dataset_cls = instantiate(cfg.dataset_config._target_)
    
    dataset = dataset_cls(
        data_path=cfg.dataset_config.data_path,
        split_ratio=cfg.dataset_config.split_ratio,
        normalization=cfg.dataset_config.normalization,
    )
    
    train_loader = DataLoader(
        dataset.train,
        batch_size=cfg.training_config.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    
    val_loader = DataLoader(
        dataset.val,
        batch_size=cfg.training_config.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    
    test_loader = DataLoader(
        dataset.test,
        batch_size=cfg.training_config.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    
    return train_loader, val_loader, test_loader
```

## Implementing Model Classes

Create model classes that match config `_target_`:

```python
# src/models/vgae.py
class VGAE(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, num_layers: int, dropout: float):
        super().__init__()
        # ... implement your VGAE
    
    def forward(self, x, edge_index):
        # ... return (recon_x, mu, logvar)
        pass
```

## Key Features

### ✅ No Hardcoded Paths
All paths are computed from config hierarchy. If path not set properly, you get informative error.

### ✅ Type Safety
IDE autocomplete and validation work perfectly.

### ✅ Reproducibility
Every run saves its exact configuration.

### ✅ No Pickle Issues
Models saved as PyTorch state dict.

### ✅ Proper Error Handling
Training fails with informative messages.

### ✅ MLflow Integration
All metrics logged to MLflow with proper organization.

## Available Config Names

All these configs are pre-generated and available:

```bash
# Format: {modality}_{dataset}_{learning_type}_{model_arch}_{model_size}_{distillation}_{training_mode}

# Unsupervised (VGAE only)
automotive_hcrlch_unsupervised_vgae_student_no_all_samples
automotive_hcrlch_unsupervised_vgae_student_no_normals_only
automotive_hcrlch_unsupervised_vgae_teacher_no_all_samples
... (all combinations)

# Classifier (GAT, DQN)
automotive_hcrlch_classifier_gat_student_no_curriculum_classifier
... (all combinations)

# Fusion (GAT, DQN)
automotive_hcrlch_fusion_gat_student_standard_curriculum_fusion
... (all combinations)
```

## Testing Your Setup

### 1. Test path generation
```python
from omegaconf import OmegaConf
from src.utils.experiment_paths import ExperimentPathManager

cfg = OmegaConf.create({
    'experiment_root': '/tmp/test_experimentruns',
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
pm.validate_structure()
```

### 2. Test config loading
```bash
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --cfg job  # Print loaded config
```

### 3. Test Slurm script generation
```bash
python oscjobmanager.py submit \
    automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --dry-run
```

## Migration from Old System

### Remove
- Delete all old `*.yaml` config files
- Delete old trainer code with dead branches
- Delete old path construction logic with fallbacks

### Keep
- Your model implementations (VGAE, GAT, DQN classes)
- Your dataset loading code (adapt to new config format)
- Your evaluation metrics

### Update
- Wrap model classes with `_target_` for instantiation
- Update data loading to accept config dict
- Update Lightning modules to use new path manager

## Troubleshooting

### Error: "Missing required configuration fields"
**Cause**: Config doesn't have all required fields
**Fix**: Use `config_store=name` with pre-generated configs, not custom overrides

### Error: "Path not properly configured"
**Cause**: `experiment_root` not set
**Fix**: Update `project_root` in `config_store.py`

### Error: "Model saved to wrong directory"
**Cause**: Paths not initialized with `get_experiment_dir_safe()`
**Fix**: The training script handles this automatically. If error occurs, check `path_manager` initialization

### Model not loading after save
**Cause**: Pickle vs torch.save difference
**Fix**: Always use `torch.load()` with the new format

## Advanced: Custom Configurations

To add a new dataset combination:

```python
# In config_store.py
store(
    ExperimentConfig(
        modality="automotive",
        dataset="set05",  # New dataset
        learning_type="unsupervised",
        model_architecture="VGAE",
        model_size="student",
        distillation="no",
        training_mode="all_samples",
        model_size_config=StudentModelSize(),
        # ... other configs
    ),
    name="automotive_set05_unsupervised_vgae_student_no_all_samples"
)
```

Then submit:
```bash
python oscjobmanager.py submit automotive_set05_unsupervised_vgae_student_no_all_samples
```

## Next Steps

1. **Implement `load_data_loaders()`** in `train_with_hydra_zen.py`
2. **Update dataset classes** with `_target_` paths
3. **Implement model classes** if not already done
4. **Update Lightning modules** to match your model interfaces
5. **Test locally** before submitting to Slurm
6. **Archive old config system** for reference
