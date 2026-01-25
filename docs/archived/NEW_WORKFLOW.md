# New Workflow Guide

**Status**: ✅ MIGRATION COMPLETE

This document describes the new consolidated workflow after the major refactoring completed on 2025-01-23.

## Summary of Changes

### Files Removed (~3,700 lines)
The following obsolete files have been **permanently deleted**:

1. `src/training/_old_modules/` (entire directory - 4 files, ~1,500 lines)
   - `can_graph_data.py`
   - `enhanced_datamodule.py`
   - `can_graph_module.py`
   - `fusion_lightning.py`

2. `src/training/lightning_modules_OLD.py` (~400 lines)
3. `src/training/train_with_hydra.py` (~800 lines)
4. `src/training/fusion_training.py` (~300 lines)
5. `src/training/memory_preserving_curriculum.py` (~200 lines)
6. `src/training/momentum_curriculum.py` (~150 lines)
7. `src/training/structure_aware_sampling.py` (~100 lines)
8. `src/training/fusion_extractor.py` (~50 lines)
9. `src/utils/experiment_paths.py` (~220 lines)

**Total:** 9 files, ~3,700 lines of obsolete code removed

## New Core Architecture

### 1. Unified Path Management: `src/paths.py` (573 lines)

**Purpose:** Single source of truth for all path resolution

```python
from src.paths import PathResolver, DATASET_PATHS, resolve_dataset_path

# Initialize
resolver = PathResolver(cfg)

# Get dataset paths
dataset_path = resolve_dataset_path("hcrl_ch", modality="automotive")

# Get cache paths
cache_info = resolver.get_cache_paths(cfg)
# Returns: {'cache_dir': Path, 'embeddings_file': Path, 'metadata_file': Path}

# Get experiment directory
exp_dir = resolver.get_experiment_dir(cfg)
# Returns: experiment_root/modality/dataset/learning_type/architecture/size/distillation/mode

# Resolve teacher model path
teacher_path = resolver.resolve_teacher_path(cfg)
```

**Key Features:**
- Validates paths exist
- Handles all experiment directory hierarchy
- Supports all modalities (automotive, robotics, etc.)

### 2. Unified Training Orchestrator: `src/training/trainer.py` (560 lines)

**Purpose:** Main entry point for all training modes

```python
from src.training.trainer import HydraZenTrainer

# Initialize
trainer = HydraZenTrainer(cfg)

# Train (automatically handles fusion/curriculum/standard modes)
result = trainer.train()

# Components:
# - setup_model(): Creates Lightning module
# - setup_trainer(): Configures PyTorch Lightning trainer
# - train(): Main orchestration logic
# - _train_fusion(): DQN-based fusion with prediction caching
# - _train_curriculum(): Curriculum learning with difficulty progression
# - _train_standard(): Standard supervised/unsupervised training
```

**Key Features:**
- Mode dispatch: Automatically selects training logic based on `cfg.training.mode`
- Path management: Uses PathResolver for all paths
- Checkpoint handling: Manages model saving/loading
- Teacher loading: Handles distillation and fusion requirements

### 3. Consolidated Lightning Modules: `src/training/lightning_modules.py` (1,187 lines)

**Purpose:** All PyTorch Lightning modules in one place

```python
from src.training.lightning_modules import (
    VAELightningModule,      # VGAE for unsupervised
    GATLightningModule,      # GAT for classification
    DQNLightningModule,      # DQN for fusion
    FusionLightningModule,   # DQN fusion with prediction cache
)

# All modules inherit from BaseKDGATModule
# path_manager is now OPTIONAL (trainer handles paths)
module = GATLightningModule(cfg, path_manager=None)
```

**Breaking Change:**
- `path_manager` parameter is now **optional** (defaults to `None`)
- Trainer handles all path management through PathResolver
- Modules no longer save metrics directly (trainer does this)

### 4. Consolidated Data Modules: `src/training/datamodules.py` (822 lines)

**Purpose:** All data loading logic unified

```python
from src.training.datamodules import (
    CANGraphDataModule,
    AdaptiveGraphDataModule,
)

# Uses PathResolver for dataset/cache paths
datamodule = CANGraphDataModule(cfg)
```

**Key Features:**
- Uses PathResolver for all dataset/cache paths
- Supports synthetic data generation
- Handles train/val/test splits
- Manages batch processing

### 5. Simplified Main Script: `train_with_hydra_zen.py` (348 lines, 74% reduction)

**Before:** 1,337 lines with embedded 990-line HydraZenTrainer class
**After:** 348 lines - just imports and CLI setup

```python
from src.training.trainer import HydraZenTrainer

def main():
    cfg = parse_cli_args()
    trainer = HydraZenTrainer(cfg)
    result = trainer.train()
```

## New Workflow Examples

### Example 1: Train VGAE (Unsupervised)

```bash
python train_with_hydra_zen.py \
    model=vgae_student \
    dataset=hcrl_ch \
    training=autoencoder
```

**What happens:**
1. `train_with_hydra_zen.py` parses CLI args
2. Creates `HydraZenTrainer(cfg)`
3. Trainer calls `setup_model()` → creates `VAELightningModule`
4. Trainer calls `setup_trainer()` → configures PyTorch Lightning
5. Trainer calls `train()` → dispatches to `_train_standard()`
6. Model saved to: `experiment_root/automotive/hcrl_ch/unsupervised/vgae/student/no/autoencoder/`

### Example 2: Train GAT with Distillation

```bash
python train_with_hydra_zen.py \
    model=gat_student \
    dataset=hcrl_ch \
    training=knowledge_distillation \
    teacher_model=/path/to/teacher.ckpt
```

**What happens:**
1. Trainer loads teacher model from `teacher_model` path
2. Creates `GATLightningModule` with distillation config
3. Training uses teacher outputs as soft targets
4. Model saved with `distillation=kd` in path hierarchy

### Example 3: Train Fusion Agent

```bash
python train_with_hydra_zen.py \
    model=dqn_fusion \
    dataset=hcrl_ch \
    training=fusion \
    teacher_model=/path/to/teacher.ckpt
```

**What happens:**
1. Trainer dispatches to `_train_fusion()`
2. Loads teacher model for generating predictions
3. Creates `FusionLightningModule` with prediction cache
4. DQN learns to select between student/teacher predictions
5. Fusion agent saved to: `experiment_root/automotive/hcrl_ch/reinforcement_learning/dqn/student/no/fusion/`

### Example 4: Curriculum Learning

```bash
python train_with_hydra_zen.py \
    model=gat_student \
    dataset=hcrl_ch \
    training=curriculum
```

**What happens:**
1. Trainer dispatches to `_train_curriculum()`
2. Uses `src/training/modes/curriculum.py` for difficulty-based sampling
3. Progressively increases difficulty during training
4. Model learns from easy → hard examples

## Path Structure Reference

```
experiment_root/
└── {modality}/              # e.g., automotive, robotics
    └── {dataset}/           # e.g., hcrl_ch, hcrl_sa
        └── {learning_type}/ # e.g., supervised, unsupervised, reinforcement_learning
            └── {architecture}/ # e.g., gat, vgae, dqn
                └── {size}/     # e.g., student, teacher
                    └── {distillation}/ # e.g., no, kd, at
                        └── {mode}/     # e.g., standard, fusion, curriculum
                            ├── checkpoints/
                            ├── logs/
                            └── models/
```

## Migration Checklist for Users

If you have custom scripts that used old modules:

- [x] ~~Replace `ExperimentPathManager`~~ → Use `PathResolver` from `src.paths`
- [x] ~~Import from old `_old_modules/`~~ → Use consolidated modules from `src.training.*`
- [x] ~~Call `train_with_hydra.py`~~ → Use `train_with_hydra_zen.py`
- [x] ~~Use old fusion_training.py~~ → Built into `HydraZenTrainer`
- [x] ~~Import path utils from multiple files~~ → Use `src.paths`

**Note:** All old files are DELETED. If you need them, restore from git history.

## Testing

After migration, all tests pass:

```bash
pytest tests/test_load_dataset.py -v
pytest tests/test_adaptive_graph_dataset.py -v
```

## Key Design Decisions

1. **Path management centralized**: No more scattered path logic across 5+ files
2. **Trainer orchestration**: One class handles all training modes instead of separate scripts
3. **Optional path_manager**: Lightning modules no longer require ExperimentPathManager
4. **Backward compatibility**: Old configs still work (PathResolver handles translation)
5. **Aggressive cleanup**: Removed ~3,700 lines of obsolete code

## Performance Impact

- **Reduced complexity**: 74% reduction in main script (1,337 → 348 lines)
- **No runtime overhead**: Same training performance
- **Faster development**: Single file to edit for training changes
- **Better testing**: Isolated components easier to test

## Troubleshooting

### "Module not found: src.utils.experiment_paths"

**Solution:** This module was deleted. Update imports to:
```python
from src.paths import PathResolver
```

### "path_manager parameter missing"

**Solution:** path_manager is now optional. Either:
- Pass `None`: `module = GATLightningModule(cfg, path_manager=None)`
- Omit parameter: `module = GATLightningModule(cfg)`

### "Old training script doesn't work"

**Solution:** Old scripts in `_old_modules/` were deleted. Use:
```bash
python train_with_hydra_zen.py [options]
```

## Related Documentation

- [MIGRATION_PLAN.md](MIGRATION_PLAN.md) - Original migration strategy
- [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) - System architecture
- [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md) - Environment setup

## Status: COMPLETE ✅

All migration tasks completed:
- [x] Consolidated modules (datamodules.py, lightning_modules.py)
- [x] Unified paths (src/paths.py)
- [x] Simplified training (src/training/trainer.py)
- [x] Removed obsolete files (9 files, ~3,700 lines)
- [x] Updated all imports
- [x] Fixed scripts (local_smoke_experiment.py)
- [x] Verified tests pass

**Migration Date:** 2025-01-23
**Code Reduction:** ~3,700 lines removed
**Files Deleted:** 9 obsolete files
