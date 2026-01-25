# Aggressive Migration to New Code Structure

## ✅ New Workflow

### Training Entry Points
**Primary (Recommended):**
```bash
# Use the simplified main script
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal
python train_with_hydra_zen.py --preset gat_normal_hcrl_sa
```

**Programmatic:**
```python
from src.training.trainer import HydraZenTrainer
from src.config.hydra_zen_configs import create_gat_normal_config

config = create_gat_normal_config('hcrl_sa')
trainer = HydraZenTrainer(config)
model, lightning_trainer = trainer.train()
```

### Module Imports
**Data Loading:**
```python
from src.training.datamodules import (
    load_dataset,              # Load and cache datasets
    create_dataloaders,        # Create PyG dataloaders
    CANGraphDataModule,        # Standard datamodule
    EnhancedCANGraphDataModule,# Curriculum datamodule
    CurriculumCallback         # Curriculum callback
)
```

**Models:**
```python
from src.training.lightning_modules import (
    CANGraphLightningModule,   # Main module (GAT/VGAE/DQN)
    FusionLightningModule,     # Fusion DQN module
    FusionPredictionCache      # Fusion cache helper
)
```

**Path Management:**
```python
from src.paths import PathResolver

resolver = PathResolver(config)
dataset_path = resolver.resolve_dataset_path('hcrl_sa')
paths = resolver.get_all_experiment_dirs(create=True)
```

**Training:**
```python
from src.training.trainer import HydraZenTrainer

trainer = HydraZenTrainer(config)
model, trainer_obj = trainer.train()
```

## 🗑️ Files to DELETE (Obsolete)

### Phase 1: Remove OLD Training Files
```bash
# Already archived - can DELETE entirely
rm -rf src/training/_old_modules/

# Obsolete duplicates
rm src/training/lightning_modules_OLD.py
rm src/training/train_with_hydra.py
rm src/training/fusion_training.py
```

### Phase 2: Remove Experimental/Unused Files
```bash
# These were experimental and not used in new structure
rm src/training/memory_preserving_curriculum.py
rm src/training/momentum_curriculum.py
rm src/training/structure_aware_sampling.py
rm src/training/fusion_extractor.py
```

### Phase 3: Update or Remove Utils
```bash
# experiment_paths.py is superseded by src/paths.py
# BUT: lightning_modules.py still uses it, needs migration first
# After migration: rm src/utils/experiment_paths.py
```

## ⚠️ Files That Need Migration

### 1. src/training/lightning_modules.py
**Issue:** Still imports `ExperimentPathManager` from old `src/utils/experiment_paths`

**Fix:** Replace with PathResolver
```python
# OLD:
from src.utils.experiment_paths import ExperimentPathManager
self.path_manager = path_manager  # ExperimentPathManager instance

# NEW:
from src.paths import PathResolver
self.path_resolver = PathResolver(config)
```

**Impact:** 16 references in lightning_modules.py need updating

### 2. scripts/local_smoke_experiment.py
**Issue:** Uses ExperimentPathManager

**Fix:** Replace with PathResolver or remove (if obsolete)

## 📋 Migration Steps

### Step 1: Fix lightning_modules.py (REQUIRED)
Remove ExperimentPathManager dependency:
- Replace `path_manager: ExperimentPathManager` parameters
- Use PathResolver or remove path manager entirely (trainer handles paths)
- Lightning modules shouldn't manage paths - that's the trainer's job

### Step 2: Delete Obsolete Files (SAFE)
```bash
# These are 100% safe to delete
rm -rf src/training/_old_modules/
rm src/training/lightning_modules_OLD.py
rm src/training/train_with_hydra.py
rm src/training/fusion_training.py
rm src/training/memory_preserving_curriculum.py
rm src/training/momentum_curriculum.py
rm src/training/structure_aware_sampling.py
rm src/training/fusion_extractor.py
```

### Step 3: Remove experiment_paths.py (After Step 1)
```bash
# Only after fixing lightning_modules.py
rm src/utils/experiment_paths.py
```

## 📊 Cleanup Impact

### Files to DELETE (8 files, ~3,500 lines):
- `src/training/_old_modules/` (4 files, ~1,500 lines)
- `src/training/lightning_modules_OLD.py` (~400 lines)
- `src/training/train_with_hydra.py` (~800 lines)
- `src/training/fusion_training.py` (~300 lines)
- `src/training/memory_preserving_curriculum.py` (~200 lines)
- `src/training/momentum_curriculum.py` (~150 lines)
- `src/training/structure_aware_sampling.py` (~100 lines)
- `src/training/fusion_extractor.py` (~50 lines)

### After Migration:
- `src/utils/experiment_paths.py` (~220 lines)

**Total cleanup:** ~3,700 lines of obsolete code removed!

## ✨ Final Structure

```
src/
├── paths.py                          # ✅ NEW: Unified path management
├── training/
│   ├── trainer.py                    # ✅ NEW: Training orchestrator
│   ├── datamodules.py                # ✅ CONSOLIDATED
│   ├── lightning_modules.py          # ✅ CONSOLIDATED (needs path fix)
│   ├── modes/
│   │   ├── fusion.py                 # ✅ Mode trainer
│   │   └── curriculum.py             # ✅ Mode trainer
│   ├── batch_optimizer.py            # ✅ Keep
│   ├── model_manager.py              # ✅ Keep
│   └── prediction_cache.py           # ✅ Keep
├── config/
│   └── hydra_zen_configs.py          # ✅ Keep
└── utils/
    └── experiment_paths.py           # ❌ DELETE after migration

train_with_hydra_zen.py               # ✅ SIMPLIFIED (348 lines)
```

## 🚀 Next Action

**Execute aggressive cleanup:**
```bash
cd /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT

# Step 1: Delete safe obsolete files
rm -rf src/training/_old_modules/
rm src/training/lightning_modules_OLD.py
rm src/training/train_with_hydra.py
rm src/training/fusion_training.py
rm src/training/memory_preserving_curriculum.py
rm src/training/momentum_curriculum.py
rm src/training/structure_aware_sampling.py
rm src/training/fusion_extractor.py

# Step 2: Fix lightning_modules.py path management (manual)
# Then: rm src/utils/experiment_paths.py

# Step 3: Test
python train_with_hydra_zen.py --help
pytest tests/test_can_graph_data_strict.py -v
```

This removes ~3,700 lines of obsolete code and completes the transition! 🎉
