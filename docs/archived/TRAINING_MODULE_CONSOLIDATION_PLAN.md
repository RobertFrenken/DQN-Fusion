# Training Module Consolidation Plan

**Goal**: Reduce 14 training-related files to 4 focused modules  
**Timeline**: Week 2-3 of refactoring roadmap  
**Status**: Planning phase

---

## Current State (14 Files)

```
src/training/
├── train_with_hydra.py              # 350 lines - OLD YAML-based trainer
├── can_graph_module.py              # 311 lines - Lightning module v1
├── lightning_modules.py             # 405 lines - Lightning module v2
├── fusion_lightning.py              # ??? lines - Fusion-specific Lightning
├── fusion_training.py               # 949 lines - Standalone fusion trainer
├── enhanced_datamodule.py           # ??? lines - Enhanced DataModule
├── can_graph_data.py                # ??? lines - Basic DataModule
├── memory_preserving_curriculum.py  # ??? lines - Curriculum training
├── prediction_cache.py              # ??? lines - Cache utilities
├── fusion_extractor.py              # ??? lines - Fusion data extraction
├── distillation.py                  # ??? lines - KD logic
├── autoencoder_training.py          # ??? lines - Autoencoder-specific
├── training_utils.py                # ??? lines - Misc utilities
└── batch_size_optimizer.py          # ??? lines - (if exists)

NEW (just created):
├── batch_optimizer.py               # 350 lines - DONE ✅
├── model_manager.py                 # 280 lines - DONE ✅
└── modes/
    ├── __init__.py                  # DONE ✅
    ├── fusion.py                    # 340 lines - DONE ✅
    └── curriculum.py                # 290 lines - DONE ✅
```

---

## Target State (4 Core Modules + Modes)

```
src/training/
├── datamodules.py          # Unified DataModule (all variants)
├── lightning_modules.py    # All Lightning modules (GAT/VGAE/DQN)
├── batch_optimizer.py      # Batch size optimization ✅
├── model_manager.py        # Model loading/saving ✅
└── modes/
    ├── __init__.py         # ✅
    ├── fusion.py           # ✅
    ├── curriculum.py       # ✅
    ├── normal.py           # Standard supervised training
    ├── autoencoder.py      # VGAE autoencoder training
    └── distillation.py     # Knowledge distillation training
```

---

## Consolidation Strategy

### Phase 1: DataModule Unification (Days 1-2)

**Goal**: Merge `can_graph_data.py` + `enhanced_datamodule.py` → `datamodules.py`

**Steps**:
1. **Analyze differences**:
   - `can_graph_data.py`: Basic DataModule + load_dataset + create_dataloaders
   - `enhanced_datamodule.py`: Curriculum-specific DataModule with hard mining

2. **Create unified datamodules.py**:
   ```python
   # src/training/datamodules.py
   
   class CANGraphDataModule(pl.LightningDataModule):
       """Standard DataModule for GAT/VGAE training."""
       pass
   
   class EnhancedCANGraphDataModule(pl.LightningDataModule):
       """Enhanced DataModule with curriculum learning."""
       pass
   
   def load_dataset(dataset_name, config, force_rebuild_cache=False):
       """Load and cache graph dataset."""
       pass
   
   def create_dataloaders(train_dataset, val_dataset, batch_size, **kwargs):
       """Create PyTorch Geometric dataloaders."""
       pass
   ```

3. **Update imports** across codebase
4. **Delete** `can_graph_data.py` and `enhanced_datamodule.py`

**Files impacted**: ~20 files import from these modules

---

### Phase 2: Lightning Module Consolidation (Days 3-4)

**Goal**: Merge `can_graph_module.py` + `lightning_modules.py` + `fusion_lightning.py` → `lightning_modules.py`

**Analysis**:
- `can_graph_module.py` (311 lines): 
  - CANGraphLightningModule (handles GAT, VGAE, DQN, all training modes)
- `lightning_modules.py` (405 lines):
  - BaseKDGATModule
  - VAELightningModule
  - GATLightningModule  
  - DQNLightningModule
- `fusion_lightning.py`:
  - FusionLightningModule
  - FusionPredictionCache

**Strategy**: Keep `can_graph_module.py` as base, extract specialized logic

**Create unified lightning_modules.py**:
```python
# src/training/lightning_modules.py

class BaseCANGraphModule(pl.LightningModule):
    """Base Lightning module with shared functionality."""
    pass

class GATModule(BaseCANGraphModule):
    """GAT classifier module."""
    pass

class VGAEModule(BaseCANGraphModule):
    """VGAE autoencoder module."""
    pass

class DQNModule(BaseCANGraphModule):
    """DQN reinforcement learning module."""
    pass

class FusionModule(BaseCANGraphModule):
    """Fusion agent module."""
    pass

# Keep CANGraphLightningModule for backward compatibility
class CANGraphLightningModule(BaseCANGraphModule):
    """Legacy unified module (delegates to specialized modules)."""
    pass
```

**Files impacted**: train_with_hydra_zen.py, all training modes, tests

---

### Phase 3: Fusion Logic Consolidation (Days 5-6)

**Goal**: Merge `fusion_training.py` + `fusion_extractor.py` + `prediction_cache.py` → `modes/fusion.py`

**Current**:
- `fusion_training.py` (949 lines): Standalone fusion pipeline
- `fusion_extractor.py`: Extract predictions from models
- `prediction_cache.py`: Cache management

**Already done**: `modes/fusion.py` ✅  
**Still needed**:
- Extract prediction_cache logic into `modes/fusion.py`
- Remove `fusion_training.py` (legacy standalone)
- Keep `prediction_cache.py` only if used elsewhere

---

### Phase 4: Training Mode Extraction (Days 7-8)

**Goal**: Extract remaining training logic from `train_with_hydra_zen.py`

**Create**:
- `modes/normal.py` - Standard supervised training
- `modes/autoencoder.py` - VGAE reconstruction training
- `modes/distillation.py` - Knowledge distillation

**Pattern** (already established in fusion.py):
```python
class NormalTrainer:
    def __init__(self, config, paths):
        self.config = config
        self.paths = paths
    
    def train(self) -> Tuple[Module, Trainer]:
        # Setup, train, save
        pass
```

---

### Phase 5: Cleanup Legacy Files (Days 9-10)

**Delete**:
- [x] ~~`train_with_hydra.py`~~ (YAML-based, replaced by train_with_hydra_zen.py)
- [ ] `fusion_training.py` (replaced by modes/fusion.py)
- [ ] `autoencoder_training.py` (merge into modes/autoencoder.py)
- [ ] `memory_preserving_curriculum.py` (functionality in enhanced_datamodule.py)
- [ ] `distillation.py` → `modes/distillation.py`
- [ ] `training_utils.py` (distribute utilities to relevant modules)

**Deprecate with warnings** (remove in next major version):
- `can_graph_module.CANGraphLightningModule` (still used in many places)

---

## Dependency Analysis

### What imports what (high-level):

```
train_with_hydra_zen.py
  ├─ can_graph_module.py (CANGraphLightningModule)
  ├─ can_graph_data.py (load_dataset, create_dataloaders, CANGraphDataModule)
  ├─ enhanced_datamodule.py (EnhancedCANGraphDataModule, CurriculumCallback)
  ├─ fusion_lightning.py (FusionLightningModule)
  └─ prediction_cache.py (create_fusion_prediction_cache)

fusion_training.py (standalone)
  ├─ models.py (GATWithJK, GraphAutoencoderNeighborhood, EnhancedDQNFusionAgent)
  ├─ fusion_extractor.py (FusionDataExtractor)
  └─ cache_manager.py

tests/
  └─ Many tests import CANGraphLightningModule, CANGraphDataModule
```

**Critical path**: `can_graph_module.py` is imported by:
- train_with_hydra_zen.py
- modes/fusion.py
- modes/curriculum.py
- Many test files

**Strategy**: Keep backward-compatible wrapper

---

## Risk Mitigation

### High Risk:
1. **Breaking existing checkpoints**: Lightning module renaming
   - Mitigation: Keep legacy CANGraphLightningModule wrapper
   
2. **Test failures**: Extensive imports
   - Mitigation: Phase changes, run tests after each phase

3. **Import cycles**: New module structure
   - Mitigation: Careful dependency planning, use TYPE_CHECKING

### Medium Risk:
4. **SLURM job compatibility**: Changing module paths
   - Mitigation: Test on small jobs before full runs

5. **Config incompatibility**: Changed module names
   - Mitigation: Add migration utility

---

## Testing Plan

After each phase:

```bash
# 1. Unit tests
pytest tests/ -v

# 2. Import smoke test
python -c "from src.training.datamodules import *; from src.training.lightning_modules import *; print('✅')"

# 3. Quick training smoke test
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal --config.training.max_epochs=1

# 4. Check for import errors in all Python files
ruff check src/ tests/ --select F401,F811
```

---

## Success Metrics

| Metric | Current | Target | After Phase |
|--------|---------|--------|-------------|
| Training modules | 14 | 4 (+modes/) | TBD |
| Duplicate code | High | Low | TBD |
| Import complexity | 20+ imports | <10 imports | TBD |
| Avg file length | ~500 lines | ~300 lines | TBD |
| Test coupling | High (mocking) | Medium | TBD |

---

## Implementation Checklist

### Phase 1: DataModules ☐
- [ ] Create `src/training/datamodules.py`
- [ ] Move CANGraphDataModule from can_graph_data.py
- [ ] Move EnhancedCANGraphDataModule from enhanced_datamodule.py
- [ ] Move load_dataset, create_dataloaders helpers
- [ ] Update imports in train_with_hydra_zen.py
- [ ] Update imports in modes/fusion.py
- [ ] Update imports in modes/curriculum.py
- [ ] Update imports in tests/
- [ ] Run tests
- [ ] Delete old files

### Phase 2: Lightning Modules ☐
- [ ] Create unified `src/training/lightning_modules.py`
- [ ] Move/merge all Lightning module classes
- [ ] Add backward-compatible wrapper
- [ ] Update imports across codebase
- [ ] Run tests (especially checkpoint loading tests)
- [ ] Delete old files

### Phase 3: Fusion Consolidation ☐
- [ ] Move prediction cache logic to modes/fusion.py
- [ ] Remove fusion_training.py (after extracting any missing logic)
- [ ] Update any external scripts using fusion_training.py
- [ ] Run fusion smoke test
- [ ] Delete old files

### Phase 4: Training Mode Extraction ☐
- [ ] Create modes/normal.py
- [ ] Create modes/autoencoder.py
- [ ] Create modes/distillation.py
- [ ] Extract logic from train_with_hydra_zen.py
- [ ] Update train_with_hydra_zen.py to use new trainers
- [ ] Run full training smoke tests

### Phase 5: Legacy Cleanup ☐
- [ ] Add deprecation warnings to legacy modules
- [ ] Update documentation
- [ ] Create migration guide
- [ ] Delete truly unused files
- [ ] Final full test suite run

---

## Timeline

| Week | Days | Phase | Deliverable |
|------|------|-------|-------------|
| 2 | Mon-Tue | Phase 1 | datamodules.py ✅ |
| 2 | Wed-Thu | Phase 2 | lightning_modules.py ✅ |
| 2 | Fri | Phase 3 | Fusion consolidation |
| 3 | Mon-Tue | Phase 4 | Training mode extraction |
| 3 | Wed-Thu | Phase 5 | Legacy cleanup |
| 3 | Fri | - | Documentation + final testing |

---

## Notes

- All new modules follow the pattern established in `batch_optimizer.py` and `model_manager.py`
- Type hints required for all new code
- Docstrings required for all public methods
- No silent exception swallowing (log all errors)
- Prefer explicit failures over fallbacks

---

**Next Step**: Begin Phase 1 (DataModule unification)  
**Blocked by**: None - can start immediately  
**Owner**: TBD
