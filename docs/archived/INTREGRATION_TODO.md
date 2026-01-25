# Integration TODO Checklist

Use this file to track your integration progress. Check off items as you complete them.

## Phase 1: Preparation (0.5 hours)

- [ ] Read `INTEGRATION_SUMMARY.md` (this gives you the big picture)
- [ ] Read `INTEGRATION_CODE_TEMPLATES.md` (shows exact code patterns)
- [ ] Understand the 5-step integration process
- [ ] Set aside 3-4 hours of focused time

**Status: Start here** ← You are probably here

---

## Phase 2: Dataset Integration (0.5 hours)

### Examine Current Code
- [ ] Open your `src/data/datasets.py` file
- [ ] Find your HCRLCHDataset class
- [ ] Find how you're currently loading and splitting data
- [ ] Take notes on current structure

### Modify Dataset Class
- [ ] Create GraphDataset wrapper class (see INTEGRATION_CODE_TEMPLATES.md Template 1)
- [ ] Update HCRLCHDataset.__init__ to:
  - [ ] Take parameters: `data_path`, `split_ratio`, `normalization`
  - [ ] Load data using your existing logic
  - [ ] Apply normalization
  - [ ] Split into train/val/test
  - [ ] Create `.train`, `.val`, `.test` as GraphDataset objects
- [ ] Create Set01Dataset, Set02Dataset, Set03Dataset, Set04Dataset (can inherit from HCRLCHDataset)
- [ ] Ensure all return PyTorch Dataset objects

### Test Dataset
- [ ] Run: `python3 << 'EOF'
from src.data.datasets import HCRLCHDataset
dataset = HCRLCHDataset('./data/automotive/hcrlch', (0.7, 0.15, 0.15), 'zscore')
print(f"✅ Train: {len(dataset.train)}, Val: {len(dataset.val)}, Test: {len(dataset.test)}")
EOF`
- [ ] Verify output shows batch counts

**Expected Time: 30 minutes**

---

## Phase 3: Model Integration (0.5 hours)

### Update VGAE Model
- [ ] Open `src/models/vgae.py`
- [ ] Find the `__init__` method
- [ ] Change from hardcoded values to parameters:
  - [ ] Add `input_dim: int = 128`
  - [ ] Add `hidden_dim: int = 64`
  - [ ] Add `latent_dim: int = 32`
  - [ ] Add `num_layers: int = 2`
  - [ ] Add `dropout: float = 0.2`
  - [ ] Add `**kwargs` at end
- [ ] Update forward() to accept `(x, edge_index)` (edge_index can be None)
- [ ] Keep all your implementation logic exactly the same

### Update GAT Model
- [ ] Open `src/models/gat.py`
- [ ] Add same parameter pattern to `__init__`
- [ ] Add: `input_dim`, `hidden_dim`, `output_dim`, `num_heads`, `num_layers`, `dropout`, `**kwargs`
- [ ] Ensure forward() signature: `forward(self, x, edge_index)`

### Update DQN Model (if present)
- [ ] Open `src/models/dqn.py`
- [ ] Add parameters: `input_dim`, `hidden_dim`, `action_dim`, `num_layers`, `dropout`, `dueling`, `**kwargs`
- [ ] Ensure forward() signature: `forward(self, x)`

### Test Models
- [ ] Run: `python3 -c "from src.models.vgae import VGAE; m = VGAE(hidden_dim=64, latent_dim=32, num_layers=2, dropout=0.1); print('✅ VGAE OK')"`
- [ ] Run: `python3 -c "from src.models.gat import GAT; m = GAT(input_dim=128, hidden_dim=64, output_dim=128, num_heads=4, num_layers=2); print('✅ GAT OK')"`
- [ ] Both should print ✅

**Expected Time: 30 minutes**

---

## Phase 4: Data Loader Integration (0.5 hours)

### Implement load_data_loaders()
- [ ] Open `src/training/train_with_hydra_zen.py`
- [ ] Find the `load_data_loaders(cfg)` placeholder function
- [ ] Add imports at top:
  ```python
  from src.data.datasets import (
      HCRLCHDataset, Set01Dataset, Set02Dataset, Set03Dataset, Set04Dataset
  )
  from torch.utils.data import DataLoader
  ```
- [ ] Create DATASET_MAP dictionary:
  ```python
  DATASET_MAP = {
      'hcrlch': HCRLCHDataset,
      'set01': Set01Dataset,
      # ... more datasets
  }
  ```
- [ ] Replace function body (see INTEGRATION_CODE_TEMPLATES.md Template 3)
- [ ] Ensure it returns (train_loader, val_loader, test_loader)

### Test Data Loaders
- [ ] Create test script and run it
- [ ] Run script and verify batch counts print
- [ ] Get first batch: `batch = next(iter(train_loader))` should work

**Expected Time: 20 minutes**

---

## Phase 5: Configuration Update (0.5 hours)

### Update config_store.py
- [ ] Open `hydra_configs/config_store.py`
- [ ] Find the top section with path definitions
- [ ] Set `project_root` to your actual KD-GAT directory
- [ ] Verify `data_root` and `experiment_root` are correct

### Update Dataset Configs
- [ ] Find dataset config classes (HCRLCHDataset, Set01Dataset, etc.)
- [ ] Update `data_path` for each to match your actual data
- [ ] Check: `ls -la ./data/automotive/hcrlch/` works
- [ ] Check: `ls -la ./data/automotive/set01/` works
- [ ] Update paths in config to match

### Update Model Configs
- [ ] Find model config dataclasses (VGAEModelConfig, GATModelConfig, etc.)
- [ ] Update `_target_` paths to match your actual class locations
- [ ] Update parameters to match your models

### Test Configuration
- [ ] Run: `python src/training/train_with_hydra_zen.py config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples --cfg job | head -30`
- [ ] Verify: No errors, config prints with your paths

**Expected Time: 20 minutes**

---

## Phase 6: Lightning Module Integration (0.5 hours)

### Update VGAE Lightning Module
- [ ] Open `src/training/lightning_modules.py`
- [ ] Find `VAELightningModule._build_vgae()` method
- [ ] Replace placeholder with actual model instantiation
- [ ] Check what parameters your VGAE needs
- [ ] Make sure all config fields exist

### Update GAT Lightning Module
- [ ] Find `GATLightningModule._build_gat()` method
- [ ] Update with correct parameters from your config

### Update DQN Lightning Module (if present)
- [ ] Find `DQNLightningModule._build_dqn()` method
- [ ] Update with correct parameters

### Test Lightning Modules
- [ ] Run basic module creation test
- [ ] Verify models instantiate without errors

**Expected Time: 20 minutes**

---

## Phase 7: Testing (1-2 hours)

### Level 1: Component Tests
- [ ] Dataset loading works
- [ ] Models instantiate correctly
- [ ] Data loaders return batches

### Level 2: Configuration Test
- [ ] Config loads: `python src/training/train_with_hydra_zen.py config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples --cfg job`

### Level 3: Single-Epoch Training
- [ ] Training runs: `python src/training/train_with_hydra_zen.py config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples device=cpu training_config.epochs=1`
- [ ] Data loads
- [ ] Model builds
- [ ] Training runs 1 epoch
- [ ] Results save to experimentruns/

### Level 4: Check Output Structure
- [ ] Results saved correctly
- [ ] Should have: `model.pt`, `config.yaml`, `checkpoints/`, `training_metrics.json`

### Level 5: Slurm Script Test
- [ ] Script generation works: `python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_student_no_all_samples --dry-run`

**Expected Time: 1-2 hours depending on debugging needed**

---

## Phase 8: Debugging (As Needed)

If you hit errors:

- [ ] Read error message carefully
- [ ] Search in `INTEGRATION_DEBUGGING.md` for matching error
- [ ] Follow solution provided
- [ ] Test component in isolation
- [ ] Verify file paths
- [ ] Check config with `--cfg job`

**Common Issues:**
- [ ] "ModuleNotFoundError" → Check directory and imports
- [ ] "Config not found" → Check config_store.py registration
- [ ] "Dataset class not found" → Check DATASET_MAP
- [ ] "Data path not found" → Check directory structure
- [ ] "Model instantiation failed" → Check model `__init__` parameters

---

## Success Checklist

When everything is working, you should be able to:

- [ ] Run single-epoch training successfully
- [ ] Results save to correct directory structure
- [ ] Config file saved in run directory
- [ ] Can reproduce results with saved config
- [ ] Understand the experiment naming convention
- [ ] Know where to find results
- [ ] Slurm script generation works
- [ ] Can run hyperparameter sweeps
- [ ] Can submit to Slurm

---

## Timeline

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 1 | Read docs & understand | 0.5h | ⭐ Start here |
| 2 | Dataset integration | 0.5h | ⬜ Not started |
| 3 | Model integration | 0.5h | ⬜ Not started |
| 4 | Data loaders | 0.5h | ⬜ Not started |
| 5 | Config update | 0.5h | ⬜ Not started |
| 6 | Lightning modules | 0.5h | ⬜ Not started |
| 7 | Testing | 1-2h | ⬜ Not started |
| 8 | Debugging | ? | ⬜ As needed |
| **Total** | **Complete integration** | **4-5h** | |

---

## Notes Section

Use this area to track issues, questions, or learnings:

```
Date: ___________

Issue encountered: _______________________________________

Solution applied: _________________________________________

Notes: ________________________________________________
```

---

**Remember:** Take this one step at a time. Test after each phase. Good luck! 🚀
