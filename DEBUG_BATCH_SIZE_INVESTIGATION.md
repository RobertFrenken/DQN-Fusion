# Debug Batch Size Investigation - Quick Reference

## What We Added

### 1. Comprehensive Test Suite

**File**: [tests/test_dataloader_batch_behavior.py](tests/test_dataloader_batch_behavior.py)

Three test scenarios:
1. **Test 1: Simple synthetic graphs** - Verifies PyG DataLoader works correctly with basic synthetic data
2. **Test 2: Real CAN dataset** - Tests DataLoader with actual hcrl_sa dataset
3. **Test 3: Curriculum datamodule** - Tests EnhancedCANGraphDataModule specifically

**To run**:
```bash
cd /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT
python tests/test_dataloader_batch_behavior.py
```

**Expected output if working correctly**:
```
PyG DataLoader Batch Behavior Investigation
================================================================================

================================================================================
TEST 1: Simple PyG DataLoader with synthetic graphs
================================================================================
Created 100 synthetic graphs

DataLoader Configuration:
  batch_size: 32
  Expected batches: 4
  len(dataloader): 4
  Batch 1: 32 graphs
  Batch 2: 32 graphs
  Batch 3: 32 graphs
  Batch 4: 4 graphs

Results:
  Total batches: 4
  Batch sizes: [32, 32, 32, 4]... (showing first 5)
  ✅ PASS: Got expected 4 batches

[... similar for tests 2 and 3 ...]

================================================================================
SUMMARY
================================================================================
synthetic            : ✅ PASS
real_dataset         : ✅ PASS
curriculum           : ✅ PASS

✅ All tests passed! DataLoader behavior is normal.
```

**Expected output if bug reproduced**:
```
  Batch 1: 9364 graphs
Results:
  Total batches: 1
  ❌ FAIL: Only 1 batch! Bug reproduced in curriculum datamodule.
```

---

### 2. Enhanced Debug Logging

#### Added to: [src/training/datamodules.py](src/training/datamodules.py)

**Lines 465-482**: `train_dataloader()` method
```python
def train_dataloader(self):
    logger.info(f"🔍 Creating train dataloader:")
    logger.info(f"   Dataset length: {len(self.train_dataset)}")
    logger.info(f"   Batch size: {self.batch_size}")
    logger.info(f"   Expected batches: {len(self.train_dataset) // self.batch_size + ...}")
    logger.info(f"   Num workers: {self.num_workers}")

    dataloader = DataLoader(...)

    logger.info(f"   Actual batches (len(dataloader)): {len(dataloader)}")
    return dataloader
```

**Lines 484-498**: `val_dataloader()` method (similar logging)

**What this tells us**:
- Dataset size at dataloader creation time
- Configured batch_size
- Expected number of batches (calculated)
- Actual number of batches (from PyG DataLoader)

---

#### Added to: [src/training/modes/curriculum.py](src/training/modes/curriculum.py)

**Lines 54-68**: `train()` method - Entry point logging
```python
logger.info("🎓 Starting GAT training with curriculum learning + hard mining")
# Load and separate dataset
datamodule, vgae_model = self._setup_curriculum_datamodule(num_ids)
logger.info(f"📊 Datamodule created with batch_size={datamodule.batch_size}")

# Create GAT model
gat_model = self._create_gat_model(num_ids)
logger.info(f"🧠 GAT model created")

# Optimize batch size
logger.info(f"🔧 About to optimize batch size (current: {datamodule.batch_size})")
gat_model = self._optimize_batch_size_for_curriculum(gat_model, datamodule)
logger.info(f"✅ Batch size optimization complete (final: {datamodule.batch_size})")
```

**Lines 99-107**: `_setup_curriculum_datamodule()` - Dataset separation logging
```python
logger.info("📊 Separating normal and attack graphs...")
train_normal = [g for g in full_dataset if g.y.item() == 0]
train_attack = [g for g in full_dataset if g.y.item() == 1]
# ... (val splits)

logger.info(f"📊 Separated dataset: {len(train_normal)} normal + {len(train_attack)} attack training")
logger.info(f"📊 Validation: {len(val_normal)} normal + {len(val_attack)} attack")
logger.info(f"📊 Total graphs: {len(train_normal) + len(train_attack) + len(val_normal) + len(val_attack)}")
```

**Lines 301-374**: `_optimize_batch_size_for_curriculum()` - Detailed optimization tracking
```python
logger.info("📊 _optimize_batch_size_for_curriculum() called")
logger.info(f"   Current datamodule.batch_size: {datamodule.batch_size}")
logger.info(f"   Current dataset size: {len(datamodule.train_dataset)}")

optimize_batch = getattr(self.config.training, 'optimize_batch_size', True)
logger.info(f"   optimize_batch_size setting: {optimize_batch}")

if not optimize_batch:
    logger.info("   Optimization disabled, using conservative batch size...")
    # ... conservative path

logger.info("🔧 Optimizing batch size using maximum curriculum dataset size...")
logger.info("   Creating max-size dataset for tuning...")
original_state = datamodule.create_max_size_dataset_for_tuning()
logger.info(f"   Max dataset size: {len(datamodule.train_dataset)}")

try:
    logger.info("   Creating BatchSizeOptimizer...")
    safety_factor = getattr(self.config.training, 'graph_memory_safety_factor', 0.5)
    max_trials = getattr(self.config.training, 'max_batch_size_trials', 10)
    mode = getattr(self.config.training, 'batch_size_mode', 'power')

    logger.info(f"   Optimizer config: safety_factor={safety_factor}, max_trials={max_trials}, mode={mode}")

    optimizer = BatchSizeOptimizer(...)

    logger.info(f"   Calling optimizer.optimize_with_datamodule()...")
    logger.info(f"   Pre-optimization: datamodule.batch_size={datamodule.batch_size}")

    safe_batch_size = optimizer.optimize_with_datamodule(model, datamodule)

    logger.info(f"   Returned safe_batch_size: {safe_batch_size}")
    logger.info(f"   Assigning to datamodule.batch_size...")
    datamodule.batch_size = safe_batch_size
    logger.info(f"   Post-assignment: datamodule.batch_size={datamodule.batch_size}")
    logger.info(f"✅ Batch size optimized for curriculum learning: {safe_batch_size}")

except Exception as e:
    logger.warning(f"❌ Batch size optimization failed: {e}. Using default batch size.")
    import traceback
    logger.warning(f"Exception traceback:\n{traceback.format_exc()}")
    # ... fallback

finally:
    logger.info("   Restoring dataset to original state...")
    if original_state:
        datamodule.restore_dataset_after_tuning(original_state)
        logger.info(f"   Restored dataset size: {len(datamodule.train_dataset)}")

logger.info(f"📊 Final state after optimization:")
logger.info(f"   datamodule.batch_size: {datamodule.batch_size}")
logger.info(f"   dataset size: {len(datamodule.train_dataset)}")
```

---

## How to Use This

### Step 1: Run the Test Suite

```bash
cd /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT
python tests/test_dataloader_batch_behavior.py
```

This will tell you if the issue is:
- ✅ **All tests pass**: PyG DataLoader works correctly in isolation. The issue is in Lightning's trainer or the training pipeline.
- ❌ **Test 1 fails**: PyG DataLoader itself has a bug (unlikely)
- ❌ **Test 2-3 fail**: Something about our dataset or datamodule is causing the issue

### Step 2: Run a Training Job with Enhanced Logging

Submit a curriculum training job as usual. The logs will now show:

**Expected good output** (if tuner runs successfully):
```
🎓 Starting GAT training with curriculum learning + hard mining
📊 Separating normal and attack graphs...
📊 Separated dataset: 6190 normal + 3174 attack training
📊 Validation: 1547 normal + 793 attack
📊 Total graphs: 11704
Loading VGAE from: /path/to/vgae.pth
📊 Curriculum batch size: 32 (base: 32, multiplier: 1.0)
📊 Datamodule created with batch_size=32
🧠 GAT model created
🔧 About to optimize batch size (current: 32)
📊 _optimize_batch_size_for_curriculum() called
   Current datamodule.batch_size: 32
   Current dataset size: 6348
   optimize_batch_size setting: True
🔧 Optimizing batch size using maximum curriculum dataset size...
   Creating max-size dataset for tuning...
   Max dataset size: 31740
   Creating BatchSizeOptimizer...
   Optimizer config: safety_factor=0.5, max_trials=10, mode=power
   Calling optimizer.optimize_with_datamodule()...
   Pre-optimization: datamodule.batch_size=32
📊 Tuner changed batch size: 32 → 1024
✅ Batch size tuner found: 1024
🛡️ Applied 50% safety factor
📊 Final safe batch size: 512
   Returned safe_batch_size: 512
   Assigning to datamodule.batch_size...
   Post-assignment: datamodule.batch_size=512
✅ Batch size optimized for curriculum learning: 512
   Restoring dataset to original state...
   Restored dataset size: 6348
📊 Final state after optimization:
   datamodule.batch_size: 512
   dataset size: 6348
✅ Batch size optimization complete (final: 512)
🔍 Creating train dataloader:
   Dataset length: 6348
   Batch size: 512
   Expected batches: 13
   Num workers: 8
   Actual batches (len(dataloader)): 13
```

**Problematic output** (if tuner fails):
```
🎓 Starting GAT training with curriculum learning + hard mining
📊 Datamodule created with batch_size=32
🔧 About to optimize batch size (current: 32)
📊 _optimize_batch_size_for_curriculum() called
   Current datamodule.batch_size: 32
   optimize_batch_size setting: True
❌ Batch size optimization failed: [some error]. Using default batch size.
Exception traceback:
[traceback here]
   Conservative fallback batch size: 16
✅ Batch size optimization complete (final: 16)
🔍 Creating train dataloader:
   Dataset length: 6348
   Batch size: 16
   Expected batches: 397
   Actual batches (len(dataloader)): 1  ← ❌ BUG REPRODUCED!
```

**Missing output** (if curriculum code doesn't run at all):
```
Training: curriculum
Found 6 CSV files to process
Total graphs created: 9364
[VGAE] Constructor called with: ...
Training: |          | 0/? [00:00<?, ?it/s]
Epoch 0: 100%|██████████| 1/1 [00:00<00:00,  1.22it/s]

← Missing ALL curriculum-specific messages!
← This suggests curriculum training didn't execute at all
```

---

## What to Look For

### Scenario A: Tuner Works, But DataLoader Still Creates 1 Batch

If you see:
```
   Returned safe_batch_size: 512
   Post-assignment: datamodule.batch_size=512
```

But then:
```
🔍 Creating train dataloader:
   Dataset length: 6348
   Batch size: 512
   Expected batches: 13
   Actual batches (len(dataloader)): 1  ← BUG!
```

**This means**: PyG DataLoader is ignoring batch_size parameter. Could be:
- Lightning overriding it
- PyG bug
- Something modifying the dataloader after creation

### Scenario B: Tuner Fails Silently

If you see:
```
❌ Batch size optimization failed: [error]
```

**Look at the exception traceback** to see why tuner failed.

Common causes:
- CUDA out of memory during tuning
- Model not compatible with Lightning Tuner
- Config missing required fields

### Scenario C: Curriculum Never Runs

If you see NONE of the curriculum messages:
```
Missing:
- 📊 Separated dataset
- Loading VGAE
- 🔧 About to optimize
- 🌊 Epoch X: Momentum Curriculum
```

**This means**: Curriculum training code path never executed. Check:
- Is `mode=curriculum` in frozen config?
- Did trainer dispatch to correct mode?
- Was there an early exception that caused fallback?

---

## Next Steps Based on Results

### If Test Suite Passes (DataLoader works in isolation)

The issue is in:
- Lightning trainer modifying dataloaders
- Something in the training pipeline
- Interaction between Lightning and PyG

**Action**: Add logging to Lightning's dataloader hooks or check if Lightning is overriding `train_dataloader()`.

### If Test Suite Fails (DataLoader broken)

The issue is in:
- PyG DataLoader itself
- Our dataset implementation
- Collate function

**Action**: File bug report with PyG or investigate dataset `__getitem__` / `__len__` methods.

### If Tuner Fails

Fix the tuner issue first. Common fixes:
- Increase GPU memory for tuning
- Reduce max_batch_size_trials
- Set batch_size_mode='binsearch' instead of 'power'

### If Curriculum Never Runs

Check dispatch logic in [src/training/trainer.py](src/training/trainer.py) to see why curriculum path wasn't taken.

---

## Files Modified

1. [tests/test_dataloader_batch_behavior.py](tests/test_dataloader_batch_behavior.py) - New test suite
2. [src/training/datamodules.py](src/training/datamodules.py) - Added dataloader creation logging
3. [src/training/modes/curriculum.py](src/training/modes/curriculum.py) - Added comprehensive optimization logging

## Original Bug Fix

**File**: [src/training/modes/curriculum.py:342](src/training/modes/curriculum.py:342)

Added missing line:
```python
safe_batch_size = optimizer.optimize_with_datamodule(model, datamodule)
datamodule.batch_size = safe_batch_size  # ← This line was missing
```

This ensures the tuner result is actually applied to the datamodule (though BatchSizeOptimizer already does this at line 109, so this is defensive/redundant).
