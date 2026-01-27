# Batch Size Tuner Investigation

## User's Question
"I thought we used tuner and batch_size factors which makes much larger batches. Are we still using those small batch sizes?"

## The Discovery

### Evidence That Tuner Is NOT Running

1. **Progress bars show 1/1 batches per epoch**
   - Example: hcrl_sa curriculum training with 9364 graphs
   - Progress: `Epoch 0: 100%|██████████| 1/1`
   - Expected: ~292 batches (9364 / 32) if batch_size=32
   - Actual: 1 batch containing ALL graphs

2. **No tuner output in logs**
   - Searched all SLURM logs for tuner-related messages
   - Expected: "🔧 Optimizing batch size...", "Tuner", "scale_batch_size"
   - Found: NONE

3. **Frozen config shows batch_size=32 (static)**
   - `config/frozen_config_*.json` all show `"batch_size": 32`
   - No evidence of dynamic batch size calculation

### The Root Cause

**CRITICAL BUG in [src/training/modes/curriculum.py](src/training/modes/curriculum.py:290-356)**

```python
def _optimize_batch_size_for_curriculum(self, model, datamodule):
    """Optimize batch size for curriculum learning."""

    optimize_batch = getattr(self.config.training, 'optimize_batch_size', True)

    if not optimize_batch:
        # ... conservative batch size path ...
        return model

    logger.info("🔧 Optimizing batch size using maximum curriculum dataset size...")

    # ... create BatchSizeOptimizer ...

    safe_batch_size = optimizer.optimize_with_datamodule(model, datamodule)
    logger.info(f"✅ Batch size optimized for curriculum learning: {safe_batch_size}")

    # ❌ BUG: Never assigns safe_batch_size to datamodule.batch_size!
    # The optimized batch size is computed but NOT APPLIED

    finally:
        # Restore dataset to original state
        if original_state:
            datamodule.restore_dataset_after_tuning(original_state)

    return model  # datamodule.batch_size unchanged!
```

**The bug**: Line 337 computes `safe_batch_size` but never updates `datamodule.batch_size`.

Compare to the exception handler (lines 346-349):
```python
except Exception as e:
    # ... warning ...
    safe_batch_size = datamodule.get_conservative_batch_size(...)
    datamodule.batch_size = safe_batch_size  # ✅ Correctly assigned
```

The exception path assigns it, but the success path doesn't!

### Why "1/1" Batches?

The datamodule is created with:
```python
# curriculum.py:125-134
datamodule = EnhancedCANGraphDataModule(
    train_normal=train_normal,
    train_attack=train_attack,
    val_normal=val_normal,
    val_attack=val_attack,
    vgae_model=vgae_model,
    batch_size=curriculum_batch_size,  # 32 from config
    num_workers=min(8, os.cpu_count() or 1),
    total_epochs=self.config.training.max_epochs
)
```

The `AdaptiveGraphDataset` used by curriculum learning starts with a small subset (~3174 graphs at epoch 0 with 1:1 ratio for 3174 attack graphs).

**Hypothesis**: The PyG DataLoader may be defaulting to loading the entire dataset when batch_size=32 is too small relative to the collate function overhead or some other PyG behavior. Need to investigate why batch_size=32 results in 1/1 batches for a ~3174 graph dataset.

### Batch Size Optimizer Infrastructure

The infrastructure EXISTS and is correctly configured:

1. **config/batch_size_factors.json** - Contains safety factors (0.35-0.6 range)
2. **[src/training/batch_optimizer.py](src/training/batch_optimizer.py)** - Implements Lightning Tuner wrapper
3. **Config settings**:
   ```json
   "optimize_batch_size": true,
   "batch_size_mode": "power",
   "max_batch_size_trials": 15,
   "use_adaptive_batch_size_factor": true
   ```

But it's not being applied due to the bug in curriculum.py line 337.

## Impact Assessment

### Memory Implications

If the entire dataset is loaded in a single batch:
- **hcrl_sa**: 9364 graphs loaded at once
- **set_02**: 203,496 graphs loaded at once (!)
- **set_03**: 166,098 graphs loaded at once

This explains:
1. Why curriculum learning uses so much memory (~13.8 GB)
2. Why set_03 OOMs (53.67 avg nodes × 166k graphs in single batch = massive memory spike)
3. Why the curriculum_memory_multiplier fix may not work as expected

### Performance Implications

Loading entire dataset in single batch:
- ✅ Fewer dataloader iterations (faster epoch time)
- ❌ Massive memory usage
- ❌ No mini-batch gradient updates (poor training dynamics)
- ❌ Single gradient update per epoch (curriculum can't respond to within-epoch changes)

## Fixes Needed

### Fix 1: Apply Optimized Batch Size (CRITICAL)

**File**: [src/training/modes/curriculum.py:337-338](src/training/modes/curriculum.py:337-338)

**Change**:
```python
safe_batch_size = optimizer.optimize_with_datamodule(model, datamodule)
datamodule.batch_size = safe_batch_size  # ADD THIS LINE
logger.info(f"✅ Batch size optimized for curriculum learning: {safe_batch_size}")
```

### Fix 2: Investigate PyG DataLoader Behavior

Need to understand why `batch_size=32` results in `1/1` batches. Possibilities:
1. AdaptiveGraphDataset `__len__()` returns wrong value
2. PyG DataLoader collate function issue
3. Some PyG internal batching behavior

**Investigation**:
```python
# Check dataset length vs batch count
print(f"Dataset length: {len(datamodule.train_dataset)}")
print(f"Batch size: {datamodule.batch_size}")
print(f"Expected batches: {len(datamodule.train_dataset) // datamodule.batch_size}")

# Check actual dataloader behavior
loader = datamodule.train_dataloader()
print(f"Actual batches: {len(loader)}")
```

### Fix 3: Verify Tuner Actually Runs

Add more verbose logging:
```python
logger.info(f"📊 Pre-optimization: batch_size={datamodule.batch_size}, dataset_size={len(datamodule.train_dataset)}")
safe_batch_size = optimizer.optimize_with_datamodule(model, datamodule)
logger.info(f"📊 Post-optimization: safe_batch_size={safe_batch_size}")
datamodule.batch_size = safe_batch_size
logger.info(f"📊 Applied: datamodule.batch_size={datamodule.batch_size}")
```

## Questions for Further Investigation

1. **Why does batch_size=32 create 1/1 batches?**
   - Is AdaptiveGraphDataset.__len__() correct?
   - Does PyG DataLoader have special behavior for graph datasets?

2. **Has tuner EVER run successfully?**
   - Check older logs
   - Verify tuner works in isolation

3. **What is the actual effective batch size?**
   - If 9364 graphs in "1 batch", what's the memory footprint?
   - Does PyG batch them internally?

4. **Why didn't we notice this earlier?**
   - Fast epoch times masked the issue?
   - Training still converged despite single batch per epoch?

## Next Steps

1. **Immediate**: Apply Fix 1 (add missing line to curriculum.py)
2. **Investigation**: Create test script to understand PyG DataLoader behavior
3. **Validation**: Run curriculum training with fix and verify tuner output appears
4. **Testing**: Confirm batch counts match expectations (dataset_size / batch_size)

## Files Referenced

- [src/training/modes/curriculum.py](src/training/modes/curriculum.py) (lines 290-356) - Bug location
- [src/training/batch_optimizer.py](src/training/batch_optimizer.py) - Tuner implementation
- [config/batch_size_factors.json](config/batch_size_factors.json) - Safety factors
- [src/training/datamodules.py](src/training/datamodules.py) - DataModule and AdaptiveGraphDataset

## User Confirmation

User was correct to question this. The tuner infrastructure exists but is not being applied due to the bug discovered above. The "small batch sizes" in the frozen config (batch_size=32) are NOT the actual effective batch sizes - the effective batch size appears to be the entire dataset per batch.
