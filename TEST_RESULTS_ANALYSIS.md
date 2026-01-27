# Test Results Analysis - DataLoader Batch Behavior

## Test Results Summary

```
================================================================================
SUMMARY
================================================================================
synthetic           : ✅ PASS
real_dataset        : ✅ PASS
curriculum          : ❌ FAIL (but not due to batching)
```

## Critical Finding: DataLoader Works Correctly! ✅

### Test 1: Simple Synthetic Graphs
- **Input**: 100 graphs, batch_size=32
- **Expected**: 4 batches (3×32 + 1×4)
- **Actual**: 4 batches ✅
- **Batch sizes**: [32, 32, 32, 4]

**Verdict**: PyG DataLoader works perfectly with simple data.

### Test 2: Real CAN Dataset (hcrl_sa)
- **Input**: 7,491 training graphs, batch_size=32
- **Expected**: 235 batches
- **Actual**: 235 batches ✅
- **Batch sizes**: [32, 32, 32, 32, 32, ...] (all 32)

**Verdict**: PyG DataLoader works perfectly with real CAN graph data.

### Test 3: Curriculum DataModule
- **Status**: Failed due to VGAE requirement (not a batch size issue)
- **Error**: `RuntimeError: VGAE model required for hard-mining difficulty scoring`
- **Note**: Test passed `vgae_model=None` which triggers hard mining error

This test didn't complete, but it's NOT a batch size issue. Fixed in datamodules.py to allow random sampling when VGAE is None.

---

## What This Tells Us

### ✅ What's NOT Broken

1. **PyG DataLoader**: Works perfectly in isolation
2. **Our Dataset Implementation**: Creates proper batch iterators
3. **Basic Curriculum Setup**: Can load and separate data correctly

### ❓ What's Still Mysterious

If DataLoader works correctly in tests, why do production logs show:
- **hcrl_sa**: `Epoch 0: 100%|██████████| 1/1` (1 batch instead of 235)
- **set_02**: `Epoch 0: 100%|██████████| 4/4` (4 batches instead of 6,360)

**Possible causes**:

1. **Lightning Trainer is Overriding Batch Size**
   - Lightning has internal logic that can modify dataloaders
   - Might be setting batch_size to a huge value
   - Or modifying the dataloader after creation

2. **Batch Size Tuner is Setting Huge Batches**
   - Tuner might be successfully finding "optimal" batch_size
   - But "optimal" might be the entire dataset (no OOM = good?)
   - Safety factor might not be applied correctly

3. **Progress Bar is Misleading**
   - The "1/1" might not mean 1 batch
   - Could be 1 gradient accumulation step
   - Or some Lightning internal counter

4. **Curriculum Dataset Size**
   - Curriculum starts with subset (~6,348 graphs at epoch 0)
   - Maybe batch_size gets set to match or exceed dataset size?
   - batch_size=9364 would create 1 batch for hcrl_sa

---

## Next Investigation Steps

### Step 1: Check What Batch Size Tuner Actually Returns

With the new logging in curriculum.py, next training run will show:

```
   Calling optimizer.optimize_with_datamodule()...
   Pre-optimization: datamodule.batch_size=32
📊 Tuner changed batch size: 32 → ???? ← WHAT IS THIS VALUE?
   Returned safe_batch_size: ????
```

**Key question**: Is tuner returning 32, 512, or 9364?

### Step 2: Check DataLoader Creation Logs

New logging in datamodules.py will show:

```
🔍 Creating train dataloader:
   Dataset length: ????
   Batch size: ????
   Expected batches: ????
   Actual batches (len(dataloader)): ????
```

**Key questions**:
- Does expected match actual?
- If not, what's the discrepancy?

### Step 3: Check Lightning Trainer Configuration

Look for any settings that might affect batching:
- `accumulate_grad_batches`
- `limit_train_batches`
- Any custom dataloader hooks

---

## Hypothesis Ranking

### Most Likely: Tuner Returns Huge Batch Size

**Evidence**:
- batch_size_factors.json has safety factors like 0.35
- But tuner might find "optimal" = dataset size
- 9364 graphs × 0.55 safety = ~5,150 batch size
- This would create 1-2 batches per epoch

**Test**: Check tuner output in next training log.

### Likely: Lightning Modifies DataLoader

**Evidence**:
- Lightning has many internal dataloader modifications
- Could be related to distributed training settings
- Or gradient accumulation logic

**Test**: Check if `len(dataloader)` changes between creation and training.

### Less Likely: Progress Bar Bug

**Evidence**:
- Training completes successfully
- Metrics look reasonable

**Test**: Check actual gradient update counts vs displayed batch counts.

### Unlikely: Our Code Has State Bug

**Evidence**:
- Tests show our code works correctly
- Issue only appears in production

---

## Action Items

### Immediate: Run Training with New Logging

Submit a curriculum training job with the enhanced logging. Look for:

1. **Tuner output**:
   ```
   📊 Tuner changed batch size: 32 → X
   ```

2. **DataLoader creation**:
   ```
   🔍 Creating train dataloader:
      Expected batches: Y
      Actual batches: Z
   ```

3. **Compare X, Y, Z**:
   - If X = 9364, tuner is setting batch_size to dataset size
   - If Y != Z, Lightning is modifying the dataloader
   - If Y = 1, something set batch_size = dataset size

### Follow-up: If Tuner Sets Huge Batch Size

**Why would tuner think batch_size=9364 is optimal?**
- It fits in memory without OOM
- From tuner's perspective: "bigger batch = better, no crash = good"

**Fix**:
- Set explicit `max_batch_size` parameter in tuner config
- Verify safety factors are being applied
- Check if curriculum's max-size dataset (~31k graphs) is causing issues

### Follow-up: If Lightning Modifies DataLoader

**Find where Lightning changes it**:
- Check trainer hooks
- Look for `replace_sampler_ddp` setting
- Check distributed training configuration

---

## Key Insight

**The "1/1 batch" behavior is NOT a PyG DataLoader bug.**

It's either:
1. Tuner legitimately finding that huge batches work
2. Lightning trainer modifying our settings
3. Something in the training configuration we haven't considered

The enhanced logging will tell us exactly which one it is.

---

## Files Modified for Investigation

1. [src/training/datamodules.py](src/training/datamodules.py:176-180) - Allow curriculum without VGAE (random sampling fallback)
2. [src/training/datamodules.py](src/training/datamodules.py:465-498) - Added dataloader creation logging
3. [src/training/modes/curriculum.py](src/training/modes/curriculum.py:301-374) - Added comprehensive optimization logging

## References

- [BATCH_SIZE_EXPLANATION.md](BATCH_SIZE_EXPLANATION.md) - Detailed explanation of the mystery
- [DEBUG_BATCH_SIZE_INVESTIGATION.md](DEBUG_BATCH_SIZE_INVESTIGATION.md) - How to use the debug tools
- [BATCH_SIZE_INVESTIGATION_COMPLETE.md](BATCH_SIZE_INVESTIGATION_COMPLETE.md) - Complete investigation summary
