# Why batch_size=32 Creates 1 Batch (When It Shouldn't)

## Your Question

> "I thought batch_size is the number of graph samples in a batch, but from you are saying it is actually more like dividing the number of training graphs (200k for example) by 32."

## The Direct Answer

**You are 100% correct.** `batch_size` IS the number of graphs per batch. With `batch_size=32`, you should get:

```
hcrl_sa: 9,364 graphs ÷ 32 = 293 batches per epoch
set_01:  151,089 graphs ÷ 32 = 4,722 batches per epoch
set_02:  203,496 graphs ÷ 32 = 6,360 batches per epoch
```

But the logs show:
- hcrl_sa: **1 batch per epoch** (entire 9,364-graph dataset!)
- set_01: **1 batch per epoch** (entire 151,089-graph dataset!)
- set_02: **4 batches per epoch** (~50,874 graphs each!)

**This is WRONG and should NOT be happening.** This is a bug, not intended behavior.

---

## What's Supposed to Happen

### Normal PyTorch/PyG DataLoader Behavior

```python
from torch_geometric.loader import DataLoader

dataset = AdaptiveGraphDataset(...)  # len=9364
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

print(f"Dataset size: {len(dataset)}")  # 9364
print(f"Batch size: 32")
print(f"Number of batches: {len(dataloader)}")  # Should be 293
```

**Expected**:
- Batch 1: graphs [0-31] (32 graphs)
- Batch 2: graphs [32-63] (32 graphs)
- ...
- Batch 293: graphs [9344-9363] (20 graphs, partial)

**Actual (in your logs)**:
- Batch 1: graphs [0-9363] (9,364 graphs - ENTIRE DATASET!)

---

## Investigation: Why Is This Happening?

### Hypothesis 1: Logging is Missing (Most Likely)

Looking at the SLURM logs, there are **NO curriculum-specific log messages**:

**Expected messages (NOT found)**:
```
🎓 Starting GAT training with curriculum learning + hard mining
📊 Separated dataset: 6190 normal + 3174 attack training
Loading VGAE from: /path/to/vgae_model.pth
🔧 Optimizing batch size using maximum curriculum dataset size...
✅ Batch size optimized for curriculum learning: 512
🌊 Epoch 0: Momentum Curriculum - Ratio 1.000:1 (3174 normals, 3174 attacks)
```

**Actual messages (only these)**:
```
Training: curriculum
Found 6 CSV files to process
Total graphs created: 9364
[VGAE] Constructor called with: ...
Training: |          | 0/? [00:00<?, ?it/s]
Epoch 0: 100%|██████████| 1/1 [00:00<00:00,  1.22it/s]
```

**This suggests**:
1. The curriculum logging was disabled/missing in the version that ran
2. OR the curriculum code path never executed
3. OR something caught an exception and fell back to simple training

The frozen config shows `"mode": "curriculum"` but the behavior looks like standard training.

### Hypothesis 2: The Dataset __len__ Is Wrong

Let me check if AdaptiveGraphDataset returns the wrong length:

```python
# From datamodules.py:331
def __len__(self):
    return len(self.epoch_samples)  # Should be correct
```

This looks correct. `epoch_samples` is generated with:
```python
self.epoch_samples = self.attack_graphs + selected_normals  # List concatenation
np.random.shuffle(self.epoch_samples)
```

For hcrl_sa at epoch 0 with 1:1 ratio:
- 3,174 attack graphs + 3,174 normal graphs = 6,348 graphs
- NOT 9,364 graphs (which is the full dataset size)

**Wait!** The logs show the FULL dataset (9,364 graphs) was loaded, but curriculum should start with a SUBSET (~6,348 graphs).

### Hypothesis 3: Curriculum Code Didn't Run

The evidence suggests curriculum training **may not have run at all**. Instead, it may have fallen back to standard training with the full dataset loaded in memory.

If standard training runs with batch_size=32 on 9,364 graphs:
- PyTorch DataLoader should create 293 batches
- But we see 1 batch

**This could happen if**:
1. `datamodule.train_dataloader()` returns something unexpected
2. PyG DataLoader has a bug with certain dataset sizes
3. Something overrides the dataloader behavior

### Hypothesis 4: PyG DataLoader vs PyTorch DataLoader

The code uses `from torch_geometric.loader import DataLoader` which is PyG's custom DataLoader for graph batching. PyG DataLoader does:

```python
def __iter__(self):
    for batch in super().__iter__():
        # batch is a list of graphs
        return Batch.from_data_list(batch)  # Combines graphs into single batch object
```

PyG's `Batch.from_data_list()` creates a single large graph with disconnected components. This is normal - each mini-batch of 32 graphs becomes one `Batch` object.

**BUT** the number of batches should still be dataset_size / batch_size = 293, not 1.

### Hypothesis 5: Silent Failure in Batch Optimizer

Looking at [BatchSizeOptimizer.optimize_with_datamodule()](src/training/batch_optimizer.py:80-122):

```python
try:
    tuner.scale_batch_size(model, datamodule=datamodule, ...)
    tuner_batch_size = getattr(model, 'batch_size', initial_bs)
    safe_batch_size = int(tuner_batch_size * self.graph_memory_safety_factor)

    # These lines UPDATE both model and datamodule
    model.batch_size = safe_batch_size
    datamodule.batch_size = safe_batch_size  # ← THIS LINE EXISTS!

    return safe_batch_size

except Exception as e:
    logger.warning(f"⚠️  Batch size optimization failed: {e}")
    return datamodule.batch_size
```

The BatchSizeOptimizer DOES set `datamodule.batch_size` at line 109. So the fix I added to curriculum.py (line 338) is actually redundant!

**BUT** if the tuner fails (exception), it returns the original batch_size without updating.

---

## The Real Problem: Tuner May Not Be Running

Looking at the complete absence of tuner output in logs, I suspect:

1. **Tuner is failing silently** - Exception caught but warning not logged
2. **OR Tuner is disabled** - Some condition prevents it from running
3. **OR Logging is completely broken** - All messages suppressed

### Missing Tuner Output

If the tuner ran successfully, we'd see:
```
📊 Tuner changed batch size: 32 → 1024
✅ Batch size tuner found: 1024
🛡️ Applied 50% safety factor
📊 Final safe batch size: 512
```

We see NONE of this in any log file.

---

## Why DataLoader Creates 1 Batch With batch_size=32

**This is the mystery.** PyG DataLoader should NOT do this. Possible causes:

### 1. DataLoader gets batch_size=None or batch_size=inf

If `datamodule.batch_size` is accidentally set to `None` or a huge number:
```python
DataLoader(dataset, batch_size=None)  # Loads entire dataset
DataLoader(dataset, batch_size=999999)  # Loads entire dataset
```

### 2. Lightning Overrides DataLoader Creation

PyTorch Lightning has internal logic that can modify dataloaders. Maybe Lightning is overriding the batch_size?

### 3. PyG DataLoader Bug

There might be a bug in PyG's DataLoader where certain conditions cause it to ignore batch_size.

### 4. The Dataset is Behaving Like IterableDataset

If the dataset is treated as an `IterableDataset` instead of a `Dataset`, PyTorch doesn't know its length and can't calculate batches properly.

---

## What To Investigate Next

### 1. Add Debug Logging

Modify [datamodules.py:465-472](src/training/datamodules.py:465-472):

```python
def train_dataloader(self):
    logger.info(f"🔍 Creating train dataloader:")
    logger.info(f"   Dataset length: {len(self.train_dataset)}")
    logger.info(f"   Batch size: {self.batch_size}")
    logger.info(f"   Expected batches: {len(self.train_dataset) // self.batch_size + 1}")

    dataloader = DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.num_workers,
        collate_fn=self._collate_graphs
    )

    logger.info(f"   Actual batches: {len(dataloader)}")
    return dataloader
```

### 2. Check If Tuner Is Being Called

Add logging to [curriculum.py:311](src/training/modes/curriculum.py:311):

```python
logger.info("🔧 Optimizing batch size using maximum curriculum dataset size...")
logger.info(f"   Current datamodule.batch_size: {datamodule.batch_size}")
logger.info(f"   Dataset size: {len(datamodule.train_dataset)}")

# Create optimizer
optimizer = BatchSizeOptimizer(...)

logger.info(f"   Calling optimizer.optimize_with_datamodule...")
safe_batch_size = optimizer.optimize_with_datamodule(model, datamodule)
logger.info(f"   Returned safe_batch_size: {safe_batch_size}")
logger.info(f"   Updated datamodule.batch_size: {datamodule.batch_size}")
```

### 3. Test DataLoader Directly

Create a test script:

```python
# test_dataloader.py
import torch
from torch_geometric.loader import DataLoader
from src.training.datamodules import load_dataset
from src.config.hydra_zen_configs import CANGraphConfig, GATConfig, CANDatasetConfig, CurriculumTrainingConfig

# Load minimal config
config = CANGraphConfig(
    model=GATConfig(...),
    dataset=CANDatasetConfig(name='hcrl_sa', ...),
    training=CurriculumTrainingConfig(...)
)

# Load dataset
train_dataset, val_dataset, num_ids = load_dataset('hcrl_sa', config)

print(f"Dataset length: {len(train_dataset)}")

# Create dataloader with batch_size=32
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

print(f"Batch size: 32")
print(f"Expected batches: {len(train_dataset) // 32 + 1}")
print(f"Actual batches from len(dataloader): {len(dataloader)}")

# Iterate and count
actual_batches = 0
for batch in dataloader:
    actual_batches += 1
    if actual_batches == 1:
        print(f"First batch size: {batch.num_graphs}")
    if actual_batches <= 3:
        print(f"Batch {actual_batches}: {batch.num_graphs} graphs")

print(f"Total batches iterated: {actual_batches}")
```

This will definitively tell us if the issue is with:
- The DataLoader itself
- Lightning's trainer
- Something else in the pipeline

---

## Current Status

After fixing [curriculum.py:338](src/training/modes/curriculum.py:338) (adding `datamodule.batch_size = safe_batch_size`):

**If the tuner is working**:
- ✅ Batch size should be set correctly
- ✅ Logging will show the optimized value
- ✅ Training will use proper mini-batches

**If the tuner is NOT working** (more likely):
- ❌ batch_size stays at 32
- ❌ DataLoader mysteriously creates 1 batch
- ❌ Need to investigate WHY this happens

The fix I applied ensures the tuner result is used, but doesn't explain why batch_size=32 creates 1 batch instead of 293 batches.

---

## Summary

**Your Understanding**: ✅ **100% CORRECT**
`batch_size=32` means 32 graphs per batch, which should give 293 batches for 9,364 graphs.

**What's Actually Happening**: ❌ **WRONG BEHAVIOR**
`batch_size=32` is creating 1 batch with all 9,364 graphs, which is NOT how DataLoader should work.

**Likely Cause**:
1. Batch size tuner is failing silently (no logs = didn't run or exception caught)
2. Curriculum training may not have executed (missing all log messages)
3. PyG DataLoader or Lightning trainer has unexpected behavior with batch_size=32
4. Something is overriding batch_size to None or a huge number

**Next Steps**:
1. Add extensive debug logging to curriculum.py and datamodules.py
2. Test DataLoader behavior in isolation
3. Check if Lightning is overriding our settings
4. Verify the tuner actually runs and doesn't fail silently

The one-line fix ensures the tuner result is used IF the tuner runs successfully. But we still need to understand why batch_size=32 behaves so strangely.
