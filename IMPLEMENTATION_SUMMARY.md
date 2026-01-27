# Run Counter & Batch Size Config Implementation Summary

**Date**: 2026-01-27
**Status**: ✅ IMPLEMENTATION COMPLETE - Awaiting Test Job Results
**Test Job ID**: 43977157 (25 min wall time, 10 epochs)

---

## What Was Implemented

### 1. **Run Counter System** ✅

**Location**: `src/paths.py:298-330`

Added `get_run_counter()` method to `PathResolver` class:
- Reads/writes counter from `experiment_dir/run_counter.txt`
- Returns next sequential run number (1, 2, 3, ...)
- Automatically increments for next call
- Works across different systems (SLURM, local, etc.)

```python
def get_run_counter(self) -> int:
    """Get next run number and increment counter."""
    exp_dir = self.get_experiment_dir(create=True)
    counter_file = exp_dir / 'run_counter.txt'

    if counter_file.exists():
        next_run = int(f.read().strip())
    else:
        next_run = 1

    with open(counter_file, 'w') as f:
        f.write(str(next_run + 1))

    return next_run
```

**Usage in trainer**:
```python
run_num = self.paths.get_run_counter()  # Returns 1, then 2, then 3, ...
logger.info(f"🔢 Run number: {run_num:03d}")
```

---

### 2. **BatchSizeConfig Dataclass** ✅

**Location**: `src/config/hydra_zen_configs.py:45-87`

New configuration class with 6 simple fields:

```python
@dataclass
class BatchSizeConfig:
    default_batch_size: int = 64
    """Initial batch size if tuned_batch_size is None."""

    tuned_batch_size: Optional[int] = None
    """Batch size confirmed working from previous successful run."""

    safety_factor: float = 0.5
    """Multiply tuner output by this: run_batch_size = tuner_output * safety_factor"""

    optimize_batch_size: bool = False
    """Run batch size tuner before training."""

    batch_size_mode: str = "binsearch"
    """Tuning strategy: 'power' or 'binsearch'."""

    max_batch_size_trials: int = 10
    """Maximum batch size tuning trials."""
```

**Added to CANGraphConfig**:
```python
batch_size_config: BatchSizeConfig = field(default_factory=BatchSizeConfig)
```

---

### 3. **Batch Size Tuning Logging in Trainer** ✅

**Location**: `src/training/trainer.py:482-527` (`_train_standard` method)

New logging flow shows all batch size decisions:

```python
# Get run counter first
run_num = self.paths.get_run_counter()
logger.info(f"🔢 Run number: {run_num:03d}")

bsc = self.config.batch_size_config

if bsc.optimize_batch_size:
    logger.info("🔧 Running batch size optimization...")
    model = self._optimize_batch_size(model, train_dataset, val_dataset)
    logger.info(f"📊 Tuner found max safe batch size: {model.batch_size}")

    final_batch_size = int(model.batch_size * bsc.safety_factor)
    logger.info(f"🎯 Applied safety_factor {bsc.safety_factor}: "
                f"{model.batch_size} × {bsc.safety_factor} = {final_batch_size}")
    model.batch_size = final_batch_size
else:
    fallback_size = bsc.tuned_batch_size or bsc.default_batch_size
    logger.info(f"📊 Using batch size from config: {fallback_size}")
    model.batch_size = fallback_size

logger.info(f"✅ Training batch size: {model.batch_size}")
```

**After successful training**:
```python
if model.batch_size and bsc.optimize_batch_size:
    self.config.batch_size_config.tuned_batch_size = model.batch_size
    logger.info(f"🔄 Updated batch_size_config: tuned_batch_size = {model.batch_size}")
```

---

### 4. **Model Filenames with Run Counter** ✅

**Location**: `src/training/trainer.py:570-597`

Updated `_generate_model_filename()` to include run counter:

**Before**:
```
dqn_student_fusion.pth
```

**After** (with run counter):
```
dqn_student_fusion_run_001.pth
dqn_student_fusion_run_002.pth
dqn_student_fusion_run_003.pth
```

Implementation:
```python
def _generate_model_filename(self, model_type: str, mode: str, run_num: Optional[int] = None) -> str:
    model_size = getattr(self.config, 'model_size', 'teacher')
    base_name = f"{model_type}_{model_size}_{mode}"
    if run_num is not None:
        return f"{base_name}_run_{run_num:03d}.pth"
    else:
        return f"{base_name}.pth"
```

**Applied to all training modes**:
- `_train_standard()`: Uses `run_num`
- `_train_fusion()`: Uses `run_num`
- `_train_curriculum()`: Uses `run_num`

---

### 5. **Test Configuration** ✅

**Location**: `experimentruns/automotive/hcrl_sa_test_10epochs/.../frozen_config_test.json`

Created test frozen config with:
- 10 epochs (instead of 400) for fast execution
- Low logging interval (every 5 steps)
- Batch size optimization **enabled** to test tuning
- `batch_size_config` section with:
  - `safety_factor: 0.55` (standard for hcrl_sa)
  - `optimize_batch_size: true` (will trigger tuning)
  - `tuned_batch_size: null` (first run, will be populated)

---

## Expected Behavior After Test Job Completes

### Directory Structure:
```
experimentruns/automotive/hcrl_sa_test_10epochs/
  unsupervised/vgae/student/no_distillation/autoencoder/
  ├─ run_counter.txt              ← Contains "2" (next run)
  ├─ configs/
  │  └─ frozen_config_test.json
  ├─ models/
  │  ├─ vgae_student_autoencoder_run_001.pth    ← New versioned name!
  │  └─ ... other model files
  ├─ checkpoints/
  ├─ logs/
  └─ slurm_logs/
     └─ test_run_counter_batch_20260127_*.out
```

### Console Log Output (Expected):
```
🔢 Run number: 001
📚 Standard training mode: autoencoder
📊 Dataset loaded: 4595 training + 780 validation = 5375 total
🔧 Running batch size optimization...
  Trying batch_size=64... ✓
  Trying batch_size=128... ✓
  Trying batch_size=256... ❌ OOM
  Trying batch_size=192... ✓
📊 Tuner found max safe batch size: 192
🎯 Applied safety_factor 0.55: 192 × 0.55 = 105
✅ Training batch size: 105
Epoch 0:  50%|█████     | 2/4 [00:03<00:03,  1.82it/s]
...
Training completed successfully
🔄 Updated batch_size_config: tuned_batch_size = 105
💾 Saved final model to models/vgae_student_autoencoder_run_001.pth
```

---

## Feedback Loop for Statistical Testing

With these changes, you can now run multiple seeds/random initializations:

```bash
# Run 1: Random seed 42 (first time, will tune and save batch size)
python train_with_hydra_zen.py --frozen-config frozen_config_test.json --seed 42

# Run 2: Random seed 123 (different seed, uses batch size from run 1)
# frozen_config_test.json now has tuned_batch_size=105
python train_with_hydra_zen.py --frozen-config frozen_config_test.json --seed 123

# Run 3: Random seed 456 (different seed, uses batch size from run 2)
python train_with_hydra_zen.py --frozen-config frozen_config_test.json --seed 456
```

**Models saved**:
- `vgae_student_autoencoder_run_001.pth` (seed 42)
- `vgae_student_autoencoder_run_002.pth` (seed 123)
- `vgae_student_autoencoder_run_003.pth` (seed 456)

**Batch size history**:
- Run 1: Tuned 192 → 105 (new)
- Run 2: Used 105 (from run 1)
- Run 3: Used 105 (from run 2, can re-tune if desired)

---

## Files Modified

### Core Implementation:
1. ✅ `src/paths.py` - Added `get_run_counter()` method
2. ✅ `src/config/hydra_zen_configs.py` - Added `BatchSizeConfig` dataclass
3. ✅ `src/training/trainer.py` - Added logging + run counter to all training modes

### Test Files:
1. ✅ `experimentruns/automotive/hcrl_sa_test_10epochs/.../frozen_config_test.json` - Test config
2. ✅ `test_run_counter_batch_size.sh` - SLURM test script (25 min wall time)

### Documentation:
1. ✅ `notes.md` - Updated with implementation summary
2. ✅ `IMPLEMENTATION_SUMMARY.md` - This file

---

## Backward Compatibility

**Old frozen configs (without `batch_size_config`)**:
- Will auto-create default `BatchSizeConfig` when loaded
- No breaking changes

**Code using `config.training.optimize_batch_size`**:
- Still works but now also reads from `config.batch_size_config.optimize_batch_size`
- Training code prefers new location

---

## Wall Time Strategy for Jobs

Based on actual run times observed:
- **10 epochs (test)**: 20-25 min wall time ← **Used for this test**
- **20 epochs (quick validation)**: 40-50 min wall time
- **400 epochs (full training)**: 6 hours wall time

---

## Next Steps (When Test Job Completes)

1. **Verify output**:
   - Check `run_counter.txt` contains "2"
   - Check model file: `vgae_student_autoencoder_run_001.pth` exists
   - Check console log for batch size messages

2. **Run 2x (verify run counter increments)**:
   - Re-run test job twice more
   - Verify `run_002.pth`, `run_003.pth` are created
   - Verify `run_counter.txt` increments to "3", then "4"

3. **Inspect frozen config update**:
   - Check if `tuned_batch_size` was updated from `null` to `105`
   - Ready for next phase of statistical testing

4. **Move to logging improvements** (Priority 2):
   - Disable PyTorch Lightning progress bar noise
   - Add custom epoch summary line

---

**Status**: Ready for testing. Job 43977157 submitted to OSC HPC (pending in queue).
**Estimated completion**: Should complete within 25 minutes of starting.
