# Complete Testing Checklist - Run Counter & Batch Size Implementation

**Date**: 2026-01-27
**Implementation Status**: ✅ COMPLETE
**Test Job ID**: 43977157
**Wall Time**: 25 minutes
**Epochs**: 10 (for fast testing)

---

## Pre-Testing: What We Implemented

- ✅ Run counter system (`src/paths.py`)
- ✅ BatchSizeConfig dataclass (`src/config/hydra_zen_configs.py`)
- ✅ Batch size logging in trainer (`src/training/trainer.py`)
- ✅ Versioned model filenames with run counter
- ✅ GPU monitoring integration into SLURM scripts
- ✅ GPU analysis script (`analyze_gpu_monitor.py`)
- ✅ Test frozen config with 10 epochs

---

## After Job 43977157 Completes: Verification Steps

### Step 1: Check Job Completion (5 min)

```bash
sacct -j 43977157 --format=JobID,JobName,State,ExitCode
```

**Expected Output**:
```
JobID           JobName      State ExitCode
43977157    test_run_+   COMPLETED    0:0
```

**If FAILED**: Check error file for OOM or other issues
```bash
tail -50 slurm-43977157.err
```

---

### Step 2: Verify Run Counter (1 min)

**File to check**:
```
experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder/run_counter.txt
```

**Expected content**: `2`

```bash
cat experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder/run_counter.txt
```

✅ Should output: `2` (because run 1 just completed, next run will be 2)

---

### Step 3: Verify Model Files (1 min)

**Directory**:
```
experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder/models/
```

**Expected files**:
```
ls -lh experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder/models/
```

✅ Should contain:
- `vgae_student_autoencoder_run_001.pth` (NEW: includes run counter!)
- Other model files from tuning/checkpointing

---

### Step 4: Analyze Batch Size Logging (5 min)

**Check console output for batch size messages**:

```bash
grep -E "🔢|🔧|📊|🎯|✅.*batch|🔄" slurm-43977157.out | head -30
```

**Expected messages**:
```
🔢 Run number: 001
🔧 Running batch size optimization...
📊 Tuner found max safe batch size: 192
🎯 Applied safety_factor 0.55: 192 × 0.55 = 105
✅ Training batch size: 105
🔄 Updated batch_size_config: tuned_batch_size = 105
```

✅ All 6 messages should appear in output

---

### Step 5: Analyze GPU Monitoring (10 min)

**Run the analysis script**:

```bash
python analyze_gpu_monitor.py gpu_monitor_43977157.csv
```

**Expected console output**:
```
======================================================================
GPU Monitoring Analysis: gpu_monitor_43977157.csv
======================================================================

📊 MEMORY STATISTICS:
  Peak Memory Used:    10,650 MiB (10.40 GB)
  Average Memory Used: 8,200 MiB (8.01 GB)
  GPU Total:           16,384 MiB (16.00 GB)
  Peak Utilization:    65.0%

⚡ COMPUTE STATISTICS:
  Peak GPU Util:       87.5%
  Average GPU Util:    72.3%
  Peak Memory Util:    65.0%
  Average Memory Util: 50.1%

🔍 MEMORY LEAK DETECTION:
  ✅ No memory leak detected (growth rate: 1.2 MiB/step)

📈 BOTTLENECK ANALYSIS:
  ⚙️  COMPUTE BOUND (balanced utilization)
     Recommendation: Good tuning! Can potentially increase batch_size

⏱️  TRAINING DURATION: 15.2 minutes
```

**Check for**:
- ✅ Peak memory < 12 GB (good safety margin on 15.77 GB GPU)
- ✅ No memory leak detected (growth rate < 2 MiB/step)
- ✅ GPU Util 70%+ (not starved for data)
- ✅ Memory Util 60-75% (balanced)

**Generated plot**: `gpu_monitor_43977157_analysis.png`

---

### Step 6: Verify Frozen Config Updated (2 min)

**Check if batch_size_config was updated**:

```bash
grep -A 8 "batch_size_config" experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder/configs/frozen_config_test.json
```

**Expected**: `tuned_batch_size` should still be `null` (only gets saved if we update frozen config, which our implementation logs but may not persist in this test)

Check the console log for:
```
🔄 Updated batch_size_config: tuned_batch_size = 105
```

This confirms the logic works; frozen config persistence is next phase.

---

## Test Success Criteria

### ✅ All MUST Pass:

- [ ] **Run counter**: run_counter.txt contains "2"
- [ ] **Model filename**: vgae_student_autoencoder_run_001.pth exists
- [ ] **Batch size logs**: All 6 messages appear in output
- [ ] **No OOM errors**: Exit code 0, no CUDA OOM in logs
- [ ] **No memory leak**: GPU analysis shows growth rate < 2 MiB/step

### ✅ Nice to Have:

- [ ] Peak memory 60-75% of GPU capacity
- [ ] GPU utilization 70%+
- [ ] Training completes in 15-20 minutes
- [ ] No warnings in logs

---

## If Test Fails: Troubleshooting

### Issue: Run Counter File Not Created

**Cause**: PathResolver.get_experiment_dir() failed or directory perms

**Fix**:
```bash
mkdir -p experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder
```

Then resubmit job.

---

### Issue: Model Filename Doesn't Have `_run_001`

**Cause**: Code change didn't apply or model save path issue

**Check**:
```bash
grep -n "_generate_model_filename" src/training/trainer.py | head -5
```

Verify run_num parameter is being passed.

---

### Issue: No Batch Size Log Messages

**Cause**: Logging configuration issue (pre-import problem from before)

**Check**:
```bash
grep -E "optimize_batch_size|safety_factor" slurm-43977157.out
```

If not present, logging config needs fixing (see LOGGING_AND_MODEL_ANALYSIS.md).

---

### Issue: GPU Peak Memory > 12 GB

**Cause**: safety_factor too high or batch_size optimization failed

**Actions**:
1. Check tuner output in logs
2. If tuner output is 192, then 192 × 0.55 = 105.6 → 105 (correct)
3. If actual memory is 12 GB, safety_factor is working but memory usage is high
4. Next: Reduce safety_factor to 0.45, test again

---

### Issue: Memory Leak Detected (growth > 50 MiB/step)

**Cause**: Validation loop or checkpoint saving allocating memory

**Fix**:
1. Add `@torch.no_grad()` to validation loop
2. Call `torch.cuda.empty_cache()` between epochs
3. Check checkpoint saving during training (should be disabled in test)

---

## Next Phase (After Successful Test)

### Phase 1: Multi-Run Testing (Verify Run Counter Increment)

1. **Run 2** (same config, should use run counter):
   ```bash
   sbatch test_run_counter_batch_size.sh
   ```
   Expected: `vgae_student_autoencoder_run_002.pth` created
   Expected: run_counter.txt contains "3"

2. **Run 3** (same config, different random seed if possible):
   ```bash
   sbatch test_run_counter_batch_size.sh --seed 123
   ```
   Expected: `vgae_student_autoencoder_run_003.pth` created
   Expected: run_counter.txt contains "4"

### Phase 2: Batch Size Consistency

1. Verify all 3 runs use same batch size (105)
2. Check GPU memory profiles are consistent
3. Confirm tuned_batch_size stays at 105 across runs

### Phase 3: Logging Improvements (Priority 2)

1. Disable PyTorch Lightning progress bar
2. Add epoch summary lines
3. Reduce log file sizes

### Phase 4: Production Runs

1. Create full frozen configs for all combinations
2. Run 3-5 iterations per config for statistical significance
3. Use GPU monitoring to validate batch sizes

---

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `src/paths.py` | `get_run_counter()` implementation |
| `src/config/hydra_zen_configs.py` | `BatchSizeConfig` dataclass |
| `src/training/trainer.py` | Batch size logging + run counter usage |
| `test_run_counter_batch_size.sh` | SLURM test script with GPU monitoring |
| `analyze_gpu_monitor.py` | GPU CSV analysis tool |
| `GPU_MONITORING_GUIDE.md` | Detailed GPU monitoring documentation |
| `IMPLEMENTATION_SUMMARY.md` | Full implementation details |
| `notes.md` | Project notes and status updates |

---

## Expected Timeline

- **Job submit**: Now
- **Job wait in queue**: 5-30 min (depends on GPU availability)
- **Job execution**: 15-20 min (10 epochs)
- **Analysis**: 5-10 min (parse logs + GPU plots)
- **Total**: 30-60 min from submission to completion

---

## Success Message (When All Checks Pass)

```
✅ RUN COUNTER: Working (run_counter.txt = 2)
✅ MODEL FILENAME: Working (vgae_student_autoencoder_run_001.pth)
✅ BATCH SIZE LOGGING: Working (all 6 messages present)
✅ GPU MONITORING: Working (no memory leak, 72% avg util)
✅ TRAINING: Completed successfully

🎉 BATCH SIZE & RUN COUNTER IMPLEMENTATION VERIFIED!

Next steps: Statistical testing with multiple runs
```

---

**Status**: Awaiting job 43977157 completion
**Estimated**: Complete within 1 hour of submission
