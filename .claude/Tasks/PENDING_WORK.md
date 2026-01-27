# Pending Work & Task Specifications

## HIGH PRIORITY: Pending Implementation

### 1. Per-Mode KD Configuration (Pipeline CLI Enhancement)

**Status**: PENDING - Needs design decision

**Current Behavior**:
```bash
./can-train pipeline --distillation with-kd
# Applies with-kd to ALL jobs (rejects if fusion present)
```

**Desired Behavior**:
```bash
./can-train pipeline \
  --training-strategyl vgae,gat,dqn \
  --training-strategy autoencoder,curriculum,fusion \
  --distillation with-kd,with-kd,no-kd
# Job 1: vgae + autoencoder + with-kd
# Job 2: gat + curriculum + with-kd
# Job 3: dqn + fusion + no-kd ← KD disabled for fusion
```

**Required Changes**:

1. **src/cli/main.py (_run_pipeline)**
   - Parse `--distillation` as comma-separated list (like `--training-strategy`)
   - Validate: `len(distillation) == num_stages` (or auto-pad with no-kd)
   - Store per-job in `job['distillation']`
   - Pass to job manager in run_type

2. **src/cli/job_manager.py (format_training_args)**
   - Already extracts distillation from run_type ✓
   - No changes needed (already handles per-job distillation)

3. **Validation Logic**
   - Auto-reject fusion+with-kd at job level (not pipeline level)
   - Allow fusion+no-kd with other jobs using with-kd
   - Warn if distillation list shorter than num_stages (auto-pad with no-kd)

**Test Cases**:
```bash
# Test 1: Per-job KD config
./can-train pipeline --training-strategyl vgae,gat --distillation with-kd,no-kd ...
# Expected: VGAE gets KD, GAT doesn't

# Test 2: Fusion + mixed KD
./can-train pipeline --training-strategyl vgae,gat,dqn --training-strategy autoencoder,curriculum,fusion \
  --distillation with-kd,with-kd,no-kd ...
# Expected: First two with KD, fusion without

# Test 3: Auto-padding
./can-train pipeline --training-strategyl vgae,gat,dqn --distillation with-kd ...
# Expected: Warning that distillation has 1 value, auto-padding to 3 (with-kd,no-kd,no-kd)
```

**Files to Modify**:
- `src/cli/main.py` (_run_pipeline function, line ~810)
- No changes to job_manager.py (already works)

**Estimated Complexity**: Medium (requires careful list validation)

---

### 2. CLI Examples Consistency Audit

**Status**: PENDING - Systematic replacement

**Issue**: Throughout codebase, CLI examples are missing `--modality`

**Locations with CLI Examples**:
1. `src/cli/main.py` - Example commands in help text
2. `train_with_hydra_zen.py` - Docstring examples
3. `README.md` (if exists)
4. Logging messages when training completes

**Template for ALL Examples**:
```bash
./can-train {command} \
  --training-strategyl {MODEL} \
  --dataset {DATASET} \
  --modality automotive \
  {OTHER_ARGS}
```

**Search & Replace Pattern**:
```bash
# Find all CLI examples missing --modality
grep -r "can-train" . --include="*.py" --include="*.md"
```

**Files to Update**:
- `src/cli/main.py` - _list_example_configs() function
- `train_with_hydra_zen.py` - docstring examples
- Any markdown docs

**Test**: After changes, search for "can-train" and verify all examples include modality

---

## MEDIUM PRIORITY: Improvements

### 3. Teacher Path Auto-Discovery

**Status**: PENDING - Nice to have

**Current**: Must explicitly pass `--teacher_path /full/path/to/model.pth`

**Desired**: Auto-find teacher models based on dataset+model_type
```bash
# Instead of:
--teacher_path /long/path/to/vgae_teacher_hcrl_sa.pth

# Use:
--auto-find-teacher
# Searches: config/teacher_models/{dataset}/vgae_teacher.pth
```

**Implementation**:
- Create `config/teacher_models/{dataset}/` directory structure
- Add `--auto-find-teacher` flag to pipeline
- In job_manager, resolve path before passing to training script

---

### 4. Distillation Safety Factor Persistence

**Status**: PENDING - Enhancement

**Current**: Safety factors stored in `config/batch_size_factors.json` (static JSON)

**Desired**: Adaptive learning system via `SafetyFactorDatabase` class
- Track per-run memory usage
- Auto-adjust safety factors based on actual OOM/success patterns
- Learn model-specific factors over time

**Note**: Code already exists in `src/training/adaptive_batch_size.py`
- Just needs to be integrated into trainer.py callbacks

---

## LOW PRIORITY: Future Improvements

### 5. Sweep/Grid Frozen Config Pattern

**Status**: FUTURE - Enhancement for hyperparameter sweeps

**Current Behavior**:
- Each job gets its own frozen config: `{experiment_dir}/configs/frozen_config_{timestamp}.json`
- Sweeps generate multiple separate frozen configs

**Potential Enhancement - Sweep Manifest**:
```json
// experimentruns/sweeps/sweep_20260126_210000/manifest.json
{
  "sweep_id": "hp_search_lr_batch",
  "created_at": "2026-01-26T21:00:00",
  "grid_definition": {
    "learning_rate": [0.001, 0.003, 0.005],
    "batch_size": [32, 64, 128]
  },
  "configs": [
    {"id": 1, "path": "config_001.json", "params": {"lr": 0.001, "batch": 32}},
    {"id": 2, "path": "config_002.json", "params": {"lr": 0.001, "batch": 64}},
    // ...
  ],
  "total_combinations": 9
}
```

**Benefits**:
- Track which configs belong to same sweep
- Easy to compare results across grid
- Reproduce entire sweep from manifest
- MLflow experiment grouping

**Alternatives Considered**:
1. **Individual frozen configs (CURRENT)**: Simple, works for most cases
2. **Grid manifest (FUTURE)**: Better for large sweeps, hyperparameter search
3. **Cartesian product expansion**: Generate all combos at submission time

**Decision**: Keep current approach (individual configs), add manifest as future enhancement when sweep functionality matures.

---

### 6. Logging Improvements

**Status**: FUTURE - Enhancement for better debugging and monitoring

**Current Logging**:
- Basic stdout/stderr via SLURM
- MLflow for training metrics
- Job ID, Node, Start/End time in SLURM output

**Proposed Improvements**:

1. **GPU Memory Tracking**
   ```python
   # In trainer.py callback
   peak_memory = torch.cuda.max_memory_allocated() / 1024**3
   logger.info(f"Peak GPU Memory: {peak_memory:.2f} GB")
   ```
   - Track peak memory usage per epoch
   - Log to MLflow as metric
   - Help diagnose OOM issues

2. **Training Summary in SLURM Output**
   - Final metrics (accuracy, F1, loss)
   - Best checkpoint info
   - Total training time
   - Parameter count

3. **Dataset Statistics Logging**
   - Total samples per split
   - Class distribution
   - Graph statistics (nodes, edges)

4. **Structured JSON Logs** (Optional)
   - Machine-parseable log format
   - Easier aggregation across runs

**Implementation Location**:
- `src/training/trainer.py` - Memory tracking callback
- `src/cli/job_manager.py` - SLURM template additions
- `src/training/lightning_modules.py` - Dataset stat logging

---

### 7. Update README with KD Examples

**Status**: PENDING

**Content**:
- KD architecture explanation
- Complete pipeline examples with per-mode configuration
- Teacher model training examples
- Troubleshooting section

---

## COMPLETED WORK (Reference)

### Session 2026-01-26 (Frozen Config & Cleanup)
✅ **Validation Simplification** - Separated duties across 4 layers:
   - `pydantic_validators.py`: CLI input + P→Q rules only
   - `config_builder.py`: Bucket parsing + config construction (removed choice validation)
   - `hydra_zen_configs.py`: Config schema definitions only (removed validate_config)
   - `validator.py`: ALL pre-flight validation consolidated here
✅ **SLURM Logs Moved** - Logs now go to `{experiment_dir}/slurm_logs/` instead of centralized folder
✅ **Loose Configs Cleanup**:
   - Removed unused `jobs/pipeline_vgae_curriculum_fusion.json`
   - Added `parameters/README.md` explaining documentation-only status
   - Kept `config/batch_size_factors.json` (actively used)
✅ **Documentation Updates**:
   - Added sweep manifest as future improvement to PENDING_WORK.md
   - Updated PIPELINE_INVESTIGATION.md with new log paths
   - Added logging improvements plan to PENDING_WORK.md

### Previous Sessions
✅ Fixed `format_training_args()` to pass `--use-kd` and `--teacher_path` flags
✅ Added pipeline validation to reject fusion+with-kd
✅ Added `--teacher_path` parameter to pipeline CLI
✅ Auto-enforce `--training-strategyl-size student` when with-kd used
✅ Created KDHelper class with projection layer support
✅ Added KD safety factors to batch_size_factors.json
✅ Implemented VGAE KD (dual-signal: latent + reconstruction)
✅ Implemented GAT KD (soft label distillation)
✅ Fixed VGAE/GAT gradient checkpoint closure bugs

---

## Ordering for Next Work Session

1. **FIRST**: Implement per-mode KD configuration (Issue #2)
   - Allows: vgae+autoencoder+with-kd, gat+curriculum+with-kd, dqn+fusion+no-kd
   - Unlocks flexible pipelines

2. **SECOND**: Audit & fix all CLI examples
   - Ensure EVERY example includes `--modality`
   - This prevents future confusion

3. **THIRD**: Test full KD pipeline end-to-end
   - With real teacher models
   - Monitor jobs for correct behavior

4. **LATER**: Optional enhancements (#3, #4, #5)
   - Auto-discovery of teacher paths
   - Adaptive safety factor learning
   - Documentation updates

---

## Known Issues & Workarounds

### Issue: Jobs seem to run but no KD happening
- **Cause**: Old SLURM scripts without `--use-kd` flag
- **Workaround**: Cancel old jobs (`scancel JOBID`), re-submit with new CLI
- **Fix**: Already applied (format_training_args updated)

### Issue: Config validation passes but training fails
- **Cause**: Teacher model file not found at runtime
- **Workaround**: Verify path exists: `ls /path/to/model.pth`
- **Fix**: Added pre-SLURM validation in pipeline CLI

### Issue: Fusion + KD rejected too aggressively
- **Cause**: Current validation rejects entire pipeline if ANY job is fusion+kd
- **Workaround**: Use separate pipelines (one with KD, one without)
- **Fix**: Implement per-mode distillation configuration (Issue #2)

---

## Testing Checklist

Before considering KD work COMPLETE, verify:

- [ ] Single job with `--use-kd` works end-to-end
- [ ] VGAE KD: latent vectors matching teacher
- [ ] GAT KD: accuracy matching/exceeding baseline
- [ ] Pipeline with 2+ jobs and mixed KD config submits correctly
- [ ] Fusion+no-kd with other jobs using KD works
- [ ] All CLI examples in code include `--modality`
- [ ] Job logs show "Knowledge Distillation: ENABLED"
- [ ] Safety factors auto-selected for KD jobs (×0.75 reduction)
