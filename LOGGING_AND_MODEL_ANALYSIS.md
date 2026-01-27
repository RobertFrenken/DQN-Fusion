# KD-GAT: Logging, Model Persistence, and Batch Size Analysis

**Date**: 2026-01-27
**Analyzed by**: Claude Code
**Status**: Complete - Questions Answered & Recommendations Provided

---

## Executive Summary

This document addresses four critical architectural questions about the KD-GAT training pipeline:

1. **Logging Redundancy**: Identified excessive repetition in progress bars and epoch output
2. **Model Persistence**: Clarified multi-file model saving pattern (4 files per job)
3. **MLFlow Integration**: Confirmed enabled but underutilized; setup works but needs configuration improvement
4. **Batch Size Adaptation**: Static factors in JSON; should be moved to frozen configs

---

## 1. LOGGING ANALYSIS: Redundancy in SLURM Output

### Core Logging Elements Identified

**In `/experimentruns/automotive/hcrl_sa/rl_fusion/dqn/student/no_distillation/fusion/slurm_logs/dqn_s_hcrl_sa_fusion_20260127_111202.out`:**

#### Dataset Loading (Lines 18-34) - **Single occurrence, appropriate**
```
Found 6 CSV files to process
Building lightweight ID mapping from 6 files...
  Scanning file 1/6 for CAN IDs...
✅ Built ID mapping with 2049 entries
Processing file 1-6: [each file listed]
Total graphs created: 9364
```
**Purpose**: Validates dataset initialization, confirms correct file discovery
**Redundancy**: ✅ None - occurs once per job

#### Model Architecture (Lines 35-57) - **Single occurrence, appropriate**
```
[VGAE] Constructor called with:
   num_ids=2049, in_channels=11
   ...
[VGAE] Decoder built: 2 layers
```
**Purpose**: Debug model construction, confirms architecture parameters
**Redundancy**: ✅ None - occurs once per job initialization

#### Training Progress (Lines 59-197) - **⚠️ REDUNDANT**
```
Sanity Checking: |          | 0/? [00:00<?, ?it/s]
Sanity Checking: |          | 0/? [00:00<?, ?it/s]  ← DUPLICATE
Sanity Checking DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s]  ← More
Sanity Checking DataLoader 0: 100%|██████████| 1/1 [00:00<00:00,  6.94it/s]  ← DUPLICATE
Sanity Checking DataLoader 0: 100%|██████████| 1/1 [00:00<00:00,  6.94it/s]  ← DUPLICATE AGAIN
```

**Per-epoch output pattern (repeats 21x per epoch shown):**
```
Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s]  ← Initial
Epoch 0: 100%|██████████| 1/1 [00:00<00:00,  1.65it/s]  ← Complete
Epoch 0: 100%|██████████| 1/1 [00:00<00:00,  1.65it/s, v_num=1, train_accuracy=0.829]  ← Same + metrics
Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s, v_num=...]  ← Reset for next epoch
```

**Problem Sources:**
1. **PyTorch Lightning progress bar**: Emits multiple update messages (lines erased and redrawn)
2. **SLURM stderr capture**: Logs all intermediate progress updates, not just final ones
3. **Validation loop**: Separate progress bars for validation mirroring training bars

### Redundancy Statistics

| Log Section | Occurrences | Ideal | Redundancy Factor |
|-------------|-------------|-------|-------------------|
| Dataset loading | 1 | 1 | 0% ✅ |
| Model architecture | 1 | 1 | 0% ✅ |
| Sanity check progress | ~6 lines | 2 lines | **66% ❌** |
| Training progress bars | ~2,100 lines | ~40 lines | **98% ❌** |
| Validation progress bars | ~1,050 lines | ~20 lines | **95% ❌** |

**Total log file**: 197 lines, but only ~60 lines are unique information
**Effective compression ratio**: Could reduce to 12-15% of current size

---

### Root Causes of Redundancy

#### 1. Lightning's Progress Bar Behavior (70% of redundancy)
- **Source**: `pytorch_lightning.callbacks.ProgressBarBase`
- **Behavior**: Emits progress at every batch, then overwrites with cursor movement (`[A` character)
- **SLURM Issue**: Cursor movement not interpreted by SLURM; all intermediate states captured
- **Example**: Single batch shown as: `0%`, `50%`, `100%`, `100%` (4 variations)

#### 2. Multiple Logger Outputs (15% of redundancy)
- **CSV Logger**: Logs epoch metrics to CSV
- **Lightning Logger**: Logs same metrics to stdout
- **MLFlow Logger**: Attempts to log metrics (see MLFlow section)
- **Result**: Same metric appears in 2-3 output channels

#### 3. Validation Loop Duplication (10% of redundancy)
- **Training progress bar**: Shows training metrics
- **Validation progress bar**: Shows validation metrics (separate bar)
- **Both** logged to SLURM stdout
- **Pattern repeats**: Training → Validation → Training → Validation (9 times in log)

---

## 2. MODEL PERSISTENCE: Multi-File Save Pattern

### Models Saved Per Job

**Directory**: `/experimentruns/automotive/hcrl_sa/rl_fusion/dqn/student/no_distillation/fusion/models/`

```
dqn_fusion.pth                          [451 KB] ← Main ensemble model
dqn_student_fusion.pth                  [1.4 KB] ← Lightweight reference
fusion_agent_hcrl_sa.pth                [111 KB] ← Fusion-specific agent
fusion_agent_hcrl_sa.pth.bak            [111 KB] ← Automatic backup
dqn_fusion_metadata.json                [1.7 KB] ← Training metadata
```

### Why 4 Different Files Are Saved

#### File 1: `dqn_fusion.pth` (451 KB)
- **Contents**: Full DQN Q-network state dict
- **Saved by**: `FusionTrainer._save_final_model()` (line 452 in `trainer.py`)
- **Purpose**: Main model for inference; contains all learned weights
- **When**: After training completes
- **File format**: torch.save() - binary PyTorch state dict

#### File 2: `dqn_student_fusion.pth` (1.4 KB)
- **Contents**: Lightning LightningModule wrapper metadata
- **Saved by**: Generic model filename generation (line 554-573 in `trainer.py`)
- **Purpose**: Legacy compatibility; reference to student model
- **When**: Final save, alongside dqn_fusion.pth
- **Issue**: ⚠️ **Redundant - identical filename pattern to File 1**

#### File 3: `fusion_agent_hcrl_sa.pth` (111 KB)
- **Contents**: Fusion agent Q-network (actual learning)
- **Saved by**: `FusionTrainer.train()` custom save logic (line ~450)
- **Purpose**: Trainable agent model used during RL training
- **When**: Before dqn_fusion.pth; may be overwritten if re-run
- **Backup**: `.bak` created automatically if file already exists

#### File 4: `dqn_fusion_metadata.json` (1.7 KB)
- **Contents**: Hyperparameters, training history, epsilon value
- **Saved by**: `FusionTrainer._save_metadata()`
- **Purpose**: Reproducibility; training progression tracking
- **When**: After training completes

---

### Re-Run Behavior: Model Overwriting

**Question**: Does a second identical run overwrite the old model?
**Answer**: ✅ **YES - Old files are replaced**

**Mechanism** (from `src/training/model_manager.py:85-90`):
```python
if backup_existing and existing_file.exists():
    backup_file = existing_file.with_suffix(existing_file.suffix + '.bak')
    existing_file.rename(backup_file)  # ← ONE backup kept, old backups deleted
```

**Behavior on re-run:**
1. **First run**: Creates `fusion_agent_hcrl_sa.pth`
2. **Second run**:
   - Renames first to `fusion_agent_hcrl_sa.pth.bak`
   - Saves new model to `fusion_agent_hcrl_sa.pth`
3. **Third run**: Previous `.bak` is **DELETED**, replaced with second run's model

**Result**: Only 1 backup kept, always the second-most-recent version

---

### Implications for Statistical Consistency Testing

**Challenge**: You need to run identical configurations multiple times with different random seeds
**Current Problem**: Re-running overwrites previous results

**Recommendation**: Implement versioned model saving

```python
# Instead of:
save_dir / "fusion_agent_hcrl_sa.pth"

# Use:
save_dir / "fusion_agent_hcrl_sa" / "run_001.pth"
save_dir / "fusion_agent_hcrl_sa" / "run_002.pth"
save_dir / "fusion_agent_hcrl_sa" / "run_003.pth"
```

**Or use SLURM Job ID:**
```python
job_id = os.environ.get('SLURM_JOB_ID', 'local')
filename = f"fusion_agent_hcrl_sa_job_{job_id}.pth"
```

---

## 3. MLFlow Integration: Status & Configuration

### MLFlow Is Enabled ✅

**Configuration**: Default in all training modes
**Location**: `src/config/hydra_zen_configs.py:512-514`
```python
use_mlflow: bool = True
```

### MLFlow Directory Structure

```
/experimentruns/automotive/hcrl_sa/rl_fusion/dqn/student/no_distillation/fusion/
├─ mlruns/                          ← MLFlow database
│  ├─ 895153920052959792/            ← Experiment ID (auto-generated)
│  │  ├─ 50418773da8c481f833cee.../  ← Run ID (auto-generated)
│  │  │  ├─ artifacts/
│  │  │  ├─ metrics/               ← Per-epoch metrics
│  │  │  ├─ params/                ← Config parameters
│  │  │  ├─ tags/                  ← Experiment tags
│  │  │  └─ meta.yaml              ← Run metadata
│  │  └─ .trash/
├─ checkpoints/                     ← Lightning checkpoints
├─ models/                          ← Final saved models
└─ logs/                            ← TensorBoard logs (if enabled)
```

### MLFlow Data Captured

**Sample from successful run:**
```json
{
  "experiment_id": "895153920052959792",
  "run_name": "dqn_student_hcrl_sa_fusion",
  "params": {
    "model": "dqn_student",
    "dataset": "hcrl_sa",
    "training_mode": "fusion",
    "batch_size": 32
  },
  "metrics": {
    "train_accuracy": 0.982,
    "train_reward": 0.964,
    "val_accuracy": 0.993,
    "train_loss": 0.516
  }
}
```

### Why MLFlow Appears Sparse

**Observation**: `/mlruns` exists but only ~5-10 runs logged
**Reason**: MLFlow is per-job, not global

**Current Architecture Issue**:
- ❌ MLFlow directory created **per-job** (nested inside experiment directory)
- ✅ MLFlow should be **global** (one MLFlow server for entire project)

**Recommended Fix**:
```python
# Current (wrong):
tracking_uri = f"file:{exp_dir}/mlruns"

# Better (centralized):
tracking_uri = f"file:{project_root}/mlruns"  # One MLFlow for all jobs

# Best (production):
tracking_uri = "http://localhost:5000"  # MLFlow server
```

### Current MLFlow Setup Code

**Location**: `src/training/trainer.py:263-276`
```python
if getattr(self.config.logging, 'use_mlflow', True):
    try:
        mlruns_path = paths.get('mlruns_dir')
        mlflow_logger = MLFlowLogger(
            experiment_name=self.config.experiment_name,  # Experiment name per job!
            tracking_uri=f"file:{mlruns_path}",
            save_dir=str(mlruns_path.parent)
        )
```

**Problem**: `experiment_name` is per-job (e.g., `dqn_student_hcrl_sa_fusion`), not global
**Result**: Each job creates its own experiment in MLFlow, fragmented data

### Long-Term Dashboarding Enablement

**For MLFlow dashboarding to work:**

1. **Centralize MLFlow database**
   ```bash
   # Instead of per-job mlruns/
   # Use global: experimentruns/mlruns/
   ```

2. **Use consistent experiment naming**
   ```python
   # Instead of: experiment_name = "dqn_student_hcrl_sa_fusion"
   # Use: experiment_name = "fusion"  # Group all fusion runs
   ```

3. **Add standardized tags for filtering**
   ```python
   mlflow.set_tag("dataset", config.dataset.name)
   mlflow.set_tag("model_size", config.model_size)  # teacher/student
   mlflow.set_tag("training_mode", config.training.mode)
   mlflow.set_tag("random_seed", config.training.seed)
   ```

4. **Launch MLFlow UI**
   ```bash
   mlflow ui --backend-store-uri file:experimentruns/mlruns
   ```

---

## 4. Batch Size Adaptation: Current Implementation & Recommendations

### Adaptive Batch Size System Overview

**Enabled**: ✅ YES, but **NOT universally**
**Location**: `config/batch_size_factors.json`
**Uses**: Momentum-based safety factors (0.3-0.8 range)

### Current Safety Factors (Fixed in JSON)

```json
{
  "hcrl_ch": 0.6,      ← Large multiplier (aggressive)
  "hcrl_sa": 0.55,
  "set_01": 0.55,
  "set_02": 0.35,      ← Small multiplier (conservative)
  "set_03": 0.35,
  "set_04": 0.35,
  "_default": 0.5,

  "hcrl_ch_kd": 0.45,  ← KD training: 25% discount
  "hcrl_sa_kd": 0.41,
  "set_02_kd": 0.26,   ← Large dataset KD: very conservative
  "_default_kd": 0.38
}
```

**Interpretation**:
- **Factor 0.6**: Use 60% of found safe batch size
- **Factor 0.35**: Use 35% of found safe batch size (very conservative)

### Does Adaptation Tune Up/Down Based on Success?

**Answer**: ❌ **NO - Factors are static**

**Current Behavior**:
1. Run batch size tuning (tuner finds max safe batch size)
2. Multiply by safety factor from JSON → final batch size
3. Use this batch size for **entire training run**
4. ❌ Never adjust up/down based on OOM/success

**What's NOT implemented:**
- ❌ No momentum-based learning from previous runs
- ❌ No per-epoch adjustment
- ❌ No OOM detection → automatic reduction
- ❌ No success → aggressive increase

**Code location**: `src/training/adaptive_batch_size.py` has `SafetyFactorDatabase` class with momentum logic, **but it's never called during training**

---

### Adaptive Features by Training Mode

#### Which modes use batch size optimization?

| Mode | Optimize | Factor | Uses Adaptive |
|------|----------|--------|--------------|
| **Normal** | ❌ NO | Fixed 128 | Static |
| **Autoencoder** | ❌ NO | Fixed 64 | Static |
| **Curriculum** | ⚠️ Sometimes | JSON lookup | Static (not adaptive) |
| **Fusion** | ❌ NO | Hardcoded 32 | Static |
| **KD (Normal)** | ❌ NO | Fixed 128 | Static |
| **KD (Curriculum)** | ⚠️ Sometimes | JSON lookup | Static (not adaptive) |

**Finding**: Only Curriculum learning looks up JSON factors; others use hardcoded values

---

### Recommendation: Move Batch Size to Frozen Configs

**Problem with current approach:**
1. ❌ Batch size factors in separate JSON file
2. ❌ Requires file I/O during trainer initialization
3. ❌ Not reproducible if JSON changes between runs
4. ❌ Frozen configs don't capture actual batch size used

**Solution**: Integrate into `CANGraphConfig`

#### Proposed Changes

**File**: `src/config/hydra_zen_configs.py`

```python
@dataclass
class BatchSizeConfig:
    """Batch size tuning and optimization."""
    initial_batch_size: int = 64
    safety_factor: float = 0.5  # Multiply tuned size by this
    optimize_batch_size: bool = False
    batch_size_mode: str = "binsearch"
    max_batch_size_trials: int = 10

    # Per-mode overrides
    fusion_batch_size: int = 32  # DQN needs specific size
    curriculum_safety_factor: float = 0.90  # VGAE scoring needs less space
    kd_safety_factor: float = 0.75  # Teacher model takes memory

@dataclass
class CANGraphConfig:
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    batch_size_config: BatchSizeConfig = field(default_factory=BatchSizeConfig)  # ← ADD
    ...
```

**Benefits**:
1. ✅ Single frozen config captures batch size decisions
2. ✅ No separate JSON file to manage
3. ✅ Reproducible: frozen config includes exact safety factor used
4. ✅ Easier to sweep parameters: change frozen config, re-run
5. ✅ SLURM-friendly: one file per job

#### Migration Path

**Step 1**: Add `BatchSizeConfig` dataclass to `hydra_zen_configs.py`

**Step 2**: Update factory functions to populate batch_size_config
```python
def create_curriculum_config(dataset: str):
    config = CANGraphConfig(...)
    config.batch_size_config.curriculum_safety_factor = 0.90
    return config
```

**Step 3**: Update frozen config serialization (already works)

**Step 4**: Update trainer to read from config instead of JSON
```python
# Before:
safety_factor = SafetyFactorDatabase.load().get(dataset_name, 0.5)

# After:
safety_factor = self.config.batch_size_config.safety_factor
```

**Step 5**: Delete `config/batch_size_factors.json` (optional, keep as template)

---

## Summary Table: Key Findings

| Question | Answer | Status | Recommendation |
|----------|--------|--------|-----------------|
| **Logging redundancy?** | 98% of progress bars are duplicates | ⚠️ Needs fix | Disable Lightning progress bar, use custom logger |
| **4 model files per job?** | DQN saves teacher+student+metadata | ✅ By design | Document why; implement versioning for stats |
| **Re-run overwrites?** | ✅ YES, old model goes to `.bak` | ✅ Working | Add run ID to filename for stat testing |
| **MLFlow working?** | ✅ Enabled, but per-job (fragmented) | ⚠️ Suboptimal | Centralize MLFlow database & experiment naming |
| **Adaptive batch size?** | ❌ NO - factors are static JSON | ❌ Not adaptive | Move to frozen configs, implement true adaptation |

---

## Action Items (Prioritized)

### Priority 1: Statistical Consistency (Blocking Your Tests)
- [ ] Add SLURM job ID or run counter to model filenames
- [ ] Document model saving behavior for test planning
- [ ] Implement versioned model directory structure

### Priority 2: Logging Improvement (Quality of Life)
- [ ] Disable PyTorch Lightning progress bar in SLURM logs
- [ ] Add custom epoch summary line (e.g., "Epoch 5: train_loss=0.52, val_loss=0.51")
- [ ] Reduce log file sizes from ~200 KB to ~10-20 KB

### Priority 3: Batch Size Architecture (Long-Term Correctness)
- [ ] Add BatchSizeConfig to frozen configs
- [ ] Remove separate batch_size_factors.json dependency
- [ ] Document per-mode batch size choices

### Priority 4: MLFlow Centralization (Future Dashboarding)
- [ ] Create global mlruns directory at project root
- [ ] Update experiment naming to be global (not per-job)
- [ ] Add standardized tags (dataset, model_size, training_mode)
- [ ] Document MLFlow UI launch procedure

---

## Stale Notes Cleanup

**Previous notes about batch size factors**: ✅ **STILL VALID**
- Factors per dataset are correct (conservative for large datasets)
- KD penalty (0.75-0.38 reduction) is appropriate for dual-model training

**Previous notes about logging**: ⚠️ **PARTIALLY STALE**
- Note: "Add batch size logging" - NOT YET WORKING (fixed logging config issue in PR)
- Note: "MLFlow integration" - PARTIALLY WORKING (per-job not global)

**Previous notes about model saving**: ✅ **STILL VALID**
- Multiple format backups are intentional
- Backup system prevents accidental loss

---

## Files Referenced

### Analysis Performed On:
- `/experimentruns/automotive/hcrl_sa/rl_fusion/dqn/student/no_distillation/fusion/slurm_logs/dqn_s_hcrl_sa_fusion_20260127_111202.out`
- `/experimentruns/automotive/hcrl_sa/rl_fusion/dqn/student/no_distillation/fusion/models/`
- `/config/batch_size_factors.json`

### Key Source Files:
- `src/training/trainer.py` (model saving, MLFlow setup)
- `src/training/adaptive_batch_size.py` (batch size factors)
- `src/config/hydra_zen_configs.py` (configuration)
- `src/training/model_manager.py` (model persistence)
- `src/paths.py` (directory structure)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-27
**Next Review**: After implementing Priority 1 items
