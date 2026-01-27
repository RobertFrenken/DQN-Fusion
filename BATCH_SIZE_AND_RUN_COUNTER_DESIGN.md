# Batch Size Tuning Documentation & Run Counter Design

**Status**: Design Phase
**Target**: Implement run counter + batch size tracking for statistical consistency
**Timeline**: Before statistical testing begins

---

## Part 1: Run Counter Implementation

### Design Rationale
- **Why run counter instead of SLURM job ID?** Can run locally, on different HPC systems, or re-submit old jobs
- **Where stored?** In `experiment_dir/run_counter.txt`
- **Format?** Simple integer, incremented each time trainer initializes

### Implementation Strategy

#### 1.1: Add run counter tracking to `src/paths.py`

```python
class PathResolver:
    def get_run_counter(self, create: bool = False) -> int:
        """Get and increment run counter for this experiment.

        Returns the next run number (1-indexed).
        Creates/reads from experiment_dir/run_counter.txt

        Example:
            Run 1: returns 1, creates file with "2"
            Run 2: returns 2, updates file to "3"
            Run 3: returns 3, updates file to "4"
        """
        exp_dir = self.get_experiment_dir(create=create)
        counter_file = exp_dir / 'run_counter.txt'

        if counter_file.exists():
            with open(counter_file, 'r') as f:
                next_run = int(f.read().strip())
        else:
            next_run = 1

        # Write next value for future runs
        with open(counter_file, 'w') as f:
            f.write(str(next_run + 1))

        return next_run
```

#### 1.2: Model filename format

```python
# Current:
filename = f"{model_type}_{model_size}_{mode}.pth"
# Example: dqn_student_fusion.pth

# New:
run_num = self.paths.get_run_counter()
filename = f"{model_type}_{model_size}_{mode}_run_{run_num:03d}.pth"
# Example: dqn_student_fusion_run_001.pth
#         dqn_student_fusion_run_002.pth
#         dqn_student_fusion_run_003.pth
```

#### 1.3: Integration points

**File**: `src/training/trainer.py`

In `_train_standard()`, `_train_fusion()`, `_train_curriculum()`:
```python
def _train_standard(self):
    """Standard training (normal, KD, autoencoder)."""
    ...

    # At start of method
    run_num = self.paths.get_run_counter()
    logger.info(f"🔢 Run number: {run_num:03d}")

    ...training code...

    # When saving models (lines ~575-590)
    model_name = self._generate_model_filename(...)
    model_name = f"{model_name[:-4]}_run_{run_num:03d}.pth"  # Insert run number
    self._save_final_model(model, model_name)
```

#### 1.4: Directory structure after 3 runs

```
experimentruns/automotive/hcrl_sa/rl_fusion/dqn/student/no_distillation/fusion/
├─ run_counter.txt                          ← File with "4" (next run)
├─ configs/
│  ├─ frozen_config_run_001.json
│  ├─ frozen_config_run_002.json
│  └─ frozen_config_run_003.json
├─ models/
│  ├─ dqn_student_fusion_run_001.pth       ← Run 1 model
│  ├─ dqn_student_fusion_run_002.pth       ← Run 2 model (new random seed)
│  ├─ dqn_student_fusion_run_003.pth       ← Run 3 model
│  ├─ fusion_agent_hcrl_sa_run_001.pth
│  ├─ fusion_agent_hcrl_sa_run_002.pth
│  ├─ fusion_agent_hcrl_sa_run_003.pth
│  └─ fusion_agent_hcrl_sa.pth.bak         ← Latest backup (from run 3)
└─ slurm_logs/
   ├─ dqn_s_hcrl_sa_fusion_run_001.out
   ├─ dqn_s_hcrl_sa_fusion_run_002.out
   └─ dqn_s_hcrl_sa_fusion_run_003.out
```

---

## Part 2: Batch Size Tuning Documentation

### Design Philosophy

**Goal**: Create a feedback loop where successful batch sizes inform future runs

```
┌─────────────────────────────────────────────────────────┐
│ First Run (Run 1)                                       │
├─────────────────────────────────────────────────────────┤
│ 1. Load frozen_config (has default_batch_size: 64)     │
│ 2. If optimize_batch_size=true:                         │
│    - Run tuner → finds max safe size (e.g., 192)       │
│    - Multiply by safety_factor (0.55) → 105            │
│ 3. USE: 105 (run_batch_size)                            │
│ 4. Train successfully ✅                                │
│ 5. SAVE: tuned_batch_size = 105 to frozen config       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Second Run (Run 2, different random seed)               │
├─────────────────────────────────────────────────────────┤
│ 1. Load frozen_config (now has tuned_batch_size: 105)  │
│ 2. If optimize_batch_size=true:                         │
│    - Could skip tuner, use 105 directly (faster)       │
│    - OR re-tune but use 105 as starting point          │
│ 3. USE: 105 (or new tuned value if re-tuned)           │
│ 4. Train successfully ✅                                │
│ 5. SAVE: tuned_batch_size updated to frozen config     │
└─────────────────────────────────────────────────────────┘
```

### 2.1: BatchSizeConfig Dataclass

**File**: `src/config/hydra_zen_configs.py`

```python
@dataclass
class BatchSizeConfig:
    """Batch size tuning and memory optimization.

    Policy:
    - On first run: tuned_batch_size = default_batch_size
    - During run: run_batch_size = tuner_output * safety_factor
    - After success: update tuned_batch_size = run_batch_size

    The tuned_batch_size acts as a feedback mechanism:
    - If optimization is disabled: use tuned_batch_size directly
    - If optimization is enabled: use as warm-start for tuner
    """

    # Core tuning parameters
    default_batch_size: int = 64
    """Initial batch size if no tuning history exists."""

    tuned_batch_size: Optional[int] = None
    """Batch size successfully used in previous run (if any).

    Set after successful training to carry forward to next run.
    If None, defaults to default_batch_size.
    """

    safety_factor: float = 0.5
    """Multiply tuner output by this factor to account for GPU memory variability.

    Range: 0.3 (very conservative, large datasets)
           0.5 (moderate, default)
           0.8 (aggressive, small datasets)

    Tuner finds max safe batch size X, we use floor(X * safety_factor).
    """

    # Tuning control
    optimize_batch_size: bool = False
    """Run batch size tuner before training.

    If True: Use PyTorch Lightning's tuner to find max safe batch size
    If False: Use tuned_batch_size (or default_batch_size if not set)
    """

    batch_size_mode: str = "binsearch"
    """Strategy for batch size tuning: 'power' or 'binsearch'."""

    max_batch_size_trials: int = 10
    """Maximum number of batch sizes to try during tuning."""

    # Mode-specific overrides (optional, can be None)
    curriculum_safety_factor: Optional[float] = None
    """Override safety_factor for curriculum learning (uses VGAE for scoring).

    If set, used instead of safety_factor when training.mode == 'curriculum'.
    Typical: 0.90 (less aggressive, VGAE needs headroom)
    """

    kd_safety_factor: Optional[float] = None
    """Override safety_factor for knowledge distillation.

    If set, used instead of safety_factor when use_knowledge_distillation=true.
    Typical: 0.75 (teacher model + projection layer need extra memory)
    """

    fusion_batch_size: Optional[int] = None
    """Override batch_size for fusion training.

    Fusion/DQN needs specific batch sizes for stable RL training.
    If set, use this directly (don't tune).
    Typical: 32 (DQN-specific requirement)
    """
```

### 2.2: Batch Size Tuning Documentation Schema

**File**: `src/config/batch_size_tuning.py` (NEW FILE)

```python
"""
Batch Size Tuning Documentation and Logging.

This module tracks:
1. Initial config parameters
2. Tuner output (max safe batch size found)
3. Safety factor applied
4. Final run batch size
5. Training result (success/failure)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchSizeTuningRecord:
    """Single record of batch size tuning for one run."""

    run_number: int
    """Run counter (1, 2, 3, ...)."""

    timestamp: str
    """ISO format: '2026-01-27T13:11:00'."""

    config_key: str
    """Composite key: 'dataset_model_mode' e.g., 'hcrl_sa_dqn_fusion'."""

    # Input parameters
    default_batch_size: int
    """Initial batch size from config."""

    tuned_batch_size_previous: Optional[int]
    """Batch size from previous successful run (if any)."""

    optimize_batch_size: bool
    """Was tuner enabled?"""

    safety_factor: float
    """Factor applied (mode-specific or default)."""

    # Tuner output (if optimize_batch_size=true)
    tuner_max_batch_size: Optional[int] = None
    """Maximum batch size found by tuner."""

    tuner_trials: Optional[int] = None
    """Number of trials tuner ran."""

    tuner_status: Optional[str] = None
    """'success', 'failed', 'skipped', etc."""

    # Run result
    run_batch_size: int
    """Actual batch size used during training.

    Computed as:
    - If optimize_batch_size=true: floor(tuner_max_batch_size * safety_factor)
    - If optimize_batch_size=false: tuned_batch_size_previous or default_batch_size
    """

    training_succeeded: bool = False
    """Did training complete without OOM or errors?"""

    training_error: Optional[str] = None
    """Error message if training_succeeded=false."""

    # Recommendation for next run
    recommended_batch_size: Optional[int] = None
    """Should be set to run_batch_size if training_succeeded=true.

    Next run should use:
    - tuned_batch_size: run_batch_size (if successful)
    - optimize_batch_size: false (no need to re-tune if we have a working size)
    """

    notes: str = ""
    """Any additional notes about this run."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'BatchSizeTuningRecord':
        """Create from dict (for loading from JSON)."""
        return BatchSizeTuningRecord(**data)


@dataclass
class BatchSizeTuningLog:
    """Collection of batch size tuning records."""

    records: Dict[int, BatchSizeTuningRecord] = field(default_factory=dict)
    """Map of run_number -> tuning record."""

    def add_record(self, record: BatchSizeTuningRecord) -> None:
        """Add a new tuning record."""
        self.records[record.run_number] = record
        logger.info(
            f"📊 Batch size tuning record saved (run {record.run_number}): "
            f"{record.run_batch_size} (prev: {record.tuned_batch_size_previous})"
        )

    def get_latest_successful(self) -> Optional[BatchSizeTuningRecord]:
        """Get most recent successful run."""
        for run_num in sorted(self.records.keys(), reverse=True):
            if self.records[run_num].training_succeeded:
                return self.records[run_num]
        return None

    def save(self, path: Path) -> None:
        """Save log to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'records': {str(k): v.to_dict() for k, v in self.records.items()},
            'last_updated': datetime.now().isoformat()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 Batch size tuning log saved to {path}")

    @staticmethod
    def load(path: Path) -> 'BatchSizeTuningLog':
        """Load log from JSON file."""
        if not path.exists():
            return BatchSizeTuningLog()

        with open(path, 'r') as f:
            data = json.load(f)

        log = BatchSizeTuningLog()
        for run_num_str, record_data in data.get('records', {}).items():
            record = BatchSizeTuningRecord.from_dict(record_data)
            log.add_record(record)

        return log
```

### 2.3: Integration into Trainer

**File**: `src/training/trainer.py`

```python
def _train_standard(self):
    """Standard training (normal, KD, autoencoder)."""

    run_num = self.paths.get_run_counter()
    logger.info(f"🔢 Run number: {run_num:03d}")

    # Load dataset
    train_dataset, val_dataset, num_ids = load_dataset(...)

    # Setup model
    model = self.setup_model(num_ids)

    # ===== BATCH SIZE TUNING SECTION (NEW) =====
    from src.config.batch_size_tuning import BatchSizeTuningRecord, BatchSizeTuningLog

    # Determine safety factor (mode-specific override or default)
    safety_factor = self.config.batch_size_config.safety_factor
    if self.config.training.mode == 'curriculum':
        if self.config.batch_size_config.curriculum_safety_factor is not None:
            safety_factor = self.config.batch_size_config.curriculum_safety_factor
    elif self.config.training.get('use_knowledge_distillation', False):
        if self.config.batch_size_config.kd_safety_factor is not None:
            safety_factor = self.config.batch_size_config.kd_safety_factor

    # Initialize tuning record
    tuning_record = BatchSizeTuningRecord(
        run_number=run_num,
        timestamp=datetime.now().isoformat(),
        config_key=f"{self.config.dataset.name}_{self.config.model.type}_{self.config.training.mode}",
        default_batch_size=self.config.batch_size_config.default_batch_size,
        tuned_batch_size_previous=self.config.batch_size_config.tuned_batch_size,
        optimize_batch_size=self.config.batch_size_config.optimize_batch_size,
        safety_factor=safety_factor,
    )

    # Run batch size optimization if enabled
    if self.config.batch_size_config.optimize_batch_size:
        logger.info("🔧 Running batch size optimization...")
        try:
            model, tuner_result = self._optimize_batch_size_with_tracking(
                model, train_dataset, val_dataset, safety_factor
            )
            tuning_record.tuner_max_batch_size = tuner_result['max_batch_size']
            tuning_record.tuner_trials = tuner_result['num_trials']
            tuning_record.tuner_status = 'success'
            tuning_record.run_batch_size = model.batch_size
        except Exception as e:
            logger.warning(f"⚠️ Batch size optimization failed: {e}")
            tuning_record.tuner_status = 'failed'
            # Fallback to tuned_batch_size or default
            fallback_size = (
                self.config.batch_size_config.tuned_batch_size or
                self.config.batch_size_config.default_batch_size
            )
            model.batch_size = fallback_size
            tuning_record.run_batch_size = fallback_size
    else:
        logger.info(f"📊 Using fixed batch size: {model.batch_size}")
        tuning_record.run_batch_size = model.batch_size

    # Log the batch size being used
    logger.info(
        f"🎯 Training batch size: {tuning_record.run_batch_size} "
        f"(tuned_prev: {tuning_record.tuned_batch_size_previous}, "
        f"safety_factor: {safety_factor})"
    )

    # ===== END BATCH SIZE SECTION =====

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, model.batch_size
    )

    # ... rest of training ...

    # After successful training
    trainer.fit(model, train_loader, val_loader)

    # ===== UPDATE TUNING RECORD WITH RESULTS =====
    tuning_record.training_succeeded = True
    tuning_record.recommended_batch_size = tuning_record.run_batch_size

    # Save tuning record
    paths = self.get_hierarchical_paths()
    tuning_log_path = paths['experiment_dir'] / 'batch_size_tuning.json'
    tuning_log = BatchSizeTuningLog.load(tuning_log_path)
    tuning_log.add_record(tuning_record)
    tuning_log.save(tuning_log_path)

    # Update frozen config with tuned batch size for next run
    self._update_frozen_config_with_tuning(
        frozen_config_path=paths['frozen_config_path'],
        tuned_batch_size=tuning_record.run_batch_size,
        tuning_record=tuning_record
    )

    # ... save models ...
```

### 2.4: Frozen Config Update Helper

```python
def _update_frozen_config_with_tuning(
    self,
    frozen_config_path: Path,
    tuned_batch_size: int,
    tuning_record: 'BatchSizeTuningRecord'
) -> None:
    """Update frozen config with tuning results for next run."""

    from src.config.frozen_config import load_frozen_config, save_frozen_config

    try:
        # Load existing config
        config = load_frozen_config(frozen_config_path)

        # Update batch size config
        config.batch_size_config.tuned_batch_size = tuned_batch_size
        config.batch_size_config.optimize_batch_size = False  # Next run can skip tuning

        # Save updated config back (overwrites)
        save_frozen_config(config, frozen_config_path)

        logger.info(
            f"✅ Updated frozen config with tuned_batch_size={tuned_batch_size}"
        )
    except Exception as e:
        logger.warning(f"Could not update frozen config: {e}")
```

---

## Part 3: Batch Size Tuning Log Output

### 3.1: Example batch_size_tuning.json after 3 runs

```json
{
  "records": {
    "1": {
      "run_number": 1,
      "timestamp": "2026-01-27T13:11:00",
      "config_key": "hcrl_sa_dqn_fusion",
      "default_batch_size": 64,
      "tuned_batch_size_previous": null,
      "optimize_batch_size": true,
      "safety_factor": 0.55,
      "tuner_max_batch_size": 192,
      "tuner_trials": 8,
      "tuner_status": "success",
      "run_batch_size": 105,
      "training_succeeded": true,
      "training_error": null,
      "recommended_batch_size": 105,
      "notes": "First run, tuner found batch size 192 safe"
    },
    "2": {
      "run_number": 2,
      "timestamp": "2026-01-27T14:22:00",
      "config_key": "hcrl_sa_dqn_fusion",
      "default_batch_size": 64,
      "tuned_batch_size_previous": 105,
      "optimize_batch_size": false,
      "safety_factor": 0.55,
      "tuner_max_batch_size": null,
      "tuner_trials": null,
      "tuner_status": "skipped",
      "run_batch_size": 105,
      "training_succeeded": true,
      "training_error": null,
      "recommended_batch_size": 105,
      "notes": "Using tuned_batch_size from run 1 (no tuning needed)"
    },
    "3": {
      "run_number": 3,
      "timestamp": "2026-01-27T15:33:00",
      "config_key": "hcrl_sa_dqn_fusion",
      "default_batch_size": 64,
      "tuned_batch_size_previous": 105,
      "optimize_batch_size": true,
      "safety_factor": 0.55,
      "tuner_max_batch_size": 200,
      "tuner_trials": 6,
      "tuner_status": "success",
      "run_batch_size": 110,
      "training_succeeded": true,
      "training_error": null,
      "recommended_batch_size": 110,
      "notes": "Re-tuned after run 2, found slightly better batch size"
    }
  },
  "last_updated": "2026-01-27T15:35:00"
}
```

### 3.2: Example console output during training

```
🔢 Run number: 001
📊 Dataset loaded: 6190 training + 1050 validation = 7240 total
🔧 Running batch size optimization...
  Trying batch_size=64... ✓ OK
  Trying batch_size=128... ✓ OK
  Trying batch_size=256... ❌ OOM (CUDA out of memory)
  Trying batch_size=192... ✓ OK
  Tuner result: max safe batch size = 192
🎯 Training batch size: 105 (tuned_prev: None, safety_factor: 0.55)
  │ Batch size = 192 * 0.55 = 105.6 → 105
📊 Batch size tuning record saved (run 001): 105 (prev: None)
Starting training...
Epoch 0: 100%|██████████| 12/12 [00:05<00:00,  2.40it/s, train_loss=0.95]
Epoch 1: 100%|██████████| 12/12 [00:05<00:00,  2.42it/s, train_loss=0.87]
...
Training completed successfully ✅
✅ Updated frozen config with tuned_batch_size=105
💾 Saved final model to models/dqn_student_fusion_run_001.pth
```

---

## Part 4: Integration with Frozen Configs

### 4.1: New frozen_config_run_001.json structure

```json
{
  "_frozen_config_version": "1.0",
  "_frozen_at": "2026-01-27T13:11:00.000000",
  "_type": "CANGraphConfig",
  "model": { ... },
  "dataset": { ... },
  "training": { ... },
  "batch_size_config": {
    "_type": "BatchSizeConfig",
    "default_batch_size": 64,
    "tuned_batch_size": null,
    "safety_factor": 0.55,
    "optimize_batch_size": true,
    "batch_size_mode": "binsearch",
    "max_batch_size_trials": 10,
    "curriculum_safety_factor": null,
    "kd_safety_factor": null,
    "fusion_batch_size": null
  },
  "logging": { ... }
}
```

### 4.2: After successful run 1, frozen_config_run_002.json becomes

```json
{
  "_frozen_config_version": "1.0",
  "_frozen_at": "2026-01-27T14:22:00.000000",
  "_type": "CANGraphConfig",
  ...
  "batch_size_config": {
    "_type": "BatchSizeConfig",
    "default_batch_size": 64,
    "tuned_batch_size": 105,                    ← UPDATED by run 1
    "safety_factor": 0.55,
    "optimize_batch_size": false,               ← Set to false, skip tuning
    "batch_size_mode": "binsearch",
    "max_batch_size_trials": 10,
    "curriculum_safety_factor": null,
    "kd_safety_factor": null,
    "fusion_batch_size": null
  },
  ...
}
```

### 4.3: Multiple runs with different seeds

```bash
# Run 1: Random seed 42
python train_with_hydra_zen.py --frozen-config frozen_config_run_001.json --seed 42

# Run 2: Random seed 123 (different seed, uses batch size from run 1)
python train_with_hydra_zen.py --frozen-config frozen_config_run_002.json --seed 123

# Run 3: Random seed 456 (different seed, uses batch size from run 2)
python train_with_hydra_zen.py --frozen-config frozen_config_run_003.json --seed 456
```

---

## Part 5: Backward Compatibility

### 5.1: For existing configs without batch_size_config

When loading a frozen config without `batch_size_config`:
```python
# In frozen_config.py dict_to_config()
if 'batch_size_config' not in config_dict:
    # Create default
    config_dict['batch_size_config'] = {
        '_type': 'BatchSizeConfig',
        'default_batch_size': 64,
        'tuned_batch_size': None,
        'safety_factor': 0.5,
        'optimize_batch_size': False,
        ...
    }
```

### 5.2: For existing old configs with optimize_batch_size in training

If `config.training.optimize_batch_size` exists (old location):
```python
# In trainer.py _train_standard()
if hasattr(self.config.training, 'optimize_batch_size'):
    # Migrate to new location
    self.config.batch_size_config.optimize_batch_size = (
        self.config.training.optimize_batch_size
    )
```

---

## Part 6: File Changes Summary

### Files to Create:
1. `src/config/batch_size_tuning.py` - Tuning record classes
2. `BATCH_SIZE_TUNING_GUIDE.md` - User documentation

### Files to Modify:
1. `src/config/hydra_zen_configs.py` - Add BatchSizeConfig dataclass
2. `src/training/trainer.py` - Integrate tuning tracking
3. `src/paths.py` - Add get_run_counter() method
4. `src/training/model_manager.py` - Use run counter in filenames

### Files Not Modified (Still Valid):
1. `config/batch_size_factors.json` - Keep as reference, values move to frozen config
2. `src/training/batch_optimizer.py` - Use as-is, just enhance output
3. `src/training/adaptive_batch_size.py` - Keep for future adaptive learning

---

## Implementation Order

1. **Add run counter** → PathResolver.get_run_counter()
2. **Create BatchSizeConfig** → hydra_zen_configs.py
3. **Create tuning record classes** → batch_size_tuning.py
4. **Integrate into trainer** → _train_standard(), _train_fusion(), _train_curriculum()
5. **Update model filenames** → _generate_model_filename()
6. **Test with single run** → verify tuning log is created
7. **Test with multiple runs** → verify run counter increments, batch sizes tracked

---

## Logging Output Examples

### Console during tuning
```
🔢 Run number: 001
🔧 Running batch size optimization...
  Trying batch_size=64... ✓
  Trying batch_size=128... ✓
  Trying batch_size=256... ❌ OOM
  Trying batch_size=192... ✓
🎯 Training batch size: 105 (tuned_prev: None, safety_factor: 0.55)
```

### After successful training
```
✅ Training completed successfully
✅ Updated frozen config with tuned_batch_size=105
💾 Saved final model to models/dqn_student_fusion_run_001.pth
📊 Batch size tuning record saved (run 001): 105 (prev: None)
```

### For re-runs (using previous tuned size)
```
🔢 Run number: 002
📊 Using fixed batch size: 105 (from previous successful run)
🎯 Training batch size: 105 (tuned_prev: 105, safety_factor: 0.55)
```

---

**Status**: Design approved and ready for implementation
