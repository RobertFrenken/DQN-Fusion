# Parameter Overlap & Mismatch Audit

**Date:** 2026-01-27
**Status:** ⚠️ 5 CRITICAL MISMATCHES FOUND

## Executive Summary

Comprehensive audit of all parameter definitions across config classes, CLI arguments, and default values revealed **13 parameters defined in multiple locations**, with **5 having conflicting default values**.

---

## CRITICAL MISMATCHES (Action Required)

### 🔴 Priority 1: Conflicting Defaults

| Parameter | Location 1 | Default 1 | Location 2 | Default 2 | Impact |
|-----------|------------|-----------|------------|-----------|--------|
| **learning_rate** | OptimizerConfig:24 | **0.005** | BaseTrainingConfig:318 | **0.003** | 40% difference - unclear which applies |
| **epochs/max_epochs** | DEFAULT_MODEL_ARGS:22 | **50** (epochs) | BaseTrainingConfig:316 | **400** (max_epochs) | 8x difference - CRITICAL naming inconsistency |
| **replay_buffer_size** | DQNConfig:212 | **100000** | DEFAULT_MODEL_ARGS:42 | **10000** | 10x difference - major memory impact |
| **weight_decay** | BaseTrainingConfig:319 | **0.0001** | DEFAULT_MODEL_ARGS:27 | **0.0** | Zero vs non-zero regularization |
| **patience** | BaseTrainingConfig:326 | **100** | DEFAULT_MODEL_ARGS:29 | **10** | 10x difference in early stopping |

### 🟡 Priority 2: Consistent but Duplicated

| Parameter | Locations | Default | Issue |
|-----------|-----------|---------|-------|
| **batch_size** | 3 places: BaseTrainingConfig, BatchSizeConfig.default_batch_size, DEFAULT_MODEL_ARGS | 64 | Consistent but unclear precedence |
| **optimize_batch_size** | 3 places: BaseTrainingConfig, BatchSizeConfig, DEFAULT_MODEL_ARGS | True | Fixed by __post_init__ sync, still duplicated |
| **gradient_checkpointing** | 3 places: MemoryOptimizationConfig, DEFAULT_MODEL_ARGS, CLI | True | Consistent but duplicated |

### 🟢 Priority 3: Minor Mismatches

| Parameter | Location 1 | Default 1 | Location 2 | Default 2 | Impact |
|-----------|------------|-----------|------------|-----------|--------|
| **gamma** (DQN) | DQNConfig:207 | **0.99** | DEFAULT_MODEL_ARGS:41 | **0.95** | 4% difference in discount factor |

---

## Complete Parameter Inventory

### 1. Config Classes (src/config/hydra_zen_configs.py)

#### OptimizerConfig (lines 22-27)
```python
name: str = "adam"
lr: float = 0.005                    # ⚠️ MISMATCH with BaseTrainingConfig.learning_rate
weight_decay: float = 0.0001
momentum: float = 0.9
```

#### SchedulerConfig (lines 30-37)
```python
use_scheduler: bool = False
scheduler_type: str = "cosine"
params: dict = field(default_factory=dict)
```

#### MemoryOptimizationConfig (lines 39-44)
```python
use_teacher_cache: bool = True
clear_cache_every_n_steps: int = 100
offload_teacher_to_cpu: bool = False
gradient_checkpointing: bool = True  # ✓ Also in DEFAULT_MODEL_ARGS (consistent)
```

#### BatchSizeConfig (lines 46-87)
```python
default_batch_size: int = 64         # ✓ Matches BaseTrainingConfig.batch_size
tuned_batch_size: Optional[int] = None
safety_factor: float = 0.5
optimize_batch_size: bool = True     # ✓ Fixed to True (was False)
batch_size_mode: str = "binsearch"
max_batch_size_trials: int = 10
```

#### BaseTrainingConfig (lines 313-399)
```python
mode: str = "normal"
max_epochs: int = 400                # ⚠️ MISMATCH with DEFAULT_MODEL_ARGS['epochs']=50
batch_size: int = 64                 # ✓ Matches BatchSizeConfig.default_batch_size
learning_rate: float = 0.003         # ⚠️ MISMATCH with OptimizerConfig.lr=0.005
weight_decay: float = 0.0001         # ⚠️ MISMATCH with DEFAULT_MODEL_ARGS=0.0
early_stopping_patience: int = 100   # ⚠️ MISMATCH with DEFAULT_MODEL_ARGS['patience']=10
gradient_clip_val: float = 1.0
accumulate_grad_batches: int = 1
precision: str = "32-true"
optimize_batch_size: bool = True     # ✓ Synced to BatchSizeConfig in __post_init__
batch_size_mode: str = "binsearch"
max_batch_size_trials: int = 10
use_adaptive_batch_size_factor: bool = True
graph_memory_safety_factor: Optional[float] = None
save_top_k: int = 3
monitor_metric: str = "val_loss"
monitor_mode: str = "min"
log_every_n_steps: int = 50
seed: Optional[int] = None
deterministic_training: bool = True
use_knowledge_distillation: bool = False
teacher_model_path: Optional[str] = None
distillation_temperature: float = 4.0
distillation_alpha: float = 0.7
```

#### DQNConfig (lines 195-221)
```python
gamma: float = 0.99                  # ⚠️ MISMATCH with DEFAULT_MODEL_ARGS=0.95
epsilon_start: float = 1.0
epsilon_end: float = 0.01
epsilon_decay: float = 0.995
replay_buffer_size: int = 100000     # ⚠️ CRITICAL 10x MISMATCH with DEFAULT_MODEL_ARGS=10000
target_update_freq: int = 100
# ... (other DQN params)
```

### 2. CLI Arguments (src/cli/main.py)

#### Model & Training Arguments (lines 218-243)
```python
--learning-rate, --lr                # Override only (no default)
--batch-size                         # Override only (no default)
--epochs                             # Override only (no default) - maps to max_epochs
--hidden-channels                    # Override only (no default)
--dropout                            # Override only (no default)
--weight-decay                       # Override only (no default)
--num-layers                         # Override only (GAT)
--heads                              # Override only (GAT)
--latent-dim                         # Override only (VGAE)
--replay-buffer-size                 # Override only (DQN)
--gamma                              # Override only (DQN)
```

#### Training Control (lines 291-340)
```python
--early-stopping / --no-early-stopping  # default=True
--patience                              # Override only (no default)
--optimize-batch-size / --no-optimize-batch-size  # default=True ✓
--gradient-checkpointing / --no-gradient-checkpointing  # default=True ✓
```

#### SLURM Options (lines 345-355)
```python
--account                            # default=None (uses DEFAULT_SLURM_ARGS)
--partition                          # default=None
--walltime                           # default=None
--memory                             # default=None
--cpus                               # default=None
--gpus                               # default=1
--gpu-type                           # default=None
--dependency                         # default=None (NEW - for job dependencies)
```

### 3. Default Values (src/cli/config_builder.py)

#### DEFAULT_MODEL_ARGS (lines 21-43)
```python
'epochs': 50,                        # ⚠️ CRITICAL: BaseTrainingConfig uses max_epochs=400
'learning_rate': 0.003,              # ✓ Matches BaseTrainingConfig
'batch_size': 64,                    # ✓ Matches BaseTrainingConfig
'hidden_channels': 128,
'dropout': 0.2,
'weight_decay': 0.0,                 # ⚠️ MISMATCH: BaseTrainingConfig=0.0001
'early_stopping': True,
'patience': 10,                      # ⚠️ MISMATCH: BaseTrainingConfig=100
'gradient_checkpointing': True,      # ✓ Matches MemoryOptimizationConfig
'optimize_batch_size': True,         # ✓ Matches BatchSizeConfig

# GAT-specific
'num_layers': 3,
'heads': 4,

# VGAE-specific
'latent_dim': 16,

# DQN-specific
'gamma': 0.95,                       # ⚠️ MISMATCH: DQNConfig=0.99
'replay_buffer_size': 10000,         # ⚠️ CRITICAL 10x MISMATCH: DQNConfig=100000
```

#### DEFAULT_SLURM_ARGS (lines 45-53)
```python
'walltime': '06:00:00',
'memory': '64G',
'cpus': 16,
'gpus': 1,
'gpu_type': 'v100',
'account': 'PAS3209',
'partition': 'gpu',
```

### 4. Validators (src/cli/pydantic_validators.py)

**No parameter defaults defined** - only validation rules:
- learning_type ↔ model consistency (lines 117-148)
- mode ↔ learning_type consistency (lines 150-180)
- distillation ↔ mode consistency (lines 182-207)
- Prerequisite checks for curriculum/fusion/distillation (lines 209-322)

---

## Parameter Precedence Flow

When a parameter is specified/resolved, the precedence is:

```
1. CLI args (--epochs 100)
     ↓
2. config_builder.py mapping logic (epochs → max_epochs)
     ↓
3. DEFAULT_MODEL_ARGS fallback (if CLI not specified)
     ↓
4. Dataclass field defaults (if not in DEFAULT_MODEL_ARGS)
     ↓
5. __post_init__ synchronization (for duplicated params)
```

**Issue**: Step 3 and 4 have conflicting values for several parameters!

---

## Current Sync Logic (Added in Recent Fix)

In `CANGraphConfig.__post_init__()` (hydra_zen_configs.py:657-660):

```python
# Sync batch_size_config with training config fields
if hasattr(self.training, 'optimize_batch_size'):
    self.batch_size_config.optimize_batch_size = self.training.optimize_batch_size
if hasattr(self.training, 'batch_size_mode'):
    self.batch_size_config.batch_size_mode = self.training.batch_size_mode
if hasattr(self.training, 'max_batch_size_trials'):
    self.batch_size_config.max_batch_size_trials = self.training.max_batch_size_trials
```

This handles **optimize_batch_size** duplication but doesn't address:
- learning_rate mismatch
- epochs/max_epochs naming inconsistency
- DQN parameter mismatches
- weight_decay/patience mismatches

---

## Recommendations

### Immediate Actions (Before Next Run)

1. **Decide on learning_rate source of truth**
   - Option A: Use OptimizerConfig.lr (0.005) everywhere
   - Option B: Use BaseTrainingConfig.learning_rate (0.003) everywhere
   - Option C: Keep both, document which applies when

2. **Standardize epochs terminology**
   - Remove `'epochs'` from DEFAULT_MODEL_ARGS entirely
   - Use only `max_epochs` throughout codebase
   - Update CLI argument name from `--epochs` to `--max-epochs` (or keep as alias)

3. **Fix DQN parameter mismatches**
   - Decide: should DEFAULT_MODEL_ARGS match DQNConfig or vice versa?
   - Current 10x difference in replay_buffer_size is likely a bug

4. **Consolidate weight_decay and patience**
   - Align DEFAULT_MODEL_ARGS with BaseTrainingConfig values
   - Document why they differ if intentional

### Long-term Refactoring

1. **Single Source of Truth Pattern**
   - Remove DEFAULT_MODEL_ARGS entirely
   - Use only dataclass field defaults
   - CLI bucket parser reads directly from config classes

2. **Eliminate __post_init__ hacks**
   - Don't define same parameter in multiple dataclasses
   - Use composition instead of duplication

3. **Parameter Documentation**
   - Add docstrings explaining precedence for every parameter
   - Create parameter registry/map for reference

---

## Files Analyzed

- ✅ src/config/hydra_zen_configs.py (all dataclasses)
- ✅ src/cli/main.py (CLI argument definitions)
- ✅ src/cli/config_builder.py (DEFAULT_MODEL_ARGS, DEFAULT_SLURM_ARGS)
- ✅ src/cli/pydantic_validators.py (validation rules)

---

## Next Steps

See `.claude/INDEX.md` TODO list for tracking resolution of each mismatch.

**Current Status**: All mismatches documented, awaiting decisions on resolution strategy.
