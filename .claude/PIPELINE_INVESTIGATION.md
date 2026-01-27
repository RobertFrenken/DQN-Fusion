# Pipeline & Frozen Config Investigation Report

**Date**: 2026-01-26
**Purpose**: Thorough analysis of pipeline structure, validation layers, and config management

---

## 1. Job Submission Flow

### 1.1 Single Job Submission

```bash
./can-train train --model gat --dataset hcrl_sa --training-strategy normal \
    --learning-type supervised --modality automotive --dry-run
```

**Flow:**
```
CLI (main.py) → args parsed
    ↓
validate_cli_config() [pydantic_validators.py]
    ↓
CANGraphCLIConfig (Pydantic model validated)
    ↓
build_config_from_buckets() [config_builder.py]
    ↓
create_can_graph_config() → CANGraphConfig [hydra_zen_configs.py]
    ↓
ExecutionRouter.execute_single() [executor.py]
    ↓
If mode='slurm': JobManager.submit_single() [job_manager.py]
    ↓
save_frozen_config() → JSON file
    ↓
generate_script() → SLURM script with --frozen-config path
    ↓
submit_job() → sbatch
```

### 1.2 Pipeline Job Submission

```bash
./can-train pipeline --model vgae,gat,dqn \
    --learning-type unsupervised,supervised,rl_fusion \
    --training-strategy autoencoder,curriculum,fusion \
    --dataset hcrl_sa --modality automotive --submit
```

**Flow:**
```
CLI (main.py) → _run_pipeline()
    ↓
Parse comma-separated params → jobs list
    ↓
Manual validation (length matching, KD constraints)
    ↓
FOR each job in jobs:
    ↓
    JobManager.submit_single(config=None, run_type=job)
        ↓
        config_builder.create_can_graph_config(run_type)
        ↓
        save_frozen_config() → JSON file (EACH JOB GETS ONE!)
        ↓
        generate_script() → SLURM with --frozen-config
        ↓
        submit_job() → sbatch
        ↓
    Add dependency: --dependency=afterok:{prev_job_id}
```

**Key Insight**: Each pipeline stage gets its own frozen config file. Dependencies are chained via SLURM afterok.

---

## 2. Validation Methods Analysis

### Current Validation Stack (4 LAYERS!)

| Layer | File | What It Does | Overlap Risk |
|-------|------|--------------|--------------|
| 1. Pydantic | `pydantic_validators.py` | Type validation, P→Q rules, learning_type↔model consistency | **HIGH** - rules duplicated in config_builder |
| 2. Config Builder | `config_builder.py` | Creates HydraZen configs, some validation | **HIGH** - repeats pydantic rules |
| 3. HydraZen | `hydra_zen_configs.py` | Dataclass structure, `__post_init__` validation, methods | **MEDIUM** - structural, less rule duplication |
| 4. Validator | `validator.py` | Pre-flight checks (artifacts exist, paths valid) | **LOW** - different purpose (runtime checks) |

### Overlap Analysis

**Rules duplicated between Pydantic and Config Builder:**
- Model ↔ learning_type mapping (vgae→unsupervised, dqn→rl_fusion, gat→supervised)
- Mode ↔ learning_type consistency
- Distillation constraints

**Recommendation**: Consolidate validation to ONE authoritative source.

### Validation Simplification Options

| Option | Pros | Cons |
|--------|------|------|
| **Keep Pydantic only** | Early validation, clear error messages, type safety | Need to update hydra_zen_configs to trust pydantic |
| **Keep HydraZen only** | Single source, closer to actual config | Lose pydantic's nice validation DSL |
| **Hybrid: Pydantic for CLI, HydraZen for runtime** | Separation of concerns | Still some duplication |

**Recommended**: Option 3 (Hybrid) with clear ownership:
- Pydantic: CLI input validation ONLY (user-facing)
- HydraZen: Config structure and business logic
- Validator: Pre-flight artifact checks

---

## 3. SLURM Log Pathing

### Current Structure (UPDATED 2026-01-26)

SLURM logs and scripts now live **inside the experiment directory**:

```
experimentruns/automotive/hcrl_sa/supervised/gat/teacher/no_distillation/normal/
├── configs/
│   └── frozen_config_20260126_202947.json
├── slurm_logs/                    ← SLURM logs and scripts live here
│   ├── gat_hcrl_sa_normal_20260126_202947.out
│   ├── gat_hcrl_sa_normal_20260126_202947.err
│   └── gat_hcrl_sa_normal.sh
├── checkpoints/
└── models/
```

**Benefits of this approach**:
- Easy to correlate logs with specific experiment runs
- All artifacts in one place for debugging
- Cleaner organization
- Can still find logs by timestamp

### Previous Structure (Legacy)

```
experimentruns/
├── slurm_runs/                    ← OLD: centralized logs (legacy fallback)
│   ├── hcrl_sa/
│   │   └── ...
└── automotive/
    └── ...
```

**Note**: Legacy behavior (centralized logs) is still available as fallback when experiment_dir is not available.

### What's Currently Logged

**In SLURM scripts:**
- Job ID, Node, Start/End time
- Frozen config path
- Python path
- Exit code

**Missing (potentially helpful):**
- GPU memory peak
- Training metrics summary
- Model parameter count
- Dataset statistics

---

## 4. Frozen Config Storage Structure

### Current Structure

```
experimentruns/{modality}/{dataset}/{learning_type}/{model}/{model_size}/{distillation}/{mode}/
└── configs/
    └── frozen_config_{timestamp}.json
```

**Example:**
```
experimentruns/automotive/hcrl_sa/supervised/gat/teacher/no_distillation/normal/
└── configs/
    ├── frozen_config_20260126_150000.json  ← Run 1
    ├── frozen_config_20260126_160000.json  ← Run 2
    └── frozen_config_20260126_170000.json  ← Run 3
```

### One Config per Combination? ✓

Yes! Each unique combination of parameters gets its own directory, and each run within that combination gets a timestamped frozen config.

### Pipeline Handling

Pipeline creates N frozen configs (one per stage):
```
experimentruns/automotive/hcrl_sa/
├── unsupervised/vgae/teacher/no_distillation/autoencoder/
│   └── configs/frozen_config_20260126_150000.json  ← Stage 1
├── supervised/gat/teacher/no_distillation/curriculum/
│   └── configs/frozen_config_20260126_150001.json  ← Stage 2
└── rl_fusion/dqn/teacher/no_distillation/fusion/
    └── configs/frozen_config_20260126_150002.json  ← Stage 3
```

### Perplexity's "Grid" vs "Individual Configs"

| Approach | Pros | Cons |
|----------|------|------|
| **Individual Configs (Current)** | Simple, one config per job, easy to reproduce | Many files for large sweeps |
| **Cartesian Grid Config** | Single file defines sweep space, compact | Complex parsing, harder to reproduce individual runs |

**Current choice is good** because:
- Each run is fully self-contained and reproducible
- Frozen configs are small (~5KB each)
- Easy debugging: just load the frozen config

**Future Improvement**: Could add a `sweep_manifest.json` that links related runs:
```json
{
  "sweep_name": "hyperparameter_search_2026_01_26",
  "configs": [
    "configs/frozen_config_20260126_150000.json",
    "configs/frozen_config_20260126_160000.json"
  ]
}
```

> **Status**: Documented as future improvement in `.claude/Tasks/PENDING_WORK.md` (Section 5)

---

## 5. Loose Configs Audit

### File Status Summary

| File | Status | Action |
|------|--------|--------|
| `config/batch_size_factors.json` | **ACTIVELY USED** | Keep - essential for batch size optimization |
| `parameters/required_cli.yaml` | **DOCUMENTATION ONLY** | Consider: remove or integrate |
| `parameters/dependencies.yaml` | **DOCUMENTATION ONLY** | Consider: remove or integrate |
| `jobs/pipeline_vgae_curriculum_fusion.json` | **APPEARS UNUSED** | Verify and potentially remove |

### Detailed Analysis

#### `config/batch_size_factors.json` - KEEP

**Used by:**
- `src/training/trainer.py:85` - Reads safety factors
- `src/training/adaptive_batch_size.py:89` - Batch optimization
- `src/training/knowledge_distillation.py:441` - KD-specific factors

**Content:**
```json
{
  "hcrl_ch": 0.6,
  "hcrl_sa": 0.55,
  "hcrl_ch_kd": 0.45,  // KD needs more memory
  ...
}
```

**Verdict**: Essential. Do NOT remove.

#### `parameters/required_cli.yaml` - DOCUMENTATION ONLY

**Problem**: Functions `load_parameter_bible()` and `load_dependency_schema()` exist but are **NEVER CALLED**.

**Evidence:**
```bash
grep -r "load_parameter_bible\|load_dependency_schema" src/
# Only returns the function definitions, no callers!
```

**Options:**
1. **Remove**: Delete these files and functions
2. **Integrate**: Actually use them for validation
3. **Keep as docs**: Rename to `parameters/README_CLI_SPEC.yaml`

**Recommendation**: Option 3 - Keep as documentation but rename to make status clear.

#### `jobs/pipeline_vgae_curriculum_fusion.json` - APPEARS UNUSED

**Content:**
```json
{
  "name": "Pipeline: VGAE → Curriculum GAT → Fusion DQN",
  "presets": ["autoencoder_hcrl_sa", "curriculum_hcrl_sa", "fusion_hcrl_sa"]
}
```

**Problem**: Searching for this file in codebase shows no usage:
```bash
grep -r "pipeline_vgae_curriculum" src/
# No results
```

**Recommendation**: Verify with user, then either:
1. Remove if truly unused
2. Integrate into a "preset pipelines" feature

---

## 6. Recommendations Summary

### Immediate Actions

1. **Test pipeline with frozen configs**:
   ```bash
   ./can-train pipeline --model vgae,gat --learning-type unsupervised,supervised \
       --training-strategy autoencoder,curriculum --dataset hcrl_sa \
       --modality automotive --dry-run
   ```

2. **Clean up unused files**:
   - Rename `parameters/*.yaml` to indicate they're documentation
   - Verify `jobs/` folder usage

3. **Fix SLURM log pathing** (optional):
   - Move logs to experiment directory for better correlation

### Medium-Term Improvements

1. **Consolidate validation**:
   - Define clear ownership for Pydantic vs HydraZen
   - Remove duplicate rules

2. **Add sweep manifest** (optional):
   - Track related frozen configs together

3. **Improve logging**:
   - Add GPU memory tracking
   - Add training summary to SLURM output

### Architecture Decision: Frozen Config vs Grid Config

**Stick with frozen configs** (individual per job) because:
- Simpler and more robust
- Each run fully reproducible from single file
- No complex cartesian expansion needed
- Perplexity's "grid" approach adds complexity without clear benefit for this use case

---

## 7. Quick Reference

### Submit Single Job
```bash
./can-train train --model gat --dataset hcrl_sa \
    --training-strategy normal --learning-type supervised \
    --modality automotive --submit
```

### Submit Pipeline
```bash
./can-train pipeline \
    --model vgae,gat,dqn \
    --learning-type unsupervised,supervised,rl_fusion \
    --training-strategy autoencoder,curriculum,fusion \
    --dataset hcrl_sa --modality automotive --submit
```

### Preview Without Submission
```bash
./can-train train ... --dry-run    # Shows SLURM script
./can-train pipeline ... --dry-run  # Shows pipeline summary
```

### Where Things Go

| What | Path |
|------|------|
| Frozen configs | `experimentruns/{canonical_path}/configs/frozen_config_{ts}.json` |
| SLURM logs | `experimentruns/{canonical_path}/slurm_logs/{job_name}_{ts}.out/.err` |
| SLURM scripts | `experimentruns/{canonical_path}/slurm_logs/{job_name}.sh` |
| Model checkpoints | `experimentruns/{canonical_path}/checkpoints/` |

> **Note**: SLURM logs now live INSIDE the experiment directory (updated 2026-01-26)
