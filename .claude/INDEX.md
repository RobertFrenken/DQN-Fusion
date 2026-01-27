# Claude Code Documentation Index

**Location**: `.claude/` folder (reference these files consistently!)

## Quick Navigation

### For Understanding the Project
👉 **Start here**: [PROJECT_OVERVIEW.md](system/PROJECT_OVERVIEW.md)
- Project architecture
- Model sizing strategy
- Dataset organization
- KD approach explanation
- File locations & modules
- Critical constraints

### For CLI Usage & Mistakes
👉 **CLI commands**: [CLI_BEST_PRACTICES.md](SOP/CLI_BEST_PRACTICES.md)
- RULE 1: Always include `--modality`
- RULE 2: Pipeline comma-separated parameters
- RULE 3: KD configuration requirements
- Common mistakes and fixes
- Template commands for common scenarios

### For Pending Work
👉 **Current tasks**: [PENDING_WORK.md](Tasks/PENDING_WORK.md)
- What needs to be done
- Why it's needed
- How to implement
- Test cases to verify

---

## File Organization

```
.claude/
├── INDEX.md (THIS FILE)
├── system/
│   ├── PROJECT_OVERVIEW.md          ← Architecture & structure
│   └── TECH_STACK.md                (Future: language versions, deps)
├── SOP/
│   ├── CLI_BEST_PRACTICES.md        ← Command patterns & rules
│   └── COMMON_MISTAKES.md           (Future: troubleshooting)
└── Tasks/
    ├── PENDING_WORK.md              ← What to do next
    ├── VGAE_KD_SETUP.md             (Future: VGAE-specific tasks)
    └── GAT_KD_SETUP.md              (Future: GAT-specific tasks)
```

---

## Key Principles to Remember

### ⭐ ALWAYS INCLUDE `--modality`
Every. Single. Time.
- Affects data loading
- Affects result paths
- Breaks without it
- **Default**: `automotive`

### ⭐ KD is Orthogonal to Mode
- KD ≠ training mode
- Can use with autoencoder, curriculum, normal
- CANNOT use with fusion (validation rejects)
- Must have teacher model path

### ⭐ Pipeline Rules
- Multi-value params must have matching lengths
- `--model`, `--training-strategy`, `--learning-type` must all have same count
- `--dataset`, `--modality`, `--distillation` are single values (same for all jobs)

### ⭐ Model Sizing
- **Teacher**: Larger models, for pretraining
- **Student**: Smaller models, for KD training
- When using `--distillation with-kd`, MUST use `--model-size student`
- CLI auto-enforces this

---

## Recent Changes & Fixes (Session Context)

### Frozen Config Pattern (NEW) ✅
- **What**: Config resolved once at job submission, saved as JSON
- **Why**: Eliminates CLI → SLURM → Config resolution chain issues
- **Files Added**:
  - `src/config/frozen_config.py` - Serialization/deserialization utilities
  - `--frozen-config` argument in train_with_hydra_zen.py
- **Usage**: `python train_with_hydra_zen.py --frozen-config /path/to/config.json`
- **See**: [MAINTENANCE.md](MAINTENANCE.md) for update tracking

### Investigation Completed ✅
- **Root Cause Found**: `format_training_args()` in job_manager.py was ignoring distillation flag
- **Impact**: Jobs submitted with `--distillation with-kd` were running without KD
- **Status**: FIXED - now passes `--use-kd` and `--teacher_path` to training script

### Fixes Applied ✅
1. Updated `format_training_args()` to extract and pass distillation flags
2. Added pipeline CLI validation for fusion+KD (rejects it)
3. Added `--teacher_path` parameter to pipeline
4. Auto-enforce `--model-size student` when with-kd detected

### Pending ⏳
1. Implement per-mode distillation configuration (comma-separated like --training-strategy)
2. Audit ALL CLI examples to include `--modality`
3. Test full KD pipeline end-to-end

---

## Common Workflows

### Starting New Work Session
1. Read [PROJECT_OVERVIEW.md](system/PROJECT_OVERVIEW.md) - 3 min
2. Check [PENDING_WORK.md](Tasks/PENDING_WORK.md) - What's next?
3. Reference [CLI_BEST_PRACTICES.md](SOP/CLI_BEST_PRACTICES.md) - For commands

### Submitting KD Pipeline
```bash
# Use this template (FROM CLI_BEST_PRACTICES.md - RULE 3):
./can-train pipeline \
  --model vgae,gat \
  --learning-type unsupervised,supervised \
  --training-strategy autoencoder,curriculum \
  --distillation with-kd \
  --dataset {DATASET} \
  --modality automotive \
  --model-size student \
  --teacher_path /path/to/teacher.pth \
  --submit
```

### Troubleshooting Job Failure
1. Check logs: `tail -f experimentruns/slurm_runs/{dataset}/*.err`
2. Look for "Knowledge Distillation: ENABLED" in logs
3. If missing, KD wasn't initialized (check --use-kd flag in script)
4. Reference [PENDING_WORK.md](Tasks/PENDING_WORK.md) - Known Issues section

---

## Related Codebase Locations

### Lightning Modules (Training)
- [src/training/lightning_modules.py](../src/training/lightning_modules.py)
  - VAELightningModule (VGAE with KD support)
  - GATLightningModule (GAT with KD support)

### Knowledge Distillation
- [src/training/knowledge_distillation.py](../src/training/knowledge_distillation.py)
  - KDHelper class (450+ lines)
  - Teacher loading, freezing, KD loss computation

### Configuration
- [src/config/hydra_zen_configs.py](../src/config/hydra_zen_configs.py)
  - KD config fields: use_knowledge_distillation, teacher_model_path, etc.
- [src/config/frozen_config.py](../src/config/frozen_config.py) **(NEW)**
  - Frozen Config Pattern: `save_frozen_config()`, `load_frozen_config()`
  - Serializes CANGraphConfig to JSON for reproducibility
- [config/batch_size_factors.json](../config/batch_size_factors.json)
  - Safety factors per dataset, with KD-specific (×0.75)

### CLI & Pipeline
- [src/cli/main.py](../src/cli/main.py)
  - _run_pipeline() function (line ~810)
  - CLI argument parser

- [src/cli/job_manager.py](../src/cli/job_manager.py)
  - format_training_args() (line ~142) - RECENTLY FIXED
  - SLURM script generation

---

## Documentation Writing Guidelines

When adding to this documentation:

1. **PROJECT_OVERVIEW.md**: Technical architecture (what & why)
2. **CLI_BEST_PRACTICES.md**: Usage patterns & rules (how to use)
3. **PENDING_WORK.md**: Implementation tasks (what to build)

Always include:
- ✅ Concrete examples
- ✅ Do's and Don'ts
- ✅ Why each rule matters
- ✅ Cross-references to other docs

---

## Version & Status

**Last Updated**: 2026-01-26 (Post-investigation)

**Documentation Coverage**:
- ✅ Project overview: Complete
- ✅ CLI best practices: Complete
- ✅ Pending work: Complete
- ⏳ Tech stack details: Planned
- ⏳ Troubleshooting guide: Planned

**Known Gaps**:
- VGAE-specific task checklist
- GAT-specific task checklist
- Teacher model training workflow

---

## How I (Claude) Use These Files

I will:
1. **Read PROJECT_OVERVIEW.md** before starting technical work
2. **Check PENDING_WORK.md** at session start to understand context
3. **Reference CLI_BEST_PRACTICES.md** when generating commands
4. **Include --modality** in EVERY CLI example
5. **Flag inconsistencies** if I notice missing modality or wrong patterns

This helps ensure:
- Consistent CLI usage (always --modality)
- Per-job KD awareness (when implemented)
- Correct pipeline configurations
- Clear task priorities
