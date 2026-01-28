# Snakemake Migration Summary

## What Was Created

I've created a complete Snakemake-based pipeline system to replace your custom `can-train` CLI. Here's what's included:

### Core Files

1. **`Snakefile`** (400 lines)
   - Main pipeline definition
   - Handles VGAE → GAT → DQN dependencies automatically
   - Supports teacher and student models
   - Includes evaluation rules
   - Automatic SLURM submission

2. **`config/snakemake_config.yaml`**
   - Dataset configuration
   - SLURM settings (account, partition, email)
   - Easy to customize per experiment

3. **`profiles/slurm/config.yaml`**
   - SLURM executor configuration
   - Resource defaults
   - Logging settings

4. **`profiles/slurm/slurm-status.py`**
   - Job status checker for SLURM
   - Integrates with Snakemake executor

5. **`envs/gnn-experiments.yaml`**
   - Conda environment specification
   - Ensures reproducibility

### Documentation

6. **`SNAKEMAKE_MIGRATION_PLAN.md`** (comprehensive guide)
   - Detailed migration strategy
   - Code changes needed
   - Testing procedures
   - Troubleshooting

7. **`SNAKEMAKE_MIGRATION_TODO.md`** (checklist)
   - Step-by-step tasks
   - Phase-by-phase breakdown
   - Success criteria

8. **`SNAKEMAKE_QUICKSTART.md`** (user guide)
   - Common commands
   - Examples
   - Quick reference

## Key Benefits

### Before (Current System)
```bash
# Complex setup with manual dependency tracking
TEACHER_DQN_JOBS=(
  "43982250"  # hcrl_sa
  "43982254"  # hcrl_ch
)

./can-train pipeline --dependency "${teacher_job}" --submit
```

### After (Snakemake)
```bash
# Simple, declarative
snakemake --profile profiles/slurm --jobs 20
```

### Advantages

1. **10x Simpler**: ~400 lines vs 2,000+ lines of CLI code
2. **Automatic Dependencies**: File-based, no manual job ID tracking
3. **Resume from Failures**: Automatic detection and rerun
4. **Reproducible**: Complete provenance and DAG visualization
5. **Standard Tool**: Community support, extensive documentation

## What You Need to Do

### Critical Change (Required)

**Modify `train_with_hydra_zen.py`** to support simple CLI arguments from Snakemake:

```python
# Add to train_with_hydra_zen.py
def create_snakemake_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-size', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--modality', required=True)
    parser.add_argument('--training', required=True)
    parser.add_argument('--learning-type', required=True)
    parser.add_argument('--output-dir', required=True)
    # ... teacher/fusion paths ...
    return parser

def main():
    if '--frozen-config' in sys.argv:
        # Legacy: use frozen config from can-train
        run_with_frozen_config()
    else:
        # New: simple args from Snakemake
        parser = create_snakemake_parser()
        args = parser.parse_args()
        config = build_config_from_simple_args(args)
        train(config)
```

This lets the training script work with both:
- **Old system**: `can-train` with frozen configs
- **New system**: Snakemake with simple arguments

## Quick Start

### 1. Install Snakemake
```bash
conda install -c conda-forge snakemake
```

### 2. Setup
```bash
# Make status script executable
chmod +x profiles/slurm/slurm-status.py

# Update your email in config
nano config/snakemake_config.yaml
```

### 3. Dry Run
```bash
snakemake --profile profiles/slurm -n
```

### 4. Visualize Pipeline
```bash
snakemake --dag | dot -Tpdf > pipeline_dag.pdf
```

### 5. Run Single Test
```bash
snakemake --profile profiles/slurm \
    experimentruns/automotive/hcrl_sa/vgae/teacher/no_distillation/autoencoder/models/vgae_teacher_no_distillation_best.pth \
    --jobs 1
```

### 6. Run Full Pipeline
```bash
# All teachers
snakemake --profile profiles/slurm all_teachers --jobs 20

# Everything (teachers + students)
snakemake --profile profiles/slurm --jobs 20
```

## How It Works

### Pipeline Structure

```
VGAE (autoencoder)
    ↓
GAT (curriculum)
    ↓ (depends on VGAE + GAT)
DQN (fusion)
    ↓
Evaluation
```

### For Each Dataset

```
hcrl_sa:
  Teacher: VGAE → GAT → DQN → Eval
  Student: (waits for teacher) → VGAE-KD → GAT-KD → DQN-KD → Eval

hcrl_ch:
  Teacher: VGAE → GAT → DQN → Eval
  Student: (waits for teacher) → VGAE-KD → GAT-KD → DQN-KD → Eval

... (set_01, set_02, set_03, set_04)
```

### Automatic Features

- **Dependency Resolution**: Snakemake knows GAT needs VGAE, DQN needs both
- **Parallel Execution**: Runs independent jobs (different datasets) in parallel
- **Failure Recovery**: If GAT fails, fix it and rerun - VGAE won't rerun
- **Provenance**: DAG shows exact execution order and dependencies

## Migration Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Setup | 1-2 hours | Install Snakemake, configure profile |
| **Code Changes** | **4-8 hours** | **Modify training script (critical)** |
| Testing | 2-4 hours | Test single jobs, dependencies |
| Validation | 1-2 days | Compare with old pipeline |
| Full Migration | 1-2 weeks | Documentation, team training |

## File Organization

```
KD-GAT/
├── Snakefile                          # ← Main pipeline
├── config/
│   └── snakemake_config.yaml         # ← Configuration
├── profiles/
│   └── slurm/
│       ├── config.yaml               # ← SLURM settings
│       └── slurm-status.py           # ← Job status checker
├── envs/
│   └── gnn-experiments.yaml          # ← Conda env
├── SNAKEMAKE_MIGRATION_PLAN.md       # ← Detailed guide
├── SNAKEMAKE_MIGRATION_TODO.md       # ← Checklist
├── SNAKEMAKE_QUICKSTART.md           # ← User guide
└── SNAKEMAKE_SUMMARY.md              # ← This file
```

## Common Commands

```bash
# Dry run (see what will execute)
snakemake --profile profiles/slurm -n

# Visualize pipeline
snakemake --dag | dot -Tpdf > dag.pdf

# Run specific target
snakemake --profile profiles/slurm results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json --jobs 3

# Resume from failures
snakemake --profile profiles/slurm --rerun-incomplete --jobs 20

# Clean checkpoints (keep best models)
snakemake clean_checkpoints
```

## Next Steps

1. **Read** `SNAKEMAKE_MIGRATION_PLAN.md` for detailed strategy
2. **Follow** `SNAKEMAKE_MIGRATION_TODO.md` checklist
3. **Modify** `train_with_hydra_zen.py` (most important!)
4. **Test** with single dataset before full run
5. **Validate** results match old pipeline
6. **Migrate** fully once confident

## Getting Help

- **Quick reference**: `SNAKEMAKE_QUICKSTART.md`
- **Detailed guide**: `SNAKEMAKE_MIGRATION_PLAN.md`
- **Task checklist**: `SNAKEMAKE_MIGRATION_TODO.md`
- **Snakemake docs**: https://snakemake.readthedocs.io/
- **Troubleshooting**: See migration plan Section 8

## Questions?

Common questions answered in the migration plan:
- What if something fails? → Automatic recovery
- How do I add a new dataset? → Edit `config/snakemake_config.yaml`
- How do I customize resources? → Edit `profiles/slurm/config.yaml`
- Can I keep using can-train? → Yes, during transition period
- What about reproducibility? → Better than before (DAG tracking)

---

**Bottom Line**: This migration will make your pipeline **simpler**, **more robust**, and **easier to maintain**. The critical change is modifying `train_with_hydra_zen.py` to support simple CLI arguments. Everything else is infrastructure that's already set up.

Good luck with the migration! 🚀
