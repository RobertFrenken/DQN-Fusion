# Snakemake Migration Plan: KD-GAT Pipeline

This document outlines the step-by-step migration from the custom `can-train` CLI system to Snakemake for managing the KD-GAT training pipeline.

## Table of Contents
1. [Overview](#overview)
2. [Benefits of Migration](#benefits-of-migration)
3. [Files Created](#files-created)
4. [Required Code Changes](#required-code-changes)
5. [Migration Steps](#migration-steps)
6. [Testing & Validation](#testing--validation)
7. [Deprecation Strategy](#deprecation-strategy)

---

## Overview

**Current System:**
- Custom CLI (`can-train`) with bucket-based configuration
- Manual SLURM script generation (`job_manager.py`)
- Manual dependency tracking via job ID arrays
- Shell scripts for pipeline orchestration (`student_kd.sh`, `run_pipeline.sh`)

**New System:**
- Declarative Snakemake pipeline (`Snakefile`)
- Automatic dependency resolution based on file I/O
- Native SLURM integration via Snakemake executor
- Automatic resume from failures
- Built-in DAG visualization

---

## Benefits of Migration

### 1. **Simplicity**
- **Before:** 2,000+ lines across `can-train`, `main.py`, `job_manager.py`, `config_builder.py`
- **After:** ~400 line Snakefile

### 2. **Automatic Dependency Management**
- **Before:** Manual job ID tracking in arrays
  ```bash
  TEACHER_DQN_JOBS=(
    "43982250"  # hcrl_sa
    "43982254"  # hcrl_ch
  )
  ```
- **After:** File-based dependencies (automatic)
  ```python
  input:
      vgae_model="models/{dataset}/vgae.pth",
      gat_model="models/{dataset}/gat.pth"
  ```

### 3. **Reproducibility**
- Complete provenance tracking
- DAG visualization shows exact execution order
- All dependencies explicit in code

### 4. **Failure Recovery**
- Automatic detection of incomplete jobs
- Resume from any point in pipeline
- No need to manually track which jobs succeeded

### 5. **Reduced Maintenance**
- No custom SLURM script templates to maintain
- No job manager code to debug
- Standard tool with community support

---

## Files Created

### Core Snakemake Files
1. **`Snakefile`** - Main pipeline definition
   - Defines all training and evaluation rules
   - Handles VGAE → GAT → DQN dependencies
   - Supports both teacher and student models

2. **`config/snakemake_config.yaml`** - Pipeline configuration
   - Lists datasets, modalities, model sizes
   - SLURM account/partition settings
   - Can be version-controlled per experiment

3. **`profiles/slurm/config.yaml`** - SLURM executor profile
   - Configures Snakemake to use SLURM
   - Sets default resources (CPUs, memory, GPUs)
   - Logging and latency settings

4. **`profiles/slurm/slurm-status.py`** - Job status checker
   - Queries SLURM for job status
   - Used by Snakemake to monitor jobs

5. **`envs/gnn-experiments.yaml`** - Conda environment
   - Ensures consistent dependencies
   - Used by all training rules

---

## Required Code Changes

### Phase 1: Training Script Modifications (CRITICAL)

#### 1.1. Modify `train_with_hydra_zen.py`

**Current Issue:**
The training script expects the frozen config pattern or complex CLI arguments from `can-train`.

**Required Changes:**

```python
# Location: train_with_hydra_zen.py

# Add a simpler CLI interface that Snakemake can call
import argparse

def create_snakemake_parser():
    """Simple parser for Snakemake invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['vgae', 'gat', 'dqn'])
    parser.add_argument('--model-size', required=True, choices=['teacher', 'student'])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--modality', required=True)
    parser.add_argument('--training', required=True,
                       choices=['autoencoder', 'curriculum', 'fusion'])
    parser.add_argument('--learning-type', required=True)
    parser.add_argument('--teacher-path', default=None)
    parser.add_argument('--vgae-path', default=None)
    parser.add_argument('--gat-path', default=None)
    parser.add_argument('--output-dir', required=True)
    return parser

def main():
    # Detect if called from Snakemake (simpler CLI)
    # vs. from can-train CLI (frozen config)

    if '--frozen-config' in sys.argv:
        # Legacy path: use frozen config
        use_frozen_config_path()
    else:
        # New path: Snakemake direct invocation
        parser = create_snakemake_parser()
        args = parser.parse_args()

        # Build config from simple arguments
        config = build_config_from_args(args)

        # Run training
        train(config)
```

**Files to Modify:**
- `train_with_hydra_zen.py` - Add Snakemake CLI interface
- Ensure model checkpoints are saved to predictable paths (e.g., `{output_dir}/models/{model}_best.pth`)

#### 1.2. Standardize Model Output Paths

**Current Issue:**
Model checkpoints may be saved with unpredictable names (e.g., timestamp-based).

**Required Changes:**

```python
# In your training code (e.g., src/training/trainer.py or similar)

def save_best_model(model, output_dir, model_name, model_size, distillation):
    """Save model with standardized naming."""
    model_path = Path(output_dir) / "models" / f"{model_name}_{model_size}_{distillation}_best.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics
    }, model_path)

    return model_path
```

**Why:**
Snakemake needs to know the exact output file paths to track dependencies.

### Phase 2: Evaluation Script Modifications

#### 2.1. Ensure Evaluation Outputs are Deterministic

**Files to Check:**
- `src/evaluation/evaluation.py`

**Required Changes:**
Ensure evaluation outputs are written to paths specified by `--json-output` and `--csv-output` arguments (already done based on earlier file reading).

### Phase 3: Directory Structure Adjustments (Optional)

**Current:**
```
experimentruns/
  automotive/
    hcrl_sa/
      unsupervised/vgae/teacher/no_distillation/autoencoder/
        models/
        checkpoints/
        logs/
```

**Snakemake-friendly (optional flattening):**
```
experimentruns/
  automotive/hcrl_sa/vgae/teacher/no_distillation/
    models/
    checkpoints/
    logs/
```

**Note:** This is optional. The current structure works fine with the helper functions in the Snakefile.

---

## Migration Steps

### Step 1: Setup Snakemake Environment

```bash
# Install Snakemake in the base conda environment
conda activate base
conda install -c conda-forge snakemake

# Or create a dedicated Snakemake environment
conda create -n snakemake -c conda-forge snakemake
conda activate snakemake
```

### Step 2: Make SLURM Status Script Executable

```bash
chmod +x profiles/slurm/slurm-status.py
```

### Step 3: Modify Training Script

```bash
# Back up original
cp train_with_hydra_zen.py train_with_hydra_zen.py.backup

# Add the Snakemake CLI interface (see Phase 1.1 above)
# This is the CRITICAL change
```

**TODO:** Edit `train_with_hydra_zen.py` to add simpler CLI parser

### Step 4: Test with Dry Run

```bash
# Activate Snakemake environment
conda activate snakemake

# Dry run: see what will be executed
snakemake --profile profiles/slurm -n

# Dry run with DAG visualization
snakemake --dag | dot -Tpdf > pipeline_dag.pdf
```

### Step 5: Test with Single Dataset

```bash
# Run just the teacher pipeline for one dataset
snakemake --profile profiles/slurm \
    results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json \
    --jobs 3
```

### Step 6: Run Full Pipeline

```bash
# Run all teacher models across all datasets
snakemake --profile profiles/slurm all_teachers --jobs 20

# Run everything (teachers + students)
snakemake --profile profiles/slurm --jobs 20
```

### Step 7: Handle Failures

```bash
# If jobs fail, fix the issue, then resume:
snakemake --profile profiles/slurm --jobs 20 --rerun-incomplete

# Force rerun of specific rule:
snakemake --profile profiles/slurm --forcerun train_gat --jobs 20
```

---

## Testing & Validation

### Test 1: Single Model Training

**Goal:** Verify VGAE training works end-to-end

```bash
snakemake --profile profiles/slurm \
    experimentruns/automotive/hcrl_sa/vgae/teacher/no_distillation/autoencoder/models/vgae_teacher_no_distillation_best.pth \
    --jobs 1
```

**Validation:**
- Check SLURM job was submitted: `squeue -u $USER`
- Check model file was created
- Check logs in `experimentruns/.../logs/`

### Test 2: Pipeline Dependencies

**Goal:** Verify GAT waits for VGAE

```bash
# Delete GAT model if it exists
rm -f experimentruns/automotive/hcrl_sa/gat/teacher/no_distillation/curriculum/models/gat_teacher_no_distillation_best.pth

# Run GAT rule
snakemake --profile profiles/slurm \
    experimentruns/automotive/hcrl_sa/gat/teacher/no_distillation/curriculum/models/gat_teacher_no_distillation_best.pth \
    --jobs 1
```

**Validation:**
- Snakemake should first ensure VGAE exists
- Check dependency chain in output

### Test 3: Student Model Distillation

**Goal:** Verify student waits for teacher

```bash
# Run student VGAE
snakemake --profile profiles/slurm \
    experimentruns/automotive/hcrl_sa/vgae/student/with_kd/autoencoder/models/vgae_student_with_kd_best.pth \
    --jobs 1
```

**Validation:**
- Teacher model should be created first (if it doesn't exist)
- Student training should receive teacher path

### Test 4: Evaluation

**Goal:** Verify evaluation runs after training

```bash
snakemake --profile profiles/slurm \
    results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json \
    --jobs 3
```

**Validation:**
- VGAE, GAT, and DQN should all be trained first
- Evaluation JSON should be created
- Metrics should be correct

### Test 5: Failure Recovery

**Goal:** Verify Snakemake can resume from failures

```bash
# Manually cancel a running job
scancel <job_id>

# Rerun pipeline
snakemake --profile profiles/slurm --jobs 20 --rerun-incomplete
```

**Validation:**
- Snakemake should detect incomplete job
- Only failed job should be rerun (not all jobs)

---

## Deprecation Strategy

### Phase 1: Parallel Operation (Recommended)

Run both systems in parallel for validation:
- Keep `can-train` CLI functional
- Run critical experiments with Snakemake
- Compare outputs to ensure consistency

**Timeline:** 2-4 weeks

### Phase 2: Snakemake as Primary

Once validated:
- Update documentation to use Snakemake
- Keep `can-train` for backward compatibility
- Add deprecation warnings

**Timeline:** 1 month

### Phase 3: Full Migration

After confidence is established:
- Archive old shell scripts (`student_kd.sh`, `run_pipeline.sh`)
- Remove `can-train` CLI (optional - can keep for quick ad-hoc runs)
- Update all docs

**Timeline:** 2-3 months

---

## Migration Checklist

### Pre-Migration
- [ ] Install Snakemake: `conda install -c conda-forge snakemake`
- [ ] Create backup branch: `git checkout -b pre-snakemake-backup`
- [ ] Review current pipeline structure
- [ ] Document current job submission patterns

### Core Changes
- [ ] Modify `train_with_hydra_zen.py` to support simple CLI (CRITICAL)
- [ ] Ensure model outputs use predictable paths
- [ ] Test training script with new CLI arguments
- [ ] Verify evaluation scripts work with Snakemake

### Snakemake Setup
- [ ] Make `slurm-status.py` executable
- [ ] Update `config/snakemake_config.yaml` with correct email/account
- [ ] Test SLURM profile with dry run
- [ ] Verify conda environment creation

### Testing
- [ ] Test single VGAE training
- [ ] Test VGAE → GAT dependency
- [ ] Test GAT → DQN fusion with multiple inputs
- [ ] Test student model with KD from teacher
- [ ] Test evaluation after training
- [ ] Test failure recovery

### Documentation
- [ ] Create Snakemake usage guide
- [ ] Update README with Snakemake commands
- [ ] Document troubleshooting procedures
- [ ] Add DAG visualization examples

### Cleanup (After Validation)
- [ ] Archive old shell scripts to `legacy/` directory
- [ ] Add deprecation warnings to `can-train`
- [ ] Remove unused `job_manager.py` code
- [ ] Clean up frozen config system (if no longer needed)

---

## Common Issues & Solutions

### Issue 1: Model Path Mismatch

**Symptom:**
```
MissingInputException: Missing input files for rule train_gat:
    experimentruns/.../vgae_teacher_no_distillation_best.pth
```

**Solution:**
- Check `get_model_path()` function in Snakefile
- Verify training script saves to correct path
- Check for typos in model naming

### Issue 2: SLURM Jobs Not Submitting

**Symptom:**
```
Error: sbatch command not found
```

**Solution:**
- Ensure you're on a SLURM cluster
- Check SLURM profile configuration
- Verify `slurm-status.py` is executable

### Issue 3: Conda Environment Issues

**Symptom:**
```
CreateCondaEnvironmentException
```

**Solution:**
```bash
# Manually create environment first
conda env create -f envs/gnn-experiments.yaml

# Or disable conda integration temporarily
snakemake --profile profiles/slurm --no-use-conda
```

### Issue 4: Student Models Start Before Teachers

**Symptom:**
Student training jobs fail because teacher models don't exist

**Solution:**
- Check `get_student_dependencies()` function
- Ensure teacher model path is added to `input:` for student rules
- Verify Snakefile has correct dependency logic

---

## Rollback Plan

If critical issues arise:

1. **Keep can-train functional:**
   ```bash
   git checkout main -- can-train src/cli/
   ```

2. **Use old scripts:**
   ```bash
   ./run_pipeline.sh  # Old method still works
   ```

3. **Report issues:**
   - Document the failure mode
   - Check Snakemake logs
   - Consider filing issue with Snakemake team

---

## Resources

### Snakemake Documentation
- [Official Tutorial](https://snakemake.readthedocs.io/en/stable/tutorial/tutorial.html)
- [SLURM Executor](https://snakemake.readthedocs.io/en/stable/executing/cluster.html)
- [Best Practices](https://snakemake.readthedocs.io/en/stable/snakefiles/best_practices.html)

### OSC-Specific
- [OSC Batch Computing](https://www.osc.edu/resources/technical_support/supercomputers/owens/batch_computing)
- [SLURM Commands](https://slurm.schedmd.com/quickstart.html)

### Community
- [Snakemake Workflows Catalog](https://snakemake.github.io/snakemake-workflow-catalog/)
- [GitHub Discussions](https://github.com/snakemake/snakemake/discussions)

---

## Next Steps

After reading this plan:

1. **Review** the Snakefile and understand the rule structure
2. **Prioritize** modifying `train_with_hydra_zen.py` (most critical change)
3. **Test** with a small dry run
4. **Iterate** based on failures
5. **Validate** results match old pipeline
6. **Document** any deviations or discoveries

Good luck with the migration!
