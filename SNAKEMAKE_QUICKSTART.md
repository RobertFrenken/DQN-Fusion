# Snakemake Quick Start Guide

This guide shows you how to use Snakemake to run your KD-GAT training pipelines.

## Installation

```bash
# Option 1: Install in base conda environment
conda activate base
conda install -c conda-forge snakemake

# Option 2: Create dedicated Snakemake environment
conda create -n snakemake -c conda-forge snakemake
conda activate snakemake
```

## Basic Usage

### 1. Dry Run (See What Will Execute)

```bash
# See all jobs that will be executed
snakemake --profile profiles/slurm -n

# See detailed reasons for execution
snakemake --profile profiles/slurm -n -r
```

### 2. Visualize Pipeline

```bash
# Generate DAG (Directed Acyclic Graph) visualization
snakemake --dag | dot -Tpdf > pipeline_dag.pdf

# Generate rule graph (simplified)
snakemake --rulegraph | dot -Tpdf > rulegraph.pdf
```

### 3. Run Specific Targets

```bash
# Train only teacher models
snakemake --profile profiles/slurm all_teachers --jobs 20

# Train only student models (will train teachers first if needed)
snakemake --profile profiles/slurm all_students --jobs 20

# Train specific dataset
snakemake --profile profiles/slurm \
    results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json \
    --jobs 3
```

### 4. Run Full Pipeline

```bash
# Run everything (all teachers and students across all datasets)
snakemake --profile profiles/slurm --jobs 20

# With specific number of parallel jobs
snakemake --profile profiles/slurm --jobs 10

# Continue on failures (run independent jobs even if some fail)
snakemake --profile profiles/slurm --jobs 20 --keep-going
```

### 5. Resume from Failures

```bash
# Automatically rerun incomplete jobs
snakemake --profile profiles/slurm --jobs 20 --rerun-incomplete

# Force rerun specific rule
snakemake --profile profiles/slurm --forcerun train_gat --jobs 20

# Force rerun specific file and all downstream dependencies
snakemake --profile profiles/slurm \
    --forcetargets results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json \
    --jobs 3
```

## Common Commands

### Check What Will Run

```bash
# List all output files that will be created
snakemake --profile profiles/slurm --list

# Show which rules will be executed
snakemake --profile profiles/slurm --list-rules
```

### Monitor Progress

```bash
# Watch SLURM queue
watch -n 10 'squeue -u $USER'

# Check Snakemake logs
tail -f .snakemake/log/*.log

# Check specific job logs
tail -f experimentruns/automotive/hcrl_sa/vgae/teacher/no_distillation/autoencoder/logs/training.log
```

### Cleanup

```bash
# Clean checkpoints (keep best models)
snakemake clean_checkpoints

# Clean logs
snakemake clean_logs

# DANGER: Clean everything
snakemake clean_all
```

## Configuration

### Edit Pipeline Configuration

```bash
# Edit datasets, modalities, SLURM settings
nano config/snakemake_config.yaml
```

### Customize SLURM Resources

```bash
# Edit SLURM profile
nano profiles/slurm/config.yaml
```

### Override Resources for Specific Run

```bash
# Use different partition
snakemake --profile profiles/slurm --jobs 20 \
    --set-resources train_vgae:partition=longserial

# Use more memory
snakemake --profile profiles/slurm --jobs 20 \
    --set-resources train_dqn:mem_mb=128000
```

## Troubleshooting

### Jobs Not Submitting

```bash
# Test SLURM status script
python profiles/slurm/slurm-status.py <job_id>

# Check SLURM account
sacctmgr show assoc where user=$USER

# Verify partition access
sinfo -o "%20P %5a %10l %6D"
```

### Missing Input Files

```bash
# Check what Snakemake thinks exists
snakemake --profile profiles/slurm --summary

# See why a file will be created
snakemake --profile profiles/slurm -n -r <target_file>
```

### Locked Directory

```bash
# If Snakemake was interrupted, unlock
snakemake --unlock
```

### Detailed Debugging

```bash
# Print shell commands
snakemake --profile profiles/slurm --printshellcmds --jobs 1

# Verbose output
snakemake --profile profiles/slurm --verbose --jobs 1

# Debug mode (very verbose)
snakemake --profile profiles/slurm --debug --jobs 1
```

## Examples

### Example 1: Train Single Model

```bash
# Train VGAE teacher on hcrl_sa
snakemake --profile profiles/slurm \
    experimentruns/automotive/hcrl_sa/vgae/teacher/no_distillation/autoencoder/models/vgae_teacher_no_distillation_best.pth \
    --jobs 1
```

### Example 2: Train Pipeline for One Dataset

```bash
# Train VGAE → GAT → DQN for hcrl_ch
snakemake --profile profiles/slurm \
    results/automotive/hcrl_ch/teacher/no_distillation/evaluation/dqn_eval.json \
    --jobs 3
```

### Example 3: Run Only Evaluation

```bash
# Assume models are trained, just run evaluation
snakemake --profile profiles/slurm \
    --forcerun evaluate_model \
    results/automotive/hcrl_sa/teacher/no_distillation/evaluation/vgae_eval.json
```

### Example 4: Train Students After Teachers

```bash
# First, ensure all teachers are trained
snakemake --profile profiles/slurm all_teachers --jobs 20

# Then train students
snakemake --profile profiles/slurm all_students --jobs 20
```

### Example 5: Generate Summary Report

```bash
# Run all evaluations and generate report
snakemake --profile profiles/slurm generate_report --jobs 20
```

## Tips & Tricks

### 1. Use Screen/Tmux for Long Runs

```bash
# Start screen session
screen -S snakemake

# Run Snakemake
snakemake --profile profiles/slurm --jobs 20

# Detach: Ctrl+A, then D
# Reattach: screen -r snakemake
```

### 2. Check Estimated Runtime

```bash
# Dry run shows what will execute
snakemake --profile profiles/slurm -n
# Count jobs × average time per job
```

### 3. Prioritize Specific Datasets

```bash
# Run hcrl_sa first, then others
snakemake --profile profiles/slurm \
    results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json \
    --jobs 3

# Then run rest
snakemake --profile profiles/slurm --jobs 20
```

### 4. Resource Profiling

```bash
# Generate benchmarks
snakemake --profile profiles/slurm --jobs 20 --benchmark-extended
```

### 5. Archive Results

```bash
# After pipeline completes
snakemake --profile profiles/slurm --archive results_archive.tar.gz
```

## Quick Reference

| Task | Command |
|------|---------|
| Dry run | `snakemake --profile profiles/slurm -n` |
| Run all | `snakemake --profile profiles/slurm --jobs 20` |
| Resume | `snakemake --profile profiles/slurm --rerun-incomplete --jobs 20` |
| Visualize | `snakemake --dag \| dot -Tpdf > dag.pdf` |
| Clean up | `snakemake clean_checkpoints` |
| Force rerun | `snakemake --forcerun <rule> --jobs 20` |
| Unlock | `snakemake --unlock` |
| Debug | `snakemake --debug --verbose` |

## Getting Help

```bash
# Snakemake help
snakemake --help

# Rule-specific help
snakemake --list-rules
snakemake --summary
```

For more information, see [SNAKEMAKE_MIGRATION_PLAN.md](SNAKEMAKE_MIGRATION_PLAN.md).
