---
name: check-status
description: Check status of running experiments, SLURM jobs, and pipeline progress
---

Check the status of experiments and running jobs.

## Arguments

`$ARGUMENTS` - Optional dataset name to filter (e.g., `hcrl_sa`). If empty, check all datasets.

## Execution Steps

1. **Check SLURM job queue**
   ```bash
   squeue -u $USER --format="%.10i %.20j %.8T %.10M %.6D %.15R" 2>&1
   ```

2. **Check experiment checkpoints** for all datasets (or filtered dataset)
   ```bash
   # List all completed stages
   for ds in hcrl_ch hcrl_sa set_01 set_02 set_03 set_04; do
     echo "=== $ds ==="
     ls -lh experimentruns/$ds/*/best_model.pt 2>/dev/null || echo "  (no checkpoints)"
   done
   ```

3. **Check for recent SLURM errors** in experiment output directories
   ```bash
   ls -lt experimentruns/*/*/slurm.err 2>/dev/null | head -10
   ```

4. **Check project DB for run status** (write-through records from cli.py)
   ```bash
   python -m pipeline.db query "SELECT run_id, stage, status, started_at FROM runs ORDER BY started_at DESC LIMIT 10"
   ```

5. **If dataset specified via `$ARGUMENTS`**, show detailed status:
   ```bash
   ls -la experimentruns/$ARGUMENTS/*/best_model.pt 2>/dev/null
   ls -la experimentruns/$ARGUMENTS/*/config.json 2>/dev/null
   ls -la experimentruns/$ARGUMENTS/*/metrics.json 2>/dev/null
   python -m pipeline.db query "SELECT run_id, stage, status FROM runs WHERE dataset='$ARGUMENTS'"
   ```

## Output Summary

Provide a concise status report:

| Dataset | Stage | Status | Last Updated |
|---------|-------|--------|--------------|
| hcrl_sa | teacher_autoencoder | complete/missing | timestamp |
| hcrl_sa | teacher_curriculum | complete/missing | timestamp |
| hcrl_sa | teacher_fusion | complete/missing | timestamp |
| ... | ... | ... | ... |

## Useful Follow-up Commands

```bash
# Watch job queue
watch -n 5 'squeue -u $USER'

# Follow specific SLURM log
tail -f experimentruns/<dataset>/<run>/slurm.err

# Check Snakemake DAG status
snakemake -s pipeline/Snakefile --summary
```
