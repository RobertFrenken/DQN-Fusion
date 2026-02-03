---
name: check-status
description: Check status of running experiments, SLURM jobs, and pipeline progress
disable-model-invocation: true
user-invocable: true
argument-hint: [dataset]
allowed-tools: Bash, Read, Glob
---

Check the status of experiments and running jobs.

## Arguments

- `$ARGUMENTS` - Optional dataset name to filter (e.g., `hcrl_sa`)

## Execution Steps

1. **Check SLURM job queue**
   ```bash
   squeue -u $USER --format="%.10i %.20j %.8T %.10M %.6D %.15R" 2>&1
   ```

2. **Check recent experiment outputs** (last modified)
   ```bash
   find experimentruns/ -name "best_model.pt" -mmin -60 -exec ls -la {} \; 2>/dev/null | head -20
   ```

3. **Check for recent errors**
   ```bash
   find experimentruns/ -name "slurm.err" -mmin -30 -exec sh -c 'echo "=== {} ===" && tail -5 {}' \; 2>/dev/null
   ```

4. **Check MLflow runs** (recent)
   ```bash
   ls -lt mlruns/*/meta.yaml 2>/dev/null | head -10
   ```

5. **If dataset specified**, show detailed status for that dataset:
   ```bash
   ls -la experimentruns/$ARGUMENTS/*/best_model.pt 2>/dev/null
   tail -20 experimentruns/$ARGUMENTS/*/slurm.err 2>/dev/null
   ```

## Output Summary

Provide a concise status report:

| Stage | Status | Last Updated |
|-------|--------|--------------|
| autoencoder | complete/running/pending | timestamp |
| curriculum | complete/running/pending | timestamp |
| fusion | complete/running/pending | timestamp |

## Common Commands

```bash
# Watch job queue
watch -n 5 'squeue -u $USER'

# Follow specific experiment logs
tail -f experimentruns/hcrl_sa/student_fusion/slurm.err

# Check Snakemake progress
cat /tmp/claude-*/tasks/*.output | tail -30
```
