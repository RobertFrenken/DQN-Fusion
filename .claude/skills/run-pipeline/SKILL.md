---
name: run-pipeline
description: Run Snakemake pipeline for a dataset and model configuration
disable-model-invocation: true
user-invocable: true
argument-hint: [dataset] [target]
allowed-tools: Bash, Read
---

Run the KD-GAT Snakemake pipeline.

## Arguments

- `$0` - Dataset name: `hcrl_sa`, `hcrl_ch`, `set_01`, `set_02`, `set_03`, `set_04`
- `$1` - Target (optional): `students_nokd`, `teachers`, `students`, or specific path

## Usage Examples

```
/run-pipeline hcrl_sa students_nokd     # Run student without KD for hcrl_sa
/run-pipeline hcrl_ch teachers          # Run teacher pipeline for hcrl_ch
/run-pipeline set_01                    # Run all targets for set_01
```

## Execution Steps

1. **Verify dataset exists**
   ```bash
   ls -la data/automotive/$0/
   ```

2. **Dry run first** to see what will be executed
   ```bash
   snakemake -s pipeline/Snakefile experimentruns/$0/$1/best_model.pt --forceall -n 2>&1 | head -50
   ```

3. **Submit to SLURM** if dry run looks correct
   ```bash
   mkdir -p slurm_logs
   snakemake -s pipeline/Snakefile experimentruns/$0/$1/best_model.pt --profile profiles/slurm --forceall
   ```

4. **Report the submitted job IDs** so the user can monitor

## Common Targets

| Target | Description | Output |
|--------|-------------|--------|
| `student_fusion` | Full student no-KD pipeline | 3 stages |
| `student_fusion_kd` | Full student with KD pipeline | 3 stages (needs teacher) |
| `teacher_fusion` | Full teacher pipeline | 3 stages |
| `students_nokd` | All datasets, student no-KD | Uses rule target |
| `teachers` | All datasets, teacher | Uses rule target |

## Notes

- Pipeline runs on SLURM with GPU resources (V100, 128GB RAM)
- Each stage takes 5-30 minutes depending on dataset size
- Logs are written to `experimentruns/{dataset}/{run}/slurm.{out,err}`
- MLflow tracking is automatic
