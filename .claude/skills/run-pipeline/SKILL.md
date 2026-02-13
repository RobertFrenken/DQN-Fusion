---
name: run-pipeline
description: Run Snakemake pipeline for a dataset and model configuration
---

Run the KD-GAT Snakemake pipeline.

## Arguments

`$ARGUMENTS` should contain: `<dataset> [target]`

- **dataset** (required): `hcrl_sa`, `hcrl_ch`, `set_01`, `set_02`, `set_03`, `set_04`
- **target** (optional): `students_nokd`, `teachers`, `students`, or a specific run like `teacher_fusion`

Parse the dataset and target from `$ARGUMENTS`. If only one word is provided, it is the dataset and no target filter is applied.

## Usage Examples

```
/run-pipeline hcrl_sa students_nokd     # Run student without KD for hcrl_sa
/run-pipeline hcrl_ch teachers          # Run teacher pipeline for hcrl_ch
/run-pipeline set_01                    # Run all targets for set_01
```

## Execution Steps

1. **Parse arguments** from `$ARGUMENTS` into dataset and optional target.

2. **Verify dataset exists**
   ```bash
   ls data/automotive/<dataset>/
   ```

3. **Dry run first** to see what will be executed. If a target is given, use it as a Snakemake rule name or build the output path `experimentruns/<dataset>/<target>/best_model.pt`. If no target, run the full DAG for that dataset.
   ```bash
   snakemake -s pipeline/Snakefile --config "datasets=[\"<dataset>\"]" -n 2>&1 | head -50
   ```

4. **Submit to SLURM** if dry run looks correct
   ```bash
   snakemake -s pipeline/Snakefile --config "datasets=[\"<dataset>\"]" --profile profiles/slurm
   ```

5. **Report the submitted job IDs** and show how to monitor with `squeue -u $USER`.

## Common Targets

| Target | Description | Output |
|--------|-------------|--------|
| `teacher_fusion` | Full teacher pipeline | 3 stages |
| `student_fusion` | Full student no-KD pipeline | 3 stages |
| `student_fusion_kd` | Full student with KD pipeline | 3 stages (needs teacher) |
| `teachers` | All datasets, teacher | Snakemake rule target |
| `students_nokd` | All datasets, student no-KD | Snakemake rule target |
| `students` | All datasets, student with KD | Snakemake rule target |

## Notes

- Pipeline runs on SLURM with GPU resources (V100, 128GB RAM)
- SLURM logs: `experimentruns/{ds}/{run}/slurm.{out,err}`
- MLflow tracking is automatic
- Write-through DB: runs are recorded in `data/project.db` before/after each stage
- Snakefile uses `sys.executable` for Python path (override with `KD_GAT_PYTHON` env var)
- Always do a dry run (`-n`) before submitting
