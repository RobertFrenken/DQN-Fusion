---
name: run-pipeline
description: Run Snakemake pipeline for a dataset and model configuration
---

Run the KD-GAT Snakemake pipeline.

## Arguments

`$ARGUMENTS` should contain: `<dataset> [target]`

- **dataset** (required): `hcrl_sa`, `hcrl_ch`, `set_01`, `set_02`, `set_03`, `set_04`
- **target** (optional): `small_nokd`, `large`, `small_kd`, or a specific rule like `dqn_large`

Parse the dataset and target from `$ARGUMENTS`. If only one word is provided, it is the dataset and no target filter is applied.

## Usage Examples

```
/run-pipeline hcrl_sa small_nokd     # Run small without KD for hcrl_sa
/run-pipeline hcrl_ch large          # Run large pipeline for hcrl_ch
/run-pipeline set_01                 # Run all targets for set_01
```

## Execution Steps

1. **Parse arguments** from `$ARGUMENTS` into dataset and optional target.

2. **Verify dataset exists**
   ```bash
   ls data/automotive/<dataset>/
   ```

3. **Dry run first** to see what will be executed. If a target is given, use it as a Snakemake rule name. If no target, run the full DAG for that dataset.
   ```bash
   PYTHONPATH=. snakemake -s pipeline/Snakefile --config "datasets=[\"<dataset>\"]" <target> -n 2>&1 | head -50
   ```

4. **Submit to SLURM** if dry run looks correct
   ```bash
   PYTHONPATH=. snakemake -s pipeline/Snakefile --config "datasets=[\"<dataset>\"]" --profile profiles/slurm
   ```

5. **Report the submitted job IDs** and show how to monitor with `squeue -u $USER`.

## Common Targets

| Target | Description | Output |
|--------|-------------|--------|
| `large` | Full large pipeline (all datasets) | 3 stages per dataset |
| `small_nokd` | Small without KD (all datasets) | 3 stages per dataset |
| `small_kd` | Small with KD (all datasets, needs large) | 3 stages per dataset |
| `evaluate_all` | Evaluation for all 3 variants | Metrics JSON per dataset |
| `vgae_large` | Single rule: VGAE large autoencoder | 1 stage |
| `gat_small_kd` | Single rule: GAT small with KD | 1 stage |
| `dqn_large` | Single rule: DQN large fusion | 1 stage |

## Notes

- Pipeline runs on SLURM with GPU resources (V100, 128GB RAM scaling to 256GB on retry)
- SLURM logs: `slurm_logs/<jobid>-<rule>.{out,err}`
- Per-rule logs: `experimentruns/{ds}/{model}_{scale}_{stage}[_{aux}]/log.{out,err}`
- MLflow tracking is automatic
- Write-through DB: runs are recorded in `data/project.db` before/after each stage
- Preprocessing is cached between workflows (`SNAKEMAKE_OUTPUT_CACHE` on scratch)
- Evaluation rules are grouped into single SLURM submissions
- Training rules retry twice with doubled memory on failure
- Snakefile needs `PYTHONPATH=.` to find the `config` package
- Always do a dry run (`-n`) before submitting
