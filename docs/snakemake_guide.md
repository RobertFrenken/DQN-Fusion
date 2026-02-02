# Snakemake Quick Reference for KD-GAT Pipeline

## What Are Profiles?

A profile is a folder containing a `config.yaml` that provides **default command-line flags** so you don't have to type them every time. When you pass `--profile profiles/slurm`, Snakemake reads `profiles/slurm/config.yaml` and applies its contents as if you'd typed them on the command line.

**Without a profile** you'd have to type:
```bash
snakemake -s pipeline/Snakefile \
  --cluster "sbatch --account=PAS3209 --partition=gpu --time=360 ..." \
  --cluster-status profiles/slurm/status.sh \
  --jobs 20 --latency-wait 120 --printshellcmds --rerun-incomplete
```

**With a profile** that becomes:
```bash
snakemake -s pipeline/Snakefile --profile profiles/slurm
```

The profile just saves you from repeating the same flags. Your profile at `profiles/slurm/config.yaml` currently sets:

| Key | What it does |
|-----|-------------|
| `cluster: sbatch ...` | Template for submitting each rule as a SLURM job. `{resources.X}` gets filled from the rule's `resources:` block |
| `cluster-status` | Script Snakemake calls to check if a submitted job is still running, succeeded, or failed |
| `cluster-cancel: scancel` | Command to cancel jobs if you Ctrl-C Snakemake |
| `jobs: 20` | Max 20 SLURM jobs in flight at once |
| `latency-wait: 120` | Wait up to 120s for output files to appear on NFS after a job finishes (NFS is slow to propagate) |
| `rerun-incomplete: true` | Re-run any jobs whose output files exist but were from a previously interrupted run |

**Local vs SLURM execution**: If you omit `--profile`, Snakemake runs rules **locally** on the login node (one at a time by default). With `--profile profiles/slurm`, each rule becomes a separate `sbatch` job on a GPU compute node.

---

## Command Reference

All commands assume you're in the project root (`KD-GAT/`). Add `-n` to any command for a **dry run** (shows what *would* run without actually running it).

### Run everything (all 6 datasets, all variants)

```bash
# Dry run first
snakemake -s pipeline/Snakefile --profile profiles/slurm -n

# Actually submit to SLURM
snakemake -s pipeline/Snakefile --profile profiles/slurm
```

This builds the `all` target: teacher fusion + student KD fusion + student no-KD fusion for all 6 datasets. That's up to 9 training jobs per dataset x 6 datasets = 54 jobs total (minus any already completed).

### Single dataset

```bash
snakemake -s pipeline/Snakefile --profile profiles/slurm \
  --config 'datasets=["hcrl_sa"]' -n
```

The `--config 'datasets=["hcrl_sa"]'` overrides the `DATASETS` variable in the Snakefile. Only rules for `hcrl_sa` will be planned.

### Subset of datasets

```bash
snakemake -s pipeline/Snakefile --profile profiles/slurm \
  --config 'datasets=["hcrl_sa","hcrl_ch"]' -n
```

### Only teachers (no students)

```bash
snakemake -s pipeline/Snakefile --profile profiles/slurm teachers -n
```

`teachers` is a **target rule** defined in the Snakefile (line 46). It only requests teacher fusion checkpoints. Named targets go at the end, after all flags.

### Only students with KD

```bash
snakemake -s pipeline/Snakefile --profile profiles/slurm students -n
```

### Only students without KD (ablation)

```bash
snakemake -s pipeline/Snakefile --profile profiles/slurm students_nokd -n
```

### Combine: teachers for one dataset

```bash
snakemake -s pipeline/Snakefile --profile profiles/slurm \
  --config 'datasets=["hcrl_sa"]' teachers -n
```

### Single specific model run (by file path)

Instead of a named target, you can request a **specific output file**:

```bash
# Just the teacher VGAE for hcrl_sa
snakemake -s pipeline/Snakefile --profile profiles/slurm \
  experimentruns/hcrl_sa/teacher_autoencoder/best_model.pt -n

# Just the teacher GAT for hcrl_sa (will also run VGAE if missing)
snakemake -s pipeline/Snakefile --profile profiles/slurm \
  experimentruns/hcrl_sa/teacher_curriculum/best_model.pt -n

# Just the teacher DQN for hcrl_sa (will also run VGAE + GAT if missing)
snakemake -s pipeline/Snakefile --profile profiles/slurm \
  experimentruns/hcrl_sa/teacher_fusion/best_model.pt -n

# Just the student KD GAT for set_01
snakemake -s pipeline/Snakefile --profile profiles/slurm \
  experimentruns/set_01/student_curriculum_kd/best_model.pt -n

# Evaluation for teacher on hcrl_sa
snakemake -s pipeline/Snakefile --profile profiles/slurm \
  experimentruns/hcrl_sa/teacher_evaluation/metrics.json -n
```

The file path pattern is always: `experimentruns/{dataset}/{size}_{stage}[_kd]/best_model.pt`

### Run locally (no SLURM, on login node)

```bash
# Omit --profile, use -j1 for one job at a time
snakemake -s pipeline/Snakefile -j1 \
  --config 'datasets=["hcrl_sa"]' teachers -n
```

Warning: login nodes have no GPU. Only useful for testing if rules start correctly (will crash at CUDA initialization).

### Visualize the DAG

```bash
snakemake -s pipeline/Snakefile --config 'datasets=["hcrl_sa"]' --dag | dot -Tpdf > dag.pdf
```

---

## How Snakemake Decides What to Run (Skip Logic)

This is the most important concept. Snakemake is **file-based** -- it decides what to run by checking whether **output files exist**.

### The core rule

> **If the output file already exists and is newer than all input files, the rule is skipped.**

### Step by step

1. You request a target (e.g., `teachers` or a specific file path)
2. Snakemake traces backwards through the DAG to find all rules needed
3. For each rule, it checks: does the `output:` file exist on disk?
   - **File exists** and is newer than inputs → **SKIP** (nothing to do)
   - **File missing** → **RUN** this rule
   - **File exists** but is older than an input → **RE-RUN** (input was updated)

### What this means for your pipeline

```
DAG for one dataset (teacher pipeline):

  vgae_teacher ──→ gat_teacher ──→ dqn_teacher
  (VGAE)           (GAT)           (DQN fusion)

  Output:          Output:          Output:
  best_model.pt    best_model.pt    best_model.pt
```

**Scenario 1: Fresh start (no files exist)**
→ Runs all 3: VGAE first, then GAT, then DQN

**Scenario 2: VGAE already completed**
→ `teacher_autoencoder/best_model.pt` exists → VGAE skipped
→ Runs GAT, then DQN

**Scenario 3: All 3 completed**
→ All `best_model.pt` files exist → **nothing runs**
→ Snakemake prints "Nothing to be done"

**Scenario 4: You delete the GAT checkpoint**
→ VGAE still exists → skipped
→ GAT output missing → **GAT re-runs**
→ DQN depends on GAT → **DQN also re-runs** (its input changed)

**Scenario 5: You want to force a re-run**
```bash
# Force re-run of a specific rule (even if output exists)
snakemake -s pipeline/Snakefile --profile profiles/slurm \
  --forcerun gat_teacher --config 'datasets=["hcrl_sa"]'

# Force re-run of everything from scratch
snakemake -s pipeline/Snakefile --profile profiles/slurm \
  --forceall --config 'datasets=["hcrl_sa"]'
```

### Summary of skip logic

| Situation | What happens |
|-----------|-------------|
| Output file missing | Rule runs |
| Output file exists, inputs unchanged | Rule skipped |
| Output file exists, but an input is newer | Rule re-runs |
| Output file exists, you used `--forcerun rulename` | Rule re-runs |
| Output file exists, you used `--forceall` | Everything re-runs |
| You deleted an upstream checkpoint | That rule + all downstream rules re-run |

---

## Useful Monitoring Commands

```bash
# Check SLURM queue for your jobs
squeue -u $USER

# Watch Snakemake's progress (it prints status to the terminal)
# Ctrl-C is safe: it cancels pending jobs but doesn't kill running ones

# Check a specific SLURM job's output
cat slurm_logs/<jobid>-<rule>.out

# Check training logs for a specific run
cat experimentruns/hcrl_sa/teacher_autoencoder/slurm.out
```

---

## All Available Output Paths (Cheat Sheet)

For any dataset `{ds}` (hcrl_ch, hcrl_sa, set_01-04):

| Path | Rule | Stage |
|------|------|-------|
| `experimentruns/{ds}/teacher_autoencoder/best_model.pt` | `vgae_teacher` | VGAE teacher |
| `experimentruns/{ds}/teacher_curriculum/best_model.pt` | `gat_teacher` | GAT teacher |
| `experimentruns/{ds}/teacher_fusion/best_model.pt` | `dqn_teacher` | DQN teacher |
| `experimentruns/{ds}/student_autoencoder_kd/best_model.pt` | `vgae_student` | VGAE student (KD) |
| `experimentruns/{ds}/student_curriculum_kd/best_model.pt` | `gat_student` | GAT student (KD) |
| `experimentruns/{ds}/student_fusion_kd/best_model.pt` | `dqn_student` | DQN student (KD) |
| `experimentruns/{ds}/student_autoencoder/best_model.pt` | `vgae_student_nokd` | VGAE student (no KD) |
| `experimentruns/{ds}/student_curriculum/best_model.pt` | `gat_student_nokd` | GAT student (no KD) |
| `experimentruns/{ds}/student_fusion/best_model.pt` | `dqn_student_nokd` | DQN student (no KD) |
| `experimentruns/{ds}/teacher_evaluation/metrics.json` | `eval_teacher` | Eval teacher |
| `experimentruns/{ds}/student_evaluation_kd/metrics.json` | `eval_student` | Eval student (KD) |
| `experimentruns/{ds}/student_evaluation/metrics.json` | `eval_student_nokd` | Eval student (no KD) |
