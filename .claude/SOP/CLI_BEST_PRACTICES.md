# CLI Best Practices & SOP

## RULE 1: Always Include --modality
**CRITICAL**: Every CLI command MUST include `--modality` (automotive, industrial, robotics)

### ✅ CORRECT - With Modality
```bash
./can-train --training-strategyl vgae --dataset hcrl_ch --training-strategy autoencoder --modality automotive --submit
```

---

## RULE 2: Pipeline Commands - Comma Separation & Matching Lengths

### Single-Value Parameters (Same for all jobs)
- `--dataset` (ONE value)
- `--modality` (ONE value)
- `--account`, `--partition`, `--walltime` (SLURM options)

### Multi-Value Parameters (ONE per job)
- `--training-strategyl` (VGAE, GAT, DQN, etc.)
- `--learning-type` (unsupervised, supervised, rl_fusion)
- `--training-strategy` (autoencoder, curriculum, fusion, normal)
- `--distillation` (with-kd, no-kd) - PER JOB
- `--teacher_path` (if using KD) - ONE path for all

### ✅ CORRECT - Matching Lengths
```bash
./can-train pipeline \
  --training-strategyl vgae,gat,dqn \
  --learning-type unsupervised,supervised,rl_fusion \
  --training-strategy autoencoder,curriculum,fusion \
  --distillation with-kd,with-kd,no-kd \
  --dataset hcrl_sa \
  --modality automotive \
  --teacher_path /path/to/teacher.pth \
  --submit
```

### ❌ WRONG - Mismatched Lengths
```bash
# Error: 3 models but only 2 modes
./can-train pipeline --training-strategyl vgae,gat,dqn --training-strategy autoencoder,curriculum ...

# Error: --distillation missing for 3 jobs (defaults to no-kd for all)
./can-train pipeline --training-strategyl vgae,gat,dqn --distillation with-kd ...
```

---

## RULE 3: Knowledge Distillation Configuration

### When to Use KD
- ✅ `autoencoder + with-kd`: VGAE student learns from teacher
- ✅ `curriculum + with-kd`: GAT student learns from teacher
- ✅ `normal + with-kd`: Standard soft label KD
- ❌ `fusion + with-kd`: NOT SUPPORTED - validation rejects

### Requirements When Using with-kd
1. **Teacher Path**: Must provide valid `--teacher_path /path/to/model.pth`
2. **File Must Exist**: Validated before SLURM submission
3. **Model Size**: Independent - any model size can use KD (student is typical but not required)

### ✅ CORRECT KD Pipeline (student model with KD - typical)
```bash
./can-train pipeline \
  --training-strategyl vgae,gat \
  --learning-type unsupervised,supervised \
  --training-strategy autoencoder,curriculum \
  --distillation with-kd,with-kd \
  --dataset hcrl_sa \
  --modality automotive \
  --training-strategyl-size student \
  --teacher_path /path/to/vgae_gat_teacher.pth \
  --submit
```

### ✅ ALSO VALID (teacher model with KD - atypical but allowed)
```bash
./can-train pipeline \
  --training-strategyl vgae,gat \
  --training-strategy autoencoder,curriculum \
  --distillation with-kd,with-kd \
  --training-strategyl-size teacher \
  --teacher_path /path/to/other_teacher.pth \
  ...
```

### ❌ WRONG KD Configs
```bash
# Missing teacher_path
./can-train pipeline ... --distillation with-kd,with-kd --submit

# Including fusion with KD (rejected)
./can-train pipeline --training-strategy autoencoder,curriculum,fusion --distillation with-kd,with-kd,with-kd ...
```

---

## RULE 4: Model Size Selection (Independent of KD)

### Key Principle
`--training-strategyl-size` and `--distillation` are INDEPENDENT dimensions:
- Model size = architecture capacity (teacher=larger, student=smaller)
- Distillation = learning strategy (with-kd=learn from teacher, no-kd=standard training)

### Teacher Models
- Larger architectures
- Training mode: `autoencoder`, `curriculum`, `normal`
- Flag: `--training-strategyl-size teacher` (or omit, defaults to teacher)
- Can optionally use KD (atypical but valid)

### Student Models
- Smaller architectures (compressed)
- Flag: `--training-strategyl-size student`
- Typically used with `--distillation with-kd` to learn from teachers
- Can also train WITHOUT KD (standard small model training)

### Examples
```bash
# Teacher without KD (typical pretraining)
--training-strategyl-size teacher --distillation no-kd

# Student with KD (typical compression)
--training-strategyl-size student --distillation with-kd

# Teacher with KD (atypical but valid)
--training-strategyl-size teacher --distillation with-kd

# Student without KD (valid small model training)
--training-strategyl-size student --distillation no-kd
```

---

## RULE 5: Common Mistakes to Avoid

| Mistake | What Happens | Fix |
|---------|--------------|-----|
| Forget `--modality` | Config validation fails cryptically | ALWAYS include it |
| Use teacher+with-kd | Valid (atypical but allowed) | Informational note displayed |
| Include fusion+with-kd | Validation rejects entire pipeline | Remove fusion from --training-strategy OR use --distillation no-kd |
| Wrong comma count | "Multi-value parameters must have same length" | Count carefully: len(model)==len(mode)==len(distillation) |
| Missing teacher_path | "Teacher model path required when with-kd" | Provide valid path with `--teacher_path` |
| Teacher file doesn't exist | Validation fails before SLURM submission | Check path: `ls /path/to/model.pth` |
| No --submit flag | Pipeline generates scripts but doesn't submit | Add `--submit` at end, or use `--dry-run` to preview |

---

## RULE 6: Dry-Run vs Submit

### Preview Pipeline Without Submitting
```bash
./can-train pipeline \
  --training-strategyl vgae,gat \
  --training-strategy autoencoder,curriculum \
  --dataset hcrl_sa \
  --modality automotive \
  --dry-run
```
**Result**: Shows pipeline summary, doesn't create SLURM scripts

### Generate Scripts (Manual Review)
```bash
./can-train pipeline \
  --training-strategyl vgae,gat \
  --training-strategy autoencoder,curriculum \
  --dataset hcrl_sa \
  --modality automotive
```
**Result**: Creates SLURM scripts in `experimentruns/slurm_runs/{dataset}/`, doesn't submit

### Submit to SLURM
```bash
./can-train pipeline \
  --training-strategyl vgae,gat \
  --training-strategy autoencoder,curriculum \
  --dataset hcrl_sa \
  --modality automotive \
  --submit
```
**Result**: Creates and immediately submits SLURM jobs

---

## RULE 7: SLURM Options

### Standard Defaults
- `--account PAS3209` (OSC account)
- `--partition gpu` (GPU partition)
- `--walltime 06:00:00` (6 hours)
- `--memory 64G` (per node)
- `--gpus 1` (single GPU v100)

### Override Example
```bash
./can-train pipeline \
  --training-strategyl vgae,gat \
  --dataset hcrl_sa \
  --modality automotive \
  --account PAS3209 \
  --partition gpu \
  --walltime 12:00:00 \
  --memory 96G \
  --gpus 1 \
  --submit
```

### When to Increase Resources
| Scenario | Recommendation |
|----------|-----------------|
| Large dataset (set_02+) | `--memory 96G`, `--walltime 12:00:00` |
| KD training | Use KD safety factors (auto-applied) |
| Multiple GPUs (not supported yet) | `--gpus 2` (experimental) |

---

## RULE 8: Checking Job Status

### View Running/Queued Jobs
```bash
squeue -u $USER
```

### View Specific Job Details
```bash
squeue -u $USER -j <JOBID>
```

### Check Job Output
```bash
# Standard output
tail -f experimentruns/slurm_runs/{dataset}/{job_name}_{timestamp}.out

# Error output
tail -f experimentruns/slurm_runs/{dataset}/{job_name}_{timestamp}.err
```

### Job Dependencies
- Job 2 depends on Job 1 (afterok:JOBID)
- Job 2 starts only when Job 1 succeeds
- If Job 1 fails, Job 2 is cancelled

---

## RULE 9: Result Paths

### Canonical Result Directory
Format: `{modality}/{dataset}/{model_size}/{learning_type}/{model_type}/{distillation}/{mode}`

### Examples
```
# Teacher VGAE Autoencoder (no KD)
experimentruns/automotive/hcrl_sa/teacher/unsupervised/vgae/no_distillation/autoencoder/

# Student VGAE Autoencoder with KD
experimentruns/automotive/hcrl_sa/student/unsupervised/vgae/with_kd/autoencoder/

# Student GAT Curriculum with KD
experimentruns/automotive/hcrl_sa/student/supervised/gat/with_kd/curriculum/
```

### SLURM Scripts Location
```
experimentruns/slurm_runs/{dataset}/{job_name}.sh
```

---

## RULE 10: Dataset Selection & OOM Risk

### Safe Datasets (Small)
- `hcrl_ch`, `hcrl_sa`, `set_01`
- Autoencoder stage: Use defaults
- Curriculum stage: Use defaults

### Risk Datasets (Large)
- `set_02`, `set_03`, `set_04`
- Autoencoder stage: Needs conservative safety factors
- KD on these: Even more conservative

### If OOM Occurs
1. Check error log: `...err` file
2. Note peak memory usage if reported
3. Decrease batch size manually (advanced)
4. Or increase safety factor and rerun

---

## Template: Complete KD Pipeline for New Dataset

```bash
# Before running: Ensure teacher models exist
ls /path/to/trained_teachers/vgae_teacher_{dataset}.pth
ls /path/to/trained_teachers/gat_teacher_{dataset}.pth

# Step 1: Dry-run to verify configuration
./can-train pipeline \
  --training-strategyl vgae,gat \
  --learning-type unsupervised,supervised \
  --training-strategy autoencoder,curriculum \
  --distillation with-kd,with-kd \
  --dataset {DATASET} \
  --modality automotive \
  --teacher_path /path/to/trained_teachers/ \
  --dry-run

# Step 2: Review dry-run output, then submit
./can-train pipeline \
  --training-strategyl vgae,gat \
  --learning-type unsupervised,supervised \
  --training-strategy autoencoder,curriculum \
  --distillation with-kd,with-kd \
  --dataset {DATASET} \
  --modality automotive \
  --teacher_path /path/to/trained_teachers/ \
  --submit

# Step 3: Monitor
squeue -u rf15
tail -f experimentruns/slurm_runs/{DATASET}/vgae_*.err
```
