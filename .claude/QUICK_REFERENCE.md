# Quick Reference Card

Print this or keep it visible while working.

---

## ⭐ THE THREE GOLDEN RULES

### RULE 1: ALWAYS ADD `--modality automotive`
```bash
# ❌ WRONG
./can-train --training-strategyl vgae --dataset hcrl_ch ...

# ✅ CORRECT
./can-train --training-strategyl vgae --dataset hcrl_ch --modality automotive ...
```

### RULE 2: COMMA PARAMS MUST MATCH LENGTH
```bash
# ❌ WRONG (3 models, 2 modes)
./can-train pipeline --training-strategyl vgae,gat,dqn --training-strategy autoencoder,curriculum ...

# ✅ CORRECT (3 each)
./can-train pipeline \
  --training-strategyl vgae,gat,dqn \
  --training-strategy autoencoder,curriculum,fusion \
  --distillation with-kd,with-kd,no-kd ...
```

### RULE 3: FUSION MODE + KD = REJECTED
```bash
# ❌ WRONG (fusion can't use KD)
./can-train pipeline --training-strategy autoencoder,curriculum,fusion \
  --distillation with-kd,with-kd,with-kd ...

# ✅ CORRECT (fusion uses no-kd)
./can-train pipeline --training-strategy autoencoder,curriculum,fusion \
  --distillation with-kd,with-kd,no-kd ...
```

---

## ⚡ Single Job (No Pipeline)

### Teacher Training (Baseline)
```bash
./can-train \
  --training-strategyl vgae \
  --dataset hcrl_ch \
  --training-strategy autoencoder \
  --modality automotive \
  --submit
```

### Student with KD
```bash
./can-train \
  --training-strategyl vgae_student \
  --dataset hcrl_ch \
  --training-strategy autoencoder \
  --modality automotive \
  --use-kd \
  --teacher_path /path/to/vgae_teacher.pth \
  --submit
```

---

## 🔗 Three-Stage Pipeline with KD

### Full Template
```bash
./can-train pipeline \
  --training-strategyl vgae,gat,dqn \
  --learning-type unsupervised,supervised,rl_fusion \
  --training-strategy autoencoder,curriculum,fusion \
  --distillation with-kd,with-kd,no-kd \
  --dataset hcrl_sa \
  --modality automotive \
  --training-strategyl-size student \
  --teacher_path /path/to/teacher.pth \
  --submit
```

### What This Does
- **Job 1**: VGAE student + autoencoder + KD
- **Job 2**: GAT student + curriculum + KD (depends on Job 1)
- **Job 3**: DQN + fusion + NO-KD (depends on Job 2)

---

## 🧪 Testing Commands

### Dry-Run (No Submission)
```bash
./can-train pipeline ... --dry-run
```

### Generate Scripts (No Submit)
```bash
./can-train pipeline ...
# (no --submit flag)
# Creates scripts in experimentruns/slurm_runs/{dataset}/
```

### Submit
```bash
./can-train pipeline ... --submit
```

---

## 📊 Check Status

### List Jobs
```bash
squeue -u rf15
```

### Check Error Log
```bash
tail -f experimentruns/slurm_runs/{DATASET}/{JOB_NAME}_{TIMESTAMP}.err
```

### Check Output Log
```bash
tail -f experimentruns/slurm_runs/{DATASET}/{JOB_NAME}_{TIMESTAMP}.out
```

### Verify KD is Running
```bash
grep -i "Knowledge Distillation: ENABLED" \
  experimentruns/slurm_runs/{DATASET}/*.out
```

---

## 🎯 Common Scenarios

### Scenario 1: Train Teacher Models (Baseline)
```bash
# VGAE Teacher
./can-train --training-strategyl vgae --dataset hcrl_sa --training-strategy autoencoder \
  --modality automotive --submit

# GAT Teacher
./can-train --training-strategyl gat --dataset hcrl_sa --training-strategy curriculum \
  --modality automotive --submit

# DQN (Fusion)
./can-train --training-strategyl dqn --dataset hcrl_sa --training-strategy fusion \
  --modality automotive --submit
```

### Scenario 2: KD Pipeline (Student Learning)
```bash
./can-train pipeline \
  --training-strategyl vgae,gat \
  --learning-type unsupervised,supervised \
  --training-strategy autoencoder,curriculum \
  --distillation with-kd,with-kd \
  --dataset hcrl_sa \
  --modality automotive \
  --training-strategyl-size student \
  --teacher_path /path/to/teacher_models/ \
  --submit
```

### Scenario 3: Full Three-Stage (Teacher + KD + Fusion)
```bash
# This requires 3 separate pipelines or manual orchestration
# (Current limitation - would need job dependencies across pipelines)
```

---

## ❌ COMMON MISTAKES

| Mistake | Error | Fix |
|---------|-------|-----|
| Missing `--modality` | Config validation fails | Add `--modality automotive` |
| Fusion + with-kd | Pipeline rejected | Use `no-kd` for fusion: `--distillation ...,no-kd` |
| Teacher + with-kd | Wrong combo | Auto-corrected to `student`, or explicitly pass `--training-strategyl-size student` |
| Wrong comma count | "Multi-value parameters must have same length" | Count: len(model)==len(mode)==len(distillation) |
| Missing teacher file | "Teacher model not found" | Verify: `ls /path/to/model.pth` |
| No `--submit` | Scripts created but not submitted | Add `--submit` to execute |
| Old SLURM scripts | Jobs run without KD | Delete old scripts: `rm experimentruns/slurm_runs/*/*_old.sh` |

---

## 📝 Parameter Reference

### Required (Always)
- `--training-strategyl`: vgae, gat, dqn
- `--dataset`: hcrl_ch, hcrl_sa, set_01, set_02, set_03, set_04
- `--modality`: automotive, industrial, robotics
- `--training-strategy`: autoencoder, curriculum, fusion, normal

### Conditional
- `--distillation`: with-kd (requires teacher_path) or no-kd (default)
- `--teacher_path`: Required when `--distillation with-kd`
- `--training-strategyl-size`: teacher (default) or student. INDEPENDENT of --distillation

### Optional
- `--walltime`: 06:00:00 (default)
- `--memory`: 64G (default)
- `--gpus`: 1 (default)
- `--dry-run`: Preview without creating scripts
- `--submit`: Actually submit to SLURM

---

## 🔄 Documentation References

Need help? Check:
- **PROJECT_OVERVIEW.md** → Architecture & design
- **CLI_BEST_PRACTICES.md** → Commands & rules
- **PENDING_WORK.md** → What's being worked on
- **INDEX.md** → Navigation guide

---

## 🚨 If Something Breaks

1. **Check the error log**: `tail -f experimentruns/slurm_runs/{DATASET}/*.err`
2. **Look for "Knowledge Distillation"**: If not present, KD didn't initialize
3. **Verify teacher path**: `ls /path/to/model.pth`
4. **Check safety factors**: `cat config/batch_size_factors.json`
5. **Compare to template**: Use commands above as reference

---

**Last Updated**: 2026-01-26
**For Questions**: See `.claude/INDEX.md`
