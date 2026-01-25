# Pipeline Submission - Quick Reference

## ⚡ TL;DR

```bash
# Submit teacher pipeline with defaults
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa

# With custom resources (applies to ALL stages)
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa \
    --walltime 12:00:00 --memory 128G
```

---

## 🔑 Key Facts

### Walltime is PER-JOB, Not Total
```
--walltime 10:00:00 means:
  ✅ Job 1 (VGAE): 10 hours
  ✅ Job 2 (GAT):  10 hours  (starts after Job 1)
  ✅ Job 3 (Fusion): 10 hours  (starts after Job 2)
  
Total pipeline time: up to 30 hours
```

### Jobs Run Independently
```
Each job:
  • Has its own SLURM allocation
  • Gets GPU only when running
  • Can be restarted individually if fails
  • Has separate log files
```

---

## 📋 Pipelines

| Name | Command | Stages | Total Time | Use Case |
|------|---------|--------|------------|----------|
| **teacher** | `--pipeline teacher` | VGAE (8h) → Curriculum GAT (12h) → Fusion (10h) | ~30h | Full teacher models |
| **student** | `--pipeline student` | VGAE (6h) → Distilled GAT (8h) | ~14h | Compact student models |
| **supervised_only** | `--pipeline supervised_only` | VGAE (8h) → GAT Normal (10h) → Fusion (10h) | ~28h | Skip curriculum |
| **curriculum_only** | `--pipeline curriculum_only` | VGAE (8h) → Curriculum GAT (12h) | ~20h | No fusion |
| **custom** | `--pipeline custom --presets "a,b,c"` | User-defined | Varies | Any sequence |

---

## 🎛️ Resource Configuration

### Default Resources (Built-In)
```bash
# Teacher pipeline uses stage-specific defaults:
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa

# Stage 1 (VGAE):      8h, 96GB
# Stage 2 (Curriculum): 12h, 128GB  ← Needs more time/memory
# Stage 3 (Fusion):    10h, 96GB
```

### Override All Stages
```bash
# All stages get same resources:
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa \
    --walltime 20:00:00 --memory 256G --cpus 32
```

### Custom Per-Stage (Advanced)
```bash
# Submit stages manually for fine control:
python oscjobmanager.py submit autoencoder_hcrl_sa --walltime 06:00:00 --memory 64G
python oscjobmanager.py submit curriculum_hcrl_sa --walltime 16:00:00 --memory 192G
python oscjobmanager.py submit fusion_hcrl_sa --walltime 08:00:00 --memory 96G
```

---

## 📊 Monitoring

```bash
# Check queue
squeue -u $USER

# Monitor logs
tail -f experimentruns/slurm_runs/hcrl_sa/*/*.log

# Cancel all
scancel -u $USER

# Cancel specific pipeline (use job IDs from submission output)
scancel 123456 123457 123458
```

---

## 🆘 Common Issues

| Problem | Solution |
|---------|----------|
| **Jobs stuck "Pending"** | Check dependency with `squeue -u $USER -o "%.18i %.20j %.8T %R %E"` |
| **"TIME LIMIT" error** | Increase `--walltime` |
| **"Out of memory"** | Increase `--memory` |
| **Script not found** | Run `python oscjobmanager.py submit <preset> --dry-run` to debug |

---

## 📚 Full Documentation

See: [docs/PIPELINE_USER_GUIDE.md](./PIPELINE_USER_GUIDE.md)
