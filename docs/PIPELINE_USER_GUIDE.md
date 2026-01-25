# Pipeline Submission System - User Guide

## Overview

The `submit_pipeline.py` script provides **automated sequential training pipelines** with SLURM dependency chaining for the KD-GAT project. It combines the configuration flexibility of `oscjobmanager.py` with automatic job orchestration.

---

## 🔑 Key Concepts

### Architecture: Independent Jobs (NOT Orchestrator-Managed)

**How It Works:**
```
┌─────────────────────────────────────────────────────────────┐
│ submit_pipeline.py (runs on login node)                    │
│                                                             │
│  1. Creates SLURM script for VGAE     → submits to SLURM  │
│  2. Creates SLURM script for GAT      → submits to SLURM  │
│  3. Creates SLURM script for Fusion   → submits to SLURM  │
│                                                             │
│  Each submission includes dependency on previous job       │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    SLURM Job Queue
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Job 123456: VGAE Autoencoder                                 │
│   - Walltime: 8 hours                                        │
│   - Resources: 16 CPUs, 1 GPU, 96GB RAM                     │
│   - Status: Running immediately                              │
└──────────────────────────────────────────────────────────────┘
                            ↓ (dependency: afterok:123456)
┌──────────────────────────────────────────────────────────────┐
│ Job 123457: Curriculum GAT                                   │
│   - Walltime: 12 hours                                       │
│   - Resources: 16 CPUs, 1 GPU, 128GB RAM                    │
│   - Status: Pending (waits for 123456 to complete)          │
└──────────────────────────────────────────────────────────────┘
                            ↓ (dependency: afterok:123457)
┌──────────────────────────────────────────────────────────────┐
│ Job 123458: Fusion DQN                                       │
│   - Walltime: 10 hours                                       │
│   - Resources: 16 CPUs, 1 GPU, 96GB RAM                     │
│   - Status: Pending (waits for 123457 to complete)          │
└──────────────────────────────────────────────────────────────┘
```

### ⏰ Walltime Logic: Per-Job, NOT Cumulative

**IMPORTANT:** Each job gets its own independent walltime allocation.

- ✅ **`--walltime 08:00:00`** means **EACH training job** gets 8 hours
- ❌ **NOT** divided between jobs (e.g., NOT 2h 40m per job if 3 jobs)
- ❌ **NOT** total pipeline time (e.g., NOT "pipeline must finish in 8 hours")

**Example:**
```bash
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa \
    --walltime 10:00:00 --memory 96G
```

This creates:
- **Job 1 (VGAE):** 10 hours, 96GB  → Runs 0-10 hours
- **Job 2 (GAT):** 10 hours, 96GB   → Runs 10-20 hours (waits for Job 1)
- **Job 3 (Fusion):** 10 hours, 96GB → Runs 20-30 hours (waits for Job 2)

**Total pipeline time:** Up to 30 hours (but only 1 GPU used at a time)

---

## 🎯 Design Decision: Why Independent Jobs?

### ✅ Pros (Current Design)

1. **No Resource Waste**
   - Each job gets GPU only when training
   - No idle GPU time during orchestration

2. **Fault Tolerance**
   - If VGAE finishes but GAT fails → Can resubmit just GAT
   - No need to re-run entire pipeline

3. **Clear Monitoring**
   - Each job has separate logs
   - `squeue` shows exact stage in progress
   - Easy to track individual stage progress

4. **Flexible Resources Per Stage**
   - VGAE can use 8 hours, 96GB
   - GAT can use 12 hours, 128GB (needs more time/memory)
   - Fusion can use 10 hours, 96GB

5. **Queue Efficiency**
   - Jobs enter queue independently
   - Can start immediately if resources available
   - No orchestrator job blocking scheduler

### ❌ Cons (Alternative: Orchestrator Job)

If we used a single orchestrator job:

```
Job 123456: Pipeline Orchestrator (24 hours, 1 GPU, 128GB)
  ├─ Train VGAE (8 hours)      ← GPU active
  ├─ Train GAT (12 hours)      ← GPU active
  └─ Train Fusion (4 hours)    ← GPU active

Problems:
  • Orchestrator needs GPU allocated for full 24 hours
  • If orchestrator crashes at hour 20, lose all progress
  • Can't restart individual stages
  • GPU sits idle during Python orchestration overhead
  • Harder to monitor individual stage progress
  • Wastes allocation if early stage fails
```

### 📊 Resource Allocation Comparison

| Design | GPU Hours | CPU Hours | Memory Efficiency | Fault Tolerance |
|--------|-----------|-----------|-------------------|-----------------|
| **Independent Jobs** (current) | 30 hrs (1 GPU × 30h) | 480 hrs (16 CPU × 30h) | ✅ Each job gets what it needs | ✅ Restart individual stages |
| **Orchestrator** | 30 hrs (1 GPU × 30h) | 480 hrs (16 CPU × 30h) | ⚠️ All or nothing | ❌ Restart entire pipeline |

**Verdict:** Independent jobs are more efficient for HPC environments with job queues.

---

## 📋 Quick Start

### List Available Pipelines
```bash
python scripts/submit_pipeline.py --list-pipelines --pipeline teacher --dataset hcrl_sa
```

### Submit a Pipeline
```bash
# Basic submission (uses default resources per stage)
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa

# Override all stages with same resources
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa \
    --walltime 12:00:00 --memory 128G

# Dry run (preview without submitting)
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa --dry-run
```

---

## 🔧 Available Pipelines

### 1. Teacher Pipeline (Recommended)
**Purpose:** Train full teacher-sized models with curriculum learning

```bash
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa
```

**Stages:**
| Stage | Model | Default Walltime | Default Memory | Purpose |
|-------|-------|------------------|----------------|---------|
| 1 | VGAE Autoencoder | 8 hours | 96GB | Unsupervised feature learning |
| 2 | Curriculum GAT | 12 hours | 128GB | Supervised classification with hard mining |
| 3 | Fusion DQN | 10 hours | 96GB | Multi-agent ensemble via RL |

**Total Time:** ~30 hours (sequential)

### 2. Student Pipeline
**Purpose:** Train compact student models with knowledge distillation

```bash
python scripts/submit_pipeline.py --pipeline student --dataset hcrl_sa
```

**Stages:**
| Stage | Model | Default Walltime | Default Memory |
|-------|-------|------------------|----------------|
| 1 | VGAE Autoencoder | 6 hours | 64GB |
| 2 | Distilled GAT Student (50% scale) | 8 hours | 96GB |

**Total Time:** ~14 hours

### 3. Supervised Only
**Purpose:** Skip curriculum learning, use standard GAT training

```bash
python scripts/submit_pipeline.py --pipeline supervised_only --dataset hcrl_sa
```

**Stages:** VGAE (8h) → GAT Normal (10h) → Fusion (10h)  
**Total Time:** ~28 hours

### 4. Curriculum Only
**Purpose:** Train up to curriculum GAT (no fusion)

```bash
python scripts/submit_pipeline.py --pipeline curriculum_only --dataset hcrl_sa
```

**Stages:** VGAE (8h) → Curriculum GAT (12h)  
**Total Time:** ~20 hours

### 5. Custom Pipeline
**Purpose:** Define your own sequence with any presets

```bash
python scripts/submit_pipeline.py --pipeline custom --dataset hcrl_sa \
    --presets "autoencoder_hcrl_sa,gat_normal_hcrl_sa,fusion_hcrl_sa"
```

---

## 🎛️ Resource Configuration

### Global Override (All Stages)
```bash
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa \
    --walltime 10:00:00 \
    --memory 96G \
    --cpus 32 \
    --gpus 2
```
**Effect:** All 3 stages get 10h walltime, 96GB RAM, 32 CPUs, 2 GPUs

### Default Resources (Pre-Defined Pipelines)
If you don't specify resources, each pipeline uses stage-specific defaults:

**Teacher Pipeline Defaults:**
- VGAE: 8h, 96GB
- Curriculum GAT: 12h, 128GB (needs more time for curriculum)
- Fusion: 10h, 96GB

### Custom Per-Stage Resources
For maximum control, use custom pipeline and submit stages manually:

```bash
# Stage 1: VGAE (short time, low memory)
python oscjobmanager.py submit autoencoder_hcrl_sa \
    --walltime 06:00:00 --memory 64G

# Wait for Job ID, then:

# Stage 2: Curriculum GAT (long time, high memory)
python oscjobmanager.py submit curriculum_hcrl_sa \
    --walltime 16:00:00 --memory 192G

# Stage 3: Fusion (medium resources)
python oscjobmanager.py submit fusion_hcrl_sa \
    --walltime 08:00:00 --memory 96G
```

---

## 📊 Monitoring & Management

### Check Job Queue
```bash
# View all your jobs
squeue -u $USER

# View specific pipeline
squeue -u $USER | grep -E "(vgae|curriculum|fusion)"

# Detailed job info
scontrol show job <JOB_ID>
```

### Check Job Dependencies
```bash
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.8T %.10M %.6D %R %E"
```
**Look for:** `(Dependency)` in reason column

### Monitor Logs (Real-Time)
```bash
# Follow VGAE logs
tail -f experimentruns/slurm_runs/hcrl_sa/autoencoder/*.log

# Follow all pipeline logs
tail -f experimentruns/slurm_runs/hcrl_sa/*/*.log
```

### Cancel Pipeline
```bash
# Cancel all your jobs
scancel -u $USER

# Cancel specific jobs (from submission output)
scancel 123456 123457 123458

# Cancel by name pattern
scancel -n curriculum_hcrl_sa
```

### Restart Failed Stage
If a stage fails, you can restart just that stage:

```bash
# Example: If Curriculum GAT fails, VGAE is already done
python oscjobmanager.py submit curriculum_hcrl_sa --walltime 12:00:00

# Then submit fusion with dependency
# (or resubmit the pipeline from curriculum stage)
```

---

## 🔬 Advanced Usage

### Multi-Dataset Pipeline Sweep
```bash
#!/bin/bash
# Submit teacher pipeline for all datasets

for dataset in hcrl_sa hcrl_ch set_01 set_02; do
    echo "Submitting pipeline for $dataset"
    python scripts/submit_pipeline.py \
        --pipeline teacher \
        --dataset $dataset \
        --walltime 10:00:00 \
        --memory 96G
    
    sleep 5  # Avoid overwhelming scheduler
done
```

### Mixed Configuration (Different Pipelines per Dataset)
```bash
# Large datasets: use teacher pipeline
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa

# Small datasets: use supervised only
python scripts/submit_pipeline.py --pipeline supervised_only --dataset set_01
```

### Custom Multi-Stage Pipeline
```bash
# Train multiple student scales sequentially
python scripts/submit_pipeline.py --pipeline custom --dataset hcrl_sa \
    --presets "autoencoder_hcrl_sa,distillation_hcrl_sa_scale_0.25,distillation_hcrl_sa_scale_0.5,distillation_hcrl_sa_scale_0.75"
```

---

## 🚨 Troubleshooting

### Issue: "Could not find generated script"
**Cause:** `oscjobmanager.py` failed to create SLURM script

**Solution:**
```bash
# Test script generation manually
python oscjobmanager.py submit autoencoder_hcrl_sa --dry-run

# Check for errors
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa --dry-run
```

### Issue: Jobs not starting (stuck in Pending)
**Cause:** Dependency job hasn't finished or failed

**Check:**
```bash
# View job status and dependencies
squeue -u $USER -o "%.18i %.20j %.8T %R %E"

# If dependency job failed:
scontrol show job <FAILED_JOB_ID>

# Release dependency (if you want job to run anyway)
scontrol release <PENDING_JOB_ID>
```

### Issue: Pipeline runs out of time
**Symptoms:** Job killed with "TIME LIMIT" in error log

**Solutions:**
```bash
# Increase walltime for specific pipeline
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa \
    --walltime 20:00:00  # Double the default

# Or use custom pipeline with per-stage times
python oscjobmanager.py submit autoencoder_hcrl_sa --walltime 12:00:00
# (manual submission for fine control)
```

### Issue: Out of Memory (OOM)
**Symptoms:** Job fails with "Out of memory" or segmentation fault

**Solutions:**
```bash
# Increase memory
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa \
    --memory 192G

# Or reduce batch size (requires manual config override)
# Edit config before submission
```

---

## 📈 Resource Planning Guide

### Estimating Requirements

**Rule of Thumb for Teacher Models:**
- **VGAE Autoencoder:** 6-10 hours, 64-96GB
- **GAT Normal:** 8-12 hours, 96-128GB
- **GAT Curriculum:** 10-16 hours, 128-192GB (longer due to curriculum)
- **Fusion DQN:** 8-12 hours, 96-128GB

**Factors that increase time/memory:**
- Larger datasets (set_01 > hcrl_sa)
- More epochs
- Larger batch sizes (more memory, possibly less time)
- Higher model complexity

### Conservative Allocation (Recommended)
```bash
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa \
    --walltime 16:00:00 --memory 192G
```
**Advantage:** Less likely to time out  
**Disadvantage:** May wait longer in queue for large allocations

### Aggressive Allocation (For Quick Iteration)
```bash
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa \
    --walltime 06:00:00 --memory 64G
```
**Advantage:** Faster queue times  
**Disadvantage:** May time out and waste partial work

---

## 🎓 Best Practices

### 1. Always Dry Run First
```bash
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa --dry-run
```
Verify scripts are generated correctly before actual submission.

### 2. Use Conservative Walltimes Initially
Better to over-allocate time than lose partial progress. You can reduce later.

### 3. Monitor First Stage Closely
If VGAE times out or OOMs, the entire pipeline fails early. Watch its progress:
```bash
tail -f experimentruns/slurm_runs/hcrl_sa/autoencoder/*.log
```

### 4. Keep Track of Job IDs
Save the submission output:
```bash
python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa \
    2>&1 | tee pipeline_submission_$(date +%Y%m%d_%H%M%S).log
```

### 5. Test on Small Dataset First
```bash
# Test pipeline on smaller dataset
python scripts/submit_pipeline.py --pipeline teacher --dataset set_01 \
    --walltime 04:00:00 --memory 64G
```

### 6. Stagger Multiple Pipelines
Don't submit 10 pipelines simultaneously:
```bash
for ds in hcrl_sa hcrl_ch set_01; do
    python scripts/submit_pipeline.py --pipeline teacher --dataset $ds
    sleep 60  # Wait 1 minute between submissions
done
```

---

## 📞 Support

### Log Locations
```
experimentruns/slurm_runs/<dataset>/<training_type>/*.log
experimentruns/slurm_runs/<dataset>/<training_type>/*.err
```

### Useful Commands Reference
```bash
# Submit
python scripts/submit_pipeline.py --pipeline <type> --dataset <name>

# Monitor
squeue -u $USER
tail -f experimentruns/slurm_runs/<dataset>/*/*.log

# Cancel
scancel <JOB_ID>
scancel -u $USER

# Check resources
sacct -j <JOB_ID> --format=JobID,JobName,Elapsed,MaxRSS,State
```

---

## 📚 Related Documentation

- **oscjobmanager.py:** Individual job submission
- **train_with_hydra_zen.py:** Direct training (no SLURM)
- **src/config/hydra_zen_configs.py:** Configuration presets
- **SLURM Documentation:** https://slurm.schedmd.com/
