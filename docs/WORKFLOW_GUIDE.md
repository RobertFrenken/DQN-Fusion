# Workflow Guide

**Complete guide to job submission, manifest management, and experiment chaining**

---

## 1. Job Submission Workflow

### Single Job Submission

```bash
# Preview job configuration
python oscjobmanager.py preview --preset gat_normal_hcrl_sa

# Dry run (validate without submitting)
python oscjobmanager.py submit --preset gat_normal_hcrl_sa --dry-run

# Submit to SLURM
python oscjobmanager.py submit --preset gat_normal_hcrl_sa

# Custom resources
python oscjobmanager.py submit \
  --preset gat_normal_hcrl_sa \
  --walltime 08:00:00 \
  --memory 64G \
  --gpus 2
```

### Available Presets

```bash
# Teacher models
gat_normal_{dataset}          # GAT classifier
vgae_autoencoder_{dataset}    # VGAE reconstruction

# Curriculum learning
gat_curriculum_{dataset}      # GAT with VGAE-guided mining

# Knowledge distillation
distillation_{dataset}_0.25   # 25% student scale
distillation_{dataset}_0.5    # 50% student scale
distillation_{dataset}_0.75   # 75% student scale

# Fusion
fusion_{dataset}              # Multi-model DQN fusion

# Datasets: hcrl_sa, hcrl_ch, set_01, set_02, set_03, set_04
```

---

## 2. Monitoring Jobs

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# Detailed info
squeue -u $USER -o "%.10i %.9P %.20j %.8T %.10M %.6D %R"

# Watch continuously
watch -n 5 squeue -u $USER
```

### View Logs

```bash
# Tail output log
tail -f slurm_jobs/job_<jobid>.out

# Check error log
tail -f slurm_jobs/job_<jobid>.err

# View completed log
less slurm_jobs/job_<jobid>.out
```

### Cancel Jobs

```bash
# Cancel single job
scancel <jobid>

# Cancel all your jobs
scancel -u $USER

# Cancel by pattern
scancel -u $USER --name gat_normal
```

---

## 3. Dependency Manifests

### What is a Manifest?

A manifest declares required pre-trained models for complex jobs (fusion, curriculum, distillation).

**Example** (`manifest.json`):
```json
{
  "autoencoder": "experiment_runs/automotive/hcrl_sa/unsupervised/vgae/teacher/no_distillation/autoencoder/vgae_autoencoder.pth",
  "classifier": "experiment_runs/automotive/hcrl_sa/supervised/gat/teacher/no_distillation/normal/gat_hcrl_sa_normal.pth"
}
```

### Creating Manifests

```bash
# Auto-generate manifest for fusion
python scripts/generate_manifests.py \
  --dataset hcrl_sa \
  --mode fusion \
  --output jobs/fusion_hcrl_sa_manifest.json

# Curriculum manifest
python scripts/generate_manifests.py \
  --dataset hcrl_sa \
  --mode curriculum \
  --output jobs/curriculum_hcrl_sa_manifest.json
```

### Validating Manifests

```bash
# Validate manifest (check all artifacts exist)
python scripts/validate_artifact.py --manifest jobs/fusion_hcrl_sa_manifest.json

# Dry-run job with manifest
python oscjobmanager.py submit \
  --preset fusion_hcrl_sa \
  --manifest jobs/fusion_hcrl_sa_manifest.json \
  --dry-run
```

### Using Manifests in Jobs

```bash
# Submit with manifest
python oscjobmanager.py submit \
  --preset fusion_hcrl_sa \
  --manifest jobs/fusion_hcrl_sa_manifest.json

# Manifest paths override auto-discovery
```

---

## 4. Job Chaining (Pipeline)

### Sequential Training Pipeline

```bash
# 1. Train VGAE (unsupervised)
JOB1=$(python oscjobmanager.py submit --preset vgae_autoencoder_hcrl_sa | grep -oP 'Job ID: \K\d+')

# 2. Train GAT (supervised) - can run in parallel
JOB2=$(python oscjobmanager.py submit --preset gat_normal_hcrl_sa | grep -oP 'Job ID: \K\d+')

# 3. Fusion - depends on both
python oscjobmanager.py submit \
  --preset fusion_hcrl_sa \
  --dependency afterok:$JOB1:$JOB2

echo "Pipeline submitted: $JOB1 → $JOB2 → Fusion"
```

### Curriculum Pipeline

```bash
# 1. Train VGAE
JOB1=$(python oscjobmanager.py submit --preset vgae_autoencoder_hcrl_sa | grep -oP 'Job ID: \K\d+')

# 2. Curriculum GAT (depends on VGAE)
python oscjobmanager.py submit \
  --preset gat_curriculum_hcrl_sa \
  --dependency afterok:$JOB1
```

### Knowledge Distillation Pipeline

```bash
# 1. Train teacher GAT
JOB1=$(python oscjobmanager.py submit --preset gat_normal_hcrl_sa | grep -oP 'Job ID: \K\d+')

# 2. Distill to student (depends on teacher)
python oscjobmanager.py submit \
  --preset distillation_hcrl_sa_0.5 \
  --dependency afterok:$JOB1 \
  --teacher-path experiment_runs/.../best_teacher_model.pth
```

---

## 5. Sweep Experiments

### Dataset Sweep

```bash
#!/bin/bash
# submit_sweep.sh - Train GAT on all datasets

datasets=("hcrl_sa" "hcrl_ch" "set_01" "set_02" "set_03" "set_04")

for dataset in "${datasets[@]}"; do
    echo "Submitting GAT for $dataset"
    python oscjobmanager.py submit --preset gat_normal_$dataset
    sleep 1  # Avoid hammering scheduler
done
```

### Hyperparameter Sweep

```bash
#!/bin/bash
# Learning rate sweep

for lr in 0.0001 0.0005 0.001 0.005; do
    echo "Submitting with lr=$lr"
    python oscjobmanager.py submit \
      --preset gat_normal_hcrl_sa \
      --override learning_rate=$lr \
      --job-name gat_lr_${lr}
    sleep 1
done
```

---

## 6. Fusion Workflow (Complete)

### Step-by-Step

```bash
# 1. Check prerequisites
python scripts/pre_submit_check.py --mode fusion --dataset hcrl_sa

# 2. Generate manifest
python scripts/generate_manifests.py \
  --dataset hcrl_sa \
  --mode fusion \
  --output jobs/fusion_hcrl_sa_manifest.json

# 3. Validate manifest
python scripts/validate_artifact.py --manifest jobs/fusion_hcrl_sa_manifest.json

# 4. Preview job
python oscjobmanager.py preview \
  --preset fusion_hcrl_sa \
  --manifest jobs/fusion_hcrl_sa_manifest.json

# 5. Submit
python oscjobmanager.py submit \
  --preset fusion_hcrl_sa \
  --manifest jobs/fusion_hcrl_sa_manifest.json \
  --walltime 06:00:00 \
  --memory 48G
```

### Automated Fusion Pipeline

```bash
#!/bin/bash
# auto_fusion.sh - Complete fusion pipeline

DATASET="hcrl_sa"

echo "=== Fusion Pipeline for $DATASET ==="

# Step 1: Train VGAE
echo "1. Training VGAE..."
JOB1=$(python oscjobmanager.py submit --preset vgae_autoencoder_$DATASET | grep -oP 'Job ID: \K\d+')

# Step 2: Train GAT
echo "2. Training GAT..."
JOB2=$(python oscjobmanager.py submit --preset gat_normal_$DATASET | grep -oP 'Job ID: \K\d+')

# Step 3: Generate manifest (after both complete)
echo "3. Will generate manifest after jobs complete..."

# Step 4: Submit fusion (depends on both)
python oscjobmanager.py submit \
  --preset fusion_$DATASET \
  --dependency afterok:$JOB1:$JOB2 \
  --walltime 06:00:00

echo "✓ Pipeline submitted: VGAE($JOB1) + GAT($JOB2) → Fusion"
```

---

## 7. Troubleshooting Jobs

### Job Won't Start

```bash
# Check job queue
squeue -u $USER

# Check job details
scontrol show job <jobid>

# Common issues:
# - Dependency not satisfied
# - Resource limits exceeded
# - Invalid account/partition
```

### Job Failed

```bash
# Check exit code
sacct -j <jobid> --format=JobID,ExitCode,State

# View error log
tail -50 slurm_jobs/job_<jobid>.err

# Common failures:
# - CUDA OOM: Reduce batch size or model size
# - Dataset not found: Check data path
# - Missing artifacts: Validate manifest
```

### Job Running Slow

```bash
# Check resource usage
seff <jobid>

# Monitor GPU
ssh <compute_node>
nvidia-smi

# If underutilizing GPU:
# - Increase batch size
# - Enable mixed precision
# - Check dataloader workers
```

---

## 8. Best Practices

### Pre-Submission Checklist

```bash
# 1. Validate config
python -c "from src.config.hydra_zen_configs import create_gat_normal_config; cfg = create_gat_normal_config('hcrl_sa'); print('✓ Config valid')"

# 2. Check dataset exists
ls -la data/automotive/hcrl_sa/

# 3. For fusion/curriculum: validate artifacts
python scripts/validate_artifact.py --manifest jobs/manifest.json

# 4. Dry run
python oscjobmanager.py submit --preset <preset> --dry-run

# 5. Preview SLURM script
python oscjobmanager.py preview --preset <preset>
```

### Resource Estimation

| Model | Walltime | Memory | GPUs | Batch Size |
|-------|----------|--------|------|------------|
| GAT Teacher | 4h | 32G | 1 | 64 |
| VGAE Teacher | 6h | 32G | 1 | 64 |
| GAT Student | 2h | 16G | 1 | 128 |
| Curriculum | 8h | 48G | 1 | 32 |
| Fusion | 6h | 48G | 1 | 256 |

### Job Naming

```bash
# Good naming (descriptive)
python oscjobmanager.py submit \
  --preset gat_normal_hcrl_sa \
  --job-name gat_hcrl_sa_lr0.001_bs64

# Avoid generic names
--job-name test  # ❌
--job-name job1  # ❌
```

---

## 9. Advanced Patterns

### Conditional Chaining

```bash
#!/bin/bash
# Only run fusion if both prerequisites succeed

JOB1=$(sbatch vgae_job.sh | grep -oP '\d+')
JOB2=$(sbatch gat_job.sh | grep -oP '\d+')

# Fusion runs only if BOTH succeed (afterok)
sbatch --dependency=afterok:$JOB1:$JOB2 fusion_job.sh

# Alternative: Run if ANY succeeds (afterany)
sbatch --dependency=afterany:$JOB1:$JOB2 fusion_job.sh
```

### Array Jobs (Parameter Sweep)

```bash
#!/bin/bash
#SBATCH --array=0-5

# Array of datasets
datasets=("hcrl_sa" "hcrl_ch" "set_01" "set_02" "set_03" "set_04")
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "Training on $dataset"
python train_with_hydra_zen.py --model gat --dataset $dataset --training normal
```

---

## 10. Workflow Templates

### Quick Start (Local Testing)

```bash
# Test locally first
python train_with_hydra_zen.py \
  --model gat \
  --dataset hcrl_sa \
  --training normal \
  --max-epochs 5 \
  --fast-dev-run

# If working, submit to cluster
python oscjobmanager.py submit --preset gat_normal_hcrl_sa
```

### Full Experiment Workflow

```bash
# 1. Train teacher models
./scripts/train_teachers.sh hcrl_sa

# 2. Validate teacher models
python scripts/validate_artifact.py --dataset hcrl_sa

# 3. Train students with distillation
./scripts/train_students.sh hcrl_sa

# 4. Run fusion
python oscjobmanager.py submit --preset fusion_hcrl_sa

# 5. Collect results
python scripts/collect_summaries.py --dataset hcrl_sa
```

---

## Summary

**Key Files**:
- `oscjobmanager.py` - Job submission
- `scripts/generate_manifests.py` - Create dependency manifests
- `scripts/validate_artifact.py` - Validate required models
- `scripts/pre_submit_check.py` - Pre-flight checks

**Key Commands**:
- `python oscjobmanager.py submit --preset <name>` - Submit job
- `python oscjobmanager.py preview --preset <name>` - Preview configuration
- `squeue -u $USER` - Check job status
- `scancel <jobid>` - Cancel job

**Best Practices**:
1. Always dry-run first
2. Validate manifests before submission
3. Use descriptive job names
4. Chain dependencies for pipelines
5. Monitor resource usage with `seff`

See [JOB_TEMPLATES.md](JOB_TEMPLATES.md) for complete preset reference.
