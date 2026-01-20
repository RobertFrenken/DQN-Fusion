# CAN-Graph Training Job Templates

This file contains copy-paste ready commands for running different training configurations using the OSC job manager.

## � **Important Parameters Quick Reference**

### Default Training Parameters
| Parameter | VGAE Autoencoder | GAT Normal | GAT Curriculum | GAT Fusion | Override Syntax |
|-----------|------------------|------------|----------------|------------|-----------------|
| **Max Epochs** | 100 | 200 | 150 | 100 | `--extra-args "max_epochs=NUMBER"` |
| **Learning Rate** | 1e-3 | 1e-3 | 1e-3 | 1e-3 | `--extra-args "learning_rate=0.001"` |
| **Batch Size** | Auto-optimized | Auto-optimized | Auto-optimized | Auto-optimized | `--extra-args "batch_size=512"` |
| **Early Stopping** | 25 epochs | 25 epochs | 25 epochs | 15 epochs | `--extra-args "early_stopping_patience=30"` |
| **Precision** | 32-bit | 32-bit | 32-bit | 32-bit | `--extra-args "precision=16-mixed"` |

### Resource Allocation
| Training Type | Wall Time (Std) | Wall Time (Complex) | Memory (Std) | Memory (Complex) | GPUs | Override Syntax |
|---------------|-----------------|---------------------|--------------|------------------|------|-----------------|
| **VGAE Autoencoder** | 2:00:00 | 8:00:00 | 32GB | 64GB | 1 | `--extra-args "time_limit=4:00:00"` |
| **GAT Normal** | 2:00:00 | 8:00:00 | 32GB | 64GB | 1 | `--extra-args "memory=48G"` |
| **GAT Curriculum** | 4:00:00 | 12:00:00 | 48GB | 80GB | 1 | `--extra-args "cpus=12"` |
| **GAT Fusion** | 3:00:00 | 3:00:00 | 48GB | 48GB | 1 | `--extra-args "gpus=2"` |

**Standard Datasets**: `hcrl_sa`, `hcrl_ch` | **Complex Datasets**: `set_01`, `set_02`, `set_03`, `set_04`

### Key Features & Parameters
| Feature | Description | Override Syntax |
|---------|-------------|-----------------|
| **Auto Batch Size Optimization** | Automatically finds optimal batch size for your GPU | `--extra-args "batch_size=1024"` (to disable auto) |
| **Gradient Clipping** | Enabled (value: 1.0) | `--extra-args "gradient_clip_val=0.5"` |
| **Mixed Precision** | Available (use for faster training) | `--extra-args "precision=16-mixed"` |
| **MLflow Logging** | Automatic experiment tracking per dataset | `--extra-args "disable_mlflow=true"` |
| **Checkpointing** | Top 3 models saved + last checkpoint | `--extra-args "save_top_k=5"` |
| **Validation Frequency** | Every epoch | `--extra-args "check_val_every_n_epoch=2"` |
| **Log Every N Steps** | Every 50 steps | `--extra-args "log_every_n_steps=100"` |

## �📋 **Quick Reference**

### Available Datasets
- `hcrl_sa`, `hcrl_ch` (Standard datasets - 2 hour wall time, 32GB memory)
- `set_01`, `set_02`, `set_03`, `set_04` (Complex datasets - extended resources)

### Training Types & Resources
- **VGAE Autoencoder**: 2h/8h wall time, 32GB/64GB memory
- **GAT Normal**: 2h/8h wall time, 32GB/64GB memory  
- **GAT Curriculum**: 4h/12h wall time, 48GB/80GB memory
- **GAT Fusion**: 3h wall time, 48GB memory

---

## 🎯 **Job Templates**

### 1. Single Model for Single Dataset

```bash
# VGAE autoencoder for hcrl_sa (2 hours, 32GB)
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training vgae_autoencoder

# GAT normal training for set_01 (8 hours, 64GB - complex dataset)
python osc_job_manager.py --submit-individual --datasets set_01 --training gat_normal

# GAT curriculum learning for hcrl_ch (4 hours, 48GB)
python osc_job_manager.py --submit-individual --datasets hcrl_ch --training gat_curriculum

# GAT fusion for hcrl_sa (3 hours, 48GB) - requires existing GAT+VGAE models
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training gat_fusion
```

### 2. VGAE Autoencoder for All Datasets

```bash
# All datasets (2h for standard, 8h for complex datasets)
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04 --training vgae_autoencoder

# Standard datasets only (2 hours, 32GB each)
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch --training vgae_autoencoder

# Complex datasets only (8 hours, 64GB each)
python osc_job_manager.py --submit-individual --datasets set_01,set_02,set_03,set_04 --training vgae_autoencoder
```

### 3. GAT Normal Training for All Datasets

```bash
# All datasets (2h for standard, 8h for complex datasets)
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04 --training gat_normal

# Standard datasets only (2 hours, 32GB each)
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch --training gat_normal

# Complex datasets only (8 hours, 64GB each)
python osc_job_manager.py --submit-individual --datasets set_01,set_02,set_03,set_04 --training gat_normal
```

### 4. GAT Curriculum Learning for All Datasets

**⚠️ IMPORTANT: Requires existing VGAE models for hard mining**

```bash
# All datasets (4h for standard, 12h for complex datasets)
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04 --training gat_curriculum

# Standard datasets only (4 hours, 48GB each)
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch --training gat_curriculum

# Complex datasets only (12 hours, 80GB each)
python osc_job_manager.py --submit-individual --datasets set_01,set_02,set_03,set_04 --training gat_curriculum
```

### 5. Fusion Model for Single Dataset

**⚠️ IMPORTANT: Requires existing GAT and VGAE models**

```bash
# Fusion for hcrl_sa (3 hours, 48GB)
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training gat_fusion

# Fusion for set_01 (3 hours, 48GB)
python osc_job_manager.py --submit-individual --datasets set_01 --training gat_fusion

# Fusion for hcrl_ch (3 hours, 48GB)
python osc_job_manager.py --submit-individual --datasets hcrl_ch --training gat_fusion
```

### 6. Fusion Model for All Datasets

**⚠️ IMPORTANT: Requires existing GAT and VGAE models for all datasets**

```bash
# All datasets (3 hours, 48GB each)
python osc_job_manager.py --submit-fusion --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04

# Standard datasets only
python osc_job_manager.py --submit-fusion --datasets hcrl_sa,hcrl_ch

# Complex datasets only  
python osc_job_manager.py --submit-fusion --datasets set_01,set_02,set_03,set_04
```

---

## 🏗️ **Pipeline Jobs (Automated Dependencies)**

### Complete Pipeline for Single Dataset
```bash
# Automatically runs: GAT → VGAE → Curriculum → Fusion (with proper dependencies)
python osc_job_manager.py --submit-pipeline --datasets hcrl_sa
```

### Complete Pipeline for All Datasets
```bash
# Runs complete pipeline for each dataset
python osc_job_manager.py --submit-pipeline --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04
```

---

## 📊 **Job Monitoring & Management**

### Monitor Running Jobs
```bash
# Check job status
python osc_job_manager.py --monitor-jobs

# Check specific jobs in SLURM queue
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>
```

### Job Output Locations
```
osc_jobs/
├── {dataset}/
│   ├── gat/
│   │   ├── normal/          # GAT normal training outputs
│   │   ├── curriculum/      # GAT curriculum training outputs  
│   │   └── fusion/          # GAT fusion training outputs
│   └── vgae/
│       └── autoencoder/     # VGAE autoencoder outputs
```

---

## ⚙️ **Advanced Parameter Customization**

### Override Default Parameters
```bash
# Custom epochs (default: 100-200 depending on model)
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training vgae_autoencoder --extra-args "max_epochs=50"

# Custom learning rate (default: 1e-3)
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training gat_normal --extra-args "learning_rate=5e-4"

# Custom early stopping patience (default: 25 epochs)
python osc_job_manager.py --submit-individual --datasets set_01 --training gat_curriculum --extra-args "early_stopping_patience=50"

# Multiple custom parameters
python osc_job_manager.py --submit-individual --datasets hcrl_ch --training vgae_autoencoder --extra-args "max_epochs=150,learning_rate=1e-4,batch_size=512"

# Enable mixed precision for faster training
python osc_job_manager.py --submit-individual --datasets set_01 --training gat_normal --extra-args "precision=16-mixed"
```

### Memory & Performance Optimization
```bash
# Force specific batch size (bypasses auto-optimization)
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training vgae_autoencoder --extra-args "batch_size=1024"

# Increase gradient accumulation for large effective batch sizes
python osc_job_manager.py --submit-individual --datasets set_01 --training gat_curriculum --extra-args "accumulate_grad_batches=4"

# Reduce precision for memory-constrained jobs
python osc_job_manager.py --submit-individual --datasets set_04 --training gat_normal --extra-args "precision=16-mixed"
```

---

## ⚙️ **Resource Allocation Details**

| Training Type | Standard Datasets | Complex Datasets | Memory | CPUs | GPUs |
|---------------|------------------|------------------|---------|------|------|
| VGAE Autoencoder | 2:00:00 | 8:00:00 | 32G/64G | 8 | 1 |
| GAT Normal | 2:00:00 | 8:00:00 | 32G/64G | 8 | 1 |
| GAT Curriculum | 4:00:00 | 12:00:00 | 48G/80G | 8 | 1 |
| GAT Fusion | 3:00:00 | 3:00:00 | 48G | 8 | 1 |

**Standard Datasets**: `hcrl_sa`, `hcrl_ch`  
**Complex Datasets**: `set_01`, `set_02`, `set_03`, `set_04`

---

## 🚀 **Recommended Training Order**

### Option 1: Sequential Training
```bash
# Step 1: Train VGAE autoencoders first (needed for curriculum & fusion)
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04 --training vgae_autoencoder

# Step 2: Train GAT models (can run in parallel with VGAE)
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04 --training gat_normal

# Step 3: Wait for VGAE completion, then train curriculum models
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04 --training gat_curriculum

# Step 4: Wait for GAT+VGAE completion, then train fusion models
python osc_job_manager.py --submit-fusion --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04
```

### Option 2: Automated Pipeline
```bash
# Let the job manager handle dependencies automatically
python osc_job_manager.py --submit-pipeline --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04
```

---

## 📝 **Notes**

- All jobs save outputs to hierarchical directories: `osc_jobs/{dataset}/{model}/{mode}/`
- MLflow tracking is dataset-specific: each dataset gets its own experiment space
- Complex datasets automatically get extended wall time and memory
- Job scripts are saved in the `osc_jobs/` directory for reference
- Use `--monitor-jobs` to track progress across all submitted jobs