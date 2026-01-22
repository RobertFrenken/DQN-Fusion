# CAN-Graph Training Job Templates & Configuration Reference

This file is your **one-stop lookup** for all CAN-Graph training configurations, including hydra-zen parameters, SLURM settings, and ready-to-use job commands.

---

## 📋 **Complete Configuration Reference**

### Model Architecture Parameters

#### GAT (Teacher) - 1.1M Parameters
```yaml
type: gat
input_dim: 11                 # CAN features (ID + 8 bytes + count + position)
embedding_dim: 32             # CAN ID embedding dimension (rich for teacher)
hidden_channels: 64           # Hidden layer size
num_layers: 5                 # Number of GAT layers (matches paper)
heads: 8                      # Attention heads per layer (matches paper)
output_dim: 2                 # Binary classification (normal/attack)
dropout: 0.2                  # Dropout rate
num_fc_layers: 3              # Fully connected layers after GAT
use_jumping_knowledge: true   # JumpingKnowledge connections
jk_mode: "cat"               # JK aggregation mode [cat, max, lstm]
use_residual: true           # Residual connections
use_batch_norm: false        # Batch normalization
activation: "relu"           # Activation function
```

#### Student GAT - 55K Parameters
```yaml
type: gat_student
input_dim: 11                 # Same as teacher
embedding_dim: 8              # Smaller embedding (matches paper)
hidden_channels: 32           # Smaller hidden size for deployment
num_layers: 2                 # Fewer layers (matches paper)
heads: 4                      # Fewer attention heads (matches paper)
output_dim: 2                 # Binary classification
dropout: 0.1                  # Lower dropout
num_fc_layers: 2              # Simpler classification head
use_jumping_knowledge: false  # Simpler architecture
use_residual: false          # Simpler for deployment
```

#### VGAE (Teacher) - 1.74M Parameters
```yaml
type: vgae
input_dim: 11                 # CAN features input
node_embedding_dim: 256       # Rich embedding for teacher
hidden_dims: [256, 128, 96, 48]  # Multi-layer compression path
latent_dim: 48                # Large latent space
output_dim: 11                # Reconstruct input features
num_layers: 3                 # Deep encoder/decoder
attention_heads: 8            # Multi-head attention
dropout: 0.15                 # Higher dropout for regularization
batch_norm: true             # Batch normalization
activation: "relu"           # Activation function
beta: 1.0                    # KL divergence weight
target_parameters: 1740000    # Parameter target
```

#### Student VGAE - 87K Parameters
```yaml
type: vgae_student
input_dim: 11                 # CAN features input
node_embedding_dim: 128       # Rich initial embedding
encoder_dims: [128, 64, 24]   # Proper compression: 128 → 64 → 24
decoder_dims: [24, 64, 128]   # Proper decompression: 24 → 64 → 128
latent_dim: 24                # Compact latent space
output_dim: 11                # Reconstruct input features
attention_heads: 2            # Lightweight attention
dropout: 0.1                  # Lower dropout
batch_norm: true             # Batch normalization
activation: "relu"           # Activation function
target_parameters: 87000      # ~87K params for deployment
memory_budget_kb: 287         # 87KB model + 200KB buffer
inference_time_ms: 5          # <5ms per CAN message
```

### Training Configuration Parameters

#### Normal Training
```yaml
type: normal
max_epochs: 200               # Maximum training epochs
learning_rate: 1e-3          # Base learning rate
batch_size: "auto"           # Auto-optimize batch size
early_stopping_patience: 25  # Early stopping patience
validation_split: 0.2        # Validation data percentage
accumulate_grad_batches: 1   # Gradient accumulation steps
gradient_clip_val: 1.0       # Gradient clipping threshold
precision: "32-true"         # Training precision [32-true, 16-mixed, bf16-mixed]
optimizer_name: "adam"       # Optimizer [adam, sgd, adamw]
weight_decay: 1e-4           # Weight decay
momentum: 0.9                # SGD momentum
scheduler_use: false         # Learning rate scheduler
scheduler_type: "cosine"     # Scheduler type [cosine, step, exponential]
```

#### Autoencoder Training
```yaml
type: autoencoder
max_epochs: 100               # Fewer epochs for reconstruction
learning_rate: 1e-3          # Learning rate
batch_size: "auto"           # Auto-optimize batch size
reconstruction_weight: 1.0   # Reconstruction loss weight
kl_weight: 0.01              # KL divergence weight (for VGAE)
validation_split: 0.2        # Validation split
early_stopping_patience: 15  # Early stopping patience
```

#### Knowledge Distillation Training
```yaml
type: knowledge_distillation
max_epochs: 150               # Moderate epochs for distillation
teacher_model_path: null      # Auto-loaded from model archive
student_model_config: null    # Student config (auto-determined)
temperature: 4.0              # Distillation temperature
alpha: 0.7                    # Balance between KD and CE loss
learning_rate: 1e-3          # Student learning rate
batch_size: "auto"           # Auto-optimize batch size
validation_split: 0.2        # Validation split
early_stopping_patience: 20  # Early stopping patience
```

#### Curriculum Learning
```yaml
type: curriculum
max_epochs: 200               # Full curriculum training
learning_rate: 1e-3          # Base learning rate
batch_size: "auto"           # Auto-optimize batch size
curriculum_stages: 4          # Number of difficulty stages
stage_epochs: [40, 40, 60, 60] # Epochs per stage
difficulty_metric: "confidence" # Curriculum ordering metric
validation_split: 0.2        # Validation split
early_stopping_patience: 30  # Longer patience for curriculum
```

#### Multi-Modal Fusion Training
```yaml
type: fusion
max_epochs: 250               # Extended training for fusion
learning_rate: 1e-4          # Lower LR for stability
batch_size: "auto"           # Auto-optimize batch size
fusion_strategy: "attention"  # Fusion method [attention, concat, gating]
modality_weights: [0.6, 0.4] # Weight per modality
validation_split: 0.2        # Validation split
early_stopping_patience: 35  # Extended patience
```

### Teacher-Student Model Configuration

#### Teacher Model Parameters
```yaml
model_type: "teacher"          # Explicitly set as teacher model
use_teacher_config: true       # Enable teacher architecture
teacher_model_path: null       # Path to pre-trained teacher (if loading)
save_teacher_model: true       # Save teacher model after training
teacher_checkpoint_dir: "osc_jobs/{dataset}/gat/normal/"  # Teacher model save directory (recommended: use oscillator per-job paths)
```

#### Student Model Parameters
```yaml
model_type: "student"          # Explicitly set as student model
use_student_config: true       # Enable student architecture
student_model_path: null       # Path to pre-trained student (if loading)
teacher_model_path: "osc_jobs/{dataset}/gat/normal/best_teacher_model_{dataset}.pth"  # Required teacher path (preferred)
save_student_model: true       # Save student model after training
student_checkpoint_dir: "saved_models/"  # Student model save directory (compatibility links)
```

#### Knowledge Distillation Specific Parameters
```yaml
# Core KD Parameters
distillation_type: "soft"      # Distillation type [soft, hard, hybrid]
temperature: 4.0               # Softmax temperature for KD
alpha: 0.7                     # Weight for distillation loss (0.0-1.0)
beta: 0.3                      # Weight for student loss (1.0-alpha)

# Advanced KD Parameters
kd_loss_type: "kl_div"         # KD loss function [kl_div, mse, cosine]
feature_matching: false        # Enable intermediate feature matching
feature_match_layers: [2, 4]   # Layers for feature matching
feature_match_weight: 0.1      # Weight for feature matching loss

# Attention Transfer (for GAT models)
attention_transfer: true       # Enable attention map transfer
attention_weight: 0.05         # Weight for attention transfer loss
attention_layers: "all"        # Layers for attention transfer [all, last, list]

# Progressive Knowledge Distillation
progressive_kd: false          # Enable progressive difficulty
progressive_stages: 3          # Number of progressive stages
stage_temperature: [6.0, 4.0, 2.0]  # Temperature schedule per stage
```

### Hydra-Zen Configuration Parameters

#### Model Selection (Hydra-Zen)
```yaml
# Use hydra-zen configs instead of YAML
use_hydra_zen: true            # Enable hydra-zen configuration system
config_store: "model_configs"  # Config store name

# Model Architecture Selection
model_config: "GATConfig"      # Teacher GAT config class
# model_config: "StudentGATConfig"     # Student GAT config class  
# model_config: "VGAEConfig"           # Teacher VGAE config class
# model_config: "StudentVGAEConfig"    # Student VGAE config class

# Training Configuration Selection
training_config: "normal"      # Training mode config
# training_config: "knowledge_distillation"  # KD training config
# training_config: "curriculum"              # Curriculum learning config
# training_config: "fusion"                  # Multi-modal fusion config

# Dataset Configuration Selection
dataset_config: "hcrl_sa"      # Dataset-specific config
# dataset_config: "hcrl_ch"    # Alternative dataset configs
# dataset_config: "set_01"     # Complex dataset configs
```

#### Hydra-Zen Override Parameters
```yaml
# Model Architecture Overrides
overrides:
  model:
    input_dim: 11              # Override input dimensions
    hidden_channels: 64        # Override hidden layer size
    num_layers: 5              # Override number of layers
    heads: 8                   # Override attention heads
    dropout: 0.2               # Override dropout rate
    
  training:
    max_epochs: 200            # Override training epochs
    learning_rate: 1e-3        # Override learning rate
    batch_size: "auto"         # Override batch size
    
  knowledge_distillation:
    temperature: 4.0           # Override KD temperature
    alpha: 0.7                 # Override KD loss weight
    teacher_model_path: "osc_jobs/hcrl_sa/gat/normal/best_teacher_model_hcrl_sa.pth"
```

### Default Training Parameters
| Parameter | VGAE Autoencoder | GAT Normal | Knowledge Distillation | GAT Curriculum | DQN Fusion | Override Syntax |
|-----------|------------------|------------|------------------------|----------------|------------|------------------|
| **Epochs** | 100 | 200 | 150 | 200 | 100 | `--extra-args "epochs=500"` |
| **Learning Rate** | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | `--extra-args "learning_rate=0.001"` |
| **Batch Size** | Auto-optimized | Auto-optimized | Auto-optimized | Auto-optimized | Auto-optimized | `--extra-args "batch_size=512"` |
| **Temperature** | N/A | N/A | 4.0 | N/A | N/A | `--extra-args "temperature=6.0"` |
| **Alpha (KD Weight)** | N/A | N/A | 0.7 | N/A | N/A | `--extra-args "alpha=0.8"` |
| **Early Stopping** | 25 epochs | 25 epochs | 20 epochs | 30 epochs | 15 epochs | `--extra-args "early_stopping_patience=30"` |
| **Precision** | 32-bit | 32-bit | 32-bit | 32-bit | 32-bit | `--extra-args "precision=16-mixed"` |

### Resource Allocation (Unified Configuration)
| Training Type | Wall Time | Memory | GPUs | Override Syntax |
|---------------|-----------|--------|------|-----------------|
| **VGAE Autoencoder** | 8:00:00 | 64GB | 1 | `--extra-args "time_limit=4:00:00"` |
| **GAT Normal** | 8:00:00 | 64GB | 1 | `--extra-args "memory=48G"` |
| **Knowledge Distillation** | 8:00:00 | 64GB | 1 | `--extra-args "temperature=6.0,alpha=0.8"` |
| **GAT Curriculum** | 8:00:00 | 64GB | 1 | `--extra-args "cpus=12"` |
| **DQN Fusion** | 8:00:00 | 64GB | 1 | `--extra-args "gpus=2"` |

**All Datasets**: `hcrl_sa`, `hcrl_ch`, `set_01`, `set_02`, `set_03`, `set_04` (8 hour wall time, 64GB memory)

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
- All datasets use unified resource allocation: **8 hour wall time, 64GB memory**
- Datasets: `hcrl_sa`, `hcrl_ch`, `set_01`, `set_02`, `set_03`, `set_04`

### Training Types & Resources
- **All Training Types**: 8h wall time, 64GB memory (unified configuration)
- **VGAE Autoencoder**: Graph autoencoder training
- **GAT Normal**: Standard GAT training  
- **GAT Curriculum**: GAT with curriculum learning (requires VGAE)
- **DQN Fusion**: Multi-modal fusion training with DQN agent (requires GAT + VGAE)

---

## 🎯 **Job Templates**

### 1. Teacher Model Training (Full Architecture)

```bash
# Train GAT teacher model for hcrl_sa (2 hours, 32GB)
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training gat_normal --extra-args "model_type=teacher,use_teacher_config=true"

# Train VGAE teacher model for set_01 (8 hours, 64GB - complex dataset)
python osc_job_manager.py --submit-individual --datasets set_01 --training vgae_autoencoder --extra-args "model_type=teacher,use_teacher_config=true"

# Train teacher models for all datasets
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04 --training gat_normal --extra-args "model_type=teacher"
```

### 2. Student Model Training (Knowledge Distillation)

```bash
# Train GAT student with knowledge distillation from teacher
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training knowledge_distillation --extra-args "model_type=student,teacher_model_path=osc_jobs/hcrl_sa/gat/normal/best_teacher_model_hcrl_sa.pth,temperature=4.0,alpha=0.7"

# Train VGAE student with knowledge distillation
python osc_job_manager.py --submit-individual --datasets set_01 --training knowledge_distillation --extra-args "model_type=student,model_config=StudentVGAEConfig,teacher_model_path=osc_jobs/set_01/gat/normal/best_teacher_model_set_01.pth"

# Knowledge distillation for all datasets (requires existing teacher models)
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch,set_01,set_02,set_03,set_04 --training knowledge_distillation --extra-args "model_type=student,temperature=4.0,alpha=0.7"
```

### 3. Hydra-Zen Configuration Examples

```bash
# Use hydra-zen GAT teacher config
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training gat_normal --extra-args "use_hydra_zen=true,model_config=GATConfig,model_type=teacher"

# Use hydra-zen GAT student config with KD
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training knowledge_distillation --extra-args "use_hydra_zen=true,model_config=StudentGATConfig,model_type=student,teacher_model_path=osc_jobs/hcrl_sa/gat/normal/best_teacher_model_hcrl_sa.pth"

# Use hydra-zen VGAE teacher config
python osc_job_manager.py --submit-individual --datasets set_01 --training vgae_autoencoder --extra-args "use_hydra_zen=true,model_config=VGAEConfig,model_type=teacher"

# Use hydra-zen VGAE student config with KD
python osc_job_manager.py --submit-individual --datasets set_01 --training knowledge_distillation --extra-args "use_hydra_zen=true,model_config=StudentVGAEConfig,model_type=student,teacher_model_path=osc_jobs/set_01/gat/normal/best_teacher_model_set_01.pth"
```

### 4. Single Model for Single Dataset

```bash
# VGAE autoencoder for hcrl_sa (2 hours, 32GB)
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training vgae_autoencoder

# GAT normal training for set_01 (8 hours, 64GB - complex dataset)
python osc_job_manager.py --submit-individual --datasets set_01 --training gat_normal

# GAT curriculum learning for hcrl_ch (4 hours, 48GB)
python osc_job_manager.py --submit-individual --datasets hcrl_ch --training gat_curriculum

# DQN fusion for hcrl_sa (3 hours, 48GB) - requires existing GAT+VGAE models
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training dqn_normal
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

### 5. DQN Fusion Model for Single Dataset

**⚠️ IMPORTANT: Requires existing GAT and VGAE models**

```bash
# DQN fusion for hcrl_sa (3 hours, 48GB)
python osc_job_manager.py --submit-individual --datasets hcrl_sa --training dqn_normal

# DQN fusion for set_01 (3 hours, 48GB)
python osc_job_manager.py --submit-individual --datasets set_01 --training dqn_normal

# DQN fusion for hcrl_ch (3 hours, 48GB)
python osc_job_manager.py --submit-individual --datasets hcrl_ch --training dqn_normal
```

### 6. DQN Fusion Model for All Datasets

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
│   │   └── dqn/             # DQN fusion training outputs
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
| DQN Fusion | 3:00:00 | 3:00:00 | 48G | 8 | 1 |

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