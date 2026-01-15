# CAN-Graph Training System Guide

## Main Access Point
**File:** `train_with_hydra_zen.py`
- Primary training interface using hydra-zen configurations
- Supports: GAT, VGAE, Knowledge Distillation, Fusion training
- Auto-logging with CSV, MLflow, optional TensorBoard

## MLflow Dashboard Access
```bash
# After training, view results:
mlflow ui --backend-store-uri file://./outputs/mlruns
# Open: http://localhost:5000
```

## Configuration System

### Dataset Configurations (`conf/dataset/`)
- `hcrl_sa.yaml` - HCRL steering angle dataset
- `hcrl_ch.yaml` - HCRL car hacking dataset  
- `set_01.yaml` to `set_04.yaml` - Complex imbalanced datasets
- `can_universal.yaml` - Base template

### Model Configurations (`conf/model/`)
- `gat.yaml` - Graph Attention Network settings
- `vgae.yaml` - Variational Graph Autoencoder settings

### Main Config (`conf/config.yaml`)
- Training parameters, paths, logging settings
- HPC optimized version: `hpc_optimized.yaml`

## Core Python Files

### Essential Files (DO NOT REMOVE)
1. **`train_with_hydra_zen.py`** - Main training script
2. **`train_models.py`** - PyTorch Lightning module
3. **`src/config/hydra_zen_configs.py`** - Type-safe configuration system
4. **`src/training/fusion_lightning.py`** - Fusion training module
5. **`src/training/prediction_cache.py`** - Fusion prediction caching
6. **`src/models/models.py`** - Model definitions (GATWithJK, GraphAutoencoderNeighborhood)

### Utility Files (EVALUATE FOR REMOVAL)
- **`src/utils/gpu_optimization.py`** - ❓ REDUNDANT (Lightning handles GPU)
- **`src/utils/legacy_compatibility.py`** - ❓ STOPGAP (remove after migration)
- **`src/training/gpu_monitor.py`** - ❓ REDUNDANT (Lightning profiler better)
- **`src/utils/utils_logging.py`** - ❓ PARTIALLY REDUNDANT (some functions used)

### Analysis Files (REMOVABLE)
- **`src/evaluation/evaluation.py`** - ❓ Can be external script
- **`src/utils/plotting_utils.py`** - ❓ Can be external script
- **`src/utils/visualization.py`** - ❓ Can be external script

## Training Options

### Basic Training
```bash
# GAT normal training
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal

# VGAE autoencoder training  
python train_with_hydra_zen.py --model vgae --dataset set_01 --training autoencoder
```

### Advanced Training
```bash
# Knowledge distillation
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training knowledge_distillation --teacher_path saved_models/teacher.pth

# Fusion training (requires pre-trained models)
python train_with_hydra_zen.py --dataset hcrl_sa --training fusion
```

### OSC Batch Submission
```bash
# Submit complex datasets with 8h walltime
python osc_job_manager.py --submit-complex --datasets set_01,set_02,set_03,set_04

# Submit individual jobs (2h walltime)  
python osc_job_manager.py --submit-individual --datasets hcrl_sa,hcrl_ch

# Submit with custom training parameters (use argument names from train_with_hydra_zen.py)
python osc_job_manager.py --submit-individual --datasets set_01 --training individual_vgae --extra-args "epochs=200"

# Multiple parameters (comma-separated)
python osc_job_manager.py --submit-individual --datasets set_01 --training individual_gat --extra-args "epochs=200,batch_size=512,learning_rate=0.0005"

# Monitor jobs
python osc_job_manager.py --monitor
```

**Important:** Use argument names from `train_with_hydra_zen.py --help`:
- `epochs` (not `training.max_epochs` or `max_epochs`)
- `batch_size` 
- `learning_rate`
- `tensorboard` (flag, use `tensorboard=true`)
- `teacher_path`, `student_scale`, `distillation_alpha`, `temperature` (for distillation)

## Configuration Parameters

### Training Parameters
- `max_epochs`: Training duration (default: 100)
- `learning_rate`: Optimizer learning rate (default: 0.001)
- `batch_size`: Training batch size (default: 256)
- `weight_decay`: L2 regularization (default: 1e-5)

### Model Parameters (GAT)
- `num_layers`: Number of GAT layers (default: 3)
- `hidden_dim`: Hidden layer size (default: 256)
- `num_heads`: Attention heads (default: 8)
- `dropout`: Dropout rate (default: 0.2)

### Model Parameters (VGAE)
- `encoder_layers`: Encoder depth (default: [512, 256])
- `latent_dim`: Latent space size (default: 128)
- `decoder_layers`: Decoder depth (default: [256, 512])

### Complex Dataset Settings
- Automatic 8-hour walltime vs 2-hour standard
- 64GB memory vs 32GB standard
- Class imbalance handling (focal loss recommended)

### Logging Options
- `enable_csv`: Always enabled (Lightning logs)
- `enable_mlflow`: Always enabled (experiment tracking)
- `enable_tensorboard`: Optional (set to true for detailed metrics)

## File Interactions

### Training Flow
1. **`train_with_hydra_zen.py`** loads config from **`src/config/hydra_zen_configs.py`**
2. Instantiates **`train_models.py`** Lightning module
3. For fusion: uses **`src/training/fusion_lightning.py`**
4. Models defined in **`src/models/models.py`**
5. Results logged to `outputs/` (CSV, MLflow, checkpoints)

### Job Submission Flow
1. **`osc_job_manager.py`** (root level) generates SLURM scripts
2. Calls **`train_with_hydra_zen.py`** on compute nodes
3. Results stored in `osc_jobs/{job_name}/`
4. Models saved to `saved_models/`

This system provides type-safe configuration, automatic experiment tracking, and scalable HPC deployment.