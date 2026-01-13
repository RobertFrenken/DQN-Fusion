# Lightning Fabric Training Workflow Guide 🚀

## 1. Recommended Lightning Optional Tools ⚡

### **Highly Recommended** ✅

#### **A. Tuner (Batch Size Optimization)**
```python
# Lightning automatically finds optimal batch size
trainer.find_optimal_batch_size(model, sample_dataloader)
```
- **Why**: Prevents OOM errors and maximizes GPU utilization
- **When**: Before full training, especially for new datasets/models
- **Benefit**: Can improve training speed by 2-3x

#### **B. Profiler (Performance Analysis)**
```python
# Enable profiler in config
config = {
    'enable_profiler': True,
    'profiler_type': 'pytorch'  # or 'simple'
}
```
- **Why**: Identifies bottlenecks in training pipeline
- **When**: During development and optimization phases
- **Benefit**: Shows exactly where time is spent (data loading, forward pass, etc.)

#### **C. Strategy Auto-Selection**
```python
# Lightning chooses best strategy
fabric_config = {
    'strategy': 'auto',  # DDP, DeepSpeed, etc.
    'devices': 'auto'
}
```
- **Why**: Optimal multi-GPU/multi-node scaling
- **When**: Always for multi-device training
- **Benefit**: Handles complex distributed training automatically

### **Conditionally Useful** ⚠️

#### **D. Learning Rate Finder**
```python
# Find optimal learning rate
lr_finder = trainer.find_learning_rate(model, dataloader)
```
- **When**: For new model architectures or datasets
- **Benefit**: Prevents training instability

#### **E. Memory Profiler**
```python
# Profile memory usage
config['profile_memory'] = True
```
- **When**: Debugging OOM issues or optimizing memory usage
- **Benefit**: Shows memory allocation patterns

## 2. Training Workflow Steps 📋

### **Step 1: Configuration Setup**
```
Files Used:
├── conf/fabric.yaml          # Lightning Fabric configuration
├── conf/base.yaml           # Model/training hyperparameters  
└── conf/default.yaml        # Dataset-specific configs
```

### **Step 2: Data Preparation**
```
Files Used:
├── src/preprocessing/preprocessing.py  # Graph creation
├── src/utils/fabric_dataloader.py     # Lightning-optimized dataloaders
└── datasets/                          # Raw data files
```

**Process:**
1. Load raw CAN data
2. Create graph representations using `graph_creation()`
3. Setup Lightning-optimized DataLoaders
4. Automatic worker/memory optimization by Lightning

### **Step 3: Model Initialization**
```
Files Used:
├── src/models/models.py              # GAT, VGAE model definitions
├── src/training/fabric_gat_trainer.py   # GAT Fabric trainer
├── src/training/fabric_vgae_trainer.py  # VGAE Fabric trainer
└── src/utils/fabric_utils.py         # Base trainer utilities
```

### **Step 4: Hardware Optimization** (Automatic)
```python
# Lightning handles automatically:
- GPU/CPU detection
- Memory allocation  
- Multi-device setup
- Mixed precision
- Optimal batch size finding
```

### **Step 5: Training Execution**
```
Main Script: train_fabric_models.py

Training Flow:
1. Load configuration (Hydra)
2. Initialize Fabric with auto-config
3. Setup model and optimizer
4. Optimize batch size (if enabled)
5. Run training loop with:
   - Automatic mixed precision
   - Gradient accumulation
   - Checkpointing
   - Logging
```

### **Step 6: Outputs & Monitoring**
```
Output Structure:
├── outputs/
│   ├── fabric_logs/         # Training logs & metrics
│   ├── figures/            # Generated plots
│   └── profiler_logs/      # Performance profiles (if enabled)
├── saved_models/           # Model checkpoints
└── checkpoints/           # Training state checkpoints
```

## 3. Complete Training Command Examples 💻

### **Basic Training**
```bash
# Teacher model training with auto-optimization
python train_fabric_models.py --model gat --dataset hcrl_sa --type teacher

# Student model with knowledge distillation  
python train_fabric_models.py --model vgae --dataset set_01 --type student --teacher saved_models/best_teacher_model_hcrl_sa.pth
```

### **With Optimization Tools**
```bash
# Enable batch size tuning and profiling
python train_fabric_models.py \
  --model gat \
  --dataset hcrl_ch \
  --type teacher \
  --optimize-batch-size \
  --enable-profiler
```

### **SLURM Cluster Training**
```bash
# Automatic SLURM integration
python train_fabric_models.py \
  --model vgae \
  --dataset set_02 \
  --type teacher \
  --use-slurm \
  --nodes 2 \
  --gpus-per-node 4
```

## 4. Key Files in Training Pipeline 📁

### **Core Training Files**
| File | Purpose | Lightning Integration |
|------|---------|----------------------|
| `train_fabric_models.py` | Main training script | Entry point, CLI interface |
| `fabric_gat_trainer.py` | GAT model trainer | Fabric training loops |
| `fabric_vgae_trainer.py` | VGAE model trainer | Fabric training loops |
| `fabric_utils.py` | Base trainer utilities | Fabric setup, checkpointing |
| `fabric_dataloader.py` | Data loading optimization | Lightning DataLoader optimization |

### **Configuration Files**
| File | Purpose | Key Settings |
|------|---------|-------------|
| `conf/fabric.yaml` | Lightning Fabric config | `accelerator: auto`, `precision: auto` |
| `conf/base.yaml` | Model hyperparameters | Architecture, learning rates |
| `conf/default.yaml` | Training defaults | Epochs, batch sizes, schedulers |

### **Model & Data Files**
| File | Purpose | Integration |
|------|---------|------------|
| `src/models/models.py` | Model architectures | Compatible with Fabric setup |
| `src/preprocessing/preprocessing.py` | Data preprocessing | Graph creation for Lightning |
| `datasets/` | Raw datasets | Automatic loading & splitting |

## 5. Performance Benefits 🎯

### **What Lightning Handles Automatically:**
- ✅ **Hardware Detection**: Auto-detects GPUs, CPUs, TPUs
- ✅ **Memory Optimization**: Efficient memory usage patterns
- ✅ **Multi-Device Training**: DDP, DeepSpeed strategies
- ✅ **Mixed Precision**: Automatic 16-bit training
- ✅ **Batch Size Optimization**: Prevents OOM, maximizes throughput
- ✅ **DataLoader Optimization**: Workers, prefetching, pin_memory
- ✅ **Checkpointing**: Robust state management
- ✅ **Logging**: Structured experiment tracking

### **Your Custom Logic:**
- 🎯 Model architectures (GAT, VGAE)
- 🎯 Loss functions (distillation, focal loss)
- 🎯 Training logic specific to CAN data
- 🎯 Evaluation metrics and visualization

## 6. Debugging & Optimization Tips 🔧

### **Enable Profiling for New Models:**
```python
config = {
    'enable_profiler': True,
    'profile_memory': True
}
```

### **Optimize Batch Size for Each Dataset:**
```bash
python train_fabric_models.py --optimize-batch-size --dataset hcrl_sa --model gat
```

### **Monitor Training Progress:**
```bash
# View logs in real-time
tail -f outputs/fabric_logs/fabric_training/version_*/metrics.csv

# Tensorboard (if profiler enabled)
tensorboard --logdir outputs/profiler_logs
```

This workflow leverages Lightning's battle-tested infrastructure while maintaining your custom model logic and domain-specific requirements for CAN intrusion detection.