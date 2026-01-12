# CAN-Graph Training Architecture Plan

## 📋 Current Architecture Analysis

### 🏗️ Architecture Issues

**Current Nested Call Chain:**
```
train_individual_model.py → 
ResourceAwareOrchestrator._execute_training_phase() → 
AD_KD_GPU.main() → 
KnowledgeDistillationPipeline + create_optimized_data_loaders()
```

**Problems Identified:**
- **Triple data loading**: Individual trainer creates config → Orchestrator overrides config → AD_KD_GPU recreates loaders from scratch
- **Conflicting batch size management**: Three different systems trying to optimize batch sizes
- **Resource management duplication**: Both orchestrator and AD_KD_GPU have GPU optimization
- **Configuration fragmentation**: Config gets modified at multiple layers

### 🔄 Functional Overlap Matrix

| Component | Data Loading | Batch Optimization | GPU Management | Memory Monitoring |
|-----------|-------------|-------------------|---------------|------------------|
| `train_individual_model.py` | ❌ | ❌ | ✅ (via orchestrator) | ✅ (via orchestrator) |
| `ResourceAwareOrchestrator` | ❌ | ✅ (optimal_batch_size) | ✅ | ✅ |
| `enhanced_resource_aware_training.py` | ✅ | ✅ (adaptive) | ✅ | ✅ |
| `AD_KD_GPU.py` | ✅ | ✅ (hardcoded logic) | ✅ | ✅ |

**Result**: Four different systems doing similar things with potential conflicts.

---

## 🎯 Ideal Training Pipeline by Model Type

### 1. Teacher Models (Individual Base Models)

**What they are**: Base autoencoder + classifier trained on normal data
**Datasets**: `hcrl_ch`, `hcrl_sa`, `set_01`, `set_02`, `set_03`, `set_04`

**🚀 Recommended Pipeline:**
```
train_individual_model.py → enhanced_resource_aware_training.py
```

**Flow:**
```mermaid
graph TD
    A[train_individual_model.py] --> B[EnhancedResourceAwareTrainer]
    B --> C[AdaptiveMemoryManager]
    B --> D[GPU Optimization]
    B --> E[Teacher Training Loop]
    E --> F[Autoencoder Phase]
    F --> G[Classifier Phase]
    G --> H[Model Saved: best_teacher_model_{dataset}.pth]
```

**Command:**
```bash
python train_individual_model.py --dataset hcrl_ch --model teacher
```

**Key Features:**
- Dynamic batch size optimization based on GPU memory
- Real-time memory profiling and adjustment
- Automatic error recovery and cleanup
- ~45 minutes per model

### 2. Student Models (Knowledge Distillation)

**What they are**: Compressed models learning from teacher models
**Dependencies**: Requires corresponding teacher model to exist

**🚀 Recommended Pipeline:**
```
train_individual_model.py → enhanced_resource_aware_training.py (KD mode)
```

**Flow:**
```mermaid
graph TD
    A[train_individual_model.py] --> B{Check Teacher Exists}
    B -->|Yes| C[EnhancedResourceAwareTrainer]
    B -->|No| D[Error: Teacher Required]
    C --> E[Load Teacher Model]
    E --> F[Knowledge Distillation Loop]
    F --> G[Student learns from Teacher]
    G --> H[Model Saved: final_student_model_{dataset}.pth]
```

**Command:**
```bash
python train_individual_model.py --dataset hcrl_ch --model student
```

**Key Features:**
- Automatic teacher model loading and validation
- Knowledge distillation with temperature scaling
- Student-specific memory optimization
- ~35 minutes per model

### 3. Fusion DQN Models

**What they are**: Deep Q-Network agents that learn to combine teacher and student predictions
**Dependencies**: Requires both teacher AND student models

**🚀 Recommended Pipeline:**
```
train_individual_model.py → fusion_training.py (enhanced)
```

**Flow:**
```mermaid
graph TD
    A[train_individual_model.py] --> B{Check Dependencies}
    B -->|Both Exist| C[FusionTrainingPipeline]
    B -->|Missing| D[Error: Teacher + Student Required]
    C --> E[Load Teacher + Student]
    E --> F[DQN Agent Training]
    F --> G[Reinforcement Learning Loop]
    G --> H[Model Saved: fusion_agent_{dataset}.pth]
```

**Command:**
```bash
python train_individual_model.py --dataset hcrl_ch --model fusion
```

**Key Features:**
- Dual model validation and loading
- DQN-specific memory management
- Episode-based training with experience replay
- ~60 minutes per model

---

## 🚀 Recommended Architecture Changes

### Phase 1: Quick Fix (Immediate)

**Change orchestrator to use enhanced trainer:**

```python
# In ResourceAwareOrchestrator._execute_training_phase()
# REPLACE:
if phase.script_path == "src/training/AD_KD_GPU.py":
    from src.training.AD_KD_GPU import main as kd_main
    kd_main(config)

# WITH:
if phase.script_path == "src/training/AD_KD_GPU.py":
    from src.training.enhanced_resource_aware_training import main as enhanced_main
    enhanced_main(config)
```

**Benefits**: Immediate access to new resource management without architectural changes.

### Phase 2: Streamlined Architecture (Recommended)

**Create simplified direct pipeline:**

```
train_individual_model.py → enhanced_resource_aware_training.py
```

**Remove layers:**
- ResourceAwareOrchestrator (for individual training)
- AD_KD_GPU.py dependencies
- Configuration conflicts

**Implementation:**
```python
# In train_individual_model.py - train_model() method
# REPLACE orchestrator call:
result = self.orchestrator._execute_training_phase(phase)

# WITH direct call:
from src.training.enhanced_resource_aware_training import EnhancedResourceAwareTrainer
trainer = EnhancedResourceAwareTrainer(config)
result = trainer.train_complete_pipeline(dataset, num_ids)
```

### Phase 3: Clean Architecture (Long-term)

**Unified system with clear boundaries:**

```
Individual Training: train_individual_model.py → EnhancedResourceAwareTrainer
Batch Training: retrain_all_models.py → ResourceAwareOrchestrator → EnhancedResourceAwareTrainer
```

---

## 📁 Ideal File Pipeline Structure

### Current Files (Problematic)
```
❌ CURRENT (Multiple overlaps):
train_individual_model.py
├── ResourceAwareOrchestrator
│   ├── AD_KD_GPU.py (old system)
│   │   ├── KnowledgeDistillationPipeline
│   │   └── create_optimized_data_loaders (hardcoded)
│   └── fusion_training.py (separate system)
└── enhanced_resource_aware_training.py (unused)
```

### Recommended Files (Clean)
```
✅ RECOMMENDED:
train_individual_model.py
├── enhanced_resource_aware_training.py
│   ├── EnhancedResourceAwareTrainer
│   ├── AdaptiveMemoryManager  
│   └── Unified data loading + resource management
├── fusion_training.py (enhanced)
│   └── For DQN-specific training
└── [OPTIONAL] ResourceAwareOrchestrator 
    └── Only for batch/multi-model training
```

---

## 🔄 Training Workflow by Model Type

### Complete Training Sequence

**1. Teacher Models (Base training)**
```bash
# Train all teacher models first (independent)
python train_individual_model.py --dataset hcrl_ch --model teacher    # 45 min
python train_individual_model.py --dataset hcrl_sa --model teacher    # 45 min
python train_individual_model.py --dataset set_01 --model teacher     # 45 min
python train_individual_model.py --dataset set_02 --model teacher     # 45 min
python train_individual_model.py --dataset set_03 --model teacher     # 45 min
python train_individual_model.py --dataset set_04 --model teacher     # 45 min
```

**2. Student Models (Knowledge Distillation)**
```bash
# Train student models (requires teachers)
python train_individual_model.py --dataset hcrl_ch --model student    # 35 min
python train_individual_model.py --dataset hcrl_sa --model student    # 35 min
python train_individual_model.py --dataset set_01 --model student     # 35 min
python train_individual_model.py --dataset set_02 --model student     # 35 min
python train_individual_model.py --dataset set_03 --model student     # 35 min
python train_individual_model.py --dataset set_04 --model student     # 35 min
```

**3. Fusion Models (DQN)**
```bash
# Train fusion agents (requires both teacher + student)
python train_individual_model.py --dataset hcrl_ch --model fusion     # 60 min
python train_individual_model.py --dataset hcrl_sa --model fusion     # 60 min
python train_individual_model.py --dataset set_01 --model fusion      # 60 min
python train_individual_model.py --dataset set_02 --model fusion      # 60 min
python train_individual_model.py --dataset set_03 --model fusion      # 60 min
python train_individual_model.py --dataset set_04 --model fusion      # 60 min
```

**Total Time**: ~13.5 hours for all 18 models (6 datasets × 3 model types)

### Priority Training (Focus on Important Datasets)

**High Priority: HCRL datasets**
```bash
# Phase 1: Core models
python train_individual_model.py --dataset hcrl_ch --model teacher
python train_individual_model.py --dataset hcrl_sa --model teacher
python train_individual_model.py --dataset hcrl_ch --model student  
python train_individual_model.py --dataset hcrl_sa --model student

# Phase 2: Advanced fusion  
python train_individual_model.py --dataset hcrl_ch --model fusion
python train_individual_model.py --dataset hcrl_sa --model fusion
```

**Medium Priority: SET datasets**
```bash
# Complete the remaining datasets as needed
for dataset in set_01 set_02 set_03 set_04; do
    python train_individual_model.py --dataset $dataset --model teacher
    python train_individual_model.py --dataset $dataset --model student  
    python train_individual_model.py --dataset $dataset --model fusion
done
```

---

## 💻 Implementation Timeline

### Week 1: Quick Fixes
- [ ] Modify orchestrator to call enhanced_resource_aware_training.py
- [ ] Test individual training with new system
- [ ] Validate resource management improvements

### Week 2: Architecture Cleanup  
- [ ] Create direct pipeline option in train_individual_model.py
- [ ] Benchmark performance improvements
- [ ] Update fusion_training.py integration

### Week 3: Full Migration
- [ ] Deprecate AD_KD_GPU.py for new workflows
- [ ] Update documentation and examples
- [ ] Performance validation across all model types

---

## 🎯 Expected Benefits

**Performance Improvements:**
- 40-60% reduction in memory conflicts
- 20-30% faster training due to optimized resource usage
- Better GPU utilization through unified management

**Maintenance Benefits:**
- Single source of truth for resource management
- Clearer debugging and error isolation
- Simpler configuration management

**User Experience:**
- More reliable individual model training
- Better error messages and recovery
- Consistent behavior across model types

---

## 🔧 Quick Start Commands

**Check what needs training:**
```bash
python train_individual_model.py --list-available
```

**Interactive guided training:**
```bash
python guided_training.py
```

**Train specific model:**
```bash
python train_individual_model.py --dataset hcrl_ch --model teacher --dry-run  # Check first
python train_individual_model.py --dataset hcrl_ch --model teacher            # Train it
```

**Force retrain existing model:**
```bash
python train_individual_model.py --dataset hcrl_ch --model teacher --force-retrain
```