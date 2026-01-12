# Enhanced Model Retraining System

This document describes the enhanced model retraining system with improved GPU utilization, resource-aware processing, and comprehensive process management.

## 🚀 Quick Start

### Simple Retraining (Recommended)
```bash
# Retrain all missing models with optimal settings
python retrain_all_models.py

# Force retrain all models
python retrain_all_models.py --force-retrain

# Dry run to see what would be done
python retrain_all_models.py --dry-run
```

### Advanced Usage
```bash
# Prioritize specific datasets
python retrain_all_models.py --priority-datasets hcrl_ch hcrl_sa

# Custom output directory
python retrain_all_models.py --output-dir outputs/my_training

# Verbose logging
python retrain_all_models.py --verbose
```

## 📋 Prerequisites

### System Requirements
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM (16GB+ recommended)
- **RAM**: 16GB+ (32GB+ recommended for large datasets)
- **Storage**: 10GB+ free space for models and logs
- **OS**: Windows 10/11, Linux, or macOS with CUDA support

### Software Requirements
- Python 3.8+
- PyTorch 1.12+ with CUDA support
- PyTorch Geometric 2.6.1+
- All project dependencies (see `requirements.txt`)

## 🏗️ System Architecture

The enhanced retraining system consists of several interconnected components:

### Core Components

1. **Training Pipeline Coordinator** (`training_pipeline_coordinator.py`)
   - Main orchestration system
   - Handles complete retraining workflow
   - Manages dependencies between training phases

2. **Resource-Aware Orchestrator** (`resource_aware_orchestrator.py`)
   - Intelligent resource allocation
   - Dynamic scheduling based on GPU availability
   - Automatic error recovery and retry logic

3. **Adaptive Memory Manager** (`adaptive_memory_manager.py`)
   - Real-time batch size optimization
   - Memory pressure detection and response
   - OOM (Out of Memory) prevention and recovery

4. **Process Manager** (`process_manager.py`)
   - Multi-process coordination
   - System resource monitoring
   - Performance analytics and reporting

5. **Enhanced Configuration Manager** (`enhanced_config_manager.py`)
   - GPU-optimized configuration generation
   - Environment-specific settings
   - Configuration validation and recommendations

## 🔧 Key Features

### GPU Optimization
- **Adaptive Batch Sizing**: Automatically adjusts batch sizes based on GPU memory
- **Memory Management**: Real-time memory monitoring and cleanup
- **Mixed Precision**: Automatic FP16/FP32 optimization for modern GPUs
- **Resource Monitoring**: Comprehensive GPU utilization tracking

### Resource Management
- **Intelligent Scheduling**: Phases scheduled based on resource availability
- **Dependency Management**: Automatic handling of model dependencies
- **Process Coordination**: Multi-process training with resource sharing
- **Error Recovery**: Automatic retry and recovery mechanisms

### Training Enhancements
- **Knowledge Distillation**: Optimized teacher-student training
- **Fusion Learning**: DQN-based model fusion with experience replay
- **Early Stopping**: Intelligent convergence detection
- **Checkpoint Management**: Automatic model saving and versioning

### Monitoring & Analytics
- **Real-time Monitoring**: Live resource usage dashboard
- **Performance Analytics**: Comprehensive training metrics
- **Visual Dashboards**: Automated chart generation
- **Recommendation System**: Post-training optimization suggestions

## 📊 Training Pipeline

The system trains models in three phases for each dataset:

### Phase 1: Teacher Training
- Full-size VGAE and GAT models
- Standard supervised learning
- ~45 minutes per dataset (typical)

### Phase 2: Student Training (Knowledge Distillation)
- Compressed models learning from teachers
- Knowledge distillation with temperature scaling
- ~35 minutes per dataset (typical)

### Phase 3: Fusion Training
- DQN agent learns optimal fusion strategies
- Experience replay and exploration
- ~60 minutes per dataset (typical)

**Total estimated time**: ~2.5 hours per dataset (6 datasets = ~15 hours)

## 📁 Generated Outputs

The system creates comprehensive outputs in the `outputs/enhanced_retraining/` directory:

### Training Results
- `{plan_id}_results.json`: Complete training results and metrics
- `{plan_id}_plan.json`: Detailed retraining plan
- `{plan_id}_dashboard.png`: Visual monitoring dashboard

### Model Files
- `saved_models/best_teacher_model_{dataset}.pth`: Optimized teacher models
- `saved_models/final_student_model_{dataset}.pth`: Compressed student models  
- `saved_models/fusion_agent_{dataset}.pth`: Trained fusion agents

### Logs and Analytics
- `coordinator_logs/`: Main coordination logs
- `orchestrator_logs/`: Resource orchestration logs
- `process_logs/`: Individual process logs
- `generated_configs/`: Auto-generated configuration files

## 🛠️ Configuration

### Automatic Configuration
The system automatically detects your GPU and creates optimized configurations:

```python
from src.config.enhanced_config_manager import create_optimized_config

# Creates config optimized for your system
config = create_optimized_config()
```

### Manual Configuration
For custom settings, modify the base configuration:

```yaml
# conf/base.yaml
batch_size: 4096        # Adjusted automatically based on GPU
lr: 0.001              # Learning rate
epochs: 10             # Training epochs
device: cuda           # Training device
mixed_precision: true  # Enable FP16 training
```

### GPU-Specific Optimizations

**A100 40GB+**:
- Batch size: 8192
- Workers: 24
- Mixed precision: Enabled
- Buffer size: 500K

**RTX 3090/4090 (24GB)**:
- Batch size: 4096
- Workers: 16  
- Mixed precision: Enabled
- Buffer size: 300K

**RTX 3080 (10GB)**:
- Batch size: 2048
- Workers: 12
- Mixed precision: Enabled
- Buffer size: 200K

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```
Solution: The system automatically reduces batch sizes and retries
Manual fix: Reduce batch_size in configuration
```

**Missing Dataset Files**
```
Error: Dataset path not found
Solution: Verify dataset paths in DATASET_PATHS configuration
```

**Slow Training**
```
Symptoms: Very low GPU utilization
Solution: Increase batch sizes, check data loading bottlenecks
```

**Process Hangs**
```
Symptoms: Training stops without error
Solution: Check system resources, restart with --verbose
```

### Debug Mode
```bash
# Enable verbose logging and monitoring
python retrain_all_models.py --verbose --dry-run

# Check system status
python -c "from src.training.training_pipeline_coordinator import create_coordinator; print(create_coordinator().get_training_status())"
```

## 📈 Performance Optimization

### Recommended Settings by GPU

| GPU Model | Batch Size | Workers | Mixed Precision | Expected Performance |
|-----------|------------|---------|----------------|---------------------|
| A100 40GB | 8192 | 24 | Yes | ~120 samples/sec |
| RTX 4090 | 4096 | 16 | Yes | ~80 samples/sec |
| RTX 3090 | 4096 | 16 | Yes | ~70 samples/sec |
| RTX 3080 | 2048 | 12 | Yes | ~50 samples/sec |

### Optimization Tips

1. **Enable Mixed Precision**: 30-50% speedup on modern GPUs
2. **Optimize Data Loading**: Use sufficient workers (but not too many)
3. **Monitor Memory**: Keep GPU memory usage at 85-90%
4. **Use SSD Storage**: Significantly improves data loading
5. **Close Other Applications**: Free up GPU memory for training

## 🔍 Monitoring

### Real-time Monitoring
The system provides real-time monitoring of:
- GPU utilization and memory usage
- CPU and RAM consumption
- Training progress and loss curves
- Process status and health

### Dashboard Generation
Automatic dashboard generation includes:
- Resource utilization over time
- Training progress visualization
- Performance metrics comparison
- System health indicators

### Log Analysis
Comprehensive logging covers:
- Training metrics and losses
- Resource allocation decisions  
- Error conditions and recoveries
- Performance bottleneck identification

## 🚀 Advanced Usage

### Custom Training Scripts
```python
from src.training.enhanced_resource_aware_training import EnhancedResourceAwareTrainer
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("conf/base.yaml")

# Initialize trainer
trainer = EnhancedResourceAwareTrainer(config)

# Train models
results = trainer.train_complete_pipeline(dataset, num_ids)
```

### Batch Processing
```python
from src.training.training_pipeline_coordinator import TrainingPipelineCoordinator

# Initialize coordinator
coordinator = TrainingPipelineCoordinator()

# Create and execute plan
plan = coordinator.create_retraining_plan(force_retrain=True)
results = coordinator.execute_retraining_plan(plan)
```

### Resource Monitoring
```python
from src.utils.process_manager import ProcessManager

# Start monitoring
manager = ProcessManager()
manager.start_monitoring()

# Get system status
status = manager.get_system_status()
print(f"GPU Usage: {status['current_resources']['gpu_utilization_percent']:.1f}%")
```

## 📝 API Reference

### Main Functions

**`run_complete_retraining()`**
```python
def run_complete_retraining(
    force_retrain: bool = False,
    priority_datasets: Optional[List[str]] = None,
    dry_run: bool = False
) -> RetrainingResults
```

**`create_optimized_config()`**
```python
def create_optimized_config(
    environment_name: str = "auto"
) -> TrainingEnvironmentConfig
```

### Configuration Classes

**`TrainingEnvironmentConfig`**
- Complete training environment setup
- GPU-optimized parameters
- Model and dataset configurations

**`RetrainingResults`**
- Comprehensive training results
- Performance metrics
- Resource utilization summary

## 🤝 Contributing

To contribute improvements:

1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include error handling and logging
4. Test with different GPU configurations
5. Update documentation accordingly

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review log files in `outputs/enhanced_retraining/`
3. Run with `--verbose --dry-run` for debugging
4. Verify system requirements and dependencies

---

*This enhanced retraining system is designed to maximize GPU utilization while providing comprehensive monitoring and automatic optimization for CAN bus intrusion detection model training.*