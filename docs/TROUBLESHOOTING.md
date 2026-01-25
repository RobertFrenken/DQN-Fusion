# Troubleshooting Guide

**Common issues and solutions for KD-GAT training**

---

## CUDA / GPU Issues

### Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Solutions**:

1. **Enable batch size optimization**:
   ```bash
   python train_with_hydra_zen.py \
     --model gat \
     --dataset hcrl_sa \
     --training normal \
     --optimize-batch-size
   ```

2. **Manually reduce batch size**:
   ```python
   config = create_gat_normal_config("hcrl_sa", batch_size=32)  # Try 32, 16, 8
   ```

3. **Use gradient accumulation**:
   ```python
   config.training.accumulate_grad_batches = 4  # Effective batch_size = 32*4 = 128
   ```

4. **Enable mixed precision**:
   ```python
   config.training.precision = "16-mixed"
   config.trainer.precision = "16-mixed"
   ```

5. **Reduce model size**:
   ```python
   # For GAT
   config.model.hidden_channels = 32  # Default: 64
   config.model.num_layers = 3        # Default: 5
   
   # For VGAE
   config.model.hidden_dims = [448, 224, 24]  # Default: [896, 448, 336, 48]
   ```

### GPU Not Detected

**Symptoms**:
```
RuntimeError: No CUDA GPUs are available
Device: cpu
```

**Solutions**:

1. **Check GPU availability**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Load CUDA module (OSC)**:
   ```bash
   module load cuda/11.8.0
   ```

3. **Check PyTorch CUDA version**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

4. **Force CPU training** (if needed):
   ```python
   config.trainer.accelerator = "cpu"
   ```

### CUDA Version Mismatch

**Symptoms**:
```
RuntimeError: CUDA error: invalid device function
cuDNN version mismatch
```

**Solutions**:

1. **Reinstall PyTorch with correct CUDA**:
   ```bash
   # For CUDA 11.8
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For CUDA 12.1
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

2. **Check compatibility**:
   ```bash
   nvcc --version  # System CUDA
   python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
   ```

---

## Data Loading Issues

### Dataset Not Found

**Symptoms**:
```
FileNotFoundError: Dataset path does not exist
Dataset path not found: datasets/can-train-and-test-v1.5/hcrl-sa
```

**Solutions**:

1. **Check data path**:
   ```bash
   ls -la data/automotive/hcrl_sa/
   ```

2. **Set explicit data path**:
   ```bash
   python train_with_hydra_zen.py \
     --model gat \
     --dataset hcrl_sa \
     --training normal \
     --data-path /full/path/to/dataset
   ```

3. **Update config data_path**:
   ```python
   config.dataset.data_path = "/users/PAS2022/rf15/data/automotive/hcrl_sa"
   ```

4. **Use environment variable**:
   ```bash
   export CAN_DATA_PATH="/path/to/datasets"
   python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal
   ```

### Slow Data Loading

**Symptoms**:
- Training stalls at "Loading data..."
- Low GPU utilization
- Long epoch times

**Solutions**:

1. **Increase dataloader workers**:
   ```python
   datamodule = CANGraphDataModule(
       dataset_name="hcrl_sa",
       num_workers=8  # Increase from 4
   )
   ```

2. **Enable data caching**:
   ```python
   config.dataset.cache_processed_data = True
   ```

3. **Use SSD for data** (not NFS):
   ```bash
   # Copy to local SSD on compute node
   cp -r $HOME/data /local/scratch/$USER/
   export CAN_DATA_PATH=/local/scratch/$USER/data
   ```

---

## Model Loading Issues

### Missing Teacher Model

**Symptoms**:
```
FileNotFoundError: Teacher model not found at specified path
ValueError: Knowledge distillation requires 'teacher_model_path'
```

**Solutions**:

1. **Check canonical path**:
   ```bash
   ls -la experiment_runs/automotive/hcrl_sa/supervised/gat/teacher/no_distillation/normal/
   ```

2. **Train teacher first**:
   ```bash
   python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal
   ```

3. **Specify explicit path**:
   ```bash
   python train_with_hydra_zen.py \
     --model gat_student \
     --dataset hcrl_sa \
     --training knowledge_distillation \
     --teacher-path /explicit/path/to/teacher.pth
   ```

### Pickle/Checkpoint Loading Errors

**Symptoms**:
```
RuntimeError: Error(s) in loading state_dict
_pickle.UnpicklingError: invalid load key
```

**Solutions**:

1. **Use weights_only loading**:
   ```python
   checkpoint = torch.load("model.pth", weights_only=True)
   ```

2. **Validate artifacts**:
   ```bash
   python scripts/validate_artifact.py --model-path path/to/model.pth
   ```

3. **Re-save with sanitized format**:
   ```python
   # Load old checkpoint
   old_ckpt = torch.load("old_model.pth", weights_only=False)
   
   # Extract weights only
   state_dict = old_ckpt["model_state_dict"]
   
   # Save clean checkpoint
   torch.save({"model_state_dict": state_dict}, "clean_model.pth")
   ```

---

## Configuration Issues

### Config Validation Failed

**Symptoms**:
```
ValueError: Unknown model type: gat_teacher
ValueError: Unknown dataset: hcrl-sa
```

**Solutions**:

1. **Check valid options**:
   ```python
   from src.config.hydra_zen_configs import CANGraphConfigStore
   
   store = CANGraphConfigStore()
   
   # Valid model types
   print(store.get_model_config.__doc__)
   # gat, gat_student, vgae, vgae_student, dqn, dqn_student
   
   # Valid datasets
   # hcrl_sa, hcrl_ch, set_01, set_02, set_03, set_04
   ```

2. **Use factory functions**:
   ```python
   from src.config.hydra_zen_configs import create_gat_normal_config
   
   config = create_gat_normal_config("hcrl_sa")  # Guaranteed valid
   ```

### Missing Required Artifacts (Fusion/Curriculum)

**Symptoms**:
```
FileNotFoundError: Fusion training requires pre-trained artifacts
FileNotFoundError: Curriculum training requires VGAE model at ...
```

**Solutions**:

1. **Check required artifacts**:
   ```python
   config = create_fusion_config("hcrl_sa")
   artifacts = config.required_artifacts()
   
   for name, path in artifacts.items():
       print(f"{name}: {path.exists()}")
   ```

2. **Train prerequisites first**:
   ```bash
   # For fusion
   python train_with_hydra_zen.py --model vgae --dataset hcrl_sa --training autoencoder
   python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal
   
   # For curriculum
   python train_with_hydra_zen.py --model vgae --dataset hcrl_sa --training autoencoder
   ```

3. **Use manifests**:
   ```bash
   python scripts/generate_manifests.py --dataset hcrl_sa --mode fusion
   python scripts/validate_artifact.py --manifest jobs/fusion_manifest.json
   ```

---

## Training Issues

### Training Not Converging

**Symptoms**:
- Loss not decreasing
- Accuracy stuck at ~0.5
- Validation worse than training

**Solutions**:

1. **Check learning rate**:
   ```python
   config.training.learning_rate = 0.001  # Try 0.0001, 0.0005, 0.001, 0.005
   ```

2. **Enable learning rate scheduling**:
   ```python
   config.training.scheduler.use_scheduler = True
   config.training.scheduler.scheduler_type = "cosine"
   ```

3. **Increase patience**:
   ```python
   config.training.early_stopping_patience = 100  # From 50
   ```

4. **Check data balance**:
   ```python
   datamodule = CANGraphDataModule(
       dataset_name="hcrl_sa",
       balance_classes=True  # Balance normal/attack samples
   )
   ```

5. **Reduce regularization**:
   ```python
   config.model.dropout = 0.1  # From 0.2
   config.training.weight_decay = 1e-5  # From 1e-4
   ```

### NaN Loss

**Symptoms**:
```
Loss is NaN
RuntimeError: Function 'LogSoftmax' returned nan values
```

**Solutions**:

1. **Enable gradient clipping**:
   ```python
   config.training.gradient_clip_val = 1.0
   ```

2. **Reduce learning rate**:
   ```python
   config.training.learning_rate = 0.0001  # From 0.001
   ```

3. **Check for inf/nan in data**:
   ```python
   import torch
   for batch in train_loader:
       if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
           print("Found NaN/Inf in data!")
           break
   ```

4. **Use mixed precision carefully**:
   ```python
   # If using 16-mixed and seeing NaN, try 32-true
   config.training.precision = "32-true"
   ```

### Overfitting

**Symptoms**:
- Training accuracy >> validation accuracy
- Loss decreasing on train, increasing on val

**Solutions**:

1. **Increase regularization**:
   ```python
   config.model.dropout = 0.3  # From 0.2
   config.training.weight_decay = 1e-4  # From 1e-5
   ```

2. **Use early stopping**:
   ```python
   config.training.early_stopping_patience = 30
   config.training.monitor_metric = "val_loss"
   ```

3. **Add batch normalization**:
   ```python
   config.model.use_batch_norm = True
   ```

4. **Reduce model capacity**:
   ```python
   config.model.hidden_channels = 32  # From 64
   config.model.num_layers = 3       # From 5
   ```

---

## Performance Issues

### Slow Training

**Symptoms**:
- Seconds per batch > 1.0
- Low GPU utilization (<50%)
- Hours to complete epoch

**Solutions**:

1. **Enable mixed precision**:
   ```python
   config.training.precision = "16-mixed"
   ```

2. **Increase batch size**:
   ```python
   config.training.batch_size = 128  # From 64
   ```

3. **More dataloader workers**:
   ```python
   datamodule = CANGraphDataModule(num_workers=8)
   ```

4. **Enable cudnn benchmark**:
   ```python
   config.training.cudnn_benchmark = True
   ```

5. **Use faster dataset loading**:
   ```python
   config.dataset.cache_processed_data = True
   ```

### Wall Time Exceeded

**Symptoms**:
```
SLURM job killed: DUE TO TIME LIMIT
```

**Solutions**:

1. **Request more time**:
   ```bash
   python oscjobmanager.py submit \
     --preset gat_normal_hcrl_sa \
     --walltime 08:00:00  # From 04:00:00
   ```

2. **Reduce epochs**:
   ```python
   config.training.max_epochs = 200  # From 400
   ```

3. **Enable checkpointing**:
   ```python
   # Training will resume from checkpoint if resubmitted
   config.training.enable_checkpointing = True
   ```

---

## Job Submission Issues

### Job Won't Start

**Symptoms**:
- Job stuck in PENDING state
- `squeue` shows job but never runs

**Solutions**:

1. **Check job status**:
   ```bash
   squeue -u $USER
   scontrol show job <jobid>
   ```

2. **Check account/partition**:
   ```bash
   # Verify account
   sacctmgr show user $USER
   
   # Use correct account
   python oscjobmanager.py submit --preset gat_normal_hcrl_sa --account PAS3209
   ```

3. **Check dependencies**:
   ```bash
   scontrol show job <jobid> | grep Dependency
   
   # If dependency never satisfied, cancel and resubmit
   scancel <jobid>
   ```

4. **Reduce resource request**:
   ```bash
   # If asking for too much
   python oscjobmanager.py submit \
     --preset gat_normal_hcrl_sa \
     --memory 32G \  # From 64G
     --gpus 1        # From 2
   ```

### Job Failed Immediately

**Symptoms**:
- Job completes in <1 minute
- Exit code non-zero

**Solutions**:

1. **Check error log**:
   ```bash
   tail -50 slurm_jobs/job_<jobid>.err
   ```

2. **Common failures**:
   - **Module not found**: Check conda environment activated
   - **CUDA not available**: Check GPU allocation and CUDA module
   - **Dataset not found**: Check data path
   - **Import error**: Check all dependencies installed

3. **Test locally first**:
   ```bash
   # Run locally to debug
   python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal --max-epochs 1
   ```

---

## Import/Module Issues

### ModuleNotFoundError

**Symptoms**:
```
ModuleNotFoundError: No module named 'src'
ImportError: cannot import name 'CANGraphConfigStore'
```

**Solutions**:

1. **Activate correct environment**:
   ```bash
   conda activate gnn-experiments
   ```

2. **Check Python path**:
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

3. **Install missing packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Reinstall in development mode**:
   ```bash
   pip install -e .
   ```

---

## Best Practices to Avoid Issues

### Before Training

```bash
# 1. Validate config
python -c "from src.config.hydra_zen_configs import create_gat_normal_config; cfg = create_gat_normal_config('hcrl_sa'); print('✓')"

# 2. Check dataset
ls -la data/automotive/hcrl_sa/

# 3. Test import
python -c "from src.training.trainer import HydraZenTrainer; print('✓')"

# 4. Dry run
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal --fast-dev-run

# 5. Check GPU
nvidia-smi
```

### During Training

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f slurm_jobs/job_*.out

# Check resource usage
seff <jobid>
```

### After Training

```bash
# Validate saved model
python scripts/validate_artifact.py --model-path experiment_runs/.../best_model.pth

# Check training metrics
python scripts/collect_summaries.py --dataset hcrl_sa

# Verify results
ls -la experiment_runs/automotive/hcrl_sa/
```

---

## Getting Help

If issues persist:

1. **Check logs carefully** - Error messages usually point to the problem
2. **Search GitHub issues** - Someone may have encountered this before
3. **Simplify** - Try with smaller model, fewer epochs, single GPU
4. **Test locally** - Eliminate cluster-specific issues
5. **Check versions** - Ensure PyTorch, CUDA, dependencies match requirements

**Quick diagnostics**:
```bash
python scripts/check_environment.py  # Verify environment setup
python scripts/check_datasets.py --dataset hcrl_sa  # Verify dataset
```

---

**Most issues are fixable!** Start with the simplest solution and work up.
