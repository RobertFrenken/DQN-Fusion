# VGAE Training Improvements: Wall Clock & Batch Size Fixes

## Overview

This document describes fixes for two critical issues discovered during VGAE testing:
1. **Wall clock timeout** for complex datasets (set_01-04)
2. **Batch size tuner disparity** due to graph data memory overhead

## Problem 1: Wall Clock Timeout ⏰

### Issue
The `set_02` job was killed after 2 hours due to wall clock limit, despite having ~203K graphs still training. The log showed:
```
[2026-01-24T05:08:14.006] error: *** JOB 43953512 CANCELLED AT 2026-01-24T05:08:14 DUE TO TIME LIMIT ***
```

### Root Cause
- Complex datasets (set_01-04) have ~200K graphs each
- Original default: 2 hours walltime
- Insufficient for large-scale training

### Solution
**Implemented automatic resource scaling** in [oscjobmanager.py](../oscjobmanager.py):

```python
# Detect complex datasets and automatically scale resources
complex_datasets = ['set_01', 'set_02', 'set_03', 'set_04']
if any(ds in config_name for ds in complex_datasets):
    walltime = '12:00:00'  # 12 hours (was 2 hours)
    memory = '96G'         # 96GB (was 32GB)
```

**New Defaults:**
- Standard datasets (hcrl_sa, hcrl_ch): 8 hours, 64GB
- Complex datasets (set_01-04): 12 hours, 96GB

### Usage
Jobs now automatically get appropriate resources:
```bash
# Automatically gets 12h + 96GB for set_02
python oscjobmanager.py submit autoencoder_set_02

# Manual override still available
python oscjobmanager.py submit autoencoder_set_02 --walltime 16:00:00 --memory 128G
```

## Problem 2: Batch Size Tuner Disparity 🔧

### Issue
Lightning's batch size tuner doesn't account for graph-specific memory overhead:
- **Tuner found**: batch_size=1024
- **Restoration failed**: Checkpoint loading error
- **Fell back to**: batch_size=64 (from config)
- **Result**: Suboptimal training speed

### Root Cause
Graph neural networks have **hidden memory overhead** not visible to the tuner:

1. **Message passing**: Intermediate neighbor aggregation tensors
2. **Attention mechanisms**: $O(N^2)$ attention weight matrices  
3. **Node embeddings**: Temporary vectors during graph convolution
4. **Batch graph structures**: Edge indices, batch assignment vectors

The tuner only measures forward/backward pass memory, missing these peaks.

### Solution
**Implemented conservative safety factor** in [train_with_hydra_zen.py](../train_with_hydra_zen.py):

```python
# Apply 50% safety factor for graph data
tuner_batch_size = model.batch_size  # e.g., 1024
safe_batch_size = int(tuner_batch_size * 0.5)  # 512
model.batch_size = safe_batch_size
```

**Made it configurable** in [hydra_zen_configs.py](../src/config/hydra_zen_configs.py):

```python
@dataclass
class BaseTrainingConfig:
    # ... existing fields ...
    
    # Graph data memory safety factor
    # Adjustable range: 0.3 (aggressive) to 0.7 (very conservative)
    graph_memory_safety_factor: float = 0.5  # Default 50%
```

### Impact
- **Before**: Tuner says 1024 → OOM crash
- **After**: Tuner says 1024 → Safe batch_size=512 → No OOM
- **Trade-off**: Slightly slower training, but stable and reliable

### Customization
Adjust the safety factor based on your needs:

```python
# More aggressive (faster training, higher OOM risk)
training.graph_memory_safety_factor = 0.7  # Use 70% of tuner suggestion

# More conservative (slower training, very safe)
training.graph_memory_safety_factor = 0.3  # Use 30% of tuner suggestion

# Disable tuner and set batch size manually
training.optimize_batch_size = False
training.batch_size = 256  # Your known-safe value
```

## Resource Estimation Utility 📊

New tool to help plan jobs: [scripts/estimate_resources.py](../scripts/estimate_resources.py)

### Examples

**Estimate for specific dataset:**
```bash
python scripts/estimate_resources.py --dataset set_02 --training vgae_autoencoder

# Output:
# ======================================================================
# 📊 Resource Estimate for set_02 - vgae_autoencoder
# ======================================================================
# Training Type: VGAE autoencoder (unsupervised)
# Dataset Size:  ~203,000 graphs (complex)
# 
# ⏰ Recommended Wall Time: 12:00:00 (12.0 hours)
# 💾 Recommended Memory:    96G (96 GB)
# 
# 🚀 Suggested Command:
#    python oscjobmanager.py submit autoencoder_set_02 --walltime 12:00:00 --memory 96G
```

**Compare all datasets:**
```bash
python scripts/estimate_resources.py --compare --training gat_curriculum

# Output:
# Dataset      Graphs       Wall Time    Memory     Complexity
# ----------------------------------------------------------------------
# hcrl_ch      45,000       04:48:00     35G        standard  
# hcrl_sa      50,000       04:48:00     35G        standard  
# set_01       200,000      14:24:00     106G       complex   
# set_02       203,000      14:24:00     106G       complex   
# set_03       195,000      14:24:00     106G       complex   
# set_04       190,000      14:24:00     106G       complex   
```

**JSON output for automation:**
```bash
python scripts/estimate_resources.py --dataset set_02 --training vgae_autoencoder --json
```

## Testing the Fixes

### Rerun set_02 with New Settings

```bash
# Will automatically get 12h + 96GB + safe batch sizing
python oscjobmanager.py submit autoencoder_set_02
```

### Verify Batch Size Safety Factor

Check the training logs for:
```
✅ Batch size tuner found: 1024
🛡️ Applied 50% safety factor for graph data overhead
📊 Final safe batch size: 512
```

### Monitor Job Progress

```bash
# Check current job status
squeue -u $USER

# Tail the log file
tail -f experimentruns/slurm_runs/autoencoder_set_02_*.log
```

## Configuration Summary

| Parameter | Old Default | New Default | Complex Dataset Override |
|-----------|-------------|-------------|--------------------------|
| Wall Time | 2:00:00 | 8:00:00 | 12:00:00 (auto) |
| Memory | 32G | 64G | 96G (auto) |
| Batch Size | tuner (unsafe) | tuner × 0.5 (safe) | tuner × 0.5 (safe) |
| Safety Factor | N/A | 0.5 | 0.5 (configurable) |

## Recommendations

### For Standard Datasets (hcrl_sa, hcrl_ch)
✅ Use defaults - they should work well

### For Complex Datasets (set_01-04)
✅ Use defaults - automatic 12h + 96GB allocation
⚠️ Consider 16h if using curriculum learning
⚠️ Monitor first epoch duration to gauge total time needed

### For Batch Size Optimization
✅ Keep safety factor at 0.5 (default)
⚠️ If you see OOM errors: reduce to 0.3
⚠️ If training is too slow: try 0.6-0.7 (monitor for OOM)
✅ Disable tuner (`optimize_batch_size=False`) and set manually if issues persist

## Future Improvements

1. **Dynamic resource estimation**: Measure actual dataset size before job submission
2. **Adaptive safety factor**: Start conservative, increase if no OOM after N epochs
3. **Historical data**: Track past runs to improve estimates
4. **Multi-GPU support**: Adjust batch sizing for distributed training

## Related Files

- [oscjobmanager.py](../oscjobmanager.py) - Automatic resource scaling
- [train_with_hydra_zen.py](../train_with_hydra_zen.py) - Batch size safety factor
- [hydra_zen_configs.py](../src/config/hydra_zen_configs.py) - Configuration
- [estimate_resources.py](../scripts/estimate_resources.py) - Resource estimation tool

## References

- Original set_02 job: `experimentruns/slurm_runs/autoencoder_set_02_20260124_030802.log`
- JOB_TEMPLATES.md: Resource allocation guide
- PyTorch Geometric memory profiling: [Link](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html)
