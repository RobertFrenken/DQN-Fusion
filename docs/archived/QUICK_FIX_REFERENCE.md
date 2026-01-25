# Quick Reference: VGAE Training Fixes

## Summary of Changes

### ✅ Fixed Issues
1. **Wall clock timeout** - Complex datasets now get 12h automatically
2. **Batch size tuner disparity** - Added 50% safety factor for graph memory overhead

### 📁 Modified Files
- `oscjobmanager.py` - Auto resource scaling
- `train_with_hydra_zen.py` - Batch size safety factor
- `src/config/hydra_zen_configs.py` - New config parameter
- `scripts/estimate_resources.py` - NEW utility tool
- `docs/VGAE_FIXES.md` - Full documentation

## Quick Usage Examples

### 1. Rerun set_02 (Now with 12h + 96GB)
```bash
python oscjobmanager.py submit autoencoder_set_02
```

### 2. Estimate Resources Before Submitting
```bash
# Single dataset estimate
python scripts/estimate_resources.py --dataset set_02 --training vgae_autoencoder

# Compare all datasets
python scripts/estimate_resources.py --compare --training vgae_autoencoder
```

### 3. Customize Batch Size Safety Factor
```python
# In your config or via extra-args
training.graph_memory_safety_factor = 0.6  # Use 60% of tuner's suggestion
```

### 4. Manual Resource Override
```bash
# Override auto-detection
python oscjobmanager.py submit autoencoder_set_02 --walltime 16:00:00 --memory 128G
```

## New Resource Defaults

| Dataset | Old | New | Notes |
|---------|-----|-----|-------|
| hcrl_sa, hcrl_ch | 2h, 32GB | 8h, 64GB | Standard datasets |
| set_01-04 | 2h, 32GB | **12h, 96GB** | **Auto-detected** |

## Batch Size Changes

| Stage | Before | After |
|-------|--------|-------|
| Tuner finds | 1024 | 1024 |
| Applied value | 1024 (crashes) | **512 (safe)** |
| Safety factor | None | **0.5 (50%)** |

## What to Expect

### When You Submit a Job
```
🕐 Detected complex dataset - using extended walltime: 12:00:00
💾 Detected complex dataset - using extended memory: 96G
✅ Created Slurm script: experimentruns/slurm_runs/autoencoder_set_02.sh
```

### During Training
```
✅ Batch size tuner found: 1024
🛡️ Applied 50% safety factor for graph data overhead
📊 Final safe batch size: 512
```

## Troubleshooting

### Still Getting OOM?
```python
# Reduce safety factor further
training.graph_memory_safety_factor = 0.3  # More aggressive reduction
```

### Training Too Slow?
```python
# Increase safety factor (risk OOM)
training.graph_memory_safety_factor = 0.7  # Use more memory

# Or disable tuner and set manually
training.optimize_batch_size = False
training.batch_size = 256
```

### Job Still Timing Out?
```bash
# Request more time manually
python oscjobmanager.py submit autoencoder_set_02 --walltime 16:00:00
```

## Learn More
- Full details: [docs/VGAE_FIXES.md](VGAE_FIXES.md)
- Resource estimates: `python scripts/estimate_resources.py --help`
- Job templates: [docs/JOB_TEMPLATES.md](JOB_TEMPLATES.md)
