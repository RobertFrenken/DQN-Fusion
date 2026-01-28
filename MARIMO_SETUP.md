# Marimo Interactive Visualization Setup

**Date**: 2026-01-28
**Status**: ✅ Ready to use

---

## What is Marimo?

[Marimo](https://marimo.io/) is a reactive Python notebook that's **better than Jupyter** for visualization development:

✅ **Reactive**: Cells auto-update when dependencies change (no hidden state)
✅ **Reproducible**: Notebooks are valid Python files (`.py`, not `.ipynb`)
✅ **Fast iteration**: See changes immediately without re-running everything
✅ **Version control friendly**: Plain Python files work great with git
✅ **No kernel restarts**: Dependency graph ensures consistency

**Perfect for**: Developing publication-quality figures iteratively

---

## Quick Start

### 1. Launch Marimo Server on SLURM

```bash
cd /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT

# Launch with default settings (4 hours, GPU)
./start_marimo.sh

# Or customize duration and resources
./start_marimo.sh 8        # 8 hours with GPU
./start_marimo.sh 2 cpu    # 2 hours, CPU only
```

**What happens**:
1. Script submits SLURM job
2. Waits for job to start
3. Prints connection instructions
4. Creates a default workspace notebook

### 2. Set Up SSH Tunnel

**On your local machine** (laptop/desktop), run the command from the connection instructions:

```bash
# Example (actual ports will vary)
ssh -L 8765:owens-gpu123:8765 rf15@owens.osc.edu
```

**Keep this terminal open!**

### 3. Open in Browser

Open the URL from connection instructions:

```
http://localhost:8765
```

You should see the Marimo interface with `visualization_workspace.py` loaded!

### 4. Start Developing

The default workspace includes:
- Model loading utilities (config-driven)
- Data loading utilities
- Example matplotlib figure
- Interactive cells for exploration

**Try it**:
1. Edit the checkpoint path
2. Click "Load Model"
3. Model loads automatically with correct dimensions
4. Create custom visualizations

---

## File Locations

```
KD-GAT/
├── start_marimo.sh                          # Helper script to launch
├── launch_marimo_visualization.sh           # SLURM batch script
├── visualizations/
│   ├── notebooks/
│   │   └── visualization_workspace.py       # Default notebook (created automatically)
│   ├── model_loader.py                      # Config-driven model loading
│   ├── data_loader.py                       # Config-driven data loading
│   └── utils.py                             # Plotting utilities
└── slurm_logs/
    ├── marimo_{JOB_ID}.out                  # Job output
    ├── marimo_{JOB_ID}.err                  # Job errors
    ├── marimo_{JOB_ID}_connection.txt       # Connection instructions
    └── marimo_{JOB_ID}_server.log           # Server log
```

---

## Usage Examples

### Load a Trained Model

```python
# In Marimo notebook
from model_loader import load_model_for_visualization

# Auto-discovers frozen config
model, config = load_model_for_visualization(
    checkpoint_path="experimentruns/automotive/hcrl_sa/.../vgae_teacher.pth"
)

# Model dimensions are correct automatically!
print(config.model.hidden_dims)    # [1024, 512, 96] from frozen config
print(config.model.latent_dim)     # 96 from frozen config
```

### Load Dataset

```python
from data_loader import load_data_for_visualization

# Load using same config as model
data = load_data_for_visualization(
    config_path="experimentruns/.../frozen_config.json",
    splits=['test'],
    max_samples=5000
)

test_data = data['test']
print(f"Loaded {len(test_data)} test samples")
```

### Extract Embeddings

```python
from model_loader import ModelLoader

loader = ModelLoader(device='cuda')

# Extract VGAE latent embeddings
embeddings = loader.extract_vgae_embeddings(
    model=model,
    data_list=test_data,
    batch_size=64
)

# embeddings contains: z_mean, z_log_var, z, labels
z = embeddings['z'].numpy()
labels = embeddings['labels'].numpy()
```

### Create Visualization

```python
from utils import setup_figure, save_figure, get_color_palette
import matplotlib.pyplot as plt

# Create publication-quality figure
fig, ax = setup_figure(width=6, height=4)

# Plot embeddings (colored by class)
colors = get_color_palette('class')
normal_mask = (labels == 0)
attack_mask = (labels == 1)

ax.scatter(z[normal_mask, 0], z[normal_mask, 1],
          c=colors[0], label='Normal', alpha=0.6, s=10)
ax.scatter(z[attack_mask, 0], z[attack_mask, 1],
          c=colors[1], label='Attack', alpha=0.6, s=10)

ax.set_xlabel('Latent Dimension 1')
ax.set_ylabel('Latent Dimension 2')
ax.set_title('VGAE Latent Space')
ax.legend()
ax.grid(True, alpha=0.3)

# Display in notebook (reactive!)
fig

# Save when satisfied
save_figure(fig, 'latent_space_exploration', output_dir='../figures')
```

---

## Features

### ✅ Reactive Updates

When you change a cell, **all dependent cells automatically re-run**:

```python
# Cell 1
checkpoint_path = "path/to/model.pth"

# Cell 2 (depends on Cell 1)
model, config = load_model(checkpoint_path)

# Cell 3 (depends on Cell 2)
embeddings = extract_embeddings(model, data)

# Cell 4 (depends on Cell 3)
plot_embeddings(embeddings)
```

**Change Cell 1** → Cells 2, 3, 4 automatically re-run!

### ✅ GPU Acceleration

The SLURM job requests a GPU, so you can:
- Load large models
- Process batches quickly
- Extract embeddings efficiently

Check GPU usage:
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### ✅ Config-Driven Loading

Uses the same pattern as evaluation:
- No hardcoded dimensions
- Automatic compatibility
- Works with all model types (teacher, student, KD, fusion)

### ✅ Auto-Save

Notebooks save automatically as you type. No need to Ctrl+S!

### ✅ Export Python Files

Your notebook IS a Python file. You can:
- Run it standalone: `python visualization_workspace.py`
- Import functions: `from visualization_workspace import my_plot`
- Version control easily

---

## Managing the Server

### Check Job Status

```bash
# View job queue
squeue -u $USER

# Check specific job
squeue -j <JOB_ID>

# View output log (live)
tail -f slurm_logs/marimo_<JOB_ID>.out

# View connection instructions again
cat slurm_logs/marimo_<JOB_ID>_connection.txt
```

### Stop the Server

```bash
scancel <JOB_ID>
```

### Extend Time

If you need more time:

```bash
# Stop current job
scancel <JOB_ID>

# Start new job with more time
./start_marimo.sh 8  # 8 hours
```

---

## Creating New Notebooks

### From Marimo UI

1. In the Marimo interface, click "File" → "New"
2. Start writing cells
3. Save with a descriptive name

### From Command Line

```bash
cd visualizations/notebooks
marimo edit my_new_notebook.py
```

### From Template

```python
"""
My New Visualization

Description of what this notebook does.
"""

import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Add visualization utilities
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))

    from model_loader import load_model_for_visualization
    from data_loader import load_data_for_visualization
    from utils import setup_figure, save_figure

    mo.md("# My New Visualization")
    return mo, plt, np, sys, Path


@app.cell
def __(mo):
    # Your code here
    mo.md("Add your cells here...")
    return


if __name__ == "__main__":
    app.run()
```

---

## Troubleshooting

### Issue: "Job stays pending"

**Cause**: Waiting for GPU resources

**Solution**:
```bash
# Check reason
squeue -j <JOB_ID> -o "%R"

# If waiting too long, use CPU-only
scancel <JOB_ID>
./start_marimo.sh 4 cpu
```

### Issue: "Cannot connect to server"

**Cause**: SSH tunnel not set up correctly

**Solution**:
1. Check connection instructions: `cat slurm_logs/marimo_<JOB_ID>_connection.txt`
2. Verify SSH tunnel command
3. Make sure tunnel terminal is still open
4. Try accessing: `http://localhost:<PORT>`

### Issue: "Marimo not found"

**Cause**: Marimo not installed in conda environment

**Solution**:
The SLURM script auto-installs Marimo, but you can also install manually:
```bash
conda activate gnn-experiments
pip install marimo
```

### Issue: "Import errors for model_loader"

**Cause**: Working directory not set correctly

**Solution**:
```python
# Add to top of notebook
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
```

### Issue: "Out of memory"

**Cause**: Loading too much data or too large model

**Solution**:
```python
# Limit samples
data = load_data_for_visualization(
    config_path=config_path,
    splits=['test'],
    max_samples=2000  # Reduce from 5000
)

# Reduce batch size
embeddings = loader.extract_vgae_embeddings(
    model=model,
    data_list=data_list,
    batch_size=16  # Reduce from 64
)
```

---

## Best Practices

### 1. Use Small Samples for Development

```python
# Start with small sample for fast iteration
data = load_data_for_visualization(
    config_path=config_path,
    splits=['test'],
    max_samples=1000  # Quick for testing
)

# Once visualization is working, scale up
data = load_data_for_visualization(
    config_path=config_path,
    splits=['test'],
    max_samples=10000  # Full dataset
)
```

### 2. Separate Cells for Long Operations

```python
# Cell 1: Load model (slow, only runs once)
model, config = load_model_for_visualization(checkpoint_path)

# Cell 2: Extract embeddings (slow, only runs if model changes)
embeddings = loader.extract_vgae_embeddings(model, data)

# Cell 3: Plot (fast, runs every time you change plot)
fig, ax = setup_figure(width=6, height=4)
ax.scatter(embeddings['z'][:, 0], embeddings['z'][:, 1])
fig
```

### 3. Cache Expensive Computations

```python
# Use @functools.lru_cache for expensive operations
from functools import lru_cache

@lru_cache(maxsize=1)
def load_cached_embeddings(checkpoint_path):
    model, config = load_model_for_visualization(checkpoint_path)
    data = load_data_for_visualization(config_path, splits=['test'])
    loader = ModelLoader()
    return loader.extract_vgae_embeddings(model, data['test'])
```

### 4. Save Final Figures

```python
# Develop interactively
fig, ax = setup_figure(width=6, height=4)
# ... plotting code ...
fig  # Display in notebook

# When satisfied, save
save_figure(fig, 'fig2_embeddings', output_dir='../figures')
```

### 5. Use UI Elements for Parameters

```python
import marimo as mo

# Interactive sliders
n_samples = mo.ui.slider(1000, 10000, value=5000, label="Samples")
n_samples

# Use value
data = load_data_for_visualization(
    config_path=config_path,
    max_samples=n_samples.value
)
```

---

## Comparison: Marimo vs Jupyter

| Feature | Marimo | Jupyter |
|---------|--------|---------|
| **Reactivity** | ✅ Automatic | ❌ Manual re-run |
| **Hidden state** | ✅ Impossible | ❌ Common problem |
| **File format** | ✅ Pure Python | ❌ JSON (.ipynb) |
| **Version control** | ✅ Clean diffs | ❌ Messy diffs |
| **Reproducibility** | ✅ Guaranteed | ⚠️ Depends on run order |
| **IDE support** | ✅ Full Python | ⚠️ Limited |
| **Execution order** | ✅ Topological | ❌ Linear |

**For publication figures**: Marimo is superior because it guarantees reproducibility!

---

## Next Steps

1. **Start the server**: `./start_marimo.sh`
2. **Connect**: Follow SSH tunnel instructions
3. **Explore**: Open default workspace notebook
4. **Develop**: Create publication figures interactively
5. **Export**: Save final figures with `save_figure()`

---

## Resources

- **Marimo Documentation**: https://docs.marimo.io/
- **Marimo Examples**: https://marimo.io/examples
- **Visualization Plan**: [VISUALIZATIONS_PLAN.md](VISUALIZATIONS_PLAN.md)
- **Model Loading**: [visualizations/README.md](visualizations/README.md)
- **Config-Driven Approach**: [EVALUATION_FIX_SUMMARY.md](EVALUATION_FIX_SUMMARY.md)

---

## Summary

You now have a **SLURM-integrated Marimo notebook server** for interactive visualization development:

✅ Launches on compute nodes with GPU
✅ Uses config-driven model/data loading
✅ Reactive updates for fast iteration
✅ Publication-quality figure generation
✅ Auto-saves and version control friendly

**To start**:
```bash
./start_marimo.sh
```

Happy visualizing! 🎨
