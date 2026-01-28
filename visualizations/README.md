# Visualization Infrastructure - Config-Driven Approach

**Date**: 2026-01-28
**Status**: ✅ Infrastructure Complete

This directory contains publication-quality visualization scripts using **config-driven model and data loading** to ensure consistency with training and evaluation.

---

## Core Philosophy: Config-Driven Loading

**Key Insight**: If you can path to a checkpoint, you can path to its frozen config. If you have the frozen config, you can instantiate the correct model architecture.

**The Pattern**:
```
Checkpoint Path → Frozen Config Discovery → Load Frozen Config →
Instantiate LightningModule → Load Weights → Extract Model/Embeddings
```

**Benefits**:
- ✅ **Zero duplication**: Reuses training/evaluation model-building logic
- ✅ **Automatic consistency**: Models match exactly what was trained
- ✅ **Correct dimensions**: No more hardcoded parameter mismatches
- ✅ **Handles all variants**: Teacher, student, student-KD, fusion models work seamlessly
- ✅ **Maintainable**: Single source of truth for model architecture

---

## Directory Structure

```
visualizations/
├── README.md                       # This file
├── __init__.py
├── utils.py                        # Plotting utilities, color palettes, figure saving
├── model_loader.py                 # Config-driven model loading ⭐ NEW
├── data_loader.py                  # Config-driven data loading ⭐ NEW
│
├── demo_visualization.py           # Test script for infrastructure
├── embedding_umap.py               # Figure 2: Embedding spaces ⭐ NEW
├── performance_comparison.py       # Figure 5: Performance comparison ⭐ NEW
│
└── [Future scripts]
    ├── vgae_reconstruction.py      # Figure 3: VGAE reconstruction analysis
    ├── dqn_policy_analysis.py      # Figure 4: DQN alpha selection strategy
    ├── roc_pr_curves.py            # Figure 6: ROC and PR curves
    ├── ablation_state_space.py     # Figure 7: 15D state space ablation
    ├── kd_impact.py                # Figure 8: Knowledge distillation impact
    └── ...
```

---

## Quick Start

### 1. Load a Model for Visualization

```python
from model_loader import load_model_for_visualization

# Automatic config discovery
model, config = load_model_for_visualization(
    checkpoint_path="experimentruns/automotive/hcrl_sa/.../models/vgae_teacher.pth",
    device='cuda'
)

# Model is ready to use!
# Config contains all training parameters (hidden_dims, latent_dim, etc.)
```

### 2. Load Dataset Using Config

```python
from data_loader import load_data_for_visualization

# Load using the same config as the model
data = load_data_for_visualization(
    config_path="experimentruns/automotive/hcrl_sa/.../configs/frozen_config.json",
    splits=['test'],
    max_samples=5000  # Limit for visualization efficiency
)

test_data = data['test']
```

### 3. Extract Embeddings

```python
from model_loader import ModelLoader

loader = ModelLoader(device='cuda')

# For VGAE
embeddings = loader.extract_vgae_embeddings(
    model=model,
    data_list=test_data,
    batch_size=64
)

# embeddings contains: z_mean, z_log_var, z, labels

# For GAT
embeddings = loader.extract_gat_embeddings(
    model=model,
    data_list=test_data,
    batch_size=64
)

# embeddings contains: pre_pooling, post_pooling, logits, labels
```

### 4. Generate a Figure

```bash
# Generate embedding space visualization
python visualizations/embedding_umap.py \
  --checkpoint experimentruns/automotive/hcrl_sa/.../models/vgae_teacher.pth \
  --split test \
  --max-samples 5000 \
  --output-dir figures

# Generate performance comparison
python visualizations/performance_comparison.py \
  --results-dirs \
    hcrl_sa:evaluation_results/hcrl_sa/teacher \
    hcrl_ch:evaluation_results/hcrl_ch/teacher \
  --metrics accuracy f1 \
  --output-dir figures
```

---

## Core Utilities

### `model_loader.py` - Config-Driven Model Loading

**Classes**:
- `ModelLoader`: Main utility for loading models

**Key Methods**:
```python
loader = ModelLoader(device='cuda')

# Load single model (VGAE or GAT)
model, config = loader.load_model(
    checkpoint_path="path/to/model.pth",
    config_path=None,  # Auto-discovered
    num_ids=None,      # Inferred from checkpoint
)

# Load fusion model with sub-models
vgae, gat, dqn, config = loader.load_fusion_model(
    checkpoint_path="path/to/dqn.pth",
    vgae_checkpoint_path="path/to/vgae.pth",
    gat_checkpoint_path="path/to/gat.pth"
)

# Extract embeddings
vgae_embeddings = loader.extract_vgae_embeddings(model, data_list)
gat_embeddings = loader.extract_gat_embeddings(model, data_list)

# Compute reconstruction errors (for VGAE analysis)
errors = loader.compute_vgae_reconstruction_errors(model, data_list)
```

**Automatic Config Discovery**:
```
experiment_dir/
├── configs/
│   └── frozen_config_TIMESTAMP.json  ← Auto-discovered from checkpoint path
└── models/
    └── model.pth  ← Provide this path
```

**Handles All Model Types**:
- VGAE Teacher
- VGAE Student
- VGAE Student + KD
- GAT Teacher
- GAT Student
- GAT Student + KD
- Fusion (DQN + VGAE + GAT)

---

### `data_loader.py` - Config-Driven Data Loading

**Classes**:
- `DataLoader`: Main utility for loading datasets

**Key Methods**:
```python
loader = DataLoader()

# Load using frozen config (recommended)
datasets = loader.load_dataset_from_config(
    config_path="path/to/frozen_config.json",
    splits=['train', 'val', 'test'],
    max_samples_per_split=5000
)

train_data = datasets['train']
val_data = datasets['val']
test_data = datasets['test']

# Load by dataset name (without config)
datasets = loader.load_dataset_by_name(
    dataset_name='hcrl_sa',
    modality='automotive',
    splits=['test'],
    max_samples_per_split=5000
)

# Get class distribution
distribution = loader.get_class_distribution(test_data)
# Returns: {0: 4200, 1: 800}  # normal: 4200, attack: 800
```

**Additional Functions**:
```python
from data_loader import load_evaluation_results, load_dqn_predictions

# Load evaluation CSVs
results = load_evaluation_results(
    results_dir='evaluation_results/hcrl_sa/teacher'
)
vgae_metrics = results['vgae_teacher_autoencoder_run_003']

# Load DQN predictions with 15D states and alpha values
states, alphas, preds, labels = load_dqn_predictions(
    'evaluation_results/hcrl_sa/fusion/dqn_predictions.npz'
)
```

---

### `utils.py` - Plotting Utilities

**Core Functions**:
```python
from utils import (
    setup_figure,
    save_figure,
    get_color_palette,
    annotate_bars,
    compute_confidence_intervals
)

# Create publication-quality figure
fig, axes = setup_figure(
    width=7.0,        # IEEE column width
    height=4.5,
    nrows=2,
    ncols=2,
    style='../paper_style.mplstyle'
)

# Get consistent colors
colors = get_color_palette('colorblind')  # Colorblind-friendly
model_colors = get_color_palette('model')  # Model-specific colors
class_colors = get_color_palette('class')  # Normal (green), Attack (red)

# Save in multiple formats
saved_files = save_figure(
    fig=fig,
    filename='my_figure',
    output_dir='figures',
    formats=['pdf', 'png'],
    dpi=300
)
```

**Color Palettes**:
- `'colorblind'`: Safe for colorblind readers
- `'model'`: VGAE (blue), GAT (orange), Fusion (purple)
- `'class'`: Normal (green), Attack (red)

---

## Example Scripts

### `embedding_umap.py` - Embedding Space Visualization

**Purpose**: Generate Figure 2 from the visualization plan - shows how embeddings separate classes

**Usage**:
```bash
python visualizations/embedding_umap.py \
  --checkpoint experimentruns/automotive/hcrl_sa/.../models/vgae_teacher.pth \
  --split test \
  --max-samples 5000 \
  --method umap \
  --output-dir figures \
  --output-name fig2_embeddings
```

**Output**: 2D UMAP/PyMDE projections of:
- VGAE latent space (z)
- VGAE mean (μ)
- GAT embeddings (if GAT model)

**Key Features**:
- Auto-discovers frozen config
- Extracts embeddings using `ModelLoader`
- Reduces dimensionality with UMAP or PyMDE
- Publication-quality scatter plots with class coloring

---

### `performance_comparison.py` - Performance Comparison

**Purpose**: Generate Figure 5 - compare models across datasets and metrics

**Usage**:
```bash
python visualizations/performance_comparison.py \
  --results-dirs \
    hcrl_sa:evaluation_results/hcrl_sa/teacher \
    hcrl_ch:evaluation_results/hcrl_ch/teacher \
  --metrics accuracy f1 precision recall \
  --output-dir figures \
  --output-name fig5_performance
```

**Output**: Bar charts showing:
- Accuracy/F1 by model type
- Model size vs performance scatter
- Statistical summaries

**Key Features**:
- Loads evaluation CSVs automatically
- Aggregates metrics across runs
- Generates clean bar charts with annotations
- Supports multiple datasets and metrics

---

## Creating New Visualization Scripts

### Template Pattern

```python
"""
Figure X: Your Figure Title

Purpose: What this figure demonstrates

Uses config-driven model and data loading for consistency.
"""

import logging
from model_loader import ModelLoader, load_model_for_visualization
from data_loader import DataLoader, load_data_for_visualization
from utils import setup_figure, save_figure, get_color_palette

logger = logging.getLogger(__name__)


def generate_your_figure(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    output_dir: str = '../figures',
    output_filename: str = 'figX_your_figure'
) -> None:
    """
    Generate your figure.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to frozen config (auto-discovered if None)
        output_dir: Where to save figure
        output_filename: Output filename (without extension)
    """
    logger.info("=" * 60)
    logger.info("Figure X: Your Figure Title")
    logger.info("=" * 60)

    # Step 1: Load model
    logger.info("\n[1/4] Loading model...")
    model, config = load_model_for_visualization(
        checkpoint_path=checkpoint_path,
        config_path=config_path
    )

    # Step 2: Load data
    logger.info("\n[2/4] Loading dataset...")
    if config_path is None:
        loader = ModelLoader()
        config_path = loader._discover_frozen_config(checkpoint_path)

    datasets = load_data_for_visualization(
        config_path=config_path,
        splits=['test'],
        max_samples=5000
    )

    # Step 3: Extract/compute what you need
    logger.info("\n[3/4] Extracting embeddings/metrics...")
    # Your analysis here...

    # Step 4: Plot
    logger.info("\n[4/4] Generating figure...")
    fig, ax = setup_figure(width=7, height=4.5)

    # Your plotting code here...

    # Save
    saved_files = save_figure(
        fig=fig,
        filename=output_filename,
        output_dir=output_dir,
        formats=['pdf', 'png'],
        dpi=300
    )

    logger.info(f"\n✓ Figure saved to: {saved_files}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate Figure X')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='../figures')
    parser.add_argument('--output-name', type=str, default='figX_your_figure')

    args = parser.parse_args()

    generate_your_figure(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        output_filename=args.output_name
    )
```

---

## Common Workflows

### Workflow 1: Visualize Embeddings from a Trained Model

```bash
# 1. Train a model (creates checkpoint + frozen_config)
python src/main.py --config configs/vgae_teacher.yaml

# 2. Generate embedding visualization
python visualizations/embedding_umap.py \
  --checkpoint experimentruns/automotive/hcrl_sa/.../models/vgae_teacher.pth \
  --split test \
  --max-samples 5000

# Output: figures/fig2_embeddings.pdf
```

### Workflow 2: Compare Multiple Models' Performance

```bash
# 1. Run evaluation on all models
python src/evaluation/evaluation.py --dataset hcrl_sa --model-path .../vgae_teacher.pth
python src/evaluation/evaluation.py --dataset hcrl_sa --model-path .../gat_teacher.pth
python src/evaluation/evaluation.py --dataset hcrl_sa --model-path .../fusion.pth

# 2. Generate comparison figure
python visualizations/performance_comparison.py \
  --results-dirs hcrl_sa:evaluation_results/hcrl_sa/teacher \
  --metrics accuracy f1

# Output: figures/fig5_performance.pdf
```

### Workflow 3: Analyze VGAE Reconstruction Errors

```python
from model_loader import ModelLoader
from data_loader import load_data_for_visualization

# Load model and data
loader = ModelLoader()
model, config = loader.load_model("path/to/vgae.pth")

data = load_data_for_visualization(
    config_path="path/to/frozen_config.json",
    splits=['test']
)

# Compute reconstruction errors
errors = loader.compute_vgae_reconstruction_errors(
    model=model,
    data_list=data['test']
)

# errors contains: node_error, neighbor_error, id_error, combined_error, labels
# Now plot histograms for Figure 3...
```

---

## Best Practices

### 1. Always Use Config-Driven Loading

**❌ Don't do this**:
```python
# Hardcoded dimensions - will break if checkpoint doesn't match!
model = GraphAutoencoderNeighborhood(
    in_channels=8,
    hidden_dims=[256, 128, 96, 48],
    latent_dim=48,
    embedding_dim=32,
    num_ids=2032
)
model.load_state_dict(checkpoint)
```

**✅ Do this instead**:
```python
# Config-driven - always correct!
from model_loader import load_model_for_visualization

model, config = load_model_for_visualization(
    checkpoint_path="path/to/checkpoint.pth"
)

# Model has correct dimensions automatically:
# - hidden_dims from config
# - latent_dim from config
# - embedding_dim from config
# - num_ids inferred from checkpoint
```

### 2. Limit Sample Sizes for Efficiency

Visualizations don't need all data - 5000-10000 samples is usually sufficient:

```python
data = load_data_for_visualization(
    config_path=config_path,
    splits=['test'],
    max_samples=5000  # Much faster than 50k+ samples
)
```

### 3. Use Logging Instead of Print

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Loading model...")  # ✅ Good
print("Loading model...")        # ❌ Avoid
```

### 4. Save in Multiple Formats

```python
save_figure(
    fig=fig,
    filename='my_figure',
    output_dir='figures',
    formats=['pdf', 'png'],  # PDF for paper, PNG for slides
    dpi=300
)
```

### 5. Follow the Visualization Plan

See [VISUALIZATIONS_PLAN.md](../VISUALIZATIONS_PLAN.md) for comprehensive figure specifications.

---

## Troubleshooting

### Issue: "Could not find frozen config"

**Solution**: Ensure frozen config exists in expected location:
```
experiment_dir/
├── configs/
│   └── frozen_config_*.json  ← Must exist here
└── models/
    └── model.pth
```

Or provide explicit path:
```python
model, config = loader.load_model(
    checkpoint_path="path/to/model.pth",
    config_path="path/to/frozen_config.json"  # Explicit
)
```

### Issue: "Dimension mismatch when loading checkpoint"

**Solution**: This should NOT happen with config-driven loading! If it does:
1. Check that frozen_config.json matches the checkpoint
2. Ensure num_ids is being inferred correctly
3. Report as a bug

### Issue: "Out of memory when extracting embeddings"

**Solution**: Reduce batch size or max samples:
```python
embeddings = loader.extract_vgae_embeddings(
    model=model,
    data_list=data_list,
    batch_size=32  # Reduce from 64
)

# Or limit data
data = load_data_for_visualization(
    config_path=config_path,
    splits=['test'],
    max_samples=2000  # Reduce from 5000
)
```

### Issue: "UMAP/PyMDE not installed"

**Solution**: Install dimensionality reduction libraries:
```bash
pip install umap-learn pymde
```

---

## Dependencies

### Required
- `matplotlib`
- `numpy`
- `pandas`
- `torch`
- `torch_geometric`

### Optional (for specific visualizations)
- `umap-learn` - For UMAP dimensionality reduction
- `pymde` - Alternative dimensionality reduction
- `seaborn` - Additional color palettes
- `shap` - Feature importance analysis (for Figure 4)

### Install all
```bash
pip install matplotlib numpy pandas torch torch_geometric umap-learn pymde seaborn shap
```

---

## Next Steps

1. **Implement remaining figures** from [VISUALIZATIONS_PLAN.md](../VISUALIZATIONS_PLAN.md):
   - Figure 3: VGAE reconstruction analysis
   - Figure 4: DQN policy analysis (15D state usage)
   - Figure 6: ROC/PR curves
   - Figure 7: 15D ablation study
   - Figure 8: Knowledge distillation impact
   - Figures 9-12: Supplementary materials

2. **Create paper_style.mplstyle** with publication settings

3. **Generate all figures** from trained models

4. **Write figure captions** in `figure_captions.md`

---

## Summary

This visualization infrastructure applies the **config-driven loading pattern** to ensure:
- ✅ Models match exactly what was trained
- ✅ Data matches exactly what was evaluated
- ✅ No hardcoded parameters
- ✅ Automatic dimension compatibility
- ✅ Reproducible, publication-quality figures

**Pattern**:
```
Checkpoint → Frozen Config → LightningModule → Model → Embeddings → Visualization
```

For questions or issues, see [EVALUATION_FIX_SUMMARY.md](../EVALUATION_FIX_SUMMARY.md) for the detailed explanation of the config-driven approach.
