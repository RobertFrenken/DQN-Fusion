#!/bin/bash
#SBATCH --job-name=marimo_viz
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --account=PAS3209
#SBATCH --partition=gpu
#SBATCH --chdir=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT
#SBATCH --output=slurm_logs/marimo_%j.out
#SBATCH --error=slurm_logs/marimo_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=firstname.lastname@osc.edu

# ============================================================================
# Launch Marimo Notebook for Interactive Visualization Development
# ============================================================================
#
# Purpose: Start a Marimo server on a compute node for interactive
#          visualization development with GPU access.
#
# Marimo is a reactive notebook that's better than Jupyter for:
# - Reproducible figures (no hidden state)
# - Real-time reactivity (auto-updates on code changes)
# - Clean Python files (notebooks are valid .py files)
#
# After job starts, you'll need to set up SSH tunnel to access the notebook.
# Instructions will be printed in the output log.
# ============================================================================

set -euo pipefail

echo "=================================================================="
echo "🎨 CAN-Graph Marimo Visualization Server"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=================================================================="

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Load environment
echo "Loading environment..."
module load miniconda3/24.1.2-py310 || true
source activate gnn-experiments || true
module load cuda/12.3.0 || module load cuda/11.8.0 || true

# Environment configuration
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "Python: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Check if marimo is installed
if ! python -c "import marimo" 2>/dev/null; then
    echo "⚠️  Marimo not found. Installing..."
    pip install marimo
    echo "✅ Marimo installed"
fi

# Get the compute node hostname and a random port
COMPUTE_NODE=$(hostname)
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "=================================================================="
echo "🚀 Starting Marimo Server"
echo "=================================================================="
echo "Compute Node: $COMPUTE_NODE"
echo "Port: $PORT"
echo "Working Directory: $(pwd)"
echo ""
echo "Server will start in 5 seconds..."
echo "=================================================================="
sleep 5

# Create a default notebook if none exists
NOTEBOOK_DIR="visualizations/notebooks"
mkdir -p "$NOTEBOOK_DIR"

DEFAULT_NOTEBOOK="$NOTEBOOK_DIR/visualization_workspace.py"

if [ ! -f "$DEFAULT_NOTEBOOK" ]; then
    echo "Creating default visualization notebook..."
    cat > "$DEFAULT_NOTEBOOK" << 'EOF'
"""
CAN-Graph Visualization Workspace

Interactive notebook for developing publication-quality visualizations.

Uses config-driven model and data loading for consistency with training/evaluation.
"""

import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from pathlib import Path

    # Add parent directory to path
    import sys
    sys.path.insert(0, str(Path.cwd().parent))

    mo.md("""
    # CAN-Graph Visualization Workspace

    Interactive development environment for publication figures.

    ## Quick Start

    1. **Load a model**: Use `model_loader.load_model_for_visualization()`
    2. **Load data**: Use `data_loader.load_data_for_visualization()`
    3. **Extract embeddings**: Use `model_loader.extract_vgae_embeddings()`
    4. **Generate figures**: Use utilities from `utils.py`
    """)
    return mo, plt, np, pd, torch, Path, sys


@app.cell
def __(mo):
    # Model and data loading utilities
    from model_loader import ModelLoader, load_model_for_visualization
    from data_loader import DataLoader, load_data_for_visualization
    from utils import (
        setup_figure,
        save_figure,
        get_color_palette,
        annotate_bars
    )

    mo.md("""
    ## 1. Load a Trained Model

    Specify the path to a checkpoint. The frozen config will be auto-discovered.
    """)
    return (
        ModelLoader,
        load_model_for_visualization,
        DataLoader,
        load_data_for_visualization,
        setup_figure,
        save_figure,
        get_color_palette,
        annotate_bars,
    )


@app.cell
def __(mo):
    # Checkpoint path selector
    checkpoint_path = mo.ui.text(
        value="experimentruns/automotive/hcrl_sa/unsupervised/vgae/teacher/no_distillation/autoencoder/models/vgae_teacher_autoencoder_run_003.pth",
        label="Checkpoint path:",
        full_width=True
    )
    checkpoint_path
    return checkpoint_path,


@app.cell
def __(checkpoint_path, load_model_for_visualization, mo):
    # Load button
    load_button = mo.ui.button(label="Load Model", on_click=lambda: None)
    load_button

    # Load model when button clicked
    if load_button.value:
        try:
            model, config = load_model_for_visualization(
                checkpoint_path=checkpoint_path.value,
                device='cuda'
            )
            mo.md(f"""
            ✅ **Model loaded successfully!**

            - Type: `{config.model.type}`
            - Size: `{config.model_size}`
            - Dataset: `{config.dataset.name}`
            - Hidden dims: `{config.model.hidden_dims}`
            - Latent dim: `{config.model.latent_dim}`
            """)
        except Exception as e:
            mo.md(f"❌ **Error loading model:** {e}")
    else:
        mo.md("👆 Click 'Load Model' to start")
    return load_button, model, config


@app.cell
def __(mo):
    mo.md("""
    ## 2. Example Visualization

    Once the model is loaded, you can extract embeddings and create visualizations.
    """)
    return


@app.cell
def __(mo, plt, np, setup_figure, get_color_palette):
    # Example plot
    fig, ax = setup_figure(width=6, height=4)

    # Generate sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    colors = get_color_palette('colorblind')
    ax.plot(x, y, color=colors[0], linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('sin(X)')
    ax.set_title('Example: Reactive Matplotlib Figure')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig
    return fig, ax, x, y, colors


@app.cell
def __(mo):
    mo.md("""
    ## 3. Next Steps

    - Load real data using `load_data_for_visualization()`
    - Extract embeddings using `ModelLoader.extract_vgae_embeddings()`
    - Create custom visualizations
    - Export figures with `save_figure()`

    All changes are reactive - modify any cell and see updates immediately!
    """)
    return


if __name__ == "__main__":
    app.run()
EOF
    echo "✅ Created default notebook: $DEFAULT_NOTEBOOK"
fi

# Create connection instructions file
INSTRUCTIONS_FILE="slurm_logs/marimo_${SLURM_JOB_ID}_connection.txt"

cat > "$INSTRUCTIONS_FILE" << EOF
================================================================
🎨 Marimo Visualization Server - Connection Instructions
================================================================

Job ID: $SLURM_JOB_ID
Compute Node: $COMPUTE_NODE
Port: $PORT

================================================================
STEP 1: Set up SSH tunnel (run on your local machine)
================================================================

ssh -L ${PORT}:${COMPUTE_NODE}:${PORT} rf15@owens.osc.edu

Keep this terminal open!

================================================================
STEP 2: Access Marimo in your browser
================================================================

Open this URL in your browser:
http://localhost:${PORT}

================================================================
STEP 3: Start working!
================================================================

Your Marimo notebook is running in:
$(pwd)/$NOTEBOOK_DIR

Features:
✅ Reactive updates (no need to re-run cells)
✅ GPU acceleration available
✅ Config-driven model loading
✅ Publication-quality figure generation

================================================================
STEP 4: When finished
================================================================

To stop the server:
scancel $SLURM_JOB_ID

================================================================
Logs
================================================================

Output: slurm_logs/marimo_${SLURM_JOB_ID}.out
Error: slurm_logs/marimo_${SLURM_JOB_ID}.err

================================================================
Quick Tips
================================================================

- Notebooks are saved as Python files (not .ipynb)
- Changes auto-save
- Use Ctrl+C in the Marimo terminal to stop
- All cells re-run automatically when dependencies change
- No hidden state - always reproducible!

================================================================
EOF

echo ""
echo "=================================================================="
echo "📋 Connection Instructions"
echo "=================================================================="
cat "$INSTRUCTIONS_FILE"
echo "=================================================================="
echo ""
echo "Instructions saved to: $INSTRUCTIONS_FILE"
echo ""
echo "Starting Marimo server now..."
echo ""

# Start Marimo server
# --headless: No auto-browser opening (we're on compute node)
# --host 0.0.0.0: Listen on all interfaces (required for SSH tunnel)
# --port: Use the random port we selected
cd "$NOTEBOOK_DIR"
marimo edit visualization_workspace.py \
    --headless \
    --host 0.0.0.0 \
    --port "$PORT" \
    --no-token \
    2>&1 | tee "../slurm_logs/marimo_${SLURM_JOB_ID}_server.log"

# If marimo exits, log it
echo ""
echo "=================================================================="
echo "Marimo server stopped"
echo "End time: $(date)"
echo "=================================================================="
