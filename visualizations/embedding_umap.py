"""
Figure 2: Embedding Space Visualization (UMAP/PyMDE)

Purpose: Demonstrate learned representations separate attack classes effectively

Layout: 2x3 grid comparing different embedding spaces
- Raw Features, VGAE Latent Space, VGAE Decoder
- GAT Layer 1, GAT Layer 2, GAT Pre-Pooling

Uses config-driven model and data loading for consistency with training/evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Config-driven loaders
from model_loader import ModelLoader
from data_loader import DataLoader, load_data_for_visualization
from utils import (
    setup_figure,
    save_figure,
    get_color_palette
)

# Dimensionality reduction
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. Install with: pip install umap-learn")

# Optional: PyMDE
try:
    import pymde
    HAS_PYMDE = True
except ImportError:
    HAS_PYMDE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reduce_dimensionality(
    embeddings: np.ndarray,
    method: str = 'umap',
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embeddings to 2D for visualization.

    Args:
        embeddings: [N, D] array of embeddings
        method: 'umap' or 'pymde'
        n_components: Target dimensions (default: 2)
        random_state: Random seed

    Returns:
        [N, n_components] reduced embeddings
    """
    if method == 'umap':
        if not HAS_UMAP:
            raise ImportError("umap-learn not installed")

        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean'
        )
        reduced = reducer.fit_transform(embeddings)

    elif method == 'pymde':
        if not HAS_PYMDE:
            raise ImportError("pymde not installed")

        # PyMDE with attractive/repulsive edges
        mde = pymde.preserve_neighbors(
            embeddings,
            embedding_dim=n_components,
            attractive_penalty=pymde.penalties.Log1p,
            repulsive_penalty=pymde.penalties.Log,
            constraint=pymde.Standardized(),
            n_neighbors=15
        )
        reduced = mde.embed().cpu().numpy()

    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap' or 'pymde'")

    return reduced


def plot_embedding_space(
    ax: plt.Axes,
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    colors: List[str],
    show_legend: bool = True
) -> None:
    """
    Plot 2D embedding space with class coloring.

    Args:
        ax: Matplotlib axes
        embeddings_2d: [N, 2] array of 2D embeddings
        labels: [N] array of class labels (0=normal, 1=attack)
        title: Subplot title
        colors: List of colors [normal_color, attack_color]
        show_legend: Whether to show legend
    """
    # Split by class
    normal_mask = (labels == 0)
    attack_mask = (labels == 1)

    normal_emb = embeddings_2d[normal_mask]
    attack_emb = embeddings_2d[attack_mask]

    # Scatter plot
    ax.scatter(
        normal_emb[:, 0], normal_emb[:, 1],
        c=colors[0], label='Normal', alpha=0.6, s=10, edgecolors='none'
    )
    ax.scatter(
        attack_emb[:, 0], attack_emb[:, 1],
        c=colors[1], label='Attack', alpha=0.6, s=10, edgecolors='none'
    )

    # Styling
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Class counts in title
    n_normal = normal_mask.sum()
    n_attack = attack_mask.sum()
    ax.text(
        0.02, 0.98, f'N={n_normal+n_attack}\n(N:{n_normal}, A:{n_attack})',
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=7,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )


def visualize_embeddings_comparison(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    dataset_split: str = 'test',
    max_samples: int = 5000,
    reduction_method: str = 'umap',
    output_dir: str = '../figures',
    output_filename: str = 'fig2_embeddings'
) -> None:
    """
    Generate Figure 2: Embedding space comparison.

    Args:
        checkpoint_path: Path to VGAE or GAT checkpoint
        config_path: Optional path to frozen config (auto-discovered if None)
        dataset_split: Which split to visualize ('train', 'val', 'test')
        max_samples: Max samples to visualize (for efficiency)
        reduction_method: 'umap' or 'pymde'
        output_dir: Where to save figure
        output_filename: Output filename (without extension)
    """
    logger.info("=" * 60)
    logger.info("Figure 2: Embedding Space Visualization")
    logger.info("=" * 60)

    # Step 1: Load model using config-driven approach
    logger.info("\n[1/5] Loading model...")
    model_loader = ModelLoader(device='cuda')
    model, config = model_loader.load_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path
    )

    # Step 2: Load dataset using config
    logger.info("\n[2/5] Loading dataset...")
    if config_path is None:
        config_path = model_loader._discover_frozen_config(checkpoint_path)

    datasets = load_data_for_visualization(
        config_path=config_path,
        splits=[dataset_split],
        max_samples=max_samples
    )
    data_list = datasets[dataset_split]

    logger.info(f"Loaded {len(data_list)} samples from {dataset_split} split")

    # Step 3: Extract embeddings
    logger.info("\n[3/5] Extracting embeddings...")

    model_type = config.model.type

    if model_type in ['vgae', 'vgae_student']:
        # Extract VGAE embeddings
        embeddings_dict = model_loader.extract_vgae_embeddings(
            model=model,
            data_list=data_list,
            batch_size=64
        )

        # Prepare embeddings for visualization
        embeddings_to_plot = {
            'VGAE Latent (z)': embeddings_dict['z'].numpy(),
            'VGAE Mean': embeddings_dict['z_mean'].numpy(),
        }

    elif model_type in ['gat', 'gat_student']:
        # Extract GAT embeddings
        embeddings_dict = model_loader.extract_gat_embeddings(
            model=model,
            data_list=data_list,
            batch_size=64
        )

        embeddings_to_plot = {
            'GAT Logits': embeddings_dict['logits'].numpy(),
        }

        if 'pre_pooling' in embeddings_dict:
            embeddings_to_plot['GAT Pre-Pooling'] = embeddings_dict['pre_pooling'].numpy()

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    labels = embeddings_dict['labels'].numpy()

    # Step 4: Reduce dimensionality
    logger.info(f"\n[4/5] Reducing dimensionality with {reduction_method}...")

    reduced_embeddings = {}
    for name, emb in embeddings_to_plot.items():
        logger.info(f"  Reducing {name}: {emb.shape}")
        reduced = reduce_dimensionality(
            embeddings=emb,
            method=reduction_method,
            n_components=2
        )
        reduced_embeddings[name] = reduced
        logger.info(f"    → {reduced.shape}")

    # Step 5: Plot
    logger.info("\n[5/5] Generating figure...")

    n_plots = len(reduced_embeddings)
    nrows = 1 if n_plots <= 3 else 2
    ncols = min(n_plots, 3)

    fig, axes = setup_figure(
        width=3.5 * ncols,
        height=3.0 * nrows,
        nrows=nrows,
        ncols=ncols
    )

    # Flatten axes for easier indexing
    if n_plots == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    # Class colors
    class_colors = get_color_palette('class')

    # Plot each embedding space
    for idx, (name, emb_2d) in enumerate(reduced_embeddings.items()):
        plot_embedding_space(
            ax=axes[idx],
            embeddings_2d=emb_2d,
            labels=labels,
            title=name,
            colors=class_colors,
            show_legend=(idx == 0)  # Only show legend on first plot
        )

    # Overall title
    model_name = config.model.type.upper()
    model_size = config.model_size.capitalize()
    dataset_name = config.dataset.name.upper().replace('_', '-')

    fig.suptitle(
        f'Embedding Space Visualization: {model_name} {model_size} on {dataset_name}',
        fontsize=13,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    saved_files = save_figure(
        fig=fig,
        filename=output_filename,
        output_dir=output_dir,
        formats=['pdf', 'png'],
        dpi=300
    )

    plt.close(fig)

    logger.info("\n" + "=" * 60)
    logger.info("✓ Figure generated successfully!")
    logger.info("=" * 60)
    logger.info(f"\nSaved to:")
    for path in saved_files:
        logger.info(f"  {path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Figure 2: Embedding Space Visualization'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to frozen config (optional, auto-discovered if not provided)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to visualize'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=5000,
        help='Max samples to visualize (for efficiency)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='umap',
        choices=['umap', 'pymde'],
        help='Dimensionality reduction method'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='../figures',
        help='Output directory for figures'
    )

    parser.add_argument(
        '--output-name',
        type=str,
        default='fig2_embeddings',
        help='Output filename (without extension)'
    )

    args = parser.parse_args()

    # Generate figure
    visualize_embeddings_comparison(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        dataset_split=args.split,
        max_samples=args.max_samples,
        reduction_method=args.method,
        output_dir=args.output_dir,
        output_filename=args.output_name
    )
