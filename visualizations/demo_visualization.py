"""
Demo script to test visualization infrastructure.

Generates sample figures to verify the setup is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    setup_figure,
    save_figure,
    get_color_palette,
    annotate_bars,
    compute_confidence_intervals
)

# Optional seaborn
try:
    import seaborn as sns
except ImportError:
    sns = None


def demo_line_plot():
    """Demo: Simple line plot with publication styling."""
    print("\n1. Testing line plot...")

    fig, ax = setup_figure(width=6, height=4)

    # Generate sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    colors = get_color_palette('colorblind')

    ax.plot(x, y1, label='sin(x)', color=colors[0], linewidth=2)
    ax.plot(x, y2, label='cos(x)', color=colors[1], linewidth=2)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('Demo: Line Plot with Publication Style')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'demo_line_plot', output_dir='../figures/test')
    plt.close(fig)
    print("  ✓ Line plot saved")


def demo_bar_chart():
    """Demo: Bar chart with error bars and annotations."""
    print("\n2. Testing bar chart...")

    fig, ax = setup_figure(width=7, height=4.5)

    # Sample data: model performance
    models = ['VGAE', 'GAT', 'Fusion 2D', 'Fusion 15D']
    accuracies = np.array([0.92, 0.94, 0.96, 0.98])
    errors = np.array([0.02, 0.015, 0.012, 0.008])

    colors = [get_color_palette('model')[i] for i in [0, 1, 2, 3]]

    bars = ax.bar(models, accuracies, yerr=errors, capsize=5,
                 color=colors, edgecolor='black', linewidth=1.2)

    # Annotate bars
    annotate_bars(ax, bars, accuracies, fmt='.3f', offset=0.01)

    ax.set_ylabel('Accuracy')
    ax.set_title('Demo: Model Performance Comparison')
    ax.set_ylim([0.88, 1.0])
    ax.grid(axis='y', alpha=0.3)

    save_figure(fig, 'demo_bar_chart', output_dir='../figures/test')
    plt.close(fig)
    print("  ✓ Bar chart saved")


def demo_heatmap():
    """Demo: Heatmap for confusion matrix or correlation."""
    print("\n3. Testing heatmap...")

    fig, ax = setup_figure(width=5, height=4.5)

    # Sample confusion matrix
    conf_matrix = np.array([
        [850, 50],
        [30, 920]
    ])

    im = ax.imshow(conf_matrix, cmap='Blues', aspect='auto')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, conf_matrix[i, j],
                          ha="center", va="center", color="black", fontsize=14)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Attack'])
    ax.set_yticklabels(['Normal', 'Attack'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Demo: Confusion Matrix')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=15)

    save_figure(fig, 'demo_heatmap', output_dir='../figures/test')
    plt.close(fig)
    print("  ✓ Heatmap saved")


def demo_scatter_plot():
    """Demo: Scatter plot with density."""
    print("\n4. Testing scatter plot...")

    fig, ax = setup_figure(width=6, height=5)

    # Sample data: 2D embeddings
    np.random.seed(42)
    normal_data = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], 500)
    attack_data = np.random.multivariate_normal([-2, -1], [[0.8, -0.3], [-0.3, 0.8]], 300)

    class_colors = get_color_palette('class')

    ax.scatter(normal_data[:, 0], normal_data[:, 1],
              alpha=0.5, s=20, c=class_colors[0], label='Normal', edgecolors='none')
    ax.scatter(attack_data[:, 0], attack_data[:, 1],
              alpha=0.5, s=20, c=class_colors[1], label='Attack', edgecolors='none')

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Demo: Embedding Space (Simulated)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'demo_scatter', output_dir='../figures/test')
    plt.close(fig)
    print("  ✓ Scatter plot saved")


def demo_subplots():
    """Demo: Multi-panel figure."""
    print("\n5. Testing subplots...")

    fig, axes = setup_figure(width=7, height=6, nrows=2, ncols=2)

    colors = get_color_palette('colorblind')

    # Panel 1: Line plot
    x = np.linspace(0, 10, 100)
    axes[0, 0].plot(x, np.sin(x), color=colors[0])
    axes[0, 0].set_title('(a) Training Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Bar chart
    data = [0.85, 0.92, 0.88]
    axes[0, 1].bar(['A', 'B', 'C'], data, color=colors[1:4])
    axes[0, 1].set_title('(b) Model Comparison')
    axes[0, 1].set_ylabel('Accuracy')

    # Panel 3: Histogram
    data = np.random.normal(0, 1, 1000)
    axes[1, 0].hist(data, bins=30, color=colors[4], alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('(c) Error Distribution')
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].set_ylabel('Frequency')

    # Panel 4: Scatter
    x = np.random.randn(200)
    y = 2*x + np.random.randn(200)*0.5
    axes[1, 1].scatter(x, y, color=colors[5], alpha=0.6, s=15)
    axes[1, 1].set_title('(d) Correlation')
    axes[1, 1].set_xlabel('Variable X')
    axes[1, 1].set_ylabel('Variable Y')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'demo_subplots', output_dir='../figures/test', tight=False)
    plt.close(fig)
    print("  ✓ Multi-panel figure saved")


if __name__ == '__main__':
    print("=" * 60)
    print("CAN-Graph Visualization Infrastructure Demo")
    print("=" * 60)

    try:
        demo_line_plot()
        demo_bar_chart()
        demo_heatmap()
        demo_scatter_plot()
        demo_subplots()

        print("\n" + "=" * 60)
        print("✓ All demo figures generated successfully!")
        print("=" * 60)
        print("\nCheck figures/test/ directory for output files.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
