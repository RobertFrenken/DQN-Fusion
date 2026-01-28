"""
Figure 5: Performance Comparison - Main Results

Purpose: Primary results figure showing model comparison across datasets

Shows:
- Accuracy and F1-score by model and dataset
- Model size vs performance trade-offs
- Demonstrates KD effectiveness and fusion superiority

Uses config-driven loading to ensure consistency with evaluation results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from data_loader import load_evaluation_results
from utils import (
    setup_figure,
    save_figure,
    get_color_palette,
    annotate_bars,
    DATASET_NAMES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def aggregate_metrics_from_results(
    results_dict: Dict[str, pd.DataFrame],
    metric: str = 'accuracy'
) -> pd.DataFrame:
    """
    Aggregate metrics across models for comparison.

    Args:
        results_dict: Dictionary mapping model_name -> results DataFrame
        metric: Metric to extract ('accuracy', 'f1', 'precision', 'recall', 'auc')

    Returns:
        DataFrame with columns: [model_name, dataset, metric_value, ...]
    """
    aggregated = []

    for model_name, df in results_dict.items():
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found in {model_name} results")
            continue

        # Extract key info from model name
        # Expected format: {model_type}_{model_size}_{training_mode}_run_{num}
        # e.g., "vgae_teacher_autoencoder_run_003"

        parts = model_name.split('_')

        if len(parts) >= 3:
            model_type = parts[0]  # vgae, gat, fusion
            model_size = parts[1]  # teacher, student
            training_mode = parts[2]  # autoencoder, curriculum, fusion
        else:
            model_type = model_name
            model_size = 'unknown'
            training_mode = 'unknown'

        # Get metric value (average if multiple rows)
        metric_value = df[metric].mean() if len(df) > 1 else df[metric].iloc[0]

        aggregated.append({
            'model_name': model_name,
            'model_type': model_type,
            'model_size': model_size,
            'training_mode': training_mode,
            'metric': metric,
            'value': metric_value
        })

    return pd.DataFrame(aggregated)


def plot_metric_comparison_bars(
    ax: plt.Axes,
    aggregated_df: pd.DataFrame,
    metric: str,
    group_by: str = 'model_type',
    colors: Optional[Dict[str, str]] = None,
    show_values: bool = True
) -> None:
    """
    Plot bar chart comparing metric across models.

    Args:
        ax: Matplotlib axes
        aggregated_df: DataFrame with aggregated metrics
        metric: Metric name for y-axis label
        group_by: How to group bars ('model_type', 'model_size', 'training_mode')
        colors: Optional color mapping
        show_values: Whether to annotate bars with values
    """
    # Group data
    grouped = aggregated_df.groupby(group_by)['value'].mean().reset_index()

    # Sort by value descending
    grouped = grouped.sort_values('value', ascending=False)

    # Colors
    if colors is None:
        palette = get_color_palette('model')
        colors = {name: palette[i % len(palette)] for i, name in enumerate(grouped[group_by])}

    bar_colors = [colors.get(name, '#888888') for name in grouped[group_by]]

    # Plot
    bars = ax.bar(
        grouped[group_by],
        grouped['value'],
        color=bar_colors,
        edgecolor='black',
        linewidth=1.2,
        alpha=0.9
    )

    # Annotate
    if show_values:
        annotate_bars(ax, bars, grouped['value'].values, fmt='.3f', offset=0.01)

    # Styling
    ax.set_ylabel(metric.capitalize(), fontsize=10)
    ax.set_title(f'{metric.capitalize()} by {group_by.replace("_", " ").title()}', fontsize=11)
    ax.set_ylim([max(0, grouped['value'].min() - 0.05), min(1.0, grouped['value'].max() + 0.05)])
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(grouped[group_by], rotation=45, ha='right')


def plot_size_vs_performance(
    ax: plt.Axes,
    model_info: List[Dict],
    colors: Optional[Dict[str, str]] = None
) -> None:
    """
    Plot model size vs performance scatter.

    Args:
        ax: Matplotlib axes
        model_info: List of dicts with keys: {name, params, f1_score, model_type}
        colors: Optional color mapping by model_type
    """
    if colors is None:
        palette = get_color_palette('model')
        model_types = list(set(m['model_type'] for m in model_info))
        colors = {mt: palette[i % len(palette)] for i, mt in enumerate(model_types)}

    # Plot each model
    for model in model_info:
        ax.scatter(
            model['params'] / 1e6,  # Convert to millions
            model['f1_score'],
            s=100,
            c=colors.get(model['model_type'], '#888888'),
            label=model['name'],
            alpha=0.8,
            edgecolors='black',
            linewidths=1.5
        )

        # Annotate
        ax.text(
            model['params'] / 1e6,
            model['f1_score'] + 0.01,
            model['name'],
            fontsize=7,
            ha='center',
            va='bottom'
        )

    # Styling
    ax.set_xlabel('Model Parameters (Millions)', fontsize=10)
    ax.set_ylabel('F1-Score', fontsize=10)
    ax.set_title('Model Size vs Performance', fontsize=11)
    ax.set_xscale('log')
    ax.set_ylim([0.8, 1.0])
    ax.grid(True, alpha=0.3)


def generate_performance_comparison_figure(
    results_dirs: Dict[str, str],
    metrics: List[str] = ['accuracy', 'f1'],
    output_dir: str = '../figures',
    output_filename: str = 'fig5_performance'
) -> None:
    """
    Generate Figure 5: Performance comparison across models and datasets.

    Args:
        results_dirs: Dictionary mapping dataset_name -> results_directory
                     e.g., {'hcrl_sa': 'evaluation_results/hcrl_sa/teacher'}
        metrics: List of metrics to plot
        output_dir: Where to save figure
        output_filename: Output filename (without extension)

    Example:
        generate_performance_comparison_figure(
            results_dirs={
                'hcrl_sa': 'evaluation_results/hcrl_sa/teacher',
                'hcrl_ch': 'evaluation_results/hcrl_ch/teacher'
            },
            metrics=['accuracy', 'f1'],
            output_dir='figures'
        )
    """
    logger.info("=" * 60)
    logger.info("Figure 5: Performance Comparison")
    logger.info("=" * 60)

    # Step 1: Load all evaluation results
    logger.info("\n[1/3] Loading evaluation results...")
    all_results = {}

    for dataset_name, results_dir in results_dirs.items():
        logger.info(f"  Loading {dataset_name} from {results_dir}")
        dataset_results = load_evaluation_results(results_dir)
        all_results[dataset_name] = dataset_results

    # Step 2: Aggregate metrics
    logger.info("\n[2/3] Aggregating metrics...")

    aggregated_metrics = {}

    for dataset_name, results_dict in all_results.items():
        for metric in metrics:
            logger.info(f"  Aggregating {metric} for {dataset_name}")
            agg_df = aggregate_metrics_from_results(results_dict, metric=metric)
            agg_df['dataset'] = dataset_name
            aggregated_metrics[f"{dataset_name}_{metric}"] = agg_df

    # Combine all
    combined_df = pd.concat(aggregated_metrics.values(), ignore_index=True)

    # Step 3: Create visualizations
    logger.info("\n[3/3] Generating figure...")

    n_metrics = len(metrics)
    fig, axes = setup_figure(
        width=7.0,
        height=3.5 * n_metrics,
        nrows=n_metrics,
        ncols=1
    )

    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for idx, metric in enumerate(metrics):
        metric_df = combined_df[combined_df['metric'] == metric]

        plot_metric_comparison_bars(
            ax=axes[idx],
            aggregated_df=metric_df,
            metric=metric,
            group_by='model_type',
            show_values=True
        )

    # Overall title
    fig.suptitle(
        'Model Performance Comparison Across Datasets',
        fontsize=14,
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

    # Print summary statistics
    logger.info("\nSummary Statistics:")
    for metric in metrics:
        metric_df = combined_df[combined_df['metric'] == metric]
        logger.info(f"\n{metric.upper()}:")
        summary = metric_df.groupby('model_type')['value'].agg(['mean', 'std', 'min', 'max'])
        logger.info(summary.to_string())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Figure 5: Performance Comparison'
    )

    parser.add_argument(
        '--results-dirs',
        type=str,
        nargs='+',
        required=True,
        help='Paths to evaluation results directories (format: dataset_name:path)'
    )

    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['accuracy', 'f1'],
        help='Metrics to plot'
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
        default='fig5_performance',
        help='Output filename (without extension)'
    )

    args = parser.parse_args()

    # Parse results_dirs from format "dataset:path"
    results_dirs = {}
    for item in args.results_dirs:
        if ':' in item:
            dataset, path = item.split(':', 1)
            results_dirs[dataset] = path
        else:
            # Assume format: evaluation_results/{dataset}/...
            path_parts = Path(item).parts
            if 'evaluation_results' in path_parts:
                idx = path_parts.index('evaluation_results')
                if idx + 1 < len(path_parts):
                    dataset = path_parts[idx + 1]
                    results_dirs[dataset] = item
            else:
                logger.warning(f"Could not parse dataset name from: {item}")

    if not results_dirs:
        logger.error("No valid results directories provided")
        exit(1)

    # Generate figure
    generate_performance_comparison_figure(
        results_dirs=results_dirs,
        metrics=args.metrics,
        output_dir=args.output_dir,
        output_filename=args.output_name
    )
