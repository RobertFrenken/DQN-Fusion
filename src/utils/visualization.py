"""
Visualization Utilities for CAN Bus Training

Publication-ready plotting and visualization utilities for GAT, VGAE, 
and Fusion training metrics, performance analysis, and results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os

# Default style configuration for publication-ready plots
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'title_size': 16,
    'label_size': 12,
    'legend_size': 10,
    'tick_size': 10,
    'line_width': 2,
    'marker_size': 6,
    'color_palette': 'husl',
    'style': 'whitegrid'
}


def setup_publication_style():
    """Configure matplotlib for publication-ready plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette(PLOT_CONFIG['color_palette'])
    
    plt.rcParams.update({
        'figure.figsize': PLOT_CONFIG['figure_size'],
        'figure.dpi': PLOT_CONFIG['dpi'],
        'font.size': PLOT_CONFIG['label_size'],
        'axes.titlesize': PLOT_CONFIG['title_size'],
        'axes.labelsize': PLOT_CONFIG['label_size'],
        'xtick.labelsize': PLOT_CONFIG['tick_size'],
        'ytick.labelsize': PLOT_CONFIG['tick_size'],
        'legend.fontsize': PLOT_CONFIG['legend_size'],
        'lines.linewidth': PLOT_CONFIG['line_width'],
        'lines.markersize': PLOT_CONFIG['marker_size']
    })


def plot_training_metrics(training_data: List[Dict], output_dir: str, prefix: str = "training"):
    """
    Plot comprehensive training metrics from training logs.
    
    Args:
        training_data: List of training episode dictionaries
        output_dir: Directory to save plots
        prefix: Filename prefix for saved plots
    """
    setup_publication_style()
    
    if not training_data:
        print("No training data provided for plotting")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(training_data)
    
    # Create comprehensive training metrics plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot 1: Loss Evolution
    if 'loss' in df.columns:
        axes[0].plot(df.index, df['loss'], color='red', alpha=0.7, linewidth=1)
        axes[0].plot(df.index, df['loss'].rolling(window=50, center=True).mean(), 
                    color='darkred', linewidth=2, label='50-ep Moving Average')
        axes[0].set_title('Training Loss Evolution', fontweight='bold')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Reward Progression
    if 'total_reward' in df.columns:
        axes[1].plot(df.index, df['total_reward'], color='green', alpha=0.6, linewidth=1)
        axes[1].plot(df.index, df['total_reward'].rolling(window=50, center=True).mean(), 
                    color='darkgreen', linewidth=2, label='50-ep Moving Average')
        axes[1].set_title('Reward Progression', fontweight='bold')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Total Reward')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Accuracy Metrics
    accuracy_cols = [col for col in df.columns if 'accuracy' in col.lower()]
    if accuracy_cols:
        for acc_col in accuracy_cols:
            axes[2].plot(df.index, df[acc_col], label=acc_col.replace('_', ' ').title(), linewidth=2)
        axes[2].set_title('Accuracy Metrics', fontweight='bold')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Accuracy')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1])
    
    # Plot 4: Epsilon Decay (if available)
    if 'epsilon' in df.columns:
        axes[3].plot(df.index, df['epsilon'], color='purple', linewidth=2)
        axes[3].set_title('Exploration Rate (Epsilon)', fontweight='bold')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('Epsilon')
        axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Processing Speed
    if 'processing_time' in df.columns:
        axes[4].plot(df.index, df['processing_time'], color='orange', alpha=0.7, linewidth=1)
        axes[4].plot(df.index, df['processing_time'].rolling(window=20, center=True).mean(), 
                    color='darkorange', linewidth=2, label='20-ep Moving Average')
        axes[4].set_title('Processing Speed', fontweight='bold')
        axes[4].set_xlabel('Episode')
        axes[4].set_ylabel('Time per Episode (s)')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
    
    # Plot 6: Memory Usage (if available)
    memory_cols = [col for col in df.columns if 'memory' in col.lower()]
    if memory_cols:
        for mem_col in memory_cols:
            if 'pct' in mem_col or 'percent' in mem_col:
                axes[5].plot(df.index, df[mem_col], label=mem_col.replace('_', ' ').title(), linewidth=2)
        axes[5].set_title('Memory Utilization', fontweight='bold')
        axes[5].set_xlabel('Episode')
        axes[5].set_ylabel('Memory Usage (%)')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{prefix}_comprehensive_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✓ Comprehensive training metrics saved to {plot_path}")


def plot_performance_comparison(gpu_stats: List[Dict], output_dir: str, 
                              title: str = "GPU Performance Analysis"):
    """
    Plot GPU performance analysis from monitoring data.
    
    Args:
        gpu_stats: List of GPU monitoring statistics
        output_dir: Directory to save plots
        title: Plot title
    """
    setup_publication_style()
    
    if not gpu_stats:
        print("No GPU stats provided for plotting")
        return
    
    df = pd.DataFrame(gpu_stats)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Memory Usage Over Time
    axes[0,0].plot(df['episode'], df['memory_utilization_pct'], 
                  color='blue', linewidth=2, label='Memory Utilization')
    axes[0,0].plot(df['episode'], df['reserved_utilization_pct'], 
                  color='lightblue', linewidth=2, alpha=0.7, label='Reserved Memory')
    axes[0,0].set_title('Memory Utilization Over Training', fontweight='bold')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Memory Usage (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim([0, 100])
    
    # GPU Utilization
    axes[0,1].plot(df['episode'], df['gpu_utilization_pct'], 
                  color='green', linewidth=2)
    axes[0,1].axhline(y=85, color='red', linestyle='--', alpha=0.7, label='Target: 85%')
    axes[0,1].set_title('GPU Utilization Over Training', fontweight='bold')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('GPU Utilization (%)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim([0, 100])
    
    # Memory Allocation in GB
    axes[1,0].plot(df['episode'], df['memory_allocated_gb'], 
                  color='purple', linewidth=2, label='Allocated')
    axes[1,0].plot(df['episode'], df['memory_reserved_gb'], 
                  color='orange', linewidth=2, alpha=0.7, label='Reserved')
    axes[1,0].set_title('Memory Allocation (GB)', fontweight='bold')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Memory (GB)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Performance Distribution
    axes[1,1].hist(df['gpu_utilization_pct'], bins=30, alpha=0.7, color='green', 
                  label=f"Mean: {df['gpu_utilization_pct'].mean():.1f}%")
    axes[1,1].axvline(df['gpu_utilization_pct'].mean(), color='red', 
                     linestyle='--', linewidth=2)
    axes[1,1].set_title('GPU Utilization Distribution', fontweight='bold')
    axes[1,1].set_xlabel('GPU Utilization (%)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'gpu_performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✓ GPU performance analysis saved to {plot_path}")


def plot_model_comparison(results: Dict[str, Dict], output_dir: str, 
                         metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1']):
    """
    Plot comparison between different models or training runs.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        output_dir: Directory to save plots
        metrics: List of metrics to compare
    """
    setup_publication_style()
    
    if not results:
        print("No results provided for comparison")
        return
    
    model_names = list(results.keys())
    
    # Create comparison bar plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics[:4]):
        if i >= 4:
            break
            
        values = []
        for model in model_names:
            if metric in results[model]:
                values.append(results[model][metric])
            else:
                values.append(0)
        
        bars = axes[i].bar(model_names, values, color=sns.color_palette("husl", len(model_names)))
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[i].set_title(f'{metric.title()} Comparison', fontweight='bold')
        axes[i].set_ylabel(metric.title())
        axes[i].set_ylim([0, 1.1])
        axes[i].grid(True, alpha=0.3)
        
        # Rotate x labels if needed
        if len(max(model_names, key=len)) > 10:
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✓ Model comparison saved to {plot_path}")


def plot_loss_landscape(losses: List[float], output_dir: str, window_size: int = 100):
    """
    Plot training loss with moving averages and trend analysis.
    
    Args:
        losses: List of loss values
        output_dir: Directory to save plots  
        window_size: Window size for moving average
    """
    setup_publication_style()
    
    if not losses:
        print("No loss data provided")
        return
    
    episodes = range(len(losses))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Raw loss plot
    ax1.plot(episodes, losses, alpha=0.3, color='lightblue', linewidth=0.5, label='Raw Loss')
    
    # Moving averages
    if len(losses) > window_size:
        ma_short = pd.Series(losses).rolling(window=window_size//4, center=True).mean()
        ma_long = pd.Series(losses).rolling(window=window_size, center=True).mean()
        
        ax1.plot(episodes, ma_short, color='blue', linewidth=2, 
                label=f'{window_size//4}-Episode MA')
        ax1.plot(episodes, ma_long, color='darkblue', linewidth=3, 
                label=f'{window_size}-Episode MA')
    
    ax1.set_title('Training Loss Evolution', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss distribution
    ax2.hist(losses, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(losses), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(losses):.4f}')
    ax2.axvline(np.median(losses), color='blue', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(losses):.4f}')
    ax2.set_title('Loss Distribution', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Loss Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'loss_landscape.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✓ Loss landscape saved to {plot_path}")


def create_training_dashboard(training_data: Dict[str, Any], gpu_stats: List[Dict], 
                            output_dir: str, title: str = "Training Dashboard"):
    """
    Create a comprehensive training dashboard with all key metrics.
    
    Args:
        training_data: Dictionary of training metrics and results
        gpu_stats: List of GPU monitoring statistics
        output_dir: Directory to save the dashboard
        title: Dashboard title
    """
    setup_publication_style()
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)
    
    # Create subplots for different metrics
    ax1 = fig.add_subplot(gs[0, :2])  # Loss evolution
    ax2 = fig.add_subplot(gs[0, 2:])  # Accuracy evolution
    ax3 = fig.add_subplot(gs[1, :2])  # GPU utilization
    ax4 = fig.add_subplot(gs[1, 2:])  # Memory usage
    ax5 = fig.add_subplot(gs[2, :2])  # Reward progression
    ax6 = fig.add_subplot(gs[2, 2:])  # Performance metrics
    ax7 = fig.add_subplot(gs[3, :])   # Summary statistics
    
    # Plot various metrics (simplified implementation)
    if 'losses' in training_data:
        ax1.plot(training_data['losses'], color='red', linewidth=2)
        ax1.set_title('Training Loss', fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    if gpu_stats:
        gpu_df = pd.DataFrame(gpu_stats)
        ax3.plot(gpu_df['episode'], gpu_df['gpu_utilization_pct'], 
                color='green', linewidth=2)
        ax3.set_title('GPU Utilization', fontweight='bold')
        ax3.set_ylabel('Utilization (%)')
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(gpu_df['episode'], gpu_df['memory_utilization_pct'], 
                color='blue', linewidth=2)
        ax4.set_title('Memory Usage', fontweight='bold')
        ax4.set_ylabel('Memory (%)')
        ax4.grid(True, alpha=0.3)
    
    # Summary table (placeholder)
    ax7.text(0.1, 0.8, "Training Summary:", fontsize=14, fontweight='bold', 
            transform=ax7.transAxes)
    ax7.text(0.1, 0.6, f"Total Episodes: {len(training_data.get('losses', []))}", 
            fontsize=12, transform=ax7.transAxes)
    ax7.axis('off')
    
    # Save dashboard
    dashboard_path = os.path.join(output_dir, 'training_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✓ Training dashboard saved to {dashboard_path}")


def save_training_summary(training_data: Dict[str, Any], gpu_stats: List[Dict], 
                         output_dir: str, filename: str = "training_summary.txt"):
    """
    Save a comprehensive text summary of training results.
    
    Args:
        training_data: Dictionary of training metrics and results  
        gpu_stats: List of GPU monitoring statistics
        output_dir: Directory to save the summary
        filename: Name of the summary file
    """
    summary_path = os.path.join(output_dir, filename)
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Training Overview
        f.write("TRAINING OVERVIEW:\n")
        f.write("-"*40 + "\n")
        if 'losses' in training_data:
            f.write(f"Total Episodes: {len(training_data['losses'])}\n")
            f.write(f"Final Loss: {training_data['losses'][-1]:.6f}\n")
            f.write(f"Best Loss: {min(training_data['losses']):.6f}\n")
        f.write("\n")
        
        # GPU Performance Summary
        if gpu_stats:
            f.write("GPU PERFORMANCE SUMMARY:\n")
            f.write("-"*40 + "\n")
            gpu_df = pd.DataFrame(gpu_stats)
            f.write(f"Average GPU Utilization: {gpu_df['gpu_utilization_pct'].mean():.1f}%\n")
            f.write(f"Peak GPU Utilization: {gpu_df['gpu_utilization_pct'].max():.1f}%\n")
            f.write(f"Average Memory Usage: {gpu_df['memory_utilization_pct'].mean():.1f}%\n")
            f.write(f"Peak Memory Usage: {gpu_df['memory_utilization_pct'].max():.1f}%\n")
            f.write("\n")
        
        # Additional metrics as available
        for key, value in training_data.items():
            if key not in ['losses'] and isinstance(value, (int, float)):
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Training summary saved to {summary_path}")