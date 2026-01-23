"""
Visualization utilities for CAN bus intrusion detection analysis.

This module provides comprehensive plotting functions for analyzing graph neural network
performance on CAN bus data, including reconstruction errors, latent spaces, and 
multi-component error analysis for anomaly detection.

FUSION TRAINING PLOTTING FUNCTIONS:
- Enhanced plotting functions for fusion training visualization
- Training progress plots with publication-ready styling
- Fusion analysis and strategy visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.manifold import TSNE
from typing import List, Optional, Tuple, Union
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Publication-ready plot configuration
PLOT_CONFIG = {
    'figure_size': (10, 6),
    'small_figure_size': (8, 5),
    'large_figure_size': (12, 10),
    'dpi': 300,
    'alpha': 0.7,
    'bins': 50,
    'colors': {
        'normal': '#4472C4',      # Professional blue
        'attack': '#E74C3C',      # Professional red
        'threshold': '#2ECC71',   # Professional green
        'accent': '#9B59B6'       # Professional purple
    },
    'style': {
        'font_size': 12,
        'title_size': 14,
        'label_size': 11,
        'legend_size': 10
    }
}

def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': PLOT_CONFIG['style']['font_size'],
        'font.family': 'serif',
        'axes.labelsize': PLOT_CONFIG['style']['label_size'],
        'axes.titlesize': PLOT_CONFIG['style']['title_size'],
        'legend.fontsize': PLOT_CONFIG['style']['legend_size'],
        'figure.dpi': PLOT_CONFIG['dpi'],
        'savefig.dpi': PLOT_CONFIG['dpi'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'grid.alpha': 0.3
    })

def ensure_directory(save_path: str) -> None:
    """Ensure the directory for save_path exists."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

def print_statistics(normal_data: np.ndarray, attack_data: np.ndarray, 
                    component_name: str) -> None:
    """Print comprehensive statistics for error components."""
    print(f"\n{component_name} Statistics:")
    print(f"  Normal - Mean: {np.mean(normal_data):.4f}, Std: {np.std(normal_data):.4f}")
    print(f"  Attack - Mean: {np.mean(attack_data):.4f}, Std: {np.std(attack_data):.4f}")
    print(f"  Separation: {np.mean(attack_data) - np.mean(normal_data):.4f}")
    
    # Cohen's d effect size
    pooled_std = np.sqrt(((len(normal_data) - 1) * np.var(normal_data) + 
                         (len(attack_data) - 1) * np.var(attack_data)) / 
                         (len(normal_data) + len(attack_data) - 2))
    cohens_d = (np.mean(attack_data) - np.mean(normal_data)) / pooled_std
    print(f"  Cohen's d: {cohens_d:.3f}")

# ==================== Core Plotting Functions ====================

def plot_feature_histograms(graphs: List, feature_names: Optional[List[str]] = None, 
                           save_path: str = "images/feature_histograms.png") -> None:
    """
    Plot histograms for all node features in the dataset.
    
    Args:
        graphs: List of graph objects
        feature_names: Names for each feature dimension
        save_path: Path to save the figure
    """
    setup_publication_style()
    
    # Concatenate all node features
    all_x = torch.cat([g.x for g in graphs], dim=0).cpu().numpy()
    num_features = all_x.shape[1]
    
    # Setup feature names
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(num_features)]
    elif len(feature_names) < num_features:
        feature_names = feature_names + [f"Feature {i}" for i in range(len(feature_names), num_features)]
    
    # Create subplot grid
    n_cols = 5
    n_rows = int(np.ceil(num_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Plot each feature
    for i in range(num_features):
        axes[i].hist(all_x[:, i], bins=PLOT_CONFIG['bins'], 
                    color=PLOT_CONFIG['colors']['normal'], 
                    alpha=PLOT_CONFIG['alpha'], edgecolor='black')
        axes[i].set_title(feature_names[i])
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✓ Feature histograms saved as '{save_path}'")

def plot_latent_space(latent_vectors: np.ndarray, labels: np.ndarray, 
                     save_path: str = "images/latent_space.png") -> None:
    """
    Visualize latent space using t-SNE dimensionality reduction.
    
    Args:
        latent_vectors: Latent representations (n_samples, n_features)
        labels: Graph labels (0=normal, 1=attack)
        save_path: Path to save the figure
    """
    setup_publication_style()
    
    print("Computing t-SNE embedding for latent space visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)//4))
    embeddings_2d = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=PLOT_CONFIG['figure_size'])
    
    # Plot normal and attack points
    for label, color, name in [(0, PLOT_CONFIG['colors']['normal'], 'Normal'),
                              (1, PLOT_CONFIG['colors']['attack'], 'Attack')]:
        mask = labels == label
        if np.any(mask):
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=color, label=f"{name} (n={np.sum(mask)})", 
                       alpha=PLOT_CONFIG['alpha'], s=20)
    
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("Latent Space Visualization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✓ Latent space visualization saved as '{save_path}'")

# ==================== Reconstruction Error Analysis ====================

def plot_recon_error_hist(errors_normal: List[float], errors_attack: List[float], 
                         threshold: float, save_path: str = 'images/recon_error_hist.png') -> None:
    """
    Plot histogram of node reconstruction errors.
    
    Args:
        errors_normal: Reconstruction errors for normal graphs
        errors_attack: Reconstruction errors for attack graphs
        threshold: Anomaly detection threshold
        save_path: Path to save the figure
    """
    if not (errors_normal and errors_attack):
        print("⚠️ Insufficient data for reconstruction error histogram")
        return
    
    setup_publication_style()
    
    plt.figure(figsize=PLOT_CONFIG['small_figure_size'])
    plt.hist(errors_normal, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Normal (n={len(errors_normal)})', 
             color=PLOT_CONFIG['colors']['normal'], density=True)
    plt.hist(errors_attack, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Attack (n={len(errors_attack)})', 
             color=PLOT_CONFIG['colors']['attack'], density=True)
    plt.axvline(threshold, color=PLOT_CONFIG['colors']['threshold'], 
                linestyle='--', label=f'Threshold: {threshold:.4f}', linewidth=2)
    
    plt.xlabel('Mean Graph Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Node Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print_statistics(np.array(errors_normal), np.array(errors_attack), "Node Reconstruction")
    print(f"✓ Reconstruction error histogram saved as '{save_path}'")

def plot_neighborhood_error_hist(neighbor_errors_normal: List[float], 
                                neighbor_errors_attack: List[float], 
                                threshold: float, 
                                save_path: str = 'images/neighborhood_error_hist.png') -> None:
    """
    Plot histogram of neighborhood reconstruction errors.
    
    Args:
        neighbor_errors_normal: Neighborhood errors for normal graphs
        neighbor_errors_attack: Neighborhood errors for attack graphs
        threshold: Anomaly detection threshold
        save_path: Path to save the figure
    """
    if not (neighbor_errors_normal and neighbor_errors_attack):
        print("⚠️ Insufficient data for neighborhood error histogram")
        return
    
    setup_publication_style()
    
    plt.figure(figsize=PLOT_CONFIG['small_figure_size'])
    plt.hist(neighbor_errors_normal, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Normal (n={len(neighbor_errors_normal)})', 
             color=PLOT_CONFIG['colors']['normal'], density=True)
    plt.hist(neighbor_errors_attack, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Attack (n={len(neighbor_errors_attack)})', 
             color=PLOT_CONFIG['colors']['attack'], density=True)
    plt.axvline(threshold, color=PLOT_CONFIG['colors']['threshold'], 
                linestyle='--', label=f'Threshold: {threshold:.4f}', linewidth=2)
    
    plt.xlabel('Neighborhood Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Neighborhood Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print_statistics(np.array(neighbor_errors_normal), np.array(neighbor_errors_attack), 
                    "Neighborhood Reconstruction")
    print(f"✓ Neighborhood error histogram saved as '{save_path}'")

# ==================== Multi-Component Error Analysis ====================

def plot_error_components_analysis(node_errors_normal: List[float], node_errors_attack: List[float],
                                 neighbor_errors_normal: List[float], neighbor_errors_attack: List[float],
                                 canid_errors_normal: List[float], canid_errors_attack: List[float],
                                 save_path: str = "images/error_components_analysis.png") -> None:
    """
    Create comprehensive 2x2 analysis of normalized error components.
    
    Args:
        node_errors_normal/attack: Node reconstruction errors
        neighbor_errors_normal/attack: Neighborhood reconstruction errors
        canid_errors_normal/attack: CAN ID prediction errors
        save_path: Path to save the figure
    """
    setup_publication_style()
    
    def normalize_errors(arr: List[float]) -> np.ndarray:
        """Normalize errors to [0,1] range."""
        arr = np.array(arr)
        if arr.max() - arr.min() == 0:
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    # Normalize all error components
    node_n, node_a = normalize_errors(node_errors_normal), normalize_errors(node_errors_attack)
    neighbor_n, neighbor_a = normalize_errors(neighbor_errors_normal), normalize_errors(neighbor_errors_attack)
    canid_n, canid_a = normalize_errors(canid_errors_normal), normalize_errors(canid_errors_attack)
    
    # Create equal-weight composite
    comp_n = (node_n + neighbor_n + canid_n) / 3
    comp_a = (node_a + neighbor_a + canid_a) / 3
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG['large_figure_size'])
    
    components = [
        (node_n, node_a, "Node Reconstruction Errors", (0, 0)),
        (neighbor_n, neighbor_a, "Neighborhood Reconstruction Errors", (0, 1)),
        (canid_n, canid_a, "CAN ID Prediction Errors", (1, 0)),
        (comp_n, comp_a, "Composite Error Score", (1, 1))
    ]
    
    for normal_data, attack_data, title, (row, col) in components:
        axes[row, col].hist(normal_data, bins=30, alpha=PLOT_CONFIG['alpha'], 
                           label='Normal', color=PLOT_CONFIG['colors']['normal'], density=True)
        axes[row, col].hist(attack_data, bins=30, alpha=PLOT_CONFIG['alpha'], 
                           label='Attack', color=PLOT_CONFIG['colors']['attack'], density=True)
        axes[row, col].set_title(title)
        axes[row, col].set_xlabel('Normalized Error')
        axes[row, col].set_ylabel('Density')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    # Print comprehensive statistics
    print(f"\n{'='*60}")
    print("NORMALIZED ERROR COMPONENTS ANALYSIS")
    print(f"{'='*60}")
    
    for normal_data, attack_data, component in [(node_n, node_a, "Node Reconstruction"),
                                               (neighbor_n, neighbor_a, "Neighborhood Reconstruction"),
                                               (canid_n, canid_a, "CAN ID Prediction"),
                                               (comp_n, comp_a, "Composite Score")]:
        print_statistics(normal_data, attack_data, component)
    
    print(f"✓ Error components analysis saved as '{save_path}'")

def plot_raw_weighted_composite_error_hist(node_errors_normal: List[float], node_errors_attack: List[float],
                                          neighbor_errors_normal: List[float], neighbor_errors_attack: List[float],
                                          node_weight: float = 1.0, neighbor_weight: float = 20.0,
                                          save_path: str = "images/raw_weighted_composite_error_hist.png") -> None:
    """
    Plot composite error using weighted raw values without normalization.
    
    Args:
        node_errors_normal/attack: Node reconstruction errors
        neighbor_errors_normal/attack: Neighborhood reconstruction errors
        node_weight: Weight for node reconstruction component
        neighbor_weight: Weight for neighborhood reconstruction component
        save_path: Path to save the figure
    """
    setup_publication_style()
    
    # Compute weighted composite scores
    comp_normal = (node_weight * np.array(node_errors_normal) + 
                   neighbor_weight * np.array(neighbor_errors_normal))
    comp_attack = (node_weight * np.array(node_errors_attack) + 
                   neighbor_weight * np.array(neighbor_errors_attack))
    
    # Calculate threshold
    comp_threshold = np.percentile(comp_normal, 95) if len(comp_normal) > 0 else 0
    
    plt.figure(figsize=PLOT_CONFIG['figure_size'])
    plt.hist(comp_normal, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Normal (n={len(comp_normal)})', 
             color=PLOT_CONFIG['colors']['normal'], density=True)
    plt.hist(comp_attack, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Attack (n={len(comp_attack)})', 
             color=PLOT_CONFIG['colors']['attack'], density=True)
    plt.axvline(comp_threshold, color=PLOT_CONFIG['colors']['threshold'], 
                linestyle='--', label=f'Threshold: {comp_threshold:.3f}', linewidth=2)
    
    plt.xlabel('Weighted Raw Composite Error')
    plt.ylabel('Density')
    plt.title(f'Raw Weighted Composite Error\n(Weights: Node={node_weight}, Neighbor={neighbor_weight})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print_statistics(comp_normal, comp_attack, "Raw Weighted Composite")
    print(f"Threshold (95th percentile): {comp_threshold:.4f}")
    print(f"✓ Raw weighted composite error histogram saved as '{save_path}'")

def plot_raw_error_components_with_composite(node_errors_normal: List[float], node_errors_attack: List[float],
                                           neighbor_errors_normal: List[float], neighbor_errors_attack: List[float],
                                           canid_errors_normal: List[float], canid_errors_attack: List[float],
                                           save_path: str = "images/raw_error_components_with_composite.png") -> None:
    """
    Create 2x2 plot showing raw error components and weighted composite.
    
    Args:
        node_errors_normal/attack: Node reconstruction errors
        neighbor_errors_normal/attack: Neighborhood reconstruction errors
        canid_errors_normal/attack: CAN ID prediction errors
        save_path: Path to save the figure
    """
    setup_publication_style()
    
    # Convert to numpy arrays
    node_n, node_a = np.array(node_errors_normal), np.array(node_errors_attack)
    neighbor_n, neighbor_a = np.array(neighbor_errors_normal), np.array(neighbor_errors_attack)
    canid_n, canid_a = np.array(canid_errors_normal), np.array(canid_errors_attack)
    
    # Learned fusion weights
    weight_node, weight_neighbor, weight_canid = 1.0, 20.0, 0.3
    
    # Compute weighted composite
    comp_n = weight_node * node_n + weight_neighbor * neighbor_n + weight_canid * canid_n
    comp_a = weight_node * node_a + weight_neighbor * neighbor_a + weight_canid * canid_a
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG['large_figure_size'])
    
    components = [
        (node_n, node_a, "Raw Node Reconstruction Errors", "Node Reconstruction Error"),
        (neighbor_n, neighbor_a, "Raw Neighborhood Reconstruction Errors", "Neighborhood Reconstruction Error"),
        (canid_n, canid_a, "Raw CAN ID Prediction Errors", "CAN ID Error (Fraction Incorrect)"),
        (comp_n, comp_a, f"Weighted Composite Error\n(Weights: {weight_node}, {weight_neighbor}, {weight_canid})", "Weighted Composite Error")
    ]
    
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for (normal_data, attack_data, title, xlabel), (row, col) in zip(components, positions):
        axes[row, col].hist(normal_data, bins=30, alpha=PLOT_CONFIG['alpha'], 
                           label='Normal', color=PLOT_CONFIG['colors']['normal'], density=True)
        axes[row, col].hist(attack_data, bins=30, alpha=PLOT_CONFIG['alpha'], 
                           label='Attack', color=PLOT_CONFIG['colors']['attack'], density=True)
        axes[row, col].set_title(title)
        axes[row, col].set_xlabel(xlabel)
        axes[row, col].set_ylabel('Density')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
        
        # Add threshold line for composite plot
        if row == 1 and col == 1:
            comp_threshold = np.percentile(comp_n, 95) if len(comp_n) > 0 else 0
            axes[row, col].axvline(comp_threshold, color=PLOT_CONFIG['colors']['threshold'], 
                                  linestyle='--', label=f'95% Threshold: {comp_threshold:.3f}', linewidth=2)
            axes[row, col].legend()
    
    plt.tight_layout()
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    # Print comprehensive statistics
    print(f"\n{'='*60}")
    print("RAW ERROR COMPONENTS ANALYSIS")
    print(f"{'='*60}")
    
    for normal_data, attack_data, component in [(node_n, node_a, "Node Reconstruction"),
                                               (neighbor_n, neighbor_a, "Neighborhood Reconstruction"),
                                               (canid_n, canid_a, "CAN ID Prediction"),
                                               (comp_n, comp_a, "Weighted Composite")]:
        print_statistics(normal_data, attack_data, component)
    
    comp_threshold = np.percentile(comp_n, 95) if len(comp_n) > 0 else 0
    print(f"Composite Threshold (95th percentile): {comp_threshold:.4f}")
    print(f"✓ Raw error components analysis saved as '{save_path}'")

# ==================== Fusion Analysis ====================

def plot_fusion_score_distributions(anomaly_scores: np.ndarray, gat_probs: np.ndarray, 
                                   labels: np.ndarray, 
                                   save_path: str = "images/fusion_score_distributions.png") -> None:
    """
    Plot distributions of anomaly detection and GAT classification scores.
    
    Args:
        anomaly_scores: Normalized anomaly detection scores
        gat_probs: GAT classification probabilities
        labels: True labels (0=normal, 1=attack)
        save_path: Path to save the figure
    """
    setup_publication_style()
    
    normal_mask = labels == 0
    attack_mask = labels == 1
    
    fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG['large_figure_size'])
    
    # Anomaly detection scores
    axes[0, 0].hist(anomaly_scores[normal_mask], bins=30, alpha=PLOT_CONFIG['alpha'], 
                   label='Normal', color=PLOT_CONFIG['colors']['normal'], density=True)
    axes[0, 0].hist(anomaly_scores[attack_mask], bins=30, alpha=PLOT_CONFIG['alpha'], 
                   label='Attack', color=PLOT_CONFIG['colors']['attack'], density=True)
    axes[0, 0].set_title('Anomaly Detection Scores')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # GAT classification probabilities
    axes[0, 1].hist(gat_probs[normal_mask], bins=30, alpha=PLOT_CONFIG['alpha'], 
                   label='Normal', color=PLOT_CONFIG['colors']['normal'], density=True)
    axes[0, 1].hist(gat_probs[attack_mask], bins=30, alpha=PLOT_CONFIG['alpha'], 
                   label='Attack', color=PLOT_CONFIG['colors']['attack'], density=True)
    axes[0, 1].set_title('GAT Classification Probabilities')
    axes[0, 1].set_xlabel('GAT Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Weighted fusion (GAT-dominant: α=0.85)
    alpha = 0.85
    fused_weighted = (1 - alpha) * anomaly_scores + alpha * gat_probs
    axes[1, 0].hist(fused_weighted[normal_mask], bins=30, alpha=PLOT_CONFIG['alpha'], 
                   label='Normal', color=PLOT_CONFIG['colors']['normal'], density=True)
    axes[1, 0].hist(fused_weighted[attack_mask], bins=30, alpha=PLOT_CONFIG['alpha'], 
                   label='Attack', color=PLOT_CONFIG['colors']['attack'], density=True)
    axes[1, 0].set_title(f'GAT-Dominant Fusion (α={alpha})')
    axes[1, 0].set_xlabel('Fused Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Product fusion (geometric mean)
    fused_product = np.sqrt(anomaly_scores * gat_probs)
    axes[1, 1].hist(fused_product[normal_mask], bins=30, alpha=PLOT_CONFIG['alpha'], 
                   label='Normal', color=PLOT_CONFIG['colors']['normal'], density=True)
    axes[1, 1].hist(fused_product[attack_mask], bins=30, alpha=PLOT_CONFIG['alpha'], 
                   label='Attack', color=PLOT_CONFIG['colors']['attack'], density=True)
    axes[1, 1].set_title('Product Fusion (Geometric Mean)')
    axes[1, 1].set_xlabel('Fused Score')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    # Print fusion analysis
    print(f"\n{'='*60}")
    print("FUSION SCORE ANALYSIS")
    print(f"{'='*60}")
    
    print_statistics(anomaly_scores[normal_mask], anomaly_scores[attack_mask], "Anomaly Detection")
    print_statistics(gat_probs[normal_mask], gat_probs[attack_mask], "GAT Classification")
    print_statistics(fused_weighted[normal_mask], fused_weighted[attack_mask], "GAT-Dominant Fusion")
    print_statistics(fused_product[normal_mask], fused_product[attack_mask], "Product Fusion")
    
    print(f"✓ Fusion score distributions saved as '{save_path}'")

# ==================== Graph-Level Visualizations ====================

def plot_node_recon_errors(pipeline, loader: DataLoader, num_graphs: int = 8, 
                          save_path: str = "images/node_recon_subplot.png") -> None:
    """
    Plot node-level reconstruction errors for sample graphs.
    
    Args:
        pipeline: Trained GAT pipeline
        loader: DataLoader containing graphs
        num_graphs: Number of graphs to visualize per class
        save_path: Path to save the figure
    """
    pipeline.autoencoder.eval()
    normal_graphs, attack_graphs = [], []
    normal_errors, attack_errors = [], []
    
    # Collect sample graphs and their node errors
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(pipeline.device)
            cont_out, canid_logits, neighbor_logits, z, kl_loss = pipeline.autoencoder(
                batch.x, batch.edge_index, batch.batch)
            node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
            
            graphs = Batch.to_data_list(batch)
            start = 0
            
            for graph in graphs:
                n = graph.x.size(0)
                errors = node_errors[start:start+n].cpu().numpy()
                
                if int(graph.y.flatten()[0]) == 0 and len(normal_graphs) < num_graphs:
                    normal_graphs.append(graph)
                    normal_errors.append(errors)
                elif int(graph.y.flatten()[0]) == 1 and len(attack_graphs) < num_graphs:
                    attack_graphs.append(graph)
                    attack_errors.append(errors)
                
                start += n
                
                if len(normal_graphs) >= num_graphs and len(attack_graphs) >= num_graphs:
                    break
            
            if len(normal_graphs) >= num_graphs and len(attack_graphs) >= num_graphs:
                break
    
    # Create visualization
    setup_publication_style()
    fig, axes = plt.subplots(2, num_graphs, figsize=(4*num_graphs, 8), sharey=True)
    
    for i in range(num_graphs):
        if i < len(normal_errors):
            axes[0, i].bar(range(len(normal_errors[i])), normal_errors[i], 
                          color=PLOT_CONFIG['colors']['normal'], alpha=PLOT_CONFIG['alpha'])
            axes[0, i].set_title(f"Normal Graph {i+1}")
            axes[0, i].set_xlabel("Node Index")
            if i == 0:
                axes[0, i].set_ylabel("Reconstruction Error")
            axes[0, i].grid(True, alpha=0.3)
        
        if i < len(attack_errors):
            axes[1, i].bar(range(len(attack_errors[i])), attack_errors[i], 
                          color=PLOT_CONFIG['colors']['attack'], alpha=PLOT_CONFIG['alpha'])
            axes[1, i].set_title(f"Attack Graph {i+1}")
            axes[1, i].set_xlabel("Node Index")
            if i == 0:
                axes[1, i].set_ylabel("Reconstruction Error")
            axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✓ Node-level reconstruction errors saved as '{save_path}'")

def plot_graph_reconstruction(pipeline, loader: DataLoader, num_graphs: int = 4, 
                            save_path: str = "images/graph_recon_examples.png") -> None:
    """
    Plot detailed reconstruction examples for sample graphs.
    
    Args:
        pipeline: Trained GAT pipeline  
        loader: DataLoader containing graphs
        num_graphs: Number of graphs to show per class
        save_path: Base path for saving figures
    """
    pipeline.autoencoder.eval()
    shown_normal, shown_attack = 0, 0
    max_per_class = num_graphs // 2
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(pipeline.device)
            cont_out, canid_logits, neighbor_logits, z, kl_loss = pipeline.autoencoder(
                batch.x, batch.edge_index, batch.batch)
            
            graphs = Batch.to_data_list(batch)
            start = 0
            
            for graph in graphs:
                n = graph.x.size(0)
                input_features = graph.x.cpu().numpy()
                recon_features = cont_out[start:start+n].cpu().numpy()
                canid_predictions = canid_logits[start:start+n].argmax(dim=1).cpu().numpy()
                
                label = int(graph.y.flatten()[0])
                
                if label == 0 and shown_normal < max_per_class:
                    graph_type, count = "Normal", shown_normal + 1
                    shown_normal += 1
                elif label == 1 and shown_attack < max_per_class:
                    graph_type, count = "Attack", shown_attack + 1  
                    shown_attack += 1
                else:
                    start += n
                    continue
                
                # Create detailed reconstruction visualization
                setup_publication_style()
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Input payload features (excluding CAN ID)
                im0 = axes[0].imshow(input_features[:, 1:].T, aspect='auto', 
                                   cmap='viridis', interpolation='nearest')
                axes[0].set_title(f"Input Features\n({graph_type} Graph {count})")
                axes[0].set_xlabel("Node Index")
                axes[0].set_ylabel("Feature Dimension")
                plt.colorbar(im0, ax=axes[0])
                
                # Reconstructed payload features  
                im1 = axes[1].imshow(recon_features.T, aspect='auto', 
                                   cmap='viridis', interpolation='nearest')
                axes[1].set_title(f"Reconstructed Features\n({graph_type} Graph {count})")
                axes[1].set_xlabel("Node Index")
                axes[1].set_ylabel("Feature Dimension")
                plt.colorbar(im1, ax=axes[1])
                
                # CAN ID comparison
                node_indices = range(len(input_features))
                axes[2].plot(node_indices, input_features[:, 0], 'o-', 
                           label="True CAN ID", color=PLOT_CONFIG['colors']['normal'], linewidth=2)
                axes[2].plot(node_indices, canid_predictions, '*-', 
                           label="Predicted CAN ID", color=PLOT_CONFIG['colors']['attack'], linewidth=2)
                axes[2].set_title(f"CAN ID Reconstruction\n({graph_type} Graph {count})")
                axes[2].set_xlabel("Node Index")
                axes[2].set_ylabel("CAN ID")
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                plt.suptitle(f"{graph_type} Graph {count} Reconstruction (Label: {label})", 
                           fontsize=16, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                
                # Save individual graph reconstruction
                graph_save_path = f"{save_path.rstrip('.png')}_{graph_type.lower()}_{count}.png"
                ensure_directory(graph_save_path)
                plt.savefig(graph_save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
                plt.close()
                
                start += n
                
                if shown_normal >= max_per_class and shown_attack >= max_per_class:
                    print(f"✓ Graph reconstruction examples saved with base path '{save_path}'")
                    return

# ==================== Dataset Analysis Functions ====================

def print_graph_stats(graphs: List, label: str) -> None:
    """Print comprehensive statistics for a collection of graphs."""
    all_x = torch.cat([g.x for g in graphs], dim=0)
    print(f"\n{'='*50}")
    print(f"{label.upper()} GRAPHS STATISTICS")
    print(f"{'='*50}")
    print(f"Number of graphs: {len(graphs):,}")
    print(f"Total nodes: {all_x.size(0):,}")
    print(f"Feature dimensions: {all_x.size(1)}")
    print(f"Node feature means: {all_x.mean(dim=0).tolist()}")
    print(f"Node feature stds: {all_x.std(dim=0).tolist()}")
    print(f"Unique CAN IDs: {sorted(all_x[:, 0].unique().tolist())}")
    print(f"Sample node features (first 5 nodes):")
    print(all_x[:5].numpy())

def print_graph_structure(graphs: List, label: str) -> None:
    """Print structural statistics for a collection of graphs."""
    num_nodes = [g.num_nodes for g in graphs]
    num_edges = [g.num_edges for g in graphs]
    
    print(f"\n{'='*50}")
    print(f"{label.upper()} GRAPHS STRUCTURE")
    print(f"{'='*50}")
    print(f"Average nodes per graph: {np.mean(num_nodes):.2f} ± {np.std(num_nodes):.2f}")
    print(f"Average edges per graph: {np.mean(num_edges):.2f} ± {np.std(num_edges):.2f}")
    print(f"Node count range: [{min(num_nodes)}, {max(num_nodes)}]")
    print(f"Edge count range: [{min(num_edges)}, {max(num_edges)}]")
    
    # Edge density analysis
    densities = []
    for g in graphs:
        max_edges = g.num_nodes * (g.num_nodes - 1)  # Directed graph
        density = g.num_edges / max_edges if max_edges > 0 else 0
        densities.append(density)
    
    print(f"Average edge density: {np.mean(densities):.4f} ± {np.std(densities):.4f}")

# ==================== Legacy/Specialized Functions ====================

def plot_canid_recon_hist(id_errors_normal: List[float], id_errors_attack: List[float], 
                         save_path: str = "images/canid_recon_hist.png") -> None:
    """Plot CAN ID reconstruction error histogram."""
    if not (id_errors_normal and id_errors_attack):
        print("⚠️ Insufficient data for CAN ID error histogram")
        return
    
    setup_publication_style()
    
    plt.figure(figsize=PLOT_CONFIG['small_figure_size'])
    plt.hist(id_errors_normal, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Normal (n={len(id_errors_normal)})', 
             color=PLOT_CONFIG['colors']['normal'], density=True)
    plt.hist(id_errors_attack, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Attack (n={len(id_errors_attack)})', 
             color=PLOT_CONFIG['colors']['attack'], density=True)
    
    plt.xlabel('Fraction of Incorrect CAN IDs per Graph')
    plt.ylabel('Density')
    plt.title('CAN ID Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print_statistics(np.array(id_errors_normal), np.array(id_errors_attack), "CAN ID Reconstruction")
    print(f"✓ CAN ID reconstruction error histogram saved as '{save_path}'")

def plot_edge_error_hist(errors_normal: List[float], errors_attack: List[float], 
                        threshold: float, save_path: str = 'images/edge_error_hist.png') -> None:
    """Plot edge reconstruction error histogram (legacy function)."""
    if not (errors_normal and errors_attack):
        print("⚠️ Insufficient data for edge error histogram")
        return
    
    setup_publication_style()
    
    plt.figure(figsize=PLOT_CONFIG['small_figure_size'])
    plt.hist(errors_normal, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Normal (n={len(errors_normal)})', 
             color=PLOT_CONFIG['colors']['normal'], density=True)
    plt.hist(errors_attack, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Attack (n={len(errors_attack)})', 
             color=PLOT_CONFIG['colors']['attack'], density=True)
    plt.axvline(threshold, color=PLOT_CONFIG['colors']['threshold'], 
                linestyle='--', label=f'Threshold: {threshold:.4f}', linewidth=2)
    
    plt.xlabel('Edge Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Edge Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print_statistics(np.array(errors_normal), np.array(errors_attack), "Edge Reconstruction")
    print(f"✓ Edge error histogram saved as '{save_path}'")

# ==================== Composite Error Functions ====================

def plot_neighborhood_composite_error_hist(node_errors_normal: List[float], node_errors_attack: List[float],
                                         neighbor_errors_normal: List[float], neighbor_errors_attack: List[float],
                                         canid_errors_normal: List[float], canid_errors_attack: List[float],
                                         save_path: str = "images/neighborhood_composite_error_hist.png") -> None:
    """
    Plot normalized composite error combining all three components with equal weights.
    
    This function creates a composite score using equal weighting (1/3 each) of normalized
    node reconstruction, neighborhood reconstruction, and CAN ID prediction errors.
    """
    setup_publication_style()
    
    def normalize_errors(arr: List[float]) -> np.ndarray:
        """Normalize errors to [0,1] range for fair combination."""
        arr = np.array(arr)
        if arr.max() - arr.min() == 0:
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    # Normalize all error components
    node_n = normalize_errors(node_errors_normal)
    node_a = normalize_errors(node_errors_attack)
    neighbor_n = normalize_errors(neighbor_errors_normal)
    neighbor_a = normalize_errors(neighbor_errors_attack)
    canid_n = normalize_errors(canid_errors_normal)
    canid_a = normalize_errors(canid_errors_attack)
    
    # Equal-weight composite (1/3 each component)
    weight_node = weight_neighbor = weight_canid = 1/3
    
    comp_normal = weight_node * node_n + weight_neighbor * neighbor_n + weight_canid * canid_n
    comp_attack = weight_node * node_a + weight_neighbor * neighbor_a + weight_canid * canid_a
    
    # Calculate 95th percentile threshold
    comp_threshold = np.percentile(comp_normal, 95) if len(comp_normal) > 0 else 0
    
    # Create plot
    plt.figure(figsize=PLOT_CONFIG['figure_size'])
    plt.hist(comp_normal, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Normal (n={len(comp_normal)})', 
             color=PLOT_CONFIG['colors']['normal'], density=True)
    plt.hist(comp_attack, bins=PLOT_CONFIG['bins'], alpha=PLOT_CONFIG['alpha'], 
             label=f'Attack (n={len(comp_attack)})', 
             color=PLOT_CONFIG['colors']['attack'], density=True)
    plt.axvline(comp_threshold, color=PLOT_CONFIG['colors']['threshold'], 
                linestyle='--', label=f'Threshold: {comp_threshold:.3f}', linewidth=2)
    
    plt.xlabel('Normalized Composite Error')
    plt.ylabel('Density')
    plt.title(f'Composite Error Distribution\n(Equal Weights: Node={weight_node:.2f}, Neighbor={weight_neighbor:.2f}, CAN_ID={weight_canid:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ensure_directory(save_path)
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()


# ===== FUSION TRAINING PLOTTING FUNCTIONS =====

def plot_fusion_training_progress(accuracies: List, rewards: List, validation_scores: List):
    """Plot training progress visualization with publication-ready styling."""
    from config.plotting_config import COLOR_SCHEMES, apply_publication_style, save_publication_figure
    
    apply_publication_style()
    plt.ioff()  # Turn off interactive mode
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    episodes = range(1, len(accuracies) + 1)
    colors = COLOR_SCHEMES['training']
    
    # Training accuracy
    ax1.plot(episodes, accuracies, color=colors['accuracy'], linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Training Accuracy')
    ax1.set_title('Training Accuracy Progression', fontweight='bold')
    ax1.set_ylim([0, 1.02])
    
    # Training rewards
    ax2.plot(episodes, rewards, color=colors['reward'], linewidth=2.5, alpha=0.8)
    ax2.set_xlabel('Training Episode')
    ax2.set_ylabel('Average Normalized Reward')
    ax2.set_title('Training Reward Evolution', fontweight='bold')
    
    # Validation accuracy
    if validation_scores:
        val_episodes = [i * 100 for i in range(1, len(validation_scores) + 1)]
        val_accuracies = [score['accuracy'] for score in validation_scores]
        ax3.plot(val_episodes, val_accuracies, color=COLOR_SCHEMES['validation']['primary'], 
                linewidth=2.5, marker='o', markersize=6, alpha=0.8)
        ax3.set_xlabel('Training Episode')
        ax3.set_ylabel('Validation Accuracy')
        ax3.set_title('Validation Performance', fontweight='bold')
        # Auto-scale validation accuracy to show data range better
        if val_accuracies:
            y_min = max(0, min(val_accuracies) - 0.01)
            y_max = min(1.0, max(val_accuracies) + 0.01)
            ax3.set_ylim([y_min, y_max])
    
    # Fusion weights used
    if validation_scores:
        val_alphas = [score['avg_alpha'] for score in validation_scores]
        ax4.plot(val_episodes, val_alphas, color=colors['q_values'], 
                linewidth=2.5, marker='s', markersize=6, alpha=0.8, label='Learned α')
        ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Balanced Fusion')
        ax4.set_xlabel('Training Episode')
        ax4.set_ylabel('Average Fusion Weight (α)')
        ax4.set_title('Adaptive Fusion Strategy', fontweight='bold')
        ax4.legend(loc='best')
    
    plt.tight_layout()
    save_publication_figure(fig, 'images/fusion_training_progress.png')
    plt.close(fig)
    plt.ion()  # Turn interactive mode back on


def plot_fusion_analysis(anomaly_scores: List, gat_probs: List, 
                        labels: List, adaptive_alphas: List,
                        dataset_key: str, current_fig=None, current_axes=None):
    """Add fusion analysis plots to the existing training progress figure."""
    from config.plotting_config import COLOR_SCHEMES, apply_publication_style, save_publication_figure
    
    # Check if we have the figure from training progress
    if current_fig is None or current_axes is None:
        # Strict: do not create implicit figures. Require caller to pass the training progress figure/axes
        raise ValueError(
            "plot_fusion_analysis requires a pre-existing figure and axes from plot_fusion_training_progress. "
            "Pass `current_fig` and `current_axes`, or call `plot_fusion_training_progress` before `plot_fusion_analysis`."
        )
    else:
        fig = current_fig
        axes = current_axes
    
    # Convert to numpy arrays
    anomaly_scores = np.array(anomaly_scores)
    gat_probs = np.array(gat_probs)
    labels = np.array(labels)
    adaptive_alphas = np.array(adaptive_alphas)
    
    colors = COLOR_SCHEMES['fusion_analysis']
    
    # Row 2, Col 2: State space visualization - simplified scatter plot
    x_min, x_max = max(0, anomaly_scores.min() - 0.02), min(1, anomaly_scores.max() + 0.02)
    y_min, y_max = max(0, gat_probs.min() - 0.02), min(1, gat_probs.max() + 0.02)
    
    scatter = axes[1,1].scatter(anomaly_scores, gat_probs, c=adaptive_alphas, 
                              cmap=COLOR_SCHEMES['contour'], s=12, alpha=0.7, 
                              edgecolors='white', linewidths=0.1)
    
    axes[1,1].set_xlabel('VGAE Anomaly Score')
    axes[1,1].set_ylabel('GAT Classification Probability')
    axes[1,1].set_title('Learned Fusion Policy', fontweight='bold')
    axes[1,1].set_xlim([x_min, x_max])
    axes[1,1].set_ylim([y_min, y_max])
    
    cbar = plt.colorbar(scatter, ax=axes[1,1], shrink=0.8)
    cbar.set_label('Fusion Weight (α)', rotation=270, labelpad=15, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # Row 2, Col 3: Fusion weight distribution by class
    normal_alphas = adaptive_alphas[labels == 0]
    attack_alphas = adaptive_alphas[labels == 1]
    
    bins = 25
    n_normal, bins_normal, _ = axes[1,2].hist(normal_alphas, bins=bins, alpha=0.7, 
                                           color=colors['normal'], edgecolor='black', linewidth=0.8,
                                           label='Normal (Count)', histtype='bar')
    n_attack, bins_attack, _ = axes[1,2].hist(attack_alphas, bins=bins, alpha=0.7,
                                           color=colors['attack'], edgecolor='black', linewidth=0.8,
                                           label='Attack (Count)', histtype='bar')
    
    axes[1,2].set_xlabel('Fusion Weight (α)')
    axes[1,2].set_ylabel('Raw Sample Count', color='black')
    axes[1,2].set_xlim([0, 1])
    axes[1,2].tick_params(axis='y', labelcolor='black')
    
    # Secondary axis - proportional distribution
    ax2_twin = axes[1,2].twinx()
    
    bin_centers = (bins_normal[:-1] + bins_normal[1:]) / 2
    normal_prop = n_normal / len(normal_alphas) if len(normal_alphas) > 0 else n_normal
    attack_prop = n_attack / len(attack_alphas) if len(attack_alphas) > 0 else n_attack
    
    ax2_twin.plot(bin_centers, normal_prop, color=colors['normal'], linewidth=1.5, 
                 linestyle='-', marker='o', markersize=2, alpha=0.9, label='Normal (Proportion)')
    ax2_twin.plot(bin_centers, attack_prop, color=colors['attack'], linewidth=1.5,
                 linestyle='-', marker='s', markersize=2, alpha=0.9, label='Attack (Proportion)')
    
    ax2_twin.set_ylabel('Proportional Distribution', color='gray')
    ax2_twin.tick_params(axis='y', labelcolor='gray')
    
    lines1, labels1 = axes[1,2].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axes[1,2].legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    axes[1,2].set_title('Fusion Strategy Distribution', fontweight='bold')
    
    # Row 2, Col 4: Model agreement analysis - simplified
    model_diff = np.abs(anomaly_scores - gat_probs)
    jitter_amount = 0.01
    jittered_alphas = adaptive_alphas + np.random.normal(0, jitter_amount, len(adaptive_alphas))
    
    normal_mask = labels == 0
    attack_mask = labels == 1
    
    axes[1,3].scatter(model_diff[normal_mask], jittered_alphas[normal_mask], 
                     alpha=0.6, s=12, c=colors['normal'], 
                     edgecolors='white', linewidths=0.3, label='Normal Traffic')
    
    axes[1,3].scatter(model_diff[attack_mask], jittered_alphas[attack_mask],
                     alpha=0.8, s=15, c=colors['attack'], 
                     edgecolors='white', linewidths=0.3, label='Attack Traffic')
    
    # Add horizontal reference lines
    key_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for alpha_val in key_alphas:
        axes[1,3].axhline(y=alpha_val, color='gray', alpha=0.2, linewidth=0.5, linestyle=':')
    
    axes[1,3].set_xlabel('Model Disagreement |VGAE Score - GAT Probability|')
    axes[1,3].set_ylabel('Fusion Weight (α)')
    axes[1,3].set_title('Strategy vs. Model Agreement', fontweight='bold')
    axes[1,3].set_xlim([0, 1])
    axes[1,3].set_ylim([0, 1])
    axes[1,3].legend(loc='best', fontsize=9)
    
    # Update the saved figure
    plt.tight_layout()
    filename = f'images/complete_fusion_training_analysis_{dataset_key}'
    save_publication_figure(fig, filename + '.png')
    plt.close(fig)
    plt.ion()


def plot_enhanced_fusion_training_progress(accuracies, rewards, losses, q_values, 
                                          action_distributions, reward_stats, validation_scores, 
                                          dataset_key, fusion_agent):
    """Enhanced training progress visualization with publication-ready styling."""
    from config.plotting_config import COLOR_SCHEMES, apply_publication_style, save_publication_figure
    
    apply_publication_style()
    plt.ioff()  # Turn off interactive mode
    fig, axes = plt.subplots(2, 4, figsize=(32, 16))  # Single 2x4 layout for all plots
    episodes = range(1, len(accuracies) + 1)
    colors = COLOR_SCHEMES['training']
    
    # Row 1, Col 1: Training accuracy
    axes[0,0].plot(episodes, accuracies, color=colors['accuracy'], linewidth=1.5, alpha=0.8)
    axes[0,0].set_xlabel('Training Episode')
    axes[0,0].set_ylabel('Training Accuracy')
    axes[0,0].set_title('Training Accuracy Progression', fontweight='bold')
    if accuracies:
        y_min = max(0, min(accuracies) - 0.01)
        y_max = min(1.0, max(accuracies) + 0.01)
        axes[0,0].set_ylim([y_min, y_max])
    
    # Row 1, Col 2: Training rewards
    axes[0,1].plot(episodes, rewards, color=colors['reward'], linewidth=1.5, alpha=0.8)
    axes[0,1].set_xlabel('Training Episode')
    axes[0,1].set_ylabel('Normalized Reward')
    axes[0,1].set_title('Training Reward Evolution', fontweight='bold')
    
    # Row 1, Col 3: Training losses
    if len(losses) > 1:
        loss_episodes = episodes[1:]
        loss_values = losses[1:]
        axes[0,2].plot(loss_episodes, loss_values, color=colors['loss'], linewidth=1.5, alpha=0.8)
    axes[0,2].set_xlabel('Training Episode')
    axes[0,2].set_ylabel('Average Loss (log scale)')
    axes[0,2].set_title('Training Loss Convergence', fontweight='bold')
    axes[0,2].set_yscale('log')
    
    # Row 1, Col 4: Action distribution heatmap
    if action_distributions:
        action_matrix = np.array(action_distributions).T
        bin_size = 50
        n_episodes = len(action_distributions)
        n_bins = (n_episodes + bin_size - 1) // bin_size
        
        binned_matrix = np.zeros((action_matrix.shape[0], n_bins))
        bin_labels = []
        
        for i in range(n_bins):
            start_ep = i * bin_size
            end_ep = min((i + 1) * bin_size, n_episodes)
            if end_ep > start_ep:
                binned_matrix[:, i] = np.mean(action_matrix[:, start_ep:end_ep], axis=1)
                bin_labels.append(f'{start_ep+1}-{end_ep}')
        
        im = axes[0,3].imshow(binned_matrix, aspect='auto', cmap=COLOR_SCHEMES['heatmap'], 
                            origin='lower', interpolation='nearest')
        axes[0,3].set_xlabel('Episode Bins')
        axes[0,3].set_ylabel('Fusion Weight (α)')
        axes[0,3].set_title(f'Action Selection Evolution', fontweight='bold')
        
        alpha_values = getattr(fusion_agent, 'alpha_values', [0.0, 0.25, 0.5, 0.75, 1.0])
        alpha_ticks = range(0, len(alpha_values), max(1, len(alpha_values)//5))
        alpha_labels = [f'{alpha_values[i]:.2f}' for i in alpha_ticks if i < len(alpha_values)]
        axes[0,3].set_yticks(alpha_ticks)
        axes[0,3].set_yticklabels(alpha_labels)
        
        x_tick_interval = max(1, n_bins // 6)  # Fewer labels for smaller subplot
        x_ticks = range(0, n_bins, x_tick_interval)
        x_labels = [bin_labels[i] for i in x_ticks]
        axes[0,3].set_xticks(x_ticks)
        axes[0,3].set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        
        cbar = plt.colorbar(im, ax=axes[0,3], shrink=0.8)
        cbar.set_label('Selection Frequency', fontsize=10)
        cbar.ax.tick_params(labelsize=9)
    
    # Row 2, Col 1: Exploration-Exploitation Balance
    if validation_scores and hasattr(fusion_agent, 'epsilon'):
        epsilon_values = []
        action_entropies = []
        
        for action_dist in action_distributions:
            action_probs = action_dist / (action_dist.sum() + 1e-8)
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
            action_entropies.append(entropy)
        
        initial_epsilon = 0.8
        epsilon_decay = 0.99
        for ep in range(len(episodes)):
            current_epsilon = max(0.15, initial_epsilon * (epsilon_decay ** (ep // 5)))
            epsilon_values.append(current_epsilon)
        
        axes[1,0].plot(episodes, epsilon_values, color='#d62728', linewidth=1.5, 
                      alpha=0.9, label='Exploration (ε)')
        axes[1,0].set_xlabel('Training Episode')
        axes[1,0].set_ylabel('Epsilon Value', color='#d62728')
        axes[1,0].tick_params(axis='y', labelcolor='#d62728')
        axes[1,0].set_ylim([0, 1])
        
        ax_twin = axes[1,0].twinx()
        window_size = min(25, len(action_entropies)//10)
        if window_size > 0:
            smoothed_entropy = np.convolve(action_entropies, np.ones(window_size)/window_size, mode='valid')
            entropy_episodes = episodes[window_size-1:]
            ax_twin.plot(entropy_episodes, smoothed_entropy, color='#ff7f0e', 
                       linewidth=1.5, alpha=0.9, label='Action Entropy')
        
        ax_twin.set_ylabel('Action Entropy (bits)', color='#ff7f0e')
        ax_twin.tick_params(axis='y', labelcolor='#ff7f0e')
        
        lines1, labels1 = axes[1,0].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axes[1,0].legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        axes[1,0].set_title('Exploration-Exploitation Balance', fontweight='bold')
    
    plt.tight_layout()
    filename = f'images/complete_fusion_training_analysis_{dataset_key}'
    save_publication_figure(fig, filename + '.png')
    plt.close(fig)
    plt.ion()
    
    # Return figure and axes for potential fusion analysis plots
    return fig, axes
    
    # Print statistics
    print_statistics(comp_normal, comp_attack, "Normalized Composite")
    print(f"Threshold (95th percentile): {comp_threshold:.4f}")
    print(f"✓ Neighborhood composite error histogram saved as '{save_path}'")