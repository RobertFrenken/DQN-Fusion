"""
Visualization utilities for CAN bus intrusion detection analysis.

This module provides comprehensive plotting functions for analyzing graph neural network
performance on CAN bus data, including reconstruction errors, latent spaces, and 
multi-component error analysis for anomaly detection.
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
    
    # Print statistics
    print_statistics(comp_normal, comp_attack, "Normalized Composite")
    print(f"Threshold (95th percentile): {comp_threshold:.4f}")
    print(f"✓ Neighborhood composite error histogram saved as '{save_path}'")