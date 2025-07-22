import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import torch
from torch_geometric.data import Batch
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_raw_error_components_with_composite(node_errors_normal, node_errors_attack,
                                           neighbor_errors_normal, neighbor_errors_attack,
                                           canid_errors_normal, canid_errors_attack,
                                           save_path="images/raw_error_components_with_composite.png"):
    """
    Create a 2x2 plot showing raw/unnormalized error components and rescaled composite.
    
    Args:
        node_errors_normal: List of node reconstruction errors for normal graphs.
        node_errors_attack: List of node reconstruction errors for attack graphs.
        neighbor_errors_normal: List of neighborhood reconstruction errors for normal graphs.
        neighbor_errors_attack: List of neighborhood reconstruction errors for attack graphs.
        canid_errors_normal: List of CAN ID errors for normal graphs.
        canid_errors_attack: List of CAN ID errors for attack graphs.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy arrays for easier manipulation
    node_n, node_a = np.array(node_errors_normal), np.array(node_errors_attack)
    neighbor_n, neighbor_a = np.array(neighbor_errors_normal), np.array(neighbor_errors_attack)
    canid_n, canid_a = np.array(canid_errors_normal), np.array(canid_errors_attack)
    
    # Create rescaled composite (weighted raw values to bring to similar scales)
    weight_node = 1.0       # Base scale (errors ~0.1-0.4)
    weight_neighbor = 20.0  # Scale up (errors ~0.005-0.04)
    weight_canid = 0.3      # Scale down (errors 0.0-1.0)
    
    comp_n = (weight_node * node_n + weight_neighbor * neighbor_n + weight_canid * canid_n)
    comp_a = (weight_node * node_a + weight_neighbor * neighbor_a + weight_canid * canid_a)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Common histogram settings
    bins = 30
    alpha = 0.7
    
    # Top Left: Raw Node Reconstruction Errors
    axes[0,0].hist(node_n, bins=bins, alpha=alpha, label='Normal', color='blue', density=True)
    axes[0,0].hist(node_a, bins=bins, alpha=alpha, label='Attack', color='red', density=True)
    axes[0,0].set_title('Raw Node Reconstruction Errors')
    axes[0,0].set_xlabel('Node Reconstruction Error')
    axes[0,0].set_ylabel('Density')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Top Right: Raw Neighborhood Reconstruction Errors
    axes[0,1].hist(neighbor_n, bins=bins, alpha=alpha, label='Normal', color='blue', density=True)
    axes[0,1].hist(neighbor_a, bins=bins, alpha=alpha, label='Attack', color='red', density=True)
    axes[0,1].set_title('Raw Neighborhood Reconstruction Errors')
    axes[0,1].set_xlabel('Neighborhood Reconstruction Error')
    axes[0,1].set_ylabel('Density')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Bottom Left: Raw CAN ID Errors
    axes[1,0].hist(canid_n, bins=bins, alpha=alpha, label='Normal', color='blue', density=True)
    axes[1,0].hist(canid_a, bins=bins, alpha=alpha, label='Attack', color='red', density=True)
    axes[1,0].set_title('Raw CAN ID Errors')
    axes[1,0].set_xlabel('CAN ID Error (Fraction Incorrect)')
    axes[1,0].set_ylabel('Density')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Bottom Right: Rescaled Composite Error
    axes[1,1].hist(comp_n, bins=bins, alpha=alpha, label='Normal', color='blue', density=True)
    axes[1,1].hist(comp_a, bins=bins, alpha=alpha, label='Attack', color='red', density=True)
    axes[1,1].set_title(f'Rescaled Composite Error\n(Weights: Node={weight_node}, Neighbor={weight_neighbor}, CAN_ID={weight_canid})')
    axes[1,1].set_xlabel('Weighted Composite Error')
    axes[1,1].set_ylabel('Density')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Add threshold line to composite plot
    comp_threshold = np.percentile(comp_n, 95) if len(comp_n) > 0 else 0
    axes[1,1].axvline(comp_threshold, color='green', linestyle='--', 
                      label=f'95% Threshold: {comp_threshold:.3f}', linewidth=2)
    axes[1,1].legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\n=== Raw Error Components Analysis ===")
    print(f"Node Errors - Normal: {np.mean(node_n):.4f}±{np.std(node_n):.4f}, Attack: {np.mean(node_a):.4f}±{np.std(node_a):.4f}")
    print(f"Neighborhood Errors - Normal: {np.mean(neighbor_n):.4f}±{np.std(neighbor_n):.4f}, Attack: {np.mean(neighbor_a):.4f}±{np.std(neighbor_a):.4f}")
    print(f"CAN ID Errors - Normal: {np.mean(canid_n):.4f}±{np.std(canid_n):.4f}, Attack: {np.mean(canid_a):.4f}±{np.std(canid_a):.4f}")
    print(f"Composite Errors - Normal: {np.mean(comp_n):.4f}±{np.std(comp_n):.4f}, Attack: {np.mean(comp_a):.4f}±{np.std(comp_a):.4f}")
    print(f"Composite Separation: {np.mean(comp_a) - np.mean(comp_n):.4f}")
    print(f"Composite Threshold (95%): {comp_threshold:.4f}")
    print(f"Saved raw error components analysis as '{save_path}'")
    
def plot_error_components_analysis(node_errors_normal, node_errors_attack,
                                 neighbor_errors_normal, neighbor_errors_attack,
                                 canid_errors_normal, canid_errors_attack,
                                 save_path="images/error_components_analysis.png"):
    """
    Create a multi-panel plot showing individual error components and their combination.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    def normalize(arr):
        arr = np.array(arr)
        if arr.max() - arr.min() == 0:
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    # Normalize errors
    node_n, node_a = normalize(node_errors_normal), normalize(node_errors_attack)
    neighbor_n, neighbor_a = normalize(neighbor_errors_normal), normalize(neighbor_errors_attack)
    canid_n, canid_a = normalize(canid_errors_normal), normalize(canid_errors_attack)
    
    # FIXED: Equal 1/3 composite weights
    comp_n = (1/3 * node_n + 1/3 * neighbor_n + 1/3 * canid_n)
    comp_a = (1/3 * node_a + 1/3 * neighbor_a + 1/3 * canid_a)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Node errors
    axes[0,0].hist(node_n, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
    axes[0,0].hist(node_a, bins=30, alpha=0.7, label='Attack', color='red', density=True)
    axes[0,0].set_title('Node Reconstruction Errors (Normalized)')
    axes[0,0].legend()
    
    # Neighborhood errors
    axes[0,1].hist(neighbor_n, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
    axes[0,1].hist(neighbor_a, bins=30, alpha=0.7, label='Attack', color='red', density=True)
    axes[0,1].set_title('Neighborhood Reconstruction Errors (Normalized)')
    axes[0,1].legend()
    
    # CAN ID errors
    axes[1,0].hist(canid_n, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
    axes[1,0].hist(canid_a, bins=30, alpha=0.7, label='Attack', color='red', density=True)
    axes[1,0].set_title('CAN ID Errors (Normalized)')
    axes[1,0].legend()
    
    # Composite
    axes[1,1].hist(comp_n, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
    axes[1,1].hist(comp_a, bins=30, alpha=0.7, label='Attack', color='red', density=True)
    axes[1,1].set_title('Composite Error (Weighted Combination)')
    axes[1,1].legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved error components analysis as '{save_path}'")

def plot_raw_weighted_composite_error_hist(node_errors_normal, node_errors_attack,
                                          neighbor_errors_normal, neighbor_errors_attack,
                                          canid_errors_normal, canid_errors_attack,
                                          save_path="images/raw_weighted_composite_error_hist.png"):
    """Plot composite error using weighted raw values (no normalization)."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Scale weights to bring all error types to similar magnitude
    weight_node = 1.0      # Base scale
    weight_neighbor = 20.0  # Scale up small neighborhood errors  
    weight_canid = 0.3     # Scale down CAN ID errors
    
    comp_n = (weight_node * np.array(node_errors_normal) + 
              weight_neighbor * np.array(neighbor_errors_normal) + 
              weight_canid * np.array(canid_errors_normal))
    
    comp_a = (weight_node * np.array(node_errors_attack) + 
              weight_neighbor * np.array(neighbor_errors_attack) + 
              weight_canid * np.array(canid_errors_attack))
    
    comp_threshold = np.percentile(comp_n, 95) if len(comp_n) > 0 else 0
    
    plt.figure(figsize=(10, 6))
    plt.hist(comp_n, bins=50, alpha=0.7, label=f'Normal (n={len(comp_n)})', color='blue', density=True)
    plt.hist(comp_a, bins=50, alpha=0.7, label=f'Attack (n={len(comp_a)})', color='red', density=True)
    plt.axvline(comp_threshold, color='green', linestyle='--', label=f'Threshold: {comp_threshold:.3f}')
    
    plt.xlabel('Weighted Raw Composite Error')
    plt.ylabel('Density')
    plt.title(f'Raw Weighted Composite Error\n(Weights: Node={weight_node}, Neighbor={weight_neighbor}, CAN_ID={weight_canid})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Raw Weighted Composite Error Statistics:")
    print(f"Normal - Mean: {np.mean(comp_n):.4f}, Std: {np.std(comp_n):.4f}")
    print(f"Attack - Mean: {np.mean(comp_a):.4f}, Std: {np.std(comp_a):.4f}")
    print(f"Separation: {np.mean(comp_a) - np.mean(comp_n):.4f}")
    print(f"Saved raw weighted composite error histogram as '{save_path}'")
    
def plot_neighborhood_composite_error_hist(node_errors_normal, node_errors_attack,
                                         neighbor_errors_normal, neighbor_errors_attack,
                                         canid_errors_normal, canid_errors_attack,
                                         save_path="images/neighborhood_composite_error_hist.png"):
    """
    Plot histogram of composite error combining node reconstruction, neighborhood reconstruction, and CAN ID errors.
    
    Args:
        node_errors_normal: List of node reconstruction errors for normal graphs.
        node_errors_attack: List of node reconstruction errors for attack graphs.
        neighbor_errors_normal: List of neighborhood reconstruction errors for normal graphs.
        neighbor_errors_attack: List of neighborhood reconstruction errors for attack graphs.
        canid_errors_normal: List of CAN ID errors for normal graphs.
        canid_errors_attack: List of CAN ID errors for attack graphs.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Normalize each error type to [0, 1] for fair combination
    def normalize(arr):
        arr = np.array(arr)
        if arr.max() - arr.min() == 0:
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    # Normalize all error types
    node_n = normalize(node_errors_normal)
    node_a = normalize(node_errors_attack)
    neighbor_n = normalize(neighbor_errors_normal)
    neighbor_a = normalize(neighbor_errors_attack)
    canid_n = normalize(canid_errors_normal)
    canid_a = normalize(canid_errors_attack)
    
    # Composite error (weighted mean of normalized errors)
    # You can adjust weights if needed
    weight_node = 1/3      # 33.33%
    weight_neighbor = 1/3  # 33.33%
    weight_canid = 1/3     # 33.33%
    
    comp_n = (weight_node * node_n + weight_neighbor * neighbor_n + weight_canid * canid_n)
    comp_a = (weight_node * node_a + weight_neighbor * neighbor_a + weight_canid * canid_a)
    
    # Calculate threshold at 95th percentile of normal graphs
    comp_threshold = np.percentile(comp_n, 95) if len(comp_n) > 0 else 0
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(comp_n, bins=50, alpha=0.7, label=f'Normal (n={len(comp_n)})', color='blue', density=True)
    plt.hist(comp_a, bins=50, alpha=0.7, label=f'Attack (n={len(comp_a)})', color='red', density=True)
    plt.axvline(comp_threshold, color='green', linestyle='--', label=f'Threshold: {comp_threshold:.3f}')
    
    plt.xlabel('Composite Error (Node + Neighborhood + CAN ID)')
    plt.ylabel('Density')
    plt.title(f'Composite Error Distribution\n(Weights: Node={weight_node}, Neighbor={weight_neighbor}, CAN_ID={weight_canid})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print some statistics
    print(f"Composite Error Statistics:")
    print(f"Normal - Mean: {np.mean(comp_n):.4f}, Std: {np.std(comp_n):.4f}")
    print(f"Attack - Mean: {np.mean(comp_a):.4f}, Std: {np.std(comp_a):.4f}")
    print(f"Separation (Attack_mean - Normal_mean): {np.mean(comp_a) - np.mean(comp_n):.4f}")
    print(f"Threshold: {comp_threshold:.4f}")
    print(f"Saved neighborhood composite error histogram as '{save_path}'")

def plot_neighborhood_error_hist(neighbor_errors_normal, neighbor_errors_attack, threshold, save_path='images/neighborhood_error_hist.png'):
    """
    Plot histogram of neighborhood reconstruction errors for normal and attack graphs.

    Args:
        neighbor_errors_normal: List of neighborhood reconstruction errors for normal graphs.
        neighbor_errors_attack: List of neighborhood reconstruction errors for attack graphs.
        threshold: Threshold value for anomaly detection.
        save_path: Path to save the figure.
    
    """
    import matplotlib.pyplot as plt
    if neighbor_errors_normal and neighbor_errors_attack:
        plt.figure(figsize=(8, 5))
        plt.hist(neighbor_errors_normal, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        plt.hist(neighbor_errors_attack, bins=50, alpha=0.6, label='Attack', color='red', density=True)
        plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
        plt.xlabel('Neighborhood Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Neighborhood Error Distribution')
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved neighborhood error histogram as '{save_path}'")
    else:
        print("Not enough data to plot neighborhood error distributions.")

def plot_structural_error_hist(structural_errors_normal, structural_errors_attack, save_path="images/neighborhood_error_hist.png"):
    """
    Plot histogram of structural feature scores for normal and attack graphs.

    This function visualizes the distribution of a graph-level structural score
    (computed from features like edge density, average degree, and degree std)
    for both normal and attack graphs. The histogram helps you see if there is
    a significant difference in structural properties between normal and attack graphs.

    Args:
        structural_errors_normal: List of scores for normal graphs.
        structural_errors_attack: List of scores for attack graphs.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(structural_errors_normal, bins=50, alpha=0.7, label=f'Normal (n={len(structural_errors_normal)})', color='blue')
    plt.hist(structural_errors_attack, bins=50, alpha=0.7, label=f'Attack (n={len(structural_errors_attack)})', color='red')
    structural_threshold = np.percentile(structural_errors_normal, 95) if structural_errors_normal else 0
    plt.axvline(structural_threshold, color='green', linestyle='--', label=f'Threshold: {structural_threshold:.4f}')
    plt.xlabel('Structural Feature Score')
    plt.ylabel('Frequency')
    plt.title('Structural Feature Distribution: Normal vs Attack Graphs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Structural error histogram saved as '{save_path}'")

def plot_connectivity_error_hist(connectivity_errors_normal, connectivity_errors_attack, save_path="images/connectivity_error_hist.png"):
    """
    Plot histogram of connectivity anomaly scores for normal and attack graphs.

    This function visualizes the distribution of a graph-level connectivity anomaly score,
    which measures how much a graph's connectivity deviates from expected patterns
    (e.g., edge density, isolated nodes, degree uniformity). The histogram helps you
    see if attack graphs have more anomalous connectivity than normal graphs.

    Args:
        connectivity_errors_normal: List of scores for normal graphs.
        connectivity_errors_attack: List of scores for attack graphs.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(connectivity_errors_normal, bins=50, alpha=0.7, label=f'Normal (n={len(connectivity_errors_normal)})', color='blue')
    plt.hist(connectivity_errors_attack, bins=50, alpha=0.7, label=f'Attack (n={len(connectivity_errors_attack)})', color='red')
    connectivity_threshold = np.percentile(connectivity_errors_normal, 95) if connectivity_errors_normal else 0
    plt.axvline(connectivity_threshold, color='green', linestyle='--', label=f'Threshold: {connectivity_threshold:.4f}')
    plt.xlabel('Connectivity Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Connectivity Anomaly Distribution: Normal vs Attack Graphs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Connectivity error histogram saved as '{save_path}'")

def plot_canid_recon_hist(id_errors_normal, id_errors_attack, save_path="images/canid_recon_hist.png"):
    # Plot
    if id_errors_normal and id_errors_attack:
        plt.figure(figsize=(8, 5))
        plt.hist(id_errors_normal, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        plt.hist(id_errors_attack, bins=50, alpha=0.6, label='Attack', color='red', density=True)
        plt.xlabel('Fraction of Incorrect CAN IDs per Graph')
        plt.ylabel('Density')
        plt.title('CAN ID Reconstruction Error Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved CAN ID reconstruction error histogram as '{save_path}'")
    else:
        print("Not enough data to plot CAN ID error distributions.")

def plot_composite_error_hist(node_errors_normal, node_errors_attack,
                             edge_errors_normal, edge_errors_attack,
                             canid_errors_normal, canid_errors_attack,
                             save_path="images/composite_error_hist.png"):
    # Normalize each error type to [0, 1] for fair combination
    def normalize(arr):
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    node_n = normalize(node_errors_normal)
    node_a = normalize(node_errors_attack)
    edge_n = normalize(edge_errors_normal)
    edge_a = normalize(edge_errors_attack)
    canid_n = normalize(canid_errors_normal)
    canid_a = normalize(canid_errors_attack)
    # Composite error (mean of normalized errors)
    comp_n = (node_n + edge_n + canid_n) / 3
    comp_a = (node_a + edge_a + canid_a) / 3
    # Plot
    plt.figure(figsize=(8, 5))
    plt.hist(comp_n, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    plt.hist(comp_a, bins=50, alpha=0.6, label='Attack', color='red', density=True)
    plt.xlabel('Composite Error (normalized)')
    plt.ylabel('Density')
    plt.title('Composite Error Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved composite error histogram as '{save_path}'")

def plot_latent_space(zs, labels, save_path="images/latent_space.png"):
    tsne = TSNE(n_components=2, random_state=42)
    zs_2d = tsne.fit_transform(zs)
    plt.figure(figsize=(8,6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(zs_2d[idx, 0], zs_2d[idx, 1], label=f"Label {label}", alpha=0.6)
    plt.legend()
    plt.title("Latent Space Visualization (t-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved latent space plot as '{save_path}'")

def plot_edge_error_hist(errors_normal, errors_attack, threshold, save_path='images/edge_error_hist.png'):
    import matplotlib.pyplot as plt
    if errors_normal and errors_attack:
        plt.figure(figsize=(8, 5))
        plt.hist(errors_normal, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        plt.hist(errors_attack, bins=50, alpha=0.6, label='Attack', color='red', density=True)
        plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
        plt.xlabel('Edge Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Edge Error Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved edge error histogram as '{save_path}'")
    else:
        print("Not enough data to plot edge error distributions.")


def plot_node_recon_errors(pipeline, loader, num_graphs=8, save_path="images/node_recon_subplot.png"):
    """Plot node-level reconstruction errors for a mix of normal and attack graphs."""
    pipeline.autoencoder.eval()
    normal_graphs = []
    attack_graphs = []
    errors_normal = []
    errors_attack = []

    # Collect graphs and their node errors
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(pipeline.device)
            # FIX: Use correct 5-output format
            cont_out, canid_logits, neighbor_logits, z, kl_loss = pipeline.autoencoder(batch.x, batch.edge_index, batch.batch)
            node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
            graphs = Batch.to_data_list(batch)
            start = 0
            for graph in graphs:
                n = graph.x.size(0)
                errs = node_errors[start:start+n].cpu().numpy()
                if int(graph.y.flatten()[0]) == 0 and len(normal_graphs) < num_graphs:
                    normal_graphs.append(graph)
                    errors_normal.append(errs)
                elif int(graph.y.flatten()[0]) == 1 and len(attack_graphs) < num_graphs:
                    attack_graphs.append(graph)
                    errors_attack.append(errs)
                start += n
                if len(normal_graphs) >= num_graphs and len(attack_graphs) >= num_graphs:
                    break
            if len(normal_graphs) >= num_graphs and len(attack_graphs) >= num_graphs:
                break

    # Create the plot
    fig, axes = plt.subplots(2, num_graphs, figsize=(4*num_graphs, 8), sharey=True)
    for i in range(num_graphs):
        if i < len(errors_normal):
            axes[0, i].bar(range(len(errors_normal[i])), errors_normal[i], color='blue')
            axes[0, i].set_title(f"Normal Graph {i+1}")
            axes[0, i].set_xlabel("Node Index")
            axes[0, i].set_ylabel("Recon Error")
        if i < len(errors_attack):
            axes[1, i].bar(range(len(errors_attack[i])), errors_attack[i], color='red')
            axes[1, i].set_title(f"Attack Graph {i+1}")
            axes[1, i].set_xlabel("Node Index")
            axes[1, i].set_ylabel("Recon Error")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved node-level reconstruction error subplot as '{save_path}'")

def plot_graph_reconstruction(pipeline, loader, num_graphs=4, save_path="images/graph_recon_examples.png"):
    """
    Plots input vs. reconstructed node features for a few graphs.
    Plots 2 normal and 2 attack graphs (if available).
    """
    pipeline.autoencoder.eval()
    shown_normal = 0
    shown_attack = 0
    max_normal = 2
    max_attack = 2
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(pipeline.device)
            # FIX: Use correct 5-output format
            cont_out, canid_logits, neighbor_logits, z, kl_loss = pipeline.autoencoder(batch.x, batch.edge_index, batch.batch)
            graphs = Batch.to_data_list(batch)
            start = 0
            for i, graph in enumerate(graphs):
                n = graph.x.size(0)
                input_feats = graph.x.cpu().numpy()
                recon_feats = cont_out[start:start+n].cpu().numpy()
                canid_pred = canid_logits[start:start+n].argmax(dim=1).cpu().numpy()
                input_canid = input_feats[:, 0]
                start += n

                # Exclude CAN ID (column 0) for main feature comparison
                input_payload = input_feats[:, 1:]
                recon_payload = recon_feats

                label = int(graph.y.flatten()[0])
                if label == 0 and shown_normal < max_normal:
                    graph_type = "Normal"
                    shown_normal += 1
                elif label == 1 and shown_attack < max_attack:
                    graph_type = "Attack"
                    shown_attack += 1
                else:
                    continue

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                # Payload/continuous features
                im0 = axes[0].imshow(input_payload, aspect='auto', interpolation='none')
                axes[0].set_title(f"Input Payload/Features\n({graph_type} Graph {shown_normal if label==0 else shown_attack})")
                plt.colorbar(im0, ax=axes[0])
                im1 = axes[1].imshow(recon_payload, aspect='auto', interpolation='none')
                axes[1].set_title(f"Reconstructed Payload/Features\n({graph_type} Graph {shown_normal if label==0 else shown_attack})")
                plt.colorbar(im1, ax=axes[1])
                # CAN ID comparison
                axes[2].plot(input_canid, label="Input CAN ID", marker='o')
                axes[2].plot(canid_pred, label="Pred CAN ID", marker='*')
                axes[2].set_title("CAN ID (Input vs Pred)")
                axes[2].set_xlabel("Node Index")
                axes[2].set_ylabel("CAN ID Value")
                axes[2].legend()
                plt.suptitle(f"{graph_type} Graph {shown_normal if label==0 else shown_attack} (Label: {label})")
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(f"{save_path.rstrip('.png')}_{graph_type.lower()}_{shown_normal if label==0 else shown_attack}.png")
                plt.close()

                # Stop if we've shown enough
                if shown_normal >= max_normal and shown_attack >= max_attack:
                    return

def plot_feature_histograms(graphs, feature_names=None, save_path="images/feature_histograms.png"):
    all_x = torch.cat([g.x for g in graphs], dim=0).cpu().numpy()
    num_features = all_x.shape[1]
    # Extend feature_names if too short
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(num_features)]
    elif len(feature_names) < num_features:
        feature_names = feature_names + [f"Feature {i}" for i in range(len(feature_names), num_features)]
    n_cols = 5
    n_rows = int(np.ceil(num_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for i in range(num_features):
        axes[i].hist(all_x[:, i], bins=50, color='skyblue', edgecolor='black')
        axes[i].set_title(feature_names[i])
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved feature histograms as '{save_path}'")

def plot_recon_error_hist(errors_normal, errors_attack, threshold, save_path='images/recon_error_hist.png'):
    """
    Plots the histogram of reconstruction errors for normal and attack graphs.
    """
    if errors_normal and errors_attack:
        plt.figure(figsize=(8, 5))
        plt.hist(errors_normal, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        plt.hist(errors_attack, bins=50, alpha=0.6, label='Attack', color='red', density=True)
        plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
        plt.xlabel('Mean Graph Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Reconstruction Error Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved reconstruction error histogram as '{save_path}'")
    else:
        print("Not enough data to plot error distributions.")         
def print_graph_stats(graphs, label):
    import torch
    all_x = torch.cat([g.x for g in graphs], dim=0)
    print(f"\n--- {label} Graphs ---")
    print(f"Num graphs: {len(graphs)}")
    print(f"Node feature means: {all_x.mean(dim=0)}")
    print(f"Node feature stds: {all_x.std(dim=0)}")
    print(f"Unique CAN IDs: {all_x[:,0].unique()}")
    print(f"Sample node features:\n{all_x[:5]}")

def print_graph_structure(graphs, label):
    num_nodes = [g.num_nodes for g in graphs]
    num_edges = [g.num_edges for g in graphs]
    print(f"\n--- {label} Graphs Structure ---")
    print(f"Avg num nodes: {sum(num_nodes)/len(num_nodes):.2f}")
    print(f"Avg num edges: {sum(num_edges)/len(num_edges):.2f}")