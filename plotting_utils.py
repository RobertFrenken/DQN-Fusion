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

def plot_node_recon_errors(pipeline, loader, num_graphs=8, save_path="node_recon_subplot.png"):
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
            x_recon, _ = pipeline.autoencoder(batch.x, batch.edge_index, batch.batch)
            node_errors = (x_recon[:, 1:] - batch.x[:, 1:]).pow(2).mean(dim=1)
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

    # --- Debug: Print last node info for each plotted graph ---
    print("Last node info for plotted graphs:")
    for i, graph in enumerate(normal_graphs):
        print(f"Normal Graph {i+1} last node features: {graph.x[-1]}")
        print(f"Normal Graph {i+1} last node CAN ID: {graph.x[-1,0]}")
        # Degree: count how many times last node index appears in edge_index
        last_idx = graph.x.size(0) - 1
        degree = (graph.edge_index[0] == last_idx).sum().item() + (graph.edge_index[1] == last_idx).sum().item()
        print(f"Normal Graph {i+1} last node degree: {degree}")
    for i, graph in enumerate(attack_graphs):
        print(f"Attack Graph {i+1} last node features: {graph.x[-1]}")
        print(f"Attack Graph {i+1} last node CAN ID: {graph.x[-1,0]}")
        last_idx = graph.x.size(0) - 1
        degree = (graph.edge_index[0] == last_idx).sum().item() + (graph.edge_index[1] == last_idx).sum().item()
        print(f"Attack Graph {i+1} last node degree: {degree}")
    for i, graph in enumerate(normal_graphs + attack_graphs):
        print(f"Graph {i+1} last node features: {graph.x[-2]}")  # -2 to skip virtual node
        n = graph.x.size(0)
        recon_feats = pipeline.autoencoder(graph.x, graph.edge_index, torch.zeros(n, dtype=torch.long, device=graph.x.device))
        print(f"Graph {i+1} last node recon: {recon_feats[-2]}")
    
    fig, axes = plt.subplots(2, num_graphs, figsize=(4*num_graphs, 8), sharey=True)
    for i in range(num_graphs):
        axes[0, i].bar(range(len(errors_normal[i])), errors_normal[i], color='blue')
        axes[0, i].set_title(f"Normal Graph {i+1}")
        axes[0, i].set_xlabel("Node Index")
        axes[0, i].set_ylabel("Recon Error")
        axes[1, i].bar(range(len(errors_attack[i])), errors_attack[i], color='red')
        axes[1, i].set_title(f"Attack Graph {i+1}")
        axes[1, i].set_xlabel("Node Index")
        axes[1, i].set_ylabel("Recon Error")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved node-level reconstruction error subplot as '{save_path}'")

def plot_graph_reconstruction(pipeline, loader, num_graphs=3, save_path="graph_recon_examples.png"):
    """
    Plots input vs. reconstructed node features for a few graphs.
    Left: Only payload/continuous features (excluding CAN ID).
    Right: CAN ID input vs. reconstructed CAN ID (for visualization).
    """
    pipeline.autoencoder.eval()
    shown = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(pipeline.device)
            x_recon, _ = pipeline.autoencoder(batch.x, batch.edge_index, batch.batch)
            graphs = Batch.to_data_list(batch)
            start = 0
            for i, graph in enumerate(graphs):
                n = graph.x.size(0)
                input_feats = graph.x.cpu().numpy()
                recon_feats = x_recon[start:start+n].cpu().numpy()
                start += n

                # Exclude CAN ID (column 0) for main feature comparison
                input_payload = input_feats[:, 1:]
                recon_payload = recon_feats[:, 1:]

                # CAN ID comparison (column 0)
                input_canid = input_feats[:, 0]
                recon_canid = recon_feats[:, 0]

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                # Payload/continuous features
                im0 = axes[0].imshow(input_payload, aspect='auto', interpolation='none')
                axes[0].set_title(f"Input Payload/Features\n(Graph {shown+1})")
                plt.colorbar(im0, ax=axes[0])
                im1 = axes[1].imshow(recon_payload, aspect='auto', interpolation='none')
                axes[1].set_title(f"Reconstructed Payload/Features\n(Graph {shown+1})")
                plt.colorbar(im1, ax=axes[1])
                # CAN ID comparison
                axes[2].plot(input_canid, label="Input CAN ID", marker='o')
                axes[2].plot(recon_canid, label="Recon CAN ID", marker='x')
                axes[2].set_title("CAN ID (Input vs Recon)")
                axes[2].set_xlabel("Node Index")
                axes[2].set_ylabel("CAN ID Value")
                axes[2].legend()
                plt.suptitle(f"Graph {shown+1} (Label: {int(graph.y.flatten()[0])})")
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(f"{save_path.rstrip('.png')}_{shown+1}.png")
                plt.close()
                shown += 1
                if shown >= num_graphs:
                    return

def plot_feature_histograms(graphs, feature_names=None, save_path="feature_histograms.png"):
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