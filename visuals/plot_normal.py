import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# load
df = pd.read_csv("D:/quadeer/task1/dataset/graphdata/preprocessed_normal_run_data.csv")

# 添加 Source 和 Target 列（如果尚未存在）
if 'Source' not in df.columns or 'Target' not in df.columns:
    df['Source'] = df['CAN_ID']
    df['Target'] = df['CAN_ID'].shift(-1)  
    df = df.dropna(subset=['Target'])     
    df['Target'] = df['Target'].astype(int)
    print(" add Source and  Target column")


window_size = 50
stride = 50

def create_graph_from_window_by_source_target(df_window):

    G = nx.DiGraph()

    edges = list(zip(df_window['Source'], df_window['Target']))
    
    # delete useless edges
    edges = [(src, dst) for src, dst in edges if pd.notnull(src) and pd.notnull(dst)]
    unique_nodes = set([src for src, _ in edges] + [dst for _, dst in edges])
    G.add_nodes_from(unique_nodes)
    G.add_edges_from(edges)

    return G

def visualize_graphs_as_subplots(graph_list, labels, rows=2, cols=3, save_path="D:/quadeer/task1/dataset/graphdata/normal_graph.png"):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    for i, (G, label) in enumerate(zip(graph_list, labels)):
        if i >= len(axes):
            break

        pos = nx.spring_layout(G, seed=42)
        sorted_nodes = sorted(G.nodes())
        id_map = {real_id: idx + 1 for idx, real_id in enumerate(sorted_nodes)}

        color = "#ADD8E6" if label == 0 else "#FF9999"
        label_text = "Attack" if label == 1 else "No Attack"

        nx.draw(
            G,
            pos,
            ax=axes[i],
            with_labels=True,
            labels=id_map,
            node_color=color,
            node_size=500,
            edge_color="gray",
            font_size=8
        )

        axes[i].set_title(f"Graph {i + 1}: {label_text}", fontsize=14)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"saved to：{save_path}")

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    for i, (G, label) in enumerate(zip(graph_list, labels)):
        if i >= len(axes): break

        pos = nx.spring_layout(G, seed=42)
        sorted_nodes = sorted(G.nodes())
        id_map = {real_id: idx + 1 for idx, real_id in enumerate(sorted_nodes)}

        color = "#ADD8E6" if label == 0 else "#FF9999"
        label_text = "Attack" if label == 1 else "No Attack"

        nx.draw(
            G,
            pos,
            ax=axes[i],
            with_labels=True,
            labels=id_map,
            node_color=color,
            node_size=500,
            edge_color="gray",
            font_size=8
        )

        axes[i].set_title(f"Graph {i + 1}: {label_text}", fontsize=14)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


graphs = []
labels = []

for start in range(0, len(df) - window_size + 1, stride):
    window = df.iloc[start:start + window_size]
    label = 1 if window['Label'].sum() > 0 else 0
    G = create_graph_from_window_by_source_target(window)
    graphs.append(G)
    labels.append(label)

visualize_graphs_as_subplots(graphs[:6], labels[:6])
