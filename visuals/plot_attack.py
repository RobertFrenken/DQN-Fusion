import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

input_dir = "D:/quadeer/task1/dataset/graphdata"
output_dir = input_dir 
window_size = 50
stride = 50

filenames = [
    f for f in os.listdir(input_dir)
    if f.startswith("preprocessed_") and f.endswith(".csv") and "normal" not in f.lower()
]

def create_graph_from_window(df_window):
   
    G = nx.DiGraph()
    edges = list(zip(df_window['Source'], df_window['Target']))
    edges = [(src, dst) for src, dst in edges if pd.notnull(src) and pd.notnull(dst)]
    G.add_edges_from(edges)

  
    label_dict = df_window.groupby('Source')['Label'].max().to_dict()
    nx.set_node_attributes(G, label_dict, name='attack')
    return G

def visualize_attack_graphs(graphs, dataset_name):
    selected_graphs = []

    for G in graphs:
        if len(G.nodes) >= 20:
            labels = nx.get_node_attributes(G, "attack")
            if any(v == 1 for v in labels.values()):
                selected_graphs.append(G)
            if len(selected_graphs) == 6:
                break

    if not selected_graphs:
        print(f" {dataset_name}: no images suit the need")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, G in enumerate(selected_graphs):
        pos = nx.spring_layout(G, seed=42)
        sorted_nodes = sorted(G.nodes())
        id_map = {real_id: idx + 1 for idx, real_id in enumerate(sorted_nodes)}

        has_attack = any(G.nodes[n].get("attack", 0) == 1 for n in G.nodes)

        node_colors = ["#FF9999" if has_attack else "#ADD8E6"] * G.number_of_nodes()

        nx.draw(
            G, pos, ax=axes[i], with_labels=True, labels=id_map,
            node_color=node_colors, node_size=500, edge_color="gray", font_size=8
        )
        axes[i].set_title(f"{dataset_name} Graph {i+1}", fontsize=14)

    for j in range(len(selected_graphs), 6):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{dataset_name}_attack_graphs.png")
    plt.savefig(out_path)
    plt.close()
    print(f"saved images：{out_path}")


for filename in filenames:
    df_path = os.path.join(input_dir, filename)
    df = pd.read_csv(df_path)


    if "Source" not in df.columns or "Target" not in df.columns:
        df["Source"] = df["CAN_ID"]
        df["Target"] = df["CAN_ID"].shift(-1)
        df.dropna(subset=["Target"], inplace=True)
        df["Target"] = df["Target"].astype(int)
    graphs = []
    for start in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[start:start + window_size]
        G = create_graph_from_window(window)
        graphs.append(G)

    dataset_name = filename.replace("preprocessed_", "").replace(".csv", "")
    visualize_attack_graphs(graphs, dataset_name)
