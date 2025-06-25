import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import torch
import sys
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from models.models import GATWithJK, Autoencoder, GraphAutoencoder
from preprocessing import graph_creation, build_id_mapping_from_normal
from training_utils import PyTorchTrainer, PyTorchDistillationTrainer, DistillationTrainer
from torch_geometric.data import Batch
from sklearn.metrics import confusion_matrix
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
class GATPipeline:
    def __init__(self, num_ids, embedding_dim=8, device='cpu'):
        self.device = device
        self.autoencoder = GraphAutoencoder(num_ids=num_ids, in_channels=11, embedding_dim=embedding_dim).to(device)
        self.classifier = GATWithJK(num_ids=num_ids, in_channels=11, hidden_channels=32, out_channels=1, num_layers=3, heads=8, embedding_dim=embedding_dim).to(device)
        self.threshold = 0.0000

    
    # NOTE: How does the train1 know to only train on normal graphs?
    # is stage 2 handling sample by sample or the entire batch?
    # NOTE: The reconstruction error is likely low since most of the nodes
    # in the graph are normal, there is really only one node that is an attack,
    # so the error would be low. I will have to check this to see if error is any
    # node or the entire graph.

    def train_stage1(self, train_loader, epochs=100):
        self.autoencoder.train()
        opt = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(epochs):
            for batch in train_loader:
                batch = batch.to(self.device)
                x_recon, z = self.autoencoder(batch.x, batch.edge_index, batch.batch)



                # Focus loss on payload features (columns 1-8)
                payload_loss = (x_recon[:, 1:9] - batch.x[:, 1:9]).pow(2).mean()
                # Optionally, add small weighted loss for count and position (last two columns)
                count_loss = (x_recon[:, -2] - batch.x[:, -2]).pow(2).mean()
                position_loss = (x_recon[:, -1] - batch.x[:, -1]).pow(2).mean()
                node_loss = payload_loss + 0.1 * (count_loss + position_loss)
                
                # Node feature loss
                # singleton_mask = (batch.x[:, -2] == 1.0)
                # node_loss = nn.MSELoss()(x_recon[~singleton_mask, 1:], batch.x[~singleton_mask, 1:])

                # Node reconstruction error (exclude CAN ID column)
                # node_errors = (x_recon[:, 1:] - batch.x[:, 1:]).pow(2).mean(dim=1)


                # # Mask out the last node of each graph in the batch
                # mask = torch.ones_like(node_errors, dtype=torch.bool)
                # graphs = Batch.to_data_list(batch)
                # start = 0
                # for graph in graphs:
                #     n = graph.x.size(0)
                #     mask[start + n - 1] = False  # Mask last node
                #     start += n

                # node_loss = node_errors[mask].mean()
                
                
                # Edge reconstruction loss
                pos_edge_index = batch.edge_index
                pos_edge_preds = self.autoencoder.decode_edge(z, pos_edge_index)
                pos_edge_labels = torch.ones(pos_edge_preds.size(0), device=pos_edge_preds.device)
                num_nodes = batch.x.size(0)
                num_neg = pos_edge_index.size(1)
                neg_edge_index = torch.randint(0, num_nodes, pos_edge_index.shape, device=pos_edge_index.device)
                neg_edge_preds = self.autoencoder.decode_edge(z, neg_edge_index)
                neg_edge_labels = torch.zeros(neg_edge_preds.size(0), device=neg_edge_preds.device)
                edge_preds = torch.cat([pos_edge_preds, neg_edge_preds])
                edge_labels = torch.cat([pos_edge_labels, neg_edge_labels])
                edge_loss = nn.BCELoss()(edge_preds, edge_labels)
                
                
                loss = node_loss + edge_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
        # Set reconstruction threshold (95th percentile)
        errors = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                x_recon, _ = self.autoencoder(batch.x, batch.edge_index, batch.batch)
                errors.append((x_recon[:, 1:] - batch.x[:, 1:]).pow(2).mean(dim=1))
        self.threshold = torch.cat(errors).quantile(0.5).item()
        # self.threshold = 0

    def train_stage2(self, full_loader, epochs=50):
        """Train classifier on filtered graphs"""
        filtered = []

        # --- Print label distribution in full_loader ---
        all_labels = []
        for batch in full_loader:
            all_labels.extend(batch.y.cpu().numpy())
        unique, counts = np.unique(all_labels, return_counts=True)
        print("Label distribution in full training set (Stage 2):")
        for u, c in zip(unique, counts):
            print(f"Label {u}: {c} samples")

        # --- Print mean reconstruction error for normal vs. attack graphs ---
        # 3x3 subplot of graphs reconstruction error distribution BY NODEs
        # based of observation, develop a new way to threhold
        errors_normal = []
        errors_attack = []
        with torch.no_grad():
            for batch in full_loader:
                batch = batch.to(self.device)
                x_recon, _ = self.autoencoder(batch.x, batch.edge_index, batch.batch)
                error = (x_recon[:, 1:] - batch.x[:, 1:]).pow(2).mean(dim=1)
                graphs = Batch.to_data_list(batch)
                batch_vec = batch.batch.cpu().numpy()
                start = 0
                for graph in graphs:
                    num_nodes = graph.x.size(0)
                    node_errors = error[start:start+num_nodes]
                    if node_errors.numel() > 0:
                        # graph_error = node_errors.mean().item()
                        graph_error = node_errors.max().item()  # <-- Use max instead of mean
                        if int(graph.y.flatten()[0]) == 0:
                            errors_normal.append(graph_error)
                        else:
                            errors_attack.append(graph_error)
                    start += num_nodes
        print(f"Processed {len(errors_normal)} normal graphs, {len(errors_attack)} attack graphs for error stats.")
        print(f"Mean reconstruction error (normal): {np.mean(errors_normal) if errors_normal else 'N/A'}")
        print(f"Mean reconstruction error (attack): {np.mean(errors_attack) if errors_attack else 'N/A'}")
        if errors_normal and errors_attack:
            plt.figure(figsize=(8, 5))
            plt.hist(errors_normal, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
            plt.hist(errors_attack, bins=50, alpha=0.6, label='Attack', color='red', density=True)
            plt.axvline(self.threshold, color='black', linestyle='--', label='Threshold')
            plt.xlabel('Mean Graph Reconstruction Error')
            plt.ylabel('Density')
            plt.title('Reconstruction Error Distribution')
            plt.legend()
            plt.tight_layout()
            plt.savefig('recon_error_hist.png')
            plt.close()
            print("Saved reconstruction error histogram as 'recon_error_hist.png'")
        else:
            print("Not enough data to plot error distributions.")
        
        print(f"Threshold: {self.threshold}")
        print(f"Error stats: min={error.min().item()}, max={error.max().item()}, mean={error.mean().item()}")
        # --- Filter using autoencoder as before ---
        with torch.no_grad():
            for batch in full_loader:
                batch = batch.to(self.device)
                x_recon, _ = self.autoencoder(batch.x, batch.edge_index, batch.batch)
                error = (x_recon[:, 1:] - batch.x[:, 1:]).pow(2).mean(dim=1)
                graphs = Batch.to_data_list(batch)
                start = 0
                for graph in graphs:
                    num_nodes = graph.x.size(0)
                    node_errors = error[start:start+num_nodes]
                    if node_errors.numel() > 0 and (node_errors > self.threshold).any():
                        filtered.append(graph)
                    start += num_nodes

        print(f"Anomaly threshold: {self.threshold}")
        print(f"Number of graphs exceeding threshold: {len(filtered)}")

        if not filtered:
            print("No graphs exceeded the anomaly threshold. Skipping classifier training.")
            return
        
        # TESTING EVEN ATTACK AND ATTACK-FREE SPLIT
        # After filtering, before training:
        labels = [int(graph.y.flatten()[0]) for graph in filtered]
        attack_graphs = [g.cpu() for g in filtered if g.y.item() == 1]
        normal_graphs = [g.cpu() for g in filtered if g.y.item() == 0]

        # After splitting filtered into attack_graphs and normal_graphs
        print_graph_stats(normal_graphs, "Normal")
        print_graph_stats(attack_graphs, "Attack")
        print_graph_structure(normal_graphs, "Normal")
        print_graph_structure(attack_graphs, "Attack")

        # Downsample normal graphs to match the number of attack graphs
        num_attack = len(attack_graphs)
        if len(normal_graphs) > num_attack:
            random.seed(42)
            normal_graphs = random.sample(normal_graphs, num_attack)

        # Combine and shuffle
        balanced_filtered = attack_graphs + normal_graphs
        np.random.shuffle(balanced_filtered)

        # Use balanced_filtered for training
        filtered = balanced_filtered
        # Print label distribution in filtered graphs
        labels = [int(graph.y.flatten()[0]) for graph in filtered]
        unique, counts = np.unique(labels, return_counts=True)
        print("Label distribution in filtered graphs (for classifier):")
        for u, c in zip(unique, counts):
            print(f"Label {u}: {c} samples")
        
        # After filtering, before training:
        labels = [int(graph.y.flatten()[0]) for graph in filtered]
        num_pos = sum(labels)
        num_neg = len(labels) - num_pos

        # Avoid division by zero
        # num_pos = 0
        if num_pos == 0:
            pos_weight = torch.tensor(1.0, device=self.device)
        else:
            pos_weight = torch.tensor(num_neg / num_pos, device=self.device)
        print(f"Using pos_weight={pos_weight.item():.4f} for BCEWithLogitsLoss")
        # Train classifier
        self.classifier.train()
        opt = torch.optim.Adam(self.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch in DataLoader(filtered, batch_size=32, shuffle=True):
                batch = batch.to(self.device)
                preds = self.classifier(batch)
                loss = criterion(preds.squeeze(), batch.y.float())
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                num_batches += 1

                if epoch == 1 and num_batches == 1:
                    print("Batch features (x):", batch.x[:5])
                    print("Batch labels (y):", batch.y[:5])
                    print("Classifier raw outputs:", preds[:5].detach().cpu().numpy())
                    print("Unique CAN ID indices in batch:", batch.x[:, 0].unique())
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            # Print accuracy every 10 epochs
            # (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            if True:
                self.classifier.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for batch in DataLoader(filtered, batch_size=32):
                        batch = batch.to(self.device)
                        out = self.classifier(batch)
                        pred_labels = (out.squeeze() > 0.5).long()
                        all_preds.append(pred_labels.cpu())
                        all_labels.append(batch.y.cpu())
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                acc = (all_preds == all_labels).float().mean().item()
                print(f"Classifier accuracy at epoch {epoch+1}: {acc:.4f}")
                
                self.classifier.train()

    def predict(self, data):
        """Full two-stage prediction"""
        data = data.to(self.device)
        
        # Stage 1: Anomaly detection
        with torch.no_grad():
            x_recon, _ = self.autoencoder(data.x, data.edge_index, data.batch)
            error = (x_recon - data.x).pow(2).mean(dim=1)
            
            # Process each graph in batch
            preds = []
            for graph in Batch.to_data_list(data):
                node_mask = (data.batch == graph.batch)
                if (error[node_mask] > self.threshold).any():
                    # Stage 2: Classification
                    prob = self.classifier(graph.x, graph.edge_index, 
                                         graph.batch).item()
                    preds.append(1 if prob > 0.5 else 0)
                else:
                    preds.append(0)
            
            return torch.tensor(preds, device=self.device)


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    config_dict = OmegaConf.to_container(config, resolve=True)
    print(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Model is using device: {device}')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    root_folders = {'hcrl_ch': r"datasets/can-train-and-test-v1.5/hcrl-ch",
                    'hcrl_sa': r"datasets/can-train-and-test-v1.5/hcrl-sa",
                    'set_01' : r"datasets/can-train-and-test-v1.5/set_01",
                    'set_02' : r"datasets/can-train-and-test-v1.5/set_02",
                    'set_03' : r"datasets/can-train-and-test-v1.5/set_03",
                    'set_04' : r"datasets/can-train-and-test-v1.5/set_04",
    }
    KEY = config_dict['root_folder']
    root_folder = root_folders[KEY]
    print(f"Root folder: {root_folder}")

    # Step 1: Build ID mapping from only normal graphs
    id_mapping = build_id_mapping_from_normal(root_folder)

    dataset = graph_creation(root_folder, id_mapping=id_mapping, window_size=100)
    num_ids = len(id_mapping)
    embedding_dim = 8  # or your choice
    print(f"Number of graphs: {len(dataset)}")

    for data in dataset:
        assert not torch.isnan(data.x).any(), "Dataset contains NaN values!"
        assert not torch.isinf(data.x).any(), "Dataset contains Inf values!"

    DATASIZE = config_dict['datasize']
    EPOCHS = config_dict['epochs']
    LR = config_dict['lr']
    BATCH_SIZE = config_dict['batch_size']
    TRAIN_RATIO = config_dict['train_ratio']
    USE_FOCAL_LOSS = config_dict['use_focal_loss']

    print("Size of the total dataset: ", len(dataset))

    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator1)
    print('Size of DATASIZE: ', DATASIZE)
    print('Size of Training dataset: ', len(train_dataset))
    print('Size of Testing dataset: ', len(test_dataset))

    # --------- FILTER NORMAL GRAPHS FOR AUTOENCODER TRAINING ----------
    # Only keep graphs with label == 0 for autoencoder training
    normal_indices = [i for i, data in enumerate(train_dataset) if int(data.y.flatten()[0]) == 0]
    if DATASIZE < 1.0:
        subset_size = int(len(normal_indices) * DATASIZE)
        indices = np.random.choice(normal_indices, subset_size, replace=False)
    else:
        indices = normal_indices
    normal_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(normal_subset, batch_size=BATCH_SIZE, shuffle=True)
    # ---------------------------------------------------------------

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('Size of Training dataloader: ', len(train_loader))
    print('Size of Testing dataloader: ', len(test_loader))
    print('Size of Training dataloader (samples): ', len(train_loader.dataset))
    print('Size of Testing dataloader (samples): ', len(test_loader.dataset))

    pipeline = GATPipeline(num_ids=num_ids, embedding_dim=embedding_dim, device=device)

    print("Stage 1: Training autoencoder for anomaly detection...")
    pipeline.train_stage1(train_loader, epochs=50)

    # Visualize input vs. reconstructed features for a few graphs
    plot_graph_reconstruction(pipeline, train_loader, num_graphs=3, save_path="graph_recon_examples.png")

    # Visualize node-level reconstruction errors
    # plot_node_recon_errors(pipeline, full_train_loader, num_graphs=5, save_path="node_recon_subplot.png")

    print("Stage 2: Training classifier on filtered graphs...")
    # For classifier, use all graphs (normal + attack)
    full_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Visualize node-level reconstruction errors
    plot_node_recon_errors(pipeline, full_train_loader, num_graphs=5, save_path="node_recon_subplot.png")
    # pipeline.train_stage2(full_train_loader, epochs=EPOCHS)

    # # Save models
    # save_folder = "saved_models"
    # os.makedirs(save_folder, exist_ok=True)
    # torch.save(pipeline.autoencoder.state_dict(), os.path.join(save_folder, f'autoencoder_{KEY}.pth'))
    # torch.save(pipeline.classifier.state_dict(), os.path.join(save_folder, f'classifier_{KEY}.pth'))
    # print(f"Models saved in '{save_folder}'.")

    # # Optionally: Evaluate on test set
    # print("Evaluating on test set...")
    # test_labels = [data.y.item() for data in test_dataset]
    # unique, counts = np.unique(test_labels, return_counts=True)
    # print("Test set label distribution:")
    # for u, c in zip(unique, counts):
    #     print(f"Label {u}: {c} samples")
    # preds = []
    # labels = []
    # for batch in test_loader:
    #     batch = batch.to(device)
    #     pred = pipeline.predict(batch)
    #     preds.append(pred.cpu())
    #     labels.append(batch.y.cpu())
    # preds = torch.cat(preds)
    # labels = torch.cat(labels)
    # accuracy = (preds == labels).float().mean().item()
    # print(f"Test Accuracy: {accuracy:.4f}")

    # # Print confusion matrix
    # cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
    # print("Confusion Matrix:")
    # print(cm)

    # # Save metrics
    # metrics = {"test_accuracy": accuracy}
    # with open('ad_metrics.json', 'w') as f:
    #     import json
    #     json.dump(metrics, f, indent=4)
    # print("Metrics saved as 'ad_metrics.json'.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")