import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import torch
import sys
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import json
from models.models import GATWithJK, Autoencoder
from preprocessing import graph_creation
from training_utils import PyTorchTrainer, PyTorchDistillationTrainer, DistillationTrainer
import torch.profiler
from torch_geometric.utils import unbatch
from torch_geometric.data import Batch

class GATPipeline:
    def __init__(self, num_ids, embedding_dim=8, device='cpu'):
        self.device = device
        self.autoencoder = Autoencoder(num_ids=num_ids, in_channels=10, embedding_dim=embedding_dim).to(device)
        self.classifier = GATWithJK(num_ids=num_ids, in_channels=10, hidden_channels=32, out_channels=1, num_layers=3, heads=8, embedding_dim=embedding_dim).to(device)
        self.threshold = 0.0000

    
    # NOTE: How does the train1 know to only train on normal graphs?
    # is stage 2 handling sample by sample or the entire batch?

    def train_stage1(self, train_loader, epochs=50):
        self.autoencoder.train()
        opt = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            for batch in train_loader:
                batch = batch.to(self.device)
                recon = self.autoencoder(batch.x, batch.edge_index)
                loss = nn.MSELoss()(recon[:, 1:], batch.x[:, 1:])
                opt.zero_grad()
                loss.backward()
                opt.step()
        # Set reconstruction threshold (95th percentile)
        errors = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                recon = self.autoencoder(batch.x, batch.edge_index)
                errors.append((recon[:, 1:] - batch.x[:, 1:]).pow(2).mean(dim=1))
        # self.threshold = torch.cat(errors).quantile(0.0001).item()
        self.threshold = 0

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
        errors_normal = []
        errors_attack = []
        with torch.no_grad():
            for batch in full_loader:
                batch = batch.to(self.device)
                recon = self.autoencoder(batch.x, batch.edge_index)
                error = (recon[:, 1:] - batch.x[:, 1:]).pow(2).mean(dim=1)
                graphs = Batch.to_data_list(batch)
                batch_vec = batch.batch.cpu().numpy()
                start = 0
                for graph in graphs:
                    num_nodes = graph.x.size(0)
                    node_errors = error[start:start+num_nodes]
                    if node_errors.numel() > 0:
                        graph_error = node_errors.mean().item()
                        if graph.y.item() == 0:
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
                recon = self.autoencoder(batch.x, batch.edge_index)
                error = (recon[:, 1:] - batch.x[:, 1:]).pow(2).mean(dim=1)
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
        
        # Print label distribution in filtered graphs
        labels = [graph.y.item() for graph in filtered]
        unique, counts = np.unique(labels, return_counts=True)
        print("Label distribution in filtered graphs (for classifier):")
        for u, c in zip(unique, counts):
            print(f"Label {u}: {c} samples")
        # Train classifier
        self.classifier.train()
        opt = torch.optim.Adam(self.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            for batch in DataLoader(filtered, batch_size=32, shuffle=True):
                batch = batch.to(self.device)
                preds = self.classifier(batch)
                loss = criterion(preds.squeeze(), batch.y.float())
                
                opt.zero_grad()
                loss.backward()
                opt.step()
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
            recon = self.autoencoder(data.x, data.edge_index)
            error = (recon - data.x).pow(2).mean(dim=1)
            
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

    dataset, id_mapping = graph_creation(root_folder, return_id_mapping=True)
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
    normal_indices = [i for i, data in enumerate(train_dataset) if data.y.item() == 0]
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
    pipeline.train_stage1(train_loader, epochs=EPOCHS)

    print("Stage 2: Training classifier on filtered graphs...")
    # For classifier, use all graphs (normal + attack)
    full_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    pipeline.train_stage2(full_train_loader, epochs=EPOCHS)

    # Save models
    save_folder = "saved_models"
    os.makedirs(save_folder, exist_ok=True)
    torch.save(pipeline.autoencoder.state_dict(), os.path.join(save_folder, f'autoencoder_{KEY}.pth'))
    torch.save(pipeline.classifier.state_dict(), os.path.join(save_folder, f'classifier_{KEY}.pth'))
    print(f"Models saved in '{save_folder}'.")

    # Optionally: Evaluate on test set
    print("Evaluating on test set...")
    preds = []
    labels = []
    for batch in test_loader:
        batch = batch.to(device)
        pred = pipeline.predict(batch)
        preds.append(pred.cpu())
        labels.append(batch.y.cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    accuracy = (preds == labels).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save metrics
    metrics = {"test_accuracy": accuracy}
    with open('ad_metrics.json', 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    print("Metrics saved as 'ad_metrics.json'.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")