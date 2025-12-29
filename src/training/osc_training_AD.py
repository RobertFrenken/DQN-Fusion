"""
GAT-Based Anomaly Detection for CAN Bus Networks
Two-stage pipeline: Autoencoder (anomaly detection) + GAT Classifier

This module implements the training pipeline for CAN bus intrusion detection
using graph neural networks with neighborhood reconstruction.
"""

import numpy as np
import os
import torch
import psutil
import hydra
import time
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
from typing import List, Tuple

from src.utils.utils_logging import setup_gpu_optimization, log_memory_usage, cleanup_memory
from src.models.pipeline import GATPipeline
from src.preprocessing.preprocessing import graph_creation, build_id_mapping_from_normal
from torch_geometric.data import Batch
from src.utils.plotting_utils import (
    plot_feature_histograms,
    plot_node_recon_errors, 
    plot_graph_reconstruction,
    plot_latent_space,
)

# Configuration Constants
DATASET_PATHS = {
    'hcrl_ch': r"datasets/can-train-and-test-v1.5/hcrl-ch",
    'hcrl_sa': r"datasets/can-train-and-test-v1.5/hcrl-sa",
    'set_01': r"datasets/can-train-and-test-v1.5/set_01",
    'set_02': r"datasets/can-train-and-test-v1.5/set_02",
    'set_03': r"datasets/can-train-and-test-v1.5/set_03",
    'set_04': r"datasets/can-train-and-test-v1.5/set_04",
}


def create_optimized_data_loaders(train_subset=None, test_dataset=None, full_train_dataset=None, 
                                 batch_size: int = 1024, device: str = 'cuda') -> List[DataLoader]:
    """Create optimized data loaders with appropriate batch sizing and worker configuration."""
    if torch.cuda.is_available() and device == 'cuda':
        batch_size = 2048
        num_workers = 6
        pin_memory = True
        prefetch_factor = 3
        persistent_workers = True
    else:
        batch_size = min(batch_size, 1024)
        num_workers = 4
        pin_memory = False
        prefetch_factor = None
        persistent_workers = False

    print(f"✓ Optimized DataLoader: batch_size={batch_size}, workers={num_workers}")

    loaders = []
    loader_configs = [
        (train_subset, True),
        (test_dataset, False),
        (full_train_dataset, True)
    ]
    for dataset, shuffle in loader_configs:
        if dataset is not None:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor
            )
            loaders.append(loader)
    return loaders if len(loaders) > 1 else loaders[0]

def extract_latent_vectors(pipeline, loader) -> Tuple[np.ndarray, np.ndarray]:
    """Extract latent vectors (graph embeddings) and labels from data loader."""
    pipeline.autoencoder.eval()
    latent_vectors, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(pipeline.device)
            _, _, _, z, _ = pipeline.autoencoder(batch.x, batch.edge_index, batch.batch)
            graphs = Batch.to_data_list(batch)
            start = 0
            for graph in graphs:
                num_nodes = graph.x.size(0)
                graph_embedding = z[start:start+num_nodes].mean(dim=0).cpu().numpy()
                latent_vectors.append(graph_embedding)
                labels.append(int(graph.y.flatten()[0]))
                start += num_nodes
    return np.array(latent_vectors), np.array(labels)

def print_test_set_distribution(test_dataset):
    """Prints the distribution of labels in the test set."""
    test_labels = [data.y.item() for data in test_dataset]
    unique, counts = np.unique(test_labels, return_counts=True)
    print("Test set distribution:")
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count:,} samples")

def print_evaluation_results(results):
    """Prints accuracy and confusion matrix for each evaluation method."""
    if 'standard' in results:
        print(f"\nStandard Two-Stage Results:")
        print(f"  Accuracy: {results['standard']['accuracy']:.4f}")
        print(f"  Confusion Matrix:\n{results['standard']['confusion_matrix']}")
    if 'fusion' in results:
        print(f"\nFusion Strategy Results:")
        print(f"  Accuracy: {results['fusion']['accuracy']:.4f}")
        print(f"  Confusion Matrix:\n{results['fusion']['confusion_matrix']}")

def final_evaluation(pipeline, test_dataset, val_loader, dataset_key):
    print(f"\n=== Final Evaluation ===")
    print_test_set_distribution(test_dataset)
    results = pipeline.evaluate(val_loader, method='both')
    print_evaluation_results(results)
    print(f"\n✓ Training completed successfully!")

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    """Main training and evaluation pipeline."""
    setup_gpu_optimization()
    config_dict = OmegaConf.to_container(config, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    # Dataset configuration
    dataset_key = config_dict['root_folder']
    if dataset_key not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    root_folder = DATASET_PATHS[dataset_key]

    # Create directories
    for dir_name in ["images", "output_model_optimized"]:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

    # === Data Loading and Preprocessing ===
    print(f"\n=== Data Loading and Preprocessing ===")
    id_mapping = build_id_mapping_from_normal(root_folder)
    start_time = time.time()
    dataset = graph_creation(root_folder, id_mapping=id_mapping, window_size=100)
    preprocessing_time = time.time() - start_time
    print(f"✓ Dataset created: {len(dataset)} graphs, {len(id_mapping)} CAN IDs")
    print(f"✓ Preprocessing time: {preprocessing_time:.2f}s")

    for data in dataset:
        assert not torch.isnan(data.x).any(), "Dataset contains NaN values"
        assert not torch.isinf(data.x).any(), "Dataset contains Inf values"

    DATASIZE = config_dict['datasize']
    TRAIN_RATIO = config_dict['train_ratio']
    BATCH_SIZE = config_dict['batch_size']
    EPOCHS = config_dict['epochs']

    feature_names = ["CAN ID", "data1", "data2", "data3", "data4", "data5", 
                     "data6", "data7", "data8", "count", "position"]
    plot_feature_histograms(dataset, feature_names=feature_names,
                           save_path=f"images/feature_histograms_{dataset_key}.png")

    # === Train/Test Split ===
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    print(f"✓ Data split: {len(train_dataset)} train, {len(test_dataset)} test")

    # Normal-only subset for autoencoder training
    normal_indices = [i for i, data in enumerate(train_dataset) if int(data.y.flatten()[0]) == 0]
    if DATASIZE < 1.0:
        subset_size = int(len(normal_indices) * DATASIZE)
        indices = np.random.choice(normal_indices, subset_size, replace=False)
    else:
        indices = normal_indices
    normal_subset = Subset(train_dataset, indices)

    # === Data Loaders ===
    train_loader = create_optimized_data_loaders(normal_subset, None, None, BATCH_SIZE, device)
    val_loader = create_optimized_data_loaders(None, test_dataset, None, BATCH_SIZE, device)
    full_train_loader = create_optimized_data_loaders(None, None, train_dataset, BATCH_SIZE, device)
    print(f"✓ Data loaders created: {len(normal_subset)} normal training samples")

    # === Model Initialization ===
    pipeline = GATPipeline(num_ids=len(id_mapping), embedding_dim=8, device=device)

    # === Stage 1: Autoencoder Training ===
    pipeline.train_stage1(train_loader, val_loader, epochs=EPOCHS)
    log_memory_usage("After autoencoder training")

    # === Visualizations ===
    print(f"\n=== Generating Visualizations ===")
    plot_graph_reconstruction(pipeline, full_train_loader, num_graphs=4,
                             save_path=f"images/graph_recon_examples_{dataset_key}.png")
    N = min(10000, len(train_dataset))
    indices = np.random.choice(len(train_dataset), size=N, replace=False)
    subsample = [train_dataset[i] for i in indices]
    subsample_loader = DataLoader(subsample, batch_size=BATCH_SIZE, shuffle=False)
    latent_vectors, labels = extract_latent_vectors(pipeline, subsample_loader)
    plot_latent_space(latent_vectors, labels, save_path=f"images/latent_space_{dataset_key}.png")
    plot_node_recon_errors(pipeline, full_train_loader, num_graphs=5,
                          save_path=f"images/node_recon_subplot_{dataset_key}.png")

    # === Stage 2: Classifier Training ===
    pipeline.train_stage2(full_train_loader, val_loader=val_loader, epochs=EPOCHS, key_suffix=dataset_key)

    # === Save Models ===
    pipeline.save_models("output_model_optimized", dataset_key, EPOCHS, 8, len(id_mapping))

    # === Final Evaluation ===
    final_evaluation(pipeline, test_dataset, val_loader, dataset_key)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")