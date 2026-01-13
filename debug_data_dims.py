#!/usr/bin/env python3
"""
Quick debug script to check actual data dimensions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.preprocessing import graph_creation, GraphDataset

# Load a small sample to check dimensions
dataset_path = "datasets/can-train-and-test-v1.5/hcrl-sa"
print(f"Loading from: {dataset_path}")

# Create just a few graphs to check dimensions
graphs, id_mapping = graph_creation(dataset_path, 'train_', return_id_mapping=True)
print(f"Number of graphs created: {len(graphs)}")
print(f"ID mapping size: {len(id_mapping)}")

if len(graphs) > 0:
    first_graph = graphs[0]
    print(f"First graph node features shape: {first_graph.x.shape}")
    print(f"First graph edge features shape: {first_graph.edge_attr.shape}")
    print(f"Sample node features (first 3 nodes):")
    print(first_graph.x[:3])
    
    # Check a few more graphs to ensure consistency
    for i in range(min(3, len(graphs))):
        print(f"Graph {i} x.shape: {graphs[i].x.shape}")