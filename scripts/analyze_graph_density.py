#!/usr/bin/env python3
"""
Analyze graph density and structure from cached datasets.
"""
import torch
from pathlib import Path
import numpy as np

def analyze_dataset(cache_path, dataset_name):
    """Analyze a cached dataset's graph statistics."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {dataset_name}")
    print(f"{'='*80}")

    try:
        # Load cached graphs
        data = torch.load(cache_path, map_location='cpu', weights_only=False)

        if isinstance(data, list):
            graphs = data
        elif isinstance(data, dict):
            # Try different possible keys
            if 'graphs' in data:
                graphs = data['graphs']
            elif 'train' in data:
                graphs = data['train']
            else:
                graphs = list(data.values())[0]
        elif hasattr(data, '__getitem__') and hasattr(data, '__len__'):
            # It's a Dataset-like object
            print(f"Dataset format: {type(data)}, length: {len(data)}")
            graphs = [data[i] for i in range(min(len(data), 1000))]  # Sample first 1000
        else:
            print(f"Unknown data format: {type(data)}")
            return

        print(f"Total graphs: {len(graphs)}")

        # Collect statistics
        node_counts = []
        edge_counts = []
        feature_dims = []

        for i, graph in enumerate(graphs):
            if i >= 1000:  # Sample first 1000 for speed
                break

            num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.shape[0]
            num_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
            feature_dim = graph.x.shape[1] if graph.x is not None else 0

            node_counts.append(num_nodes)
            edge_counts.append(num_edges)
            feature_dims.append(feature_dim)

        node_counts = np.array(node_counts)
        edge_counts = np.array(edge_counts)

        # Calculate densities
        max_edges = node_counts * (node_counts - 1)  # Directed graph
        densities = np.divide(edge_counts, max_edges, where=max_edges>0, out=np.zeros_like(edge_counts, dtype=float))

        # Memory estimates
        bytes_per_node = feature_dims[0] * 4 if len(feature_dims) > 0 else 0  # float32
        bytes_per_edge = 2 * 8  # 2 int64 values

        avg_memory_per_graph = (np.mean(node_counts) * bytes_per_node +
                                np.mean(edge_counts) * bytes_per_edge)

        print(f"\nNode Statistics:")
        print(f"  Mean nodes/graph: {np.mean(node_counts):.2f}")
        print(f"  Median nodes/graph: {np.median(node_counts):.2f}")
        print(f"  Min nodes: {np.min(node_counts)}")
        print(f"  Max nodes: {np.max(node_counts)}")
        print(f"  Std dev: {np.std(node_counts):.2f}")

        print(f"\nEdge Statistics:")
        print(f"  Mean edges/graph: {np.mean(edge_counts):.2f}")
        print(f"  Median edges/graph: {np.median(edge_counts):.2f}")
        print(f"  Min edges: {np.min(edge_counts)}")
        print(f"  Max edges: {np.max(edge_counts)}")
        print(f"  Std dev: {np.std(edge_counts):.2f}")

        print(f"\nDensity:")
        print(f"  Mean density: {np.mean(densities):.4f}")
        print(f"  Median density: {np.median(densities):.4f}")

        print(f"\nMemory Estimates:")
        print(f"  Feature dimensions: {feature_dims[0] if len(feature_dims) > 0 else 'N/A'}")
        print(f"  Avg bytes/node (features): {bytes_per_node:.2f}")
        print(f"  Bytes/edge (edge_index): {bytes_per_edge}")
        print(f"  Avg memory/graph: {avg_memory_per_graph/1024:.2f} KB")
        print(f"  Total for {len(graphs)} graphs: {len(graphs) * avg_memory_per_graph / (1024**2):.2f} MB")
        print(f"  Doubled (curriculum loading): {2 * len(graphs) * avg_memory_per_graph / (1024**2):.2f} MB")

        return {
            'dataset': dataset_name,
            'num_graphs': len(graphs),
            'mean_nodes': np.mean(node_counts),
            'mean_edges': np.mean(edge_counts),
            'mean_density': np.mean(densities),
            'avg_memory_kb': avg_memory_per_graph / 1024
        }

    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    base_dir = Path('/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/automotive')

    datasets = [
        ('hcrl_sa', base_dir / 'hcrl_sa' / 'cache' / 'processed_graphs.pt'),
        ('hcrl_ch', base_dir / 'hcrl_ch' / 'cache' / 'processed_graphs.pt'),
        ('set_01', base_dir / 'set_01' / 'cache' / 'processed_graphs.pt'),
        ('set_02', base_dir / 'set_02' / 'cache' / 'processed_graphs.pt'),
        ('set_03', base_dir / 'set_03' / 'cache' / 'processed_graphs.pt'),
        ('set_04', base_dir / 'set_04' / 'cache' / 'processed_graphs.pt'),
    ]

    results = []
    for name, path in datasets:
        if path.exists():
            result = analyze_dataset(path, name)
            if result:
                results.append(result)
        else:
            print(f"⚠️  Cache not found: {path}")

    # Summary table
    print(f"\n{'='*120}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*120}")
    print(f"{'Dataset':<10} {'Graphs':<10} {'Avg Nodes':<12} {'Avg Edges':<12} {'Density':<10} {'Mem/Graph':<12} {'Status':<10}")
    print(f"{'='*120}")

    for r in results:
        status = 'OOM' if r['dataset'] == 'set_03' else 'SUCCESS'
        print(f"{r['dataset']:<10} {r['num_graphs']:<10} {r['mean_nodes']:<12.2f} {r['mean_edges']:<12.2f} "
              f"{r['mean_density']:<10.4f} {r['avg_memory_kb']:<12.2f} {status:<10}")

    print(f"{'='*120}")

if __name__ == '__main__':
    main()
