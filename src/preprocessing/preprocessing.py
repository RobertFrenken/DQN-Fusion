"""
CAN Bus Data Preprocessing for Graph Neural Network Analysis

This module handles the preprocessing of CAN bus network data, converting raw CSV files
into graph representations suitable for anomaly detection and intrusion detection tasks.
The preprocessing pipeline includes data normalization, temporal windowing, and graph
construction with rich node and edge features.

Key Features:
- Parallel processing for large datasets
- Robust hex-to-decimal conversion with error handling
- Temporal windowing with configurable stride
- Rich edge features including temporal and structural properties
- Categorical encoding for CAN IDs with out-of-vocabulary handling
"""

import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

# ==================== Configuration Constants ====================

DEFAULT_WINDOW_SIZE = 100
DEFAULT_STRIDE = 100
EXCLUDED_ATTACK_TYPES = ['suppress', 'masquerade']  # Attack types to exclude from processing
MAX_DATA_BYTES = 8  # CAN bus data field maximum bytes
HEX_CHARS = '0123456789abcdefABCDEF'

# Node feature configuration
NODE_FEATURE_COUNT = 11  # CAN_ID + 8 data bytes + count + position
EDGE_FEATURE_COUNT = 11  # Streamlined edge features

# Column indices for processed DataFrame (after conversion to numpy)
# Columns: [CAN_ID, Data1-8, Source, Target, label]
COL_CAN_ID = 0
COL_DATA_START = 1
COL_DATA_END = 9  # exclusive
COL_SOURCE = -3
COL_TARGET = -2
COL_LABEL = -1

# ==================== Core Data Processing Functions ====================

def safe_hex_to_int(value: Union[str, int, float]) -> Optional[int]:
    """
    Safely convert hex string or numeric value to integer.
    
    Args:
        value: Input value (hex string, int, or float)
        
    Returns:
        int or None: Converted integer value or None if conversion fails
    """
    if pd.isna(value):
        return None
    
    try:
        if isinstance(value, str):
            value = value.strip()
            if all(c in HEX_CHARS for c in value):
                return int(value, 16)
            else:
                return int(value)  # Try as decimal
        else:
            return int(value)
    except (ValueError, TypeError):
        return None

def apply_dynamic_id_mapping(df: pd.DataFrame, id_mapping: Dict, verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply ID mapping with dynamic expansion for new IDs encountered during processing.

    Args:
        df: DataFrame to process
        id_mapping: Existing ID mapping
        verbose: Whether to print information about new IDs

    Returns:
        Tuple of (processed DataFrame, updated ID mapping)
    """
    # Make a copy to avoid modifying the original mapping
    dynamic_mapping = id_mapping.copy()
    new_ids_found = []

    # First pass: collect all new IDs from all columns
    for col in ['CAN ID', 'Source', 'Target']:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            for val in unique_vals:
                if val not in dynamic_mapping:
                    new_id_index = len(dynamic_mapping) - 1  # Insert before OOV
                    dynamic_mapping[val] = new_id_index
                    dynamic_mapping['OOV'] = len(dynamic_mapping) - 1
                    new_ids_found.append(val)

    # Second pass: apply mapping using vectorized .map() instead of .apply()
    oov_index = dynamic_mapping['OOV']
    for col in ['CAN ID', 'Source', 'Target']:
        if col in df.columns:
            df[col] = df[col].map(dynamic_mapping).fillna(oov_index).astype(int)

    if new_ids_found and verbose:
        print(f"  Dynamically added {len(new_ids_found)} new CAN IDs: {new_ids_found[:5]}{'...' if len(new_ids_found) > 5 else ''}")

    return df, dynamic_mapping

def build_id_mapping_from_normal(root_folder: str, folder_type: str = 'train_') -> Dict[Union[int, str], int]:
    """
    Build CAN ID mapping using only normal (non-attack) data for cleaner feature space.

    This function reads full CSV files to filter by attack label. For faster preprocessing
    that includes all IDs (without filtering), use build_lightweight_id_mapping() instead.

    Args:
        root_folder: Path to root folder containing CSV files
        folder_type: Type of folder to process (e.g., 'train_', 'test_')

    Returns:
        Dictionary mapping CAN IDs to integer indices, with 'OOV' for out-of-vocabulary
    """
    csv_files = find_csv_files(root_folder, folder_type)
    all_ids = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']

            # Filter to normal traffic only
            normal_df = df[df['attack'] == 0]
            if len(normal_df) == 0:
                continue

            # Extract and convert CAN IDs
            can_ids = normal_df['arbitration_id'].dropna().unique()
            converted = [safe_hex_to_int(x) for x in can_ids]
            all_ids.extend([x for x in converted if x is not None])

        except Exception as e:
            logger.warning(f"Could not process {csv_file}: {e}")
            continue

    if not all_ids:
        return {'OOV': 0}

    # Create sorted mapping with OOV index
    unique_ids = sorted(set(all_ids))
    id_mapping = {can_id: idx for idx, can_id in enumerate(unique_ids)}
    id_mapping['OOV'] = len(id_mapping)

    return id_mapping

def find_csv_files(root_folder: str, folder_type: str = 'train_') -> List[str]:
    """
    Find all relevant CSV files in the directory structure.
    
    Args:
        root_folder: Root directory to search
        folder_type: Type of folder to include in search (e.g. 'train_')
        
    Returns:
        List of CSV file paths
    """
    csv_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Check if current directory contains the folder_type pattern
        dir_basename = os.path.basename(dirpath).lower()
        if folder_type.lower().rstrip('_') in dir_basename:
            for filename in filenames:
                if (filename.endswith('.csv') and 
                    not any(attack_type in filename.lower() for attack_type in EXCLUDED_ATTACK_TYPES)):
                    csv_files.append(os.path.join(dirpath, filename))
    
    return csv_files

# ==================== Dataset Creation Functions ====================

def pad_data_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pad data fields to ensure consistent 8-byte structure.
    
    Args:
        df: DataFrame with DLC and Data columns
        
    Returns:
        DataFrame with padded data fields
    """
    # Create mask for rows where DLC is less than 8
    mask = df['DLC'] < 8
    
    # Pad missing bytes with '00'
    for i in range(MAX_DATA_BYTES):
        column_name = f'Data{i+1}'
        df.loc[mask & (df['DLC'] <= i), column_name] = '00'
    
    # Fill any remaining NaN values
    df.fillna('00', inplace=True)
    
    return df

def dataset_creation_streaming(csv_path: str, id_mapping: Optional[Dict] = None, chunk_size: int = 10000) -> pd.DataFrame:
    """
    Process a CSV file in chunks to reduce memory usage.
    
    Args:
        csv_path: Path to CSV file
        id_mapping: Pre-built CAN ID mapping for categorical encoding
        chunk_size: Number of rows to process at once
        
    Returns:
        Processed DataFrame ready for graph construction
    """
    processed_chunks = []
    
    try:
        # Use the Python engine and explicit dtype to avoid C-parser segfaults on malformed files.
        # Reading with chunks reduces memory usage.
        reader = pd.read_csv(csv_path, chunksize=chunk_size, engine='python', dtype=str)
        for chunk in reader:
            # Ensure consistent columns even when dtype=str
            chunk.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
            chunk.rename(columns={'arbitration_id': 'CAN ID'}, inplace=True)

            # Process this chunk
            processed_chunk = _process_dataframe_chunk(chunk, id_mapping)
            processed_chunks.append(processed_chunk)

    except Exception as e:
        # As a last resort, try reading the entire file with engine='python' and no chunks
        try:
            full = pd.read_csv(csv_path, engine='python', dtype=str)
            full.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
            full.rename(columns={'arbitration_id': 'CAN ID'}, inplace=True)
            processed_full = _process_dataframe_chunk(full, id_mapping)
            processed_chunks.append(processed_full)
        except Exception as e2:
            logger.warning(f"Error processing {csv_path} (also failed full-read fallback): {e2}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    # Combine all chunks
    if processed_chunks:
        return pd.concat(processed_chunks, ignore_index=True)
    else:
        return pd.DataFrame()

def _process_dataframe_chunk(chunk: pd.DataFrame, id_mapping: Optional[Dict] = None) -> pd.DataFrame:
    """
    Process a single chunk of DataFrame.

    Args:
        chunk: DataFrame chunk to process
        id_mapping: Pre-built CAN ID mapping

    Returns:
        Processed DataFrame chunk
    """
    # Handle data field parsing - use vectorized string operations
    chunk['data_field'] = chunk['data_field'].astype(str).fillna('').str.strip()
    chunk['DLC'] = chunk['data_field'].str.len() // 2  # Vectorized length

    # Extract bytes using vectorized string slicing
    data_field = chunk['data_field'].values
    for i in range(MAX_DATA_BYTES):
        start = i * 2
        end = start + 2
        # Vectorized extraction with padding for short strings
        chunk[f'Data{i+1}'] = [s[start:end] if len(s) >= end else '00' for s in data_field]

    # Add graph structure columns
    chunk['Source'] = chunk['CAN ID']
    chunk['Target'] = chunk['CAN ID'].shift(-1)

    # Pad and clean data
    chunk = pad_data_field(chunk)

    # Convert hex values to decimal - use vectorized approach where possible
    hex_columns = ['CAN ID', 'Source', 'Target'] + [f'Data{i+1}' for i in range(MAX_DATA_BYTES)]
    for col in hex_columns:
        chunk[col] = chunk[col].apply(safe_hex_to_int)

    # Apply dynamic ID mapping (handles new IDs gracefully)
    if id_mapping is not None:
        chunk, _ = apply_dynamic_id_mapping(chunk, id_mapping, verbose=False)

    # Clean up and normalize - make explicit copy to avoid SettingWithCopyWarning
    chunk = chunk.iloc[:-1].copy()  # Remove last row (no target) and create copy
    chunk.loc[:, 'label'] = chunk['attack'].astype(int)

    # Normalize payload bytes to [0, 1] - vectorized
    byte_columns = [f'Data{i+1}' for i in range(MAX_DATA_BYTES)]
    for col in byte_columns:
        chunk[col] = chunk[col] / 255.0

    # Return essential columns
    return chunk[['CAN ID'] + byte_columns + ['Source', 'Target', 'label']]


# ==================== Graph Construction Functions ====================

def create_graphs_numpy(data: pd.DataFrame, window_size: int = DEFAULT_WINDOW_SIZE, 
                       stride: int = DEFAULT_STRIDE) -> List[Data]:
    """
    Transform DataFrame into PyTorch Geometric Data objects using sliding windows.
    
    Args:
        data: Processed DataFrame
        window_size: Size of sliding window
        stride: Stride for sliding window
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    data_array = data.to_numpy()
    num_windows = max(1, (len(data_array) - window_size) // stride + 1)
    start_indices = range(0, num_windows * stride, stride)
    
    return [create_graph_from_window(data_array[start:start + window_size]) 
            for start in start_indices]

def create_graph_from_window(window_data: np.ndarray) -> Data:
    """
    Transform a data window into a PyTorch Geometric Data object with rich features.

    Args:
        window_data: NumPy array of shape (window_size, num_features)
                    Columns: [CAN_ID, Data1-8, Source, Target, label]

    Returns:
        PyTorch Geometric Data object with node and edge features
    """
    # Extract components using column constants
    source = window_data[:, COL_SOURCE]
    target = window_data[:, COL_TARGET]
    labels = window_data[:, COL_LABEL]
    
    # Build edge information
    edges = np.column_stack((source, target))
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    
    # Create node mapping
    nodes = np.unique(np.concatenate((source, target)))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Convert to edge indices
    edge_index = np.array([[node_to_idx[src], node_to_idx[tgt]] 
                          for src, tgt in unique_edges]).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Create edge features
    edge_features = compute_edge_features(window_data, unique_edges, edge_counts)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Create node features
    node_features = compute_node_features(window_data, nodes, source)
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create graph label
    label_value = 1 if np.any(labels == 1) else 0
    y = torch.tensor(label_value, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def compute_edge_features(window_data: np.ndarray, unique_edges: np.ndarray,
                         edge_counts: np.ndarray) -> np.ndarray:
    """
    Compute comprehensive edge features for graph representation (optimized).

    Args:
        window_data: Full window data array
        unique_edges: Array of unique (source, target) pairs
        edge_counts: Count of each unique edge

    Returns:
        Array of edge features (num_edges, EDGE_FEATURE_COUNT)
    """
    source = window_data[:, COL_SOURCE]
    target = window_data[:, COL_TARGET]
    window_length = len(window_data)
    num_edges = len(unique_edges)

    # Pre-allocate output array (zeros handle default cases)
    edge_features = np.zeros((num_edges, EDGE_FEATURE_COUNT), dtype=np.float32)

    # Frequency features (vectorized)
    edge_features[:, 0] = edge_counts
    edge_features[:, 1] = edge_counts / window_length

    # Pre-compute node degrees (vectorized with bincount)
    all_nodes = np.concatenate([source, target])
    unique_nodes = np.unique(all_nodes)
    node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
    node_indices = np.array([node_to_idx[n] for n in all_nodes])
    degree_counts = np.bincount(node_indices, minlength=len(unique_nodes))
    node_degrees = {n: degree_counts[node_to_idx[n]] for n in unique_nodes}

    # Pre-compute reverse edge lookup
    edge_set = set(map(tuple, unique_edges))

    # Position array for temporal calculations
    positions = np.arange(window_length)

    for i, (src, tgt) in enumerate(unique_edges):
        # Edge mask - unavoidable O(W) per edge
        edge_mask = (source == src) & (target == tgt)
        edge_positions = positions[edge_mask]

        # Temporal analysis
        n_occurrences = len(edge_positions)
        if n_occurrences > 1:
            intervals = np.diff(edge_positions)
            avg_interval = intervals.mean()
            std_interval = intervals.std()
            edge_features[i, 2] = avg_interval
            edge_features[i, 3] = std_interval
            edge_features[i, 4] = 1.0 / (1.0 + std_interval) if std_interval > 0 else 1.0

        # Temporal position features
        if n_occurrences > 0:
            first_occ = edge_positions[0] / window_length
            last_occ = edge_positions[-1] / window_length
            edge_features[i, 5] = first_occ
            edge_features[i, 6] = last_occ
            edge_features[i, 7] = last_occ - first_occ

        # Network structure features (use pre-computed degrees)
        edge_features[i, 8] = float((tgt, src) in edge_set)
        src_deg = node_degrees.get(src, 0)
        tgt_deg = node_degrees.get(tgt, 0)
        edge_features[i, 9] = src_deg * tgt_deg
        edge_features[i, 10] = src_deg / max(tgt_deg, 1e-8)

    return edge_features

def compute_node_features(window_data: np.ndarray, nodes: np.ndarray,
                         source: np.ndarray) -> np.ndarray:
    """
    Compute node features including CAN ID, payload statistics, and temporal information.

    Args:
        window_data: Full window data array
        nodes: Array of unique node IDs
        source: Source column from window data

    Returns:
        Array of node features (num_nodes, NODE_FEATURE_COUNT)
    """
    num_nodes = len(nodes)
    window_length = len(source)
    positions = np.arange(window_length)

    # Pre-allocate output
    node_features = np.zeros((num_nodes, NODE_FEATURE_COUNT), dtype=np.float32)
    occurrence_counts = np.zeros(num_nodes, dtype=np.float32)

    for i, node in enumerate(nodes):
        # Find all occurrences of this node as source
        node_mask = source == node
        node_data = window_data[node_mask]

        if len(node_data) > 0:
            # CAN ID and payload features (columns 0-8)
            node_features[i, :COL_DATA_END] = node_data[:, :COL_DATA_END].mean(axis=0)

            # Occurrence count
            occurrence_counts[i] = len(node_data)

            # Temporal position (last occurrence)
            node_positions = positions[node_mask]
            node_features[i, -1] = node_positions[-1] / max(window_length - 1, 1)
        else:
            # Handle nodes that only appear as targets
            node_features[i, COL_CAN_ID] = node

    # Normalize occurrence counts to [0, 1]
    count_min, count_max = occurrence_counts.min(), occurrence_counts.max()
    if count_max > count_min:
        node_features[:, -2] = (occurrence_counts - count_min) / (count_max - count_min)
    else:
        node_features[:, -2] = occurrence_counts

    return node_features

# ==================== Main Interface Function ====================

def graph_creation(
    root_folder: str,
    folder_type: str = 'train_',
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    verbose: bool = False,
    return_id_mapping: bool = False,
    id_mapping: Optional[Dict] = None,
) -> Union[List[Data], Tuple[List[Data], Dict]]:
    """
    Create graphs from CAN bus data with streaming and reduced memory usage.

    Args:
        root_folder: Path to root folder containing CSV files
        folder_type: Type of folder to process (e.g., 'train_', 'test_')
        window_size: Size of sliding window for temporal graphs
        stride: Stride for sliding window (overlap control)
        verbose: Whether to print detailed processing information
        return_id_mapping: Whether to return the CAN ID mapping dictionary
        id_mapping: Pre-built CAN ID mapping (optional, will build if None)

    Returns:
        GraphDataset or tuple of (GraphDataset, id_mapping) if return_id_mapping=True
    """
    csv_files = find_csv_files(root_folder, folder_type)
    
    if not csv_files:
        if verbose:
            print(f"No CSV files found in {root_folder}")
        dataset = GraphDataset([])
        return (dataset, {'OOV': 0}) if return_id_mapping else dataset
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files to process")
    
    # Build ID mapping if not provided
    if id_mapping is None:
        # Use lightweight ID mapping (just scan for IDs, don't load full files)
        global_id_mapping = build_lightweight_id_mapping(csv_files, verbose=verbose)
    else:
        global_id_mapping = id_mapping
    
    # Process files with streaming to reduce memory usage
    all_graphs = []
    
    for i, csv_file in enumerate(csv_files):
        if verbose:
            print(f"Processing file {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        
        try:
            # Use streaming processing to reduce memory usage
            df = dataset_creation_streaming(csv_file, id_mapping=global_id_mapping, chunk_size=5000)
            
            if df.empty:
                continue
                
            if df.isnull().values.any():
                if verbose:
                    print(f"  Warning: NaN values found, filling with 0")
                df.fillna(0, inplace=True)
            
            # Create graphs from processed DataFrame
            graphs = create_graphs_numpy(df, window_size=window_size, stride=stride)
            all_graphs.extend(graphs)
            
            if verbose:
                print(f"  Created {len(graphs)} graphs ({len(all_graphs)} total so far)")
            
        except Exception as e:
            if verbose:
                print(f"  Error processing {csv_file}: {e}")
    
    if verbose:
        print(f"Total graphs created: {len(all_graphs)}")
    
    dataset = GraphDataset(all_graphs)
    
    if return_id_mapping:
        return dataset, global_id_mapping
    return dataset

def build_lightweight_id_mapping(csv_files: List[str], verbose: bool = False) -> Dict[Union[int, str], int]:
    """
    Build ID mapping with minimal memory usage by only scanning CAN ID column.
    
    Args:
        csv_files: List of CSV file paths
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping CAN IDs to integer indices
    """
    unique_ids = set()
    
    if verbose:
        print(f"Building lightweight ID mapping from {len(csv_files)} files...")
    
    for i, csv_file in enumerate(csv_files):
        if verbose and i % 10 == 0:
            print(f"  Scanning file {i+1}/{len(csv_files)} for CAN IDs...")
        
        try:
            # Only read the CAN ID column to minimize memory usage
            df_ids = pd.read_csv(csv_file, usecols=[1])  # arbitration_id is column 1
            df_ids.columns = ['arbitration_id']
            
            # Extract unique CAN IDs
            can_ids = df_ids['arbitration_id'].dropna().unique()
            file_ids = set()
            
            for can_id in can_ids:
                converted_id = safe_hex_to_int(can_id)
                if converted_id is not None:
                    file_ids.add(converted_id)
            
            unique_ids.update(file_ids)
            del df_ids  # Free memory immediately
            
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not scan {csv_file}: {e}")
            continue
    
    # Create mapping
    sorted_ids = sorted(list(unique_ids))
    id_mapping = {can_id: idx for idx, can_id in enumerate(sorted_ids)}
    id_mapping['OOV'] = len(id_mapping)
    
    if verbose:
        print(f"✅ Built ID mapping with {len(id_mapping)} entries")
    
    return id_mapping

class GraphDataset(Dataset):
    """
    PyTorch Geometric Dataset wrapper for CAN bus graph data.
    
    This class provides a convenient interface for handling collections of
    graph objects with PyTorch Geometric DataLoader compatibility.
    
    Args:
        data_list: List of PyTorch Geometric Data objects
        
    Examples:
        >>> graphs = [Data(x=..., edge_index=...), ...]
        >>> dataset = GraphDataset(graphs)
        >>> print(f"Dataset size: {len(dataset)}")
        >>> first_graph = dataset[0]
    """
    
    def __init__(self, data_list: List[Data]):
        """
        Initialize dataset with list of graph objects.
        
        Args:
            data_list: List of PyTorch Geometric Data objects
        """
        self.data_list = data_list
        
        # Validate data consistency
        if data_list:
            self._validate_data_consistency()

    def __len__(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        """
        Get a graph by index.
        
        Args:
            idx: Index of the graph to retrieve
            
        Returns:
            PyTorch Geometric Data object
        """
        return self.data_list[idx]
    
    def _validate_data_consistency(self) -> None:
        """Validate that all graphs have consistent feature dimensions."""
        if not self.data_list:
            return
        
        # Check node feature consistency
        first_graph = self.data_list[0]
        expected_node_features = first_graph.x.size(1) if first_graph.x is not None else 0
        expected_edge_features = first_graph.edge_attr.size(1) if first_graph.edge_attr is not None else 0
        
        for i, graph in enumerate(self.data_list):
            if graph.x is not None and graph.x.size(1) != expected_node_features:
                raise ValueError(f"Graph {i} has inconsistent node features: "
                               f"expected {expected_node_features}, got {graph.x.size(1)}")
            
            if graph.edge_attr is not None and graph.edge_attr.size(1) != expected_edge_features:
                raise ValueError(f"Graph {i} has inconsistent edge features: "
                               f"expected {expected_edge_features}, got {graph.edge_attr.size(1)}")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if not self.data_list:
            return {"num_graphs": 0}
        
        num_nodes = [g.num_nodes for g in self.data_list]
        num_edges = [g.num_edges for g in self.data_list]
        labels = [g.y.item() if g.y.dim() == 0 else g.y[0].item() for g in self.data_list]
        
        stats = {
            "num_graphs": len(self.data_list),
            "avg_nodes": np.mean(num_nodes),
            "std_nodes": np.std(num_nodes),
            "avg_edges": np.mean(num_edges),
            "std_edges": np.std(num_edges),
            "min_nodes": min(num_nodes),
            "max_nodes": max(num_nodes),
            "min_edges": min(num_edges),
            "max_edges": max(num_edges),
            "normal_graphs": sum(1 for label in labels if label == 0),
            "attack_graphs": sum(1 for label in labels if label == 1),
            "node_features": self.data_list[0].x.size(1) if self.data_list[0].x is not None else 0,
            "edge_features": self.data_list[0].edge_attr.size(1) if self.data_list[0].edge_attr is not None else 0
        }
        
        return stats
    
    def print_stats(self) -> None:
        """Print comprehensive dataset statistics."""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print("DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Total graphs: {stats['num_graphs']:,}")
        print(f"Normal graphs: {stats['normal_graphs']:,} ({stats['normal_graphs']/stats['num_graphs']*100:.1f}%)")
        print(f"Attack graphs: {stats['attack_graphs']:,} ({stats['attack_graphs']/stats['num_graphs']*100:.1f}%)")
        print(f"\nGraph Structure:")
        print(f"  Nodes per graph: {stats['avg_nodes']:.1f} ± {stats['std_nodes']:.1f} [{stats['min_nodes']}-{stats['max_nodes']}]")
        print(f"  Edges per graph: {stats['avg_edges']:.1f} ± {stats['std_edges']:.1f} [{stats['min_edges']}-{stats['max_edges']}]")
        print(f"\nFeature Dimensions:")
        print(f"  Node features: {stats['node_features']}")
        print(f"  Edge features: {stats['edge_features']}")

