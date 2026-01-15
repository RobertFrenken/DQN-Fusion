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

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import os
import unittest
from typing import List, Dict, Tuple, Optional, Union
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# ==================== Configuration Constants ====================

DEFAULT_WINDOW_SIZE = 100
DEFAULT_STRIDE = 100
EXCLUDED_ATTACK_TYPES = ['suppress', 'masquerade']  # Attack types to exclude from processing
MAX_DATA_BYTES = 8  # CAN bus data field maximum bytes
HEX_CHARS = '0123456789abcdefABCDEF'

# Node feature configuration
NODE_FEATURE_COUNT = 11  # CAN_ID + 8 data bytes + count + position
EDGE_FEATURE_COUNT = 11  # Streamlined edge features

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

def normalize_payload_bytes(df: pd.DataFrame, byte_columns: List[str]) -> pd.DataFrame:
    """
    Normalize payload byte columns to [0, 1] range.
    
    Args:
        df: DataFrame containing payload data
        byte_columns: List of column names containing byte data
        
    Returns:
        DataFrame with normalized payload columns
    """
    df_copy = df.copy()
    for col in byte_columns:
        df_copy[col] = df_copy[col] / 255.0
    return df_copy

def build_complete_id_mapping_streaming(csv_files: List[str], verbose: bool = False) -> Dict[Union[int, str], int]:
    """
    Build COMPLETE CAN ID mapping using streaming approach with minimal memory.
    Gets 100% of all CAN IDs by reading each file once and extracting only the IDs.
    
    Args:
        csv_files: List of CSV file paths
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping ALL CAN IDs to integer indices
    """
    unique_ids = set()
    
    if verbose:
        print(f"Building COMPLETE ID mapping by streaming through {len(csv_files)} files...")
    
    for i, csv_file in enumerate(csv_files):
        if verbose:
            print(f"  📂 Processing {os.path.basename(csv_file)} ({i+1}/{len(csv_files)})")
        
        try:
            # Read the ENTIRE file (but we'll only extract IDs)
            df = pd.read_csv(csv_file)
            df.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
            
            # Extract ALL unique CAN IDs from this file
            can_ids = df['arbitration_id'].dropna().unique()
            file_ids = set()
            
            for can_id in can_ids:
                converted_id = safe_hex_to_int(can_id)
                if converted_id is not None:
                    file_ids.add(converted_id)
            
            # Add to global set
            unique_ids.update(file_ids)
            
            # IMMEDIATELY discard the DataFrame to free memory
            del df
            
            if verbose:
                print(f"     Found {len(file_ids)} unique IDs in this file")
                print(f"     Total unique IDs so far: {len(unique_ids)}")
                    
        except Exception as e:
            if verbose:
                print(f"     Warning: Could not process {csv_file}: {e}")
            continue
    
    if verbose:
        print(f"✅ Found {len(unique_ids)} TOTAL unique CAN IDs (100% coverage)")
    
    # Create mapping from ALL unique IDs
    sorted_ids = sorted(list(unique_ids))
    id_mapping = {can_id: idx for idx, can_id in enumerate(sorted_ids)}
    
    # Add out-of-vocabulary index (for truly corrupted data)
    oov_index = len(id_mapping)
    id_mapping['OOV'] = oov_index
    
    if verbose:
        print(f"📋 Final mapping size: {len(id_mapping)} entries (including OOV)")
    
    return id_mapping

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
    oov_index = dynamic_mapping['OOV']
    new_ids_found = []
    
    # Check all CAN ID columns for new IDs
    for col in ['CAN ID', 'Source', 'Target']:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            
            for val in unique_vals:
                if val not in dynamic_mapping:
                    # Found a new ID not in our mapping
                    new_id_index = len(dynamic_mapping) - 1  # Insert before OOV
                    dynamic_mapping[val] = new_id_index
                    dynamic_mapping['OOV'] = len(dynamic_mapping) - 1  # Update OOV index
                    new_ids_found.append(val)
            
            # Apply the mapping (now including new IDs)
            df[col] = df[col].apply(lambda x: dynamic_mapping.get(x, dynamic_mapping['OOV']))
    
    if new_ids_found and verbose:
        print(f"  Dynamically added {len(new_ids_found)} new CAN IDs: {new_ids_found[:5]}{'...' if len(new_ids_found) > 5 else ''}")
    
    return df, dynamic_mapping

def build_id_mapping(df: pd.DataFrame) -> Dict[Union[int, str], int]:
    """
    Build a mapping from CAN IDs to indices for categorical encoding.
    
    Args:
        df: DataFrame containing CAN ID columns
        
    Returns:
        Dictionary mapping CAN IDs to integer indices, with 'OOV' for out-of-vocabulary
    """
    # Extract all unique CAN IDs from relevant columns
    id_columns = ['CAN ID', 'Source', 'Target']
    all_ids = []
    
    for col in id_columns:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            converted_ids = [safe_hex_to_int(x) for x in unique_vals]
            all_ids.extend([x for x in converted_ids if x is not None])
    
    # Create mapping from unique IDs
    unique_ids = sorted(list(set(all_ids)))
    id_mapping = {can_id: idx for idx, can_id in enumerate(unique_ids)}
    
    # Add out-of-vocabulary index
    oov_index = len(id_mapping)
    id_mapping['OOV'] = oov_index
    
    return id_mapping

def build_id_mapping_from_normal(root_folder: str, folder_type: str = 'train_') -> Dict[Union[int, str], int]:
    """
    Build CAN ID mapping using only normal (non-attack) data for cleaner feature space.
    
    Args:
        root_folder: Path to root folder containing CSV files
        folder_type: Type of folder to process (e.g., 'train_', 'test_')
        
    Returns:
        Dictionary mapping CAN IDs to integer indices
    """
    csv_files = find_csv_files(root_folder, folder_type)
    normal_dfs = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
            df.rename(columns={'arbitration_id': 'CAN ID'}, inplace=True)
            
            # Add Source and Target columns for graph construction
            df['Source'] = df['CAN ID']
            df['Target'] = df['CAN ID'].shift(-1)
            
            # Filter to normal traffic only
            normal_df = df[df['attack'] == 0]
            if len(normal_df) > 0:
                normal_dfs.append(normal_df)
                
        except Exception as e:
            print(f"Warning: Could not process {csv_file}: {e}")
            continue
    
    if normal_dfs:
        combined_df = pd.concat(normal_dfs, ignore_index=True)
        return build_id_mapping(combined_df)
    else:
        return {'OOV': 0}

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
        # Read CSV in chunks to reduce memory usage
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            chunk.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
            chunk.rename(columns={'arbitration_id': 'CAN ID'}, inplace=True)
            
            # Process this chunk
            processed_chunk = _process_dataframe_chunk(chunk, id_mapping)
            processed_chunks.append(processed_chunk)
    
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
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
    # Handle data field parsing
    chunk['data_field'] = chunk['data_field'].astype(str).fillna('')
    chunk['DLC'] = chunk['data_field'].apply(lambda x: len(x) // 2)
    
    # Split data field into individual bytes
    chunk['data_field'] = chunk['data_field'].str.strip()
    chunk['bytes'] = chunk['data_field'].apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)])
    
    # Create Data1-Data8 columns
    for i in range(MAX_DATA_BYTES):
        chunk[f'Data{i+1}'] = chunk['bytes'].apply(lambda x: x[i] if i < len(x) else '00')
    
    # Add graph structure columns
    chunk['Source'] = chunk['CAN ID']
    chunk['Target'] = chunk['CAN ID'].shift(-1)
    
    # Pad and clean data
    chunk = pad_data_field(chunk)
    
    # Convert hex values to decimal
    hex_columns = ['CAN ID', 'Source', 'Target'] + [f'Data{i+1}' for i in range(MAX_DATA_BYTES)]
    for col in hex_columns:
        chunk[col] = chunk[col].apply(safe_hex_to_int)
    
    # Apply dynamic ID mapping (handles new IDs gracefully)
    if id_mapping is not None:
        chunk, _ = apply_dynamic_id_mapping(chunk, id_mapping, verbose=False)
    
    # Clean up and normalize
    chunk = chunk.iloc[:-1]  # Remove last row (no target)
    chunk['label'] = chunk['attack'].astype(int)
    
    # Normalize payload bytes to [0, 1]
    byte_columns = [f'Data{i+1}' for i in range(MAX_DATA_BYTES)]
    chunk = normalize_payload_bytes(chunk, byte_columns)
    
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
    # Extract components
    source = window_data[:, -3]  # Source column
    target = window_data[:, -2]  # Target column
    labels = window_data[:, -1]  # Label column
    
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
    Compute comprehensive edge features for graph representation.
    
    Args:
        window_data: Full window data array
        unique_edges: Array of unique (source, target) pairs
        edge_counts: Count of each unique edge
        
    Returns:
        Array of edge features (num_edges, EDGE_FEATURE_COUNT)
    """
    source = window_data[:, -3]
    target = window_data[:, -2]
    window_length = len(window_data)
    
    edge_features = []
    
    for i, (src, tgt) in enumerate(unique_edges):
        # Basic frequency features
        frequency = edge_counts[i]
        relative_frequency = frequency / window_length
        
        # Temporal analysis
        edge_mask = (source == src) & (target == tgt)
        edge_positions = np.where(edge_mask)[0]
        
        if len(edge_positions) > 1:
            intervals = np.diff(edge_positions)
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            regularity = 1.0 / (1.0 + std_interval) if std_interval > 0 else 1.0
        else:
            avg_interval = 0.0
            std_interval = 0.0
            regularity = 0.0
        
        # Temporal position features
        first_occurrence = edge_positions[0] / window_length if len(edge_positions) > 0 else 0.0
        last_occurrence = edge_positions[-1] / window_length if len(edge_positions) > 0 else 0.0
        temporal_spread = last_occurrence - first_occurrence
        
        # Network structure features
        reverse_edge_exists = float(np.any((source == tgt) & (target == src)))
        src_degree = np.sum((source == src) | (target == src))
        tgt_degree = np.sum((source == tgt) | (target == tgt))
        degree_product = src_degree * tgt_degree
        degree_ratio = src_degree / max(tgt_degree, 1e-8)
        
        # Payload variance
        edge_data = window_data[edge_mask]
        payload_variance = np.var(edge_data[:, 1:9]) if len(edge_data) > 1 else 0.0
        
        # Combine features (11 features total)
        feature_vector = np.array([
            frequency, relative_frequency,                    # Frequency (2)
            avg_interval, std_interval, regularity,           # Temporal regularity (3)
            first_occurrence, last_occurrence, temporal_spread, # Temporal position (3)
            reverse_edge_exists, degree_product, degree_ratio   # Network structure (3)
        ])
        
        edge_features.append(feature_vector)
    
    return np.array(edge_features)

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
    node_features = np.zeros((len(nodes), NODE_FEATURE_COUNT))
    
    for i, node in enumerate(nodes):
        # Find all occurrences of this node as source
        node_mask = (source == node)
        node_data = window_data[node_mask]
        
        if len(node_data) > 0:
            # CAN ID and payload features (columns 0-8)
            node_features[i, :9] = node_data[:, :9].mean(axis=0)
            
            # Occurrence count (normalized)
            occurrence_count = len(node_data)
            
            # Temporal position (last occurrence)
            occurrence_indices = np.where(node_mask)[0]
            last_position = occurrence_indices[-1] / (len(source) - 1) if len(source) > 1 else 0.0
        else:
            # Handle nodes that only appear as targets
            node_features[i, 0] = node  # CAN ID
            occurrence_count = 0
            last_position = 0.0
        
        node_features[i, -2] = occurrence_count  # Count feature
        node_features[i, -1] = last_position     # Position feature
    
    # Normalize occurrence counts
    counts = node_features[:, -2]
    if counts.max() > counts.min():
        node_features[:, -2] = (counts - counts.min()) / (counts.max() - counts.min())
    
    return node_features

# ==================== Optimized Processing Functions ====================







# ==================== Main Interface Function ====================

def graph_creation(root_folder: str, folder_type: str = 'train_',
                  window_size: int = DEFAULT_WINDOW_SIZE, stride: int = DEFAULT_STRIDE,
                  verbose: bool = False, return_id_mapping: bool = False,
                  id_mapping: Optional[Dict] = None, parallel: bool = False) -> Union[List[Data], Tuple[List[Data], Dict]]:
    """
    Create graphs from CAN bus data with optimized sequential processing.
    
    Lightning DataLoader handles parallelism, so we use single-threaded preprocessing
    to avoid conflicts and reduce memory usage.
    
    Args:
        root_folder: Path to root folder containing CSV files
        folder_type: Type of folder to process (e.g., 'train_', 'test_')
        window_size: Size of sliding window for temporal graphs
        stride: Stride for sliding window (overlap control)
        verbose: Whether to print detailed processing information
        return_id_mapping: Whether to return the CAN ID mapping dictionary
        id_mapping: Pre-built CAN ID mapping (optional, will build if None)
        parallel: Ignored - always uses sequential processing
        
    Returns:
        GraphDataset or tuple of (GraphDataset, id_mapping) if return_id_mapping=True
    """
    # Always use optimized sequential processing
    return graph_creation_optimized(
        root_folder, folder_type, window_size, stride,
        verbose, return_id_mapping, id_mapping
    )

def graph_creation_optimized(root_folder: str, folder_type: str = 'train_',
                           window_size: int = DEFAULT_WINDOW_SIZE, stride: int = DEFAULT_STRIDE,
                           verbose: bool = False, return_id_mapping: bool = False,
                           id_mapping: Optional[Dict] = None) -> Union[List[Data], Tuple[List[Data], Dict]]:
    """
    Optimized graph creation with streaming and reduced memory usage.
    
    Args:
        root_folder: Path to root folder containing CSV files
        folder_type: Type of folder to process
        window_size: Size of sliding window
        stride: Stride for sliding window
        verbose: Whether to print verbose output
        return_id_mapping: Whether to return ID mapping
        id_mapping: Pre-built ID mapping (optional)
        
    Returns:
        GraphDataset or tuple of (GraphDataset, id_mapping)
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

# ==================== Testing Framework ====================

class TestPreprocessing(unittest.TestCase):
    """
    Comprehensive test suite for preprocessing functionality.
    
    Tests cover data loading, graph creation, feature validation,
    and edge case handling to ensure robust preprocessing.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_root = r"datasets/can-train-and-test-v1.5/hcrl-sa"
        cls.small_window_size = 10  # Smaller for faster testing
        cls.test_stride = 5
    
    def test_id_mapping_creation(self):
        """Test CAN ID mapping creation and consistency."""
        print("Testing CAN ID mapping creation...")
        
        # Test normal data mapping
        id_mapping = build_id_mapping_from_normal(self.test_root)
        
        self.assertIsInstance(id_mapping, dict)
        self.assertIn('OOV', id_mapping)
        self.assertGreater(len(id_mapping), 1)  # Should have more than just OOV
        
        # Check that all values are integers
        for key, value in id_mapping.items():
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)
        
        print(f"✓ ID mapping created with {len(id_mapping)} entries")
    
    def test_hex_conversion(self):
        """Test hex-to-decimal conversion robustness."""
        print("Testing hex conversion...")
        
        # Test valid hex strings
        self.assertEqual(safe_hex_to_int("1A"), 26)
        self.assertEqual(safe_hex_to_int("FF"), 255)
        self.assertEqual(safe_hex_to_int("0"), 0)
        
        # Test invalid inputs
        self.assertIsNone(safe_hex_to_int("XYZ"))
        self.assertIsNone(safe_hex_to_int(""))
        self.assertIsNone(safe_hex_to_int(None))
        
        # Test numeric inputs
        self.assertEqual(safe_hex_to_int(123), 123)
        self.assertEqual(safe_hex_to_int(0), 0)
        
        print("✓ Hex conversion tests passed")
    
    def test_single_file_processing(self):
        """Test processing of a single CSV file."""
        print("Testing single file processing...")
        
        csv_files = find_csv_files(self.test_root, 'train_')
        if not csv_files:
            self.skipTest("No CSV files found for testing")
        
        # Test with first available file
        test_file = csv_files[0]
        id_mapping = build_id_mapping_from_normal(self.test_root)
        
        df = dataset_creation_streaming(test_file, id_mapping=id_mapping)
        
        # Validate DataFrame structure
        expected_columns = ['CAN ID'] + [f'Data{i+1}' for i in range(8)] + ['Source', 'Target', 'label']
        self.assertEqual(list(df.columns), expected_columns)
        
        # Check normalization
        for col in [f'Data{i+1}' for i in range(8)]:
            self.assertTrue(df[col].between(0, 1).all(), f"{col} not properly normalized")
        
        # Check for missing values
        self.assertFalse(df.isnull().any().any(), "DataFrame contains NaN values")
        
        print(f"✓ Single file processing: {len(df)} rows processed")
    
    def test_graph_creation_basic(self):
        """Test basic graph creation functionality."""
        print("Testing basic graph creation...")
        
        dataset = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride
        )
        
        self.assertIsInstance(dataset, GraphDataset)
        self.assertGreater(len(dataset), 0)
        
        print(f"✓ Created {len(dataset)} graphs")
    
    def test_graph_structure_validation(self):
        """Test graph structure and feature validation."""
        print("Testing graph structure validation...")
        
        dataset = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride
        )
        
        for i, graph in enumerate(dataset):
            # Basic structure validation
            self.assertIsInstance(graph, Data)
            self.assertIsNotNone(graph.x)
            self.assertIsNotNone(graph.edge_index)
            self.assertIsNotNone(graph.y)
            
            # Feature dimension validation
            self.assertEqual(graph.x.size(1), NODE_FEATURE_COUNT, 
                           f"Graph {i}: incorrect node feature count")
            
            if graph.edge_attr is not None:
                self.assertEqual(graph.edge_attr.size(1), EDGE_FEATURE_COUNT,
                               f"Graph {i}: incorrect edge feature count")
            
            # Data type validation
            self.assertEqual(graph.x.dtype, torch.float, f"Graph {i}: incorrect node feature dtype")
            self.assertEqual(graph.edge_index.dtype, torch.long, f"Graph {i}: incorrect edge index dtype")
            
            # Value range validation
            self.assertFalse(torch.isnan(graph.x).any(), f"Graph {i}: NaN in node features")
            self.assertFalse(torch.isinf(graph.x).any(), f"Graph {i}: Inf in node features")
            
            # Payload normalization check (columns 1-8)
            payload = graph.x[:, 1:9]
            self.assertTrue(torch.all(payload >= 0) and torch.all(payload <= 1),
                          f"Graph {i}: payload features not normalized to [0,1]")
            
            # Edge attribute validation
            if graph.edge_attr is not None:
                self.assertFalse(torch.isnan(graph.edge_attr).any(), 
                               f"Graph {i}: NaN in edge features")
                self.assertFalse(torch.isinf(graph.edge_attr).any(),
                               f"Graph {i}: Inf in edge features")
            
            # Test only first 10 graphs for speed
            if i >= 9:
                break
        
        print("✓ Graph structure validation passed")
    
    def test_optimized_processing(self):
        """Test optimized processing with streaming."""
        print("Testing optimized processing...")
        
        # Test optimized processing
        dataset = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride
        )
        
        self.assertIsInstance(dataset, GraphDataset)
        self.assertGreater(len(dataset), 0)
        
        print(f"✓ Optimized processing created {len(dataset)} graphs")
    
    def test_dataset_statistics(self):
        """Test dataset statistics computation."""
        print("Testing dataset statistics...")
        
        dataset = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride
        )
        
        stats = dataset.get_stats()
        
        # Validate statistics structure
        required_keys = ['num_graphs', 'avg_nodes', 'avg_edges', 'normal_graphs', 'attack_graphs']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Validate statistics values
        self.assertEqual(stats['num_graphs'], len(dataset))
        self.assertGreaterEqual(stats['normal_graphs'], 0)
        self.assertGreaterEqual(stats['attack_graphs'], 0)
        self.assertEqual(stats['normal_graphs'] + stats['attack_graphs'], stats['num_graphs'])
        
        print("✓ Dataset statistics validation passed")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("Testing edge cases...")
        
        # Test with non-existent directory
        empty_dataset = graph_creation("/non/existent/path")
        self.assertEqual(len(empty_dataset), 0)
        
        # Test with empty ID mapping
        empty_mapping = {'OOV': 0}
        dataset = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride,
            id_mapping=empty_mapping
        )
        self.assertIsInstance(dataset, GraphDataset)
        
        print("✓ Edge case testing passed")
    
    def test_full_pipeline_integration(self):
        """Test complete preprocessing pipeline integration."""
        print("Testing full pipeline integration...")
        
        # Test with ID mapping return
        dataset, id_mapping = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride,
            return_id_mapping=True
        )
        
        self.assertIsInstance(dataset, GraphDataset)
        self.assertIsInstance(id_mapping, dict)
        self.assertIn('OOV', id_mapping)
        
        # Test reusing ID mapping
        dataset2 = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride,
            id_mapping=id_mapping
        )
        
        self.assertIsInstance(dataset2, GraphDataset)
        
        # Print final statistics
        dataset.print_stats()
        
        print("✓ Full pipeline integration test passed")

def run_comprehensive_tests():
    """Run all preprocessing tests with detailed output."""
    print(f"\n{'='*80}")
    print("CAN-GRAPH PREPROCESSING COMPREHENSIVE TEST SUITE")
    print(f"{'='*80}")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessing)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run comprehensive test suite if called directly
    success = run_comprehensive_tests()
    exit(0 if success else 1)