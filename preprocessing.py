import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import os
import unittest
from torch_geometric.transforms import VirtualNode

def build_id_mapping(df):
    """Build a mapping from CAN IDs to indices for embedding, with OOV index.
    
    Args:
        df (pandas.DataFrame): DataFrame containing CAN ID columns.
        
    Returns:
        dict: Mapping from CAN IDs to integer indices, with 'OOV' key for out-of-vocabulary.
    """
    # Convert all IDs to integer (from hex string)
    def to_int(x):
        try:
            return int(x, 16) if isinstance(x, str) and all(c in '0123456789abcdefABCDEF' for c in x) else int(x)
        except Exception:
            return None
    unique_ids = pd.concat([df['CAN ID'], df['Source'], df['Target']]).dropna().unique()
    unique_ids = [to_int(x) for x in unique_ids if x is not None]
    id_mapping = {can_id: idx for idx, can_id in enumerate(unique_ids)}
    oov_index = len(id_mapping)
    id_mapping['OOV'] = oov_index
    return id_mapping

def build_id_mapping_from_normal(root_folder, folder_type='train_'):
    """Build CAN ID mapping using only normal (attack==0) rows from all CSVs.
    
    Args:
        root_folder (str): Path to the root folder containing CSV files.
        folder_type (str, optional): Type of folder to process. Defaults to 'train_'.
        
    Returns:
        dict: Mapping from CAN IDs to integer indices, with 'OOV' key for out-of-vocabulary.
    """
    train_csv_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if folder_type in dirpath.lower():
            for filename in filenames:
                if filename.endswith('.csv') and 'suppress' not in filename.lower() and 'masquerade' not in filename.lower():
                    train_csv_files.append(os.path.join(dirpath, filename))
    all_dfs = []
    for csv_file in train_csv_files:
        df = pd.read_csv(csv_file)
        df.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
        df.rename(columns={'arbitration_id': 'CAN ID'}, inplace=True)
        df['Source'] = df['CAN ID']
        df['Target'] = df['CAN ID'].shift(-1)
        # Only keep normal rows
        df = df[df['attack'] == 0]
        all_dfs.append(df)
    if all_dfs:
        # --- Use integer conversion for mapping ---
        def to_int(x):
            try:
                return int(x, 16) if isinstance(x, str) and all(c in '0123456789abcdefABCDEF' for c in x) else int(x)
            except Exception:
                return None
        concat_df = pd.concat(all_dfs, ignore_index=True)
        unique_ids = pd.concat([concat_df['CAN ID'], concat_df['Source'], concat_df['Target']]).dropna().unique()
        unique_ids = [to_int(x) for x in unique_ids if x is not None]
        id_mapping = {can_id: idx for idx, can_id in enumerate(unique_ids)}
        oov_index = len(id_mapping)
        id_mapping['OOV'] = oov_index
        return id_mapping
    else:
        return {'OOV': 0}

def graph_creation(root_folder, folder_type='train_', window_size=50, stride=50, verbose=False, return_id_mapping=False, id_mapping=None):
    """Create graphs from CAN bus data in the specified folder.
    
    Args:
        root_folder (str): Path to the root folder containing CSV files.
        folder_type (str, optional): Type of folder to process. Defaults to 'train_'.
        window_size (int, optional): Size of sliding window. Defaults to 50.
        stride (int, optional): Stride for sliding window. Defaults to 50.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        return_id_mapping (bool, optional): Whether to return ID mapping. Defaults to False.
        id_mapping (dict, optional): Pre-built ID mapping. Defaults to None.
        
    Returns:
        GraphDataset or tuple: Dataset of graphs, optionally with ID mapping if return_id_mapping=True.
    """
    train_csv_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if folder_type in dirpath.lower():
            for filename in filenames:
                if filename.endswith('.csv') and 'suppress' not in filename.lower() and 'masquerade' not in filename.lower():
                    train_csv_files.append(os.path.join(dirpath, filename))
    # Build global mapping ONLY if not provided
    all_dfs = []
    for csv_file in train_csv_files:
        df = pd.read_csv(csv_file)
        df.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
        df.rename(columns={'arbitration_id': 'CAN ID'}, inplace=True)
        # Add Source and Target columns here
        df['Source'] = df['CAN ID']
        df['Target'] = df['CAN ID'].shift(-1)
        all_dfs.append(df)
    if id_mapping is None:
        if all_dfs:
            global_id_mapping = build_id_mapping(pd.concat(all_dfs, ignore_index=True))
        else:
            global_id_mapping = {'OOV': 0}
    else:
        global_id_mapping = id_mapping
    combined_list = []
    
    for csv_file in train_csv_files:
        if verbose:
            print(f"Processing file: {csv_file}")
        df = dataset_creation_vectorized(csv_file, id_mapping=global_id_mapping)
        if df.isnull().values.any():
            if verbose:
                print(f"NaN values found in DataFrame from file: {csv_file}")
                print(df[df.isnull().any(axis=1)])
            df.fillna(0, inplace=True)
        graphs = create_graphs_numpy(df, window_size=window_size, stride=stride)
        combined_list.extend(graphs)
    dataset = GraphDataset(combined_list)
    if return_id_mapping:
        return dataset, global_id_mapping
    return dataset

def create_graphs_numpy(data, window_size, stride):
    """Transform a pandas dataframe into a list of PyTorch Geometric Data objects.
    
    Args:
        data (pandas.DataFrame): DataFrame representing a window of data.
                                 Assumes column structure: [Source, Target, Data1, Data2, ..., DataN, label]
        window_size (int): The size of the sliding window.
        stride (int): The stride for the sliding window.
    
    Returns:
        list: A list of PyTorch Geometric Data objects.
    """
    # Calculate the number of windows
    data = data.to_numpy()  # Convert DataFrame to NumPy array if necessary
    num_windows = (len(data) - window_size) // stride + 1
    start_indices = range(0, num_windows * stride, stride)

    # list comprehension to create graphs for each window
    return [window_data_transform_numpy(data[start:start + window_size]) 
            for start in start_indices]



def window_data_transform_numpy(data):
    """Transform a NumPy array window into a PyTorch Geometric Data object with rich edge features.
    
    Args:
        data (numpy.ndarray): A NumPy array representing a window of data.
                              Assumes column structure: [Source, Target, Data1, Data2, ..., DataN, label]
    
    Returns:
        torch_geometric.data.Data: A PyTorch Geometric Data object with enhanced edge features.
    """
    # Extract Source, Target, and label columns
    source = data[:, 0]
    target = data[:, -2]
    labels = data[:, -1]

    # Calculate edge counts for each unique (Source, Target) pair
    unique_edges, edge_counts = np.unique(np.stack((source, target), axis=1), axis=0, return_counts=True)

    # Create node index mapping
    nodes = np.unique(np.concatenate((source, target)))
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Convert edges to indices
    edge_index = np.vectorize(node_to_idx.get)(unique_edges).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # CREATE RICH EDGE FEATURES
    edge_features = []
    
    for i, (src, tgt) in enumerate(unique_edges):
        # Get all occurrences of this edge
        edge_mask = (source == src) & (target == tgt)
        edge_data = data[edge_mask]
        
        # 1. FREQUENCY FEATURES
        frequency = edge_counts[i]
        relative_frequency = frequency / len(data)
        
        # 2. TEMPORAL FEATURES
        edge_positions = np.where(edge_mask)[0]
        if len(edge_positions) > 1:
            # Time intervals between consecutive occurrences
            intervals = np.diff(edge_positions)
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            regularity = 1.0 / (1.0 + std_interval) if std_interval > 0 else 1.0
        else:
            avg_interval = 0.0
            std_interval = 0.0
            regularity = 0.0
        
        # First and last occurrence (normalized)
        first_occurrence = edge_positions[0] / len(data) if len(edge_positions) > 0 else 0.0
        last_occurrence = edge_positions[-1] / len(data) if len(edge_positions) > 0 else 0.0
        temporal_spread = last_occurrence - first_occurrence
        
        # 3. PAYLOAD FEATURES
        payload_data = edge_data[:, 1:9]  # Data1 to Data8
        if len(payload_data) > 0:
            # Statistical measures of payload
            payload_mean = np.mean(payload_data, axis=0)
            payload_std = np.std(payload_data, axis=0)
            payload_entropy = -np.sum(payload_mean * np.log(payload_mean + 1e-8))  # Simple entropy
            
            # Payload change patterns
            if len(payload_data) > 1:
                payload_changes = np.sum(np.diff(payload_data, axis=0) != 0, axis=1)
                avg_payload_change = np.mean(payload_changes)
                payload_volatility = np.std(payload_changes)
            else:
                avg_payload_change = 0.0
                payload_volatility = 0.0
        else:
            payload_mean = np.zeros(8)
            payload_std = np.zeros(8)
            payload_entropy = 0.0
            avg_payload_change = 0.0
            payload_volatility = 0.0
        
        # 4. COMMUNICATION PATTERN FEATURES
        # Bidirectional communication check
        reverse_edge_exists = np.any((source == tgt) & (target == src))
        
        # Node degree influence (how connected are source and target)
        src_degree = np.sum((source == src) | (target == src))
        tgt_degree = np.sum((source == tgt) | (target == tgt))
        degree_product = src_degree * tgt_degree
        degree_ratio = src_degree / (tgt_degree + 1e-8)
        
        # 5. ATTACK CORRELATION FEATURES
        edge_attack_ratio = np.mean(edge_data[:, -1]) if len(edge_data) > 0 else 0.0
        
        # Combine all features into a single vector
        edge_feature_vector = np.concatenate([
            [frequency, relative_frequency],                    # Frequency features (2)
            [avg_interval, std_interval, regularity],           # Temporal regularity (3)
            [first_occurrence, last_occurrence, temporal_spread], # Temporal position (3)
            payload_mean[:4],                                   # First 4 payload means (4)
            payload_std[:4],                                    # First 4 payload stds (4)
            [payload_entropy, avg_payload_change, payload_volatility], # Payload dynamics (3)
            [float(reverse_edge_exists), degree_product, degree_ratio], # Network structure (3)
            [edge_attack_ratio]                                 # Attack correlation (1)
        ])
        edge_features.append(edge_feature_vector)
    
    # Convert to tensor
    edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)

    # ...existing node feature computation...
    node_features = np.zeros((len(nodes), 11))  # 10 features + 1 for position
    node_counts = np.zeros(len(nodes))
    node_positions = np.zeros(len(nodes))

    for i, node in enumerate(nodes):
        mask = (source == node)
        node_data = data[mask, 0:9]
        if len(node_data) > 0:
            node_data_norm = node_data.copy()
            node_features[i, :-2] = node_data_norm.mean(axis=0)
        node_counts[i] = mask.sum()
        indices = np.where(source == node)[0]
        if len(indices) > 0:
            node_positions[i] = indices[-1] / (len(source) - 1) if len(source) > 1 else 0.0

    # Normalize occurrence count
    occ = node_counts
    if occ.max() > occ.min():
        node_features[:, -2] = (occ - occ.min()) / (occ.max() - occ.min())
    else:
        node_features[:, -2] = 0.0

    node_features[:, -1] = node_positions

    x = torch.tensor(node_features, dtype=torch.float)
    label_value = 1 if (labels == 1).any() else 0
    y = torch.tensor(label_value, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def pad_row_vectorized(df):
    """
    Pads rows in a Pandas DataFrame where the data length code (DLC) is less than 8
    by filling missing columns with a hex code value of '00'.

    Args:
        df (pandas.DataFrame): A pandas DataFrame.

    Returns:
        df: A pandas DataFrame with missing columns padded to '00'.
    """
    # Create a mask for rows where DLC is less than 8
    mask = df['DLC'] < 8

    # Iterate over the range of Data1 to Data8 columns
    for i in range(8):
        # Only pad columns where the index is greater than or equal to the DLC
        column_name = f'Data{i+1}'
        df.loc[mask & (df['DLC'] <= i), column_name] = '00'

    # Fill any remaining NaN values with '00'
    df.fillna('00', inplace=True)

    return df

def dataset_creation_vectorized(path, id_mapping=None):
    """
    Takes a csv file containing CAN data. Creates a pandas dataframe,
    adds source and target columns, pads the rows with missing values,
    transforms the hex values to decimal values, and reencodes the labels
    to a binary classifcation problem of 0 for attack free and 1 for attack.
    
    Args:
        path (string): A string containing the path to a CAN data csv file.
    
    Returns:
        df: a pandas dataframe.
    """
    df = pd.read_csv(path)
    df.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
    df.rename(columns={'arbitration_id': 'CAN ID'}, inplace=True)

    # Ensure 'data_field' is a string and handle missing values
    df['data_field'] = df['data_field'].astype(str).fillna('')

    # Add the DLC column based on the length of the data_field
    df['DLC'] = df['data_field'].apply(lambda x: len(x) // 2)

    # Unpack the data_field column into individual bytes
    df['data_field'] = df['data_field'].astype(str).str.strip()  # Ensure it's a string and strip whitespace
    df['bytes'] = df['data_field'].apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)])  # Split into bytes

    # Determine the maximum number of bytes (should be 8 or fewer)
    max_bytes = 8
    # Create Data1 to Data8 columns, padding with '00' if fewer than 8 bytes
    for i in range(max_bytes):
        df[f'Data{i+1}'] = df['bytes'].apply(lambda x: x[i] if i < len(x) else '00')

    df['Source'] = df['CAN ID']
    df['Target'] = df['CAN ID'].shift(-1)

    # Pad rows and fill missing values
    df = pad_row_vectorized(df)

    # Convert hex columns to decimal
    hex_columns = ['CAN ID', 'Data1', 'Data2', 'Data3', 'Data4', 
                   'Data5', 'Data6', 'Data7', 'Data8', 'Source', 'Target']
    # Convert hex values to decimal
    for col in hex_columns:
        df[col] = df[col].apply(lambda x: int(x, 16) if pd.notnull(x) and isinstance(x, str) and all(c in '0123456789abcdefABCDEF' for c in x) else None)

    # CHANGE: Apply ID mapping (categorical encoding)
    if id_mapping is not None:
        oov_index = id_mapping['OOV']
        for col in ['CAN ID', 'Source', 'Target']:
            df[col] = df[col].apply(lambda x: id_mapping.get(x, oov_index))
        # print("Mapped CAN IDs in DataFrame:", df['CAN ID'].unique())

    # Drop the last row and reencode labels
    df = df.iloc[:-1]
    df['label'] = df['attack'].astype(int)

    # CHANGE: Normalize payload columns to [0, 1]
    for col in [f'Data{i+1}' for i in range(8)]:
        df[col] = df[col] / 255.0

    return df[['CAN ID', 'Data1', 'Data2', 'Data3', 'Data4', 
               'Data5', 'Data6', 'Data7', 'Data8', 'Source', 'Target', 'label']]

class GraphDataset(Dataset):
    """
    Takes a list of pytorch geometric Data objects and creates a dataset.
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

#################################
# Testing Class                 #
#################################
class TestPreprocessing(unittest.TestCase):
    # NOTE: Add test about the ID embedding mapping
    def test_graph_creation(self):
        print("Testing graph creation...")
        root_folder = r"datasets/can-train-and-test-v1.5/hcrl-sa"
        graph_dataset = graph_creation(root_folder)
        self.assertGreater(len(graph_dataset), 0)

        # Check for NaN and Inf values in the graph dataset
        for data in graph_dataset:
            self.assertFalse(torch.isnan(data.x).any(), "Graph dataset contains NaN values!")
            self.assertFalse(torch.isinf(data.x).any(), "Graph dataset contains Inf values!")
            # Check normalization for payload columns in node features (columns 1-8)
            payload = data.x[:, 1:9]
            id = data.x[:, 0]  # Assuming first column is ID
            count = data.x[:, -1]  # Assuming last column is count
            # self.assertTrue(((id >= 0).all() and (id <= 1).all()), "ID features not normalized!")
            self.assertTrue(((payload >= 0).all() and (payload <= 1).all()), "Payload features not normalized!")
            self.assertTrue(((count >= 0).all() and (count <= 1).all()), "Count node features not normalized!")
            
            # Check that edge attributes exist and have correct dimensions
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                self.assertEqual(data.edge_attr.size(0), data.edge_index.size(1), "Edge attributes size mismatch!")
                self.assertEqual(data.edge_attr.size(1), 23, "Edge attributes should have 23 features!")
                self.assertFalse(torch.isnan(data.edge_attr).any(), "Edge attributes contain NaN values!")
                self.assertFalse(torch.isinf(data.edge_attr).any(), "Edge attributes contain Inf values!")
        
        print("Graph creation test passed.")

if __name__ == "__main__":
    unittest.main()