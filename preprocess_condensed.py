import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import os
import unittest

HEX_COLUMNS = ['CAN ID', 'Data1', 'Data2', 'Data3', 'Data4', 
               'Data5', 'Data6', 'Data7', 'Data8', 'Source', 'Target']

def hex_to_decimal(val):
    if pd.notnull(val) and isinstance(val, str) and all(c in '0123456789abcdefABCDEF' for c in val):
        return int(val, 16)
    return None

def normalize_payload(df):
    for col in [f'Data{i+1}' for i in range(8)]:
        df[col] = df[col] / 255.0
    return df

def map_ids(df, id_mapping):
    for col in ['CAN ID', 'Source', 'Target']:
        df[col] = df[col].map(id_mapping)
    return df

def pad_row_vectorized(df):
    mask = df['DLC'] < 8
    for i in range(8):
        col = f'Data{i+1}'
        df.loc[mask & (df['DLC'] <= i), col] = '00'
    df.fillna('00', inplace=True)
    return df

def build_id_mapping(df):
    unique_ids = pd.concat([df['CAN ID'], df['Source'], df['Target']]).unique()
    return {can_id: idx for idx, can_id in enumerate(unique_ids)}

def process_dataframe(path, id_mapping=None):
    df = pd.read_csv(path)
    df.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
    df.rename(columns={'arbitration_id': 'CAN ID'}, inplace=True)
    df['data_field'] = df['data_field'].astype(str).fillna('')
    df['DLC'] = df['data_field'].apply(lambda x: len(x) // 2)
    df['bytes'] = df['data_field'].str.strip().apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)])
    for i in range(8):
        df[f'Data{i+1}'] = df['bytes'].apply(lambda x: x[i] if i < len(x) else '00')
    df['Source'] = df['CAN ID']
    df['Target'] = df['CAN ID'].shift(-1)
    df = pad_row_vectorized(df)
    for col in HEX_COLUMNS:
        df[col] = df[col].apply(hex_to_decimal)
    if id_mapping is not None:
        df = map_ids(df, id_mapping)
    df = df.iloc[:-1]
    df['label'] = df['attack'].astype(int)
    df = normalize_payload(df)
    return df[['CAN ID', 'Data1', 'Data2', 'Data3', 'Data4', 
               'Data5', 'Data6', 'Data7', 'Data8', 'Source', 'Target', 'label']]

def graph_creation(root_folder, folder_type='train_', window_size=50, stride=50, verbose=False):
    train_csv_files = [os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(root_folder)
        if folder_type in dirpath.lower()
        for filename in filenames
        if filename.endswith('.csv') and 'suppress' not in filename.lower() and 'masquerade' not in filename.lower()]
    all_dfs = [pd.read_csv(f) for f in train_csv_files]
    if all_dfs:
        for df in all_dfs:
            df.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
            df.rename(columns={'arbitration_id': 'CAN ID'}, inplace=True)
        global_id_mapping = build_id_mapping(pd.concat(all_dfs, ignore_index=True))
    else:
        global_id_mapping = {}
    combined_list = []
    for csv_file in train_csv_files:
        if verbose:
            print(f"Processing file: {csv_file}")
        df = process_dataframe(csv_file, id_mapping=global_id_mapping)
        if df.isnull().values.any():
            if verbose:
                print(f"NaN values found in DataFrame from file: {csv_file}")
                print(df[df.isnull().any(axis=1)])
            df.fillna(0, inplace=True)
        graphs = create_graphs_numpy(df, window_size=window_size, stride=stride)
        combined_list.extend(graphs)
    return GraphDataset(combined_list)

def create_graphs_numpy(data, window_size, stride):
    data = data.to_numpy()
    num_windows = (len(data) - window_size) // stride + 1
    start_indices = range(0, num_windows * stride, stride)
    return [window_data_transform_numpy(data[start:start + window_size]) 
            for start in start_indices]

def window_data_transform_numpy(data):
    source, target, labels = data[:, 0], data[:, -2], data[:, -1]
    unique_edges, edge_counts = np.unique(np.stack((source, target), axis=1), axis=0, return_counts=True)
    nodes = np.unique(np.concatenate((source, target)))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edge_index = np.vectorize(node_to_idx.get)(unique_edges).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_counts, dtype=torch.float).view(-1, 1)
    node_features = np.zeros((len(nodes), 10))
    node_counts = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        mask = (source == node)
        node_data = data[mask, 0:9]
        if len(node_data) > 0:
            node_data_norm = node_data.copy()
            node_data_norm[:, 1:9] = node_data_norm[:, 1:9] / 255.0
            node_features[i, :-1] = node_data_norm.mean(axis=0)
        node_counts[i] = mask.sum()
    occ = node_counts
    node_features[:, -1] = (occ - occ.min()) / (occ.max() - occ.min()) if occ.max() > occ.min() else 0.0
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([1 if 1 in labels else 0], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

# Optional: Keep your tests as is for validation
class TestPreprocessing(unittest.TestCase):
    def test_dataset_creation_vectorized(self):
        test_path = r"datasets/can-train-and-test-v1.5/hcrl-ch/train_02_with_attacks/fuzzing-train.csv"
        df = process_dataframe(test_path)
        self.assertEqual(len(df.columns), 12)
        self.assertTrue('label' in df.columns)
        self.assertFalse(df.isnull().values.any(), "Dataset contains NaN values!")
    def test_graph_creation(self):
        root_folder = r"datasets/can-train-and-test-v1.5/set_02"
        graph_dataset = graph_creation(root_folder)
        self.assertGreater(len(graph_dataset), 0)
        for data in graph_dataset:
            self.assertFalse(torch.isnan(data.x).any(), "Graph dataset contains NaN values!")
            self.assertFalse(torch.isinf(data.x).any(), "Graph dataset contains Inf values!")

if __name__ == "__main__":
    unittest.main()