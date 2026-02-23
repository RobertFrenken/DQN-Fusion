"""Domain-agnostic graph construction engine.

Receives an IR DataFrame (conforming to ``IRSchema``) and produces
PyTorch Geometric ``Data`` objects via sliding-window graph construction.

The engine knows nothing about CAN buses, network flows, or any other
domain — it only operates on the standardized column layout.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from config.constants import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STRIDE,
    EDGE_FEATURE_COUNT,
    NODE_FEATURE_COUNT,
)
from .schema import IRSchema

log = logging.getLogger(__name__)


class GraphEngine:
    """Converts IR DataFrames into PyG graph objects.

    Parameters
    ----------
    schema : IRSchema
        Describes the column layout of incoming DataFrames.
    window_size : int
        Number of rows per sliding window.
    stride : int
        Step size between consecutive windows.
    """

    def __init__(
        self,
        schema: IRSchema,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
    ):
        self.schema = schema
        self.window_size = window_size
        self.stride = stride

    def create_graphs(self, ir_df) -> list[Data]:
        """Transform an IR DataFrame into a list of PyG Data objects.

        Parameters
        ----------
        ir_df : pd.DataFrame
            DataFrame conforming to ``self.schema``.

        Returns
        -------
        list[Data]
            One graph per sliding window.
        """
        data_array = ir_df.to_numpy()
        n = len(data_array)
        ws, st = self.window_size, self.stride
        num_windows = max(1, (n - ws) // st + 1)

        graphs = []
        for w in range(num_windows):
            start = w * st
            window = data_array[start : start + ws]
            graphs.append(self._window_to_graph(window))

        return graphs

    # ------------------------------------------------------------------
    # Internal: per-window graph construction
    # ------------------------------------------------------------------

    def _window_to_graph(self, window: np.ndarray) -> Data:
        """Build a single PyG Data from a numpy window slice."""
        s = self.schema

        source = window[:, s.col_source]
        target = window[:, s.col_target]
        labels = window[:, s.col_label]

        # Unique edges and counts
        edges = np.column_stack((source, target))
        unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)

        # Node mapping (dense re-indexing)
        nodes = np.unique(np.concatenate((source, target)))
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        edge_index = np.array(
            [[node_to_idx[src], node_to_idx[tgt]] for src, tgt in unique_edges]
        ).T
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Features
        edge_features = self._compute_edge_features(
            window, source, target, unique_edges, edge_counts, nodes, node_to_idx,
        )
        node_features = self._compute_node_features(
            window, nodes, source,
        )

        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float)
        label_value = 1 if np.any(labels == 1) else 0
        y = torch.tensor(label_value, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def _compute_edge_features(
        self,
        window: np.ndarray,
        source: np.ndarray,
        target: np.ndarray,
        unique_edges: np.ndarray,
        edge_counts: np.ndarray,
        nodes: np.ndarray,
        node_to_idx: dict,
    ) -> np.ndarray:
        """Compute 11-D edge features.

        Feature layout (matches legacy ``EDGE_FEATURE_COUNT=11``):
            [0]  raw count
            [1]  frequency (count / window_length)
            [2]  mean interval between occurrences
            [3]  std interval
            [4]  regularity  1/(1+std)
            [5]  first occurrence position (normalized)
            [6]  last occurrence position (normalized)
            [7]  temporal span (last - first)
            [8]  bidirectionality flag
            [9]  degree product (src_deg * tgt_deg)
            [10] degree ratio (src_deg / tgt_deg)
        """
        window_length = len(window)
        num_edges = len(unique_edges)
        features = np.zeros((num_edges, EDGE_FEATURE_COUNT), dtype=np.float32)

        # Frequency features (vectorized)
        features[:, 0] = edge_counts
        features[:, 1] = edge_counts / window_length

        # Pre-compute node degrees
        all_nodes = np.concatenate([source, target])
        uniq_nodes = np.unique(all_nodes)
        n2i = {n: i for i, n in enumerate(uniq_nodes)}
        node_indices = np.array([n2i[n] for n in all_nodes])
        degree_counts = np.bincount(node_indices, minlength=len(uniq_nodes))
        node_degrees = {n: degree_counts[n2i[n]] for n in uniq_nodes}

        # Reverse edge set for bidirectionality
        edge_set = set(map(tuple, unique_edges))

        positions = np.arange(window_length)

        for i, (src, tgt) in enumerate(unique_edges):
            edge_mask = (source == src) & (target == tgt)
            edge_positions = positions[edge_mask]

            n_occ = len(edge_positions)
            if n_occ > 1:
                intervals = np.diff(edge_positions)
                avg_interval = intervals.mean()
                std_interval = intervals.std()
                features[i, 2] = avg_interval
                features[i, 3] = std_interval
                features[i, 4] = 1.0 / (1.0 + std_interval) if std_interval > 0 else 1.0

            if n_occ > 0:
                first_occ = edge_positions[0] / window_length
                last_occ = edge_positions[-1] / window_length
                features[i, 5] = first_occ
                features[i, 6] = last_occ
                features[i, 7] = last_occ - first_occ

            features[i, 8] = float((tgt, src) in edge_set)
            src_deg = node_degrees.get(src, 0)
            tgt_deg = node_degrees.get(tgt, 0)
            features[i, 9] = src_deg * tgt_deg
            features[i, 10] = src_deg / max(tgt_deg, 1e-8)

        return features

    def _compute_node_features(
        self,
        window: np.ndarray,
        nodes: np.ndarray,
        source: np.ndarray,
    ) -> np.ndarray:
        """Compute 11-D node features.

        Feature layout (matches legacy ``NODE_FEATURE_COUNT=11``):
            [0]        entity_id mean (CAN ID)
            [1:n+1]    mean of continuous features (payload bytes)
            [n+1]      normalized occurrence count
            [n+2]      last temporal position (normalized)

        Where n = schema.num_features.
        """
        s = self.schema
        num_nodes = len(nodes)
        window_length = len(source)
        positions = np.arange(window_length)

        node_features = np.zeros((num_nodes, NODE_FEATURE_COUNT), dtype=np.float32)
        occurrence_counts = np.zeros(num_nodes, dtype=np.float32)

        # entity_id is column 0, features are columns 1..num_features
        feat_end = 1 + s.num_features  # exclusive

        for i, node in enumerate(nodes):
            node_mask = source == node
            node_data = window[node_mask]

            if len(node_data) > 0:
                # entity_id + continuous features
                node_features[i, :feat_end] = node_data[:, :feat_end].mean(axis=0)
                occurrence_counts[i] = len(node_data)
                node_positions = positions[node_mask]
                node_features[i, -1] = node_positions[-1] / max(window_length - 1, 1)
            else:
                # Nodes that only appear as targets
                node_features[i, 0] = node

        # Normalize occurrence counts to [0, 1]
        c_min, c_max = occurrence_counts.min(), occurrence_counts.max()
        if c_max > c_min:
            node_features[:, -2] = (occurrence_counts - c_min) / (c_max - c_min)
        else:
            node_features[:, -2] = occurrence_counts

        return node_features
