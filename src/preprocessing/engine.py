"""Domain-agnostic graph construction engine.

Receives an IR DataFrame (conforming to ``IRSchema``) and produces
PyTorch Geometric ``Data`` objects via sliding-window graph construction.

The engine knows nothing about CAN buses, network flows, or any other
domain — it only operates on the standardized column layout.

Phase 3.3: Edge and node feature computation is fully vectorized using
``np.unique(return_inverse=True)`` + ``np.add.at`` scatter operations,
replacing the O(E×W) and O(N×W) Python loops.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch_geometric.data import Data

from config.constants import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STRIDE,
    EDGE_FEATURE_COUNT,
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
        self.node_feature_count = schema.num_features + 3  # entity_id + features + count + position

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

        # Unique edges, counts, and inverse mapping for scatter ops
        edges = np.column_stack((source, target))
        unique_edges, edge_inverse, edge_counts = np.unique(
            edges,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        # Node mapping (dense re-indexing)
        nodes = np.unique(np.concatenate((source, target)))
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        edge_index = np.array([[node_to_idx[src], node_to_idx[tgt]] for src, tgt in unique_edges]).T
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Features (vectorized)
        edge_features = self._compute_edge_features_vectorized(
            window,
            source,
            target,
            unique_edges,
            edge_counts,
            edge_inverse,
            nodes,
        )
        node_features = self._compute_node_features_vectorized(
            window,
            nodes,
            source,
        )

        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float)
        label_value = 1 if np.any(labels == 1) else 0
        y = torch.tensor(label_value, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # Per-node binary labels (attack=1, normal=0) via scatter-max
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        num_nodes = len(nodes)
        node_labels = np.zeros(num_nodes, dtype=np.int64)
        for row_idx in range(len(source)):
            nidx = node_to_idx[source[row_idx]]
            if labels[row_idx] == 1:
                node_labels[nidx] = 1
        data.node_y = torch.tensor(node_labels, dtype=torch.long)

        # Attack type metadata (graph-level and per-node)
        col_at = s.col_attack_type
        if col_at is not None:
            attack_types = window[:, col_at].astype(np.int64)
            # Graph-level: dominant non-normal attack type, or 0 if all normal
            attack_counts = np.bincount(attack_types)
            if len(attack_counts) > 1:
                # Zero out normal (code 0) to find dominant attack type
                attack_counts_no_normal = attack_counts.copy()
                attack_counts_no_normal[0] = 0
                if attack_counts_no_normal.sum() > 0:
                    data.attack_type = torch.tensor(
                        int(np.argmax(attack_counts_no_normal)), dtype=torch.long
                    )
                else:
                    data.attack_type = torch.tensor(0, dtype=torch.long)
            else:
                data.attack_type = torch.tensor(0, dtype=torch.long)

            # Per-node attack type via scatter (take max code per node)
            node_attack_type = np.zeros(num_nodes, dtype=np.int64)
            for row_idx in range(len(source)):
                nidx = node_to_idx[source[row_idx]]
                if attack_types[row_idx] > node_attack_type[nidx]:
                    node_attack_type[nidx] = attack_types[row_idx]
            data.node_attack_type = torch.tensor(node_attack_type, dtype=torch.long)

        return data

    def _compute_edge_features_vectorized(
        self,
        window: np.ndarray,
        source: np.ndarray,
        target: np.ndarray,
        unique_edges: np.ndarray,
        edge_counts: np.ndarray,
        edge_inverse: np.ndarray,
        nodes: np.ndarray,
    ) -> np.ndarray:
        """Compute 11-D edge features using vectorized operations.

        Replaces the O(E×W) Python loop with scatter-based aggregation.

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
        W = len(window)
        E = len(unique_edges)
        features = np.zeros((E, EDGE_FEATURE_COUNT), dtype=np.float32)

        # [0-1] Frequency features (already vectorized)
        features[:, 0] = edge_counts
        features[:, 1] = edge_counts / W

        # --- Temporal features via scatter ---
        # edge_inverse[row] = which unique edge this row belongs to
        positions = np.arange(W, dtype=np.float64)

        # First and last occurrence per edge group
        first_pos = np.full(E, W, dtype=np.float64)  # init to max
        last_pos = np.full(E, -1.0, dtype=np.float64)  # init to min
        np.minimum.at(first_pos, edge_inverse, positions)
        np.maximum.at(last_pos, edge_inverse, positions)

        # [5-7] Temporal position features (normalized)
        valid_mask = edge_counts > 0
        features[valid_mask, 5] = first_pos[valid_mask] / W
        features[valid_mask, 6] = last_pos[valid_mask] / W
        features[valid_mask, 7] = (last_pos[valid_mask] - first_pos[valid_mask]) / W

        # [2-4] Interval statistics: mean, std, regularity
        # For edges with count > 1, compute intervals between sorted positions
        multi_mask = edge_counts > 1
        if np.any(multi_mask):
            # Sort positions by edge group for interval computation
            sort_idx = np.lexsort((positions, edge_inverse))
            sorted_groups = edge_inverse[sort_idx]
            sorted_positions = positions[sort_idx]

            # Compute intervals: diff between consecutive same-group positions
            # A pair (i, i+1) belongs to the same group when sorted_groups match
            same_group = sorted_groups[:-1] == sorted_groups[1:]
            intervals = np.diff(sorted_positions)

            # Only keep intervals within the same edge group
            valid_intervals = intervals[same_group]
            valid_groups = sorted_groups[:-1][same_group]

            if len(valid_intervals) > 0:
                # Sum and sum-of-squares per edge group for mean/std
                interval_sum = np.zeros(E, dtype=np.float64)
                interval_sq_sum = np.zeros(E, dtype=np.float64)
                interval_count = np.zeros(E, dtype=np.float64)

                np.add.at(interval_sum, valid_groups, valid_intervals)
                np.add.at(interval_sq_sum, valid_groups, valid_intervals**2)
                np.add.at(interval_count, valid_groups, 1.0)

                has_intervals = interval_count > 0
                mean_interval = np.zeros(E, dtype=np.float64)
                std_interval = np.zeros(E, dtype=np.float64)

                mean_interval[has_intervals] = (
                    interval_sum[has_intervals] / interval_count[has_intervals]
                )
                # Variance = E[X^2] - E[X]^2, then sqrt for std
                variance = np.zeros(E, dtype=np.float64)
                variance[has_intervals] = (
                    interval_sq_sum[has_intervals] / interval_count[has_intervals]
                    - mean_interval[has_intervals] ** 2
                )
                # Clamp negative variance from float precision
                variance = np.maximum(variance, 0.0)
                std_interval[has_intervals] = np.sqrt(variance[has_intervals])

                features[has_intervals, 2] = mean_interval[has_intervals]
                features[has_intervals, 3] = std_interval[has_intervals]
                # Regularity: 1/(1+std), with std=0 → regularity=1
                reg = np.ones(E, dtype=np.float64)
                nonzero_std = std_interval > 0
                reg[nonzero_std] = 1.0 / (1.0 + std_interval[nonzero_std])
                features[multi_mask, 4] = reg[multi_mask]

        # [8] Bidirectionality: check if reverse edge exists
        # Build set of (src, tgt) tuples for reverse lookup
        edge_set = set(map(tuple, unique_edges))
        features[:, 8] = np.array(
            [float((tgt, src) in edge_set) for src, tgt in unique_edges],
            dtype=np.float32,
        )

        # [9-10] Degree features
        all_nodes_arr = np.concatenate([source, target])
        _, node_inv = np.unique(all_nodes_arr, return_inverse=True)
        degree = np.bincount(node_inv)
        # Map unique_edges src/tgt to node indices for degree lookup
        all_uniq = np.unique(all_nodes_arr)
        n2i = {n: i for i, n in enumerate(all_uniq)}
        src_deg = np.array([degree[n2i[s]] for s, _ in unique_edges], dtype=np.float32)
        tgt_deg = np.array([degree[n2i[t]] for _, t in unique_edges], dtype=np.float32)
        features[:, 9] = src_deg * tgt_deg
        features[:, 10] = src_deg / np.maximum(tgt_deg, 1e-8)

        return features

    def _compute_node_features_vectorized(
        self,
        window: np.ndarray,
        nodes: np.ndarray,
        source: np.ndarray,
    ) -> np.ndarray:
        """Compute node features using vectorized operations.

        Replaces the O(N×W) Python loop with scatter-based aggregation.

        Feature layout (``num_features + 3`` columns total):
            [0]        entity_id mean (CAN ID or IP)
            [1:n+1]    mean of continuous features (payload bytes / flow stats)
            [n+1]      normalized occurrence count
            [n+2]      last temporal position (normalized)

        Where n = schema.num_features.
        """
        s = self.schema
        N = len(nodes)
        W = len(source)
        feat_end = 1 + s.num_features  # columns 0..feat_end (exclusive)
        node_feat_count = s.num_features + 3  # entity_id + features + count + position

        node_features = np.zeros((N, node_feat_count), dtype=np.float32)

        # Map source values to node indices
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        source_node_idx = np.array([node_to_idx[s] for s in source], dtype=np.int64)

        # --- Scatter-sum for feature means ---
        # Sum entity_id + continuous features per node
        feature_sums = np.zeros((N, feat_end), dtype=np.float64)
        occurrence_counts = np.zeros(N, dtype=np.float64)

        # Scatter: add each row's features to the corresponding node
        row_features = window[:, :feat_end].astype(np.float64)
        for col in range(feat_end):
            np.add.at(feature_sums[:, col], source_node_idx, row_features[:, col])
        np.add.at(occurrence_counts, source_node_idx, 1.0)

        # Compute means where count > 0
        has_data = occurrence_counts > 0
        for col in range(feat_end):
            node_features[has_data, col] = feature_sums[has_data, col] / occurrence_counts[has_data]

        # For target-only nodes (no source occurrences), set entity_id directly
        target_only = ~has_data
        if np.any(target_only):
            node_features[target_only, 0] = nodes[target_only]

        # --- Last temporal position per node (vectorized) ---
        positions = np.arange(W, dtype=np.float64)
        last_pos = np.full(N, -1.0, dtype=np.float64)
        np.maximum.at(last_pos, source_node_idx, positions)
        has_pos = last_pos >= 0
        node_features[has_pos, -1] = last_pos[has_pos] / max(W - 1, 1)

        # --- Normalized occurrence counts ---
        c_min, c_max = occurrence_counts.min(), occurrence_counts.max()
        if c_max > c_min:
            node_features[:, -2] = ((occurrence_counts - c_min) / (c_max - c_min)).astype(
                np.float32
            )
        else:
            node_features[:, -2] = occurrence_counts.astype(np.float32)

        return node_features
