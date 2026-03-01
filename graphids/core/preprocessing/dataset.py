"""Graph dataset wrapper for PyTorch Geometric Data objects.

Extracted from the monolithic preprocessing.py to decouple dataset
management from graph construction logic.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Union

import numpy as np
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)


class GraphDataset(torch.utils.data.Dataset):
    """Dataset wrapper for CAN bus graph data.

    Provides a convenient interface for handling collections of
    graph objects with PyTorch Geometric DataLoader compatibility.
    """

    def __init__(self, data_list: List[Data]):
        super().__init__()
        self.data_list = data_list
        if data_list:
            self._validate_data_consistency()

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]

    def _validate_data_consistency(self) -> None:
        """Validate that all graphs have consistent feature dimensions."""
        if not self.data_list:
            return

        first = self.data_list[0]
        expected_node = first.x.size(1) if first.x is not None else 0
        expected_edge = first.edge_attr.size(1) if first.edge_attr is not None else 0

        for i, g in enumerate(self.data_list):
            if g.x is not None and g.x.size(1) != expected_node:
                raise ValueError(
                    f"Graph {i} has inconsistent node features: "
                    f"expected {expected_node}, got {g.x.size(1)}"
                )
            if g.edge_attr is not None and g.edge_attr.size(1) != expected_edge:
                raise ValueError(
                    f"Graph {i} has inconsistent edge features: "
                    f"expected {expected_edge}, got {g.edge_attr.size(1)}"
                )

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get comprehensive statistics about the dataset."""
        if not self.data_list:
            return {"num_graphs": 0}

        num_nodes = [g.num_nodes for g in self.data_list]
        num_edges = [g.num_edges for g in self.data_list]
        labels = [
            g.y.item() if g.y.dim() == 0 else g.y[0].item()
            for g in self.data_list
        ]

        return {
            "num_graphs": len(self.data_list),
            "avg_nodes": np.mean(num_nodes),
            "std_nodes": np.std(num_nodes),
            "avg_edges": np.mean(num_edges),
            "std_edges": np.std(num_edges),
            "min_nodes": min(num_nodes),
            "max_nodes": max(num_nodes),
            "min_edges": min(num_edges),
            "max_edges": max(num_edges),
            "normal_graphs": sum(1 for lbl in labels if lbl == 0),
            "attack_graphs": sum(1 for lbl in labels if lbl == 1),
            "node_features": (
                self.data_list[0].x.size(1) if self.data_list[0].x is not None else 0
            ),
            "edge_features": (
                self.data_list[0].edge_attr.size(1)
                if self.data_list[0].edge_attr is not None
                else 0
            ),
        }

    def print_stats(self) -> None:
        """Log comprehensive dataset statistics."""
        stats = self.get_stats()
        log.info("=" * 60)
        log.info("DATASET STATISTICS")
        log.info("=" * 60)
        log.info("Total graphs: %d", stats["num_graphs"])
        log.info(
            "Normal graphs: %d (%.1f%%)",
            stats["normal_graphs"],
            stats["normal_graphs"] / stats["num_graphs"] * 100,
        )
        log.info(
            "Attack graphs: %d (%.1f%%)",
            stats["attack_graphs"],
            stats["attack_graphs"] / stats["num_graphs"] * 100,
        )
        log.info("Graph Structure:")
        log.info(
            "  Nodes per graph: %.1f +/- %.1f [%d-%d]",
            stats["avg_nodes"],
            stats["std_nodes"],
            stats["min_nodes"],
            stats["max_nodes"],
        )
        log.info(
            "  Edges per graph: %.1f +/- %.1f [%d-%d]",
            stats["avg_edges"],
            stats["std_edges"],
            stats["min_edges"],
            stats["max_edges"],
        )
        log.info("Feature Dimensions:")
        log.info("  Node features: %d", stats["node_features"])
        log.info("  Edge features: %d", stats["edge_features"])
