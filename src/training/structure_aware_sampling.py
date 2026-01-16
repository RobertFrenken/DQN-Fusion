"""
Graph-Structure Aware Sampling for Class Imbalance

Novel approach: Sample based on graph structural similarity to attacks,
not just reconstruction error.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from torch_geometric.utils import degree
from sklearn.cluster import KMeans


class GraphStructureSampler:
    """
    Sample normal graphs that are structurally similar to attacks.
    This creates harder negative examples for better boundary learning.
    """
    
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters
        self.attack_prototypes = None
        self.normal_clusters = None
        
    def extract_graph_features(self, graphs: List) -> np.ndarray:
        """Extract structural features from graphs."""
        features = []
        
        for graph in graphs:
            # Node-level features
            num_nodes = graph.num_nodes
            num_edges = graph.num_edges
            
            # Degree statistics
            degrees = degree(graph.edge_index[0], num_nodes=num_nodes)
            degree_mean = degrees.mean().item()
            degree_std = degrees.std().item()
            degree_max = degrees.max().item()
            
            # Edge density
            max_edges = num_nodes * (num_nodes - 1) / 2
            edge_density = num_edges / max_edges if max_edges > 0 else 0
            
            # Node feature statistics (first few dimensions)
            node_feat_mean = graph.x.mean(dim=0)[:5].cpu().numpy()
            node_feat_std = graph.x.std(dim=0)[:5].cpu().numpy()
            
            # Combine features
            struct_features = np.concatenate([
                [num_nodes, num_edges, degree_mean, degree_std, degree_max, edge_density],
                node_feat_mean,
                node_feat_std
            ])
            
            features.append(struct_features)
        
        return np.array(features)
    
    def fit(self, normal_graphs: List, attack_graphs: List):
        """Learn structural patterns from attack and normal graphs."""
        print("Learning graph structural patterns...")
        
        # Extract features
        attack_features = self.extract_graph_features(attack_graphs)
        normal_features = self.extract_graph_features(normal_graphs)
        
        # Learn attack prototypes (centroids)
        attack_kmeans = KMeans(n_clusters=min(self.n_clusters, len(attack_graphs)))
        attack_kmeans.fit(attack_features)
        self.attack_prototypes = attack_kmeans.cluster_centers_
        
        # Cluster normal graphs
        normal_kmeans = KMeans(n_clusters=self.n_clusters)
        normal_labels = normal_kmeans.fit_predict(normal_features)
        
        self.normal_clusters = {
            'features': normal_features,
            'labels': normal_labels,
            'centroids': normal_kmeans.cluster_centers_,
            'graphs': normal_graphs
        }
        
        print(f"Found {len(self.attack_prototypes)} attack prototypes")
        print(f"Clustered {len(normal_graphs)} normal graphs into {self.n_clusters} clusters")
    
    def compute_attack_similarity(self, normal_features: np.ndarray) -> np.ndarray:
        """Compute how similar normal graphs are to attack patterns."""
        similarities = []
        
        for normal_feat in normal_features:
            # Distance to closest attack prototype
            distances = [np.linalg.norm(normal_feat - proto) 
                        for proto in self.attack_prototypes]
            min_distance = min(distances)
            similarity = 1.0 / (1.0 + min_distance)  # Higher = more similar
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def sample_hard_normals(self, target_count: int, 
                           similarity_percentile: float = 75) -> List:
        """Sample normal graphs that are structurally similar to attacks."""
        if self.normal_clusters is None:
            raise ValueError("Must call fit() first")
        
        # Compute attack similarity for all normals
        similarities = self.compute_attack_similarity(
            self.normal_clusters['features']
        )
        
        # Select top percentile most attack-like normals
        threshold = np.percentile(similarities, similarity_percentile)
        hard_indices = np.where(similarities >= threshold)[0]
        
        # Sample from hard examples
        if len(hard_indices) >= target_count:
            selected_indices = np.random.choice(hard_indices, target_count, replace=False)
        else:
            # Not enough hard examples, fill with random
            remaining = target_count - len(hard_indices)
            easy_indices = np.where(similarities < threshold)[0]
            extra_indices = np.random.choice(easy_indices, remaining, replace=False)
            selected_indices = np.concatenate([hard_indices, extra_indices])
        
        return [self.normal_clusters['graphs'][i] for i in selected_indices]
    
    def create_adversarial_batch(self, attack_graphs: List, 
                               normal_ratio: float = 10.0) -> List:
        """Create batch with attack-like normal examples."""
        n_attacks = len(attack_graphs)
        n_normals = int(n_attacks * normal_ratio)
        
        # Sample structurally hard normals
        hard_normals = self.sample_hard_normals(n_normals)
        
        return attack_graphs + hard_normals


class StructureAwareLoss(torch.nn.Module):
    """
    Loss that considers graph structural relationships.
    """
    
    def __init__(self, structure_weight: float = 0.5):
        super().__init__()
        self.structure_weight = structure_weight
        
    def compute_structure_penalty(self, predictions: torch.Tensor, 
                                 graph_similarities: torch.Tensor) -> torch.Tensor:
        """
        Penalize predictions that don't respect structural similarity.
        Similar graphs should have similar predictions.
        """
        # Convert predictions to probabilities
        probs = torch.sigmoid(predictions)
        
        # Compute prediction similarity matrix
        pred_sim = 1.0 - torch.abs(probs.unsqueeze(0) - probs.unsqueeze(1))
        
        # Structure consistency loss
        structure_loss = F.mse_loss(pred_sim, graph_similarities)
        
        return structure_loss
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                graph_similarities: torch.Tensor = None) -> torch.Tensor:
        """Compute structure-aware loss."""
        # Base classification loss
        base_loss = F.binary_cross_entropy_with_logits(predictions, targets)
        
        if graph_similarities is not None:
            # Add structure consistency penalty
            structure_loss = self.compute_structure_penalty(predictions, graph_similarities)
            total_loss = base_loss + self.structure_weight * structure_loss
        else:
            total_loss = base_loss
            
        return total_loss