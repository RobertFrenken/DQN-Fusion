"""
Fusion Prediction Cache Builder

Pre-computes VGAE and GAT predictions for efficient fusion training.
Caches predictions to disk to avoid recomputation.

This module bridges the gap between pre-trained models and fusion training,
allowing the Lightning fusion module to work with cached predictions only.
"""

from contextlib import nullcontext
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging
from tqdm import tqdm
import pickle
import hashlib

from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)

# Fusion weights for composite anomaly scoring
FUSION_WEIGHTS = {
    'node_reconstruction': 0.4,
    'neighborhood_prediction': 0.35,
    'can_id_prediction': 0.25
}


# ============================================================================
# FUSION DATA EXTRACTOR (inlined from fusion_extractor.py)
# ============================================================================

class FusionDataExtractor:
    """Extract anomaly scores and GAT probabilities for fusion training - GPU OPTIMIZED."""
    
    def __init__(self, autoencoder: nn.Module, classifier: nn.Module, 
                 device: str, threshold: float = 0.0):
        self.autoencoder = autoencoder.to(device)
        self.classifier = classifier.to(device)
        self.device = torch.device(device)
        self.threshold = threshold
        
        # Set models to evaluation mode
        self.autoencoder.eval()
        self.classifier.eval()
        
        # Pre-compute fusion weights as tensor for GPU operations
        self.fusion_weights = torch.tensor([
            FUSION_WEIGHTS['node_reconstruction'],
            FUSION_WEIGHTS['neighborhood_prediction'],
            FUSION_WEIGHTS['can_id_prediction']
        ], dtype=torch.float32, device=self.device)
        
        logger.info(f"✓ Fusion Data Extractor initialized (GPU-Optimized) with threshold: {threshold:.4f}")

    def compute_anomaly_scores(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enhanced VGAE feature extraction for rich DQN state space.

        Returns:
            Tuple of (error_components, latent_summary, vgae_confidence):
            - error_components: [num_graphs, 3] (node, neighbor, canid errors)
            - latent_summary: [num_graphs, 4] (mean, std, max, min of latent z per graph)
            - vgae_confidence: [num_graphs] (inverse of error variance across 3 components)
        """
        with torch.no_grad():
            # Forward pass - capture latent z
            cont_out, canid_logits, neighbor_logits, z, _ = self.autoencoder(
                batch.x, batch.edge_index, batch.batch
            )

            # Create neighborhood targets more efficiently
            neighbor_targets = self.autoencoder.create_neighborhood_targets(
                batch.x, batch.edge_index, batch.batch
            )

            # Node-level errors (efficient)
            node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
            neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                neighbor_logits, neighbor_targets
            ).mean(dim=1)
            canid_pred = canid_logits.argmax(dim=1)
            true_canids = batch.x[:, 0].long()
            canid_errors = (canid_pred != true_canids).float()

            # MEMORY OPTIMIZATION: Process graphs in smaller chunks
            num_graphs = batch.batch.max().item() + 1
            chunk_size = min(2048, num_graphs)  # Process 2048 graphs at a time

            graph_errors_list = []
            latent_summary_list = []

            for chunk_start in range(0, num_graphs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_graphs)
                chunk_graph_errors = torch.zeros(chunk_end - chunk_start, 3,
                                            device=self.device, dtype=node_errors.dtype)
                chunk_latent_summary = torch.zeros(chunk_end - chunk_start, 4,
                                            device=self.device, dtype=z.dtype)

                for i, graph_idx in enumerate(range(chunk_start, chunk_end)):
                    node_mask = (batch.batch == graph_idx)
                    if node_mask.any():
                        # Error components (max error across nodes in graph)
                        graph_node_errors = torch.stack([
                            node_errors[node_mask],
                            neighbor_errors[node_mask],
                            canid_errors[node_mask]
                        ], dim=1)
                        chunk_graph_errors[i] = graph_node_errors.max(dim=0)[0]

                        # Latent space statistics (aggregated per graph)
                        graph_latent = z[node_mask]  # [num_nodes_in_graph, latent_dim]
                        chunk_latent_summary[i, 0] = graph_latent.mean()
                        chunk_latent_summary[i, 1] = graph_latent.std()
                        chunk_latent_summary[i, 2] = graph_latent.max()
                        chunk_latent_summary[i, 3] = graph_latent.min()

                graph_errors_list.append(chunk_graph_errors)
                latent_summary_list.append(chunk_latent_summary)

            # Combine chunks
            graph_errors = torch.cat(graph_errors_list, dim=0)  # [num_graphs, 3]
            latent_summary = torch.cat(latent_summary_list, dim=0)  # [num_graphs, 4]

            # VGAE confidence: inverse of error variance across 3 error components
            # High variance = uncertain (different errors disagree), low variance = confident
            error_variance = graph_errors.std(dim=1)  # [num_graphs]
            vgae_confidence = 1.0 / (1.0 + error_variance)  # Normalize to [0, 1]

            return graph_errors, latent_summary, vgae_confidence

    def compute_gat_probabilities(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enhanced GAT feature extraction for rich DQN state space.

        Returns:
            Tuple of (logits, embedding_summary, gat_confidence):
            - logits: [num_graphs, 2] (raw logits for both classes)
            - embedding_summary: [num_graphs, 4] (mean, std, max, min of pre-pooling embeddings per graph)
            - gat_confidence: [num_graphs] (1 - normalized entropy, higher = more confident)
        """
        import math

        with torch.no_grad():
            # Get intermediate representations (before pooling)
            xs = self.classifier(batch, return_intermediate=True)

            # Use the last layer's output before pooling
            pre_pooling_embeddings = xs[-1]  # [num_nodes, hidden_dim]

            # Extract per-graph embedding statistics
            num_graphs = batch.batch.max().item() + 1
            embedding_summary = torch.zeros(num_graphs, 4, device=self.device)

            for graph_idx in range(num_graphs):
                node_mask = (batch.batch == graph_idx)
                if node_mask.any():
                    graph_embeddings = pre_pooling_embeddings[node_mask]  # [num_nodes_in_graph, hidden_dim]
                    embedding_summary[graph_idx, 0] = graph_embeddings.mean()
                    embedding_summary[graph_idx, 1] = graph_embeddings.std()
                    embedding_summary[graph_idx, 2] = graph_embeddings.max()
                    embedding_summary[graph_idx, 3] = graph_embeddings.min()

            # Now get final logits for classification
            logits = self.classifier(batch)  # [num_graphs, 2]

            # Handle both [batch_size] and [batch_size, 2] output shapes
            if logits.dim() == 1:
                # Single output per graph - convert to 2-class format
                logits = torch.stack([1.0 - logits, logits], dim=1)
            elif logits.shape[0] == 1 and logits.dim() > 1:
                logits = logits.squeeze(0)

            # Compute GAT confidence using softmax entropy
            # Lower entropy = more confident (peaked distribution)
            # Higher entropy = less confident (uniform distribution)
            probs = torch.softmax(logits, dim=1)  # [num_graphs, 2]
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)  # [num_graphs]
            max_entropy = math.log(2)  # Max entropy for 2-class problem
            gat_confidence = 1.0 - (entropy / max_entropy)  # Normalize to [0, 1], higher = more confident

            return logits, embedding_summary, gat_confidence

    def extract_fusion_data(self, data_loader: DataLoader, max_samples: int = None) -> dict:
        """
        Enhanced fusion data extraction with rich 15D state space.

        Returns:
            Dict with keys:
            - 'vgae_errors': List[torch.Tensor] - 3 error components per graph
            - 'vgae_latent': List[torch.Tensor] - 4 latent statistics per graph
            - 'vgae_confidence': List[torch.Tensor] - 1 confidence per graph
            - 'gat_logits': List[torch.Tensor] - 2 logits per graph
            - 'gat_embeddings': List[torch.Tensor] - 4 embedding statistics per graph
            - 'gat_confidence': List[torch.Tensor] - 1 confidence per graph
            - 'labels': List[torch.Tensor] - Ground truth labels
        """
        logger.info("🚀 GPU-Optimized Fusion Data Extraction (Enhanced 15D State Space)...")

        # Pre-allocate GPU tensors to avoid repeated allocation
        device_tensors = {
            'vgae_errors': [],           # [num_graphs, 3]
            'vgae_latent': [],           # [num_graphs, 4]
            'vgae_confidence': [],       # [num_graphs]
            'gat_logits': [],            # [num_graphs, 2]
            'gat_embeddings': [],        # [num_graphs, 4]
            'gat_confidence': [],        # [num_graphs]
            'labels': []                 # [num_graphs]
        }

        samples_processed = 0
        total_batches = len(data_loader)

        # Process in larger chunks to reduce Python overhead
        with torch.cuda.stream(torch.cuda.Stream()) if self.device.type == 'cuda' else nullcontext() as stream:
            with tqdm(data_loader, desc="GPU Extraction (15D)", total=total_batches,
                    miniters=max(1, total_batches//20)) as pbar:

                for batch_idx, batch in enumerate(pbar):
                    if self.device.type == 'cuda':
                        # Async GPU transfer with stream
                        batch = batch.to(self.device, non_blocking=True)

                    # Vectorized computation without intermediate CPU transfers
                    with torch.no_grad():
                        # VGAE features: errors (3), latent (4), confidence (1) = 8 dimensions
                        vgae_errors, vgae_latent, vgae_conf = self.compute_anomaly_scores(batch)

                        # GAT features: logits (2), embeddings (4), confidence (1) = 7 dimensions
                        gat_logits, gat_embeddings, gat_conf = self.compute_gat_probabilities(batch)

                        # Extract labels efficiently (keep on GPU)
                        if hasattr(batch, 'y') and batch.y.shape[0] == batch.num_graphs:
                            batch_labels = batch.y
                        else:
                            # Handle per-node labels -> per-graph labels
                            num_graphs = batch.batch.max().item() + 1
                            batch_labels = torch.zeros(num_graphs, device=self.device, dtype=batch.y.dtype)

                            for graph_idx in range(num_graphs):
                                node_mask = (batch.batch == graph_idx)
                                if node_mask.any():
                                    batch_labels[graph_idx] = batch.y[node_mask].max()

                    # Accumulate on GPU (no CPU transfer yet)
                    device_tensors['vgae_errors'].append(vgae_errors)
                    device_tensors['vgae_latent'].append(vgae_latent)
                    device_tensors['vgae_confidence'].append(vgae_conf)
                    device_tensors['gat_logits'].append(gat_logits)
                    device_tensors['gat_embeddings'].append(gat_embeddings)
                    device_tensors['gat_confidence'].append(gat_conf)
                    device_tensors['labels'].append(batch_labels)
                    
                    samples_processed += batch.num_graphs
                    
                    # Update progress less frequently for speed
                    if batch_idx % 50 == 0:
                        # Get GPU utilization if available (nvidia-ml-py may not be installed)
                        gpu_util = "N/A"
                        if self.device.type == 'cuda':
                            try:
                                gpu_util = f"{torch.cuda.utilization():.0f}%"
                            except (ModuleNotFoundError, RuntimeError):
                                # nvidia-ml-py not available
                                gpu_util = "N/A"
                        pbar.set_postfix({
                            'samples': f"{samples_processed:,}",
                            'gpu_util': gpu_util
                        })
                    
                    if max_samples and samples_processed >= max_samples:
                        break
                    
                    # Less frequent GPU cache clearing
                    if batch_idx % 100 == 0 and batch_idx > 0:
                        torch.cuda.empty_cache()
        
        # Single GPU→CPU transfer at the end (minimizes transfer overhead)
        logger.info("📥 Transferring results from GPU to CPU (15D state space)...")

        result = {
            # VGAE features (8 dims total)
            'vgae_errors': torch.cat(device_tensors['vgae_errors']).cpu().numpy(),  # [N, 3]
            'vgae_latent': torch.cat(device_tensors['vgae_latent']).cpu().numpy(),  # [N, 4]
            'vgae_confidence': torch.cat(device_tensors['vgae_confidence']).cpu().numpy(),  # [N]

            # GAT features (7 dims total)
            'gat_logits': torch.cat(device_tensors['gat_logits']).cpu().numpy(),  # [N, 2]
            'gat_embeddings': torch.cat(device_tensors['gat_embeddings']).cpu().numpy(),  # [N, 4]
            'gat_confidence': torch.cat(device_tensors['gat_confidence']).cpu().numpy(),  # [N]

            # Labels
            'labels': torch.cat(device_tensors['labels']).cpu().numpy()  # [N]
        }

        # Clean up GPU memory
        del device_tensors
        torch.cuda.empty_cache()

        num_samples = len(result['labels'])
        logger.info(f"✓ Extracted {num_samples} samples with 15D state space (GPU-optimized)")
        logger.info(f"  VGAE: 3 errors + 4 latent + 1 confidence = 8 dims")
        logger.info(f"  GAT: 2 logits + 4 embeddings + 1 confidence = 7 dims")
        logger.info(f"  Total: 15 dimensions per sample")

        return result


# ============================================================================
# PREDICTION CACHE BUILDER
# ============================================================================


class PredictionCacheBuilder:
    """
    Builds and manages prediction caches for fusion training.
    
    Caching strategy:
    1. Load pre-trained VGAE and GAT models
    2. Forward pass on all data → collect predictions
    3. Save to disk for quick loading
    4. Hash cache files to detect stale caches
    """
    
    def __init__(self, device: str = 'cuda', cache_dir: str = 'cache/fusion'):
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Will be set by setup()
        self.data_extractor: Optional[FusionDataExtractor] = None
    
    def setup(self, autoencoder: nn.Module, classifier: nn.Module):
        """Initialize the data extractor with loaded models."""
        self.data_extractor = FusionDataExtractor(
            autoencoder=autoencoder,
            classifier=classifier,
            device=str(self.device)
        )
        logger.info("✓ Prediction cache builder initialized")
    
    def get_cache_path(self, dataset_name: str, split: str = 'train') -> Path:
        """Get path for cached predictions."""
        return self.cache_dir / f"{dataset_name}_{split}_predictions.pkl"
    
    def build_cache(self, data_loader: DataLoader, dataset_name: str,
                   split: str = 'train', max_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Build prediction cache from data loader with 15D state space.

        Args:
            data_loader: PyG DataLoader with graph data
            dataset_name: Name for cache file
            split: 'train' or 'val' for naming
            max_samples: Maximum samples to cache (None = all)

        Returns:
            Dict with 15D features:
            - 'vgae_errors': [N, 3]
            - 'vgae_latent': [N, 4]
            - 'vgae_confidence': [N]
            - 'gat_logits': [N, 2]
            - 'gat_embeddings': [N, 4]
            - 'gat_confidence': [N]
            - 'labels': [N]
        """
        cache_path = self.get_cache_path(dataset_name, split)

        # Check if cache exists and is valid
        if cache_path.exists():
            logger.info(f"💾 Loading cached predictions from {cache_path}")
            return self._load_cache(cache_path)

        logger.info(f"🔄 Building prediction cache for {split} split (15D state space)...")

        if self.data_extractor is None:
            raise RuntimeError("Must call setup() before building cache")

        # Extract predictions (returns dict with 15D features)
        feature_dict = self.data_extractor.extract_fusion_data(
            data_loader,
            max_samples=max_samples
        )

        # Save cache
        self._save_cache(cache_path, feature_dict)

        logger.info(f"✓ Cached {len(feature_dict['labels'])} samples (15D state) to {cache_path}")

        return feature_dict
    
    def _save_cache(self, path: Path, feature_dict: Dict[str, np.ndarray]):
        """Save predictions to disk (15D state space)."""
        cache_data = {
            **feature_dict,  # Include all 15D features
            'metadata': {
                'created': time.time(),
                'num_samples': len(feature_dict['labels']),
                'state_dim': 15,
                'vgae_dims': 8,
                'gat_dims': 7
            }
        }

        with open(path, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info(f"💾 Saved 15D cache to {path}")
    
    def _load_cache(self, path: Path) -> Dict[str, np.ndarray]:
        """Load predictions from disk (15D state space)."""
        with open(path, 'rb') as f:
            cache_data = pickle.load(f)

        # Check if this is old 2D format or new 15D format
        if 'vgae_errors' in cache_data:
            # New 15D format
            return {
                'vgae_errors': cache_data['vgae_errors'],
                'vgae_latent': cache_data['vgae_latent'],
                'vgae_confidence': cache_data['vgae_confidence'],
                'gat_logits': cache_data['gat_logits'],
                'gat_embeddings': cache_data['gat_embeddings'],
                'gat_confidence': cache_data['gat_confidence'],
                'labels': cache_data['labels']
            }
        else:
            # Old 2D format - raise error to force regeneration
            raise ValueError(
                f"Cache at {path} uses old 2D format. Please delete and regenerate with new 15D format."
            )
    
    def clear_cache(self, dataset_name: str):
        """Clear cached predictions for a dataset."""
        for split in ['train', 'val']:
            cache_path = self.get_cache_path(dataset_name, split)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"🗑️  Cleared cache: {cache_path}")


def create_fusion_prediction_cache(
    autoencoder: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset_name: str,
    device: str = 'cuda',
    cache_dir: str = 'cache/fusion',
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Convenience function to create both training and validation caches with 15D state space.

    Returns:
        Tuple of (train_features, val_features) where each is a dict with:
        - 'vgae_errors': [N, 3]
        - 'vgae_latent': [N, 4]
        - 'vgae_confidence': [N]
        - 'gat_logits': [N, 2]
        - 'gat_embeddings': [N, 4]
        - 'gat_confidence': [N]
        - 'labels': [N]
    """
    builder = PredictionCacheBuilder(device=device, cache_dir=cache_dir)
    builder.setup(autoencoder, classifier)

    # Build training cache (returns dict with 15D features)
    train_features = builder.build_cache(
        train_loader,
        dataset_name,
        split='train',
        max_samples=max_train_samples
    )

    # Build validation cache (returns dict with 15D features)
    val_features = builder.build_cache(
        val_loader,
        dataset_name,
        split='val',
        max_samples=max_val_samples
    )

    logger.info(f"✓ Fusion caches created: {len(train_features['labels'])} train, {len(val_features['labels'])} val samples")

    return train_features, val_features
