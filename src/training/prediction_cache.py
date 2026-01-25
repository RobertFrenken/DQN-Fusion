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
from typing import Tuple, Optional, List
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

    def compute_anomaly_scores(self, batch) -> torch.Tensor:
        """Memory-efficient computation without massive tensors."""
        with torch.no_grad():
            # Forward pass (normal)
            cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
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
            for chunk_start in range(0, num_graphs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_graphs)
                chunk_graph_errors = torch.zeros(chunk_end - chunk_start, 3, 
                                            device=self.device, dtype=node_errors.dtype)
                
                for i, graph_idx in enumerate(range(chunk_start, chunk_end)):
                    node_mask = (batch.batch == graph_idx)
                    if node_mask.any():
                        graph_node_errors = torch.stack([
                            node_errors[node_mask], 
                            neighbor_errors[node_mask], 
                            canid_errors[node_mask]
                        ], dim=1)
                        chunk_graph_errors[i] = graph_node_errors.max(dim=0)[0]
                
                graph_errors_list.append(chunk_graph_errors)
            
            # Combine chunks
            graph_errors = torch.cat(graph_errors_list, dim=0)
            
            # Weighted composite score (efficient)
            composite_scores = (graph_errors * self.fusion_weights).sum(dim=1)
            return torch.sigmoid(composite_scores * 3 - 1.5)

    def compute_gat_probabilities(self, batch) -> torch.Tensor:
        """
        GPU-accelerated computation of GAT classification probabilities.
        Already efficient - just needs to stay on GPU.
        """
        with torch.no_grad():
            logits = self.classifier(batch)
            probabilities = torch.sigmoid(logits.squeeze())
            return probabilities  # Keep on GPU!

    def extract_fusion_data(self, data_loader: DataLoader, max_samples: int = None) -> Tuple[List, List, List]:
        """Eliminate CPU serialization bottleneck."""
        logger.info("🚀 GPU-Optimized Fusion Data Extraction...")
        
        # Pre-allocate GPU tensors to avoid repeated allocation
        device_tensors = {
            'anomaly_scores': [],
            'gat_probs': [],  
            'labels': []
        }
        
        samples_processed = 0
        total_batches = len(data_loader)
        
        # Process in larger chunks to reduce Python overhead
        with torch.cuda.stream(torch.cuda.Stream()) if self.device.type == 'cuda' else nullcontext() as stream:
            with tqdm(data_loader, desc="GPU Extraction", total=total_batches, 
                    miniters=max(1, total_batches//20)) as pbar:
                
                for batch_idx, batch in enumerate(pbar):
                    if self.device.type == 'cuda':
                        # Async GPU transfer with stream
                        batch = batch.to(self.device, non_blocking=True)
                        
                    # Vectorized computation without intermediate CPU transfers
                    with torch.no_grad():
                        batch_anomaly_scores = self.compute_anomaly_scores(batch)
                        batch_gat_probs = self.compute_gat_probabilities(batch)
                        
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
                    device_tensors['anomaly_scores'].append(batch_anomaly_scores)
                    device_tensors['gat_probs'].append(batch_gat_probs)
                    device_tensors['labels'].append(batch_labels)
                    
                    samples_processed += batch.num_graphs
                    
                    # Update progress less frequently for speed
                    if batch_idx % 50 == 0:
                        pbar.set_postfix({
                            'samples': f"{samples_processed:,}",
                            'gpu_util': f"{torch.cuda.utilization():.0f}%" if self.device.type == 'cuda' else "N/A"
                        })
                    
                    if max_samples and samples_processed >= max_samples:
                        break
                    
                    # Less frequent GPU cache clearing
                    if batch_idx % 100 == 0 and batch_idx > 0:
                        torch.cuda.empty_cache()
        
        # Single GPU→CPU transfer at the end (minimizes transfer overhead)
        logger.info("📥 Transferring results from GPU to CPU...")
        anomaly_scores = torch.cat(device_tensors['anomaly_scores']).cpu().numpy().tolist()
        gat_probabilities = torch.cat(device_tensors['gat_probs']).cpu().numpy().tolist()
        labels = torch.cat(device_tensors['labels']).cpu().numpy().tolist()
        
        # Clean up GPU memory
        del device_tensors
        torch.cuda.empty_cache()
        
        logger.info(f"✓ Extracted {len(anomaly_scores)} predictions (GPU-optimized)")
        return anomaly_scores, gat_probabilities, labels


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
                   split: str = 'train', max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build prediction cache from data loader.
        
        Args:
            data_loader: PyG DataLoader with graph data
            dataset_name: Name for cache file
            split: 'train' or 'val' for naming
            max_samples: Maximum samples to cache (None = all)
        
        Returns:
            Tuple of (anomaly_scores, gat_probs, labels) as numpy arrays
        """
        cache_path = self.get_cache_path(dataset_name, split)
        
        # Check if cache exists and is valid
        if cache_path.exists():
            logger.info(f"💾 Loading cached predictions from {cache_path}")
            return self._load_cache(cache_path)
        
        logger.info(f"🔄 Building prediction cache for {split} split...")
        
        if self.data_extractor is None:
            raise RuntimeError("Must call setup() before building cache")
        
        # Extract predictions
        anomaly_scores, gat_probs, labels = self.data_extractor.extract_fusion_data(
            data_loader, 
            max_samples=max_samples
        )
        
        # Convert to numpy
        anomaly_scores = np.array(anomaly_scores, dtype=np.float32)
        gat_probs = np.array(gat_probs, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Save cache
        self._save_cache(cache_path, anomaly_scores, gat_probs, labels)
        
        logger.info(f"✓ Cached {len(anomaly_scores)} predictions to {cache_path}")
        
        return anomaly_scores, gat_probs, labels
    
    def _save_cache(self, path: Path, anomaly_scores: np.ndarray, 
                   gat_probs: np.ndarray, labels: np.ndarray):
        """Save predictions to disk."""
        cache_data = {
            'anomaly_scores': anomaly_scores,
            'gat_probs': gat_probs,
            'labels': labels,
            'metadata': {
                'num_samples': len(anomaly_scores),
                'dtype': str(anomaly_scores.dtype)
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"💾 Saved cache to {path}")
    
    def _load_cache(self, path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load predictions from disk."""
        with open(path, 'rb') as f:
            cache_data = pickle.load(f)
        
        return (
            cache_data['anomaly_scores'],
            cache_data['gat_probs'],
            cache_data['labels']
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to create both training and validation caches.
    
    Returns:
        Tuple of (train_anomaly, train_gat, train_labels, val_anomaly, val_gat, val_labels)
    """
    builder = PredictionCacheBuilder(device=device, cache_dir=cache_dir)
    builder.setup(autoencoder, classifier)
    
    # Build training cache
    train_anomaly, train_gat, train_labels = builder.build_cache(
        train_loader,
        dataset_name,
        split='train',
        max_samples=max_train_samples
    )
    
    # Build validation cache
    val_anomaly, val_gat, val_labels = builder.build_cache(
        val_loader,
        dataset_name,
        split='val',
        max_samples=max_val_samples
    )

    # Ensure all returned arrays have the same length by trimming to the smallest length (robustness for tiny datasets)
    def _trim_to_min(a, b, c):
        min_len = min(len(a), len(b), len(c))
        if min_len != len(a) or min_len != len(b) or min_len != len(c):
            logger.warning(f"Trimming fusion prediction arrays to min length={min_len}: (was {len(a)},{len(b)},{len(c)})")
        return a[:min_len], b[:min_len], c[:min_len]

    train_anomaly, train_gat, train_labels = _trim_to_min(train_anomaly, train_gat, train_labels)
    val_anomaly, val_gat, val_labels = _trim_to_min(val_anomaly, val_gat, val_labels)
    
    return train_anomaly, train_gat, train_labels, val_anomaly, val_gat, val_labels
