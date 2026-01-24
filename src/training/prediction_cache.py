"""
Fusion Prediction Cache Builder

Pre-computes VGAE and GAT predictions for efficient fusion training.
Caches predictions to disk to avoid recomputation.

This module bridges the gap between pre-trained models and fusion training,
allowing the Lightning fusion module to work with cached predictions only.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from tqdm import tqdm
import pickle
import hashlib

from torch_geometric.loader import DataLoader
from src.training.fusion_extractor import FusionDataExtractor

logger = logging.getLogger(__name__)


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
