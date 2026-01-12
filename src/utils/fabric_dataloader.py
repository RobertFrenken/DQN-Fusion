"""
Optimized data loading utilities for PyTorch Lightning Fabric training.

This module provides:
- High-performance DataLoader configurations
- Memory-efficient graph data handling
- Prefetching and caching strategies
- Dynamic batch sizing integration
- Multi-worker optimization
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import logging
from pathlib import Path
import pickle
import psutil
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import gc

logger = logging.getLogger(__name__)


class OptimizedGraphDataLoader:
    """
    Optimized DataLoader for graph data with Fabric integration.
    """
    
    def __init__(self,
                 dataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = None,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 prefetch_factor: int = 2,
                 drop_last: bool = False,
                 memory_fraction: float = 0.8):
        """
        Initialize optimized graph data loader.
        
        Args:
            dataset: PyTorch Geometric dataset
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to use pinned memory
            persistent_workers: Whether to keep workers alive between epochs
            prefetch_factor: Number of batches to prefetch per worker
            drop_last: Whether to drop the last incomplete batch
            memory_fraction: Fraction of available memory to use for caching
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.memory_fraction = memory_fraction
        
        # Auto-detect optimal number of workers if not specified
        if num_workers is None:
            num_workers = self._determine_optimal_workers()
        
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        
        # Initialize caching
        self.cache_enabled = self._should_enable_cache()
        self.cached_batches = {}
        
        logger.info(f"OptimizedGraphDataLoader initialized:")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Workers: {num_workers}")
        logger.info(f"  - Pin memory: {self.pin_memory}")
        logger.info(f"  - Persistent workers: {self.persistent_workers}")
        logger.info(f"  - Prefetch factor: {prefetch_factor}")
        logger.info(f"  - Cache enabled: {self.cache_enabled}")
    
    def _determine_optimal_workers(self) -> int:
        """Determine optimal number of worker processes."""
        cpu_count = os.cpu_count() or 4
        
        # Conservative approach: use 50% of available CPUs, capped at 8
        optimal_workers = min(cpu_count // 2, 8)
        
        # Adjust based on dataset size
        dataset_size = len(self.dataset)
        if dataset_size < 1000:
            optimal_workers = min(optimal_workers, 2)
        elif dataset_size < 10000:
            optimal_workers = min(optimal_workers, 4)
        
        return max(optimal_workers, 1)
    
    def _should_enable_cache(self) -> bool:
        """Determine if batch caching should be enabled."""
        # Get available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # Enable caching if we have sufficient memory and small dataset
        dataset_size = len(self.dataset)
        estimated_memory_per_sample = 0.001  # 1MB per sample (rough estimate)
        estimated_total_memory = dataset_size * estimated_memory_per_sample
        
        return (available_gb > 4.0 and 
                estimated_total_memory < available_gb * self.memory_fraction)
    
    def create_dataloader(self) -> GeometricDataLoader:
        """
        Create optimized PyTorch Geometric DataLoader.
        
        Returns:
            Configured DataLoader
        """
        # Custom collate function for better memory handling
        def optimized_collate_fn(batch):
            return self._optimized_batch_creation(batch)
        
        dataloader = GeometricDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
            # collate_fn=optimized_collate_fn  # Uncomment if needed
        )
        
        return dataloader
    
    def _optimized_batch_creation(self, batch_list: List[Data]) -> Batch:
        """
        Create optimized batches with memory management.
        
        Args:
            batch_list: List of Data objects
            
        Returns:
            Batched Data object
        """
        # Pre-allocate tensors for efficiency
        total_nodes = sum(data.x.size(0) for data in batch_list)
        total_edges = sum(data.edge_index.size(1) for data in batch_list)
        
        # Create batch more efficiently
        batch = Batch.from_data_list(batch_list)
        
        # Optimize tensor storage
        if hasattr(batch.x, 'contiguous'):
            batch.x = batch.x.contiguous()
        if hasattr(batch.edge_index, 'contiguous'):
            batch.edge_index = batch.edge_index.contiguous()
        
        return batch


class DynamicBatchSampler:
    """
    Dynamic batch sampler that adjusts batch size based on memory usage.
    """
    
    def __init__(self,
                 dataset_size: int,
                 initial_batch_size: int,
                 max_batch_size: int,
                 min_batch_size: int = 1,
                 memory_threshold: float = 0.85):
        """
        Initialize dynamic batch sampler.
        
        Args:
            dataset_size: Total number of samples in dataset
            initial_batch_size: Starting batch size
            max_batch_size: Maximum allowed batch size
            min_batch_size: Minimum allowed batch size
            memory_threshold: GPU memory usage threshold for batch size adjustment
        """
        self.dataset_size = dataset_size
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.memory_threshold = memory_threshold
        
        # Performance tracking
        self.batch_times = []
        self.memory_usage = []
        
    def __iter__(self):
        """Iterate through dynamic batch indices."""
        indices = list(range(self.dataset_size))
        
        i = 0
        while i < len(indices):
            # Get current batch
            batch_end = min(i + self.current_batch_size, len(indices))
            batch_indices = indices[i:batch_end]
            
            yield batch_indices
            
            # Update for next iteration
            i = batch_end
            
            # Adjust batch size based on performance
            self._adjust_batch_size()
    
    def _adjust_batch_size(self):
        """Adjust batch size based on memory usage and performance."""
        if not torch.cuda.is_available():
            return
        
        # Get current memory usage
        memory_allocated = torch.cuda.memory_allocated()
        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_fraction = memory_allocated / memory_total
        
        # Adjust batch size
        if memory_fraction > self.memory_threshold:
            # Reduce batch size if memory usage is high
            new_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
            if new_batch_size != self.current_batch_size:
                logger.debug(f"Reducing batch size: {self.current_batch_size} -> {new_batch_size}")
                self.current_batch_size = new_batch_size
        elif memory_fraction < self.memory_threshold * 0.7:
            # Increase batch size if memory usage is low
            new_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
            if new_batch_size != self.current_batch_size:
                logger.debug(f"Increasing batch size: {self.current_batch_size} -> {new_batch_size}")
                self.current_batch_size = new_batch_size
    
    def __len__(self):
        """Return number of batches."""
        return (self.dataset_size + self.current_batch_size - 1) // self.current_batch_size


class CachedGraphDataset:
    """
    Dataset wrapper with intelligent caching for graph data.
    """
    
    def __init__(self,
                 dataset,
                 cache_dir: str = None,
                 memory_cache_size: int = 1000,
                 preprocess_fn: Callable = None):
        """
        Initialize cached dataset.
        
        Args:
            dataset: Original dataset
            cache_dir: Directory for disk cache
            memory_cache_size: Number of samples to keep in memory
            preprocess_fn: Optional preprocessing function
        """
        self.dataset = dataset
        self.memory_cache_size = memory_cache_size
        self.preprocess_fn = preprocess_fn
        
        # Memory cache
        self.memory_cache = {}
        self.cache_access_order = []
        
        # Disk cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.disk_cache_enabled = True
        else:
            self.disk_cache_enabled = False
        
        logger.info(f"CachedGraphDataset initialized with memory cache size: {memory_cache_size}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Check memory cache first
        if idx in self.memory_cache:
            self._update_cache_access(idx)
            return self.memory_cache[idx]
        
        # Check disk cache
        if self.disk_cache_enabled:
            cached_item = self._load_from_disk_cache(idx)
            if cached_item is not None:
                self._add_to_memory_cache(idx, cached_item)
                return cached_item
        
        # Load from original dataset
        item = self.dataset[idx]
        
        # Apply preprocessing if provided
        if self.preprocess_fn:
            item = self.preprocess_fn(item)
        
        # Add to caches
        self._add_to_memory_cache(idx, item)
        if self.disk_cache_enabled:
            self._save_to_disk_cache(idx, item)
        
        return item
    
    def _add_to_memory_cache(self, idx, item):
        """Add item to memory cache with LRU eviction."""
        # Remove oldest item if cache is full
        if len(self.memory_cache) >= self.memory_cache_size:
            oldest_idx = self.cache_access_order.pop(0)
            del self.memory_cache[oldest_idx]
        
        self.memory_cache[idx] = item
        self.cache_access_order.append(idx)
    
    def _update_cache_access(self, idx):
        """Update access order for LRU cache."""
        if idx in self.cache_access_order:
            self.cache_access_order.remove(idx)
        self.cache_access_order.append(idx)
    
    def _load_from_disk_cache(self, idx) -> Optional[Data]:
        """Load item from disk cache."""
        cache_path = self.cache_dir / f"sample_{idx}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached item {idx}: {e}")
        return None
    
    def _save_to_disk_cache(self, idx, item):
        """Save item to disk cache."""
        cache_path = self.cache_dir / f"sample_{idx}.pkl"
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(item, f)
        except Exception as e:
            logger.warning(f"Failed to cache item {idx}: {e}")


class PrefetchingDataLoader:
    """
    DataLoader with advanced prefetching capabilities.
    """
    
    def __init__(self,
                 dataloader,
                 prefetch_batches: int = 2,
                 device: torch.device = None):
        """
        Initialize prefetching DataLoader.
        
        Args:
            dataloader: Base DataLoader
            prefetch_batches: Number of batches to prefetch
            device: Target device for prefetched data
        """
        self.dataloader = dataloader
        self.prefetch_batches = prefetch_batches
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.prefetch_queue = Queue(maxsize=prefetch_batches)
        self.prefetch_thread = None
        self.stop_prefetching = threading.Event()
    
    def __iter__(self):
        """Iterator with prefetching."""
        self.stop_prefetching.clear()
        
        # Start prefetching thread
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(iter(self.dataloader),)
        )
        self.prefetch_thread.start()
        
        # Yield prefetched batches
        try:
            while True:
                try:
                    batch = self.prefetch_queue.get(timeout=1.0)
                    if batch is None:  # Sentinel value indicating end
                        break
                    yield batch
                except Empty:
                    if not self.prefetch_thread.is_alive():
                        break
        finally:
            # Cleanup
            self.stop_prefetching.set()
            if self.prefetch_thread and self.prefetch_thread.is_alive():
                self.prefetch_thread.join()
    
    def _prefetch_worker(self, data_iter):
        """Worker function for prefetching batches."""
        try:
            for batch in data_iter:
                if self.stop_prefetching.is_set():
                    break
                
                # Move batch to device
                if self.device.type == 'cuda':
                    batch = batch.to(self.device, non_blocking=True)
                
                # Add to queue
                self.prefetch_queue.put(batch)
            
            # Signal end of data
            self.prefetch_queue.put(None)
            
        except Exception as e:
            logger.error(f"Prefetching error: {e}")
            self.prefetch_queue.put(None)
    
    def __len__(self):
        return len(self.dataloader)


class FabricDataLoaderFactory:
    """
    Factory for creating optimized DataLoaders for Fabric training.
    """
    
    @staticmethod
    def create_gat_dataloader(dataset,
                             batch_size: int,
                             split: str = 'train',
                             num_workers: int = None,
                             use_cache: bool = True,
                             use_prefetch: bool = True) -> DataLoader:
        """
        Create optimized DataLoader for GAT training.
        
        Args:
            dataset: PyTorch Geometric dataset
            batch_size: Batch size
            split: Dataset split ('train', 'val', 'test')
            num_workers: Number of workers
            use_cache: Whether to use caching
            use_prefetch: Whether to use prefetching
            
        Returns:
            Optimized DataLoader
        """
        # Apply caching if enabled
        if use_cache:
            cache_dir = f"cache/gat_{split}" if split != 'train' else None
            dataset = CachedGraphDataset(
                dataset,
                cache_dir=cache_dir,
                memory_cache_size=min(1000, len(dataset) // 10)
            )
        
        # Create optimized loader
        loader_factory = OptimizedGraphDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        dataloader = loader_factory.create_dataloader()
        
        # Add prefetching if enabled
        if use_prefetch and torch.cuda.is_available():
            dataloader = PrefetchingDataLoader(
                dataloader,
                prefetch_batches=2
            )
        
        return dataloader
    
    @staticmethod
    def create_vgae_dataloader(dataset,
                              batch_size: int,
                              split: str = 'train',
                              num_workers: int = None,
                              use_cache: bool = True,
                              use_prefetch: bool = True) -> DataLoader:
        """
        Create optimized DataLoader for VGAE training.
        
        Args:
            dataset: PyTorch Geometric dataset
            batch_size: Batch size (typically smaller for VGAE)
            split: Dataset split
            num_workers: Number of workers
            use_cache: Whether to use caching
            use_prefetch: Whether to use prefetching
            
        Returns:
            Optimized DataLoader
        """
        # VGAE typically uses smaller batches due to memory requirements
        adjusted_batch_size = min(batch_size, 64)
        
        # Apply caching
        if use_cache:
            cache_dir = f"cache/vgae_{split}" if split != 'train' else None
            
            # Custom preprocessing for VGAE (if needed)
            def vgae_preprocess(data):
                # Any VGAE-specific preprocessing
                return data
            
            dataset = CachedGraphDataset(
                dataset,
                cache_dir=cache_dir,
                memory_cache_size=min(500, len(dataset) // 20),  # Smaller cache for VGAE
                preprocess_fn=vgae_preprocess
            )
        
        # Create optimized loader
        loader_factory = OptimizedGraphDataLoader(
            dataset,
            batch_size=adjusted_batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            memory_fraction=0.6  # More conservative for VGAE
        )
        
        dataloader = loader_factory.create_dataloader()
        
        # Add prefetching
        if use_prefetch and torch.cuda.is_available():
            dataloader = PrefetchingDataLoader(
                dataloader,
                prefetch_batches=1  # Smaller prefetch for VGAE
            )
        
        return dataloader


def optimize_dataloader_for_hardware() -> Dict[str, Any]:
    """
    Determine optimal DataLoader configuration based on hardware.
    
    Returns:
        Dictionary with recommended DataLoader settings
    """
    config = {}
    
    # CPU configuration
    cpu_count = os.cpu_count() or 4
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Determine optimal number of workers
    if memory_gb < 8:
        config['num_workers'] = min(2, cpu_count // 2)
    elif memory_gb < 32:
        config['num_workers'] = min(4, cpu_count // 2)
    else:
        config['num_workers'] = min(8, cpu_count // 2)
    
    # GPU configuration
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = device_props.total_memory / (1024**3)
        
        # Adjust settings based on GPU memory
        if gpu_memory_gb >= 40:  # A100, H100
            config.update({
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 4,
                'use_cache': True,
                'use_prefetch': True
            })
        elif gpu_memory_gb >= 16:  # V100, RTX 3080
            config.update({
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 2,
                'use_cache': True,
                'use_prefetch': True
            })
        else:  # Smaller GPUs
            config.update({
                'pin_memory': True,
                'persistent_workers': False,
                'prefetch_factor': 2,
                'use_cache': False,
                'use_prefetch': False
            })
    else:
        # CPU-only configuration
        config.update({
            'pin_memory': False,
            'persistent_workers': True,
            'prefetch_factor': 2,
            'use_cache': True,
            'use_prefetch': False
        })
    
    logger.info(f"Hardware-optimized DataLoader config: {config}")
    return config