"""
Data Module System for CAN-Graph Training

Components:
- CANGraphDataModule: Standard DataModule for training
- load_dataset(): Dataset loading with intelligent caching
"""

import json
import os
import glob
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl

from src.preprocessing.preprocessing import (
    GraphDataset,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STRIDE,
    NODE_FEATURE_COUNT,
    EDGE_FEATURE_COUNT,
)

logger = logging.getLogger(__name__)

# Default multiprocessing start method for DataLoader workers.
# 'spawn' is required when CUDA is initialized before DataLoader workers
# are created (e.g. difficulty scoring in curriculum stage). 'fork' will
# crash with "Cannot re-initialize CUDA in forked subprocess".
DEFAULT_MP_CONTEXT = "spawn"


# ============================================================================
# Standard DataModule
# ============================================================================

class CANGraphDataModule(pl.LightningDataModule):
    """
    Standard Lightning DataModule for CAN graph training.

    Used for normal training modes (GAT, VGAE, DQN).
    Provides efficient batch loading with PyTorch Geometric DataLoader.
    """

    def __init__(self, train_dataset, val_dataset, batch_size: int, num_workers: int = 8,
                 mp_start_method: str = DEFAULT_MP_CONTEXT):
        """
        Initialize standard datamodule.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            num_workers: Number of dataloader workers
            mp_start_method: Multiprocessing start method ('spawn' for CUDA safety)
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mp_start_method = mp_start_method

    def train_dataloader(self):
        nw = self.num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=nw > 0,
            multiprocessing_context=self.mp_start_method if nw > 0 else None,
        )

    def val_dataloader(self):
        nw = self.num_workers
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=nw > 0,
            multiprocessing_context=self.mp_start_method if nw > 0 else None,
        )


# ============================================================================
# Dataset Loading and Caching
# ============================================================================

def load_dataset(
    dataset_name: str,
    dataset_path: Path,
    cache_dir_path: Path,
    force_rebuild_cache: bool = False,
):
    """
    Load and prepare dataset with intelligent caching.

    Args:
        dataset_name: Dataset name (hcrl_sa, set_01, etc.)
        dataset_path: Path to the raw dataset directory
        cache_dir_path: Path to the cache directory for processed graphs
        force_rebuild_cache: Force rebuild cached data

    Returns:
        Tuple of (train_dataset, val_dataset, num_unique_ids)
    """
    cache_file = cache_dir_path / "processed_graphs.pt"
    id_mapping_file = cache_dir_path / "id_mapping.pkl"
    
    graphs, id_mapping = None, None

    if not force_rebuild_cache:
        graphs, id_mapping = _load_cached_data(
            cache_file,
            id_mapping_file,
            dataset_name,
        )

    # Process from scratch if needed
    if graphs is None or id_mapping is None:
        graphs, id_mapping = _process_dataset_from_scratch(
            dataset_path,
            dataset_name,
            cache_file,
            id_mapping_file,
            force_rebuild_cache,
        )
    
    # Create dataset and split
    dataset = GraphDataset(graphs)
    logger.info(f"📊 Created dataset with {len(dataset)} total graphs")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    logger.info(
        f"📊 Dataset split: {len(train_dataset)} training, "
        f"{len(val_dataset)} validation"
    )
    
    num_ids = len(id_mapping) if id_mapping else 1000
    return train_dataset, val_dataset, num_ids


# ============================================================================
# Internal Helper Functions
# ============================================================================


def _load_cached_data(cache_file, id_mapping_file, dataset_name):
    """Load cached graphs and ID mapping with robust error handling.
    
    Note: Uses weights_only=False to support PyTorch Geometric Data objects.
    This is safe for our own cached data but should not be used with untrusted files.
    TODO: Migrate to safetensors format for improved security.
    """
    if not (cache_file.exists() and id_mapping_file.exists()):
        return None, None
    
    try:
        import pickle
        # Note: weights_only=False is required for PyG Data objects
        # This is safe for our own cached data but be cautious with untrusted files
        try:
            graphs = torch.load(cache_file, map_location='cpu', weights_only=False)
        except Exception as e:
            logger.warning(f"Cache load failed (possibly corrupted): {e}")
            return None, None
            
        with open(id_mapping_file, 'rb') as f:
            id_mapping = pickle.load(f)
        
        # Validate loaded data
        if not isinstance(graphs, list) or not isinstance(id_mapping, dict):
            logger.warning(f"Invalid cache format. Expected list of graphs and dict mapping.")
            return None, None
        
        logger.info(f"Loaded {len(graphs)} cached graphs with {len(id_mapping)} unique IDs")

        # Validate cache using metadata if available
        metadata_file = cache_file.parent / "cache_metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                expected = metadata.get("num_graphs", 0)
                actual = len(graphs)
                version = metadata.get("preprocessing_version", "unknown")

                if expected > 0 and actual < expected * 0.1:
                    logger.warning(
                        f"CACHE ISSUE: Only {actual} graphs found, expected {expected} "
                        f"(preprocessing v{version}). Rebuilding cache."
                    )
                    return None, None
                elif expected > 0 and actual < expected * 0.5:
                    logger.warning(
                        f"Cache has fewer graphs than expected: {actual} vs {expected}. "
                        "Use --force-rebuild to recreate."
                    )
                else:
                    logger.info(f"Cache validated: {actual} graphs (expected {expected}, v{version})")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Cache metadata unreadable: {e}. Proceeding with loaded data.")
        else:
            logger.info(f"No cache metadata found. Loaded {len(graphs)} graphs.")

        return graphs, id_mapping
        
    except (pickle.UnpicklingError, AttributeError, EOFError) as e:
        logger.warning(f"Cache file corrupted ({type(e).__name__}). Deleting and rebuilding.")
        try:
            cache_file.unlink(missing_ok=True)
            id_mapping_file.unlink(missing_ok=True)
        except:
            pass
        return None, None
    except Exception as e:
        logger.warning(f"Failed to load cached data: {e}. Processing from scratch.")
        return None, None


def _process_dataset_from_scratch(
    dataset_path,
    dataset_name,
    cache_file,
    id_mapping_file,
    force_rebuild,
):
    """Process dataset from CSV files."""
    logger.info(
        f"Processing dataset: "
        f"{'forced rebuild' if force_rebuild else 'processing from scratch'}..."
    )
    logger.info(f"Dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Find CSV files
    csv_files = []
    for train_folder in ['train_01_attack_free', 'train_02_with_attacks', 'train_*']:
        pattern = os.path.join(dataset_path, train_folder, '*.csv')
        csv_files.extend(glob.glob(pattern))
    
    if not csv_files:
        csv_files = glob.glob(
            os.path.join(dataset_path, '**', '*train*.csv'), 
            recursive=True
        )
    
    logger.info(f"Found {len(csv_files)} CSV files in {dataset_path}")
    
    if len(csv_files) == 0:
        logger.error(f"🚨 NO CSV FILES FOUND in {dataset_path}!")
        all_files = glob.glob(os.path.join(dataset_path, '**', '*.csv'), recursive=True)[:20]
        for f in all_files:
            logger.error(f"  - {f}")
        raise FileNotFoundError(f"No train CSV files found in {dataset_path}")
    
    logger.info("🔄 Starting graph creation from CSV files...")
    
    from src.preprocessing.preprocessing import graph_creation
    graphs, id_mapping = graph_creation(
        dataset_path, 
        'train_', 
        return_id_mapping=True, 
        verbose=True
    )
    
    # Save cache atomically
    import pickle
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving processed data to cache: {cache_file}")

    temp_cache = cache_file.with_suffix('.tmp')
    temp_mapping = id_mapping_file.with_suffix('.tmp')

    try:
        torch.save(graphs, temp_cache, pickle_protocol=4)
        with open(temp_mapping, 'wb') as f:
            pickle.dump(id_mapping, f, protocol=4)

        temp_cache.rename(cache_file)
        temp_mapping.rename(id_mapping_file)
        logger.info(f"Cache saved: {len(graphs)} graphs")

        # Write cache metadata for validation on future loads
        _write_cache_metadata(
            cache_file.parent, dataset_name, graphs, id_mapping, csv_files
        )
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")
        temp_cache.unlink(missing_ok=True)
        temp_mapping.unlink(missing_ok=True)

    return graphs, id_mapping


def _write_cache_metadata(cache_dir, dataset_name, graphs, id_mapping, csv_files):
    """Write cache_metadata.json alongside processed cache files."""
    import torch_geometric

    metadata = {
        "dataset": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "window_size": DEFAULT_WINDOW_SIZE,
        "stride": DEFAULT_STRIDE,
        "num_graphs": len(graphs),
        "num_unique_ids": len(id_mapping) if id_mapping else 0,
        "node_feature_dim": NODE_FEATURE_COUNT,
        "edge_feature_dim": EDGE_FEATURE_COUNT,
        "source_csv_count": len(csv_files),
        "preprocessing_version": "1.0",
        "torch_version": torch.__version__,
        "pyg_version": torch_geometric.__version__,
    }
    metadata_file = Path(cache_dir) / "cache_metadata.json"
    try:
        metadata_file.write_text(json.dumps(metadata, indent=2))
        logger.info(f"Cache metadata written to {metadata_file}")
    except Exception as e:
        logger.warning(f"Failed to write cache metadata: {e}")
