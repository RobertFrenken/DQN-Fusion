"""
CAN-Graph Data Handling
Implements Lightning DataModule, dataset loading, and dataloader creation for CAN graph training.
"""

import os
import torch
from pathlib import Path
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl
from src.preprocessing.preprocessing import GraphDataset

class CANGraphDataModule(pl.LightningDataModule):
    """Lightning DataModule for efficient batch size tuning."""
    def __init__(self, train_dataset, val_dataset, batch_size: int):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = min(os.cpu_count() or 1, 8)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )

def load_dataset(dataset_name: str, config, force_rebuild_cache: bool = False):
    """Load and prepare dataset"""
    import logging
    logger = logging.getLogger(__name__)
    from pathlib import Path
    import os
    import torch
    # Resolve dataset path with fallbacks so runs are more robust on different systems.
    dataset_path = None
    # 1) Explicit config value
    if hasattr(config.dataset, 'data_path') and config.dataset.data_path:
        dataset_path = str(config.dataset.data_path)

    # 2) CLI override via env var commonly used by job managers
    if not dataset_path:
        for env_var in ('CAN_DATA_PATH', 'DATA_PATH', 'EXPERIMENT_DATA_PATH'):
            if os.environ.get(env_var):
                dataset_path = os.environ.get(env_var)
                break

    # 3) Try sensible locations inside the project (datasets/ or data/automotive)
    project_root = Path(__file__).resolve().parents[2]
    candidates = []
    if dataset_path:
        candidates.append(Path(dataset_path))
    # common dataset layout used by this repo
    candidates.append(project_root / 'datasets' / f'can-train-and-test-v1.5' / dataset_name)
    candidates.append(project_root / 'data' / 'automotive' / dataset_name)
    # also accept dataset_name as a direct folder under datasets
    candidates.append(project_root / 'datasets' / dataset_name)

    resolved = None
    for c in candidates:
        try:
            if c and c.exists():
                resolved = str(c)
                break
        except Exception:
            continue

    if resolved is None:
        # Provide helpful guidance rather than a terse traceback
        msg_lines = [
            f"Dataset not found. Tried the following locations:",
        ]
        for c in candidates:
            msg_lines.append(f"  - {c}")
        msg_lines.append("\nSuggestions:")
        msg_lines.append("  - Set `config.dataset.data_path` in your preset to the absolute dataset folder.")
        msg_lines.append("  - Or run the script with `--data-path /path/to/dataset` to override the location.")
        msg_lines.append("  - Or set the environment variable CAN_DATA_PATH or DATA_PATH to point to the dataset.")
        full_msg = "\n".join(msg_lines)
        raise FileNotFoundError(full_msg)

    dataset_path = resolved
    if hasattr(config.dataset, 'get'):
        cache_enabled = config.dataset.get('preprocessing', {}).get('cache_processed_data', True)
        cache_dir = config.dataset.get('cache_dir', f"datasets/cache/{dataset_name}")
    else:
        cache_enabled = getattr(config.dataset, 'cache_processed_data', True)
        cache_dir = getattr(config.dataset, 'cache_dir', None)
        if cache_dir is None:
            cache_dir = f"datasets/cache/{dataset_name}"
    cache_file = Path(cache_dir) / "processed_graphs.pt"
    id_mapping_file = Path(cache_dir) / "id_mapping.pkl"
    if cache_enabled and cache_file.exists() and id_mapping_file.exists() and not force_rebuild_cache:
        try:
            import pickle
            graphs = torch.load(cache_file)
            with open(id_mapping_file, 'rb') as f:
                id_mapping = pickle.load(f)
            logger.info(f"Loaded {len(graphs)} cached graphs with {len(id_mapping)} unique IDs")
            expected_sizes = {
                'set_01': 300000, 'set_02': 400000, 'set_03': 330000, 'set_04': 240000,
                'hcrl_sa': 18000, 'hcrl_ch': 290000
            }
            if dataset_name in expected_sizes:
                expected = expected_sizes[dataset_name]
                actual = len(graphs)
                if actual < expected * 0.1:
                    logger.warning(f"🚨 CACHE ISSUE DETECTED: Only {actual} graphs found, expected ~{expected}")
                    logger.warning(f"Cache appears corrupted or incomplete. Rebuilding from scratch.")
                    graphs, id_mapping = None, None
                elif actual < expected * 0.5:
                    logger.warning(f"⚠️  Cache has fewer graphs than expected: {actual} vs ~{expected}")
                    logger.warning(f"This might be a debug/test cache. Use --force-rebuild to recreate.")
                else:
                    logger.info(f"✅ Cache size looks good: {actual} graphs (expected ~{expected})")
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}. Processing from scratch.")
            graphs, id_mapping = None, None
    else:
        graphs, id_mapping = None, None
    if graphs is None or id_mapping is None:
        logger.info(f"Processing dataset: {'forced rebuild' if force_rebuild_cache else 'processing dataset from scratch'}...")
        logger.info(f"Dataset path: {dataset_path}")
        if os.path.exists(dataset_path):
            import glob
            csv_files = []
            for train_folder in ['train_01_attack_free', 'train_02_with_attacks', 'train_*']:
                pattern = os.path.join(dataset_path, train_folder, '*.csv')
                csv_files.extend(glob.glob(pattern))
            if not csv_files:
                csv_files = glob.glob(os.path.join(dataset_path, '**', '*train*.csv'), recursive=True)
            logger.info(f"Found {len(csv_files)} CSV files in {dataset_path}")
            if len(csv_files) == 0:
                logger.error(f"🚨 NO CSV FILES FOUND in {dataset_path}!")
                logger.error(f"Available files:")
                all_files = glob.glob(os.path.join(dataset_path, '**', '*.csv'), recursive=True)[:20]
                for f in all_files:
                    logger.error(f"  - {f}")
                train_folders = glob.glob(os.path.join(dataset_path, 'train*'))
                if train_folders:
                    logger.error(f"Found training folders: {train_folders}")
                    for folder in train_folders:
                        folder_files = glob.glob(os.path.join(folder, '*.csv'))
                        logger.error(f"  {folder}: {len(folder_files)} CSV files")
                raise FileNotFoundError(f"No train CSV files found in {dataset_path}")
            elif len(csv_files) < 50:
                logger.info(f"CSV files found: {csv_files[:10]}")
        else:
            logger.error(f"Dataset path does not exist: {dataset_path}")
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        logger.info("🔄 Starting graph creation from CSV files (this may take several minutes for large datasets)...")
        from src.preprocessing.preprocessing import graph_creation, GraphDataset
        graphs, id_mapping = graph_creation(dataset_path, 'train_', return_id_mapping=True, verbose=True)
        if cache_enabled:
            import pickle
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving processed data to cache: {cache_file}")
            torch.save(graphs, cache_file)
            with open(id_mapping_file, 'wb') as f:
                pickle.dump(id_mapping, f)
    from src.preprocessing.preprocessing import GraphDataset
    dataset = GraphDataset(graphs)
    logger.info(f"📊 Created dataset with {len(dataset)} total graphs")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    logger.info(f"📊 Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation")
    num_ids = len(id_mapping) if id_mapping else 1000
    return train_dataset, val_dataset, num_ids

def create_dataloaders(train_dataset, val_dataset, batch_size: int):
    """Create optimized dataloaders - standalone function."""
    import os
    from torch_geometric.loader import DataLoader
    num_workers = min(os.cpu_count() or 1, 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Created dataloaders with {num_workers} workers")
    return train_loader, val_loader
