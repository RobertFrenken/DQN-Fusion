"""
Unified Data Module System for CAN-Graph Training

Consolidates data loading, caching, and curriculum learning into a single module.

Components:
- CANGraphDataModule: Standard DataModule for normal training
- EnhancedCANGraphDataModule: Curriculum learning with hard mining
- AdaptiveGraphDataset: Dynamic sampling based on curriculum + difficulty
- CurriculumCallback: Lightning callback for curriculum management
- load_dataset(): Dataset loading with intelligent caching
- create_dataloaders(): Optimized dataloader creation

Replaces:
- src/training/can_graph_data.py
- src/training/enhanced_datamodule.py
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import lightning.pytorch as pl

from src.preprocessing.preprocessing import GraphDataset
from src.paths import PathResolver

logger = logging.getLogger(__name__)


# ============================================================================
# Standard DataModule
# ============================================================================

class CANGraphDataModule(pl.LightningDataModule):
    """
    Standard Lightning DataModule for CAN graph training.
    
    Used for normal training modes (GAT, VGAE, DQN).
    Provides efficient batch loading with PyTorch Geometric DataLoader.
    """
    
    def __init__(self, train_dataset, val_dataset, batch_size: int, num_workers: int = 8):
        """
        Initialize standard datamodule.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            num_workers: Number of dataloader workers
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

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


# ============================================================================
# Curriculum Learning Components
# ============================================================================

class AdaptiveGraphDataset(Dataset):
    """
    Dataset with dynamic sampling for curriculum learning.
    
    Progressively increases dataset difficulty using:
    1. Curriculum scheduling (normal:attack ratio)
    2. Hard sample mining (VGAE-based difficulty scores)
    """
    
    def __init__(
        self, 
        normal_graphs: List, 
        attack_graphs: List, 
        vgae_model=None, 
        current_epoch: int = 0, 
        total_epochs: int = 200
    ):
        """
        Initialize adaptive dataset.
        
        Args:
            normal_graphs: List of normal traffic graphs
            attack_graphs: List of attack traffic graphs
            vgae_model: Trained VGAE for difficulty scoring
            current_epoch: Current training epoch
            total_epochs: Total training epochs
        """
        self.normal_graphs = normal_graphs
        self.attack_graphs = attack_graphs
        self.vgae_model = vgae_model
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        
        # Curriculum parameters
        self.start_ratio = 1.0  # 1:1 (easy - balanced)
        self.end_ratio = 10.0   # 10:1 (realistic but manageable)
        
        # Hard mining parameters
        self.difficulty_percentile = 50.0  # Start with median difficulty
        self.difficulty_cache = {}  # Cache VGAE scores
        
        # Pre-compute all combinations for this epoch
        self._generate_epoch_samples()
    
    def get_max_dataset_size(self) -> int:
        """Calculate maximum expected dataset size at end of curriculum."""
        n_attacks = len(self.attack_graphs)
        max_normals = min(int(n_attacks * self.end_ratio), len(self.normal_graphs))
        return n_attacks + max_normals
    
    def _compute_curriculum_ratio(self) -> float:
        """Compute current normal:attack ratio using momentum curriculum."""
        if not hasattr(self, 'momentum_scheduler'):
            from src.training.momentum_curriculum import MomentumCurriculumScheduler
            self.momentum_scheduler = MomentumCurriculumScheduler(
                total_epochs=self.total_epochs,
                initial_ratio=self.start_ratio,
                target_ratio=self.end_ratio,
                momentum=0.9,
                confidence_threshold=0.75,
                warmup_epochs=max(10, int(self.total_epochs * 0.1))
            )
            
        # Use GAT confidence from callback if available
        confidence = getattr(self, 'last_confidence', 0.5)
        
        # Get momentum-based ratio
        ratio, metrics = self.momentum_scheduler.update_ratio(self.current_epoch, confidence)
        
        # Store metrics for logging
        self.curriculum_metrics = metrics
        
        return ratio
    
    def _get_difficulty_scores(self, graphs: List) -> np.ndarray:
        """
        Get VGAE reconstruction difficulty scores.
        
        Args:
            graphs: List of graphs to score
            
        Returns:
            Array of difficulty scores (higher = harder)
        """
        if self.vgae_model is None:
            raise RuntimeError(
                "VGAE model required for hard-mining difficulty scoring. "
                "Provide a trained VGAE model or disable VGAE-based mining."
            )
        
        scores = []
        self.vgae_model.eval()
        
        with torch.no_grad():
            for i, graph in enumerate(graphs):
                # Use cache if available
                graph_id = id(graph)
                if graph_id in self.difficulty_cache:
                    scores.append(self.difficulty_cache[graph_id])
                    continue
                
                try:
                    # Forward pass through VGAE
                    x, edge_index = graph.x, graph.edge_index
                    z = self.vgae_model.encode(x, edge_index)
                    adj_recon = self.vgae_model.decode(z, edge_index)
                    
                    # Compute reconstruction loss as difficulty
                    loss = self.vgae_model.recon_loss(adj_recon, graph.edge_index)
                    score = loss.item()
                    
                    # Cache the score
                    self.difficulty_cache[graph_id] = score
                    scores.append(score)
                    
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to compute VGAE difficulty score for graph {i}: {e}"
                    ) from e
        
        return np.array(scores)
    
    def _select_hard_normals(self, n_needed: int) -> List:
        """
        Select hard normal examples using VGAE reconstruction error.
        
        Args:
            n_needed: Number of normal samples needed
            
        Returns:
            List of selected hard normal graphs
        """
        if len(self.normal_graphs) <= n_needed:
            return self.normal_graphs
        
        # Get difficulty scores
        difficulty_scores = self._get_difficulty_scores(self.normal_graphs)
        
        # Select based on percentile threshold
        threshold = np.percentile(difficulty_scores, self.difficulty_percentile)
        hard_indices = np.where(difficulty_scores >= threshold)[0]
        
        if len(hard_indices) >= n_needed:
            # Sample from hard examples
            selected = np.random.choice(hard_indices, n_needed, replace=False)
        else:
            # Not enough hard examples, fill with random
            remaining = n_needed - len(hard_indices)
            easy_indices = np.where(difficulty_scores < threshold)[0]
            if len(easy_indices) > 0:
                extra = np.random.choice(
                    easy_indices, 
                    min(remaining, len(easy_indices)), 
                    replace=False
                )
                selected = np.concatenate([hard_indices, extra])
            else:
                selected = hard_indices
        
        return [self.normal_graphs[i] for i in selected]
    
    def _generate_epoch_samples(self):
        """Generate samples for current epoch based on curriculum + hard mining."""
        current_ratio = self._compute_curriculum_ratio()
        
        # Determine batch composition
        n_attacks = len(self.attack_graphs)
        n_normals_needed = min(int(n_attacks * current_ratio), len(self.normal_graphs))
        
        # Select hard normals
        selected_normals = self._select_hard_normals(n_normals_needed)
        
        # Combine and shuffle
        self.epoch_samples = self.attack_graphs + selected_normals
        np.random.shuffle(self.epoch_samples)
        
        logger.info(
            f"🌊 Epoch {self.current_epoch}: Momentum Curriculum - "
            f"Ratio {current_ratio:.3f}:1 ({len(selected_normals)} normals, "
            f"{len(self.attack_graphs)} attacks)"
        )
        
        # Log momentum metrics if available
        if hasattr(self, 'curriculum_metrics'):
            metrics = self.curriculum_metrics
            momentum = metrics.get('momentum_accumulator', 0.0)
            progress = metrics.get('progress_signal', 0.0)
            logger.info(f"   Momentum: {momentum:.3f}, Progress Signal: {progress:+.3f}")
    
    def update_epoch(self, epoch: int, gat_confidence: float = None):
        """
        Update epoch and difficulty based on GAT performance.
        
        Args:
            epoch: New epoch number
            gat_confidence: GAT model confidence on normal samples
        """
        self.current_epoch = epoch
        
        # Store confidence for momentum curriculum
        if gat_confidence is not None:
            self.last_confidence = gat_confidence
        
        # Adaptive difficulty based on GAT confidence
        if gat_confidence is not None:
            if gat_confidence < 0.1:  # Too confident on normals
                self.difficulty_percentile = min(90.0, self.difficulty_percentile + 5.0)
            elif gat_confidence > 0.3:  # Not confident enough
                self.difficulty_percentile = max(10.0, self.difficulty_percentile - 5.0)
        
        # Regenerate samples for new epoch
        old_size = len(self.epoch_samples) if hasattr(self, 'epoch_samples') else 0
        self._generate_epoch_samples()
        new_size = len(self.epoch_samples)
        
        # Check if dataset size changed significantly
        recalc_threshold = getattr(self, 'recalc_threshold', 2.0)
        if old_size > 0 and new_size > old_size * recalc_threshold:
            logger.warning(
                f"⚠️  Dataset size increased significantly: {old_size} -> {new_size} "
                f"({new_size/old_size:.1f}x > {recalc_threshold}x threshold). "
                "May need batch size recalculation."
            )
            self.needs_batch_size_recalc = True
        else:
            self.needs_batch_size_recalc = False
    
    def __len__(self):
        return len(self.epoch_samples)
    
    def __getitem__(self, idx):
        return self.epoch_samples[idx]


class EnhancedCANGraphDataModule(pl.LightningDataModule):
    """
    Enhanced DataModule with curriculum learning and hard mining.
    
    Used for curriculum training mode with progressive difficulty increase.
    Integrates AdaptiveGraphDataset for dynamic sampling.
    """
    
    def __init__(
        self, 
        train_normal: List, 
        train_attack: List,
        val_normal: List, 
        val_attack: List,
        vgae_model=None, 
        batch_size: int = 32, 
        num_workers: int = 4,
        total_epochs: int = 200
    ):
        """
        Initialize enhanced datamodule.
        
        Args:
            train_normal: Normal training graphs
            train_attack: Attack training graphs
            val_normal: Normal validation graphs
            val_attack: Attack validation graphs
            vgae_model: Trained VGAE for difficulty scoring
            batch_size: Batch size for training
            num_workers: Number of dataloader workers
            total_epochs: Total training epochs
        """
        super().__init__()
        self.train_normal = train_normal
        self.train_attack = train_attack
        self.val_normal = val_normal  
        self.val_attack = val_attack
        self.vgae_model = vgae_model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_epochs = total_epochs
        
        # Create adaptive datasets
        self.train_dataset = AdaptiveGraphDataset(
            train_normal, train_attack, vgae_model, 0, total_epochs
        )
        
        # Validation remains balanced for consistent evaluation
        val_samples = val_attack + val_normal[:len(val_attack)*5]  # 5:1 ratio
        self.val_dataset = GraphDataset(val_samples)
    
    def get_conservative_batch_size(self, target_batch_size: int = 64) -> int:
        """
        Calculate conservative batch size for curriculum learning.
        
        Accounts for dataset growth during training.
        
        Args:
            target_batch_size: Desired batch size
            
        Returns:
            Conservative batch size
        """
        max_size = self.train_dataset.get_max_dataset_size()
        current_size = len(self.train_dataset)
        
        # If current size is much smaller than max, reduce batch size proportionally
        if current_size > 0 and max_size > current_size * 2:
            conservative_ratio = min(0.5, current_size / max_size)
            conservative_batch_size = max(8, int(target_batch_size * conservative_ratio))
            logger.info(
                f"🛡️ Conservative batch sizing: {target_batch_size} -> "
                f"{conservative_batch_size} (current: {current_size}, "
                f"max expected: {max_size})"
            )
            return conservative_batch_size
        
        return target_batch_size
    
    def create_max_size_dataset_for_tuning(self):
        """
        Create temporary dataset at maximum size for batch size optimization.
        
        Returns:
            Original state tuple (epoch, ratio) or None
        """
        max_size = self.train_dataset.get_max_dataset_size()
        current_size = len(self.train_dataset)
        
        if max_size <= current_size:
            return None  # Already at max size
            
        # Save original state
        original_epoch = self.train_dataset.current_epoch
        original_ratio = self.train_dataset.end_ratio
        
        # Set to end curriculum state (maximum dataset size)
        self.train_dataset.current_epoch = self.train_dataset.total_epochs - 1
        self.train_dataset._generate_epoch_samples()
        
        logger.info(
            f"🔧 Created max-size dataset for batch optimization: "
            f"{len(self.train_dataset)} samples"
        )
        
        return original_epoch, original_ratio
    
    def restore_dataset_after_tuning(self, original_state):
        """
        Restore dataset to original state after optimization.
        
        Args:
            original_state: Tuple from create_max_size_dataset_for_tuning()
        """
        if original_state:
            original_epoch, original_ratio = original_state
            self.train_dataset.current_epoch = original_epoch
            self.train_dataset._generate_epoch_samples()
            logger.info(
                f"🔄 Restored dataset to original state: "
                f"{len(self.train_dataset)} samples"
            )
    
    def setup(self, stage: str = None):
        """Setup datasets (already done in __init__)."""
        pass
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_graphs
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_graphs
        )
    
    def _collate_graphs(self, batch):
        """Collate function to batch graph data."""
        return Batch.from_data_list(batch)
    
    def update_training_epoch(self, epoch: int, gat_confidence: float = None):
        """Update training dataset for new epoch."""
        self.train_dataset.update_epoch(epoch, gat_confidence)


class CurriculumCallback(pl.Callback):
    """
    Lightning callback to manage curriculum learning and hard mining.
    
    Handles:
    - Curriculum schedule updates
    - Confidence tracking
    - Memory preservation (EWC)
    - Dynamic batch size adjustment
    """
    
    def __init__(self):
        super().__init__()
        self.normal_confidences = []
        self.balanced_phase_buffer = []  # Store early balanced examples
        self.ewc_initialized = False
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Update curriculum at start of each epoch."""
        datamodule = trainer.datamodule
        
        # Store samples from balanced phase for replay
        current_epoch = trainer.current_epoch
        if current_epoch < trainer.max_epochs * 0.2:  # First 20% = balanced phase
            train_loader = datamodule.train_dataloader()
            if len(self.balanced_phase_buffer) < 1000:  # Limit buffer size
                batch = next(iter(train_loader))
                self.balanced_phase_buffer.extend(batch.to_data_list()[:10])
        
        # Initialize EWC after balanced phase
        elif not self.ewc_initialized and current_epoch == int(trainer.max_epochs * 0.2):
            self._initialize_memory_preservation(trainer, pl_module)
            self.ewc_initialized = True
        
        # Calculate GAT confidence on normals
        avg_confidence = None
        if len(self.normal_confidences) > 0:
            avg_confidence = np.mean(self.normal_confidences[-10:])
        
        # Update curriculum
        datamodule.update_training_epoch(trainer.current_epoch, avg_confidence)
        
        # Log metrics
        if hasattr(datamodule.train_dataset, 'curriculum_metrics'):
            metrics = datamodule.train_dataset.curriculum_metrics
            current_ratio = datamodule.train_dataset._compute_curriculum_ratio()
            normal_percentage = (current_ratio / (1 + current_ratio)) * 100
            
            pl_module.log('curriculum/normal_ratio', current_ratio)
            pl_module.log('curriculum/normal_percentage', normal_percentage)
            pl_module.log('curriculum/momentum', metrics.get('momentum_accumulator', 0.0))
            pl_module.log('curriculum/progress_signal', metrics.get('progress_signal', 0.0))
        
        if avg_confidence is not None:
            pl_module.log('curriculum/normal_confidence', avg_confidence)
    
    def _initialize_memory_preservation(self, trainer, pl_module):
        """Initialize EWC for memory preservation."""
        logger.info(f"🧠 Initializing memory preservation at epoch {trainer.current_epoch}")
        
        if not hasattr(pl_module, 'ewc'):
            from src.training.memory_preserving_curriculum import ElasticWeightConsolidation
            pl_module.ewc = ElasticWeightConsolidation(pl_module.model)
            
            # Compute Fisher Information on balanced examples
            if len(self.balanced_phase_buffer) > 0:
                def collate_fn(batch):
                    return Batch.from_data_list(batch)
                
                balanced_loader = TorchDataLoader(
                    self.balanced_phase_buffer, 
                    batch_size=32, 
                    collate_fn=collate_fn
                )
                
                pl_module.ewc.compute_fisher_information(
                    balanced_loader, 
                    pl_module.device
                )
                logger.info("✅ EWC initialized with balanced examples")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Track confidence and add EWC loss."""
        if batch_idx % 10 != 0:  # Sample every 10 batches
            return
            
        # Track normal confidence
        with torch.no_grad():
            logits = pl_module(batch)
            probs = torch.sigmoid(logits)
            
            normal_mask = batch.y == 0
            if normal_mask.sum() > 0:
                normal_confidence = probs[normal_mask].mean().item()
                self.normal_confidences.append(normal_confidence)
        
        # Add EWC loss after balanced phase
        if hasattr(pl_module, 'ewc') and trainer.current_epoch > trainer.max_epochs * 0.3:
            ewc_loss = pl_module.ewc.compute_ewc_loss()
            
            if 'loss' in outputs:
                outputs['loss'] += ewc_loss * 0.1
            
            pl_module.log('ewc_loss', ewc_loss)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log curriculum statistics."""
        datamodule = trainer.datamodule
        current_ratio = datamodule.train_dataset._compute_curriculum_ratio()
        difficulty_percentile = datamodule.train_dataset.difficulty_percentile
        
        pl_module.log('curriculum/ratio', current_ratio)
        pl_module.log('curriculum/difficulty_percentile', difficulty_percentile)
        
        if len(self.normal_confidences) > 0:
            avg_confidence = np.mean(self.normal_confidences)
            pl_module.log('curriculum/normal_confidence', avg_confidence)


# ============================================================================
# Dataset Loading and Caching
# ============================================================================

def load_dataset(
    dataset_name: str, 
    config, 
    force_rebuild_cache: bool = False
) -> Tuple[Dataset, Dataset, int]:
    """
    Load and prepare dataset with intelligent caching.
    
    Features:
    - Automatic path resolution (config, env vars, common locations)
    - Disk caching for processed graphs
    - Cache validation (size checks)
    - Train/val split
    
    Args:
        dataset_name: Dataset name (hcrl_sa, set_01, etc.)
        config: Configuration object
        force_rebuild_cache: Force rebuild cached data
        
    Returns:
        Tuple of (train_dataset, val_dataset, num_unique_ids)
    """
    # Use unified path resolver
    path_resolver = PathResolver(config)
    dataset_path = path_resolver.resolve_dataset_path(dataset_name)
    
    # Check cache
    cache_enabled, cache_file, id_mapping_file = path_resolver.get_cache_paths(dataset_name)
    
    graphs, id_mapping = None, None
    
    if cache_enabled and not force_rebuild_cache:
        graphs, id_mapping = _load_cached_data(
            cache_file, 
            id_mapping_file, 
            dataset_name
        )
    
    # Process from scratch if needed
    if graphs is None or id_mapping is None:
        graphs, id_mapping = _process_dataset_from_scratch(
            dataset_path,
            dataset_name,
            cache_enabled,
            cache_file,
            id_mapping_file,
            force_rebuild_cache
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


def create_dataloaders(
    train_dataset, 
    val_dataset, 
    batch_size: int, 
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized PyTorch Geometric dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
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
    
    logger.info(f"Created dataloaders with {num_workers} workers")
    return train_loader, val_loader


# ============================================================================
# Internal Helper Functions
# ============================================================================

# Path resolution functions moved to src.paths.PathResolver
# These are kept as thin wrappers for backward compatibility


def _load_cached_data(cache_file, id_mapping_file, dataset_name):
    """Load cached graphs and ID mapping."""
    if not (cache_file.exists() and id_mapping_file.exists()):
        return None, None
    
    try:
        import pickle
        graphs = torch.load(cache_file)
        with open(id_mapping_file, 'rb') as f:
            id_mapping = pickle.load(f)
        
        logger.info(f"Loaded {len(graphs)} cached graphs with {len(id_mapping)} unique IDs")
        
        # Validate cache size
        expected_sizes = {
            'set_01': 300000, 'set_02': 400000, 'set_03': 330000, 'set_04': 240000,
            'hcrl_sa': 18000, 'hcrl_ch': 290000
        }
        
        if dataset_name in expected_sizes:
            expected = expected_sizes[dataset_name]
            actual = len(graphs)
            
            if actual < expected * 0.1:
                logger.warning(
                    f"🚨 CACHE ISSUE: Only {actual} graphs found, expected ~{expected}. "
                    "Rebuilding cache."
                )
                return None, None
            elif actual < expected * 0.5:
                logger.warning(
                    f"⚠️  Cache has fewer graphs than expected: {actual} vs ~{expected}. "
                    "Use --force-rebuild to recreate."
                )
            else:
                logger.info(f"✅ Cache size looks good: {actual} graphs (expected ~{expected})")
        
        return graphs, id_mapping
        
    except Exception as e:
        logger.warning(f"Failed to load cached data: {e}. Processing from scratch.")
        return None, None


def _process_dataset_from_scratch(
    dataset_path, 
    dataset_name, 
    cache_enabled, 
    cache_file, 
    id_mapping_file,
    force_rebuild
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
    
    # Save cache
    if cache_enabled:
        import pickle
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving processed data to cache: {cache_file}")
        torch.save(graphs, cache_file)
        with open(id_mapping_file, 'wb') as f:
            pickle.dump(id_mapping, f)
    
    return graphs, id_mapping
