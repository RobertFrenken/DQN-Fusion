"""
Enhanced DataModule integrating Dynamic Hard Mining + Curriculum Learning
with Lightning training loop.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
from torch_geometric.data import Batch

from src.preprocessing.preprocessing import GraphDataset


class AdaptiveGraphDataset(Dataset):
    """Dataset that dynamically samples based on curriculum + hard mining."""
    
    def __init__(self, normal_graphs: List, attack_graphs: List, 
                 vgae_model=None, current_epoch: int = 0, total_epochs: int = 200):
        self.normal_graphs = normal_graphs
        self.attack_graphs = attack_graphs
        self.vgae_model = vgae_model
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        
        # Curriculum parameters
        self.start_ratio = 1.0  # 1:1 (easy)
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
                initial_ratio=self.start_ratio,  # Use configured start ratio
                target_ratio=self.end_ratio,     # Use configured end ratio
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
        """Get VGAE reconstruction difficulty scores."""
        if self.vgae_model is None:
            # Fallback to random if no VGAE
            return np.random.random(len(graphs))
        
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
                    # Fallback for problematic graphs
                    scores.append(0.5)
        
        return np.array(scores)
    
    def _select_hard_normals(self, n_needed: int) -> List:
        """Select hard normal examples using VGAE reconstruction error."""
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
                extra = np.random.choice(easy_indices, min(remaining, len(easy_indices)), 
                                       replace=False)
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
        
        print(f"🌊 Epoch {self.current_epoch}: Momentum Curriculum - Ratio {current_ratio:.3f}:1 "
              f"({len(selected_normals)} normals, {len(self.attack_graphs)} attacks)")
        
        # Log momentum metrics if available
        if hasattr(self, 'curriculum_metrics'):
            metrics = self.curriculum_metrics
            momentum = metrics.get('momentum_accumulator', 0.0)
            progress = metrics.get('progress_signal', 0.0)
            print(f"   Momentum: {momentum:.3f}, Progress Signal: {progress:+.3f}")
    
    def update_epoch(self, epoch: int, gat_confidence: float = None):
        """Update epoch and difficulty based on GAT performance."""
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
        
        # Check if dataset size changed significantly (configurable threshold)
        recalc_threshold = getattr(self, 'recalc_threshold', 2.0)  # Default 2x growth
        if old_size > 0 and new_size > old_size * recalc_threshold:
            print(f"⚠️  Dataset size increased significantly: {old_size} -> {new_size} "
                  f"({new_size/old_size:.1f}x > {recalc_threshold}x threshold). May need batch size recalculation.")
            self.needs_batch_size_recalc = True
        else:
            self.needs_batch_size_recalc = False
    
    def __len__(self):
        return len(self.epoch_samples)
    
    def __getitem__(self, idx):
        return self.epoch_samples[idx]


class EnhancedCANGraphDataModule(pl.LightningDataModule):
    """Enhanced DataModule with curriculum learning and hard mining."""
    
    def __init__(self, train_normal: List, train_attack: List,
                 val_normal: List, val_attack: List,
                 vgae_model=None, batch_size: int = 32, num_workers: int = 4,
                 total_epochs: int = 200):
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
        """Calculate conservative batch size based on maximum expected dataset size."""
        max_size = self.train_dataset.get_max_dataset_size()
        current_size = len(self.train_dataset)
        
        # If current size is much smaller than max, reduce batch size proportionally
        if current_size > 0 and max_size > current_size * 2:
            # Conservative scaling: use smaller batch size to account for growth
            conservative_ratio = min(0.5, current_size / max_size)
            conservative_batch_size = max(8, int(target_batch_size * conservative_ratio))
            print(f"🛡️ Conservative batch sizing: {target_batch_size} -> {conservative_batch_size} "
                  f"(current: {current_size}, max expected: {max_size})")
            return conservative_batch_size
        
        return target_batch_size
    
    def create_max_size_dataset_for_tuning(self):
        """Create a temporary dataset at maximum expected size for batch size optimization."""
        max_size = self.train_dataset.get_max_dataset_size()
        current_size = len(self.train_dataset)
        
        if max_size <= current_size:
            return  # Already at max size
            
        # Temporarily set the dataset to maximum size for tuning
        original_epoch = self.train_dataset.current_epoch
        original_ratio = self.train_dataset.end_ratio
        
        # Set to end curriculum state (maximum dataset size)
        self.train_dataset.current_epoch = self.train_dataset.total_epochs - 1
        self.train_dataset._generate_epoch_samples()
        
        print(f"🔧 Created max-size dataset for batch optimization: {len(self.train_dataset)} samples")
        
        return original_epoch, original_ratio
    
    def restore_dataset_after_tuning(self, original_state):
        """Restore dataset to original state after batch size optimization."""
        if original_state:
            original_epoch, original_ratio = original_state
            self.train_dataset.current_epoch = original_epoch
            self.train_dataset._generate_epoch_samples()
            print(f"🔄 Restored dataset to original state: {len(self.train_dataset)} samples")
    
    def recalculate_batch_size_if_needed(self, trainer):
        """Dynamically recalculate batch size if dataset grew significantly."""
        if not hasattr(self.train_dataset, 'needs_batch_size_recalc') or not self.train_dataset.needs_batch_size_recalc:
            return False
        
        print(f"🔄 Recalculating batch size due to significant dataset growth...")
        
        try:
            # Create a temporary trainer for batch size optimization
            from lightning.pytorch.tuner import Tuner
            temp_trainer = pl.Trainer(
                accelerator=trainer.accelerator,
                devices=1,  # Use single device for tuning
                precision='32-true',
                max_epochs=1,
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=False
            )
            
            # Get the model from current trainer
            model = trainer.lightning_module
            old_batch_size = self.batch_size
            
            # Run batch size optimization with current dataset size
            tuner = Tuner(temp_trainer)
            tuner.scale_batch_size(
                model,
                datamodule=self,
                mode="power",  # Efficient search
                steps_per_trial=3,
                init_val=max(8, old_batch_size // 4),  # Start with smaller size
                max_trials=10
            )
            
            new_batch_size = model.batch_size
            self.batch_size = new_batch_size
            
            print(f"✅ Batch size recalculated: {old_batch_size} -> {new_batch_size}")
            
            # Reset the flag
            self.train_dataset.needs_batch_size_recalc = False
            return True
            
        except Exception as e:
            print(f"⚠️ Batch size recalculation failed: {e}. Keeping current size: {self.batch_size}")
            self.train_dataset.needs_batch_size_recalc = False
            return False
    
    def setup(self, stage: str = None):
        """Setup datasets."""
        pass  # Already done in __init__
    
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
    """Lightning callback to manage curriculum learning and hard mining."""
    
    def __init__(self):
        super().__init__()
        self.normal_confidences = []
        self.balanced_phase_buffer = []  # Store early balanced examples
        self.ewc_initialized = False
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Update curriculum at start of each epoch."""
        datamodule = trainer.datamodule
        
        # Store samples from balanced phase for replay (first 20% of training)
        current_epoch = trainer.current_epoch
        if current_epoch < trainer.max_epochs * 0.2:  # First 20% = balanced phase
            # Sample some examples to store in buffer
            train_loader = datamodule.train_dataloader()
            if len(self.balanced_phase_buffer) < 1000:  # Limit buffer size
                batch = next(iter(train_loader))
                self.balanced_phase_buffer.extend(batch.to_data_list()[:10])  # Store 10 per epoch
        
        # Initialize EWC after balanced phase (20% instead of 30% for smoother momentum)
        elif not self.ewc_initialized and current_epoch == int(trainer.max_epochs * 0.2):
            self._initialize_memory_preservation(trainer, pl_module)
            self.ewc_initialized = True
        
        # Calculate GAT confidence on normals (if we have history)
        avg_confidence = None
        if len(self.normal_confidences) > 0:
            avg_confidence = np.mean(self.normal_confidences[-10:])  # Last 10 batches
        
        # Update curriculum with momentum scheduling
        datamodule.update_training_epoch(trainer.current_epoch, avg_confidence)
        
        # Check if we need to recalculate batch size due to dataset growth
        if hasattr(datamodule, 'recalculate_batch_size_if_needed'):
            batch_size_changed = datamodule.recalculate_batch_size_if_needed(trainer)
            if batch_size_changed:
                # Force recreation of train dataloader with new batch size
                trainer.fit_loop.setup_data()
        
        # Log momentum curriculum metrics to Lightning
        if hasattr(datamodule.train_dataset, 'curriculum_metrics'):
            metrics = datamodule.train_dataset.curriculum_metrics
            
            # Get current ratio from dataset
            current_ratio = datamodule.train_dataset._compute_curriculum_ratio()
            normal_percentage = (current_ratio / (1 + current_ratio)) * 100
            
            pl_module.log('curriculum/normal_ratio', current_ratio)
            pl_module.log('curriculum/normal_percentage', normal_percentage)
            pl_module.log('curriculum/momentum', metrics.get('momentum_accumulator', 0.0))
            pl_module.log('curriculum/progress_signal', metrics.get('progress_signal', 0.0))
        
        if avg_confidence is not None:
            pl_module.log('curriculum/normal_confidence', avg_confidence)
    
    def _initialize_memory_preservation(self, trainer, pl_module):
        """Initialize memory preservation mechanisms."""
        print(f"🧠 Initializing memory preservation at epoch {trainer.current_epoch}")
        
        # Add EWC if not already present
        if not hasattr(pl_module, 'ewc'):
            from src.training.memory_preserving_curriculum import ElasticWeightConsolidation
            pl_module.ewc = ElasticWeightConsolidation(pl_module.model)
            
            # Compute Fisher Information on balanced examples
            if len(self.balanced_phase_buffer) > 0:
                # Create temporary dataloader with balanced examples
                from torch.utils.data import DataLoader
                from torch_geometric.data import Batch
                
                def collate_fn(batch):
                    return Batch.from_data_list(batch)
                
                balanced_loader = DataLoader(
                    self.balanced_phase_buffer, 
                    batch_size=32, 
                    collate_fn=collate_fn
                )
                
                pl_module.ewc.compute_fisher_information(balanced_loader, pl_module.device)
                print("✅ EWC initialized with balanced examples")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Track model confidence and add EWC loss if needed."""
        if batch_idx % 10 != 0:  # Sample every 10 batches to avoid overhead
            return
            
        # Track normal confidence
        with torch.no_grad():
            logits = pl_module(batch)
            probs = torch.sigmoid(logits)
            
            # Find normal examples (y=0) and their confidence
            normal_mask = batch.y == 0
            if normal_mask.sum() > 0:
                normal_confidence = probs[normal_mask].mean().item()
                self.normal_confidences.append(normal_confidence)
        
        # Add EWC loss after balanced phase
        if hasattr(pl_module, 'ewc') and trainer.current_epoch > trainer.max_epochs * 0.3:
            ewc_loss = pl_module.ewc.compute_ewc_loss()
            
            # Add to the loss (this modifies the training step loss)
            if 'loss' in outputs:
                outputs['loss'] += ewc_loss * 0.1  # Weight the EWC loss
            
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