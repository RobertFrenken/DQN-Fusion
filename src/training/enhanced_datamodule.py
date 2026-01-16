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
    
    def _compute_curriculum_ratio(self) -> float:
        """Compute current normal:attack ratio based on curriculum."""
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        # Exponential increase: slow start, rapid ramp-up
        ratio = self.start_ratio * (self.end_ratio / self.start_ratio) ** (progress ** 1.5)
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
        
        print(f"Epoch {self.current_epoch}: Ratio {current_ratio:.2f}:1 "
              f"({len(selected_normals)} normals, {len(self.attack_graphs)} attacks)")
    
    def update_epoch(self, epoch: int, gat_confidence: float = None):
        """Update epoch and difficulty based on GAT performance."""
        self.current_epoch = epoch
        
        # Adaptive difficulty based on GAT confidence
        if gat_confidence is not None:
            if gat_confidence < 0.1:  # Too confident on normals
                self.difficulty_percentile = min(90.0, self.difficulty_percentile + 5.0)
            elif gat_confidence > 0.3:  # Not confident enough
                self.difficulty_percentile = max(10.0, self.difficulty_percentile - 5.0)
        
        # Regenerate samples for new epoch
        self._generate_epoch_samples()
    
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
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Update curriculum at start of each epoch."""
        datamodule = trainer.datamodule
        
        # Calculate GAT confidence on normals (if we have history)
        avg_confidence = None
        if len(self.normal_confidences) > 0:
            avg_confidence = np.mean(self.normal_confidences[-10:])  # Last 10 batches
        
        # Update curriculum
        datamodule.update_training_epoch(trainer.current_epoch, avg_confidence)
        
        # Clear confidence history for new epoch
        self.normal_confidences = []
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Track model confidence on normal examples."""
        if batch_idx % 10 != 0:  # Sample every 10 batches to avoid overhead
            return
            
        # Get predictions and targets
        with torch.no_grad():
            logits = pl_module(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(logits)
            
            # Find normal examples (y=0) and their confidence
            normal_mask = batch.y == 0
            if normal_mask.sum() > 0:
                normal_confidence = probs[normal_mask].mean().item()
                self.normal_confidences.append(normal_confidence)
    
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