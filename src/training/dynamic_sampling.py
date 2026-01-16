"""
Dynamic Hard Mining for Extreme Class Imbalance

Builds on your successful VGAE approach with adaptive sampling.
"""

import torch
import numpy as np
from typing import List, Tuple
import lightning.pytorch as pl


class DynamicHardMiner:
    """
    Adaptively samples hard normal examples using VGAE reconstruction error.
    Updates sampling strategy during training based on GAT performance.
    """
    
    def __init__(self, vgae_model, target_ratio: float = 10.0, 
                 update_frequency: int = 5):
        """
        Args:
            vgae_model: Trained VGAE for computing reconstruction errors
            target_ratio: Desired normal:attack ratio (10:1 instead of 100:1)  
            update_frequency: How often to update hard mining (epochs)
        """
        self.vgae_model = vgae_model
        self.target_ratio = target_ratio
        self.update_frequency = update_frequency
        self.difficulty_threshold = 0.5  # Start with median difficulty
        
    def compute_difficulty_scores(self, normal_graphs):
        """Compute difficulty scores using VGAE reconstruction error."""
        self.vgae_model.eval()
        scores = []
        
        with torch.no_grad():
            for graph in normal_graphs:
                # Get reconstruction error from VGAE
                recon_error = self.vgae_model.compute_reconstruction_loss(graph)
                scores.append(recon_error.item())
        
        return np.array(scores)
    
    def update_sampling_strategy(self, gat_model, normal_graphs, attack_graphs, epoch):
        """Update difficulty threshold based on GAT performance."""
        if epoch % self.update_frequency != 0:
            return
            
        # Get GAT predictions on normal samples
        gat_model.eval()
        with torch.no_grad():
            normal_predictions = []
            for graph in normal_graphs[:1000]:  # Sample for speed
                pred = gat_model(graph).sigmoid()
                normal_predictions.append(pred.item())
        
        # Increase difficulty if GAT is getting too confident on normals
        avg_confidence = np.mean(normal_predictions)
        if avg_confidence < 0.1:  # Very confident it's normal
            self.difficulty_threshold *= 0.9  # Make sampling harder
        elif avg_confidence > 0.3:  # Not confident enough
            self.difficulty_threshold *= 1.1  # Make sampling easier
            
        print(f"Epoch {epoch}: Updated difficulty threshold to {self.difficulty_threshold:.3f}")
    
    def create_balanced_batch(self, normal_graphs, attack_graphs):
        """Create dynamically balanced batch."""
        # Compute difficulty scores
        difficulty_scores = self.compute_difficulty_scores(normal_graphs)
        
        # Select hard normal samples
        hard_mask = difficulty_scores > np.percentile(difficulty_scores, 
                                                    self.difficulty_threshold * 100)
        hard_normals = [normal_graphs[i] for i in np.where(hard_mask)[0]]
        
        # Balance the batch
        n_attacks = len(attack_graphs)
        n_normals_needed = int(n_attacks * self.target_ratio)
        
        if len(hard_normals) < n_normals_needed:
            # Add some random normals if not enough hard ones
            remaining = n_normals_needed - len(hard_normals)
            easy_normals = [normal_graphs[i] for i in np.where(~hard_mask)[0]]
            hard_normals.extend(np.random.choice(easy_normals, remaining, replace=False))
        else:
            # Sample from hard normals
            hard_normals = np.random.choice(hard_normals, n_normals_needed, replace=False)
            
        return list(hard_normals) + attack_graphs


class AdaptiveWeightedLoss(pl.LightningModule):
    """
    Loss that adapts weights based on model confidence and class distribution.
    """
    
    def __init__(self, initial_pos_weight: float = 100.0):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor(initial_pos_weight))
        self.confidence_history = []
        
    def forward(self, predictions, targets):
        """Compute adaptive weighted BCE loss."""
        # Standard weighted BCE
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, pos_weight=self.pos_weight
        )
        
        # Track prediction confidence
        with torch.no_grad():
            probs = torch.sigmoid(predictions)
            confidence = torch.max(probs, 1 - probs).mean()
            self.confidence_history.append(confidence.item())
            
            # Adapt pos_weight based on confidence (every 100 steps)
            if len(self.confidence_history) % 100 == 0:
                avg_confidence = np.mean(self.confidence_history[-100:])
                
                if avg_confidence > 0.95:  # Too confident, increase challenge
                    self.pos_weight *= 1.1
                elif avg_confidence < 0.7:  # Not confident enough, reduce challenge  
                    self.pos_weight *= 0.9
                    
                self.pos_weight = torch.clamp(self.pos_weight, 10.0, 1000.0)
        
        return loss


# ==================== Integration Example ====================

class ImbalanceAwareGAT(pl.LightningModule):
    """GAT with dynamic hard mining and adaptive loss."""
    
    def __init__(self, vgae_model, gat_model):
        super().__init__()
        self.gat = gat_model
        self.hard_miner = DynamicHardMiner(vgae_model)
        self.adaptive_loss = AdaptiveWeightedLoss()
        
    def training_step(self, batch, batch_idx):
        # Update sampling strategy periodically
        self.hard_miner.update_sampling_strategy(
            self.gat, batch['normal_graphs'], batch['attack_graphs'], 
            self.current_epoch
        )
        
        # Get balanced batch
        balanced_graphs = self.hard_miner.create_balanced_batch(
            batch['normal_graphs'], batch['attack_graphs']
        )
        
        # Forward pass
        predictions = []
        targets = []
        for graph in balanced_graphs:
            pred = self.gat(graph)
            predictions.append(pred)
            targets.append(graph.y)
            
        predictions = torch.cat(predictions)
        targets = torch.cat(targets).float()
        
        # Adaptive loss
        loss = self.adaptive_loss(predictions, targets)
        
        self.log('train_loss', loss)
        self.log('pos_weight', self.adaptive_loss.pos_weight)
        return loss