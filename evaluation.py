import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import torch
import sys
import time
from omegaconf import DictConfig, OmegaConf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
from models.models import GATWithJK, GraphAutoencoderNeighborhood
from preprocessing import graph_creation, build_id_mapping_from_normal
from training_utils import PyTorchTrainer, PyTorchDistillationTrainer, DistillationTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from torch_geometric.data import Batch, Data
import random

# FIXED: Direct model creation instead of importing from training file
def create_student_models(num_ids, embedding_dim, device):
    """Create student models with exact same architecture as training."""
    
    # Student autoencoder (small, efficient)
    student_autoencoder = GraphAutoencoderNeighborhood(
        num_ids=num_ids, 
        in_channels=11, 
        embedding_dim=embedding_dim,
        hidden_dim=16,           # Must match training
        latent_dim=16,           # Must match training
        num_encoder_layers=2,    # Must match training
        num_decoder_layers=2,    # Must match training
        encoder_heads=2,         # Must match training
        decoder_heads=2          # Must match training
    ).to(device)
    
    # Student classifier (small, efficient)
    student_classifier = GATWithJK(
        num_ids=num_ids, 
        in_channels=11, 
        hidden_channels=16,      # Must match training
        out_channels=1, 
        num_layers=2,            # Must match training
        heads=4,                 # Must match training
        embedding_dim=embedding_dim
    ).to(device)
    
    return student_autoencoder, student_classifier


def create_data_loaders(train_subset, test_dataset, full_train_dataset, batch_size, device):
    """AGGRESSIVE data loaders for evaluation - much larger batch sizes."""
    
    # Check GPU memory for optimal batch sizing
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected GPU memory: {gpu_memory_gb:.1f}GB")
        
        if gpu_memory_gb < 20:  # 16GB Tesla V100
            eval_batch_size = 512    # 8x larger than current
        else:  # 32GB Tesla V100
            eval_batch_size = 1024   # 16x larger than current
    else:
        eval_batch_size = 256
    
    # EVALUATION-OPTIMIZED settings
    pin_memory = True if torch.cuda.is_available() else False
    num_workers = 4          # Fewer workers for evaluation
    prefetch_factor = 2      # Conservative prefetch
    persistent_workers = True
    
    print(f"Evaluation DataLoader: batch_size={eval_batch_size}, workers={num_workers}")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=eval_batch_size,  # MUCH larger
        shuffle=False,               # No shuffling needed for eval
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False             # Keep all samples
    )
    
    return test_loader
class StudentEvaluationPipeline:
    """Evaluation pipeline for knowledge distilled student models."""
    
    def __init__(self, num_ids, embedding_dim=8, device='cpu'):
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'
        
        # Create student models
        self.student_autoencoder, self.student_classifier = create_student_models(
            num_ids=num_ids, embedding_dim=embedding_dim, device=device
        )
        
        self.threshold = 0.0
        print(f"Initialized FAST Student Evaluation Pipeline on {device}")

    def load_student_models(self, autoencoder_path, classifier_path, threshold_path=None):
        """Load the trained student models (threshold now optional)."""
        print(f"Loading student autoencoder from: {autoencoder_path}")
        autoencoder_state_dict = torch.load(autoencoder_path, map_location=self.device, weights_only=True)
        self.student_autoencoder.load_state_dict(autoencoder_state_dict)
        
        print(f"Loading student classifier from: {classifier_path}")
        classifier_state_dict = torch.load(classifier_path, map_location=self.device, weights_only=True)
        self.student_classifier.load_state_dict(classifier_state_dict)
        
        # OPTIONAL: Threshold is no longer critical for fusion methods
        if threshold_path and os.path.exists(threshold_path):
            threshold_data = torch.load(threshold_path, map_location=self.device, weights_only=True)
            if isinstance(threshold_data, dict) and 'threshold' in threshold_data:
                self.threshold = threshold_data['threshold']
            else:
                self.threshold = float(threshold_data)
            print(f"Loaded threshold: {self.threshold} (used only for reference)")
        else:
            self.threshold = None
            print("No threshold loaded - using threshold-free fusion methods")
        
        # Set to eval mode
        self.student_autoencoder.eval()
        self.student_classifier.eval()
        
        print("Student models loaded and set to evaluation mode!")

    def set_threshold(self, threshold):
        """Set the anomaly detection threshold."""
        self.threshold = threshold

    def predict_batch_optimized(self, data):
        """ULTRA-FAST batch prediction with temperature-scaled confidence fusion."""
        if self.is_cuda:
            data = data.to(self.device, non_blocking=True)
        else:
            data = data.to(self.device)
        
        with torch.no_grad():
            # Step 1: BATCH autoencoder forward pass
            cont_out, canid_logits, neighbor_logits, _, _ = self.student_autoencoder(
                data.x, data.edge_index, data.batch)
            
            # Step 2: VECTORIZED anomaly score computation
            node_errors = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            neighbor_targets = self.student_autoencoder.create_neighborhood_targets(
                data.x, data.edge_index, data.batch)
            neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                neighbor_logits, neighbor_targets).mean(dim=1)
            
            # Step 3: EFFICIENT graph-level aggregation
            batch_size = data.batch.max().item() + 1
            graph_node_errors = torch.zeros(batch_size, device=self.device)
            graph_neighbor_errors = torch.zeros(batch_size, device=self.device)
            
            # Ultra-fast scatter operations
            graph_node_errors.scatter_reduce_(0, data.batch, node_errors, reduce='amax')
            graph_neighbor_errors.scatter_reduce_(0, data.batch, neighbor_errors, reduce='amax')
            
            # Step 4: VECTORIZED composite scoring (same as training)
            weight_node = 1.0
            weight_neighbor = 20.0
            raw_anomaly_scores = (weight_node * graph_node_errors + 
                                weight_neighbor * graph_neighbor_errors)
            
            # Step 5: SINGLE classifier forward pass for ENTIRE batch
            classifier_logits = self.student_classifier(data).squeeze()
            
            # Step 6: IMPROVED Raw Score Fusion (NO THRESHOLD DEPENDENCY)
            temperature = 2.0
            
            # Get GAT probabilities and confidence
            calibrated_logits = classifier_logits / temperature
            gat_probs = torch.sigmoid(calibrated_logits)
            gat_confidence = torch.abs(calibrated_logits)
            
            # Normalize anomaly scores to [0,1] for fusion
            if raw_anomaly_scores.max() > raw_anomaly_scores.min():
                norm_anomaly_scores = (raw_anomaly_scores - raw_anomaly_scores.min()) / (
                    raw_anomaly_scores.max() - raw_anomaly_scores.min())
            else:
                norm_anomaly_scores = torch.zeros_like(raw_anomaly_scores)
            
            # Normalize GAT confidence to [0,1]
            if gat_confidence.max() > gat_confidence.min():
                norm_gat_confidence = (gat_confidence - gat_confidence.min()) / (
                    gat_confidence.max() - gat_confidence.min())
            else:
                norm_gat_confidence = torch.ones_like(gat_confidence)
            
            # IMPROVED: Dual confidence weighting
            # Both models contribute to confidence estimation
            anomaly_confidence = norm_anomaly_scores  # Higher score = more confident it's anomalous
            
            # Adaptive alpha based on BOTH model confidences
            # When both are confident about anomaly: high alpha (trust anomaly detection more)
            # When GAT is confident but anomaly is low: lower alpha (trust GAT more)
            combined_confidence = (norm_gat_confidence + anomaly_confidence) / 2
            min_alpha, max_alpha = 0.3, 0.7
            adaptive_alpha = min_alpha + (max_alpha - min_alpha) * combined_confidence
            
            # THRESHOLD-FREE two-stage: Use percentile-based decision
            # Instead of fixed threshold, use dynamic threshold based on score distribution
            dynamic_threshold = torch.quantile(raw_anomaly_scores, 0.8)  # Top 20% as anomalies
            anomaly_mask = raw_anomaly_scores > dynamic_threshold
            two_stage_preds = torch.zeros(batch_size, device=self.device)
            two_stage_preds[anomaly_mask] = (classifier_logits[anomaly_mask] > 0.0).float()
            
            # IMPROVED adaptive fusion prediction
            fused_scores = adaptive_alpha * norm_anomaly_scores + (1 - adaptive_alpha) * gat_probs
            fusion_preds = (fused_scores > 0.5).float()
            
            return {
                'two_stage_preds': two_stage_preds.long(),
                'fusion_preds': fusion_preds.long(),
                'anomaly_scores': norm_anomaly_scores,
                'gat_probs': gat_probs,
                'gat_confidence': norm_gat_confidence,
                'anomaly_confidence': anomaly_confidence,    # NEW
                'adaptive_alpha': adaptive_alpha,
                'raw_anomaly_scores': raw_anomaly_scores,
                'dynamic_threshold': dynamic_threshold       # NEW: For analysis
            }

    def predict_two_stage(self, data):
        """Standard two-stage prediction: anomaly detection + classification."""
        if self.is_cuda:
            data = data.to(self.device, non_blocking=True)
        else:
            data = data.to(self.device)
        
        with torch.no_grad():
            # Stage 1: Anomaly detection using autoencoder
            cont_out, _, _, _, _ = self.student_autoencoder(data.x, data.edge_index, data.batch)
            
            # FIXED: Use the same composite scoring as training
            node_errors = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            # Get neighbor reconstruction errors
            neighbor_targets = self.student_autoencoder.create_neighborhood_targets(
                data.x, data.edge_index, data.batch)
            _, _, neighbor_logits, _, _ = self.student_autoencoder(data.x, data.edge_index, data.batch)
            neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                neighbor_logits, neighbor_targets).mean(dim=1)
            
            # Vectorized graph-level error computation with composite scoring
            batch_size = data.batch.max().item() + 1
            graph_node_errors = torch.zeros(batch_size, device=self.device)
            graph_neighbor_errors = torch.zeros(batch_size, device=self.device)
            
            # Efficient scatter operations
            graph_node_errors.scatter_reduce_(0, data.batch, node_errors, reduce='amax')
            graph_neighbor_errors.scatter_reduce_(0, data.batch, neighbor_errors, reduce='amax')
            
            # FIXED: Use same composite scoring as training
            weight_node = 1.0
            weight_neighbor = 20.0  # High weight for neighborhood errors
            
            composite_errors = (weight_node * graph_node_errors + 
                              weight_neighbor * graph_neighbor_errors)
            
            # Stage 2: Classification for anomalous graphs
            preds = torch.zeros(batch_size, device=self.device)
            anomaly_mask = composite_errors > self.threshold
            
            if anomaly_mask.any():
                anomaly_indices = torch.nonzero(anomaly_mask).flatten()
                
                # Process anomalous graphs
                for idx in anomaly_indices:
                    # Get the graph for this index
                    graphs = Batch.to_data_list(data)
                    graph = graphs[idx]
                    try:
                        # Run classifier on the graph
                        graph_batch = graph.to(self.device)
                        classifier_logit = self.student_classifier(graph_batch).item()
                        
                        # Use threshold 0.0 for classification
                        preds[idx] = (classifier_logit > 0.0).float()
                        
                    except Exception as e:
                        # Fallback: if composite error is very high, classify as attack
                        preds[idx] = (composite_errors[idx] > self.threshold * 1.5).float()
            
            return preds.long()

    def predict_with_score_fusion(self, data, fusion_method='adaptive_confidence'):
        """Advanced fusion methods using raw reconstruction scores."""
        if self.is_cuda:
            data = data.to(self.device, non_blocking=True)
        else:
            data = data.to(self.device)
        
        with torch.no_grad():
            # Get all raw outputs (same computation as predict_batch_optimized)
            cont_out, canid_logits, neighbor_logits, _, _ = self.student_autoencoder(
                data.x, data.edge_index, data.batch)
            
            # Compute raw anomaly scores
            node_errors = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            neighbor_targets = self.student_autoencoder.create_neighborhood_targets(
                data.x, data.edge_index, data.batch)
            neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                neighbor_logits, neighbor_targets).mean(dim=1)
            
            batch_size = data.batch.max().item() + 1
            graph_node_errors = torch.zeros(batch_size, device=self.device)
            graph_neighbor_errors = torch.zeros(batch_size, device=self.device)
            
            graph_node_errors.scatter_reduce_(0, data.batch, node_errors, reduce='amax')
            graph_neighbor_errors.scatter_reduce_(0, data.batch, neighbor_errors, reduce='amax')
            
            raw_anomaly_scores = (1.0 * graph_node_errors + 20.0 * graph_neighbor_errors)
            classifier_logits = self.student_classifier(data).squeeze()
            
            # Different fusion strategies
            if fusion_method == 'percentile_gating':
                # Use percentile-based gating instead of fixed threshold
                anomaly_percentile = torch.quantile(raw_anomaly_scores, 0.75)  # Top 25%
                anomaly_mask = raw_anomaly_scores > anomaly_percentile
                
                final_preds = torch.zeros(batch_size, device=self.device)
                final_preds[anomaly_mask] = (classifier_logits[anomaly_mask] > 0.0).float()
                
            elif fusion_method == 'soft_voting':
                # Soft voting: combine normalized scores directly
                norm_anomaly = (raw_anomaly_scores - raw_anomaly_scores.min()) / (
                    raw_anomaly_scores.max() - raw_anomaly_scores.min() + 1e-8)
                gat_probs = torch.sigmoid(classifier_logits)
                
                # Equal weighted soft voting
                combined_score = (norm_anomaly + gat_probs) / 2
                final_preds = (combined_score > 0.5).float()
                
            elif fusion_method == 'uncertainty_weighted':
                # Weight by uncertainty: more uncertain predictions get less weight
                norm_anomaly = (raw_anomaly_scores - raw_anomaly_scores.min()) / (
                    raw_anomaly_scores.max() - raw_anomaly_scores.min() + 1e-8)
                gat_probs = torch.sigmoid(classifier_logits)
                
                # Uncertainty as distance from decision boundary
                anomaly_uncertainty = torch.abs(norm_anomaly - 0.5)  # Closer to 0.5 = more uncertain
                gat_uncertainty = torch.abs(gat_probs - 0.5)
                
                # Confidence = 1 - uncertainty
                anomaly_confidence = 1 - anomaly_uncertainty
                gat_confidence = 1 - gat_uncertainty
                
                # Weight by confidence
                total_confidence = anomaly_confidence + gat_confidence + 1e-8
                anomaly_weight = anomaly_confidence / total_confidence
                gat_weight = gat_confidence / total_confidence
                
                combined_score = anomaly_weight * norm_anomaly + gat_weight * gat_probs
                final_preds = (combined_score > 0.5).float()
                
            elif fusion_method == 'adaptive_confidence':
                # Most sophisticated: adapt based on both raw scores and confidence
                norm_anomaly = (raw_anomaly_scores - raw_anomaly_scores.min()) / (
                    raw_anomaly_scores.max() - raw_anomaly_scores.min() + 1e-8)
                gat_probs = torch.sigmoid(classifier_logits)
                gat_confidence = torch.abs(classifier_logits)
                
                # Normalize GAT confidence
                if gat_confidence.max() > gat_confidence.min():
                    norm_gat_confidence = (gat_confidence - gat_confidence.min()) / (
                        gat_confidence.max() - gat_confidence.min())
                else:
                    norm_gat_confidence = torch.ones_like(gat_confidence)
                
                # High raw anomaly scores indicate high confidence in anomaly detection
                anomaly_confidence = norm_anomaly
                
                # Dynamic alpha based on relative confidence
                # When anomaly score is high and GAT confidence is low: trust anomaly more
                # When GAT confidence is high and anomaly score is low: trust GAT more
                confidence_ratio = (anomaly_confidence + 1e-8) / (norm_gat_confidence + 1e-8)
                adaptive_alpha = torch.sigmoid(confidence_ratio - 1.0)  # Centers around 0.5
                
                combined_score = adaptive_alpha * norm_anomaly + (1 - adaptive_alpha) * gat_probs
                final_preds = (combined_score > 0.5).float()
                
            else:
                # Fallback: simple weighted average
                norm_anomaly = (raw_anomaly_scores - raw_anomaly_scores.min()) / (
                    raw_anomaly_scores.max() - raw_anomaly_scores.min() + 1e-8)
                gat_probs = torch.sigmoid(classifier_logits)
                combined_score = 0.6 * norm_anomaly + 0.4 * gat_probs
                final_preds = (combined_score > 0.5).float()
            
            return final_preds.long(), norm_anomaly, gat_probs
    
    def predict_with_fusion(self, data, fusion_method='weighted', alpha=0.6):
        """
        Improved fusion-based prediction with better decision logic.
        """
        if self.is_cuda:
            data = data.to(self.device, non_blocking=True)
        else:
            data = data.to(self.device)
        
        with torch.no_grad():
            # Get autoencoder outputs for anomaly detection
            cont_out, canid_logits, neighbor_logits, _, _ = self.student_autoencoder(
                data.x, data.edge_index, data.batch)
            
            # Compute composite anomaly scores
            node_errors = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            neighbor_targets = self.student_autoencoder.create_neighborhood_targets(
                data.x, data.edge_index, data.batch)
            neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                neighbor_logits, neighbor_targets).mean(dim=1)
            
            # Vectorized graph-level error computation
            batch_size = data.batch.max().item() + 1
            graph_node_errors = torch.zeros(batch_size, device=self.device)
            graph_neighbor_errors = torch.zeros(batch_size, device=self.device)
            
            # Efficient scatter operations
            graph_node_errors.scatter_reduce_(0, data.batch, node_errors, reduce='amax')
            graph_neighbor_errors.scatter_reduce_(0, data.batch, neighbor_errors, reduce='amax')
            
            # Compute RAW composite anomaly scores
            weight_node = 1.0
            weight_neighbor = 20.0
            raw_anomaly_scores = (weight_node * graph_node_errors + 
                                weight_neighbor * graph_neighbor_errors)
            
            # Get GAT classification logits (NOT probabilities yet)
            graphs = Batch.to_data_list(data)
            gat_logits = []
            
            for i, graph in enumerate(graphs):
                graph_batch = graph.to(self.device)
                gat_logit = self.student_classifier(graph_batch).item()
                gat_logits.append(gat_logit)
            
            gat_logits = torch.tensor(gat_logits, device=self.device)
            
            # IMPROVED FUSION STRATEGY - Use hierarchical decision making
            final_preds = torch.zeros(batch_size, device=self.device)
            
            if fusion_method == 'hierarchical':
                # Strategy 1: Hierarchical - Anomaly detection first, then GAT
                anomaly_mask = raw_anomaly_scores > self.threshold
                
                for i in range(batch_size):
                    if anomaly_mask[i]:
                        # If anomaly detected, use GAT to classify
                        final_preds[i] = (gat_logits[i] > 0.0).float()
                    else:
                        # If no anomaly detected, classify as normal
                        final_preds[i] = 0.0
                        
            elif fusion_method == 'confidence_weighted':
                # Strategy 2: Weight by confidence levels
                # Normalize anomaly scores to [0,1]
                if raw_anomaly_scores.max() > raw_anomaly_scores.min():
                    norm_anomaly = (raw_anomaly_scores - raw_anomaly_scores.min()) / (
                        raw_anomaly_scores.max() - raw_anomaly_scores.min())
                else:
                    norm_anomaly = torch.zeros_like(raw_anomaly_scores)
                
                # Convert GAT logits to probabilities but keep confidence info
                gat_probs = torch.sigmoid(gat_logits)
                gat_confidence = torch.abs(gat_logits)  # Higher absolute value = more confident
                
                # Normalize confidence to [0,1]
                if gat_confidence.max() > gat_confidence.min():
                    norm_gat_conf = (gat_confidence - gat_confidence.min()) / (
                        gat_confidence.max() - gat_confidence.min())
                else:
                    norm_gat_conf = torch.ones_like(gat_confidence)
                
                # Weight by confidence - more confident GAT gets higher weight
                adaptive_alpha = 0.3 + 0.4 * norm_gat_conf  # Alpha ranges from 0.3 to 0.7
                
                fused_scores = adaptive_alpha * norm_anomaly + (1 - adaptive_alpha) * gat_probs
                final_preds = (fused_scores > 0.5).float()
                
            elif fusion_method == 'conservative':
                # Strategy 3: Conservative - Both must agree for attack classification
                anomaly_binary = (raw_anomaly_scores > self.threshold).float()
                gat_binary = (gat_logits > 0.0).float()
                
                # Only classify as attack if BOTH agree
                final_preds = anomaly_binary * gat_binary
                
            elif fusion_method == 'optimistic':
                # Strategy 4: Optimistic - Either can classify as attack
                anomaly_binary = (raw_anomaly_scores > self.threshold).float()
                gat_binary = (gat_logits > 0.0).float()
                
                # Classify as attack if EITHER agrees
                final_preds = torch.maximum(anomaly_binary, gat_binary)
                
            elif fusion_method == 'gat_priority':
                # Strategy 5: GAT has priority, anomaly detection as backup
                gat_confident_mask = torch.abs(gat_logits) > 5.0  # High confidence threshold
                
                for i in range(batch_size):
                    if gat_confident_mask[i]:
                        # If GAT is confident, use its decision
                        final_preds[i] = (gat_logits[i] > 0.0).float()
                    else:
                        # If GAT is not confident, fall back to anomaly detection
                        final_preds[i] = (raw_anomaly_scores[i] > self.threshold).float()
                        
            else:
                # Default: Traditional weighted fusion
                if raw_anomaly_scores.max() > raw_anomaly_scores.min():
                    norm_anomaly = (raw_anomaly_scores - raw_anomaly_scores.min()) / (
                        raw_anomaly_scores.max() - raw_anomaly_scores.min())
                else:
                    norm_anomaly = torch.zeros_like(raw_anomaly_scores)
                
                gat_probs = torch.sigmoid(gat_logits)
                fused_scores = alpha * norm_anomaly + (1 - alpha) * gat_probs
                final_preds = (fused_scores > 0.5).float()
            
            # For evaluation, return normalized scores
            if raw_anomaly_scores.max() > raw_anomaly_scores.min():
                normalized_anomaly_scores = (raw_anomaly_scores - raw_anomaly_scores.min()) / (
                    raw_anomaly_scores.max() - raw_anomaly_scores.min())
            else:
                normalized_anomaly_scores = torch.zeros_like(raw_anomaly_scores)
                
            gat_probs = torch.sigmoid(gat_logits)
            
            return final_preds.long(), normalized_anomaly_scores, gat_probs
    

# Also add the print_sample_evaluations function call to evaluate_fusion_methods:
def print_sample_evaluations(all_labels, all_anomaly_scores, all_gat_probs, num_samples=10):
    """Print sample-by-sample evaluation for debugging fusion behavior."""
    print(f"\n=== Sample-by-Sample Fusion Analysis (first {num_samples} samples) ===")
    print("Sample | Label | Anomaly Score | GAT Prob | Weighted(0.6) | Product | Max | Final Pred")
    print("-" * 85)
    
    for i in range(min(num_samples, len(all_labels))):
        label = all_labels[i]
        # FIXED: Convert to scalar values
        anomaly_score = float(all_anomaly_scores[i])
        gat_prob = float(all_gat_probs[i])
        
        # Calculate different fusion scores
        weighted_score = 0.6 * anomaly_score + 0.4 * gat_prob
        product_score = (anomaly_score * gat_prob) ** 0.5
        max_score = max(anomaly_score, gat_prob)  # Now works with scalars
        
        # Final prediction (using weighted as example)
        final_pred = 1 if weighted_score > 0.5 else 0
        
        print(f"{i:6d} | {label:5d} | {anomaly_score:12.4f} | {gat_prob:8.4f} | "
              f"{weighted_score:12.4f} | {product_score:7.4f} | {max_score:3.4f} | {final_pred:9d}")
        
def evaluate_student_model_fast(pipeline, data_loader, device):
    """ULTRA-FAST evaluation with support for new fusion metrics."""
    
    # Pre-allocate lists for speed  
    all_labels = []
    all_two_stage_preds = []
    all_fusion_preds = []
    all_anomaly_scores = []
    all_gat_probs = []
    all_gat_confidence = []     # NEW
    all_adaptive_alpha = []     # NEW
    
    print(f"Fast evaluation with {len(data_loader)} batches...")
    
    # FAST batch processing
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device, non_blocking=True)
            
            # SINGLE optimized prediction call
            results = pipeline.predict_batch_optimized(batch)
            
            # Collect results
            all_two_stage_preds.append(results['two_stage_preds'].cpu())
            all_fusion_preds.append(results['fusion_preds'].cpu())
            all_anomaly_scores.append(results['anomaly_scores'].cpu())
            all_gat_probs.append(results['gat_probs'].cpu())
            all_labels.append(batch.y.cpu())
            
            # FIXED: Collect new metrics if available - check properly
            if 'gat_confidence' in results and results['gat_confidence'] is not None:
                all_gat_confidence.append(results['gat_confidence'].cpu())
            if 'adaptive_alpha' in results and results['adaptive_alpha'] is not None:
                all_adaptive_alpha.append(results['adaptive_alpha'].cpu())
            
            # Progress update every 100 batches
            if batch_idx % 100 == 0:
                print(f"Processed batch {batch_idx}/{len(data_loader)}")
    
    # FAST concatenation
    all_labels = torch.cat(all_labels).numpy()
    all_two_stage_preds = torch.cat(all_two_stage_preds).numpy()
    all_fusion_preds = torch.cat(all_fusion_preds).numpy()
    all_anomaly_scores = torch.cat(all_anomaly_scores).numpy()
    all_gat_probs = torch.cat(all_gat_probs).numpy()
    
    # FIXED: Concatenate new metrics if available - check list length instead of truth value
    if len(all_gat_confidence) > 0:
        all_gat_confidence = torch.cat(all_gat_confidence).numpy()
    else:
        all_gat_confidence = None
        
    if len(all_adaptive_alpha) > 0:
        all_adaptive_alpha = torch.cat(all_adaptive_alpha).numpy()
    else:
        all_adaptive_alpha = None
    
    # FAST metrics computation (existing code unchanged)
    results = {}
    
    # Two-stage metrics
    results['two_stage'] = {
        'accuracy': accuracy_score(all_labels, all_two_stage_preds),
        'precision': precision_score(all_labels, all_two_stage_preds, zero_division=0),
        'recall': recall_score(all_labels, all_two_stage_preds, zero_division=0),
        'f1': f1_score(all_labels, all_two_stage_preds, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_two_stage_preds)
    }
    
    # Fusion metrics
    results['fusion'] = {
        'accuracy': accuracy_score(all_labels, all_fusion_preds),
        'precision': precision_score(all_labels, all_fusion_preds, zero_division=0),
        'recall': recall_score(all_labels, all_fusion_preds, zero_division=0),
        'f1': f1_score(all_labels, all_fusion_preds, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_fusion_preds)
    }
    
    # Individual component performance
    anomaly_only_preds = (all_anomaly_scores > 0.5).astype(int)
    gat_only_preds = (all_gat_probs > 0.5).astype(int)
    
    results['anomaly_only'] = {
        'accuracy': accuracy_score(all_labels, anomaly_only_preds),
        'f1': f1_score(all_labels, anomaly_only_preds, zero_division=0)
    }
    
    results['gat_only'] = {
        'accuracy': accuracy_score(all_labels, gat_only_preds),
        'f1': f1_score(all_labels, gat_only_preds, zero_division=0)
    }
    
    # Store all data for analysis
    results['labels'] = all_labels
    results['anomaly_scores'] = all_anomaly_scores
    results['gat_probs'] = all_gat_probs
    results['fusion_preds'] = all_fusion_preds
    
    # FIXED: Store new metrics if available - use proper None checks
    if all_gat_confidence is not None:
        results['gat_confidence'] = all_gat_confidence
    if all_adaptive_alpha is not None:
        results['adaptive_alpha'] = all_adaptive_alpha
    
    return results

def analyze_score_distributions(results, num_samples=20):
    """Analyze the distribution of raw and normalized scores."""
    print(f"\n=== Score Distribution Analysis ===")
    
    raw_scores = results.get('raw_anomaly_scores', [])
    norm_scores = results.get('anomaly_scores', [])
    gat_probs = results.get('gat_probs', [])
    
    if len(raw_scores) > 0:
        print(f"Raw Anomaly Scores:")
        print(f"  Min: {np.min(raw_scores):.6f}")
        print(f"  Max: {np.max(raw_scores):.6f}")
        print(f"  Mean: {np.mean(raw_scores):.6f}")
        print(f"  75th percentile: {np.percentile(raw_scores, 75):.6f}")
        print(f"  90th percentile: {np.percentile(raw_scores, 90):.6f}")
        print(f"  95th percentile: {np.percentile(raw_scores, 95):.6f}")
    
    if len(norm_scores) > 0:
        print(f"Normalized Anomaly Scores:")
        print(f"  Min: {np.min(norm_scores):.6f}")
        print(f"  Max: {np.max(norm_scores):.6f}")  
        print(f"  Mean: {np.mean(norm_scores):.6f}")
    
    if len(gat_probs) > 0:
        print(f"GAT Probabilities:")
        print(f"  Min: {np.min(gat_probs):.6f}")
        print(f"  Max: {np.max(gat_probs):.6f}")
        print(f"  Mean: {np.mean(gat_probs):.6f}")
    
    # Show threshold-free decision making
    if 'dynamic_threshold' in results:
        print(f"Dynamic Threshold: {results['dynamic_threshold']:.6f}")

def analyze_fusion_behavior(results, num_samples=20):
    """Analyze how the adaptive fusion behaves."""
    print(f"\n=== Adaptive Fusion Analysis (first {num_samples} samples) ===")
    print("Sample | Label | Anomaly | GAT_Prob | Confidence | Alpha | Fused | Pred")
    print("-" * 75)
    
    labels = results['labels'][:num_samples]
    anomaly_scores = results['anomaly_scores'][:num_samples] 
    gat_probs = results['gat_probs'][:num_samples]
    gat_confidence = results.get('gat_confidence', np.array([0.5] * num_samples))[:num_samples]
    adaptive_alpha = results.get('adaptive_alpha', np.array([0.6] * num_samples))[:num_samples]
    fusion_preds = results['fusion_preds'][:num_samples]
    
    for i in range(min(num_samples, len(labels))):
        # FIXED: Safe array indexing without truth value evaluation
        try:
            if hasattr(adaptive_alpha, '__len__') and len(adaptive_alpha) > i:
                alpha = float(adaptive_alpha[i])
            else:
                alpha = 0.6  # Default value
        except (IndexError, TypeError):
            alpha = 0.6
            
        try:
            if hasattr(gat_confidence, '__len__') and len(gat_confidence) > i:
                confidence = float(gat_confidence[i])
            else:
                confidence = 0.5
        except (IndexError, TypeError):
            confidence = 0.5
            
        fused = alpha * gat_probs[i] + (1 - alpha) * anomaly_scores[i]
        
        print(f"{i:6d} | {labels[i]:5d} | {anomaly_scores[i]:7.3f} | {gat_probs[i]:8.3f} | "
              f"{confidence:10.3f} | {alpha:5.3f} | {fused:5.3f} | {fusion_preds[i]:4d}")
    
    # FIXED: Summary statistics with safe array checking
    try:
        if hasattr(adaptive_alpha, '__len__') and len(adaptive_alpha) > 0:
            alpha_array = np.array(adaptive_alpha)
            if alpha_array.size > 0:
                print(f"\nAlpha Statistics:")
                print(f"  Mean: {np.mean(alpha_array):.3f}")
                print(f"  Min:  {np.min(alpha_array):.3f}")  
                print(f"  Max:  {np.max(alpha_array):.3f}")
                print(f"  Std:  {np.std(alpha_array):.3f}")
            else:
                print(f"\nAlpha Statistics: Using default value (0.6)")
        else:
            print(f"\nAlpha Statistics: Using default value (0.6)")
    except Exception as e:
        print(f"\nAlpha Statistics: Error computing statistics - {e}")
        print(f"Using default value (0.6)")

def main(evaluate_known_only=True):
    """OPTIMIZED main evaluation function."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check available models
    available_models = []
    model_patterns = [
        ("set_01", "datasets/can-train-and-test-v1.5/set_01"),
        ("set_02", "datasets/can-train-and-test-v1.5/set_02"), 
        ("set_03", "datasets/can-train-and-test-v1.5/set_03"),
        ("set_04", "datasets/can-train-and-test-v1.5/set_04"),
        ("hcrl_ch", "datasets/can-train-and-test-v1.5/hcrl-ch"),
        ("hcrl_sa", "datasets/can-train-and-test-v1.5/hcrl-sa"),
    ]
    
    test_datasets = []
    for model_key, folder_path in model_patterns:
        autoencoder_path = f"saved_models/student_autoencoder_{model_key}.pth"
        classifier_path = f"saved_models/student_classifier_{model_key}.pth"
        threshold_path = f"saved_models/student_threshold_{model_key}.pth"
        
        if os.path.exists(autoencoder_path) and os.path.exists(classifier_path):
            test_datasets.append({
                "folder": folder_path,
                "student_autoencoder": autoencoder_path,
                "student_classifier": classifier_path,
                "threshold": threshold_path
            })
            available_models.append(model_key)

    if not available_models:
        print("No student models found!")
        return

    print("=" * 80)
    print("OPTIMIZED STUDENT MODEL EVALUATION")
    print("=" * 80)
    print(f"Found models: {', '.join(available_models)}")

    for dataset_info in test_datasets:
        root_folder = dataset_info["folder"]
        student_autoencoder_path = dataset_info["student_autoencoder"]
        student_classifier_path = dataset_info["student_classifier"]
        threshold_path = dataset_info["threshold"]

        print(f"\n{'='*50}")
        print(f"Evaluating: {root_folder.split('/')[-1]}")
        print(f"{'='*50}")

        try:
            eval_start = time.time()
            
            # Build dataset
            id_mapping = build_id_mapping_from_normal(root_folder)
            
            if "hcrl-ch" in root_folder:
                combined_dataset = []
                for subfolder_name in os.listdir(root_folder):
                    subfolder_path = os.path.join(root_folder, subfolder_name)
                    if os.path.isdir(subfolder_path) and subfolder_name.startswith("test_"):
                        test_data = graph_creation(subfolder_path, folder_type="test_", 
                                                 id_mapping=id_mapping, window_size=100)
                        combined_dataset.extend(test_data)
                test_dataset = combined_dataset
                
            elif "hcrl-sa" in root_folder:
                test_subfolder = os.path.join(root_folder, "test_01_known_vehicle_known_attack")
                if not os.path.exists(test_subfolder):
                    print(f"Test folder not found: {test_subfolder}")
                    continue
                test_dataset = graph_creation(test_subfolder, folder_type="test_", 
                                            id_mapping=id_mapping, window_size=100)
            else:
                test_subfolder = os.path.join(root_folder, "test_01_known_vehicle_known_attack")
                if not os.path.exists(test_subfolder):
                    print(f"Test folder not found: {test_subfolder}")
                    continue
                test_dataset = graph_creation(test_subfolder, folder_type="test_", 
                                            id_mapping=id_mapping, window_size=100)

            print(f"Loaded {len(test_dataset)} test graphs")

            # OPTIMIZED data loader with large batch size
            test_loader = create_data_loaders(None, test_dataset, None, 512, device)

            # Initialize pipeline
            pipeline = StudentEvaluationPipeline(
                num_ids=len(id_mapping), embedding_dim=8, device=device)
            pipeline.load_student_models(student_autoencoder_path, student_classifier_path, threshold_path)

            # FAST evaluation
            results = evaluate_student_model_fast(pipeline, test_loader, device)

            eval_time = time.time() - eval_start
            print(f"\nEvaluation completed in {eval_time:.2f} seconds")

            # Print results
            print(f"\n=== RESULTS ===")
            
            two_stage = results['two_stage']
            print(f"Two-Stage: Acc={two_stage['accuracy']:.4f}, F1={two_stage['f1']:.4f}")
            
            fusion = results['fusion']
            print(f"Fusion:    Acc={fusion['accuracy']:.4f}, F1={fusion['f1']:.4f}")
            
            anomaly = results['anomaly_only']
            gat = results['gat_only']
            print(f"Anomaly Only: Acc={anomaly['accuracy']:.4f}, F1={anomaly['f1']:.4f}")
            print(f"GAT Only:     Acc={gat['accuracy']:.4f}, F1={gat['f1']:.4f}")

            # FIXED: Show analysis with proper error handling
            try:
                analyze_score_distributions(results)
            except Exception as e:
                print(f"Error in score distribution analysis: {e}")
            
            # FIXED: Show adaptive fusion analysis with better error handling
            try:
                if 'adaptive_alpha' in results and results['adaptive_alpha'] is not None:
                    analyze_fusion_behavior(results, num_samples=15)
                else:
                    print("\nNo adaptive alpha data available for fusion analysis")
            except Exception as e:
                print(f"Error in fusion behavior analysis: {e}")

        except Exception as e:
            print(f"Error evaluating {root_folder}: {str(e)}")
            import traceback
            traceback.print_exc()  # This will show the full error traceback for debugging
            continue  # Continue to next dataset instead of crashing


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")