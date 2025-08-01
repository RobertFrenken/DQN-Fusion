import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import torch
import sys
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import json
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

class StudentEvaluationPipeline:
    """Evaluation pipeline for knowledge distilled student models."""
    
    def __init__(self, num_ids, embedding_dim=8, device='cpu'):
        """Initialize the evaluation pipeline with student models."""
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'
        
        # FIXED: Create student models directly
        self.student_autoencoder, self.student_classifier = create_student_models(
            num_ids=num_ids, embedding_dim=embedding_dim, device=device
        )
        
        self.threshold = 0.0
        print(f"Initialized Student Evaluation Pipeline on {device}")

    def load_student_models(self, autoencoder_path, classifier_path, threshold_path=None):
        """Load the trained student models and threshold."""
        print(f"Loading student autoencoder from: {autoencoder_path}")
        autoencoder_state_dict = torch.load(autoencoder_path, map_location=self.device)
        self.student_autoencoder.load_state_dict(autoencoder_state_dict)
        
        print(f"Loading student classifier from: {classifier_path}")
        classifier_state_dict = torch.load(classifier_path, map_location=self.device)
        self.student_classifier.load_state_dict(classifier_state_dict)
        
        # FIXED: Try to load the correct threshold from training
        if threshold_path and os.path.exists(threshold_path):
            threshold_data = torch.load(threshold_path, map_location=self.device)
            if isinstance(threshold_data, dict) and 'threshold' in threshold_data:
                self.threshold = threshold_data['threshold']
            else:
                self.threshold = float(threshold_data)
            print(f"Loaded threshold: {self.threshold}")
        else:
            # FIXED: Use a more reasonable default threshold
            self.threshold = 0.01  # Much lower threshold
            print(f"Using default threshold: {self.threshold}")
        
        # Set to eval mode
        self.student_autoencoder.eval()
        self.student_classifier.eval()
        
        print("Student models loaded and set to evaluation mode!")

    def set_threshold(self, threshold):
        """Set the anomaly detection threshold."""
        self.threshold = threshold

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
    
def evaluate_student_model(pipeline, data_loader, device, evaluation_mode='fusion'):
    """
    Evaluate student model with different prediction modes.
    """
    all_labels = []
    all_two_stage_preds = []
    all_fusion_preds = []
    all_anomaly_scores = []
    all_gat_probs = []

    # FIXED: Check if data_loader is empty
    if len(data_loader) == 0:
        print("Warning: DataLoader is empty, cannot evaluate")
        return {'error': 'Empty dataset'}

    # Collect all predictions and labels
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Two-stage prediction
            if evaluation_mode in ['two_stage', 'both']:
                two_stage_preds = pipeline.predict_two_stage(batch)
                all_two_stage_preds.append(two_stage_preds.cpu())
            
            # Fusion prediction
            if evaluation_mode in ['fusion', 'both']:
                fusion_preds, anomaly_scores, gat_probs = pipeline.predict_with_fusion(
                    batch, fusion_method='weighted', alpha=0.6)
                all_fusion_preds.append(fusion_preds.cpu())
                all_anomaly_scores.append(anomaly_scores.cpu())
                all_gat_probs.append(gat_probs.cpu())
                # REMOVED: Don't call evaluate_fusion_methods here!
            
            all_labels.append(batch.y.cpu())

    # FIXED: Check if we have any data before concatenation
    if not all_labels:
        print("Warning: No data collected, cannot evaluate")
        return {'error': 'No data collected'}

    # Convert to numpy arrays
    all_labels = torch.cat(all_labels).numpy()
    
    results = {}
    
    if evaluation_mode in ['two_stage', 'both'] and all_two_stage_preds:
        all_two_stage_preds = torch.cat(all_two_stage_preds).numpy()
        
        # Compute metrics for two-stage
        accuracy = accuracy_score(all_labels, all_two_stage_preds)
        precision = precision_score(all_labels, all_two_stage_preds, zero_division=0)
        recall = recall_score(all_labels, all_two_stage_preds, zero_division=0)
        f1 = f1_score(all_labels, all_two_stage_preds, zero_division=0)
        
        results['two_stage'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_two_stage_preds,
            'confusion_matrix': confusion_matrix(all_labels, all_two_stage_preds)
        }
    
    if evaluation_mode in ['fusion', 'both'] and all_fusion_preds:
        all_fusion_preds = torch.cat(all_fusion_preds).numpy()
        all_anomaly_scores = torch.cat(all_anomaly_scores).numpy()
        all_gat_probs = torch.cat(all_gat_probs).numpy()
        
        # FIXED: NOW call evaluate_fusion_methods ONCE after collecting all data
        fusion_results = evaluate_fusion_methods(
            all_labels, all_anomaly_scores, all_gat_probs)
        
        results['fusion'] = fusion_results
        results['anomaly_scores'] = all_anomaly_scores
        results['gat_probs'] = all_gat_probs
    
    results['labels'] = all_labels
    return results

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
        
def evaluate_fusion_methods(all_labels, all_anomaly_scores, all_gat_probs):
    """Evaluate multiple fusion methods and return detailed results."""
    # Convert back to raw logits for better fusion strategies
    # Approximate GAT logits from probabilities (not perfect but workable)
    epsilon = 1e-8
    gat_probs_clipped = np.clip(all_gat_probs, epsilon, 1-epsilon)
    approx_gat_logits = np.log(gat_probs_clipped / (1 - gat_probs_clipped))
    
    fusion_methods = ['conservative', 'hierarchical', 'confidence_weighted', 'gat_priority']
    results = {}
    
    print_sample_evaluations(all_labels, all_anomaly_scores, all_gat_probs, num_samples=15)
    
    # Print score distribution statistics
    print(f"\n=== Score Distribution Statistics ===")
    print(f"Anomaly Scores - Mean: {np.mean(all_anomaly_scores):.4f}, Std: {np.std(all_anomaly_scores):.4f}")
    print(f"GAT Probabilities - Mean: {np.mean(all_gat_probs):.4f}, Std: {np.std(all_gat_probs):.4f}")
    print(f"Approx GAT Logits - Mean: {np.mean(approx_gat_logits):.4f}, Std: {np.std(approx_gat_logits):.4f}")
    
    # Show distribution by class
    normal_mask = all_labels == 0
    attack_mask = all_labels == 1
    
    if np.any(normal_mask):
        print(f"\nNormal samples (label=0):")
        print(f"  Anomaly scores - Mean: {np.mean(all_anomaly_scores[normal_mask]):.4f}")
        print(f"  GAT probs - Mean: {np.mean(all_gat_probs[normal_mask]):.4f}")
    
    if np.any(attack_mask):
        print(f"Attack samples (label=1):")
        print(f"  Anomaly scores - Mean: {np.mean(all_anomaly_scores[attack_mask]):.4f}")
        print(f"  GAT probs - Mean: {np.mean(all_gat_probs[attack_mask]):.4f}")
    
    print(f"\n=== Fusion Strategy Evaluation ===")
    
    # Test different strategies
    strategies = {
        'conservative': lambda a, g: (a > 0.5) & (g > 0.5),  # Both must agree
        'hierarchical': lambda a, g: np.where(a > 0.1, g > 0.5, 0),  # Anomaly first, then GAT
        'gat_priority': lambda a, g: np.where(np.abs(approx_gat_logits) > 2, g > 0.5, a > 0.5),  # GAT priority if confident
        'weighted_09': lambda a, g: (0.9 * a + 0.1 * g) > 0.5,  # Heavily weight anomaly
        'weighted_01': lambda a, g: (0.1 * a + 0.9 * g) > 0.5,  # Heavily weight GAT
    }
    
    best_acc = 0
    best_method = None
    
    for method_name, strategy_func in strategies.items():
        preds = strategy_func(all_anomaly_scores, all_gat_probs).astype(int)
        
        acc = accuracy_score(all_labels, preds)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        print(f"\n--- {method_name.replace('_', ' ').title()} Strategy ---")
        print(f"  Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_method = method_name
        
        results[method_name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': preds,
            'confusion_matrix': confusion_matrix(all_labels, preds)
        }
    
    # Individual component performance
    print(f"\n=== Individual Component Performance ===")
    
    anomaly_only_preds = (all_anomaly_scores > 0.5).astype(int)
    gat_only_preds = (all_gat_probs > 0.5).astype(int)
    
    anomaly_acc = accuracy_score(all_labels, anomaly_only_preds)
    gat_acc = accuracy_score(all_labels, gat_only_preds)
    
    print(f"Anomaly Detection Only: Acc={anomaly_acc:.4f}")
    print(f"GAT Classification Only: Acc={gat_acc:.4f}")
    
    print(f"\nBest Strategy: {best_method} (Accuracy: {best_acc:.4f})")
    
    return results

# Remove the debugging prints from predict_with_fusion (they're cluttering the output):

def predict_with_fusion(self, data, fusion_method='weighted', alpha=0.6):
    """
    Fusion-based prediction combining RAW anomaly scores and GAT probabilities.
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
        
        # Compute RAW composite anomaly scores (same weights as training)
        weight_node = 1.0
        weight_neighbor = 20.0
        raw_anomaly_scores = (weight_node * graph_node_errors + 
                            weight_neighbor * graph_neighbor_errors)
        
        # Get GAT classification probabilities
        graphs = Batch.to_data_list(data)
        gat_probs = []
        
        for i, graph in enumerate(graphs):
            graph_batch = graph.to(self.device)
            gat_logit = self.student_classifier(graph_batch)
            
            # REMOVED: Debugging prints - they clutter the output
            gat_prob = torch.sigmoid(gat_logit).item()
            gat_probs.append(gat_prob)
        
        gat_probs = torch.tensor(gat_probs, device=self.device)
        
        # Normalize raw anomaly scores to [0,1] for fusion
        if raw_anomaly_scores.max() > raw_anomaly_scores.min():
            normalized_anomaly_scores = (raw_anomaly_scores - raw_anomaly_scores.min()) / (
                raw_anomaly_scores.max() - raw_anomaly_scores.min())
        else:
            normalized_anomaly_scores = raw_anomaly_scores
        
        # Apply fusion between normalized anomaly scores and GAT probabilities
        if fusion_method == 'weighted':
            fused_scores = alpha * normalized_anomaly_scores + (1 - alpha) * gat_probs
        elif fusion_method == 'product':
            fused_scores = (normalized_anomaly_scores * gat_probs) ** 0.5
        elif fusion_method == 'max':
            fused_scores = torch.maximum(normalized_anomaly_scores, gat_probs)
        elif fusion_method == 'learned':
            fused_scores = 0.7 * normalized_anomaly_scores + 0.3 * gat_probs
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Final predictions: threshold the fused scores
        final_preds = (fused_scores > 0.5).long()
        
        return final_preds, normalized_anomaly_scores, gat_probs

def main(evaluate_known_only=True):
    """Main evaluation function for knowledge distilled student models."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # FIXED: Check what models actually exist
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
        threshold_path = f"saved_models/student_threshold_{model_key}.pth"  # Try to load threshold
        
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
        print("Please run knowledge distillation training first.")
        return

    print("=" * 80)
    print("KNOWLEDGE DISTILLED STUDENT MODEL EVALUATION")
    print("=" * 80)
    print(f"Found student models for: {', '.join(available_models)}")

    for dataset_info in test_datasets:
        root_folder = dataset_info["folder"]
        student_autoencoder_path = dataset_info["student_autoencoder"]
        student_classifier_path = dataset_info["student_classifier"]
        threshold_path = dataset_info["threshold"]

        print(f"\nEvaluating student models for dataset: {root_folder}")

        try:
            # Build ID mapping and create dataset
            id_mapping = build_id_mapping_from_normal(root_folder)
            
            # FIXED: Load ONLY known vehicle, known attack for fair comparison
            if "hcrl-ch" in root_folder:
                # For hcrl-ch, load all test subfolders
                combined_dataset = []
                subfolder_found = False
                
                for subfolder_name in os.listdir(root_folder):
                    subfolder_path = os.path.join(root_folder, subfolder_name)
                    
                    if os.path.isdir(subfolder_path) and subfolder_name.startswith("test_"):
                        subfolder_found = True
                        print(f"Processing subfolder: {subfolder_path}")
                        test_data = graph_creation(subfolder_path, folder_type="test_", 
                                                 id_mapping=id_mapping, window_size=100)
                        combined_dataset.extend(test_data)
                
                test_dataset = combined_dataset
                
            elif "hcrl-sa" in root_folder:
                # FIXED: Load ONLY known vehicle, known attack
                test_subfolder = os.path.join(root_folder, "test_01_known_vehicle_known_attack")
                if not os.path.exists(test_subfolder):
                    print(f"Known vehicle/known attack test folder not found: {test_subfolder}")
                    continue
                    
                print(f"Loading only known vehicle/known attack: {test_subfolder}")
                test_dataset = graph_creation(test_subfolder, folder_type="test_", 
                                            id_mapping=id_mapping, window_size=100)
                
            else:
                # Standard dataset structure
                test_subfolder = os.path.join(root_folder, "test_01_known_vehicle_known_attack")
                if not os.path.exists(test_subfolder):
                    print(f"Test subfolder not found: {test_subfolder}")
                    continue
                    
                test_dataset = graph_creation(test_subfolder, folder_type="test_", 
                                            id_mapping=id_mapping, window_size=100)

            print(f"Loaded {len(test_dataset)} test graphs with {len(id_mapping)} unique CAN IDs")

            # Create data loader
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                                   pin_memory=(device.type == 'cuda'), num_workers=0)

            # Initialize student evaluation pipeline
            pipeline = StudentEvaluationPipeline(
                num_ids=len(id_mapping), embedding_dim=8, device=device)

            # FIXED: Load models with proper threshold
            pipeline.load_student_models(student_autoencoder_path, student_classifier_path, threshold_path)

            # Evaluate with both methods
            print("\n--- Evaluating Student Models ---")
            results = evaluate_student_model(pipeline, test_loader, device, evaluation_mode='both')

            # Print results
            print(f"\n=== Results for {root_folder.split('/')[-1]} ===")
            
            # Two-stage results
            if 'two_stage' in results:
                two_stage = results['two_stage']
                print(f"\nTwo-Stage Prediction:")
                print(f"  Accuracy:  {two_stage['accuracy']:.4f}")
                print(f"  Precision: {two_stage['precision']:.4f}")
                print(f"  Recall:    {two_stage['recall']:.4f}")
                print(f"  F1 Score:  {two_stage['f1']:.4f}")
                print(f"  Confusion Matrix:")
                print(f"  {two_stage['confusion_matrix']}")

            # Rest of the results printing code stays the same...
            if 'fusion' in results:
                fusion_results = results['fusion']
                print(f"\nFusion-Based Prediction:")
                
                print(f"\nIndividual Components:")
                anomaly_only = fusion_results['anomaly_only']
                gat_only = fusion_results['gat_only']
                print(f"  Anomaly Detection Only: Acc={anomaly_only['accuracy']:.4f}, "
                      f"F1={anomaly_only['f1']:.4f}")
                print(f"  GAT Classification Only: Acc={gat_only['accuracy']:.4f}, "
                      f"F1={gat_only['f1']:.4f}")
                
                print(f"\nFusion Methods:")
                for method in ['weighted', 'product']:  # Updated to match your fusion_methods
                    if method in fusion_results:
                        result = fusion_results[method]
                        alpha_str = f" (α={result['alpha']:.1f})" if 'alpha' in result else ""
                        print(f"  {method.capitalize()}{alpha_str}: Acc={result['accuracy']:.4f}, "
                              f"F1={result['f1']:.4f}, Prec={result['precision']:.4f}, "
                              f"Rec={result['recall']:.4f}")

                # Find best fusion method
                available_methods = [m for m in ['weighted', 'product'] if m in fusion_results]
                if available_methods:
                    best_method = max(available_methods, 
                                    key=lambda x: fusion_results[x]['accuracy'])
                    print(f"\nBest Fusion Method: {best_method} "
                          f"(Accuracy: {fusion_results[best_method]['accuracy']:.4f})")
                    print(f"Best Fusion Confusion Matrix:")
                    print(fusion_results[best_method]['confusion_matrix'])

            print("-" * 80)

        except Exception as e:
            print(f"Error evaluating {root_folder}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")