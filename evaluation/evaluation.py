import numpy as np
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import time
import gc
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report
)
from torch_geometric.data import Batch
import hydra
from omegaconf import DictConfig, OmegaConf

from models.models import GATWithJK, GraphAutoencoderNeighborhood
from old_code.preprocessing import graph_creation, build_id_mapping_from_normal


def create_student_models(num_ids, embedding_dim, device):
    """Create student models with exact same architecture as training."""
    student_autoencoder = GraphAutoencoderNeighborhood(
        num_ids=num_ids, 
        in_channels=11, 
        embedding_dim=embedding_dim,
        hidden_dim=16,
        latent_dim=16,
        num_encoder_layers=2,
        num_decoder_layers=2,
        encoder_heads=2,
        decoder_heads=2
    ).to(device)
    
    student_classifier = GATWithJK(
        num_ids=num_ids, 
        in_channels=11, 
        hidden_channels=16,
        out_channels=1, 
        num_layers=2,
        heads=4,
        embedding_dim=embedding_dim
    ).to(device)
    
    return student_autoencoder, student_classifier


def create_teacher_models(num_ids, embedding_dim, device):
    """Create teacher models with larger architecture."""
    teacher_autoencoder = GraphAutoencoderNeighborhood(
        num_ids=num_ids, 
        in_channels=11, 
        embedding_dim=embedding_dim,
        hidden_dim=32,  
        latent_dim=32,  
        num_encoder_layers=3,  
        num_decoder_layers=3,  
        encoder_heads=4,  
        decoder_heads=4   
    ).to(device)
    
    teacher_classifier = GATWithJK(
        num_ids=num_ids, 
        in_channels=11, 
        hidden_channels=32,  
        out_channels=1, 
        num_layers=5,  
        heads=8,  
        embedding_dim=embedding_dim
    ).to(device)
    
    return teacher_autoencoder, teacher_classifier


def create_optimized_data_loader(dataset, batch_size, device, shuffle=False):
    """Create optimized data loader for evaluation."""
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb < 20:
            eval_batch_size = 512
        else:
            eval_batch_size = 1024
    else:
        eval_batch_size = 256
    
    print(f"Evaluation DataLoader: batch_size={eval_batch_size}, workers=4")
    
    return DataLoader(
        dataset, 
        batch_size=eval_batch_size,
        shuffle=shuffle,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )


def optimize_threshold_for_method(pipeline, validation_loader, method='anomaly_only', target_metric='f1', model_type='student'):
    """Optimize threshold for specific method using validation data."""
    print(f"\n=== OPTIMIZING THRESHOLD FOR {model_type.upper()} {method.upper()} (target: {target_metric}) ===")
    
    all_scores = []
    all_labels = []
    
    # Select appropriate models based on type
    if model_type == 'student':
        autoencoder = pipeline.student_autoencoder
        classifier = pipeline.student_classifier
    else:  # teacher
        autoencoder = pipeline.teacher_autoencoder
        classifier = pipeline.teacher_classifier
    
    autoencoder.eval()
    classifier.eval()
    
    with torch.no_grad():
        for batch in validation_loader:
            batch = batch.to(pipeline.device, non_blocking=True)
            
            # Get anomaly scores
            cont_out, canid_logits, neighbor_logits, _, _ = autoencoder(
                batch.x, batch.edge_index, batch.batch)
            
            node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
            neighbor_targets = autoencoder.create_neighborhood_targets(
                batch.x, batch.edge_index, batch.batch)
            neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                neighbor_logits, neighbor_targets).mean(dim=1)
            
            batch_size = batch.batch.max().item() + 1
            graph_node_errors = torch.zeros(batch_size, device=pipeline.device)
            graph_neighbor_errors = torch.zeros(batch_size, device=pipeline.device)
            
            graph_node_errors.scatter_reduce_(0, batch.batch, node_errors, reduce='amax')
            graph_neighbor_errors.scatter_reduce_(0, batch.batch, neighbor_errors, reduce='amax')
            
            composite_scores = (1.0 * graph_node_errors + 20.0 * graph_neighbor_errors)
            
            all_scores.extend(composite_scores.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    print(f"Validation data: {len(all_labels)} samples")
    print(f"  Normal: {np.sum(all_labels == 0)} ({np.mean(all_labels == 0)*100:.1f}%)")
    print(f"  Attack: {np.sum(all_labels == 1)} ({np.mean(all_labels == 1)*100:.1f}%)")
    
    # Define threshold candidates based on method
    if method == 'anomaly_only':
        score_percentiles = np.linspace(50, 99.5, 100)
        thresholds = [np.percentile(all_scores, p) for p in score_percentiles]
    elif method == 'two_stage':
        score_percentiles = np.linspace(60, 95, 71)
        thresholds = [np.percentile(all_scores, p) for p in score_percentiles]
        # Add attack-focused thresholds
        attack_scores = all_scores[all_labels == 1]
        if len(attack_scores) > 0:
            for p in [10, 5, 1]:
                thresholds.append(np.percentile(attack_scores, p))
    else:
        min_score, max_score = np.min(all_scores), np.max(all_scores)
        thresholds = np.linspace(min_score, max_score, 200)
    
    thresholds = sorted(list(set(thresholds)))
    print(f"Evaluating {len(thresholds)} threshold candidates...")
    
    best_threshold = None
    best_score = -1
    best_metrics = None
    
    # Teacher models typically perform better, so use more realistic GAT accuracy
    if model_type == 'teacher':
        gat_accuracy_on_normal = 0.995  # Teachers are better
        gat_accuracy_on_attack = 0.96   # Teachers are better
    else:
        gat_accuracy_on_normal = 0.98   # Students
        gat_accuracy_on_attack = 0.92   # Students
    
    for threshold in thresholds:
        if method == 'anomaly_only':
            preds = (all_scores > threshold).astype(int)
        elif method == 'two_stage':
            # Simulate GAT performance on detected samples
            anomaly_mask = all_scores > threshold
            preds = np.zeros_like(all_labels)
            
            if np.any(anomaly_mask):
                detected_labels = all_labels[anomaly_mask]
                
                for j, label in enumerate(detected_labels):
                    idx = np.where(anomaly_mask)[0][j]
                    if label == 0:
                        preds[idx] = 1 if np.random.random() > gat_accuracy_on_normal else 0
                    else:
                        preds[idx] = 1 if np.random.random() < gat_accuracy_on_attack else 0
        
        # Compute metrics
        if len(np.unique(preds)) > 1 and len(np.unique(all_labels)) > 1:
            accuracy = accuracy_score(all_labels, preds)
            precision = precision_score(all_labels, preds, zero_division=0)
            recall = recall_score(all_labels, preds, zero_division=0)
            f1 = f1_score(all_labels, preds, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (recall + specificity) / 2
            miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'balanced_accuracy': balanced_acc,
                'miss_rate': miss_rate,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
            
            # Select best based on target metric
            if target_metric == 'f1':
                current_score = f1
            elif target_metric == 'low_miss_rate':
                current_score = 1 - miss_rate
            elif target_metric == 'security_score':
                current_score = 0.7 * recall + 0.3 * precision
            else:
                current_score = accuracy
            
            if current_score > best_score:
                best_score = current_score
                best_threshold = threshold
                best_metrics = metrics.copy()
    
    if best_threshold is None:
        print("WARNING: Could not find optimal threshold, using median")
        best_threshold = np.median(all_scores)
        best_metrics = {'threshold': best_threshold}
    
    print(f"\nBest threshold: {best_threshold:.6f}")
    print(f"Best {target_metric}: {best_score:.4f}")
    
    if best_metrics and 'accuracy' in best_metrics:
        print(f"Detailed metrics:")
        print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {best_metrics['f1']:.4f}")
        print(f"  Miss Rate: {best_metrics['miss_rate']:.4f}")
    
    return best_threshold, best_metrics


def create_validation_split(train_dataset, validation_ratio=0.2, seed=42):
    """Create validation split from training data."""
    np.random.seed(seed)
    
    normal_indices = []
    attack_indices = []
    
    for i, data in enumerate(train_dataset):
        if data.y.item() == 0:
            normal_indices.append(i)
        else:
            attack_indices.append(i)
    
    n_val_normal = int(len(normal_indices) * validation_ratio)
    n_val_attack = int(len(attack_indices) * validation_ratio) if len(attack_indices) > 0 else 0
    
    val_normal_indices = np.random.choice(normal_indices, n_val_normal, replace=False)
    val_attack_indices = np.random.choice(attack_indices, n_val_attack, replace=False) if n_val_attack > 0 else []
    
    val_indices = list(val_normal_indices) + list(val_attack_indices)
    val_dataset = [train_dataset[i] for i in val_indices]
    
    print(f"Validation split: {len(val_normal_indices)} normal, {len(val_attack_indices)} attack")
    return val_dataset


class ComprehensiveEvaluationPipeline:
    """Unified evaluation pipeline for both student and teacher models."""
    
    def __init__(self, num_ids, embedding_dim=8, device='cpu'):
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'
        
        # Create both student and teacher models
        self.student_autoencoder, self.student_classifier = create_student_models(
            num_ids=num_ids, embedding_dim=embedding_dim, device=device
        )
        
        self.teacher_autoencoder, self.teacher_classifier = create_teacher_models(
            num_ids=num_ids, embedding_dim=embedding_dim, device=device
        )
        
        # Thresholds for both model types
        self.student_threshold = 0.0
        self.teacher_threshold = 0.0
        self.student_optimized_thresholds = {}
        self.teacher_optimized_thresholds = {}
        
        # Model availability flags
        self.student_models_loaded = False
        self.teacher_models_loaded = False
        
        print(f"Initialized Comprehensive Evaluation Pipeline on {device}")

    def load_student_models(self, autoencoder_path, classifier_path, threshold_path=None):
        """Load trained student models."""
        if not (os.path.exists(autoencoder_path) and os.path.exists(classifier_path)):
            print("Student models not found, skipping...")
            return False
            
        print(f"Loading student models...")
        
        autoencoder_state_dict = torch.load(autoencoder_path, map_location=self.device, weights_only=True)
        self.student_autoencoder.load_state_dict(autoencoder_state_dict)
        
        classifier_state_dict = torch.load(classifier_path, map_location=self.device, weights_only=True)
        self.student_classifier.load_state_dict(classifier_state_dict)
        
        if threshold_path and os.path.exists(threshold_path):
            threshold_data = torch.load(threshold_path, map_location=self.device, weights_only=True)
            if isinstance(threshold_data, dict) and 'threshold' in threshold_data:
                self.student_threshold = threshold_data['threshold']
            else:
                self.student_threshold = float(threshold_data)
            print(f"Loaded student threshold: {self.student_threshold}")
        else:
            self.student_threshold = None
            print("No student threshold loaded")
        
        self.student_autoencoder.eval()
        self.student_classifier.eval()
        self.student_models_loaded = True
        return True

    def load_teacher_models(self, autoencoder_path, classifier_path, threshold_path=None):
        """Load trained teacher models."""
        if not (os.path.exists(autoencoder_path) and os.path.exists(classifier_path)):
            print("Teacher models not found, skipping...")
            return False
            
        print(f"Loading teacher models...")
        
        autoencoder_state_dict = torch.load(autoencoder_path, map_location=self.device, weights_only=True)
        self.teacher_autoencoder.load_state_dict(autoencoder_state_dict)
        
        classifier_state_dict = torch.load(classifier_path, map_location=self.device, weights_only=True)
        self.teacher_classifier.load_state_dict(classifier_state_dict)
        
        if threshold_path and os.path.exists(threshold_path):
            threshold_data = torch.load(threshold_path, map_location=self.device, weights_only=True)
            if isinstance(threshold_data, dict) and 'threshold' in threshold_data:
                self.teacher_threshold = threshold_data['threshold']
            else:
                self.teacher_threshold = float(threshold_data)
            print(f"Loaded teacher threshold: {self.teacher_threshold}")
        else:
            self.teacher_threshold = None
            print("No teacher threshold loaded")
        
        self.teacher_autoencoder.eval()
        self.teacher_classifier.eval()
        self.teacher_models_loaded = True
        return True

    def optimize_thresholds(self, train_dataset):
        """Optimize thresholds using training data for both model types."""
        print("\n" + "="*60)
        print("OPTIMIZING THRESHOLDS FOR ALL MODELS")
        print("="*60)
        
        val_dataset = create_validation_split(train_dataset, validation_ratio=0.2)
        val_loader = create_optimized_data_loader(val_dataset, 512, self.device)
        
        # Optimize student thresholds if models loaded
        if self.student_models_loaded:
            print("\n" + "="*40)
            print("STUDENT MODEL THRESHOLD OPTIMIZATION")
            print("="*40)
            
            student_anomaly_threshold, student_anomaly_metrics = optimize_threshold_for_method(
                self, val_loader, method='anomaly_only', target_metric='f1', model_type='student'
            )
            
            student_two_stage_threshold, student_two_stage_metrics = optimize_threshold_for_method(
                self, val_loader, method='two_stage', target_metric='low_miss_rate', model_type='student'
            )
            
            self.student_optimized_thresholds = {
                'anomaly_only': {'threshold': student_anomaly_threshold, 'metrics': student_anomaly_metrics},
                'two_stage': {'threshold': student_two_stage_threshold, 'metrics': student_two_stage_metrics}
            }
            
            self.student_threshold = student_two_stage_threshold
        
        # Optimize teacher thresholds if models loaded
        if self.teacher_models_loaded:
            print("\n" + "="*40)
            print("TEACHER MODEL THRESHOLD OPTIMIZATION")
            print("="*40)
            
            teacher_anomaly_threshold, teacher_anomaly_metrics = optimize_threshold_for_method(
                self, val_loader, method='anomaly_only', target_metric='f1', model_type='teacher'
            )
            
            teacher_two_stage_threshold, teacher_two_stage_metrics = optimize_threshold_for_method(
                self, val_loader, method='two_stage', target_metric='low_miss_rate', model_type='teacher'
            )
            
            self.teacher_optimized_thresholds = {
                'anomaly_only': {'threshold': teacher_anomaly_threshold, 'metrics': teacher_anomaly_metrics},
                'two_stage': {'threshold': teacher_two_stage_threshold, 'metrics': teacher_two_stage_metrics}
            }
            
            self.teacher_threshold = teacher_two_stage_threshold
        
        return {
            'student': self.student_optimized_thresholds if self.student_models_loaded else None,
            'teacher': self.teacher_optimized_thresholds if self.teacher_models_loaded else None
        }

    def predict_batch_comprehensive(self, data, model_type='both'):
        """Comprehensive prediction with specified model type."""
        if self.is_cuda:
            data = data.to(self.device, non_blocking=True)
        else:
            data = data.to(self.device)
        
        results = {}
        
        with torch.no_grad():
            # Student model predictions
            if model_type in ['student', 'both'] and self.student_models_loaded:
                student_results = self._predict_with_models(
                    data, self.student_autoencoder, self.student_classifier,
                    self.student_threshold, self.student_optimized_thresholds
                )
                results['student'] = student_results
            
            # Teacher model predictions
            if model_type in ['teacher', 'both'] and self.teacher_models_loaded:
                teacher_results = self._predict_with_models(
                    data, self.teacher_autoencoder, self.teacher_classifier,
                    self.teacher_threshold, self.teacher_optimized_thresholds
                )
                results['teacher'] = teacher_results
        
        return results

    def _predict_with_models(self, data, autoencoder, classifier, threshold, optimized_thresholds):
        """Helper method to make predictions with given models."""
        # Autoencoder forward pass
        cont_out, canid_logits, neighbor_logits, _, _ = autoencoder(
            data.x, data.edge_index, data.batch)
        
        # Compute anomaly scores
        node_errors = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
        neighbor_targets = autoencoder.create_neighborhood_targets(
            data.x, data.edge_index, data.batch)
        neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
            neighbor_logits, neighbor_targets).mean(dim=1)
        
        batch_size = data.batch.max().item() + 1
        graph_node_errors = torch.zeros(batch_size, device=self.device)
        graph_neighbor_errors = torch.zeros(batch_size, device=self.device)
        
        graph_node_errors.scatter_reduce_(0, data.batch, node_errors, reduce='amax')
        graph_neighbor_errors.scatter_reduce_(0, data.batch, neighbor_errors, reduce='amax')
        
        raw_anomaly_scores = (1.0 * graph_node_errors + 20.0 * graph_neighbor_errors)
        
        # Classifier forward pass
        classifier_logits = classifier(data).squeeze()
        
        # Normalize scores
        if raw_anomaly_scores.max() > raw_anomaly_scores.min():
            norm_anomaly_scores = (raw_anomaly_scores - raw_anomaly_scores.min()) / (
                raw_anomaly_scores.max() - raw_anomaly_scores.min())
        else:
            norm_anomaly_scores = torch.zeros_like(raw_anomaly_scores)
        
        gat_probs = torch.sigmoid(classifier_logits)
        
        # Two-stage predictions
        two_stage_preds = torch.zeros(batch_size, device=self.device)
        if threshold is not None:
            anomaly_mask = raw_anomaly_scores > threshold
            two_stage_preds[anomaly_mask] = (classifier_logits[anomaly_mask] > 0.0).float()
        
        # Optimized predictions if available
        optimized_anomaly_preds = torch.zeros(batch_size, device=self.device)
        optimized_two_stage_preds = torch.zeros(batch_size, device=self.device)
        
        if 'anomaly_only' in optimized_thresholds:
            opt_threshold = optimized_thresholds['anomaly_only']['threshold']
            optimized_anomaly_preds = (raw_anomaly_scores > opt_threshold).float()
        
        if 'two_stage' in optimized_thresholds:
            opt_threshold = optimized_thresholds['two_stage']['threshold']
            opt_anomaly_mask = raw_anomaly_scores > opt_threshold
            optimized_two_stage_preds[opt_anomaly_mask] = (classifier_logits[opt_anomaly_mask] > 0.0).float()
        
        # Fusion strategy (GAT-dominant)
        performance_weight_gat = 0.85
        performance_weight_anomaly = 0.15
        fusion_scores = (performance_weight_anomaly * norm_anomaly_scores + 
                       performance_weight_gat * gat_probs)
        fusion_preds = (fusion_scores > 0.5).float()
        
        return {
            'two_stage_preds': two_stage_preds.long(),
            'fusion_preds': fusion_preds.long(),
            'optimized_anomaly_preds': optimized_anomaly_preds.long(),
            'optimized_two_stage_preds': optimized_two_stage_preds.long(),
            'anomaly_scores': norm_anomaly_scores,
            'gat_probs': gat_probs,
            'raw_anomaly_scores': raw_anomaly_scores
        }


def compute_comprehensive_metrics(y_true, y_pred, name="method"):
    """Compute comprehensive metrics for evaluation."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Support counts
    metrics['support_class_0'] = np.sum(y_true == 0)
    metrics['support_class_1'] = np.sum(y_true == 1)
    metrics['total_samples'] = len(y_true)
    
    # Error counts
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    metrics['true_negatives'] = tn
    
    return metrics


def evaluate_comprehensive(pipeline, test_loader, device):
    """Comprehensive evaluation with all available models."""
    print(f"Comprehensive evaluation with {len(test_loader)} batches...")
    
    all_labels = []
    student_results = {
        'two_stage_preds': [], 'fusion_preds': [], 'optimized_anomaly_preds': [],
        'optimized_two_stage_preds': [], 'anomaly_scores': [], 'gat_probs': []
    }
    teacher_results = {
        'two_stage_preds': [], 'fusion_preds': [], 'optimized_anomaly_preds': [],
        'optimized_two_stage_preds': [], 'anomaly_scores': [], 'gat_probs': []
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device, non_blocking=True)
            
            batch_results = pipeline.predict_batch_comprehensive(batch, model_type='both')
            
            all_labels.append(batch.y.cpu())
            
            # Collect student results
            if 'student' in batch_results:
                for key in student_results:
                    if key in batch_results['student']:
                        student_results[key].append(batch_results['student'][key].cpu())
            
            # Collect teacher results
            if 'teacher' in batch_results:
                for key in teacher_results:
                    if key in batch_results['teacher']:
                        teacher_results[key].append(batch_results['teacher'][key].cpu())
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(test_loader)}")
    
    # Concatenate results
    all_labels = torch.cat(all_labels).numpy()
    
    # Process student results
    for key in student_results:
        if len(student_results[key]) > 0:
            student_results[key] = torch.cat(student_results[key]).numpy()
    
    # Process teacher results
    for key in teacher_results:
        if len(teacher_results[key]) > 0:
            teacher_results[key] = torch.cat(teacher_results[key]).numpy()
    
    # Compute metrics for all methods
    results = {'labels': all_labels}
    
    # Student model metrics
    if pipeline.student_models_loaded and len(student_results['anomaly_scores']) > 0:
        anomaly_only_preds = (student_results['anomaly_scores'] > 0.5).astype(int)
        gat_only_preds = (student_results['gat_probs'] > 0.5).astype(int)
        
        results['student_two_stage'] = compute_comprehensive_metrics(all_labels, student_results['two_stage_preds'], 'student_two_stage')
        results['student_fusion'] = compute_comprehensive_metrics(all_labels, student_results['fusion_preds'], 'student_fusion')
        results['student_anomaly_only'] = compute_comprehensive_metrics(all_labels, anomaly_only_preds, 'student_anomaly_only')
        results['student_gat_only'] = compute_comprehensive_metrics(all_labels, gat_only_preds, 'student_gat_only')
        
        # Steelmanned student methods
        if len(student_results['optimized_anomaly_preds']) > 0:
            results['student_steelmanned_anomaly'] = compute_comprehensive_metrics(
                all_labels, student_results['optimized_anomaly_preds'], 'student_steelmanned_anomaly')
        
        if len(student_results['optimized_two_stage_preds']) > 0:
            results['student_steelmanned_two_stage'] = compute_comprehensive_metrics(
                all_labels, student_results['optimized_two_stage_preds'], 'student_steelmanned_two_stage')
    
    # Teacher model metrics
    if pipeline.teacher_models_loaded and len(teacher_results['anomaly_scores']) > 0:
        anomaly_only_preds = (teacher_results['anomaly_scores'] > 0.5).astype(int)
        gat_only_preds = (teacher_results['gat_probs'] > 0.5).astype(int)
        
        results['teacher_two_stage'] = compute_comprehensive_metrics(all_labels, teacher_results['two_stage_preds'], 'teacher_two_stage')
        results['teacher_fusion'] = compute_comprehensive_metrics(all_labels, teacher_results['fusion_preds'], 'teacher_fusion')
        results['teacher_anomaly_only'] = compute_comprehensive_metrics(all_labels, anomaly_only_preds, 'teacher_anomaly_only')
        results['teacher_gat_only'] = compute_comprehensive_metrics(all_labels, gat_only_preds, 'teacher_gat_only')
        
        # Steelmanned teacher methods
        if len(teacher_results['optimized_anomaly_preds']) > 0:
            results['teacher_steelmanned_anomaly'] = compute_comprehensive_metrics(
                all_labels, teacher_results['optimized_anomaly_preds'], 'teacher_steelmanned_anomaly')
        
        if len(teacher_results['optimized_two_stage_preds']) > 0:
            results['teacher_steelmanned_two_stage'] = compute_comprehensive_metrics(
                all_labels, teacher_results['optimized_two_stage_preds'], 'teacher_steelmanned_two_stage')
    
    return results


def print_results(results, dataset_name):
    """Print comprehensive results for both student and teacher models."""
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*100}")
    
    # Separate student and teacher methods
    student_methods = []
    teacher_methods = []
    all_method_names = []
    
    method_mapping = {
        'student_two_stage': 'S-TwoStage',
        'student_fusion': 'S-Fusion',
        'student_anomaly_only': 'S-Anomaly',
        'student_gat_only': 'S-GAT',
        'student_steelmanned_anomaly': 'S-Steel-Anomaly',
        'student_steelmanned_two_stage': 'S-Steel-TwoStage',
        'teacher_two_stage': 'T-TwoStage',
        'teacher_fusion': 'T-Fusion',
        'teacher_anomaly_only': 'T-Anomaly',
        'teacher_gat_only': 'T-GAT',
        'teacher_steelmanned_anomaly': 'T-Steel-Anomaly',
        'teacher_steelmanned_two_stage': 'T-Steel-TwoStage'
    }
    
    # Collect available methods
    available_methods = []
    available_names = []
    
    for method_key, display_name in method_mapping.items():
        if method_key in results:
            available_methods.append(method_key)
            available_names.append(display_name)
    
    if not available_methods:
        print("No model results available!")
        return
    
    # Metrics table
    metrics = [('Accuracy', 'accuracy'), ('Precision', 'precision'), ('Recall', 'recall'), 
               ('F1-Score', 'f1'), ('Specificity', 'specificity'), ('Balanced Acc', 'balanced_accuracy')]
    
    print(f"\n{'Metric':<15}", end='')
    for name in available_names:
        print(f"{name:>15}", end='')
    print()
    print('-' * (15 + 15 * len(available_names)))
    
    for metric_name, metric_key in metrics:
        print(f"{metric_name:<15}", end='')
        for method in available_methods:
            if method in results:
                value = results[method][metric_key]
                print(f"{value:>15.4f}", end='')
            else:
                print(f"{'N/A':>15}", end='')
        print()
    
    # Dataset characteristics (use any available method for stats)
    first_method = available_methods[0]
    total = results[first_method]['total_samples']
    normal = results[first_method]['support_class_0']
    attack = results[first_method]['support_class_1']
    
    print(f"\nDataset: {total:,} samples ({normal:,} normal, {attack:,} attack)")
    print(f"Class imbalance: {normal/attack:.1f}:1 (Normal:Attack)")
    
    # Model comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    # Student vs Teacher comparison
    if 'student_gat_only' in results and 'teacher_gat_only' in results:
        student_f1 = results['student_gat_only']['f1']
        teacher_f1 = results['teacher_gat_only']['f1']
        improvement = teacher_f1 - student_f1
        print(f"GAT Performance - Student: {student_f1:.4f}, Teacher: {teacher_f1:.4f} (Δ: {improvement:+.4f})")
    
    if 'student_fusion' in results and 'teacher_fusion' in results:
        student_f1 = results['student_fusion']['f1']
        teacher_f1 = results['teacher_fusion']['f1']
        improvement = teacher_f1 - student_f1
        print(f"Fusion Performance - Student: {student_f1:.4f}, Teacher: {teacher_f1:.4f} (Δ: {improvement:+.4f})")
    
    # Show steelmanned improvements
    print(f"\n{'='*60}")
    print("STEELMANNED IMPROVEMENTS")
    print(f"{'='*60}")
    
    for model_type in ['student', 'teacher']:
        if f'{model_type}_steelmanned_anomaly' in results and f'{model_type}_anomaly_only' in results:
            original_f1 = results[f'{model_type}_anomaly_only']['f1']
            steelmanned_f1 = results[f'{model_type}_steelmanned_anomaly']['f1']
            improvement = steelmanned_f1 - original_f1
            print(f"{model_type.title()} Anomaly-Only F1 improvement: {improvement:+.4f}")
        
        if f'{model_type}_steelmanned_two_stage' in results and f'{model_type}_two_stage' in results:
            original_f1 = results[f'{model_type}_two_stage']['f1']
            steelmanned_f1 = results[f'{model_type}_steelmanned_two_stage']['f1']
            improvement = steelmanned_f1 - original_f1
            print(f"{model_type.title()} Two-Stage F1 improvement: {improvement:+.4f}")


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    """Main evaluation function for both student and teacher models."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataset from config
    config_dict = OmegaConf.to_container(config, resolve=True)
    dataset_key = config_dict['root_folder']
    
    # Dataset paths
    root_folders = {
        'hcrl_ch': r"datasets/can-train-and-test-v1.5/hcrl-ch",
        'hcrl_sa': r"datasets/can-train-and-test-v1.5/hcrl-sa",
        'set_01': r"datasets/can-train-and-test-v1.5/set_01",
        'set_02': r"datasets/can-train-and-test-v1.5/set_02",
        'set_03': r"datasets/can-train-and-test-v1.5/set_03",
        'set_04': r"datasets/can-train-and-test-v1.5/set_04",
    }
    
    if dataset_key not in root_folders:
        print(f"Dataset {dataset_key} not found!")
        return
    
    root_folder = root_folders[dataset_key]
    
    # Model paths
    student_autoencoder_path = f"saved_models/student_autoencoder_{dataset_key}.pth"
    student_classifier_path = f"saved_models/student_classifier_{dataset_key}.pth"
    student_threshold_path = f"saved_models/student_threshold_{dataset_key}.pth"
    
    teacher_autoencoder_path = f"saved_models/teacher_autoencoder_{dataset_key}.pth"
    teacher_classifier_path = f"saved_models/teacher_classifier_{dataset_key}.pth"
    teacher_threshold_path = f"saved_models/teacher_threshold_{dataset_key}.pth"
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {dataset_key}")
    print(f"{'='*60}")
    
    try:
        eval_start = time.time()
        
        # Build dataset
        id_mapping = build_id_mapping_from_normal(root_folder)
        
        # Load training data for threshold optimization
        train_folder = os.path.join(root_folder, "training")
        train_dataset = None
        if os.path.exists(train_folder):
            print("Loading training data for threshold optimization...")
            train_dataset = graph_creation(train_folder, folder_type="training_", 
                                         id_mapping=id_mapping, window_size=100)
            print(f"Loaded {len(train_dataset)} training graphs")
        
        # Load test data
        if "hcrl-ch" in root_folder:
            combined_dataset = []
            for subfolder_name in os.listdir(root_folder):
                subfolder_path = os.path.join(root_folder, subfolder_name)
                if os.path.isdir(subfolder_path) and subfolder_name.startswith("test_"):
                    test_data = graph_creation(subfolder_path, folder_type="test_", 
                                             id_mapping=id_mapping, window_size=100)
                    combined_dataset.extend(test_data)
            test_dataset = combined_dataset
        else:
            test_subfolder = os.path.join(root_folder, "test_01_known_vehicle_known_attack")
            if not os.path.exists(test_subfolder):
                print(f"Test folder not found: {test_subfolder}")
                return
            test_dataset = graph_creation(test_subfolder, folder_type="test_", 
                                        id_mapping=id_mapping, window_size=100)
        
        print(f"Loaded {len(test_dataset)} test graphs")
        
        # Create data loader
        test_loader = create_optimized_data_loader(test_dataset, 512, device)
        
        # Initialize comprehensive pipeline
        pipeline = ComprehensiveEvaluationPipeline(
            num_ids=len(id_mapping), embedding_dim=8, device=device)
        
        # Load student models
        student_loaded = pipeline.load_student_models(
            student_autoencoder_path, student_classifier_path, student_threshold_path)
        
        # Load teacher models
        teacher_loaded = pipeline.load_teacher_models(
            teacher_autoencoder_path, teacher_classifier_path, teacher_threshold_path)
        
        if not student_loaded and not teacher_loaded:
            print("No models found! Please train models first.")
            return
        
        # Optimize thresholds if training data available
        if train_dataset is not None and len(train_dataset) > 100:
            pipeline.optimize_thresholds(train_dataset)
        else:
            print("Skipping threshold optimization (insufficient training data)")
        
        # Comprehensive evaluation
        results = evaluate_comprehensive(pipeline, test_loader, device)
        
        eval_time = time.time() - eval_start
        print(f"\nEvaluation completed in {eval_time:.2f} seconds")
        
        # Print results
        print_results(results, dataset_key)
        
        # Memory cleanup
        del pipeline, test_loader, test_dataset
        if train_dataset is not None:
            del train_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error evaluating {dataset_key}: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")