"""
Metric computation module for comprehensive model evaluation.

Provides functions to compute classification, security, and threshold-independent metrics
for binary classification tasks (normal vs attack detection).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, auc, roc_curve, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score, classification_report
)
from typing import Dict, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                  sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground truth labels (0=normal, 1=attack)
        y_pred: Predicted labels (binary)
        sample_weight: Optional sample weights

    Returns:
        Dictionary with:
        - accuracy, precision, recall, f1
        - specificity, balanced_accuracy
        - mcc, kappa
    """
    if len(np.unique(y_true)) < 2:
        # Only one class present
        return {
            'accuracy': float(np.mean(y_true == y_pred)),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'specificity': 0.0,
            'balanced_accuracy': 0.0,
            'mcc': 0.0,
            'kappa': 0.0
        }

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    precision = precision_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
    recall = recall_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
    f1 = f1_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)

    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2

    # Correlation metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'balanced_accuracy': float(balanced_accuracy),
        'mcc': float(mcc),
        'kappa': float(kappa)
    }


def compute_security_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute security-focused metrics for intrusion detection.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with:
        - false_positive_rate (FPR)
        - false_negative_rate (FNR)
        - true_positive_rate (TPR) / recall
        - true_negative_rate (TNR) / specificity
        - detection_rate (same as recall)
        - miss_rate (same as FNR)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # = specificity

    return {
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'true_positive_rate': float(tpr),
        'true_negative_rate': float(tnr),
        'detection_rate': float(tpr),  # Same as recall for attack detection
        'miss_rate': float(fnr),       # Same as FNR
        'false_alarms': int(fp),
        'missed_attacks': int(fn),
        'correct_detections': int(tp),
        'correct_normal': int(tn)
    }


def compute_threshold_independent_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics independent of classification threshold (ROC-AUC, PR-AUC).

    Args:
        y_true: Ground truth labels
        y_scores: Confidence scores (continuous, 0-1)

    Returns:
        Dictionary with:
        - roc_auc: Area under ROC curve
        - pr_auc: Area under precision-recall curve
    """
    # Ensure scores are between 0 and 1
    y_scores = np.clip(y_scores, 0, 1)

    if len(np.unique(y_true)) < 2:
        return {'roc_auc': 0.0, 'pr_auc': 0.0}

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = 0.0

    # Precision-Recall AUC
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_vals, precision_vals)
    except ValueError:
        pr_auc = 0.0

    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    }


def detect_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray,
                            metric: str = 'f1') -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold that maximizes a target metric.

    Args:
        y_true: Ground truth labels
        y_scores: Confidence scores (continuous)
        metric: Metric to optimize ('f1', 'balanced_accuracy', 'youden')

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, {}

    # Get ROC curve for threshold search
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    best_threshold = 0.5
    best_score = -1
    best_metrics = {}

    # Test a range of thresholds
    for threshold in np.linspace(0, 1, 101):
        y_pred = (y_scores >= threshold).astype(int)

        if len(np.unique(y_pred)) < 2:
            continue

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced_accuracy':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = (recall + specificity) / 2
        elif metric == 'youden':
            # Youden's J = TPR + TNR - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = tpr + tnr - 1
        else:
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'threshold': float(threshold),
                metric: float(score),
                **compute_classification_metrics(y_true, y_pred),
                **compute_security_metrics(y_true, y_pred)
            }

    return best_threshold, best_metrics


def compute_detection_rate_at_fpr(y_true: np.ndarray, y_scores: np.ndarray,
                                 fpr_thresholds: Optional[list] = None) -> Dict[float, float]:
    """
    Compute detection rate (TPR) at specific false positive rate thresholds.

    Useful for security: "How many attacks detected while keeping false alarms < 5%?"

    Args:
        y_true: Ground truth labels
        y_scores: Confidence scores
        fpr_thresholds: FPR values to measure at (e.g., [0.05, 0.01, 0.001])

    Returns:
        Dictionary mapping FPR threshold → detection rate
    """
    if fpr_thresholds is None:
        fpr_thresholds = [0.05, 0.01, 0.001]

    if len(np.unique(y_true)) < 2:
        return {fpr: 0.0 for fpr in fpr_thresholds}

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    result = {}
    for fpr_target in fpr_thresholds:
        # Find threshold where FPR is closest to target
        closest_idx = np.argmin(np.abs(fpr - fpr_target))
        result[fpr_target] = float(tpr[closest_idx])

    return result


def compute_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Compute confusion matrix and derived counts.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total': int(tp + tn + fp + fn)
    }


def compute_class_distribution(y_true: np.ndarray) -> Dict[str, Any]:
    """
    Compute class distribution statistics.

    Args:
        y_true: Ground truth labels

    Returns:
        Dictionary with class counts, percentages, and imbalance ratio
    """
    unique, counts = np.unique(y_true, return_counts=True)

    normal_count = counts[0] if 0 in unique else 0
    attack_count = counts[1] if 1 in unique else 0
    total = normal_count + attack_count

    if attack_count > 0:
        imbalance_ratio = normal_count / attack_count
    else:
        imbalance_ratio = float('inf')

    return {
        'normal_count': int(normal_count),
        'attack_count': int(attack_count),
        'total_samples': int(total),
        'normal_percent': float(100 * normal_count / total) if total > 0 else 0.0,
        'attack_percent': float(100 * attack_count / total) if total > 0 else 0.0,
        'imbalance_ratio': float(imbalance_ratio) if attack_count > 0 else 0.0
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       y_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Compute all metrics in one call.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Optional confidence scores for threshold-independent metrics

    Returns:
        Dictionary with all computed metrics organized by category
    """
    metrics = {
        'classification': compute_classification_metrics(y_true, y_pred),
        'security': compute_security_metrics(y_true, y_pred),
        'confusion': compute_confusion_metrics(y_true, y_pred),
        'class_distribution': compute_class_distribution(y_true)
    }

    # Add threshold-independent metrics if scores provided
    if y_scores is not None and len(np.unique(y_true)) >= 2:
        metrics['threshold_independent'] = compute_threshold_independent_metrics(y_true, y_scores)

        # Add detection rates at security thresholds
        metrics['detection_at_fpr'] = compute_detection_rate_at_fpr(
            y_true, y_scores, fpr_thresholds=[0.05, 0.01, 0.001]
        )

    return metrics


def flatten_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested metrics dictionary for CSV export.

    Args:
        metrics_dict: Nested metrics dictionary from compute_all_metrics

    Returns:
        Flattened dictionary with keys like 'classification_accuracy', 'security_fpr', etc.
    """
    flattened = {}

    for category, values in metrics_dict.items():
        if isinstance(values, dict):
            for metric_name, metric_value in values.items():
                key = f"{category}_{metric_name}" if category != 'class_distribution' else metric_name
                flattened[key] = metric_value
        else:
            flattened[category] = values

    return flattened


def format_metrics_for_display(metrics_dict: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as readable string.

    Args:
        metrics_dict: Flat dictionary of metrics
        precision: Decimal places for float values

    Returns:
        Formatted string representation
    """
    lines = []
    for key, value in sorted(metrics_dict.items()):
        if isinstance(value, float):
            lines.append(f"  {key:<30} : {value:.{precision}f}")
        else:
            lines.append(f"  {key:<30} : {value}")
    return '\n'.join(lines)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate sklearn-style classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Formatted classification report string
    """
    return classification_report(y_true, y_pred, target_names=['Normal', 'Attack'])
