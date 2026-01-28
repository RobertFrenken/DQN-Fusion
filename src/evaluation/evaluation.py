"""
Comprehensive Evaluation Framework for CAN-Graph Models

Supports evaluation of three primary pipelines:
1. Teacher: Full-size models without knowledge distillation
2. Student No-KD: Compressed models without knowledge distillation
3. Student With-KD: Compressed models trained with teacher knowledge distillation

Computes comprehensive metrics across train/val/test subsets and exports results
in multiple formats (CSV, JSON) for LaTeX paper integration and ablation studies.
"""

import argparse
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    compute_all_metrics, flatten_metrics, detect_optimal_threshold
)
from src.preprocessing.preprocessing import graph_creation, build_id_mapping_from_normal
from src.models.vgae import GraphAutoencoderNeighborhood
from src.models.models import GATWithJK
from src.models.dqn import EnhancedDQNFusionAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationConfig:
    """Configuration for evaluation pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.dataset = args.dataset
        self.model_path = args.model_path
        self.teacher_path = args.teacher_path
        self.vgae_path = getattr(args, 'vgae_path', None)
        self.gat_path = getattr(args, 'gat_path', None)
        self.training_mode = args.training_mode
        self.kd_mode = args.mode
        self.batch_size = args.batch_size
        self.device = torch.device(args.device if args.device != 'auto' else
                                  ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.csv_output = args.csv_output
        self.json_output = args.json_output
        self.plots_dir = args.plots_dir
        self.threshold_optimization = args.threshold_optimization
        self.verbose = args.verbose

        # Dataset paths
        self.dataset_paths = {
            'hcrl_ch': 'data/automotive/hcrl_ch',
            'hcrl_sa': 'data/automotive/hcrl_sa',
            'set_01': 'data/automotive/set_01',
            'set_02': 'data/automotive/set_02',
            'set_03': 'data/automotive/set_03',
            'set_04': 'data/automotive/set_04',
        }

        self.root_folder = self.dataset_paths.get(self.dataset)
        if not self.root_folder:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        # Validation
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if self.teacher_path and not os.path.exists(self.teacher_path):
            raise FileNotFoundError(f"Teacher model not found: {self.teacher_path}")

        # Fusion mode requires VGAE and GAT models
        if self.training_mode == 'fusion':
            if not self.vgae_path:
                raise ValueError("Fusion mode requires --vgae-path argument")
            if not self.gat_path:
                raise ValueError("Fusion mode requires --gat-path argument")
            if not os.path.exists(self.vgae_path):
                raise FileNotFoundError(f"VGAE model not found: {self.vgae_path}")
            if not os.path.exists(self.gat_path):
                raise FileNotFoundError(f"GAT model not found: {self.gat_path}")

    def log(self):
        """Log configuration."""
        logger.info("=" * 70)
        logger.info("EVALUATION CONFIGURATION")
        logger.info("=" * 70)
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Model Path: {self.model_path}")
        logger.info(f"Teacher Path: {self.teacher_path or 'None'}")
        if self.training_mode == 'fusion':
            logger.info(f"VGAE Path: {self.vgae_path}")
            logger.info(f"GAT Path: {self.gat_path}")
        logger.info(f"Training Mode: {self.training_mode}")
        logger.info(f"KD Mode: {self.kd_mode}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Threshold Optimization: {self.threshold_optimization}")
        logger.info("=" * 70)


class ModelLoader:
    """Load models from disk."""

    def __init__(self, device: torch.device):
        self.device = device

    def load_model(self, model_path: str) -> nn.Module:
        """Load model from file."""
        logger.info(f"Loading model from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            # Try to infer model type and create appropriate model instance
            # For now, just load state dict - actual model creation depends on architecture
            logger.info("Model loaded successfully")
            return state_dict
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load complete checkpoint (for Lightning models)."""
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            logger.info("Checkpoint loaded successfully")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


class DatasetHandler:
    """Handle dataset loading and preprocessing."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.id_mapping = None

    def load_datasets(self) -> Tuple[List, List, List]:
        """
        Load train, validation, and test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Loading datasets...")

        # Build ID mapping from normal samples
        self.id_mapping = build_id_mapping_from_normal(self.config.root_folder)
        logger.info(f"Built ID mapping with {len(self.id_mapping)} IDs")

        # Load training data
        train_dataset = self._load_train_data()

        # Split into train/val
        train_dataset, val_dataset = self._split_train_val(train_dataset, val_ratio=0.2)

        # Load test data
        test_dataset = self._load_test_data()

        logger.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset

    def _load_train_data(self) -> List:
        """Load all training data (combine train_* folders)."""
        combined_dataset = []
        train_root = os.path.join(self.config.root_folder, 'train_01_attack_free')

        # Try standard structure first
        if os.path.exists(train_root):
            dataset = graph_creation(train_root, folder_type='train_',
                                    id_mapping=self.id_mapping, window_size=100)
            combined_dataset.extend(dataset)

        # Try alternative structure with train_*_ folders
        for folder in os.listdir(self.config.root_folder):
            if folder.startswith('train_') and '_' not in folder.split('_', 1)[1] if len(folder.split('_')) > 1 else False:
                folder_path = os.path.join(self.config.root_folder, folder)
                if os.path.isdir(folder_path):
                    dataset = graph_creation(folder_path, folder_type='train_',
                                            id_mapping=self.id_mapping, window_size=100)
                    combined_dataset.extend(dataset)

        if not combined_dataset:
            raise RuntimeError(f"No training data found in {self.config.root_folder}")

        return combined_dataset

    def _load_test_data(self) -> List:
        """Load test data (dataset-specific structure)."""
        if 'hcrl_ch' in self.config.root_folder:
            # For hcrl_ch: glob all test_*.csv files
            combined_dataset = []
            for folder in os.listdir(self.config.root_folder):
                if folder.startswith('test_'):
                    folder_path = os.path.join(self.config.root_folder, folder)
                    if os.path.isdir(folder_path):
                        dataset = graph_creation(folder_path, folder_type='test_',
                                               id_mapping=self.id_mapping, window_size=100)
                        combined_dataset.extend(dataset)
            return combined_dataset
        else:
            # For other datasets: use test_01_known_vehicle_known_attack
            test_folder = os.path.join(self.config.root_folder,
                                      'test_01_known_vehicle_known_attack')
            if not os.path.exists(test_folder):
                raise RuntimeError(f"Test folder not found: {test_folder}")
            return graph_creation(test_folder, folder_type='test_',
                                id_mapping=self.id_mapping, window_size=100)

    def _split_train_val(self, dataset: List, val_ratio: float = 0.2) -> Tuple[List, List]:
        """Split dataset into train/val."""
        normal_indices = [i for i, data in enumerate(dataset) if data.y.item() == 0]
        attack_indices = [i for i, data in enumerate(dataset) if data.y.item() == 1]

        n_val_normal = int(len(normal_indices) * val_ratio)
        n_val_attack = int(len(attack_indices) * val_ratio)

        np.random.seed(42)
        val_normal_idx = np.random.choice(normal_indices, n_val_normal, replace=False)
        val_attack_idx = np.random.choice(attack_indices, n_val_attack, replace=False)

        val_indices = set(list(val_normal_idx) + list(val_attack_idx))

        val_dataset = [dataset[i] for i in val_indices]
        train_dataset = [dataset[i] for i in range(len(dataset)) if i not in val_indices]

        logger.info(f"Train/Val split: {len(train_dataset)} train, {len(val_dataset)} val")
        return train_dataset, val_dataset


class Evaluator:
    """Main evaluation pipeline."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model_loader = ModelLoader(config.device)
        self.dataset_handler = DatasetHandler(config)
        self.model = None
        self.model_type = None

    def evaluate(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        logger.info("Starting evaluation pipeline...")
        start_time = time.time()

        # Load datasets
        train_dataset, val_dataset, test_dataset = self.dataset_handler.load_datasets()

        # Load and instantiate model
        self._load_model(train_dataset)

        # Perform inference on all subsets
        logger.info("Running inference on train subset...")
        train_predictions, train_scores = self._infer_subset(train_dataset)
        train_labels = np.array([data.y.item() for data in train_dataset])

        logger.info("Running inference on val subset...")
        val_predictions, val_scores = self._infer_subset(val_dataset)
        val_labels = np.array([data.y.item() for data in val_dataset])

        logger.info("Running inference on test subset...")
        test_predictions, test_scores = self._infer_subset(test_dataset)
        test_labels = np.array([data.y.item() for data in test_dataset])

        # Compute metrics for each subset
        logger.info("Computing metrics...")
        results = {
            'train': self._compute_subset_metrics('train', train_labels, train_predictions, train_scores),
            'val': self._compute_subset_metrics('val', val_labels, val_predictions, val_scores),
            'test': self._compute_subset_metrics('test', test_labels, test_predictions, test_scores)
        }

        # Threshold optimization on val set
        if self.config.threshold_optimization and val_scores is not None:
            logger.info("Optimizing threshold on validation set...")
            optimal_threshold, opt_metrics = detect_optimal_threshold(
                val_labels, val_scores, metric='f1'
            )
            results['threshold_optimization'] = {
                'optimal_threshold': float(optimal_threshold),
                'optimized_metrics': opt_metrics
            }
            logger.info(f"Optimal threshold: {optimal_threshold:.6f}")

        elapsed = time.time() - start_time
        results['metadata'] = {
            'dataset': self.config.dataset,
            'model_path': self.config.model_path,
            'teacher_path': self.config.teacher_path,
            'training_mode': self.config.training_mode,
            'kd_mode': self.config.kd_mode,
            'device': str(self.config.device),
            'evaluation_time_seconds': elapsed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        logger.info(f"Evaluation completed in {elapsed:.2f} seconds")
        return results

    def _load_model(self, sample_dataset: List) -> None:
        """
        Load and instantiate model based on training mode.

        For fusion mode, also loads VGAE and GAT models.

        Args:
            sample_dataset: Sample dataset to infer num_ids from ID mapping
        """
        logger.info(f"Loading model from {self.config.model_path}...")

        # Load checkpoint
        checkpoint = self.model_loader.load_model(self.config.model_path)

        # Determine model type from training mode
        self.model_type = self.config.training_mode
        logger.info(f"Model type: {self.model_type}")

        # Get num_ids from dataset handler
        num_ids = len(self.dataset_handler.id_mapping) if self.dataset_handler.id_mapping else 1000
        logger.info(f"Number of IDs: {num_ids}")

        # Instantiate model based on training mode
        if self.model_type == 'autoencoder':
            self.model = self._build_vgae_model(num_ids)
        elif self.model_type in ['normal', 'curriculum']:
            self.model = self._build_gat_model(num_ids)
        elif self.model_type == 'fusion':
            # For fusion, load DQN agent + VGAE + GAT models
            self.model = self._build_dqn_model()

            # Load VGAE model
            logger.info(f"Loading VGAE model from {self.config.vgae_path}...")
            vgae_checkpoint = self.model_loader.load_model(self.config.vgae_path)
            self.vgae_model = self._build_vgae_model(num_ids)
            self._load_state_dict(self.vgae_model, vgae_checkpoint)
            self.vgae_model.to(self.config.device)
            self.vgae_model.eval()
            logger.info("✅ VGAE model loaded")

            # Load GAT model
            logger.info(f"Loading GAT model from {self.config.gat_path}...")
            gat_checkpoint = self.model_loader.load_model(self.config.gat_path)
            self.gat_model = self._build_gat_model(num_ids)
            self._load_state_dict(self.gat_model, gat_checkpoint)
            self.gat_model.to(self.config.device)
            self.gat_model.eval()
            logger.info("✅ GAT model loaded")
        else:
            raise ValueError(f"Unknown training mode: {self.model_type}")

        # Load state dict for main model
        self._load_state_dict(self.model, checkpoint)

        self.model.to(self.config.device)
        self.model.eval()
        logger.info("✅ Model loaded and ready for inference")

    def _load_state_dict(self, model: nn.Module, checkpoint: Any) -> None:
        """Load state dict into model, handling different checkpoint formats."""
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # Lightning checkpoint format
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            cleaned_state = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned_state, strict=False)
        elif isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
            # DQN agent checkpoint format
            # For EnhancedDQNFusionAgent, load Q-network weights
            if hasattr(model, 'q_network'):
                model.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            # Raw state dict
            model.load_state_dict(checkpoint, strict=False)

    def _build_vgae_model(self, num_ids: int) -> nn.Module:
        """Build VGAE model for inference."""
        model = GraphAutoencoderNeighborhood(
            num_ids=num_ids,
            in_channels=8,  # Standard input dimension
            hidden_dims=[256, 128, 96, 48],
            latent_dim=48,
            encoder_heads=4,
            decoder_heads=4,
            embedding_dim=32,
            dropout=0.15,
            batch_norm=True
        )
        return model

    def _build_gat_model(self, num_ids: int) -> nn.Module:
        """Build GAT model for inference."""
        model = GATWithJK(
            in_channels=8,
            hidden_channels=128,
            out_channels=2,  # Binary classification
            num_layers=3,
            heads=4,
            dropout=0.15,
            num_fc_layers=2,
            embedding_dim=32,
            num_ids=num_ids
        )
        return model

    def _build_dqn_model(self) -> nn.Module:
        """Build DQN fusion model for inference."""
        # DQN fusion agent expects 2 inputs (VGAE anomaly score + GAT probability)
        model = EnhancedDQNFusionAgent(input_dim=2, hidden_dim=64)
        return model

    def _infer_subset(self, dataset: List) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run real inference on dataset subset using actual model predictions.

        Returns:
            Tuple of (predictions, scores) where:
            - predictions: binary labels (0 or 1)
            - scores: confidence scores in [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        logger.info(f"Running {self.model_type} inference on {len(dataset)} samples...")

        predictions = []
        scores = []

        # Create dataloader for batched inference
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = batch.to(self.config.device)

                if self.model_type == 'autoencoder':
                    # VGAE inference: compute reconstruction error as anomaly score
                    batch_preds, batch_scores = self._infer_vgae_batch(batch)

                elif self.model_type in ['normal', 'curriculum']:
                    # GAT inference: logits → softmax → predictions
                    batch_preds, batch_scores = self._infer_gat_batch(batch)

                elif self.model_type == 'fusion':
                    # DQN fusion: use combined scores
                    batch_preds, batch_scores = self._infer_fusion_batch(batch)

                predictions.extend(batch_preds)
                scores.extend(batch_scores)

                if (batch_idx + 1) % max(1, len(dataloader) // 10) == 0:
                    logger.info(f"  Processed {min((batch_idx + 1) * self.config.batch_size, len(dataset))}/{len(dataset)} samples")

        return np.array(predictions, dtype=np.int32), np.array(scores, dtype=np.float32)

    def _infer_vgae_batch(self, batch) -> Tuple[List[int], List[float]]:
        """VGAE inference: anomaly detection via reconstruction error."""
        cont_out, canid_logits, neighbor_logits, z, kl_loss = self.model(batch)

        # Reconstruction error as anomaly score
        continuous_features = batch.x[:, 1:]
        reconstruction_error = F.mse_loss(cont_out, continuous_features, reduction='none').mean(dim=1)

        # Normalize reconstruction errors to [0, 1] range
        # Use simple percentile-based normalization
        error_scores = reconstruction_error.cpu().numpy()
        # Threshold at 50th percentile (median) for classification
        threshold = np.median(error_scores) if len(error_scores) > 0 else 0.5

        # Higher error = more anomalous = prediction of 1 (attack)
        predictions = (error_scores > threshold).astype(int).tolist()

        # Normalize errors to [0, 1] as confidence score
        # Use sigmoid-like transformation: score = 1 / (1 + exp(-10 * (error - threshold)))
        scores = (1 / (1 + np.exp(-10 * (error_scores - threshold)))).tolist()

        return predictions, scores

    def _infer_gat_batch(self, batch) -> Tuple[List[int], List[float]]:
        """GAT inference: classification via softmax probabilities."""
        logits = self.model(batch)

        # Get predictions and confidence scores
        probs = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1).cpu().numpy().tolist()

        # Confidence score: max probability from softmax
        scores = probs.max(dim=1)[0].cpu().numpy().tolist()

        return predictions, scores

    def _infer_fusion_batch(self, batch) -> Tuple[List[int], List[float]]:
        """
        DQN fusion inference with 15D state space.

        Extracts rich features from both VGAE and GAT, then uses DQN to select fusion weight.
        """
        batch_size = batch.num_graphs

        # ========== VGAE Feature Extraction (8 dims) ==========
        cont_out, canid_logits, neighbor_logits, z, _ = self.vgae_model(batch)

        # Error components (3 dims)
        continuous_features = batch.x[:, 1:]
        node_errors = F.mse_loss(cont_out, continuous_features, reduction='none').mean(dim=1)
        canid_errors = F.cross_entropy(canid_logits, batch.x[:, 0].long(), reduction='none')
        neighbor_errors = F.binary_cross_entropy_with_logits(
            neighbor_logits, batch.edge_attr.float(), reduction='none'
        ).mean(dim=1) if batch.edge_attr.numel() > 0 else torch.zeros_like(node_errors)

        # Aggregate errors per graph
        vgae_errors = []
        vgae_latent = []
        vgae_confidence = []

        for graph_idx in range(batch_size):
            node_mask = (batch.batch == graph_idx)
            graph_node_errors = node_errors[node_mask].mean().item()
            graph_canid_errors = canid_errors[node_mask].mean().item()
            graph_neighbor_errors = neighbor_errors[node_mask].mean().item() if neighbor_errors.numel() > 0 else 0.0

            vgae_errors.append([graph_node_errors, graph_canid_errors, graph_neighbor_errors])

            # Latent statistics (4 dims: mean, std, max, min)
            graph_latent = z[node_mask]
            vgae_latent.append([
                graph_latent.mean().item(),
                graph_latent.std().item(),
                graph_latent.max().item(),
                graph_latent.min().item()
            ])

            # VGAE confidence: inverse of error variance
            error_vec = torch.tensor([graph_node_errors, graph_canid_errors, graph_neighbor_errors])
            vgae_confidence.append(1.0 / (1.0 + error_vec.var().item()))

        vgae_errors = torch.tensor(vgae_errors, dtype=torch.float32)
        vgae_latent = torch.tensor(vgae_latent, dtype=torch.float32)
        vgae_confidence = torch.tensor(vgae_confidence, dtype=torch.float32)

        # ========== GAT Feature Extraction (7 dims) ==========
        # Get intermediate representations (pre-pooling embeddings)
        xs = self.gat_model(batch, return_intermediate=True)
        pre_pooling_embeddings = xs[-1]  # Last layer's output before pooling [num_nodes, hidden_dim * heads]

        # Also get final logits for classification
        gat_logits = self.gat_model(batch, return_intermediate=False)  # [num_nodes, 2]

        # Aggregate features per graph
        gat_logits_per_graph = []
        gat_embeddings = []
        gat_confidence = []

        for graph_idx in range(batch_size):
            node_mask = (batch.batch == graph_idx)
            graph_logits = gat_logits[node_mask]
            graph_embeddings = pre_pooling_embeddings[node_mask]

            # Aggregate logits per graph (mean across nodes)
            graph_logits_mean = graph_logits.mean(dim=0)
            gat_logits_per_graph.append(graph_logits_mean.cpu().numpy())

            # Embedding statistics (4 dims: mean, std, max, min of pre-pooling embeddings)
            gat_embeddings.append([
                graph_embeddings.mean().item(),
                graph_embeddings.std().item(),
                graph_embeddings.max().item(),
                graph_embeddings.min().item()
            ])

            # GAT confidence: 1 - normalized entropy
            probs = F.softmax(graph_logits_mean, dim=0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            normalized_entropy = entropy / np.log(2)  # Normalize by log(num_classes)
            gat_confidence.append((1.0 - normalized_entropy).item())

        gat_logits_tensor = torch.tensor(gat_logits_per_graph, dtype=torch.float32)
        gat_embeddings = torch.tensor(gat_embeddings, dtype=torch.float32)
        gat_confidence = torch.tensor(gat_confidence, dtype=torch.float32)

        # ========== Stack 15D State and Use DQN ==========
        predictions = []
        scores = []

        for i in range(batch_size):
            # Stack 15D state vector
            state_15d = np.concatenate([
                vgae_errors[i].numpy(),      # 3 dims
                vgae_latent[i].numpy(),      # 4 dims
                [vgae_confidence[i].item()], # 1 dim
                gat_logits_tensor[i].numpy(),# 2 dims
                gat_embeddings[i].numpy(),   # 4 dims
                [gat_confidence[i].item()]   # 1 dim
            ])  # Total: 15 dims

            # Use DQN to select fusion weight
            alpha, _, _ = self.model.select_action(state_15d, training=False)

            # Derive scalar scores for fusion
            vgae_weights = np.array([0.4, 0.35, 0.25])
            anomaly_score = np.clip(np.sum(vgae_errors[i].numpy() * vgae_weights), 0.0, 1.0)

            gat_probs = np.exp(gat_logits_tensor[i].numpy()) / np.sum(np.exp(gat_logits_tensor[i].numpy()))
            gat_prob = gat_probs[1]

            # Fused prediction
            fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
            prediction = 1 if fused_score > 0.5 else 0

            predictions.append(prediction)
            scores.append(fused_score)

        return predictions, scores

    def _compute_subset_metrics(self, subset_name: str, y_true: np.ndarray,
                               y_pred: np.ndarray, y_scores: Optional[np.ndarray]) -> Dict[str, Any]:
        """Compute all metrics for a subset."""
        metrics = compute_all_metrics(y_true, y_pred, y_scores)
        metrics['subset_name'] = subset_name
        return metrics

    def export_results(self, results: Dict[str, Any]) -> None:
        """Export results to CSV and JSON formats."""
        logger.info("Exporting results...")

        # Prepare wide format CSV
        if self.config.csv_output:
            self._export_wide_csv(results, self.config.csv_output)

        # Prepare JSON summary
        if self.config.json_output:
            self._export_json(results, self.config.json_output)

        logger.info("Results exported successfully")

    def _export_wide_csv(self, results: Dict[str, Any], output_path: str) -> None:
        """Export results in wide CSV format (one row per subset)."""
        rows = []

        for subset_name in ['train', 'val', 'test']:
            if subset_name not in results:
                continue

            subset_results = results[subset_name]
            row = {
                'dataset': self.config.dataset,
                'subset': subset_name,
                'model': Path(self.config.model_path).stem,
                'training_mode': self.config.training_mode,
                'kd_mode': self.config.kd_mode
            }

            # Flatten metrics
            flattened = flatten_metrics(subset_results)
            row.update(flattened)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Wide CSV exported to {output_path}")

    def _export_json(self, results: Dict[str, Any], output_path: str) -> None:
        """Export results as JSON summary."""
        # Make results JSON-serializable
        json_results = {
            'metadata': results.get('metadata', {}),
            'results': {}
        }

        for subset_name in ['train', 'val', 'test']:
            if subset_name not in results:
                continue
            json_results['results'][subset_name] = self._serialize_metrics(results[subset_name])

        if 'threshold_optimization' in results:
            json_results['threshold_optimization'] = self._serialize_metrics(results['threshold_optimization'])

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"JSON exported to {output_path}")

    def _serialize_metrics(self, metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metrics to JSON-serializable format."""
        serialized = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_metrics(value)
            else:
                serialized[key] = value
        return serialized

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print evaluation results to console."""
        logger.info("=" * 90)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 90)

        for subset_name in ['train', 'val', 'test']:
            if subset_name not in results:
                continue

            logger.info(f"\n{subset_name.upper()} SET METRICS")
            logger.info("-" * 90)

            subset_results = results[subset_name]

            # Print class distribution
            if 'class_distribution' in subset_results:
                dist = subset_results['class_distribution']
                logger.info(f"Samples: {dist['total_samples']} (Normal: {dist['normal_count']}, "
                          f"Attack: {dist['attack_count']}, Ratio: {dist['imbalance_ratio']:.1f}:1)")

            # Print key metrics
            if 'classification' in subset_results:
                logger.info("\nClassification Metrics:")
                for key, value in subset_results['classification'].items():
                    logger.info(f"  {key:<25} : {value:.4f}")

            if 'security' in subset_results:
                logger.info("\nSecurity Metrics:")
                for key, value in subset_results['security'].items():
                    if isinstance(value, float):
                        logger.info(f"  {key:<25} : {value:.4f}")
                    else:
                        logger.info(f"  {key:<25} : {value}")

            if 'threshold_independent' in subset_results:
                logger.info("\nThreshold-Independent Metrics:")
                for key, value in subset_results['threshold_independent'].items():
                    logger.info(f"  {key:<25} : {value:.4f}")

        logger.info("\n" + "=" * 90)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation framework for CAN-Graph models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Evaluate student model without KD
  python -m src.evaluation.evaluation \\
    --dataset hcrl_sa \\
    --model-path saved_models/gat_student.pth \\
    --training-mode normal \\
    --mode standard

  # Evaluate student model with KD
  python -m src.evaluation.evaluation \\
    --dataset hcrl_sa \\
    --model-path saved_models/gat_student_with_kd.pth \\
    --teacher-path saved_models/gat_teacher.pth \\
    --training-mode normal \\
    --mode with-kd

  # Evaluate with CSV and JSON output
  python -m src.evaluation.evaluation \\
    --dataset hcrl_sa \\
    --model-path saved_models/gat_student.pth \\
    --training-mode autoencoder \\
    --mode standard \\
    --csv-output results.csv \\
    --json-output results.json \\
    --threshold-optimization true
        '''
    )

    # Required arguments
    parser.add_argument('--dataset', required=True,
                       choices=['hcrl_sa', 'hcrl_ch', 'set_01', 'set_02', 'set_03', 'set_04'],
                       help='Dataset to evaluate on')
    parser.add_argument('--model-path', required=True,
                       help='Path to primary model (student or teacher)')
    parser.add_argument('--training-mode', required=True,
                       choices=['normal', 'autoencoder', 'curriculum', 'fusion'],
                       help='Training mode the model was trained with')
    parser.add_argument('--mode', required=True,
                       choices=['standard', 'with-kd'],
                       help='Knowledge distillation mode')

    # Optional arguments
    parser.add_argument('--teacher-path', default=None,
                       help='Path to teacher model (for KD or baseline comparison)')
    parser.add_argument('--vgae-path', default=None,
                       help='Path to VGAE model (required for fusion mode evaluation)')
    parser.add_argument('--gat-path', default=None,
                       help='Path to GAT model (required for fusion mode evaluation)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for inference')
    parser.add_argument('--device', default='auto',
                       choices=['cuda', 'cpu', 'auto'],
                       help='Device to use for inference')
    parser.add_argument('--csv-output', default='evaluation_results.csv',
                       help='Path to output CSV file')
    parser.add_argument('--json-output', default='evaluation_results.json',
                       help='Path to output JSON file')
    parser.add_argument('--plots-dir', default='evaluation_plots',
                       help='Directory to save metric plots')
    parser.add_argument('--threshold-optimization', type=bool, default=True,
                       help='Whether to optimize detection threshold')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')

    args = parser.parse_args()

    # Create config
    try:
        config = EvaluationConfig(args)
        config.log()
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return 1

    # Run evaluation
    try:
        evaluator = Evaluator(config)
        results = evaluator.evaluate()
        evaluator.print_results(results)
        evaluator.export_results(results)
        logger.info("Evaluation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
