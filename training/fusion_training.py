"""
Fusion Training Pipeline for CAN Bus Intrusion Detection

This module implements reinforcement learning-based fusion training for combining
VGAE anomaly detection and GAT classification outputs using a DQN agent.

Key Features:
- Load pre-trained VGAE and GAT models
- Extract anomaly scores and GAT probabilities from validation data
- Train DQN fusion agent with experience replay
- Evaluate and compare fusion strategies
- Save trained fusion agent for deployment
"""
import sys
import os
from pathlib import Path
# Clean path setup - add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.data import random_split, Subset
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any
import warnings
import random
from tqdm import tqdm

# Import your existing modules
from models.models import GATWithJK, GraphAutoencoderNeighborhood
from models.adaptive_fusion import EnhancedDQNFusionAgent
from archive.preprocessing import graph_creation, build_id_mapping_from_normal
from utils.utils_logging import setup_gpu_optimization, log_memory_usage, cleanup_memory

warnings.filterwarnings('ignore', category=UserWarning)




# Configuration Constants
DATASET_PATHS = {
    'hcrl_ch': r"datasets/can-train-and-test-v1.5/hcrl-ch",
    'hcrl_sa': r"datasets/can-train-and-test-v1.5/hcrl-sa",
    'set_01': r"datasets/can-train-and-test-v1.5/set_01",
    'set_02': r"datasets/can-train-and-test-v1.5/set_02",
    'set_03': r"datasets/can-train-and-test-v1.5/set_03",
    'set_04': r"datasets/can-train-and-test-v1.5/set_04",
}

# Fusion weights for composite anomaly scoring
FUSION_WEIGHTS = {
    'node_reconstruction': 0.4,
    'neighborhood_prediction': 0.35,
    'can_id_prediction': 0.25
}

class FusionDataExtractor:
    """Extract anomaly scores and GAT probabilities for fusion training."""
    
    def __init__(self, autoencoder: nn.Module, classifier: nn.Module, 
                 device: str, threshold: float = 0.0):
        self.autoencoder = autoencoder.to(device)
        self.classifier = classifier.to(device)
        self.device = torch.device(device)
        self.threshold = threshold
        
        # Set models to evaluation mode
        self.autoencoder.eval()
        self.classifier.eval()
        
        print(f"✓ Fusion Data Extractor initialized with threshold: {threshold:.4f}")

    def compute_anomaly_scores(self, batch) -> torch.Tensor:
        """
        Compute normalized anomaly scores for each graph in the batch.
        
        Args:
            batch: Batch of graph data
            
        Returns:
            Tensor of anomaly scores [0, 1] for each graph
        """
        with torch.no_grad():
            # Forward pass through autoencoder
            cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                batch.x, batch.edge_index, batch.batch
            )
            
            # Compute component errors
            node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
            
            # Neighborhood reconstruction errors
            neighbor_targets = self.autoencoder.create_neighborhood_targets(
                batch.x, batch.edge_index, batch.batch
            )
            neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                neighbor_logits, neighbor_targets
            ).mean(dim=1)
            
            # CAN ID prediction errors
            canid_pred = canid_logits.argmax(dim=1)
            true_canids = batch.x[:, 0].long()
            canid_errors = (canid_pred != true_canids).float()
            
            # Aggregate to graph-level scores
            graphs = Batch.to_data_list(batch)
            anomaly_scores = []
            start_idx = 0
            
            for graph in graphs:
                num_nodes = graph.x.size(0)
                end_idx = start_idx + num_nodes
                
                # Graph-level error aggregation (max pooling for anomalies)
                graph_node_error = node_errors[start_idx:end_idx].max().item()
                graph_neighbor_error = neighbor_errors[start_idx:end_idx].max().item()
                graph_canid_error = canid_errors[start_idx:end_idx].max().item()
                
                # Weighted composite score
                composite_score = (
                    FUSION_WEIGHTS['node_reconstruction'] * graph_node_error +
                    FUSION_WEIGHTS['neighborhood_prediction'] * graph_neighbor_error +
                    FUSION_WEIGHTS['can_id_prediction'] * graph_canid_error
                )
                
                # Normalize to [0, 1] using sigmoid with learned scaling
                normalized_score = torch.sigmoid(torch.tensor(composite_score * 3 - 1.5)).item()
                anomaly_scores.append(normalized_score)
                
                start_idx = end_idx
            
            return torch.tensor(anomaly_scores, dtype=torch.float32)

    def compute_gat_probabilities(self, batch) -> torch.Tensor:
        """
        Compute GAT classification probabilities for each graph.
        
        Args:
            batch: Batch of graph data
            
        Returns:
            Tensor of GAT probabilities [0, 1] for each graph
        """
        with torch.no_grad():
            logits = self.classifier(batch)
            probabilities = torch.sigmoid(logits.squeeze())
            return probabilities.cpu()

    def extract_fusion_data(self, data_loader: DataLoader, max_samples: int = None) -> Tuple[List, List, List]:
        """
        Extract anomaly scores, GAT probabilities, and labels for fusion training.
        
        Args:
            data_loader: DataLoader with graph data
            max_samples: Maximum number of samples to extract (None for all)
            
        Returns:
            Tuple of (anomaly_scores, gat_probabilities, labels)
        """
        print("Extracting fusion training data...")
        
        anomaly_scores = []
        gat_probabilities = []
        labels = []
        
        samples_processed = 0
        
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting data")):
            batch = batch.to(self.device)
            
            # Extract scores and probabilities
            batch_anomaly_scores = self.compute_anomaly_scores(batch)
            batch_gat_probs = self.compute_gat_probabilities(batch)
            batch_labels = batch.y.cpu()
            
            # Store results
            anomaly_scores.extend(batch_anomaly_scores.tolist())
            gat_probabilities.extend(batch_gat_probs.tolist())
            labels.extend(batch_labels.tolist())
            
            samples_processed += len(batch_labels)
            
            # Check if we've reached the sample limit
            if max_samples is not None and samples_processed >= max_samples:
                break
            
            # Memory cleanup
            del batch, batch_anomaly_scores, batch_gat_probs, batch_labels
            if batch_idx % 50 == 0:
                cleanup_memory()
        
        print(f"✓ Extracted {len(anomaly_scores)} samples for fusion training")
        print(f"  Normal samples: {sum(1 for l in labels if l == 0)}")
        print(f"  Attack samples: {sum(1 for l in labels if l == 1)}")
        
        return anomaly_scores, gat_probabilities, labels

class FusionTrainingPipeline:
    """Complete pipeline for training the fusion agent."""
    
    def __init__(self, num_ids: int, embedding_dim: int = 8, device: str = 'cpu'):
        self.device = torch.device(device)
        self.num_ids = num_ids
        self.embedding_dim = embedding_dim
        
        # Models (will be loaded)
        self.autoencoder = None
        self.classifier = None
        self.fusion_agent = None
        self.data_extractor = None
        
        # Training data
        self.training_data = None
        self.validation_data = None
        
        print(f"✓ Fusion Training Pipeline initialized on {device}")

    def load_pretrained_models(self, autoencoder_path: str, classifier_path: str):
        """
        Load pre-trained VGAE and GAT models.
        
        Args:
            autoencoder_path: Path to autoencoder checkpoint
            classifier_path: Path to classifier checkpoint
        """
        print(f"\n=== Loading Pre-trained Models ===")
        print(f"Autoencoder: {autoencoder_path}")
        print(f"Classifier: {classifier_path}")
        
        # Initialize models
        self.autoencoder = GraphAutoencoderNeighborhood(
            num_ids=self.num_ids,
            in_channels=11,
            embedding_dim=self.embedding_dim,
            hidden_dim=32,
            latent_dim=32,
            num_encoder_layers=3,
            num_decoder_layers=3,
            encoder_heads=4,
            decoder_heads=4
        ).to(self.device)
        
        self.classifier = GATWithJK(
            num_ids=self.num_ids,
            in_channels=11,
            hidden_channels=32,
            out_channels=1,
            num_layers=5,
            heads=8,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        # Load checkpoints
        try:
            # Load autoencoder
            ae_checkpoint = torch.load(autoencoder_path, map_location=self.device)
            if isinstance(ae_checkpoint, dict):
                ae_state_dict = ae_checkpoint.get('state_dict', ae_checkpoint)
                threshold = ae_checkpoint.get('threshold', 0.0)
            else:
                ae_state_dict = ae_checkpoint
                threshold = 0.0
            
            self.autoencoder.load_state_dict(ae_state_dict)
            
            # Load classifier
            clf_checkpoint = torch.load(classifier_path, map_location=self.device)
            if isinstance(clf_checkpoint, dict):
                clf_state_dict = clf_checkpoint.get('state_dict', clf_checkpoint)
            else:
                clf_state_dict = clf_checkpoint
                
            self.classifier.load_state_dict(clf_state_dict)
            
            print("✓ Models loaded successfully")
            
            # Create data extractor
            self.data_extractor = FusionDataExtractor(
                self.autoencoder, self.classifier, str(self.device), threshold
            )
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e

    def prepare_fusion_data(self, train_loader: DataLoader, val_loader: DataLoader, 
                          max_train_samples: int = 50000, max_val_samples: int = 10000):
        """
        Prepare training and validation data for fusion learning.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_train_samples: Maximum training samples to use
            max_val_samples: Maximum validation samples to use
        """
        print(f"\n=== Preparing Fusion Data ===")
        
        # Extract training data
        print("Extracting training data...")
        train_anomaly_scores, train_gat_probs, train_labels = self.data_extractor.extract_fusion_data(
            train_loader, max_train_samples
        )
        
        # Extract validation data
        print("Extracting validation data...")
        val_anomaly_scores, val_gat_probs, val_labels = self.data_extractor.extract_fusion_data(
            val_loader, max_val_samples
        )
        
        # Store as tuples for easy access
        self.training_data = list(zip(train_anomaly_scores, train_gat_probs, train_labels))
        self.validation_data = list(zip(val_anomaly_scores, val_gat_probs, val_labels))
        
        print(f"✓ Fusion data prepared:")
        print(f"  Training samples: {len(self.training_data)}")
        print(f"  Validation samples: {len(self.validation_data)}")
        
        # Data quality checks
        self._analyze_fusion_data()

    def _analyze_fusion_data(self):
        """Analyze the extracted fusion data for quality and distribution."""
        if not self.training_data:
            return
        
        train_anomaly_scores = [x[0] for x in self.training_data]
        train_gat_probs = [x[1] for x in self.training_data]
        train_labels = [x[2] for x in self.training_data]
        
        print(f"\n📊 Fusion Data Analysis:")
        print(f"Anomaly Score Range: [{min(train_anomaly_scores):.3f}, {max(train_anomaly_scores):.3f}]")
        print(f"GAT Probability Range: [{min(train_gat_probs):.3f}, {max(train_gat_probs):.3f}]")
        print(f"Class Distribution: Normal={train_labels.count(0)}, Attack={train_labels.count(1)}")
        
        # Correlation analysis
        normal_data = [(a, g) for a, g, l in self.training_data if l == 0]
        attack_data = [(a, g) for a, g, l in self.training_data if l == 1]
        
        if normal_data:
            normal_anomaly_avg = np.mean([x[0] for x in normal_data])
            normal_gat_avg = np.mean([x[1] for x in normal_data])
            print(f"Normal Class Averages: Anomaly={normal_anomaly_avg:.3f}, GAT={normal_gat_avg:.3f}")
        
        if attack_data:
            attack_anomaly_avg = np.mean([x[0] for x in attack_data])
            attack_gat_avg = np.mean([x[1] for x in attack_data])
            print(f"Attack Class Averages: Anomaly={attack_anomaly_avg:.3f}, GAT={attack_gat_avg:.3f}")

    def initialize_fusion_agent(self, alpha_steps: int = 21, lr: float = 1e-3, 
                               epsilon: float = 0.3, buffer_size: int = 100000,
                               batch_size: int = 128, target_update_freq: int = 100):
        """
        Initialize the DQN fusion agent.
        
        Args:
            alpha_steps: Number of discrete fusion weights
            lr: Learning rate
            epsilon: Initial exploration rate
            buffer_size: Experience replay buffer size
        """
        print(f"\n=== Initializing Fusion Agent ===")
        
        self.fusion_agent = EnhancedDQNFusionAgent(
            alpha_steps=alpha_steps,
            lr=lr,
            gamma=0.95,  # Slightly higher discount for fusion learning
            epsilon=epsilon,
            epsilon_decay=0.995,
            min_epsilon=0.1,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=str(self.device)
        )
        
        print(f"✓ Fusion Agent initialized:")
        print(f"  Action space: {alpha_steps} fusion weights")
        print(f"  State space: 4D (anomaly_score, gat_prob, confidence_diff, avg_confidence)")
        print(f"  Buffer size: {buffer_size:,}")

    def train_fusion_agent(self, episodes: int = 50, validation_interval: int = 10,  # Shorter episodes for testing
                          early_stopping_patience: int = 20, save_interval: int = 25):
        """
        Enhanced training with better instrumentation and learning dynamics.
        """
        print(f"\n=== Training Fusion Agent ===")
        print(f"Episodes: {episodes}, Validation every {validation_interval} episodes")
        
        if not self.training_data or not self.fusion_agent:
            raise ValueError("Training data and fusion agent must be initialized first")
        
        # Enhanced tracking
        episode_rewards = []
        episode_accuracies = []
        episode_losses = []
        episode_q_values = []
        action_distributions = []
        reward_stats = []
        validation_scores = []
        best_validation_score = -float('inf')
        patience_counter = 0
        
        # Create save directory
        os.makedirs("saved_models/fusion_checkpoints", exist_ok=True)
        
        for episode in range(episodes):
            episode_reward = 0
            episode_correct = 0
            episode_samples = 0
            episode_loss_sum = 0
            episode_loss_count = 0
            episode_q_sum = 0
            episode_action_counts = np.zeros(len(self.fusion_agent.alpha_values))
            episode_raw_rewards = []
            
            # Shuffle training data for each episode
            shuffled_data = random.sample(self.training_data, min(len(self.training_data), 5000))  # Limit episode size
            
            # Process samples with sequential next_states
            for i, (anomaly_score, gat_prob, true_label) in enumerate(shuffled_data):
                # Current state
                current_state = self.fusion_agent.normalize_state(anomaly_score, gat_prob)
                
                # Select action (fusion weight)
                alpha, action_idx, _ = self.fusion_agent.select_action(
                    anomaly_score, gat_prob, training=True
                )
                
                # Track action distribution
                episode_action_counts[action_idx] += 1
                
                # Make fused prediction
                fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
                prediction = 1 if fused_score > 0.5 else 0
                
                # Compute raw reward
                raw_reward = self.fusion_agent.compute_fusion_reward(
                    prediction, true_label, anomaly_score, gat_prob, alpha
                )
                
                # Clip and normalize reward for better learning
                clipped_reward = np.clip(raw_reward, -5.0, 5.0)  # Clip extreme rewards
                normalized_reward = clipped_reward / 5.0  # Normalize to [-1, 1]
                
                episode_raw_rewards.append(raw_reward)
                
                # Get next state (use next sample if available, else current)
                if i + 1 < len(shuffled_data):
                    next_anomaly, next_gat, _ = shuffled_data[i + 1]
                    next_state = self.fusion_agent.normalize_state(next_anomaly, next_gat)
                    done = False  # Not terminal, agent can bootstrap
                else:
                    next_state = current_state  # Terminal state
                    done = True
                
                # Store experience with improved dynamics
                self.fusion_agent.store_experience(
                    current_state, action_idx, normalized_reward, next_state, done
                )
                
                # Train more frequently with smaller batches
                if len(self.fusion_agent.replay_buffer) >= self.fusion_agent.batch_size:
                    loss = self.fusion_agent.train_step()
                    if loss is not None:
                        episode_loss_sum += loss
                        episode_loss_count += 1
                
                # Track Q-values for sample states
                with torch.no_grad():
                    state_tensor = torch.tensor(current_state).unsqueeze(0).to(self.fusion_agent.device)
                    q_vals = self.fusion_agent.q_network(state_tensor)
                    episode_q_sum += q_vals.mean().item()
                
                # Track statistics
                episode_reward += normalized_reward
                episode_correct += (prediction == true_label)
                episode_samples += 1
                
                # More frequent memory cleanup
                if i % 100 == 0:
                    cleanup_memory()
            
            # End episode
            self.fusion_agent.end_episode()
            
            # Calculate episode statistics
            episode_accuracy = episode_correct / episode_samples if episode_samples > 0 else 0
            avg_episode_reward = episode_reward / episode_samples if episode_samples > 0 else 0
            avg_episode_loss = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0
            avg_q_value = episode_q_sum / episode_samples if episode_samples > 0 else 0
            
            # Store episode stats
            episode_rewards.append(avg_episode_reward)
            episode_accuracies.append(episode_accuracy)
            episode_losses.append(avg_episode_loss)
            episode_q_values.append(avg_q_value)
            action_distributions.append(episode_action_counts / episode_samples)
            
            # Reward statistics
            reward_stats.append({
                'raw_mean': np.mean(episode_raw_rewards),
                'raw_std': np.std(episode_raw_rewards),
                'raw_min': np.min(episode_raw_rewards),
                'raw_max': np.max(episode_raw_rewards)
            })
            
            # Decay exploration rate more gradually
            if episode % 2 == 0:  # Every 2 episodes instead of 5
                self.fusion_agent.decay_epsilon()
            
            # Enhanced logging every episode for debugging
            if episode % 5 == 0 or episode < 10:  # More frequent early logging
                print(f"\n📊 Episode {episode + 1}/{episodes} Stats:")
                print(f"  Accuracy: {episode_accuracy:.4f}")
                print(f"  Normalized Reward: {avg_episode_reward:.4f}")
                print(f"  Raw Reward Range: [{reward_stats[-1]['raw_min']:.2f}, {reward_stats[-1]['raw_max']:.2f}]")
                print(f"  Avg Loss: {avg_episode_loss:.6f}")
                print(f"  Avg Q-Value: {avg_q_value:.4f}")
                print(f"  Epsilon: {self.fusion_agent.epsilon:.4f}")
                print(f"  Buffer Size: {len(self.fusion_agent.replay_buffer)}")
                
                # Action distribution analysis
                most_used_actions = np.argsort(episode_action_counts)[-3:][::-1]
                print(f"  Top Actions: {[f'α={self.fusion_agent.alpha_values[i]:.2f}({episode_action_counts[i]:.0f})' for i in most_used_actions]}")
                
                # Q-value analysis for sample states
                self._analyze_q_values_for_sample_states()
            
            # Validation and logging
            if (episode + 1) % validation_interval == 0:
                val_results = self.fusion_agent.validate_agent(self.validation_data, num_samples=1000)
                validation_scores.append(val_results)
                
                print(f"\n🎯 Validation Results (Episode {episode + 1}):")
                print(f"  Validation Accuracy: {val_results['accuracy']:.4f}")
                print(f"  Validation Reward: {val_results['avg_reward']:.4f}")
                print(f"  Avg Alpha: {val_results['avg_alpha']:.4f} ± {val_results['alpha_std']:.4f}")
                
                # Early stopping check
                current_val_score = val_results['accuracy']
                if current_val_score > best_validation_score:
                    best_validation_score = current_val_score
                    patience_counter = 0
                    self.save_fusion_agent("saved_models/fusion_checkpoints", "best")
                    print(f"  🏆 New best validation score!")
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"\n🛑 Early stopping triggered after {patience_counter} validation cycles")
                    break
            
            # Periodic checkpoints
            if (episode + 1) % save_interval == 0:
                self.save_fusion_agent("saved_models/fusion_checkpoints", f"episode_{episode+1}")
        
        print(f"\n✓ Fusion agent training completed!")
        print(f"Best validation accuracy: {best_validation_score:.4f}")
        
        # Enhanced analysis plots
        self._plot_enhanced_training_progress(
            episode_accuracies, episode_rewards, episode_losses, 
            episode_q_values, action_distributions, reward_stats, validation_scores
        )
        
        return {
            'episode_accuracies': episode_accuracies,
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'episode_q_values': episode_q_values,
            'action_distributions': action_distributions,
            'reward_stats': reward_stats,
            'validation_scores': validation_scores,
            'best_validation_score': best_validation_score
        }

    def evaluate_fusion_strategies(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Comprehensive evaluation comparing different fusion strategies.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation results for different strategies
        """
        print(f"\n=== Evaluating Fusion Strategies ===")
        
        # Extract test data
        test_anomaly_scores, test_gat_probs, test_labels = self.data_extractor.extract_fusion_data(
            test_loader, max_samples=None
        )
        
        results = {}
        
        # 1. Individual model performance
        print("Evaluating individual models...")
        
        # VGAE only (using threshold)
        vgae_predictions = [1 if score > 0.5 else 0 for score in test_anomaly_scores]
        results['vgae_only'] = self._calculate_metrics(test_labels, vgae_predictions, "VGAE Only")
        
        # GAT only
        gat_predictions = [1 if prob > 0.5 else 0 for prob in test_gat_probs]
        results['gat_only'] = self._calculate_metrics(test_labels, gat_predictions, "GAT Only")
        
        # 2. Fixed fusion strategies
        print("Evaluating fixed fusion strategies...")
        
        fixed_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        for alpha in fixed_alphas:
            fused_scores = [(1 - alpha) * a + alpha * g for a, g in zip(test_anomaly_scores, test_gat_probs)]
            fused_predictions = [1 if score > 0.5 else 0 for score in fused_scores]
            results[f'fixed_alpha_{alpha}'] = self._calculate_metrics(
                test_labels, fused_predictions, f"Fixed α={alpha}"
            )
        
        # 3. Adaptive fusion (trained agent)
        print("Evaluating adaptive fusion...")
        
        adaptive_predictions = []
        adaptive_alphas = []
        
        for anomaly_score, gat_prob in zip(test_anomaly_scores, test_gat_probs):
            alpha, _, _ = self.fusion_agent.select_action(anomaly_score, gat_prob, training=False)
            fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
            prediction = 1 if fused_score > 0.5 else 0
            
            adaptive_predictions.append(prediction)
            adaptive_alphas.append(alpha)
        
        results['adaptive_fusion'] = self._calculate_metrics(
            test_labels, adaptive_predictions, "Adaptive Fusion"
        )
        results['adaptive_fusion']['avg_alpha'] = np.mean(adaptive_alphas)
        results['adaptive_fusion']['alpha_std'] = np.std(adaptive_alphas)
        
        # 4. Print comparison table
        self._print_comparison_table(results)
        
        # 5. Generate analysis plots
        self._plot_fusion_analysis(test_anomaly_scores, test_gat_probs, test_labels, adaptive_alphas)
        
        return results

    def _calculate_metrics(self, true_labels: List, predictions: List, method_name: str) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        return {
            'method': method_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }

    def _print_comparison_table(self, results: Dict):
        """Print a formatted comparison table of all methods."""
        print(f"\n📊 FUSION STRATEGY COMPARISON")
        print("=" * 80)
        print(f"{'Method':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 80)
        
        # Sort by accuracy for better presentation
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for method_key, metrics in sorted_results:
            method_name = metrics['method']
            print(f"{method_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
        
        print("=" * 80)
        
        # Highlight best performance
        best_method = sorted_results[0]
        print(f"🏆 Best Performance: {best_method[1]['method']} "
              f"(Accuracy: {best_method[1]['accuracy']:.4f})")

    def _plot_training_progress(self, accuracies: List, rewards: List, validation_scores: List):
        """Plot training progress visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(1, len(accuracies) + 1)
        
        # Training accuracy
        ax1.plot(episodes, accuracies, 'b-', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Training Accuracy')
        ax1.set_title('Training Accuracy Over Episodes')
        ax1.grid(True, alpha=0.3)
        
        # Training rewards
        ax2.plot(episodes, rewards, 'g-', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Training Rewards Over Episodes')
        ax2.grid(True, alpha=0.3)
        
        # Validation accuracy
        if validation_scores:
            val_episodes = [i * 100 for i in range(1, len(validation_scores) + 1)]
            val_accuracies = [score['accuracy'] for score in validation_scores]
            ax3.plot(val_episodes, val_accuracies, 'r-', linewidth=2, marker='o')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Validation Accuracy')
            ax3.set_title('Validation Accuracy')
            ax3.grid(True, alpha=0.3)
        
        # Fusion weights used
        if validation_scores:
            val_alphas = [score['avg_alpha'] for score in validation_scores]
            ax4.plot(val_episodes, val_alphas, 'm-', linewidth=2, marker='s')
            ax4.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Balanced')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Average Fusion Weight')
            ax4.set_title('Learned Fusion Weights')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/fusion_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_fusion_analysis(self, anomaly_scores: List, gat_probs: List, 
                            labels: List, adaptive_alphas: List):
        """Plot detailed fusion analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert to numpy arrays
        anomaly_scores = np.array(anomaly_scores)
        gat_probs = np.array(gat_probs)
        labels = np.array(labels)
        adaptive_alphas = np.array(adaptive_alphas)
        
        # 1. State space visualization with adaptive weights
        scatter = ax1.scatter(anomaly_scores, gat_probs, c=adaptive_alphas, 
                            cmap='viridis', alpha=0.6, s=20)
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('GAT Probability')
        ax1.set_title('Learned Fusion Policy\n(Color = Fusion Weight)')
        plt.colorbar(scatter, ax=ax1, label='Fusion Weight (α)')
        
        # 2. Fusion weight distribution by class
        normal_alphas = adaptive_alphas[labels == 0]
        attack_alphas = adaptive_alphas[labels == 1]
        
        ax2.hist([normal_alphas, attack_alphas], bins=20, alpha=0.7, 
                label=['Normal', 'Attack'], color=['blue', 'red'])
        ax2.set_xlabel('Fusion Weight (α)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Fusion Weight Distribution by Class')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Model agreement analysis
        model_diff = np.abs(anomaly_scores - gat_probs)
        ax3.scatter(model_diff, adaptive_alphas, alpha=0.6, s=20)
        ax3.set_xlabel('Model Disagreement |VGAE - GAT|')
        ax3.set_ylabel('Fusion Weight (α)')
        ax3.set_title('Fusion Strategy vs Model Agreement')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance by confidence level
        confidence_levels = (anomaly_scores + gat_probs) / 2
        high_conf_mask = confidence_levels > 0.7
        low_conf_mask = confidence_levels < 0.3
        
        ax4.boxplot([adaptive_alphas[high_conf_mask], adaptive_alphas[low_conf_mask]], 
                   labels=['High Confidence', 'Low Confidence'])
        ax4.set_ylabel('Fusion Weight (α)')
        ax4.set_title('Fusion Weights by Confidence Level')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/fusion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_fusion_agent(self, save_folder: str, suffix: str = "final"):
        """Save the trained fusion agent."""
        os.makedirs(save_folder, exist_ok=True)
        filepath = os.path.join(save_folder, f'fusion_agent_{suffix}.pth')
        self.fusion_agent.save_agent(filepath)
        
        # Also save configuration
        config_path = os.path.join(save_folder, f'fusion_config_{suffix}.json')
        import json
        config = {
            'num_ids': self.num_ids,
            'embedding_dim': self.embedding_dim,
            'alpha_steps': len(self.fusion_agent.alpha_values),
            'fusion_weights': FUSION_WEIGHTS,
            'device': str(self.device)
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Fusion agent and config saved to {save_folder}")
    
    def _analyze_q_values_for_sample_states(self):
        """Analyze Q-values for sample states to check learning."""
        if not self.training_data:
            return
            
        # Sample a few representative states
        sample_indices = [0, len(self.training_data)//4, len(self.training_data)//2, 
                         3*len(self.training_data)//4, len(self.training_data)-1]
        
        print(f"  Q-Value Analysis for Sample States:")
        for i, idx in enumerate(sample_indices[:3]):  # Just show first 3
            if idx < len(self.training_data):
                anomaly_score, gat_prob, true_label = self.training_data[idx]
                state = self.fusion_agent.normalize_state(anomaly_score, gat_prob)
                
                with torch.no_grad():
                    state_tensor = torch.tensor(state).unsqueeze(0).to(self.fusion_agent.device)
                    q_values = self.fusion_agent.q_network(state_tensor).squeeze()
                    best_action = torch.argmax(q_values).item()
                    best_alpha = self.fusion_agent.alpha_values[best_action]
                    max_q = q_values[best_action].item()
                
                print(f"    State{i+1}: VGAE={anomaly_score:.3f}, GAT={gat_prob:.3f}, Label={true_label} "
                      f"→ Best α={best_alpha:.2f} (Q={max_q:.3f})")

    def _plot_enhanced_training_progress(self, accuracies, rewards, losses, q_values, 
                                       action_distributions, reward_stats, validation_scores):
        """Enhanced training progress visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        episodes = range(1, len(accuracies) + 1)
        
        # Training accuracy
        axes[0,0].plot(episodes, accuracies, 'b-', linewidth=2, alpha=0.7)
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Training Accuracy')
        axes[0,0].set_title('Training Accuracy Over Episodes')
        axes[0,0].grid(True, alpha=0.3)
        
        # Training rewards (normalized)
        axes[0,1].plot(episodes, rewards, 'g-', linewidth=2, alpha=0.7)
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Normalized Reward')
        axes[0,1].set_title('Training Rewards (Normalized)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Training losses
        axes[0,2].plot(episodes, losses, 'r-', linewidth=2, alpha=0.7)
        axes[0,2].set_xlabel('Episode')
        axes[0,2].set_ylabel('Average Loss')
        axes[0,2].set_title('Training Loss Over Episodes')
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].set_yscale('log')  # Log scale for loss
        
        # Q-values
        axes[1,0].plot(episodes, q_values, 'm-', linewidth=2, alpha=0.7)
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Average Q-Value')
        axes[1,0].set_title('Q-Values Over Episodes')
        axes[1,0].grid(True, alpha=0.3)
        
        # Action distribution heatmap
        if action_distributions:
            action_matrix = np.array(action_distributions).T
            im = axes[1,1].imshow(action_matrix, aspect='auto', cmap='viridis', origin='lower')
            axes[1,1].set_xlabel('Episode')
            axes[1,1].set_ylabel('Action Index (α value)')
            axes[1,1].set_title('Action Distribution Heatmap')
            
            # Add alpha value labels
            alpha_ticks = range(0, len(self.fusion_agent.alpha_values), 5)
            alpha_labels = [f'{self.fusion_agent.alpha_values[i]:.1f}' for i in alpha_ticks]
            axes[1,1].set_yticks(alpha_ticks)
            axes[1,1].set_yticklabels(alpha_labels)
            plt.colorbar(im, ax=axes[1,1], label='Usage Frequency')
        
        # Raw reward statistics
        if reward_stats:
            raw_means = [rs['raw_mean'] for rs in reward_stats]
            raw_stds = [rs['raw_std'] for rs in reward_stats]
            axes[1,2].plot(episodes, raw_means, 'orange', linewidth=2, label='Mean')
            axes[1,2].fill_between(episodes, 
                                 np.array(raw_means) - np.array(raw_stds),
                                 np.array(raw_means) + np.array(raw_stds),
                                 alpha=0.3, color='orange')
            axes[1,2].set_xlabel('Episode')
            axes[1,2].set_ylabel('Raw Reward')
            axes[1,2].set_title('Raw Reward Statistics')
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/enhanced_fusion_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_optimized_data_loaders(train_subset=None, test_dataset=None, full_train_dataset=None, 
                                 batch_size: int = 1024, device: str = 'cuda') -> List[DataLoader]:
    """Create optimized data loaders for fusion training."""
    if torch.cuda.is_available() and device == 'cuda':
        batch_size = 2048
        num_workers = 6
        pin_memory = True
        prefetch_factor = 3
        persistent_workers = True
    else:
        batch_size = min(batch_size, 1024)
        num_workers = 4
        pin_memory = False
        prefetch_factor = None
        persistent_workers = False

    print(f"✓ Optimized DataLoader: batch_size={batch_size}, workers={num_workers}")

    loaders = []
    loader_configs = [
        (train_subset, True),
        (test_dataset, False),
        (full_train_dataset, True)
    ]
    for dataset, shuffle in loader_configs:
        if dataset is not None:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor
            )
            loaders.append(loader)
    return loaders if len(loaders) > 1 else loaders[0]

@hydra.main(config_path="../conf", config_name="base", version_base=None)
def main(config: DictConfig):
    """
    Main fusion training pipeline.
    
    Args:
        config: Hydra configuration object
    """
    print(f"\n{'='*80}")
    print("FUSION TRAINING FOR CAN BUS ANOMALY DETECTION")
    print(f"{'='*80}")
    
    # Setup
    setup_gpu_optimization()
    config_dict = OmegaConf.to_container(config, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # Dataset configuration
    dataset_key = config_dict['root_folder']
    if dataset_key not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    
    root_folder = DATASET_PATHS[dataset_key]
    
    # Create directories
    for dir_name in ["images", "saved_models", "saved_models/fusion_checkpoints"]:
        os.makedirs(dir_name, exist_ok=True)
    
    # === Data Loading and Preprocessing ===
    print(f"\n=== Data Loading and Preprocessing ===")
    id_mapping = build_id_mapping_from_normal(root_folder)
    
    start_time = time.time()
    dataset = graph_creation(root_folder, id_mapping=id_mapping, window_size=100)
    preprocessing_time = time.time() - start_time
    
    print(f"✓ Dataset: {len(dataset)} graphs, {len(id_mapping)} CAN IDs")
    print(f"✓ Preprocessing time: {preprocessing_time:.2f}s")
    
    # Configuration
    TRAIN_RATIO = config_dict.get('train_ratio', 0.8)
    BATCH_SIZE = config_dict.get('batch_size', 1024)
    FUSION_EPISODES = config_dict.get('fusion_episodes', 1000)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train/test split
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    # Create data loaders
    train_loader = create_optimized_data_loaders(None, None, train_dataset, BATCH_SIZE, str(device))
    test_loader = create_optimized_data_loaders(None, test_dataset, None, BATCH_SIZE, str(device))
    
    print(f"✓ Data split: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # === Initialize Fusion Pipeline ===
    pipeline = FusionTrainingPipeline(
        num_ids=len(id_mapping), 
        embedding_dim=8, 
        device=str(device)
    )
    
    # === Load Pre-trained Models ===
    autoencoder_path = f"saved_models1/autoencoder_best_{dataset_key}.pth"
    classifier_path = f"saved_models1/classifier_{dataset_key}.pth"

    try:
        pipeline.load_pretrained_models(autoencoder_path, classifier_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run osc_training_AD.py first to train the base models!")
        return
    
    # === Prepare Fusion Data ===
    pipeline.prepare_fusion_data(
        train_loader, 
        test_loader, 
        max_train_samples=50000,  # Limit for memory efficiency
        max_val_samples=10000
    )
    
    # === Initialize and Train Fusion Agent ===
    pipeline.initialize_fusion_agent(
        alpha_steps=21,           # 0.0, 0.05, ..., 1.0
        lr=1e-3,                 # Learning rate
        epsilon=0.3,             # Initial exploration
        buffer_size=100000       # Experience replay buffer
    )
    
    # Train the fusion agent
    training_results = pipeline.train_fusion_agent(
        episodes=FUSION_EPISODES,
        validation_interval=100,
        early_stopping_patience=10,
        save_interval=200
    )
    
    # === Final Evaluation ===
    print(f"\n=== Final Evaluation and Comparison ===")
    evaluation_results = pipeline.evaluate_fusion_strategies(test_loader)
    
    # === Save Final Model ===
    pipeline.save_fusion_agent("saved_models", dataset_key)
    
    # === Summary ===
    print(f"\n🎉 FUSION TRAINING COMPLETED!")
    print(f"Best validation accuracy: {training_results['best_validation_score']:.4f}")
    
    # Find best method from evaluation
    best_method = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"Best fusion method: {best_method[1]['method']} (Acc: {best_method[1]['accuracy']:.4f})")
    
    log_memory_usage("Final")

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        end_time = time.time()
        print(f"\n⏱️  Total runtime: {end_time - start_time:.2f} seconds")
        cleanup_memory()