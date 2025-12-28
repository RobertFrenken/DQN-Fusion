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

from contextlib import nullcontext
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any
import warnings
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import psutil

# Import your existing modules
from models.models import GATWithJK, GraphAutoencoderNeighborhood
from models.adaptive_fusion import EnhancedDQNFusionAgent
from archive.preprocessing import graph_creation, build_id_mapping_from_normal
from utils.utils_logging import setup_gpu_optimization, log_memory_usage, cleanup_memory
from utils.cache_manager import CacheManager

# Import new organized modules
from config.fusion_config import DATASET_PATHS, FUSION_WEIGHTS
from config.plotting_config import COLOR_SCHEMES, apply_publication_style, save_publication_figure
from gpu_monitor import GPUMonitor
from fusion_extractor import FusionDataExtractor
from utils.gpu_utils import detect_gpu_capabilities_unified, create_optimized_data_loaders

warnings.filterwarnings('ignore', category=UserWarning)

class FusionTrainingPipeline:
    """Complete pipeline for training the fusion agent."""
    
    def __init__(self, num_ids: int, embedding_dim: int = 8, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Require GPU for this optimized pipeline
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError("This fusion training pipeline requires CUDA GPU. CPU mode not supported.")
        
        self.num_ids = num_ids
        self.embedding_dim = embedding_dim
        self.gpu_info = self._detect_gpu_capabilities()
        
        # Initialize GPU monitoring
        self.gpu_monitor = GPUMonitor(self.device)
        
        # Models (will be loaded)
        self.autoencoder = None
        self.classifier = None
        self.fusion_agent = None
        self.data_extractor = None
        
        # Training data
        self.training_data = None
        self.validation_data = None
        
        print(f"✓ Fusion Training Pipeline initialized on {device}")
        if self.gpu_info:
            print(f"  GPU: {self.gpu_info['name']} ({self.gpu_info['memory_gb']:.1f}GB)")
            print(f"  Batch size: {self.gpu_info['batch_size']}")
        print(f"  GPU tracking enabled")
    
    def _detect_gpu_capabilities(self):
        """Detect GPU capabilities using the unified configuration function."""
        return detect_gpu_capabilities_unified(str(self.device))

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




    def _process_gpu_batch_fully(self, processing_indices: List[int]) -> Dict:
        """Keep everything on GPU, eliminate CPU bottlenecks."""
        
        # Convert all data to GPU tensors at once (no Python loops)
        batch_anomaly_scores = torch.tensor([self.training_data[i][0] for i in processing_indices], 
                                        device=self.device, dtype=torch.float32)
        batch_gat_probs = torch.tensor([self.training_data[i][1] for i in processing_indices],
                                    device=self.device, dtype=torch.float32)  
        batch_labels = torch.tensor([self.training_data[i][2] for i in processing_indices],
                                device=self.device, dtype=torch.float32)
        
        # Get states from GPU cache
        batch_states = self.training_states_gpu[processing_indices]
        
        # GPU-vectorized action selection
        with torch.no_grad():
            q_values = self.fusion_agent.q_network(batch_states)
            if np.random.rand() < self.fusion_agent.epsilon:
                actions = torch.randint(0, self.fusion_agent.action_dim, 
                                    (len(processing_indices),), device=self.device)
            else:
                actions = q_values.argmax(dim=1)
        
        # GPU-vectorized reward computation
        alpha_tensor = torch.tensor([self.fusion_agent.alpha_values[a] for a in actions.cpu()],
                                device=self.device, dtype=torch.float32)
        
        # All fusion math on GPU
        fusion_scores = (1 - alpha_tensor) * batch_anomaly_scores + alpha_tensor * batch_gat_probs
        predictions = (fusion_scores > 0.5).float()
        correct = (predictions == batch_labels)  # Keep as boolean for torch.where
        
        # Vectorized rewards (all on GPU)
        base_rewards = torch.where(correct, 3.0, -3.0)
        agreement = 1.0 - torch.abs(batch_anomaly_scores - batch_gat_probs)
        agreement_bonus = torch.where(correct, agreement, -agreement)
        total_rewards = torch.clamp((base_rewards + agreement_bonus) * 0.5, -1.0, 1.0)
        
        return {
            'states': batch_states,
            'actions': actions,
            'rewards': total_rewards,
            'episode_reward': total_rewards.sum().item(),
            'episode_correct': correct.float().sum().item(),
            'episode_samples': len(processing_indices)
        }

    def prepare_fusion_data(self, train_loader: DataLoader, val_loader: DataLoader, 
                        max_train_samples: int = 200000, max_val_samples: int = 50000):
        """
        GPU-optimized preparation of training and validation data for fusion learning.
        Pre-computes and caches all states on GPU to eliminate CPU bottleneck.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_train_samples: Maximum training samples to use
            max_val_samples: Maximum validation samples to use
        """
        print(f"\n=== Preparing Fusion Data (GPU-Optimized) ===")
        
        # Check for cached fusion predictions first
        cached_fusion_data = None
        if hasattr(self, 'cache_mgr') and self.cache_mgr:
            cached_fusion_data = self.cache_mgr.load_cache('fusion_predictions')
        
        if cached_fusion_data is not None:
            print("📥 Loading fusion predictions from cache...")
            self.training_data = cached_fusion_data['training_data']
            self.validation_data = cached_fusion_data['validation_data']
            self.test_data = self.validation_data
            print(f"🚀 Cache hit! Loaded {len(self.training_data)} train, {len(self.validation_data)} val samples")
            print("💾 Saved ~25 minutes of GPU extraction time!")
        else:
            print("🔄 Cache miss - extracting fusion predictions...")
            
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
            self.test_data = self.validation_data  # val_loader is actually test data in main()
            
            # Cache the fusion predictions
            if hasattr(self, 'cache_mgr') and self.cache_mgr:
                fusion_cache_data = {
                    'training_data': self.training_data,
                    'validation_data': self.validation_data
                }
                self.cache_mgr.save_cache(fusion_cache_data, 'fusion_predictions', metadata={
                    'train_samples': len(self.training_data),
                    'val_samples': len(self.validation_data),
                    'max_train_samples': max_train_samples,
                    'max_val_samples': max_val_samples
                })
        
        print(f"✓ Fusion data prepared:")
        print(f"  Training samples: {len(self.training_data)}")
        print(f"  Validation samples: {len(self.validation_data)}")
        
        # ===== NEW: Pre-compute all states on GPU =====
        if self.device.type == 'cuda':
            print("🚀 Pre-computing states on GPU for maximum throughput...")
            
            try:
                # Pre-compute training states
                train_states_list = []
                for anomaly_score, gat_prob, _ in self.training_data:
                    state = self.fusion_agent.normalize_state(anomaly_score, gat_prob)
                    train_states_list.append(state)
                
                self.training_states_gpu = torch.tensor(
                    np.array(train_states_list), 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                # Pre-compute validation states
                val_states_list = []
                for anomaly_score, gat_prob, _ in self.validation_data:
                    state = self.fusion_agent.normalize_state(anomaly_score, gat_prob)
                    val_states_list.append(state)
                
                self.validation_states_gpu = torch.tensor(
                    np.array(val_states_list), 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                print(f"✓ GPU state cache created: {self.training_states_gpu.shape} training, {self.validation_states_gpu.shape} validation")
                print(f"  GPU memory used: {self.training_states_gpu.element_size() * self.training_states_gpu.nelement() / (1024**2):.2f}MB")
                
            except RuntimeError as e:
                print(f"⚠️ GPU memory error during state precomputation: {e}")
                print("Falling back to CPU computation during training...")
                self.training_states_gpu = None
                self.validation_states_gpu = None
        else:
            # Ensure GPU state tensors are None for CPU mode
            self.training_states_gpu = None
            self.validation_states_gpu = None
        
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

    def initialize_fusion_agent(self, config_dict: dict = None):
        """Initialize fusion agent using unified GPU configuration."""
        state_dim = 4
        
        # Use unified GPU configuration
        batch_size = self.gpu_info['dqn_batch_size']
        buffer_size = self.gpu_info['buffer_size']
            
        self.fusion_agent = EnhancedDQNFusionAgent(
            alpha_steps=21, lr=1e-3, epsilon=0.3,
            buffer_size=buffer_size, batch_size=batch_size,
            device=str(self.device), state_dim=state_dim
        )
        
        print(f"✓ Fusion Agent initialized: {batch_size:,} batch, {buffer_size:,} buffer")


    def train_fusion_agent(self, episodes: int = 50, dataset_key: str = 'default'):
        """GPU-optimized fusion agent training."""
        print(f"\n=== Training Fusion Agent (GPU-Optimized) ===")
        print(f"Episodes: {episodes}")
        
        if not self.training_data or not self.fusion_agent:
            raise ValueError("Training data and fusion agent must be initialized first")
        
        # ===== USE UNIFIED GPU CONFIGURATION =====
        gpu_processing_batch = self.gpu_info['gpu_processing_batch']
        dqn_training_batch = self.gpu_info['dqn_training_batch'] 
        training_steps_per_episode = self.gpu_info['training_steps_per_episode']
        episode_sample_ratio = self.gpu_info['episode_sample_ratio']
        
        print(f"🎯 GPU Batch Configuration:")
        print(f"  GPU Processing Batch: {gpu_processing_batch:,}")
        print(f"  DQN Training Batch: {dqn_training_batch:,}")
        print(f"  Training Steps per Episode: {training_steps_per_episode}")
        print(f"  Episode Sample Ratio: {episode_sample_ratio:.1%}")
        
        # Training setup
        training_start_time = time.time()
        episode_rewards = []
        episode_accuracies = []
        episode_losses = []
        episode_q_values = []
        action_distributions = []
        validation_scores = []
        best_validation_score = -float('inf')
        patience_counter = 0
        
        # Create checkpoint directory
        checkpoint_dir = f"saved_models/fusion_checkpoints/{dataset_key}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # ===== MAIN TRAINING LOOP =====
        for episode in range(episodes):
            episode_start_time = time.time()
            
            # Determine episode sample size
            episode_size = max(1000, int(len(self.training_data) * episode_sample_ratio))
            episode_indices = np.random.choice(len(self.training_data), 
                                            min(episode_size, len(self.training_data)), 
                                            replace=False)
            
            # Progress reporting
            if episode % 10 == 0:
                elapsed = time.time() - training_start_time
                eps_per_min = (episode / elapsed * 60) if elapsed > 0 else 0
                print(f"\n📊 Episode {episode}/{episodes} | {eps_per_min:.1f} eps/min | "
                    f"ε={self.fusion_agent.epsilon:.3f} | Buffer: {len(self.fusion_agent.replay_buffer):,}")
            
            # Episode statistics
            episode_reward = 0.0
            episode_correct = 0
            episode_samples = 0
            episode_losses_list = []
            episode_q_list = []
            episode_action_counts = np.zeros(len(self.fusion_agent.alpha_values))
            
            # ===== BATCH ACCUMULATION STRATEGY =====
            # Process small batches and accumulate for GPU efficiency
            
            # ===== GPU PROCESSING PATH =====
            if episode % 50 == 0:
                print(f"🚀 GPU batch processing: {gpu_processing_batch} samples")
            
            # Process episode in GPU-sized batches
            for batch_start in range(0, len(episode_indices), gpu_processing_batch):
                batch_end = min(batch_start + gpu_processing_batch, len(episode_indices))
                processing_indices = episode_indices[batch_start:batch_end]
                
                # GPU batch processing
                batch_results = self._process_gpu_batch_fully(processing_indices)
                
                # Store experiences in batches
                self.fusion_agent.store_batch_experiences_gpu(
                    batch_results['states'],
                    batch_results['actions'], 
                    batch_results['rewards']
                )
                
                # Update episode statistics
                episode_reward += batch_results['episode_reward']
                episode_correct += batch_results['episode_correct']
                episode_samples += batch_results['episode_samples']
                
                # Track actions for episode statistics
                for action_idx in batch_results['actions'].cpu().numpy():
                    episode_action_counts[action_idx] += 1
                        
            
            # ===== DQN TRAINING PHASE =====
            # Train with large, stable batches for good learning
            if len(self.fusion_agent.replay_buffer) >= dqn_training_batch:
                for training_step in range(training_steps_per_episode):
                    loss = self.fusion_agent.train_step()
                    
                    if loss is not None:
                        episode_losses_list.append(loss)
                        
                        # Sample Q-values for monitoring (less frequently)
                        if training_step == 0:
                            with torch.no_grad():
                                sample_idx = np.random.randint(0, len(self.training_data))
                                if (hasattr(self, 'training_states_gpu') and 
                                    self.training_states_gpu is not None):
                                    sample_state = self.training_states_gpu[sample_idx:sample_idx+1]
                                else:
                                    anomaly_s, gat_p, _ = self.training_data[sample_idx]
                                    sample_state = torch.tensor(
                                        self.fusion_agent.normalize_state(anomaly_s, gat_p),
                                        dtype=torch.float32, device=self.device
                                    ).unsqueeze(0)
                                
                                q_vals = self.fusion_agent.q_network(sample_state)
                                episode_q_list.append(q_vals.mean().item())
            
            # ===== END EPISODE PROCESSING =====
            # Decay exploration
            self.fusion_agent.decay_epsilon()
            
            # Calculate episode metrics
            episode_accuracy = episode_correct / episode_samples if episode_samples > 0 else 0
            avg_episode_reward = episode_reward / episode_samples if episode_samples > 0 else 0
            avg_episode_loss = np.mean(episode_losses_list) if episode_losses_list else 0
            avg_q_value = np.mean(episode_q_list) if episode_q_list else 0
            
            # Store episode statistics
            episode_rewards.append(avg_episode_reward)
            episode_accuracies.append(episode_accuracy)
            episode_losses.append(avg_episode_loss)
            episode_q_values.append(avg_q_value)
            action_distributions.append(episode_action_counts / max(episode_samples, 1))
            
            # ===== VALIDATION AND CHECKPOINTING =====
            if (episode + 1) % 25 == 0:  # Fixed validation interval
                val_results = self.fusion_agent.validate_agent(
                    self.validation_data, 
                    num_samples=min(5000, len(self.validation_data))
                )
                validation_scores.append(val_results)
                
                print(f"  📈 Validation: Acc={val_results['accuracy']:.4f}, "
                    f"Reward={val_results['avg_reward']:.3f}, α={val_results['avg_alpha']:.3f}")
                
                # Early stopping check
                if val_results['accuracy'] > best_validation_score:
                    best_validation_score = val_results['accuracy']
                    patience_counter = 0
                    self.save_fusion_agent(checkpoint_dir, f"best_ep{episode}", dataset_key)
                else:
                    patience_counter += 1
                
                if patience_counter >= 50:  # Fixed patience
                    print(f"⏹️  Early stopping at episode {episode}")
                    break
            
            # Periodic checkpoint save
            if (episode + 1) % 50 == 0:  # Fixed save interval
                self.save_fusion_agent(checkpoint_dir, f"checkpoint_ep{episode}", dataset_key)
            
            # GPU monitoring
            if episode % 10 == 0:
                self.gpu_monitor.record_gpu_stats(episode)
                episode_time = time.time() - episode_start_time
                if episode > 0:
                    print(f"  ⏱️  Episode time: {episode_time:.2f}s, "
                        f"Acc: {episode_accuracy:.3f}, Reward: {avg_episode_reward:.3f}")
        
        # ===== TRAINING COMPLETION =====
        total_training_time = time.time() - training_start_time
        print(f"\n✅ Fusion agent training completed in {total_training_time/60:.1f} minutes!")
        print(f"🏆 Best validation accuracy: {best_validation_score:.4f}")
        
        # Performance summary
        self.gpu_monitor.print_performance_summary()
        
        # Generate training plots
        self._plot_enhanced_training_progress(
            episode_accuracies, episode_rewards, episode_losses, 
            episode_q_values, action_distributions, [], validation_scores, dataset_key
        )
        
        return {
            'episode_accuracies': episode_accuracies,
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'episode_q_values': episode_q_values,
            'action_distributions': action_distributions,
            'validation_scores': validation_scores,
            'best_validation_score': best_validation_score,
            'performance_summary': self.gpu_monitor.get_performance_summary(),
            'total_training_time_minutes': total_training_time / 60
        }

    def evaluate_fusion_strategies(self, dataset_key: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation comparing different fusion strategies.
        
        Returns:
            Dictionary with evaluation results for different strategies
        """
        print(f"\n=== Evaluating Fusion Strategies ===")
        
        # Use already extracted test data (from prepare_fusion_data)
        if not hasattr(self, 'test_data') or not self.test_data:
            raise ValueError("No test data available. Run prepare_fusion_data() first.")
            
        test_anomaly_scores = [data[0] for data in self.test_data]
        test_gat_probs = [data[1] for data in self.test_data]
        test_labels = [data[2] for data in self.test_data]
        
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
        self._plot_fusion_analysis(test_anomaly_scores, test_gat_probs, test_labels, adaptive_alphas, dataset_key)
        
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
        """Plot training progress visualization with publication-ready styling."""
        apply_publication_style()
        plt.ioff()  # Turn off interactive mode
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        episodes = range(1, len(accuracies) + 1)
        colors = COLOR_SCHEMES['training']
        
        # Training accuracy
        ax1.plot(episodes, accuracies, color=colors['accuracy'], linewidth=2.5, alpha=0.8)
        ax1.set_xlabel('Training Episode')
        ax1.set_ylabel('Training Accuracy')
        ax1.set_title('Training Accuracy Progression', fontweight='bold')
        ax1.set_ylim([0, 1.02])
        
        # Training rewards
        ax2.plot(episodes, rewards, color=colors['reward'], linewidth=2.5, alpha=0.8)
        ax2.set_xlabel('Training Episode')
        ax2.set_ylabel('Average Normalized Reward')
        ax2.set_title('Training Reward Evolution', fontweight='bold')
        
        # Validation accuracy
        if validation_scores:
            val_episodes = [i * 100 for i in range(1, len(validation_scores) + 1)]
            val_accuracies = [score['accuracy'] for score in validation_scores]
            ax3.plot(val_episodes, val_accuracies, color=COLOR_SCHEMES['validation']['primary'], 
                    linewidth=2.5, marker='o', markersize=6, alpha=0.8)
            ax3.set_xlabel('Training Episode')
            ax3.set_ylabel('Validation Accuracy')
            ax3.set_title('Validation Performance', fontweight='bold')
            # Auto-scale validation accuracy to show data range better
            if val_accuracies:
                y_min = max(0, min(val_accuracies) - 0.01)
                y_max = min(1.0, max(val_accuracies) + 0.01)
                ax3.set_ylim([y_min, y_max])
        
        # Fusion weights used
        if validation_scores:
            val_alphas = [score['avg_alpha'] for score in validation_scores]
            ax4.plot(val_episodes, val_alphas, color=colors['q_values'], 
                    linewidth=2.5, marker='s', markersize=6, alpha=0.8, label='Learned α')
            ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Balanced Fusion')
            ax4.set_xlabel('Training Episode')
            ax4.set_ylabel('Average Fusion Weight (α)')
            ax4.set_title('Adaptive Fusion Strategy', fontweight='bold')
            ax4.legend(loc='best')
        
        plt.tight_layout()
        save_publication_figure(fig, 'images/fusion_training_progress.png')
        plt.close(fig)
        plt.ion()  # Turn interactive mode back on

    def _plot_fusion_analysis(self, anomaly_scores: List, gat_probs: List, 
                            labels: List, adaptive_alphas: List,
                            dataset_key: str):
        """Add fusion analysis plots to the existing training progress figure."""
        # Check if we have the figure from training progress
        if not hasattr(self, '_current_fig') or not hasattr(self, '_current_axes'):
            # Fallback: create new figure if training plots weren't called first
            apply_publication_style()
            plt.ioff()
            fig, axes = plt.subplots(2, 4, figsize=(32, 16))
            # Hide first 5 subplots if training wasn't called
            for i in range(2):
                for j in range(4):
                    if i == 0 or (i == 1 and j == 0):
                        axes[i,j].set_visible(False)
        else:
            fig = self._current_fig
            axes = self._current_axes
        
        # Convert to numpy arrays
        anomaly_scores = np.array(anomaly_scores)
        gat_probs = np.array(gat_probs)
        labels = np.array(labels)
        adaptive_alphas = np.array(adaptive_alphas)
        
        colors = COLOR_SCHEMES['fusion_analysis']
        
        # Row 2, Col 2: State space visualization - simplified scatter plot
        x_min, x_max = max(0, anomaly_scores.min() - 0.02), min(1, anomaly_scores.max() + 0.02)
        y_min, y_max = max(0, gat_probs.min() - 0.02), min(1, gat_probs.max() + 0.02)
        
        scatter = axes[1,1].scatter(anomaly_scores, gat_probs, c=adaptive_alphas, 
                                  cmap=COLOR_SCHEMES['contour'], s=12, alpha=0.7, 
                                  edgecolors='white', linewidths=0.1)
        
        axes[1,1].set_xlabel('VGAE Anomaly Score')
        axes[1,1].set_ylabel('GAT Classification Probability')
        axes[1,1].set_title('Learned Fusion Policy', fontweight='bold')
        axes[1,1].set_xlim([x_min, x_max])
        axes[1,1].set_ylim([y_min, y_max])
        
        cbar = plt.colorbar(scatter, ax=axes[1,1], shrink=0.8)
        cbar.set_label('Fusion Weight (α)', rotation=270, labelpad=15, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        
        # Row 2, Col 3: Fusion weight distribution by class
        normal_alphas = adaptive_alphas[labels == 0]
        attack_alphas = adaptive_alphas[labels == 1]
        
        bins = 25
        n_normal, bins_normal, _ = axes[1,2].hist(normal_alphas, bins=bins, alpha=0.7, 
                                               color=colors['normal'], edgecolor='black', linewidth=0.8,
                                               label='Normal (Count)', histtype='bar')
        n_attack, bins_attack, _ = axes[1,2].hist(attack_alphas, bins=bins, alpha=0.7,
                                               color=colors['attack'], edgecolor='black', linewidth=0.8,
                                               label='Attack (Count)', histtype='bar')
        
        axes[1,2].set_xlabel('Fusion Weight (α)')
        axes[1,2].set_ylabel('Raw Sample Count', color='black')
        axes[1,2].set_xlim([0, 1])
        axes[1,2].tick_params(axis='y', labelcolor='black')
        
        # Secondary axis - proportional distribution
        ax2_twin = axes[1,2].twinx()
        
        bin_centers = (bins_normal[:-1] + bins_normal[1:]) / 2
        normal_prop = n_normal / len(normal_alphas) if len(normal_alphas) > 0 else n_normal
        attack_prop = n_attack / len(attack_alphas) if len(attack_alphas) > 0 else n_attack
        
        ax2_twin.plot(bin_centers, normal_prop, color=colors['normal'], linewidth=1.5, 
                     linestyle='-', marker='o', markersize=2, alpha=0.9, label='Normal (Proportion)')
        ax2_twin.plot(bin_centers, attack_prop, color=colors['attack'], linewidth=1.5,
                     linestyle='-', marker='s', markersize=2, alpha=0.9, label='Attack (Proportion)')
        
        ax2_twin.set_ylabel('Proportional Distribution', color='gray')
        ax2_twin.tick_params(axis='y', labelcolor='gray')
        
        lines1, labels1 = axes[1,2].get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        axes[1,2].legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        axes[1,2].set_title('Fusion Strategy Distribution', fontweight='bold')
        
        # Row 2, Col 4: Model agreement analysis - simplified
        model_diff = np.abs(anomaly_scores - gat_probs)
        jitter_amount = 0.01
        jittered_alphas = adaptive_alphas + np.random.normal(0, jitter_amount, len(adaptive_alphas))
        
        normal_mask = labels == 0
        attack_mask = labels == 1
        
        axes[1,3].scatter(model_diff[normal_mask], jittered_alphas[normal_mask], 
                         alpha=0.6, s=12, c=colors['normal'], 
                         edgecolors='white', linewidths=0.3, label='Normal Traffic')
        
        axes[1,3].scatter(model_diff[attack_mask], jittered_alphas[attack_mask],
                         alpha=0.8, s=15, c=colors['attack'], 
                         edgecolors='white', linewidths=0.3, label='Attack Traffic')
        
        # Add horizontal reference lines
        key_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        for alpha_val in key_alphas:
            axes[1,3].axhline(y=alpha_val, color='gray', alpha=0.2, linewidth=0.5, linestyle=':')
        
        axes[1,3].set_xlabel('Model Disagreement |VGAE Score - GAT Probability|')
        axes[1,3].set_ylabel('Fusion Weight (α)')
        axes[1,3].set_title('Strategy vs. Model Agreement', fontweight='bold')
        axes[1,3].set_xlim([0, 1])
        axes[1,3].set_ylim([0, 1])
        axes[1,3].legend(loc='best', fontsize=9)
        
        # Update the saved figure
        plt.tight_layout()
        filename = f'images/complete_fusion_training_analysis_{dataset_key}'
        save_publication_figure(fig, filename + '.png')
        plt.close(fig)
        plt.ion()

    def save_fusion_agent(self, save_folder: str, suffix: str = "final", dataset_key: str = "default"):
        """Save the trained fusion agent."""
        os.makedirs(save_folder, exist_ok=True)
        filepath = os.path.join(save_folder, f'fusion_agent_{dataset_key}_{suffix}.pth')
        self.fusion_agent.save_agent(filepath)
        
        # Also save configuration
        config_path = os.path.join(save_folder, f'fusion_config_{dataset_key}_{suffix}.json')
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

    def _plot_enhanced_training_progress(self, accuracies, rewards, losses, q_values, 
                                       action_distributions, reward_stats, validation_scores, dataset_key):
        """Enhanced training progress visualization with publication-ready styling."""
        apply_publication_style()
        plt.ioff()  # Turn off interactive mode
        fig, axes = plt.subplots(2, 4, figsize=(32, 16))  # Single 2x4 layout for all plots
        episodes = range(1, len(accuracies) + 1)
        colors = COLOR_SCHEMES['training']
        
        # Row 1, Col 1: Training accuracy
        axes[0,0].plot(episodes, accuracies, color=colors['accuracy'], linewidth=1.5, alpha=0.8)
        axes[0,0].set_xlabel('Training Episode')
        axes[0,0].set_ylabel('Training Accuracy')
        axes[0,0].set_title('Training Accuracy Progression', fontweight='bold')
        if accuracies:
            y_min = max(0, min(accuracies) - 0.01)
            y_max = min(1.0, max(accuracies) + 0.01)
            axes[0,0].set_ylim([y_min, y_max])
        
        # Row 1, Col 2: Training rewards
        axes[0,1].plot(episodes, rewards, color=colors['reward'], linewidth=1.5, alpha=0.8)
        axes[0,1].set_xlabel('Training Episode')
        axes[0,1].set_ylabel('Normalized Reward')
        axes[0,1].set_title('Training Reward Evolution', fontweight='bold')
        
        # Row 1, Col 3: Training losses
        if len(losses) > 1:
            loss_episodes = episodes[1:]
            loss_values = losses[1:]
            axes[0,2].plot(loss_episodes, loss_values, color=colors['loss'], linewidth=1.5, alpha=0.8)
        axes[0,2].set_xlabel('Training Episode')
        axes[0,2].set_ylabel('Average Loss (log scale)')
        axes[0,2].set_title('Training Loss Convergence', fontweight='bold')
        axes[0,2].set_yscale('log')
        
        # Row 1, Col 4: Action distribution heatmap
        if action_distributions:
            action_matrix = np.array(action_distributions).T
            bin_size = 50
            n_episodes = len(action_distributions)
            n_bins = (n_episodes + bin_size - 1) // bin_size
            
            binned_matrix = np.zeros((action_matrix.shape[0], n_bins))
            bin_labels = []
            
            for i in range(n_bins):
                start_ep = i * bin_size
                end_ep = min((i + 1) * bin_size, n_episodes)
                if end_ep > start_ep:
                    binned_matrix[:, i] = np.mean(action_matrix[:, start_ep:end_ep], axis=1)
                    bin_labels.append(f'{start_ep+1}-{end_ep}')
            
            im = axes[0,3].imshow(binned_matrix, aspect='auto', cmap=COLOR_SCHEMES['heatmap'], 
                                origin='lower', interpolation='nearest')
            axes[0,3].set_xlabel('Episode Bins')
            axes[0,3].set_ylabel('Fusion Weight (α)')
            axes[0,3].set_title(f'Action Selection Evolution', fontweight='bold')
            
            alpha_values = getattr(self.fusion_agent, 'alpha_values', [0.0, 0.25, 0.5, 0.75, 1.0])
            alpha_ticks = range(0, len(alpha_values), max(1, len(alpha_values)//5))
            alpha_labels = [f'{alpha_values[i]:.2f}' for i in alpha_ticks if i < len(alpha_values)]
            axes[0,3].set_yticks(alpha_ticks)
            axes[0,3].set_yticklabels(alpha_labels)
            
            x_tick_interval = max(1, n_bins // 6)  # Fewer labels for smaller subplot
            x_ticks = range(0, n_bins, x_tick_interval)
            x_labels = [bin_labels[i] for i in x_ticks]
            axes[0,3].set_xticks(x_ticks)
            axes[0,3].set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
            
            cbar = plt.colorbar(im, ax=axes[0,3], shrink=0.8)
            cbar.set_label('Selection Frequency', fontsize=10)
            cbar.ax.tick_params(labelsize=9)
        
        # Row 2, Col 1: Exploration-Exploitation Balance
        if validation_scores and hasattr(self.fusion_agent, 'epsilon'):
            epsilon_values = []
            action_entropies = []
            
            for action_dist in action_distributions:
                action_probs = action_dist / (action_dist.sum() + 1e-8)
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
                action_entropies.append(entropy)
            
            initial_epsilon = 0.8
            epsilon_decay = 0.99
            for ep in range(len(episodes)):
                current_epsilon = max(0.15, initial_epsilon * (epsilon_decay ** (ep // 5)))
                epsilon_values.append(current_epsilon)
            
            axes[1,0].plot(episodes, epsilon_values, color='#d62728', linewidth=1.5, 
                          alpha=0.9, label='Exploration (ε)')
            axes[1,0].set_xlabel('Training Episode')
            axes[1,0].set_ylabel('Epsilon Value', color='#d62728')
            axes[1,0].tick_params(axis='y', labelcolor='#d62728')
            axes[1,0].set_ylim([0, 1])
            
            ax_twin = axes[1,0].twinx()
            window_size = min(25, len(action_entropies)//10)
            if window_size > 0:
                smoothed_entropy = np.convolve(action_entropies, np.ones(window_size)/window_size, mode='valid')
                entropy_episodes = episodes[window_size-1:]
                ax_twin.plot(entropy_episodes, smoothed_entropy, color='#ff7f0e', 
                           linewidth=1.5, alpha=0.9, label='Action Entropy')
            
            ax_twin.set_ylabel('Action Entropy (bits)', color='#ff7f0e')
            ax_twin.tick_params(axis='y', labelcolor='#ff7f0e')
            
            lines1, labels1 = axes[1,0].get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            axes[1,0].legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
            axes[1,0].set_title('Exploration-Exploitation Balance', fontweight='bold')
        
        # Store for fusion analysis plots
        self._current_fig = fig
        self._current_axes = axes
        
        plt.tight_layout()
        filename = f'images/complete_fusion_training_analysis_{dataset_key}'
        save_publication_figure(fig, filename + '.png')
        plt.close(fig)
        plt.ion()



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
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for fusion training. No GPU detected.")
        
    device = torch.device("cuda")
    print(f"✓ Using device: {device}")
    
    # Dataset configuration
    dataset_key = config_dict['root_folder']
    if dataset_key not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    
    root_folder = DATASET_PATHS[dataset_key]
    
    # Create directories
    for dir_name in ["images", "saved_models", "saved_models/fusion_checkpoints"]:
        os.makedirs(dir_name, exist_ok=True)
    
    # === Data Loading and Preprocessing with Caching ===
    print(f"\n=== Data Loading and Preprocessing ===")
    print(f"✓ Dataset: {dataset_key}, Path: {root_folder}")
    
    # Initialize cache manager
    cache_mgr = CacheManager(dataset_name=dataset_key)
    
    # Clear GPU memory for preprocessing
    torch.cuda.empty_cache()
    print(f"🧠 Starting preprocessing - GPU memory cleared")
    
    # Try loading from cache first
    print("💾 Checking for cached data...")
    id_mapping = cache_mgr.load_cache('id_mapping')
    dataset = cache_mgr.load_cache('raw_dataset')
    
    if id_mapping is None or dataset is None:
        print("🔄 Cache miss - performing full preprocessing...")
        
        # Timing diagnostics with progress
        print("🔄 Step 1/2: Building ID mapping...")
        id_mapping = build_id_mapping_from_normal(root_folder)
        
        # Cache ID mapping
        cache_mgr.save_cache(id_mapping, 'id_mapping')

        print("🔄 Step 2/2: Creating graph dataset...")
        
        start_time = time.time()
        dataset = graph_creation(root_folder, id_mapping=id_mapping, 
                               window_size=config_dict.get('window_size', 100))
        graph_creation_time = time.time() - start_time
        
        # Cache dataset
        cache_mgr.save_cache(dataset, 'raw_dataset', metadata={
            'num_graphs': len(dataset),
            'num_ids': len(id_mapping),
            'window_size': config_dict.get('window_size', 100)
        })
        
        print(f"✓ Dataset: {len(dataset)} graphs, {len(id_mapping)} CAN IDs")
        print(f"✓ Graph creation time: {graph_creation_time:.2f}s")
        
    else:
        print(f"🚀 Cache hit! Loaded {len(dataset)} graphs and {len(id_mapping)} IDs from cache")
        print(f"💾 Saved ~23 minutes of preprocessing time!")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train/test split
    train_size = int(config_dict.get('train_ratio', 0.8) * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    # Create GPU-optimized data loaders
    print("🔧 Creating GPU-optimized data loaders...")
    train_loader = create_optimized_data_loaders(train_dataset, None, None, device=str(device))
    test_loader = create_optimized_data_loaders(None, test_dataset, None, device=str(device))
    
    print(f"✓ Data split: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # === Initialize Fusion Pipeline ===
    pipeline = FusionTrainingPipeline(
        num_ids=len(id_mapping), 
        embedding_dim=8, 
        device=str(device)
    )
    
    # Store dataloader config for fusion training optimization
    resource_config = detect_gpu_capabilities_unified(str(device))
    pipeline._dataloader_config = resource_config
    
    # === Load Pre-trained Models ===
    autoencoder_path = f"saved_models1/autoencoder_best_{dataset_key}.pth"
    classifier_path = f"saved_models1/classifier_{dataset_key}.pth"

    try:
        pipeline.load_pretrained_models(autoencoder_path, classifier_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run osc_training_AD.py first to train the base models!")
        return
    
    # Use all available data - consistent batch sizes regardless of dataset size
    max_train_samples = len(train_dataset)  # Use all training data
    max_val_samples = len(test_dataset)     # Use all validation data
    
    print(f"📊 Using all available data:")
    print(f"   Training samples: {max_train_samples:,}")
    print(f"   Validation samples: {max_val_samples:,}")
    
    # Initialize fusion agent
    pipeline.initialize_fusion_agent()
    
    # Add fusion data caching
    pipeline.cache_mgr = cache_mgr  # Pass cache manager to pipeline
    
    pipeline.prepare_fusion_data(
        train_loader, 
        test_loader, 
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples
    )
    
    # Train the fusion agent
    training_results = pipeline.train_fusion_agent(
        episodes=config_dict.get('fusion_episodes', 200),  # Reasonable default
        dataset_key=dataset_key
    )
    
    # === Final Evaluation ===
    print(f"\n=== Final Evaluation and Comparison ===")
    evaluation_results = pipeline.evaluate_fusion_strategies(dataset_key)
    
    # === Save Final Model ===
    pipeline.save_fusion_agent("saved_models", "final", dataset_key)
    
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