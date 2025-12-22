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

# Publication-Ready Plotting Configuration (Scientific Paper Style)
PLOT_CONFIG = {
    # Font settings - clean, professional
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    
    # Figure settings - high DPI, clean background
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'none',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.facecolor': 'white',
    
    # Line and marker settings - thin, precise
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'patch.linewidth': 0.8,
    
    # Grid and axes - minimal, clean
    'axes.grid': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.axisbelow': True,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'axes.facecolor': 'white',
    
    # Ticks - clean, minimal
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    
    # Colors - scientific, muted palette
    'axes.prop_cycle': plt.cycler('color', [
        '#2E86AB',  # Blue
        '#A23B72',  # Magenta  
        '#F18F01',  # Orange
        '#C73E1D',  # Red
        '#592E83',  # Purple
        '#1B5F40',  # Green
        '#8B4513',  # Brown
        '#708090'   # Slate Gray
    ]),
    
    # LaTeX rendering
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',
    
    # Legend settings - clean, minimal
    'legend.frameon': False,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1
}

# Color schemes for different plot types (Scientific Paper Style)
COLOR_SCHEMES = {
    'training': {
        'accuracy': '#2E86AB',
        'reward': '#1B5F40', 
        'loss': '#C73E1D',
        'q_values': '#592E83'
    },
    'validation': {
        'primary': '#A23B72',
        'secondary': '#8B4513'
    },
    'fusion_analysis': {
        'normal': '#2E86AB',
        'attack': '#A23B72',
        'adaptive': '#F18F01'
    },
    'heatmap': 'RdYlBu_r',
    'contour': 'viridis'
}

def apply_publication_style():
    """Apply publication-ready matplotlib style."""
    plt.rcParams.update(PLOT_CONFIG)
    plt.style.use('default')  # Reset to default first
    for key, value in PLOT_CONFIG.items():
        if key in plt.rcParams:
            plt.rcParams[key] = value

def save_publication_figure(fig, filename, additional_formats=None):
    """Save figure in multiple publication-ready formats.
    
    Args:
        fig: matplotlib figure object
        filename: base filename with extension
        additional_formats: list of additional formats to save ['pdf', 'svg', 'eps'] or None for PNG only
    """
    base_path = filename.rsplit('.', 1)[0]
    
    # Always save PNG (default)
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    # Save additional formats only if specified
    if additional_formats:
        for fmt in additional_formats:
            try:
                fig.savefig(f"{base_path}.{fmt}", bbox_inches='tight', pad_inches=0.05)
            except Exception as e:
                print(f"Warning: Could not save {fmt} format: {e}")

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
        GPU-optimized computation of normalized anomaly scores for batch.
        
        Args:
            batch: Batch of graph data
            
        Returns:
            Tensor of anomaly scores [0, 1] for each graph
        """
        with torch.no_grad():
            # GPU-optimized forward pass with larger batch processing
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
        GPU-accelerated computation of GAT classification probabilities.
        
        Args:
            batch: Batch of graph data
            
        Returns:
            Tensor of GAT probabilities [0, 1] for each graph
        """
        with torch.no_grad():
            # GPU-optimized classifier forward pass
            logits = self.classifier(batch)
            probabilities = torch.sigmoid(logits.squeeze())
            return probabilities.cpu()

    def extract_fusion_data(self, data_loader: DataLoader, max_samples: int = None) -> Tuple[List, List, List]:
        """
        GPU-accelerated extraction of anomaly scores, GAT probabilities, and labels.
        
        Args:
            data_loader: DataLoader with graph data
            max_samples: Maximum number of samples to extract (None for all)
            
        Returns:
            Tuple of (anomaly_scores, gat_probabilities, labels)
        """
        print("Fusion Data Extraction...")
        
        anomaly_scores = []
        gat_probabilities = []
        labels = []
        
        samples_processed = 0
        
        # Direct batch processing for better GPU/CPU utilization
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="GPU-Accelerated Extraction", disable=False)):
            batch = batch.to(self.device, non_blocking=True)
            
            # Process batch immediately without accumulation
            batch_anomaly_scores = self.compute_anomaly_scores(batch)
            batch_gat_probs = self.compute_gat_probabilities(batch) 
            
            # Efficiently extract labels
            batch_labels = [graph.y.item() for graph in Batch.to_data_list(batch)]
            
            # Convert to lists and extend
            anomaly_scores.extend(batch_anomaly_scores.cpu().numpy().tolist())
            gat_probabilities.extend(batch_gat_probs.cpu().numpy().tolist())
            labels.extend(batch_labels)
            samples_processed += len(batch_labels)
            
            # Check sample limit
            if max_samples is not None and samples_processed >= max_samples:
                break
            
            # Memory cleanup every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
        
        print(f"✅ GPU-Accelerated Extraction Complete: {len(anomaly_scores)} samples")
        print(f"  Normal samples: {sum(1 for l in labels if l == 0)}")
        print(f"  Attack samples: {sum(1 for l in labels if l == 1)}")
        
        return anomaly_scores, gat_probabilities, labels
    
    def _process_batch_parallel(self, batch_list):
        """Process multiple batches efficiently."""
        if not batch_list:
            return []
        
        results = []
        with torch.no_grad():
            for batch in batch_list:
                anomaly_scores = self.compute_anomaly_scores(batch)
                gat_probs = self.compute_gat_probabilities(batch) 
                labels = batch.y.cpu().tolist()
                results.append((anomaly_scores.tolist(), gat_probs.tolist(), labels))
        return results


class FusionTrainingPipeline:
    """Complete pipeline for training the fusion agent."""
    
    def __init__(self, num_ids: int, embedding_dim: int = 8, device: str = 'cpu'):
        self.device = torch.device(device)
        self.num_ids = num_ids
        self.embedding_dim = embedding_dim
        self.gpu_info = self._detect_gpu_capabilities()
        
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
            print(f"  Optimized batch size: {self.gpu_info['optimal_batch_size']}")
    
    def _detect_gpu_capabilities(self):
        """Detect GPU capabilities and optimize parameters accordingly."""
        if not torch.cuda.is_available():
            return None
        
        gpu_props = torch.cuda.get_device_properties(self.device)
        memory_gb = gpu_props.total_memory / (1024**3)

        if memory_gb >= 30:  # A100 40GB/80GB
                optimal_batch_size = 16384  # Large batch for maximum throughput
                buffer_size = 100000  # Keep smaller buffer for speed
                training_steps = 4
        else:
            optimal_batch_size = 8192   # Large batch for good throughput
            buffer_size = 75000   # Keep smaller buffer for speed
            training_steps = 3
        
        return {
            'name': gpu_props.name,
            'memory_gb': memory_gb,
            'optimal_batch_size': optimal_batch_size,
            'buffer_size': buffer_size,
            'training_steps': training_steps,
            'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
        }

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
        
        # Store test data for evaluation (avoid redundant extraction)
        self.test_data = self.validation_data  # val_loader is actually test data in main()
        
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
                               batch_size: int = 2048, target_update_freq: int = 100,
                               config_dict: dict = None):
        """Initialize fusion agent."""
        state_dim = 4  # anomaly_score, gat_prob, disagreement, avg_confidence
        
        # GPU optimization
        if self.device.type == 'cuda' and self.gpu_info:
            batch_size = max(batch_size, self.gpu_info['optimal_batch_size'])
            buffer_size = max(buffer_size, self.gpu_info['buffer_size'])
        elif self.device.type == 'cuda':
            batch_size = max(batch_size, 4096)
            buffer_size = max(buffer_size, 500000)
            
        self.fusion_agent = EnhancedDQNFusionAgent(
            alpha_steps=alpha_steps, lr=lr, 
            gamma=config_dict.get('fusion_gamma', 0.95) if config_dict else 0.95, 
            epsilon=epsilon,
            epsilon_decay=config_dict.get('fusion_epsilon_decay', 0.995) if config_dict else 0.995,
            min_epsilon=config_dict.get('fusion_min_epsilon', 0.1) if config_dict else 0.1,
            buffer_size=buffer_size, batch_size=batch_size, target_update_freq=target_update_freq,
            device=str(self.device), state_dim=state_dim
        )
        
        print(f"✓ Fusion Agent initialized with {state_dim}D state space")

    def _get_curriculum_phase(self, episode: int, total_episodes: int) -> dict:
        """Return default sampling strategy (simplified)."""
        return {'phase': 'natural', 'high_disagreement_prob': 0.2, 
               'extreme_confidence_prob': 0.2, 'balanced_prob': 0.6}



    def _sample_by_curriculum(self, episode: int, total_episodes: int, episode_size: int) -> List:
        """Sample training data (simplified without curriculum learning)."""
        # Simple random sampling without curriculum complexity
        return random.sample(self.training_data, min(episode_size, len(self.training_data)))
    
    def _process_experience_batch_parallel(self, batch_data, num_workers):
        """Process experiences with optimized parallelization."""
        def process_experience(data):
            anomaly_score, gat_prob, true_label = data
            current_state = self.fusion_agent.normalize_state(anomaly_score, gat_prob)
            alpha, action_idx, _ = self.fusion_agent.select_action(anomaly_score, gat_prob, training=True)
            raw_reward = self.fusion_agent.compute_fusion_reward(
                1 if (1 - alpha) * anomaly_score + alpha * gat_prob > 0.5 else 0,
                true_label, anomaly_score, gat_prob, alpha
            )
            normalized_reward = np.clip(raw_reward * 0.5, -1.0, 1.0)
            return (anomaly_score, gat_prob, true_label, current_state, alpha, action_idx, raw_reward, normalized_reward)
        
        # Use parallel processing only for larger batches to avoid overhead
        if len(batch_data) > 64:  # Increased threshold
            with ThreadPoolExecutor(max_workers=min(num_workers, 16)) as executor:  # Limit max workers
                return list(executor.map(process_experience, batch_data))
        return [process_experience(data) for data in batch_data]

    def train_fusion_agent(self, episodes: int = 50, validation_interval: int = 25,  # Less frequent validation
                          early_stopping_patience: int = 50, save_interval: int = 50,  # Much longer patience
                          dataset_key: str = 'default', config_dict: dict = None):
        """
        Enhanced training with better instrumentation and learning dynamics.
        """
        torch.set_default_dtype(torch.float32)
        
        # Optimize PyTorch for maximum CPU utilization
        torch.set_num_threads(cpu_count())  # Use all CPU cores for tensor operations
        torch.set_num_interop_threads(cpu_count())  # Parallel operations between ops
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Optimize CUDA kernels
            torch.cuda.set_sync_debug_mode(0)  # Disable synchronous debugging
        
        print(f"\n=== Training Fusion Agent ===")
        print(f"Episodes: {episodes}, Validation every {validation_interval} episodes")
        print(f"CPU Optimization: {cpu_count()} threads, GPU async: {torch.cuda.is_available()}")
        
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
        base_episode_size = config_dict.get('episode_sample_size', 10000) if config_dict else 10000
        
        # Create save directory
        checkpoint_dir = f"saved_models/fusion_checkpoints/{dataset_key}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for episode in range(episodes):
            # Determine episode size (reduced for faster training)
            base_episode_size = config_dict.get('episode_sample_size', 2000) if config_dict else 2000
            current_episode_size = min(base_episode_size, len(self.training_data))
            
            # Use simplified sampling
            shuffled_data = self._sample_by_curriculum(episode, episodes, current_episode_size)
            
            # Print progress info
            if episode % 10 == 0:
                total_progress = episode / episodes
                print(f"    Episode {episode:4d} - Total Progress: {total_progress:.1%}")

            
            episode_reward = 0
            episode_correct = 0
            episode_samples = 0
            episode_loss_sum = 0
            episode_loss_count = 0
            episode_q_sum = 0
            episode_q_count = 0  # Track Q-value sample count separately
            episode_action_counts = np.zeros(len(getattr(self.fusion_agent, 'alpha_values', [0.0, 0.25, 0.5, 0.75, 1.0])))
            episode_raw_rewards = []
            

            
            # GPU-optimized training for maximum throughput
            if self.gpu_info:
                training_step_interval = 32  # Balance between GPU utilization and speed
                gpu_training_steps = 8   # Reasonable training steps for GPU efficiency
            else:
                training_step_interval = config_dict.get('training_step_interval', 64 if self.device.type == 'cuda' else 32)
                gpu_training_steps = config_dict.get('gpu_training_steps', 4 if self.device.type == 'cuda' else 2)
            
            experience_batch = []  # Collect experiences for batch processing
            state_cache = []      # Cache states for GPU batch processing
            continuous_training = False  # Enable continuous training mode after warmup
            
            # CPU-optimized parallel processing of samples
            cpu_batch_size = 256  # Process samples in CPU batches
            num_cpu_workers = min(8, cpu_count())  # Use available CPU cores
            
            # Process samples in parallel batches for better CPU utilization
            for batch_start in range(0, len(shuffled_data), cpu_batch_size):
                batch_end = min(batch_start + cpu_batch_size, len(shuffled_data))
                batch_data = shuffled_data[batch_start:batch_end]
                
                # Process batch in parallel (with validation)
                if len(batch_data) > 0:
                    batch_experiences = self._process_experience_batch_parallel(batch_data, num_cpu_workers)
                    
                    # Validate batch_experiences structure
                    if not batch_experiences:
                        print(f"Warning: Empty batch_experiences for batch {batch_start}-{batch_end}")
                        continue
                else:
                    batch_experiences = []
                
                # Add experiences to agent
                for exp_idx, exp_data in enumerate(batch_experiences):
                    # Safely unpack experience data with error handling
                    try:
                        anomaly_score, gat_prob, true_label, current_state, alpha, action_idx, raw_reward, normalized_reward = exp_data
                        
                        # Ensure scalar values (convert arrays to scalars if needed)
                        if hasattr(anomaly_score, 'item'):
                            anomaly_score = anomaly_score.item()
                        if hasattr(gat_prob, 'item'):
                            gat_prob = gat_prob.item()
                        if hasattr(true_label, 'item'):
                            true_label = true_label.item()
                        if hasattr(alpha, 'item'):
                            alpha = alpha.item()
                            
                    except (ValueError, TypeError) as e:
                        print(f"Error unpacking experience data: {e}")
                        print(f"Experience data shape/type: {[type(x) for x in exp_data]}")
                        continue
                    
                    # Get next state for experience
                    next_idx = batch_start + exp_idx + 1
                    if next_idx < len(shuffled_data):
                        next_anomaly, next_gat, _ = shuffled_data[next_idx]
                        next_state = self.fusion_agent.normalize_state(next_anomaly, next_gat)
                        done = False
                    else:
                        next_state = current_state
                        done = True
                    
                    # Store experience and update counters
                    experience_batch.append({
                        'state': current_state,
                        'action': action_idx, 
                        'reward': normalized_reward,
                        'next_state': next_state,
                        'done': done
                    })
                    
                    # Update episode statistics
                    episode_action_counts[action_idx] += 1
                    episode_reward += normalized_reward
                    episode_correct += (1 if (1 - alpha) * anomaly_score + alpha * gat_prob > 0.5 else 0) == true_label
                    episode_samples += 1
                    episode_raw_rewards.append(raw_reward)
            
            # Convert to old loop variable for compatibility
            i = episode_samples - 1
            
            # All processing now handled by parallel batch processing above
            
            # Final GPU utilization boost
            if experience_batch:
                # Store remaining experiences
                for exp in experience_batch:
                    self.fusion_agent.store_experience(
                        exp['state'], exp['action'], exp['reward'], exp['next_state'], exp['done']
                    )
                
                # Aggressive final training to fully utilize buffer
                if len(self.fusion_agent.replay_buffer) >= self.fusion_agent.batch_size:
                    # Utilize full training potential
                    max_final_steps = gpu_training_steps * 3  # Remove conservative cap
                    batch_loss = 0
                    successful_steps = 0
                    
                    # Train until buffer is optimally utilized
                    for step in range(max_final_steps):
                        loss = self.fusion_agent.train_step()
                        if loss is not None:
                            batch_loss += loss
                            successful_steps += 1
                        
                        # Stop when buffer becomes too small for efficient training
                        if len(self.fusion_agent.replay_buffer) < self.fusion_agent.batch_size * 3:
                            break
                    
                    if successful_steps > 0:
                        episode_loss_sum += batch_loss / successful_steps
                        episode_loss_count += 1
                        
                    # Final target network sync
                    self.fusion_agent.update_target_network()
            
            # End-of-episode minimal training burst
            if len(self.fusion_agent.replay_buffer) >= self.fusion_agent.batch_size * 20:  # Higher threshold
                # Minimal additional training to prevent timeout
                bonus_training_steps = gpu_training_steps  # No multiplication
                for _ in range(bonus_training_steps):
                    loss = self.fusion_agent.train_step()
                    if loss is not None:
                        episode_loss_sum += loss
                        episode_loss_count += 1
            
            # End episode
            if hasattr(self.fusion_agent, 'end_episode'):
                self.fusion_agent.end_episode()
            
            # Calculate episode statistics with action diversity bonus
            episode_accuracy = episode_correct / episode_samples if episode_samples > 0 else 0
            avg_episode_reward = episode_reward / episode_samples if episode_samples > 0 else 0
            avg_episode_loss = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0
            avg_q_value = episode_q_sum / episode_q_count if episode_q_count > 0 else 0.0  # Correct Q-value averaging
            
            # Calculate action diversity (entropy) for this episode
            action_probs = episode_action_counts / max(episode_samples, 1)
            action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
            max_entropy = np.log(len(episode_action_counts))
            diversity_ratio = action_entropy / max_entropy if max_entropy > 0 else 0
            
            # Store episode stats
            episode_rewards.append(avg_episode_reward)
            episode_accuracies.append(episode_accuracy)
            episode_losses.append(avg_episode_loss)
            episode_q_values.append(avg_q_value)
            # Safe division to avoid division by zero
            action_distributions.append(episode_action_counts / max(episode_samples, 1))
            
            # Reward statistics (with safety checks for empty lists)
            if episode_raw_rewards:
                reward_stats.append({
                    'raw_mean': np.mean(episode_raw_rewards),
                    'raw_std': np.std(episode_raw_rewards),
                    'raw_min': np.min(episode_raw_rewards),
                    'raw_max': np.max(episode_raw_rewards)
                })
            else:
                reward_stats.append({
                    'raw_mean': 0.0,
                    'raw_std': 0.0,
                    'raw_min': 0.0,
                    'raw_max': 0.0
                })
            
            # Adaptive learning rate decay for stability in later episodes
            if episode > 200 and episode_accuracy > 0.995:  # High performance phase
                # Reduce learning rate for fine-tuning stability
                if hasattr(self.fusion_agent, 'optimizer'):
                    for param_group in self.fusion_agent.optimizer.param_groups:
                        if param_group['lr'] > 0.0001:  # Don't go too low
                            param_group['lr'] *= config_dict.get('lr_decay_factor', 0.98) if config_dict else 0.98
            
            # More gradual exploration decay for better learning
            if episode % 5 == 0 and hasattr(self.fusion_agent, 'decay_epsilon'):  # Every 5 episodes for stability
                self.fusion_agent.decay_epsilon()
            
            # episode logging
            if episode % 50 == 0 or episode < 3:
                print(f"\n📊 Episode {episode + 1}/{episodes} Stats:")
                print(f"  Accuracy: {episode_accuracy:.4f}")
                print(f"  Normalized Reward: {avg_episode_reward:.4f}")
                print(f"  Raw Reward Range: [{reward_stats[-1]['raw_min']:.2f}, {reward_stats[-1]['raw_max']:.2f}]")
                print(f"  Avg Loss: {avg_episode_loss:.6f}")
                print(f"  Avg Q-Value: {avg_q_value:.6f} (Sum: {episode_q_sum:.3f}, Q-Samples: {episode_q_count})")
                print(f"  Action Diversity: {diversity_ratio:.3f} (Entropy: {action_entropy:.3f}/{max_entropy:.3f})")
                print(f"  Epsilon: {getattr(self.fusion_agent, 'epsilon', 0.1):.4f}")
                print(f"  Buffer Size: {len(self.fusion_agent.replay_buffer)}")
                
                # Early convergence detection for stability
                if episode > 250 and episode_accuracy > 0.9975 and avg_episode_loss < 0.001:
                    print(f"\n🎯 Early convergence detected at episode {episode+1}")
                    print(f"  High accuracy: {episode_accuracy:.6f}")
                    print(f"  Low loss: {avg_episode_loss:.6f}")
                    # Reduce training intensity for stability
                    if hasattr(self.fusion_agent, 'epsilon') and self.fusion_agent.epsilon > 0.05:
                        decay_factor = config_dict.get('epsilon_fast_decay', 0.95) if config_dict else 0.95
                        self.fusion_agent.epsilon *= decay_factor
                
                # Action distribution analysis (with safety checks)
                if len(episode_action_counts) > 0:
                    # Take min of 3 or available actions to avoid index errors
                    num_actions_to_show = min(3, len(episode_action_counts))
                    most_used_actions = np.argsort(episode_action_counts)[-num_actions_to_show:][::-1]
                    alpha_values = getattr(self.fusion_agent, 'alpha_values', [0.0, 0.25, 0.5, 0.75, 1.0])
                    print(f"  Top Actions: {[f'α={alpha_values[i] if i < len(alpha_values) else 0.5:.2f}({episode_action_counts[i]:.0f})' for i in most_used_actions]}")
                else:
                    print("  Top Actions: No actions recorded")
                
                # Q-value analysis for sample states
                self._analyze_q_values_for_sample_states()
            
            # Validation and logging (optimized for CPU utilization)
            if (episode + 1) % validation_interval == 0 and hasattr(self.fusion_agent, 'validate_agent'):
                # Use smaller validation sample for faster processing
                val_samples = 500 if episode < episodes // 2 else 1000  # Increase validation near end
                val_results = self.fusion_agent.validate_agent(self.validation_data, num_samples=val_samples)
                validation_scores.append(val_results)
                
                print(f"\n🎯 Validation Results (Episode {episode + 1}):")
                print(f"  Validation Accuracy: {val_results['accuracy']:.4f}")
                print(f"  Validation Reward: {val_results['avg_reward']:.4f}")
                print(f"  Avg Alpha: {val_results['avg_alpha']:.4f} ± {val_results['alpha_std']:.4f}")
                
                # Aggressive early stopping - count any improvement
                current_val_score = val_results['accuracy']
                
                # Any improvement counts
                if current_val_score > best_validation_score:
                    best_validation_score = current_val_score
                    patience_counter = 0
                    self.save_fusion_agent("saved_models/fusion_checkpoints", "best", dataset_key)
                    print(f"  🏆 New best validation score: {current_val_score:.6f}!")
                elif current_val_score < best_validation_score - 0.005:  # Only count meaningful degradation
                    patience_counter += 1
                    print(f"  ⚠️  Validation declined: {current_val_score:.6f} < {best_validation_score:.6f} (patience: {patience_counter}/{early_stopping_patience})")
                else:
                    # Small changes don't affect patience - model is still learning
                    print(f"  ➖ Validation stable: {current_val_score:.6f} (patience unchanged: {patience_counter}/{early_stopping_patience})")
                
                if patience_counter >= early_stopping_patience:
                    print(f"\n🛑 Early stopping triggered after {patience_counter} validation cycles")
                    print(f"   Best validation score achieved: {best_validation_score:.6f}")
                    print(f"   Current validation score: {current_val_score:.6f}")
                    break
            
            # Periodic checkpoints
            if (episode + 1) % save_interval == 0:
                self.save_fusion_agent("saved_models/fusion_checkpoints", f"episode_{episode+1}", dataset_key)
        
        print(f"\n✓ Fusion agent training completed!")
        print(f"Best validation accuracy: {best_validation_score:.4f}")
        
        # Enhanced analysis plots
        self._plot_enhanced_training_progress(
            episode_accuracies, episode_rewards, episode_losses, 
            episode_q_values, action_distributions, reward_stats, validation_scores,
            dataset_key
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
                    device = getattr(self.fusion_agent, 'device', self.device)
                    q_network = getattr(self.fusion_agent, 'q_network', None)
                    alpha_values = getattr(self.fusion_agent, 'alpha_values', [0.0, 0.25, 0.5, 0.75, 1.0])
                    
                    if q_network is not None:
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        q_values = q_network(state_tensor).squeeze()
                        best_action = torch.argmax(q_values).item()
                        best_alpha = alpha_values[best_action] if best_action < len(alpha_values) else 0.5
                        max_q = q_values[best_action].item()
                    else:
                        best_alpha = 0.5  # Fallback
                        max_q = 0.0
                
                print(f"    State{i+1}: VGAE={anomaly_score:.3f}, GAT={gat_prob:.3f}, Label={true_label} "
                      f"→ Best α={best_alpha:.2f} (Q={max_q:.3f})")

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

def calculate_dynamic_resources(dataset_size: int, device: str = 'cuda'):
    """Dynamically calculate optimal resource allocation based on dataset size and available hardware."""
    
    # Get system resources
    cpu_count_available = cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Get GPU info if available
    gpu_memory_gb = 0
    cuda_available = torch.cuda.is_available() and 'cuda' in device
    if cuda_available:
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            cuda_available = False
    
    print(f"🔍 System Resources: {cpu_count_available} CPUs, {memory_gb:.1f}GB RAM, GPU: {gpu_memory_gb:.1f}GB")
    print(f"📊 Dataset size: {dataset_size:,} samples")
    
    # Estimate memory per sample (rough heuristic based on graph data)
    # Typical CAN graph: ~50-200 nodes, ~100-500 edges, features = ~1-10KB per graph
    memory_per_sample_mb = 0.01  # 10KB per sample estimate
    
    # Calculate optimal batch size based on available memory
    if cuda_available:
        # Use 95% of GPU memory for maximum throughput
        available_gpu_memory = gpu_memory_gb * 0.95
        # GPU can handle larger batches due to parallel processing
        max_batch_from_gpu = int(available_gpu_memory * 1024 / (memory_per_sample_mb * 3))  # Reduced overhead factor
        
        # Remove artificial caps - let GPU memory be the limiting factor
        target_batch = max_batch_from_gpu
        target_workers = max(16, min(cpu_count_available - 2, int(cpu_count_available * 0.9)))  # Use 90% of CPUs
        prefetch_factor = min(8, max(2, target_workers // 8))
            
        config = {
            'batch_size': target_batch,
            'num_workers': target_workers,
            'prefetch_factor': prefetch_factor,
            'pin_memory': True,
            'persistent_workers': True
        }
    else:
        # CPU mode - more aggressive memory usage
        available_memory = memory_gb * 0.8  # Only reserve 20% for system
        max_batch_from_memory = int(available_memory * 1024 / memory_per_sample_mb)
        
        target_batch = max_batch_from_memory  # Remove artificial 2048 cap
        # Use most CPUs but leave some for system
        target_workers = max(8, min(cpu_count_available - 1, int(cpu_count_available * 0.85)))
            
        config = {
            'batch_size': target_batch,
            'num_workers': target_workers, 
            'prefetch_factor': max(2, target_workers // 4),
            'pin_memory': False,
            'persistent_workers': False
        }
    
    print(f"🎯 Calculated config: batch={config['batch_size']}, workers={config['num_workers']}, "
          f"prefetch={config['prefetch_factor']}, cuda={cuda_available}")
    
    return config, cuda_available

def create_optimized_data_loaders(train_subset=None, test_dataset=None, full_train_dataset=None, 
                                 batch_size: int = 1024, device: str = 'cuda'):
    """Create optimized data loaders with dynamic resource allocation."""
    # Find dataset to get size for dynamic allocation
    dataset = next((d for d in [train_subset, test_dataset, full_train_dataset] if d is not None), None)
    if dataset is None:
        raise ValueError("No valid dataset provided to create_optimized_data_loaders")
    
    dataset_size = len(dataset)
    
    # Get dynamic resource allocation
    config, cuda_available = calculate_dynamic_resources(dataset_size, device)
    
    # Debug output to understand resource allocation
    print(f"🔍 Dataset analysis: size={dataset_size:,}")
    print(f"🎯 Calculated resources: workers={config['num_workers']}, batch={config['batch_size']}")
    print(f"📊 CPU info: available={cpu_count()}, using={config['num_workers']} ({config['num_workers']/cpu_count()*100:.1f}%)")
    
    # Override batch_size if explicitly provided and reasonable
    if batch_size != 1024:  # Non-default batch_size provided
        config['batch_size'] = min(batch_size, config['batch_size'])  # Don't exceed calculated max
    
    print(f"✓ Dynamic DataLoader: batch_size={config['batch_size']}, workers={config['num_workers']}")
    print(f"✓ DataLoader config: pin_memory={config['pin_memory']}, prefetch_factor={config['prefetch_factor']}, persistent_workers={config['persistent_workers']}")
    print(f"✓ CUDA available: {cuda_available}, Device: {device}")
    
    # Clear GPU cache before creating DataLoader for large datasets
    if cuda_available:
        torch.cuda.empty_cache()
        print(f"✓ GPU memory cleared")

    # Find the first non-None dataset and create loader for it
    datasets = [train_subset, test_dataset, full_train_dataset]
    shuffles = [True, False, True]
    
    for dataset, shuffle in zip(datasets, shuffles):
        if dataset is not None:
            try:
                print(f"🔧 Attempting DataLoader creation with {config['num_workers']} workers...")
                loader = DataLoader(
                    dataset,
                    batch_size=config['batch_size'],
                    shuffle=shuffle,
                    pin_memory=config['pin_memory'],
                    num_workers=config['num_workers'],
                    persistent_workers=config['persistent_workers'],
                    prefetch_factor=config['prefetch_factor']
                )
                # Test the loader to make sure it actually works
                print(f"✅ Testing DataLoader with {config['num_workers']} workers...")
                test_batch = next(iter(loader))
                print(f"✅ DataLoader verified successfully with {config['num_workers']} workers!")
                return loader
            except Exception as e:
                import traceback
                print(f"❌ DataLoader creation failed with {config['num_workers']} workers")
                print(f"❌ Error type: {type(e).__name__}")
                print(f"❌ Error message: {str(e)}")
                print(f"❌ Full traceback:")
                print(traceback.format_exc())
                # Aggressive fallback strategy - only reduce if absolutely necessary
                fallback_attempts = [
                    max(24, int(config['num_workers'] * 0.8)),  # Try 80% first
                    max(16, int(config['num_workers'] * 0.6)),  # Then 60%
                    max(12, config['num_workers'] // 2),        # Then 50%
                    8  # Final fallback
                ]
                
                for attempt, reduced_workers in enumerate(fallback_attempts, 1):
                    try:
                        print(f"🔄 Fallback attempt {attempt}: trying {reduced_workers} workers...")
                        loader = DataLoader(
                            dataset,
                            batch_size=config['batch_size'],
                            shuffle=shuffle,
                            pin_memory=config['pin_memory'],
                            num_workers=reduced_workers,
                            persistent_workers=config['persistent_workers'],
                            prefetch_factor=config['prefetch_factor']
                        )
                        # Test the fallback loader
                        test_batch = next(iter(loader))
                        print(f"✅ Fallback successful with {reduced_workers} workers")
                        return loader
                    except Exception as fallback_e:
                        print(f"❌ Fallback attempt {attempt} failed with {reduced_workers} workers: {fallback_e}")
                        continue
                
                # If all fallbacks fail, there's a deeper issue
                raise RuntimeError(f"All DataLoader configurations failed for dataset of size {dataset_size}")
    
    raise ValueError("No valid dataset provided to create_optimized_data_loaders")

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
    print(f"✓ Dataset: {dataset_key}, Path: {root_folder}")
    
    # Timing diagnostics
    io_start_time = time.time()
    id_mapping = build_id_mapping_from_normal(root_folder)
    io_mapping_time = time.time() - io_start_time
    print(f"✓ ID mapping built in {io_mapping_time:.2f}s")

    start_time = time.time()
    dataset = graph_creation(root_folder, id_mapping=id_mapping, 
                           window_size=config_dict.get('window_size', 100))
    preprocessing_time = time.time() - start_time
    
    print(f"✓ Dataset: {len(dataset)} graphs, {len(id_mapping)} CAN IDs")
    print(f"✓ Preprocessing time: {preprocessing_time:.2f}s")
    
    # Configuration (optimized for GPU utilization and speed)
    TRAIN_RATIO = config_dict.get('train_ratio', 0.8)
    BATCH_SIZE = config_dict.get('batch_size', 1024)
    FUSION_EPISODES = config_dict.get('fusion_episodes', 1000)
    ALPHA_STEPS = 21
    FUSION_LR = 0.0005  # Much lower for stability
    FUSION_EPSILON = 0.8  # Higher exploration for better Q-learning
    # Dynamic parameters - will be set by GPU detection and resource calculation
    BUFFER_SIZE = None  # Will be calculated dynamically
    FUSION_BATCH_SIZE = None  # Will be calculated dynamically  
    TARGET_UPDATE_FREQ = None  # Will be calculated dynamically
    
    # Enhanced exploration parameters
    FUSION_EPSILON = 0.9  # Higher initial exploration (was 0.8)
    FUSION_EPSILON_DECAY = 0.995  # Slower decay for longer exploration (was 0.99)
    FUSION_MIN_EPSILON = 0.2  # Higher minimum exploration (was 0.15)

    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train/test split
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    # Create GPU-optimized data loaders
    print("🔧 Creating GPU-optimized data loaders...")
    train_loader = create_optimized_data_loaders(train_dataset, None, None, BATCH_SIZE, str(device))
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
    
    # Dynamic sampling based on dataset size and available resources
    def calculate_optimal_sampling(dataset_size: int, device_type: str) -> Dict[str, int]:
        """Calculate optimal train/val sampling based on dataset characteristics."""
        # Dynamic ratios based purely on dataset size and compute power
        if dataset_size < 50000:  # Small datasets - use most data
            train_ratio = 0.9
            val_ratio = 0.3
        elif dataset_size < 200000:  # Medium datasets
            train_ratio = 0.7
            val_ratio = 0.25
        else:  # Large datasets
            train_ratio = 0.5
            val_ratio = 0.15
            
        # Scale up for better hardware
        if device_type == 'cuda':
            train_ratio *= 1.5  # Use more data on GPU
            val_ratio *= 1.2
            
        max_train = int(dataset_size * train_ratio)
        max_val = int(dataset_size * val_ratio)
        
        # Remove artificial caps - let hardware determine limits
        max_train = max(10000, max_train)  # Only minimum bound
        max_val = max(2000, max_val)       # Only minimum bound
        
        return {'max_train': max_train, 'max_val': max_val}
    
    total_samples = len(train_dataset) + len(test_dataset)
    sampling_config = calculate_optimal_sampling(total_samples, device.type)
    
    print(f"🎯 Dynamic sampling for {total_samples:,} total samples:")
    print(f"   Training samples: {sampling_config['max_train']:,}")
    print(f"   Validation samples: {sampling_config['max_val']:,}")
    
    # GPU memory optimization for extraction phase
    if torch.cuda.is_available():
        print("🚀 Pre-allocating GPU memory for accelerated extraction...")
        torch.cuda.empty_cache()  # Clear cache before large extraction
        torch.cuda.set_per_process_memory_fraction(0.99)  # Use 99% of GPU memory
    
    pipeline.prepare_fusion_data(
        train_loader, 
        test_loader, 
        max_train_samples=sampling_config['max_train'],
        max_val_samples=sampling_config['max_val']
    )
    
    
    # Initialize fusion agent with stability-focused config
    pipeline.initialize_fusion_agent(
        alpha_steps=config_dict.get('alpha_steps', ALPHA_STEPS),           # Use stable config
        lr=config_dict.get('fusion_lr', FUSION_LR),                       # Use stable learning rate
        epsilon=config_dict.get('fusion_epsilon', FUSION_EPSILON),        # Use enhanced exploration
        buffer_size=config_dict.get('fusion_buffer_size', BUFFER_SIZE), # Use optimized buffer size
        batch_size=config_dict.get('fusion_batch_size', FUSION_BATCH_SIZE), # Use optimized batch size
        target_update_freq=config_dict.get('fusion_target_update', TARGET_UPDATE_FREQ), # Use optimized update freq
        config_dict={**config_dict, 'fusion_epsilon_decay': FUSION_EPSILON_DECAY, 'fusion_min_epsilon': FUSION_MIN_EPSILON}
    )
    
    # Train the fusion agent with patience for continued learning
    training_results = pipeline.train_fusion_agent(
        episodes=config_dict.get('fusion_episodes', FUSION_EPISODES),  # Use faster default
        validation_interval=25,   # Less frequent validation for speed
        early_stopping_patience=75,   # Much longer patience - continue learning!
        save_interval=50,         # Less frequent checkpoints for speed
        dataset_key=dataset_key,
        config_dict=config_dict  # Pass config_dict here
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