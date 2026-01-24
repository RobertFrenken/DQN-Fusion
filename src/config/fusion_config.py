"""
Fusion Training Configuration for Hydra-Zen

Contains dataset paths, fusion weights, and Hydra-Zen dataclass configs
for DQN-based multi-model fusion training.

Note: DATASET_PATHS is now centralized in src.paths.DATASET_PATHS
This module re-exports it for backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from src.config.hydra_zen_configs import BaseTrainingConfig
from src.paths import DATASET_PATHS  # Unified dataset paths

# Fusion weights for composite anomaly scoring
FUSION_WEIGHTS = {
    'node_reconstruction': 0.4,
    'neighborhood_prediction': 0.35,
    'can_id_prediction': 0.25
}


# ============================================================================
# Hydra-Zen Fusion Configuration Classes
# ============================================================================

@dataclass
class FusionAgentConfig:
    """DQN Fusion Agent configuration."""
    # Action space
    alpha_steps: int = 21  # Number of discrete fusion weights [0.0, 0.05, ..., 1.0]
    
    # DQN hyperparameters
    fusion_lr: float = 0.001  # Learning rate for Q-network
    gamma: float = 0.9  # Discount factor
    
    # Exploration
    fusion_epsilon: float = 0.9  # Initial exploration rate
    fusion_epsilon_decay: float = 0.995  # Decay per episode
    fusion_min_epsilon: float = 0.2  # Minimum exploration rate
    
    # Experience replay
    fusion_buffer_size: int = 100000  # Replay buffer size
    fusion_batch_size: int = 256  # Mini-batch size for Q-learning
    
    # Target network
    target_update_freq: int = 100  # Update target every N steps
    
    # Network architecture
    hidden_dim: int = 64  # Hidden layer size
    
    # State normalization
    normalize_anomaly_score: bool = True
    normalize_gat_prob: bool = True


@dataclass
class FusionDataConfig:
    """Configuration for fusion data extraction and caching."""
    # Data usage
    max_train_samples: Optional[int] = None  # Use all if None
    max_val_samples: Optional[int] = None
    
    # Caching
    use_cache: bool = True
    cache_dir: str = "cache/fusion"
    cache_predictions: bool = True  # Cache GAT/VGAE predictions
    
    # Batch processing
    extraction_batch_size: int = 1024  # Batch size for extracting predictions
    num_extraction_workers: int = 8  # Workers for data loading


@dataclass
class FusionTrainingConfig(BaseTrainingConfig):
    """
    Complete fusion training configuration.
    
    Combines base training config with fusion-specific settings.
    """
    mode: str = "fusion"
    description: str = "DQN-based multi-model fusion training"
    
    # Training parameters
    max_epochs: int = 50  # Episodes to train fusion agent
    batch_size: int = 256  # Minibatch size for Q-learning
    learning_rate: float = 0.001  # For Q-network optimizer
    
    # Fusion agent
    fusion_agent: FusionAgentConfig = field(default_factory=FusionAgentConfig)
    
    # Data extraction and caching
    fusion_data: FusionDataConfig = field(default_factory=FusionDataConfig)
    
    # Early stopping
    early_stopping_patience: int = 20
    monitor_metric: str = "val_accuracy"
    monitor_mode: str = "max"
    
    # Logging and checkpointing
    log_every_n_steps: int = 50
    save_top_k: int = 3
    
    # Validation
    val_check_interval: float = 1.0  # Check every epoch
    limit_val_batches: float = 1.0  # Use all validation data
    
    # Hardware
    precision: str = "32-true"
    accelerator: str = "auto"
    devices: str = "auto"


@dataclass
class FusionEvaluationConfig:
    """Configuration for fusion evaluation and comparison."""
    # Evaluation modes
    evaluate_fixed_alphas: bool = True
    fixed_alpha_values: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Comparison methods
    compare_with_vgae_only: bool = True
    compare_with_gat_only: bool = True
    compare_with_learned_policy: bool = True
    
    # Metrics
    compute_confusion_matrix: bool = True
    compute_classification_report: bool = True
    
    # Plotting
    plot_policy_heatmap: bool = True
    plot_fusion_weights_distribution: bool = True
    plot_training_curves: bool = True