"""
Hydra-Zen Configuration System for CAN-Graph

This module replaces the YAML configuration files with type-safe Python configurations
using hydra-zen. Benefits:
- Type safety and IDE support
- Programmatic config generation
- Reduced file clutter
- Better validation
- Dynamic configuration composition
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from hydra_zen import make_config, store, zen
import torch


# ============================================================================
# Base Configuration Classes
# ============================================================================

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9  # For SGD


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    use_scheduler: bool = False
    scheduler_type: str = "cosine"
    params: Dict[str, Any] = field(default_factory=lambda: {"T_max": 100})


@dataclass
class MemoryOptimizationConfig:
    """Memory optimization settings."""
    use_teacher_cache: bool = True
    clear_cache_every_n_steps: int = 100
    offload_teacher_to_cpu: bool = False
    gradient_checkpointing: bool = False


# ============================================================================
# Model Configurations
# ============================================================================

@dataclass
class GATConfig:
    """Graph Attention Network configuration (Teacher Model)."""
    type: str = "gat"
    input_dim: int = 11
    hidden_channels: int = 64
    output_dim: int = 2           # Binary classification (normal/attack)
    num_layers: int = 5           # Matches paper: 5 GAT layers
    heads: int = 8                # Matches paper: 8 attention heads
    dropout: float = 0.2
    num_fc_layers: int = 3
    embedding_dim: int = 32       # Matches paper: 32 dimensions
    use_jumping_knowledge: bool = True
    jk_mode: str = "cat"
    use_residual: bool = True
    use_batch_norm: bool = False
    activation: str = "relu"


@dataclass
class StudentGATConfig:
    """Student GAT configuration for on-board deployment (55K parameters)."""
    type: str = "gat_student"
    input_dim: int = 11
    hidden_channels: int = 32     # Smaller for student
    output_dim: int = 2           # Binary classification
    num_layers: int = 2           # Matches paper: 2 GAT layers
    heads: int = 4                # Matches paper: 4 attention heads
    dropout: float = 0.1          # Lower dropout for student
    num_fc_layers: int = 2        # Simpler classification head
    embedding_dim: int = 8        # Matches paper: 8 dimensions
    use_jumping_knowledge: bool = False  # Simpler architecture
    jk_mode: str = "max"
    use_residual: bool = False    # Simpler for deployment
    use_batch_norm: bool = False
    activation: str = "relu"


@dataclass
class VGAEConfig:
    """Teacher VGAE configuration (~1.74M parameters, matching teacher_student.yaml)."""
    type: str = "vgae"
    input_dim: int = 11           # Same input as student
    node_embedding_dim: int = 256 # Richer embedding (from teacher_student.yaml)
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 96, 48])  # Multi-layer with proper compression
    latent_dim: int = 48          # Larger latent space than student (from teacher_student.yaml)
    output_dim: int = 11          # Reconstruct input features
    
    # Architecture settings (from teacher_student.yaml)
    num_layers: int = 3           # Deep architecture for rich representations  
    attention_heads: int = 8      # 8 heads per layer (multi-head attention for complex patterns)
    dropout: float = 0.15         # Higher dropout for regularization
    batch_norm: bool = True
    activation: str = "relu"
    
    # Training parameters (from teacher_student.yaml)
    target_parameters: int = 1740000  # ~1.74M params
    curriculum_stages: List[str] = field(default_factory=lambda: ["pretrain", "distill"])
    
    # Legacy compatibility
    hidden_channels: int = 128    # Larger hidden size for teacher
    embedding_dim: int = 32       # CAN ID embedding dimension (larger for teacher)
    beta: float = 1.0             # KL divergence weight

@dataclass
class StudentVGAEConfig:
    """Student VGAE configuration for on-board deployment (~87K parameters)."""
    type: str = "vgae_student"
    input_dim: int = 11           # CAN features input
    node_embedding_dim: int = 128 # Rich initial embedding (from teacher_student.yaml)
    encoder_dims: List[int] = field(default_factory=lambda: [128, 64, 24])  # Proper compression
    decoder_dims: List[int] = field(default_factory=lambda: [24, 64, 128])  # Proper decompression
    latent_dim: int = 24          # Compact latent space (from teacher_student.yaml)
    output_dim: int = 11          # Reconstruct input features
    
    # Architecture settings
    attention_heads: int = 2      # Lightweight attention
    dropout: float = 0.1          # Lower dropout for student
    batch_norm: bool = True
    activation: str = "relu"
    
    # Deployment constraints (from teacher_student.yaml)
    target_parameters: int = 87000    # ~87K params
    memory_budget_kb: int = 287       # 87KB model + 200KB buffer
    inference_time_ms: int = 5        # Must be <20ms CAN message period
    
    # Legacy compatibility
    hidden_channels: int = 64     # For backward compatibility
    num_layers: int = 2           # Simple encoder/decoder
    embedding_dim: int = 8        # CAN ID embedding dimension
    beta: float = 1.0             # KL divergence weight

@dataclass
class DQNConfig:
    """DQN Teacher configuration (~687K parameters)."""
    type: str = "dqn"
    input_dim: int = 20           # State space dimension
    output_dim: int = 11          # Action space size (|A|)
    
    # Architecture settings (3 layers, 576 hidden units)
    num_layers: int = 3           # Teacher depth
    hidden_units: int = 576       # Hidden layer size (tuned for 687K params)
    hidden_channels: int = 576    # Channel dimension
    
    # DQN-specific parameters
    gamma: float = 0.99           # Discount factor
    lr: float = 1e-3              # Learning rate
    epsilon: float = 0.1          # Exploration rate
    epsilon_decay: float = 0.995  # Epsilon decay
    min_epsilon: float = 0.01     # Minimum epsilon
    buffer_size: int = 100000     # Experience replay buffer
    batch_size: int = 128         # Training batch size
    target_update_freq: int = 100 # Target network update frequency
    
    # Training parameters
    target_parameters: int = 687000  # ~687K params (actual)
    dropout: float = 0.2          # Regularization
    activation: str = "relu"
    use_double_dqn: bool = True   # Use Double DQN
    use_dueling: bool = False     # Dueling architecture (optional)

@dataclass
class StudentDQNConfig:
    """Student DQN configuration for on-board deployment (~32K parameters)."""
    type: str = "dqn_student"
    input_dim: int = 20           # State space dimension
    output_dim: int = 11          # Action space size (|A|)
    
    # Architecture settings (2 layers, 160 hidden units)
    num_layers: int = 2           # Student depth (lightweight)
    hidden_units: int = 160       # Hidden layer size (tuned for 32K params)
    hidden_channels: int = 160    # Channel dimension
    
    # DQN-specific parameters (same as teacher but optimized)
    gamma: float = 0.99           # Discount factor
    lr: float = 1e-3              # Learning rate
    epsilon: float = 0.05         # Lower exploration for student
    epsilon_decay: float = 0.99   # Faster decay for student
    min_epsilon: float = 0.001    # Lower minimum for student
    buffer_size: int = 50000      # Smaller replay buffer
    batch_size: int = 64          # Smaller batch size
    target_update_freq: int = 50  # More frequent updates
    
    # Deployment constraints
    target_parameters: int = 32000    # ~32K params (actual)
    memory_budget_kb: int = 128       # 32KB model + 96KB buffer
    inference_time_ms: int = 3        # Must be <20ms CAN message period
    dropout: float = 0.1              # Lower dropout for student
    activation: str = "relu"
    use_double_dqn: bool = True       # Keep Double DQN for stability
    use_dueling: bool = False         # Skip dueling to save parameters

# ============================================================================
# Dataset Configurations
# ============================================================================

@dataclass
class BaseDatasetConfig:
    """Base dataset configuration."""
    name: str
    data_path: Optional[str] = None
    cache_dir: Optional[str] = None
    cache_processed_data: bool = True
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "cache_processed_data": True,
        "normalize_features": False,
        "feature_scaling": "standard"
    })


@dataclass
class CANDatasetConfig(BaseDatasetConfig):
    """CAN bus dataset configuration."""
    time_window: int = 100
    overlap: int = 100
    max_graphs: Optional[int] = None
    balance_classes: bool = True
    attack_types: List[str] = field(default_factory=lambda: ["normal", "attack"])


# ============================================================================
# Training Configurations
# ============================================================================

@dataclass
class BaseTrainingConfig:
    """Base training configuration."""
    mode: str = "normal"
    max_epochs: int = 400
    batch_size: int = 64  # Starting point for Lightning Tuner
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Optimization
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Training behavior
    early_stopping_patience: int = 100  # Increased for more robust training
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Hardware optimization
    precision: str = "32-true"
    find_unused_parameters: bool = False
    
    # Lightning Tuner - enabled for optimal batch size
    optimize_batch_size: bool = True
    batch_size_mode: str = "power"
    max_batch_size_trials: int = 10
    
    # Evaluation
    run_test: bool = True
    test_every_n_epochs: int = 5
    
    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    
    # Logging
    log_every_n_steps: int = 50
    save_hyperparameters: bool = True


@dataclass
class NormalTrainingConfig(BaseTrainingConfig):
    """Normal training configuration."""
    mode: str = "normal"
    description: str = "Standard supervised training"


@dataclass
class AutoencoderTrainingConfig(BaseTrainingConfig):
    """Autoencoder training configuration."""
    mode: str = "autoencoder"
    description: str = "Unsupervised autoencoder training on normal samples only"
    max_epochs: int = 400
    learning_rate: float = 0.0005
    reconstruction_loss: str = "mse"
    use_normal_samples_only: bool = True
    early_stopping_patience: int = 100  # Longer patience for autoencoder convergence


@dataclass
class KnowledgeDistillationConfig(BaseTrainingConfig):
    """Knowledge distillation training configuration."""
    mode: str = "knowledge_distillation"
    description: str = "Knowledge distillation from teacher to student model"
    
    # Distillation specific
    teacher_model_path: Optional[str] = None
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    student_model_scale: float = 1.0
    
    # Adjusted defaults for distillation
    max_epochs: int = 400
    batch_size: int = 64  # Starting point for Lightning Tuner
    learning_rate: float = 0.002
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 2
    
    # Memory optimization
    memory_optimization: MemoryOptimizationConfig = field(default_factory=MemoryOptimizationConfig)
    
    # Enhanced monitoring for distillation
    log_teacher_student_comparison: bool = True
    compare_with_teacher: bool = True
    
    # More conservative batch size optimization
    optimize_batch_size: bool = True
    max_batch_size_trials: int = 15


@dataclass
class StudentBaselineTrainingConfig(BaseTrainingConfig):
    """Baseline student training without knowledge distillation."""
    mode: str = "student_baseline"
    description: str = "Student-only supervised training (no teacher)"
    max_epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 0.001
    precision: str = "16-mixed"
    early_stopping_patience: int = 50


@dataclass
class FusionAgentConfig:
    """DQN Fusion Agent configuration (nested within FusionTrainingConfig)."""
    alpha_steps: int = 21
    fusion_lr: float = 0.001
    gamma: float = 0.9
    fusion_epsilon: float = 0.9
    fusion_epsilon_decay: float = 0.995
    fusion_min_epsilon: float = 0.2
    fusion_buffer_size: int = 100000
    fusion_batch_size: int = 32768
    target_update_freq: int = 100
    hidden_dim: int = 128


@dataclass
class FusionTrainingConfig(BaseTrainingConfig):
    """Fusion training configuration."""
    mode: str = "fusion"
    description: str = "Multi-model fusion with reinforcement learning"
    
    # Fusion specific parameters
    fusion_episodes: int = 500
    max_train_samples: int = 150000
    max_val_samples: int = 30000
    
    # Pipeline parallelism
    episode_sample_size: int = 20000
    training_step_interval: int = 32
    gpu_training_steps: int = 16
    
    # Model paths for fusion (optional - will be auto-detected if not provided)
    autoencoder_path: Optional[str] = None
    classifier_path: Optional[str] = None
    
    # Fusion agent configuration
    fusion_agent_config: FusionAgentConfig = field(default_factory=FusionAgentConfig)


@dataclass
class CurriculumTrainingConfig(BaseTrainingConfig):
    """Curriculum learning training configuration."""
    mode: str = "curriculum"
    description: str = "GAT training with curriculum learning and VGAE-based hard mining"
    
    # Curriculum learning parameters
    vgae_model_path: Optional[str] = None
    start_ratio: float = 1.0       # 1:1 normal:attack (easy start)
    end_ratio: float = 10.0        # 10:1 normal:attack (realistic end, not 100:1)
    difficulty_percentile: float = 75.0  # Use top 75% difficult samples
    
    # Training adjustments for curriculum
    max_epochs: int = 400
    batch_size: int = 32           # Starting batch size for optimization
    learning_rate: float = 0.001
    early_stopping_patience: int = 150  # Longer patience for curriculum
    
    # Batch size optimization
    optimize_batch_size: bool = True   # Enable batch size optimization using max dataset size
    batch_size_mode: str = "power"     # Lightning tuner mode: power, binsearch
    max_batch_size_trials: int = 15    # More trials for curriculum learning
    dynamic_batch_recalc_threshold: float = 2.0  # Recalculate if dataset grows >2x
    
    # Hard mining parameters
    use_vgae_mining: bool = True       # Enable VGAE-based hard mining
    difficulty_cache_update: int = 10  # Update difficulty cache every N epochs
    
    # Memory preservation
    use_memory_preservation: bool = True  # Prevent catastrophic forgetting
    memory_strength: float = 0.1          # EWC regularization strength


# ============================================================================
# Lightning Trainer Configuration
# ============================================================================

@dataclass
class TrainerConfig:
    """PyTorch Lightning Trainer configuration."""
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "32-true"
    max_epochs: int = 400
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    num_sanity_val_steps: int = 2


# ============================================================================
# Complete Application Configurations
# ============================================================================

@dataclass
class CANGraphConfig:
    """Complete CAN-Graph application configuration."""
    # Core components
    model: Union[GATConfig, StudentGATConfig, VGAEConfig, StudentVGAEConfig, DQNConfig, StudentDQNConfig]
    dataset: CANDatasetConfig
    training: Union[NormalTrainingConfig, AutoencoderTrainingConfig, 
                   KnowledgeDistillationConfig, StudentBaselineTrainingConfig,
                   FusionTrainingConfig, CurriculumTrainingConfig]
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    
    # Global settings
    seed: int = 42
    project_name: str = "can_graph_lightning"
    experiment_name: str = field(init=False)
    
    # Output directories
    output_dir: str = "outputs"
    model_save_dir: str = "saved_models"
    log_dir: str = "outputs/lightning_logs"
    
    # Logging
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "enable_tensorboard": False,
        "save_top_k": 3,
        "monitor_metric": "val_loss",
        "monitor_mode": "min"
    })
    
    def __post_init__(self):
        """Generate experiment name after initialization."""
        self.experiment_name = f"{self.model.type}_{self.dataset.name}_{self.training.mode}"


# ============================================================================
# Configuration Store and Presets
# ============================================================================

class CANGraphConfigStore:
    """Central configuration store with presets."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.store = store
            self._register_base_configs()
            self._register_presets()
            CANGraphConfigStore._initialized = True
    
    def _register_base_configs(self):
        """Register base configuration components."""
        # Models
        self.store(GATConfig, name="gat", group="model")  # Teacher GAT
        self.store(StudentGATConfig, name="gat_student", group="model")  # Student GAT
        self.store(VGAEConfig, name="vgae", group="model")  # Teacher VGAE
        self.store(StudentVGAEConfig, name="vgae_student", group="model")  # Student VGAE
        self.store(DQNConfig, name="dqn", group="model")  # Teacher DQN
        self.store(StudentDQNConfig, name="dqn_student", group="model")  # Student DQN
        
        # Training modes
        self.store(NormalTrainingConfig, name="normal", group="training")
        self.store(AutoencoderTrainingConfig, name="autoencoder", group="training")
        self.store(KnowledgeDistillationConfig, name="knowledge_distillation", group="training")
        self.store(StudentBaselineTrainingConfig, name="student_baseline", group="training")
        self.store(FusionTrainingConfig, name="fusion", group="training")
        
        # Trainer
        self.store(TrainerConfig, name="default", group="trainer")
    
    def _register_presets(self):
        """Register common preset configurations."""
        # Dataset presets
        datasets = {
            "hcrl_sa": CANDatasetConfig(name="hcrl_sa", data_path="datasets/can-train-and-test-v1.5/hcrl-sa"),
            "hcrl_ch": CANDatasetConfig(name="hcrl_ch", data_path="datasets/can-train-and-test-v1.5/hcrl-ch"),
            "set_01": CANDatasetConfig(name="set_01", data_path="datasets/can-train-and-test-v1.5/set_01"),
            "set_02": CANDatasetConfig(name="set_02", data_path="datasets/can-train-and-test-v1.5/set_02"),
            "set_03": CANDatasetConfig(name="set_03", data_path="datasets/can-train-and-test-v1.5/set_03"),
            "set_04": CANDatasetConfig(name="set_04", data_path="datasets/can-train-and-test-v1.5/set_04"),
        }
        
        for name, config in datasets.items():
            self.store(config, name=name, group="dataset")
    
    def create_config(self, model_type: str = "gat", dataset_name: str = "hcrl_sa", 
                     training_mode: str = "normal", **overrides) -> CANGraphConfig:
        """Create a complete configuration with specified components."""
        
        # Get base components
        model_config = self.get_model_config(model_type)
        dataset_config = self.get_dataset_config(dataset_name)
        training_config = self.get_training_config(training_mode)
        trainer_config = TrainerConfig()
        
        # Apply any overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(training_config, key):
                    setattr(training_config, key, value)
        
        return CANGraphConfig(
            model=model_config,
            dataset=dataset_config,
            training=training_config,
            trainer=trainer_config
        )
    
    def get_model_config(self, model_type: str) -> Union[GATConfig, StudentGATConfig, VGAEConfig, StudentVGAEConfig, DQNConfig, StudentDQNConfig]:
        """Get model configuration by type."""
        if model_type == "gat":
            return GATConfig()
        elif model_type == "gat_student":
            return StudentGATConfig()
        elif model_type == "vgae":
            return VGAEConfig()
        elif model_type == "vgae_student":
            return StudentVGAEConfig()
        elif model_type == "dqn":
            return DQNConfig()
        elif model_type == "dqn_student":
            return StudentDQNConfig()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_dataset_config(self, dataset_name: str) -> CANDatasetConfig:
        """Get dataset configuration by name."""
        dataset_configs = {
            "hcrl_sa": CANDatasetConfig(name="hcrl_sa", data_path="datasets/can-train-and-test-v1.5/hcrl-sa"),
            "hcrl_ch": CANDatasetConfig(name="hcrl_ch", data_path="datasets/can-train-and-test-v1.5/hcrl-ch"),
            "set_01": CANDatasetConfig(name="set_01", data_path="datasets/can-train-and-test-v1.5/set_01"),
            "set_02": CANDatasetConfig(name="set_02", data_path="datasets/can-train-and-test-v1.5/set_02"),
            "set_03": CANDatasetConfig(name="set_03", data_path="datasets/can-train-and-test-v1.5/set_03"),
            "set_04": CANDatasetConfig(name="set_04", data_path="datasets/can-train-and-test-v1.5/set_04")
        }
        
        if dataset_name not in dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_configs.keys())}")
        
        return dataset_configs[dataset_name]
    
    def get_training_config(self, training_mode: str):
        """Get training configuration by mode."""
        training_configs = {
            "normal": NormalTrainingConfig(),
            "autoencoder": AutoencoderTrainingConfig(),
            "knowledge_distillation": KnowledgeDistillationConfig(),
            "student_baseline": StudentBaselineTrainingConfig(),
            "fusion": FusionTrainingConfig(),
            "curriculum": CurriculumTrainingConfig()
        }
        
        if training_mode not in training_configs:
            raise ValueError(f"Unknown training mode: {training_mode}. Available: {list(training_configs.keys())}")
        
        return training_configs[training_mode]


# ============================================================================
# Quick Configuration Functions
# ============================================================================

def create_gat_normal_config(dataset: str = "hcrl_sa", **kwargs) -> CANGraphConfig:
    """Quick config for normal GAT training."""
    store_manager = CANGraphConfigStore()
    return store_manager.create_config("gat", dataset, "normal", **kwargs)


def create_distillation_config(dataset: str = "hcrl_sa", teacher_path: str = None, 
                             student_scale: float = 1.0, **kwargs) -> CANGraphConfig:
    """Quick config for knowledge distillation."""
    store_manager = CANGraphConfigStore()
    overrides = {"teacher_model_path": teacher_path, "student_model_scale": student_scale}
    overrides.update(kwargs)
    return store_manager.create_config("gat", dataset, "knowledge_distillation", **overrides)


def create_autoencoder_config(dataset: str = "hcrl_sa", **kwargs) -> CANGraphConfig:
    """Quick config for autoencoder training."""
    store_manager = CANGraphConfigStore()
    return store_manager.create_config("gat", dataset, "autoencoder", **kwargs)


def create_fusion_config(dataset: str = "hcrl_sa", **kwargs) -> CANGraphConfig:
    """Quick config for fusion training."""
    store_manager = CANGraphConfigStore()
    return store_manager.create_config("gat", dataset, "fusion", **kwargs)


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config(config: CANGraphConfig) -> bool:
    """Validate configuration for common issues."""
    issues = []
    
    # Check knowledge distillation requirements
    if config.training.mode == "knowledge_distillation":
        if not config.training.teacher_model_path:
            issues.append("Knowledge distillation requires teacher_model_path")
        elif not Path(config.training.teacher_model_path).exists():
            issues.append(f"Teacher model not found: {config.training.teacher_model_path}")
    
    # Check dataset path
    if config.dataset.data_path and not Path(config.dataset.data_path).exists():
        issues.append(f"Dataset path not found: {config.dataset.data_path}")
    
    # Check compatibility
    if config.training.precision == "16-mixed" and config.trainer.precision != "16-mixed":
        issues.append("Training precision and trainer precision should match")
    
    if issues:
        for issue in issues:
            print(f"❌ {issue}")
        return False
    
    print("✅ Configuration validation passed")
    return True


# Initialize the global store
config_store = CANGraphConfigStore()