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
    """Graph Attention Network configuration."""
    type: str = "gat"
    input_dim: int = 11
    hidden_channels: int = 64
    output_dim: int = 2
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.2
    num_fc_layers: int = 3
    embedding_dim: int = 8
    use_jumping_knowledge: bool = True
    jk_mode: str = "cat"
    use_residual: bool = True
    use_batch_norm: bool = False
    activation: str = "relu"


@dataclass
class VGAEConfig:
    """Variational Graph Autoencoder configuration."""
    type: str = "vgae"
    input_dim: int = 11
    hidden_channels: int = 64
    latent_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.2


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
    time_window: int = 50
    overlap: int = 10
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
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Optimization
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Training behavior
    early_stopping_patience: int = 25  # Increased for more robust training
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Hardware optimization
    precision: str = "32-true"
    find_unused_parameters: bool = False
    
    # Lightning Tuner - disabled to avoid batch size warnings
    optimize_batch_size: bool = False
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
    max_epochs: int = 150
    learning_rate: float = 0.0005
    reconstruction_loss: str = "mse"
    use_normal_samples_only: bool = True
    early_stopping_patience: int = 30  # Longer patience for autoencoder convergence


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
    max_epochs: int = 80
    batch_size: int = 32
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
class FusionTrainingConfig(BaseTrainingConfig):
    """Fusion training configuration."""
    mode: str = "fusion"
    description: str = "Multi-model fusion with reinforcement learning"
    
    # Fusion specific parameters
    fusion_episodes: int = 250
    max_train_samples: int = 150000
    max_val_samples: int = 30000
    alpha_steps: int = 21
    fusion_lr: float = 0.001
    fusion_epsilon: float = 0.9
    fusion_epsilon_decay: float = 0.995
    fusion_min_epsilon: float = 0.2
    
    # Pipeline parallelism
    fusion_buffer_size: int = 100000
    fusion_batch_size: int = 32768
    fusion_target_update: int = 15
    episode_sample_size: int = 20000
    training_step_interval: int = 32
    gpu_training_steps: int = 16


# ============================================================================
# Lightning Trainer Configuration
# ============================================================================

@dataclass
class TrainerConfig:
    """PyTorch Lightning Trainer configuration."""
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "32-true"
    max_epochs: int = 100
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
    model: Union[GATConfig, VGAEConfig]
    dataset: CANDatasetConfig
    training: Union[NormalTrainingConfig, AutoencoderTrainingConfig, 
                   KnowledgeDistillationConfig, FusionTrainingConfig]
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
        self.store(GATConfig, name="gat", group="model")
        self.store(VGAEConfig, name="vgae", group="model")
        
        # Training modes
        self.store(NormalTrainingConfig, name="normal", group="training")
        self.store(AutoencoderTrainingConfig, name="autoencoder", group="training")
        self.store(KnowledgeDistillationConfig, name="knowledge_distillation", group="training")
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
    
    def get_model_config(self, model_type: str) -> Union[GATConfig, VGAEConfig]:
        """Get model configuration by type."""
        if model_type == "gat":
            return GATConfig()
        elif model_type == "vgae":
            return VGAEConfig()
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
            "fusion": FusionTrainingConfig()
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