"""
Hydra-Zen Configuration System for CAN-Graph
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from hydra_zen import make_config, store, zen
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class BatchSizeSafetyConfig:
    """Safety Factor for Graph Data Batch Size"""

    hcrl_ch: 0.6
    hcrl_sa: 0.55
    set_01: 0.55
    set_02: 0.35
    set_03: 0.35
    set_04: 0.35

    hcrl_ch_kd: 0.45
    hcrl_sa_kd: 0.41
    set_01_kd: 0.41
    set_02_kd: 0.26
    set_03_kd: 0.26
    set_04_kd: 0.26

# ============================================================================
# Base Configuration Classes
# ============================================================================
@dataclass
class TrainingConfigBase:

    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "32-true"
    max_epochs: int = 400
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    num_sanity_val_steps: int = 2
    """Base training configuration."""
    mode: str = "normal"
    max_epochs: int = 500
    batch_size: int = 512
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    momentum: float = 0.9

    """Learning rate scheduler configuration."""
    use_scheduler: bool = False
    scheduler_type: str = "cosine"
    params: Dict[str, Any] = field(default_factory=lambda: {"T_max": 100})


    """Memory optimization settings."""
    use_teacher_cache: bool = True
    clear_cache_every_n_steps: int = 100
    offload_teacher_to_cpu: bool = False
    gradient_checkpointing: bool = True  # Enabled by default for GNN memory efficiency

    """Batch size tuning and optimization settings."""
    default_batch_size: int = 64
    tuned_batch_size: Optional[int] = None
    safety_factor: float = 0.5

    optimize_batch_size: bool = False
    batch_size_mode: str = "binsearch"
    max_batch_size_trials: int = 10

    # Training behavior
    early_stopping_patience: int = 100  # Increased for more robust training
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1

    # Hardware optimization
    precision: str = "32-true"
    find_unused_parameters: bool = False


    use_adaptive_batch_size_factor: bool = True  # Enable adaptive learning (recommended)
    graph_memory_safety_factor: Optional[float] = None  # None = use JSON file or auto-select

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

    # Reproducibility and validation
    seed: Optional[int] = None
    deterministic_training: bool = True
    cudnn_benchmark: bool = False
    # Automatically validate saved artifacts after training (torch.load weights_only) and write sanitized copies if needed
    validate_artifacts: bool = True

    use_knowledge_distillation: bool = False  # Toggle KD on/off
    teacher_model_path: Optional[str] = None  # Path to teacher .pth or .ckpt file
    distillation_temperature: float = 4.0     # Temperature for soft labels (higher = softer)
    distillation_alpha: float = 0.7 
    
    """Autoencoder training configuration."""
    mode: str = "autoencoder"
    description: str = "Unsupervised autoencoder training on normal samples only"
    max_epochs: int = 400
    learning_rate: float = 0.002
    batch_size: int = 64  # Starting point for Lightning Tuner
    reconstruction_loss: str = "mse"
    use_normal_samples_only: bool = True
    early_stopping_patience: int = 100 
    
    
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

    # Note: memory_optimization now inherited from BaseTrainingConfig

    # Enhanced monitoring for distillation
    log_teacher_student_comparison: bool = True
    compare_with_teacher: bool = True
    
    # More conservative batch size optimization
    optimize_batch_size: bool = True
    max_batch_size_trials: int = 15

@dataclass
class EvaluationTrainingConfig(BaseTrainingConfig):
    """Evaluation configuration to run comprehensive evaluation pipelines."""
    mode: str = "evaluation"
    description: str = "Run comprehensive evaluation (student & teacher) and save reports"

    # Optional explicit model paths (if not provided, will use hierarchical defaults)
    autoencoder_path: Optional[str] = None
    classifier_path: Optional[str] = None
    threshold_path: Optional[str] = None

    # Execution options
    device: str = "cuda"  # 'cuda' or 'cpu' or 'auto'
    optimize_thresholds: bool = True
    threshold_methods: List[str] = field(default_factory=lambda: ["anomaly_only", "two_stage"])

    # Reporting & plotting
    save_report_dir: str = "outputs/evaluation"
    save_plots: bool = True
    save_plots_dir: str = "outputs/workbench_plots"
    plot_examples: int = 3  # Number of example graphs to plot for visualizations

    # MLflow options
    use_mlflow: bool = True
    mlflow_experiment_name: Optional[str] = None  # override experiment name if desired

    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "roc_auc"])
    evaluate_student: bool = True
    evaluate_teacher: bool = True
    run_on_test: bool = True
    batch_size: Optional[int] = None
    memory_strength: float = 0.1          # EWC regularization strength


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
    num_layers: int = 5           # Fixed shallow architecture: 5 GAT layers
    heads: int = 8                # Attention heads
    dropout: float = 0.2
    # Reduce fully-connected head MLP depth to keep parameter budget within targets
    num_fc_layers: int = 1
    embedding_dim: int = 32       # Matches paper: 32 dimensions
    use_jumping_knowledge: bool = True
    jk_mode: str = "cat"
    use_residual: bool = True
    use_batch_norm: bool = False
    activation: str = "relu"
    # Expected parameter budget for this teacher GAT architecture
    target_parameters: int = 1100000  # ~1.1M params (tunable)


@dataclass
class StudentGATConfig:
    """Student GAT configuration for on-board deployment (55K parameters)."""
    type: str = "gat_student"
    input_dim: int = 11
    hidden_channels: int = 24     # Reduced for student budget
    output_dim: int = 2           # Binary classification
    num_layers: int = 2           # Matches paper: 2 GAT layers
    heads: int = 4                # Matches paper: 4 attention heads
    dropout: float = 0.1          # Lower dropout for student
    num_fc_layers: int = 2        # Simpler classification head
    embedding_dim: int = 8        # Matches paper: 8 dimensions
    use_jumping_knowledge: bool = False  # Simpler architecture
    jk_mode: str = "cat"
    use_residual: bool = False    # Simpler for deployment
    use_batch_norm: bool = False
    activation: str = "relu"    # Expected parameter budget for student GAT (on-board deployment)
    target_parameters: int = 55000  # ~55K params (tunable)

@dataclass
class VGAEConfig:
    """Teacher VGAE configuration (~1.71M parameters, tuned for 20x compression)."""
    type: str = "vgae"
    input_dim: int = 11           # Same input as student
    node_embedding_dim: int = 256 # Richer embedding (from teacher_student.yaml)
    # Tuned hidden_dims to meet teacher parameter budget (~1.71M) with reduced MLP
    # NOTE: Must include latent_dim at end ([1024, 512, 96]) for proper decoder mirroring
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 96])  # Two-layer compression + latent
    latent_dim: int = 96          # Larger latent space for rich representations
    output_dim: int = 11          # Reconstruct input features
    
    # Architecture settings (tuned for ~1.71M params)
    num_layers: int = 2           # Encoder/decoder depth
    attention_heads: int = 4      # 4 heads per layer (tuned)
    embedding_dim: int = 64       # CAN ID embedding dimension
    mlp_hidden: int = 256         # Neighborhood decoder MLP hidden size (tuned)
    dropout: float = 0.15         # Higher dropout for regularization
    batch_norm: bool = True
    activation: str = "relu"
    
    # Training parameters
    target_parameters: int = 1710000  # ~1.71M params (tuned)
    curriculum_stages: List[str] = field(default_factory=lambda: ["pretrain", "distill"])
    
    # Legacy compatibility
    hidden_channels: int = 128    # Larger hidden size for teacher
    beta: float = 1.0             # KL divergence weight

@dataclass
class StudentVGAEConfig:
    """Student VGAE configuration for on-board deployment (~86K parameters, tuned)."""
    type: str = "vgae_student"
    input_dim: int = 11           # CAN features input
    node_embedding_dim: int = 80  # Initial embedding (tuned)
    # Tuned encoder_dims for ~86K parameters with reduced MLP
    # NOTE: For compatibility with decoder, include latent_dim at end
    encoder_dims: List[int] = field(default_factory=lambda: [80, 40, 16])  # Compact two-layer + latent
    decoder_dims: List[int] = field(default_factory=lambda: [40, 80])  # Mirror of encoder
    latent_dim: int = 16          # Compact latent space (tuned)
    output_dim: int = 11          # Reconstruct input features
    
    # Architecture settings (tuned for ~86K params)
    attention_heads: int = 1      # Single head for efficiency
    mlp_hidden: int = 16          # Compact neighborhood decoder MLP (tuned)
    dropout: float = 0.1          # Lower dropout for student
    batch_norm: bool = True
    activation: str = "relu"
    
    # Deployment constraints
    target_parameters: int = 86000    # ~86K params (tuned)
    memory_budget_kb: int = 287       # 86KB model + 200KB buffer
    inference_time_ms: int = 5        # Must be <20ms CAN message period    
    # Legacy compatibility
    hidden_channels: int = 40     # For backward compatibility (tuned)
    num_layers: int = 2           # Simple encoder/decoder
    embedding_dim: int = 4        # CAN ID embedding dimension (tuned)
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
    """Base dataset configuration.

    New fields:
      - modality: high-level modality name (default: 'automotive')
      - experiment_root: where experiment outputs will be placed (default: 'experiment_runs')

    The dataset config will infer reasonable defaults for `data_path` and `cache_dir`
    if they are not provided. This is strict configuration construction but does not
    perform filesystem validation (validation is performed by `validate_config`).
    """
    name: str
    modality: str = "automotive"
    data_path: Optional[str] = None
    cache_dir: Optional[str] = None
    experiment_root: str = "experimentruns"
    cache_processed_data: bool = True
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "cache_processed_data": True,
        "normalize_features": False,
        "feature_scaling": "standard"
    })

    def __post_init__(self):
        # If a data_path is not supplied, infer it from the standard datasets layout
        # CORRECT PATH: data/automotive/{name}
        if not self.data_path:
            self.data_path = f"data/automotive/{self.name}"

        # If no cache directory given, place it under the canonical experiment root
        if not self.cache_dir:
            self.cache_dir = str(Path(self.experiment_root) / self.modality / self.name / "cache")

        # Keep attributes as strings to avoid accidental Path issues in configs
        self.data_path = str(self.data_path)
        self.cache_dir = str(self.cache_dir)



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
       # Weight: alpha*KD_loss + (1-alpha)*task_loss


@dataclass
class KnowledgeDistillationConfig(BaseTrainingConfig):
    

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
    curriculum_memory_multiplier: float = 1.0  # Batch size multiplier for curriculum mode (use 0.5 for dense datasets)
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


# ============================================================================
# Complete Application Configurations
# ============================================================================

@dataclass
class CANGraphConfig:
    """Complete CAN-Graph application configuration.

    New fields introduced (strict; no legacy fallbacks):
      - experiment_root: base directory for all experiments (default: 'experiment_runs')
      - modality: high-level modality (e.g., 'automotive')
      - model_size: 'teacher' or 'student' (defaults inferred from model type)
      - distillation: 'distilled' or 'no_distillation' (inferred from training.teacher_model_path)

    The config exposes helper methods to compute canonical experiment directories and
    enforces strict validation for modes that require pre-trained artifacts (e.g., fusion,
    curriculum, knowledge_distillation). If required artifacts are missing the validator
    raises informative errors (no implicit fallbacks).
    """
    # Core components
    model: Union[GATConfig, StudentGATConfig, VGAEConfig, StudentVGAEConfig, DQNConfig, StudentDQNConfig]
    dataset: CANDatasetConfig
    training: Union[NormalTrainingConfig, AutoencoderTrainingConfig,
                   KnowledgeDistillationConfig, StudentBaselineTrainingConfig,
                   FusionTrainingConfig, CurriculumTrainingConfig, EvaluationTrainingConfig]
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    batch_size_config: BatchSizeConfig = field(default_factory=BatchSizeConfig)

    # Global settings
    seed: int = 42
    project_name: str = "can_graph_lightning"
    experiment_name: str = field(init=False)

    # Canonical experiment layout root (will be created by training when needed)
    experiment_root: str = "experimentruns"
    modality: str = "automotive"
    model_size: str = field(default=None)  # 'teacher' or 'student' (inferred if None)
    distillation: str = field(default=None)  # 'distilled' or 'no_distillation' (inferred if None)

    # Output directories (kept for backward compatibility inside code but not used for canonical saving)
    output_dir: str = "outputs"
    model_save_dir: str = "saved_models"
    log_dir: str = "outputs/lightning_logs"
    
    # Logging
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "use_tensorboard": False,  # Use MLFlow instead
        "use_mlflow": True,
        "save_top_k": 3,
        "monitor_metric": "val_loss",
        "monitor_mode": "min",
        "log_interval": 50,
        "val_check_interval": 1
    })

    def __post_init__(self):
        """Generate experiment name and infer model size/distillation after initialization."""
        # Name used for human readability and for backward compatibility in some logs
        self.experiment_name = f"{self.model.type}_{self.dataset.name}_{self.training.mode}"

        # Infer model_size if not provided
        if self.model_size is None:
            self.model_size = "student" if "student" in getattr(self.model, "type", "") else "teacher"

        # Infer distillation if not explicitly provided
        if self.distillation is None:
            self.distillation = "distilled" if getattr(self.training, "teacher_model_path", None) else "no_distillation"

    def canonical_experiment_dir(self) -> Path:
        """Return the canonical experiment directory as a Path.

        Layout:
            {experiment_root}/{modality}/{dataset}/{model_size}/{learning_type}/{model_arch}/{distillation}/{training_mode}/
        """
        # Determine base model architecture (e.g., 'vgae' from 'vgae_student')
        raw_type = getattr(self.model, "type", "unknown")
        model_arch = raw_type.replace('_student', '').replace('_teacher', '')

        # Determine model_size explicitly if type contains student/teacher hints
        model_size = self.model_size

        # Determine learning_type with some special-casing for student distillation:
        # - autoencoder training is unsupervised
        # - fusion is rl_fusion
        # - knowledge_distillation: use the teacher's learning type inferred from model_arch
        if self.training.mode == "autoencoder":
            learning_type = "unsupervised"
        elif self.training.mode == "fusion":
            learning_type = "rl_fusion"
        elif self.training.mode == "knowledge_distillation":
            # If the student model is derived from an unsupervised architecture (vgae), keep it under unsupervised;
            # otherwise default to supervised
            if model_arch == "vgae":
                learning_type = "unsupervised"
            else:
                learning_type = "supervised"
        else:
            learning_type = "supervised"

        base = Path(self.experiment_root)
        parts = [self.modality, self.dataset.name, learning_type, model_arch, model_size, self.distillation, self.training.mode]
        return (base.joinpath(*parts)).resolve()

    def required_artifacts(self) -> Dict[str, Path]:
        """Return a mapping of artifact name -> required canonical Path for this config.

        This makes it explicit which pre-trained models are required for the current
        `training.mode`. Uses glob-based discovery to find models flexibly rather
        than hardcoding exact filenames.
        
        Resolution priority for each artifact:
        1. Glob-based discovery in canonical directory (handles any *.pth matching the pattern)
        2. Fallback to hardcoded default path (for pre-existence checks before training)
        """
        from src.paths import PathResolver
        
        artifacts = {}
        exp_dir = self.canonical_experiment_dir()
        resolver = PathResolver(self)

        if self.training.mode == "knowledge_distillation":
            # Teacher model must exist and be specified
            teacher_path = getattr(self.training, "teacher_model_path", None)
            if teacher_path:
                artifacts["teacher_model"] = Path(teacher_path)
            else:
                # Infer teacher location based on student model type
                raw_type = getattr(self.model, "type", "unknown")
                model_arch = raw_type.replace('_student', '').replace('_teacher', '')
                
                if model_arch == "vgae":
                    # VGAE teacher is in autoencoder mode - use discovery
                    teacher_dir = Path(self.experiment_root) / self.modality / self.dataset.name / "unsupervised" / "vgae" / "teacher" / "no_distillation" / "autoencoder"
                    discovered = resolver.discover_model(teacher_dir, 'vgae', require_exists=False)
                    artifacts["teacher_model"] = discovered or (teacher_dir / "models" / "vgae_autoencoder.pth")
                elif model_arch == "gat":
                    # GAT teacher - try curriculum first, then normal (using discovery)
                    teacher_dir_curriculum = Path(self.experiment_root) / self.modality / self.dataset.name / "supervised" / "gat" / "teacher" / "no_distillation" / "curriculum"
                    teacher_dir_normal = Path(self.experiment_root) / self.modality / self.dataset.name / "supervised" / "gat" / "teacher" / "no_distillation" / "normal"
                    
                    # Try discovery in each location
                    discovered = resolver.discover_model(teacher_dir_curriculum, 'gat', require_exists=False)
                    if not discovered:
                        discovered = resolver.discover_model(teacher_dir_normal, 'gat', require_exists=False)
                    
                    # Fallback to default path if nothing discovered
                    artifacts["teacher_model"] = discovered or (teacher_dir_curriculum / "models" / "gat_curriculum.pth")
                else:
                    # Generic fallback
                    artifacts["teacher_model"] = exp_dir / "teacher" / f"best_teacher_model_{self.dataset.name}.pth"

        # Use discovery for fusion mode artifacts
        if self.training.mode == "fusion":
            # Determine if this is teacher fusion or student fusion
            is_student_fusion = self.model_size == "student" or self.distillation == "distilled"
            
            if is_student_fusion:
                # Student fusion needs student VGAE (autoencoder mode) and student GAT (curriculum mode)
                # Student models are trained with their base modes (autoencoder/curriculum), not knowledge_distillation mode
                ae_dir = Path(self.experiment_root) / self.modality / self.dataset.name / "unsupervised" / "vgae" / "student" / self.distillation / "autoencoder"
                clf_dir = Path(self.experiment_root) / self.modality / self.dataset.name / "supervised" / "gat" / "student" / self.distillation / "curriculum"

                discovered_ae = resolver.discover_model(ae_dir, 'vgae', require_exists=False)
                discovered_clf = resolver.discover_model(clf_dir, 'gat', require_exists=False)

                artifacts["autoencoder"] = discovered_ae or (ae_dir / "models" / "vgae_student_autoencoder.pth")
                artifacts["classifier"] = discovered_clf or (clf_dir / "models" / "gat_student_curriculum.pth")
            else:
                # Teacher fusion needs teacher VGAE and teacher GAT
                ae_dir = Path(self.experiment_root) / self.modality / self.dataset.name / "unsupervised" / "vgae" / "teacher" / "no_distillation" / "autoencoder"
                clf_dir = Path(self.experiment_root) / self.modality / self.dataset.name / "supervised" / "gat" / "teacher" / "no_distillation" / "curriculum"
                
                discovered_ae = resolver.discover_model(ae_dir, 'vgae', require_exists=False)
                discovered_clf = resolver.discover_model(clf_dir, 'gat', require_exists=False)
                
                artifacts["autoencoder"] = discovered_ae or (ae_dir / "models" / "vgae_autoencoder.pth")
                artifacts["classifier"] = discovered_clf or (clf_dir / "models" / "gat_curriculum.pth")

        if self.training.mode == "curriculum":
            # Curriculum requires the VGAE teacher - use discovery
            vgae_dir = Path(self.experiment_root) / self.modality / self.dataset.name / "unsupervised" / "vgae" / "teacher" / "no_distillation" / "autoencoder"
            discovered = resolver.discover_model(vgae_dir, 'vgae', require_exists=False)
            artifacts["vgae"] = discovered or (vgae_dir / "models" / "vgae_autoencoder.pth")

        return artifacts

    def get_effective_safety_factor(self) -> float:
        """
        Get the effective batch size safety factor for this configuration.

        Returns either:
        1. Adaptive factor from database (learned from previous runs) if enabled
        2. Static fallback factor from training config

        The adaptive system learns optimal factors for each dataset×model×mode
        combination by monitoring actual memory usage and adjusting with momentum
        targeting 90% GPU utilization.

        Returns:
            Safety factor (0.3 - 0.8 range)

        Example:
            >>> config = store.create_config('gat', 'hcrl_ch', 'normal')
            >>> factor = config.get_effective_safety_factor()
            >>> # Returns adaptive factor like 0.55 (learned) or 0.6 (fallback)
        """
        # Check if adaptive mode is enabled
        use_adaptive = getattr(self.training, 'use_adaptive_batch_size_factor', True)

        if use_adaptive:
            try:
                # Import here to avoid circular dependency
                from src.training.memory_monitor_callback import get_adaptive_safety_factor
                return get_adaptive_safety_factor(self)
            except Exception as e:
                # Fallback to static if adaptive fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to get adaptive safety factor: {e}")
                logger.warning("Falling back to static factor")

        # Use static factor from config
        static_factor = getattr(self.training, 'graph_memory_safety_factor', 0.6)
        return static_factor



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
        """Get model configuration by type.

        Returns a new instance of the requested model config. Raises ValueError for unknown types.
        """
        model_map = {
            "gat": GATConfig,
            "gat_student": StudentGATConfig,
            "vgae": VGAEConfig,
            "vgae_student": StudentVGAEConfig,
            "dqn": DQNConfig,
            "dqn_student": StudentDQNConfig,
        }

        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")

        # Return a fresh instance (avoids accidental shared mutable state)
        return model_map[model_type]()
    
    def get_dataset_config(self, dataset_name: str) -> CANDatasetConfig:
        """Get dataset configuration by name."""
        # CORRECT PATHS: data/automotive/{dataset}
        dataset_configs = {
            "hcrl_sa": CANDatasetConfig(name="hcrl_sa", data_path="data/automotive/hcrl_sa"),
            "hcrl_ch": CANDatasetConfig(name="hcrl_ch", data_path="data/automotive/hcrl_ch"),
            "set_01": CANDatasetConfig(name="set_01", data_path="data/automotive/set_01"),
            "set_02": CANDatasetConfig(name="set_02", data_path="data/automotive/set_02"),
            "set_03": CANDatasetConfig(name="set_03", data_path="data/automotive/set_03"),
            "set_04": CANDatasetConfig(name="set_04", data_path="data/automotive/set_04")
        }
        
        if dataset_name not in dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_configs.keys())}")
        
        return dataset_configs[dataset_name]
    
    def get_training_config(self, training_mode: str):
        """Get training configuration by mode name."""
        training_map = {
            "normal": NormalTrainingConfig,
            "autoencoder": AutoencoderTrainingConfig,
            "knowledge_distillation": KnowledgeDistillationConfig,
            "student_baseline": StudentBaselineTrainingConfig,
            "fusion": FusionTrainingConfig,
            "curriculum": CurriculumTrainingConfig,
            "evaluation": EvaluationTrainingConfig,
        }
        if training_mode not in training_map:
            raise ValueError(f"Unknown training mode: {training_mode}. Available: {list(training_map.keys())}")
        return training_map[training_mode]()
    


# ---------------------------------------------------------------------------
# Convenience factory helpers for presets (used by CLI and tests)
# ---------------------------------------------------------------------------

def create_curriculum_config(dataset: str, vgae_model_path: Optional[str] = None, **overrides) -> CANGraphConfig:
    """Create curriculum learning config for GAT with VGAE-guided hard mining.
    
    Uses glob-based discovery to find the VGAE model if not explicitly provided.
    """
    from src.paths import PathResolver
    
    store = CANGraphConfigStore()
    cfg = store.create_config(model_type="gat", dataset_name=dataset, training_mode="curriculum", **overrides)
    
    # Set VGAE path if provided, otherwise use discovery-based resolution
    if vgae_model_path:
        cfg.training.vgae_model_path = vgae_model_path
    else:
        # Try to discover existing VGAE model using glob patterns
        vgae_dir = Path(cfg.experiment_root) / cfg.modality / cfg.dataset.name / "unsupervised" / "vgae" / "teacher" / "no_distillation" / "autoencoder"
        resolver = PathResolver(cfg)
        discovered = resolver.discover_model(vgae_dir, 'vgae', require_exists=False)
        
        if discovered:
            cfg.training.vgae_model_path = str(discovered)
        else:
            # Fallback to canonical path (will be validated at runtime)
            cfg.training.vgae_model_path = str(vgae_dir / "models" / "vgae_autoencoder.pth")
    
    return cfg


"""
Example adapter code for train_with_hydra_zen.py to support Snakemake.

This shows how to modify your training script to accept simple CLI arguments
from Snakemake while maintaining backward compatibility with the frozen config
system used by can-train.

Add this code to your train_with_hydra_zen.py file.
"""

import argparse
import sys
from pathlib import Path


def create_snakemake_parser():
    """
    Create simple argument parser for Snakemake invocation.

    This parser accepts straightforward CLI arguments that Snakemake can easily
    pass from the Snakefile, avoiding the complex bucket-based config system.
    """
    parser = argparse.ArgumentParser(
        description='Train GNN models (Snakemake-compatible interface)'
    )

    # Core configuration
    parser.add_argument(
        '--model',
        required=True,
        choices=['vgae', 'gat', 'dqn'],
        help='Model architecture'
    )
    parser.add_argument(
        '--model-size',
        required=True,
        choices=['teacher', 'student'],
        help='Model size variant'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset name (e.g., hcrl_sa, hcrl_ch)'
    )
    parser.add_argument(
        '--modality',
        required=True,
        choices=['automotive', 'industrial', 'robotics'],
        help='Application domain'
    )
    parser.add_argument(
        '--training',
        required=True,
        choices=['autoencoder', 'curriculum', 'fusion', 'normal'],
        help='Training strategy/mode'
    )
    parser.add_argument(
        '--learning-type',
        required=True,
        choices=['supervised', 'unsupervised', 'rl_fusion'],
        help='Learning paradigm'
    )

    # Paths for special modes
    parser.add_argument(
        '--teacher-path',
        type=Path,
        default=None,
        help='Path to teacher model for knowledge distillation'
    )
    parser.add_argument(
        '--vgae-path',
        type=Path,
        default=None,
        help='Path to VGAE model for fusion'
    )
    parser.add_argument(
        '--gat-path',
        type=Path,
        default=None,
        help='Path to GAT model for fusion'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for models and logs'
    )

    # Optional hyperparameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')

    return parser


def build_config_from_simple_args(args):
    """
    Build full config object from simple Snakemake arguments.

    This function translates the simple CLI arguments into whatever config
    structure your training code expects. Adapt this to match your actual
    config structure.

    Args:
        args: Parsed arguments from create_snakemake_parser()

    Returns:
        Config object suitable for train() function
    """
    # Example structure - adapt to your actual config format
    config = {
        'model': {
            'type': args.model,
            'size': args.model_size,
        },
        'dataset': {
            'name': args.dataset,
            'modality': args.modality,
        },
        'training': {
            'mode': args.training,
            'learning_type': args.learning_type,
        },
        'paths': {
            'output_dir': args.output_dir,
        },
    }

    # Add distillation config if teacher path provided
    if args.teacher_path:
        config['distillation'] = {
            'enabled': True,
            'teacher_path': args.teacher_path,
        }

    # Add fusion paths if provided
    if args.vgae_path and args.gat_path:
        config['fusion'] = {
            'vgae_path': args.vgae_path,
            'gat_path': args.gat_path,
        }

    # Add optional hyperparameters
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    return config


def main():
    """
    Main entry point with dual-mode support.

    Detects whether called from:
    1. can-train CLI (legacy) → use frozen config
    2. Snakemake (new) → use simple args
    """

    # Mode 1: Legacy frozen config path (from can-train)
    if '--frozen-config' in sys.argv:
        print("Running in LEGACY mode (frozen config from can-train)")
        # Your existing frozen config loading code
        # Example:
        # from src.config.frozen_config import load_frozen_config
        # config_path = sys.argv[sys.argv.index('--frozen-config') + 1]
        # config = load_frozen_config(config_path)
        # train(config)
        pass

    # Mode 2: New Snakemake mode (simple CLI args)
    else:
        print("Running in SNAKEMAKE mode (simple CLI arguments)")
        parser = create_snakemake_parser()
        args = parser.parse_args()

        # Convert simple args to config
        config = build_config_from_simple_args(args)

        # Run training with config
        # train(config)  # Your actual training function
        print(f"Would train with config: {config}")


if __name__ == '__main__':
    main()


# ============================================================================
# Integration Instructions
# ============================================================================
"""
To integrate this into your existing train_with_hydra_zen.py:

1. Copy create_snakemake_parser() function
2. Copy build_config_from_simple_args() function (adapt to your config format!)
3. Modify your main() function to add the dual-mode detection:

    def main():
        if '--frozen-config' in sys.argv:
            # Existing frozen config path
            run_with_frozen_config()
        else:
            # New Snakemake path
            parser = create_snakemake_parser()
            args = parser.parse_args()
            config = build_config_from_simple_args(args)
            train(config)

4. Test both modes:

   # Old way (should still work)
   python train_with_hydra_zen.py --frozen-config /path/to/config.json

   # New way (for Snakemake)
   python train_with_hydra_zen.py \
       --model vgae \
       --model-size teacher \
       --dataset hcrl_sa \
       --modality automotive \
       --training autoencoder \
       --learning-type unsupervised \
       --output-dir test_output

5. Verify model outputs:
   - Models saved to: {output_dir}/models/{model}_{size}_{distillation}_best.pth
   - Checkpoints saved to: {output_dir}/checkpoints/last.ckpt
   - Logs saved to: {output_dir}/logs/training.log

That's it! Your training script now works with both systems.
"""


# Initialize the global store
config_store = CANGraphConfigStore()