"""
Hydra-Zen Configuration System for CAN-Graph

This module replaces the YAML configuration files with type-safe Python configurations
using hydra-zen. Benefits:
- Type safety and IDE support
- Programmatic config generation
- Reduced file clutter
- Better validation
- Dynamic configuration composition

Looking to change the configuration to the new pathing system:

        self.osc_settings = {
            "account": "PAS3209",  # Your account
            "email": "frenken.2@osu.edu",  # Your email (used for SBATCH notifications if enabled)
            "project_path": str(self.project_root),  # Project path (keeps previous behaviour but derived)
            "conda_env": "gnn-gpu",  # Your conda environment
            "notify_webhook": "",  # Optional: Slack/Teams webhook URL for concise completion notifications
            "notify_email": "frenken.2@osu.edu"     # Optional: single email to receive job completion summaries (one per job)
        }

        self.osc_parameters = {
            "wall_time": "02:00:00",
            "memory": "32G",
            "cpus": 8,
            "gpus": 1
        }

        self.training_configurations = {
            "modalities": ["automotive", "internet", "water_treatment"],
            # will need to handle the pathing based on modalities in future
            # right now automotive is the only modality used
            "datasets": {"automotive": ["hcrl_sa", "hcrl_ch", "set_01", "set_02", "set_03", "set_04"],
                         "internet": [],
                         "water_treatment": []},
            "learning_types": ["unsupervised", "supervised", "rl_fusion"],
            "model_architectures": {"unsupervised": ["vgae"], "supervised": ["gat"], "rl_fusion": ["dqn"]},
            # right now small = student and teacher = large with options to expand
            "model_sizes": ["small", "medium", "large"],
            "distillation": ["no", "standard"],
            "training_modes": ["all_samples", "normals_only","curriculum_classifier", "curriculum_fusion"],
        }

"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from hydra_zen import make_config, store, zen
import torch
import logging

logger = logging.getLogger(__name__)


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
    experiment_root: str = "experiment_runs"
    cache_processed_data: bool = True
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "cache_processed_data": True,
        "normalize_features": False,
        "feature_scaling": "standard"
    })

    def __post_init__(self):
        # If a data_path is not supplied, infer it from the standard datasets layout
        if not self.data_path:
            self.data_path = f"datasets/can-train-and-test-v1.5/{self.name}"

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
    
    # Global settings
    seed: int = 42
    project_name: str = "can_graph_lightning"
    experiment_name: str = field(init=False)

    # Canonical experiment layout root (will be created by training when needed)
    experiment_root: str = "experiment_runs"
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
        "enable_tensorboard": False,
        "save_top_k": 3,
        "monitor_metric": "val_loss",
        "monitor_mode": "min"
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
            {experiment_root}/{modality}/{dataset}/{learning_type}/{model_arch}/{model_size}/{distillation}/{training_mode}/
        """
        # Determine base model architecture (e.g., 'vgae' from 'vgae_student')
        raw_type = getattr(self.model, "type", "unknown")
        model_arch = raw_type.replace('_student', '').replace('_teacher', '')

        # Determine model_size explicitly if type contains student/teacher hints
        model_size = self.model_size or ("student" if "student" in raw_type else "teacher")

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
        `training.mode`. The caller should check for existence and fail if missing.
        """
        artifacts = {}
        exp_dir = self.canonical_experiment_dir()

        if self.training.mode == "knowledge_distillation":
            # Teacher model must exist and be specified
            teacher_path = getattr(self.training, "teacher_model_path", None)
            if teacher_path:
                artifacts["teacher_model"] = Path(teacher_path)
            else:
                artifacts["teacher_model"] = exp_dir / "teacher" / f"best_teacher_model_{self.dataset.name}.pth"

        # Use canonical absolute locations (consistent with canonical_experiment_dir grammar)
        if self.training.mode == "fusion":
            # VGAE autoencoder (unsupervised teacher)
            ae_dir = Path(self.experiment_root) / self.modality / self.dataset.name / "unsupervised" / "vgae" / "teacher" / self.distillation / "autoencoder"
            artifacts["autoencoder"] = ae_dir / "vgae_autoencoder.pth"

            # Classifier (supervised GAT, normal training)
            clf_dir = Path(self.experiment_root) / self.modality / self.dataset.name / "supervised" / "gat" / "teacher" / self.distillation / "normal"
            artifacts["classifier"] = clf_dir / f"gat_{self.dataset.name}_normal.pth"

        if self.training.mode == "curriculum":
            # Curriculum requires the VGAE teacher saved under the canonical unsupervised path
            vgae_dir = Path(self.experiment_root) / self.modality / self.dataset.name / "unsupervised" / "vgae" / "teacher" / self.distillation / "autoencoder"
            artifacts["vgae"] = vgae_dir / "vgae_autoencoder.pth"

        return artifacts



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
        self.store(EvaluationTrainingConfig, name="evaluation", group="training")
        
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
    


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config(config: CANGraphConfig) -> bool:
    """Validate configuration for common issues.

    This validator is strict (no fallbacks). If a training mode requires pre-existing
    artifacts the validator raises a descriptive error (rather than silently falling back).

    Additionally, performs a lightweight *sanity check* on dataset paths and prints
    helpful warnings (does not raise) so developers can see canonical expected paths.
    """
    issues = []

    # Lightweight sanity check: dataset path existence (warn only, don't raise)
    try:
        ds_path = getattr(config.dataset, 'data_path', None)
        if ds_path and not Path(ds_path).exists():
            logger.warning(
                f"Dataset path does not exist: {ds_path}\n"
                f"  → Canonical experiment dir (where outputs will be written): {config.canonical_experiment_dir()}\n"
                f"  Suggestion: set config.dataset.data_path to the correct local path or provide --data-path when running the smoke script."
            )
    except Exception:
        # Be permissive for unexpected config shapes; the strict validations follow below
        logger.debug("Dataset path sanity check skipped due to unexpected config structure")

    # Check knowledge distillation requirements strictly
    if config.training.mode == "knowledge_distillation":
        teacher_path = getattr(config.training, "teacher_model_path", None)
        if not teacher_path:
            raise ValueError("Knowledge distillation requires 'teacher_model_path' in the training config.")
        if not Path(teacher_path).exists():
            raise FileNotFoundError(f"Teacher model not found at specified path: {teacher_path}. Please provide the canonical path under experiment_runs and ensure it exists.")

    # Fusion requires both autoencoder and classifier artifacts
    if config.training.mode == "fusion":
        artifacts = config.required_artifacts()
        missing = []
        for name, p in artifacts.items():
            if not p.exists():
                missing.append(f"{name} missing at {p}")
        if missing:
            raise FileNotFoundError("Fusion training requires pre-trained artifacts:\n" + "\n".join(missing) + "\nPlease train and save the required models under the canonical experiment directory.")

    # Curriculum also requires the VGAE to exist
    if config.training.mode == "curriculum":
        artifacts = config.required_artifacts()
        vgae = artifacts.get("vgae")
        if vgae and not vgae.exists():
            raise FileNotFoundError(f"Curriculum training requires VGAE model at {vgae}. Please ensure it's available under experiment_runs.")

    # Additional lightweight checks
    if getattr(config, 'modality', None) is None:
        issues.append("Modality should be set (e.g., 'automotive')")

    if issues:
        for i in issues:
            logger.error(i)
        return False

    # Check dataset path - warn only (allow iterative development without immediate data present)
    if config.dataset.data_path and not Path(config.dataset.data_path).exists():
        logger.warning(f"Dataset path not found (warning only): {config.dataset.data_path} - training may fail if the dataset is not provided.")

    # Check precision compatibility strictly
    if config.training.precision == "16-mixed" and config.trainer.precision != "16-mixed":
        raise ValueError("Training precision and trainer.precision should both be '16-mixed' when using mixed precision training.")

    logger.info("✅ Configuration validation passed")
    return True


# Initialize the global store
config_store = CANGraphConfigStore()