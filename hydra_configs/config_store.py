# ============================================================================
# KD-GAT Hydra-Zen Configuration Store
# Type-safe, reproducible experiment configuration for all model variations
# ============================================================================

from hydra_zen import Store, field
from dataclasses import dataclass, field as dc_field
from typing import Literal, Optional
import torch

# ============================================================================
# MODEL ARCHITECTURE CONFIGS
# ============================================================================

@dataclass
class VAEConfig:
    """VGAE (Variational Graph AutoEncoder) configuration"""
    _target_: str = "src.models.vgae.VGAE"
    hidden_dim: int = 64
    latent_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"

@dataclass
class GATConfig:
    """Graph Attention Network configuration"""
    _target_: str = "src.models.gat.GAT"
    input_dim: int = 128  # Will be set by dataset
    hidden_dim: int = 128
    output_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"
    concat: bool = True

@dataclass
class DQNConfig:
    """Deep Q-Network configuration"""
    _target_: str = "src.models.dqn.DQN"
    input_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    action_dim: int = 10  # Set per dataset
    dropout: float = 0.1
    dueling: bool = True
    double_q: bool = True

# ============================================================================
# TEACHER/STUDENT MODEL SIZES
# ============================================================================

@dataclass
class TeacherModelSize:
    """Teacher model: larger capacity"""
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.05

@dataclass
class StudentModelSize:
    """Student model: compressed for deployment"""
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1

@dataclass
class IntermediateModelSize:
    """Intermediate size: balance between teacher and student"""
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.08

@dataclass
class HugeModelSize:
    """Large capacity model for maximum accuracy"""
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.05

@dataclass
class TinyModelSize:
    """Minimal size for edge deployment"""
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.15

# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================

@dataclass
class TrainingConfig:
    """Base training configuration"""
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: Optional[str] = "cosine"  # cosine, linear, exponential, None
    warmup_epochs: int = 5
    gradient_clip: Optional[float] = 1.0
    early_stopping_patience: int = 20
    early_stopping_metric: str = "val_loss"

@dataclass
class AllSamplesTraining(TrainingConfig):
    """Train on all available samples"""
    name: str = "all_samples"
    epochs: int = 100

@dataclass
class NormalsOnlyTraining(TrainingConfig):
    """Train only on normal/benign samples"""
    name: str = "normals_only"
    epochs: int = 80
    learning_rate: float = 1e-3

@dataclass
class CurriculumClassifierTraining(TrainingConfig):
    """Curriculum learning schedule for classification"""
    name: str = "curriculum_classifier"
    epochs: int = 120
    curriculum_steps: int = 4
    difficulty_ramp: float = 1.5  # Increase difficulty multiplier per curriculum step

@dataclass
class CurriculumFusionTraining(TrainingConfig):
    """Curriculum learning schedule for fusion models"""
    name: str = "curriculum_fusion"
    epochs: int = 150
    curriculum_steps: int = 5
    fusion_weight_schedule: str = "linear"  # linear, exponential, step

# ============================================================================
# DISTILLATION CONFIGURATIONS
# ============================================================================

@dataclass
class NoDistillation:
    """Standard training without knowledge distillation"""
    _target_: str = "src.training.distillation.NoDistillation"
    name: str = "no"

@dataclass
class StandardDistillation:
    """Standard knowledge distillation (KD)"""
    _target_: str = "src.training.distillation.StandardDistillation"
    name: str = "standard"
    temperature: float = 3.0
    alpha: float = 0.7  # Weight between KD loss and standard loss
    kd_loss_type: str = "kl_divergence"  # kl_divergence, mse

@dataclass
class TopologyPreservingDistillation:
    """Topology-preserving distillation for graph models"""
    _target_: str = "src.training.distillation.TopologyPreservingDistillation"
    name: str = "topology_preserving"
    temperature: float = 4.0
    alpha: float = 0.6
    beta: float = 0.4  # Weight for topology preservation loss

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

@dataclass
class HCRLCHDataset:
    """HCRL-CH CAN bus dataset"""
    _target_: str = "src.data.datasets.HCRLCHDataset"
    name: str = "hcrlch"
    modality: str = "automotive"
    data_path: str = "${oc.select:data_root}/automotive/hcrlch"
    split_ratio: tuple = (0.7, 0.15, 0.15)
    normalization: str = "zscore"

@dataclass
class Set01Dataset:
    """Set01 CAN bus dataset"""
    _target_: str = "src.data.datasets.Set01Dataset"
    name: str = "set01"
    modality: str = "automotive"
    data_path: str = "${oc.select:data_root}/automotive/set01"
    split_ratio: tuple = (0.7, 0.15, 0.15)
    normalization: str = "zscore"

@dataclass
class Set02Dataset:
    """Set02 CAN bus dataset"""
    _target_: str = "src.data.datasets.Set02Dataset"
    name: str = "set02"
    modality: str = "automotive"
    data_path: str = "${oc.select:data_root}/automotive/set02"
    split_ratio: tuple = (0.7, 0.15, 0.15)
    normalization: str = "zscore"

# ============================================================================
# LEARNING TYPE CONFIGURATIONS
# ============================================================================

@dataclass
class UnsupervisedConfig:
    """Unsupervised learning (autoencoder-based)"""
    learning_type: str = "unsupervised"
    reconstruction_loss: str = "mse"
    kl_weight: float = 0.0001  # For variational autoencoders

@dataclass
class ClassifierConfig:
    """Supervised classification"""
    learning_type: str = "classifier"
    loss_function: str = "cross_entropy"
    class_weights: Optional[list] = None

@dataclass
class FusionConfig:
    """Multi-modal fusion learning (e.g., structural + traffic features)"""
    learning_type: str = "fusion"
    fusion_strategy: str = "concatenation"  # concatenation, attention, bilinear
    loss_function: str = "cross_entropy"
    fusion_weight: float = 0.5

# ============================================================================
# EXPERIMENT CONFIGURATION (Root Config)
# ============================================================================

@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration combining all components.
    Follows hierarchical structure:
    L1: modality (automotive, internet, watertreatment)
    L2: dataset (hcrlch, set01, set02, set03, set04)
    L3: learning_type (unsupervised, classifier, fusion)
    L4: model_architecture (VGAE, GAT, DQN)
    L5: model_size (teacher, student, intermediate, huge, tiny)
    L6: distillation (no, standard, topology_preserving)
    L7: training_type (all_samples, normals_only, curriculum_classifier, curriculum_fusion)
    """

    # Paths and logging
    project_root: str = "/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT"
    data_root: str = "${project_root}/data"
    experiment_root: str = "${project_root}/experiment_runs"
    
    # Core experiment config
    modality: Literal["automotive"] = "automotive"
    dataset: Literal["hcrlch", "set01", "set02", "set03", "set04"] = "hcrlch"
    learning_type: Literal["unsupervised", "classifier", "fusion"] = "unsupervised"
    model_architecture: Literal["VGAE", "GAT", "DQN"] = "VGAE"
    model_size: Literal["teacher", "student", "intermediate", "huge", "tiny"] = "student"
    distillation: Literal["no", "standard", "topology_preserving"] = "no"
    training_mode: Literal["all_samples", "normals_only", "curriculum_classifier", "curriculum_fusion"] = "all_samples"
    
    # Component configs
    dataset_config: dict = dc_field(default_factory=dict)  # Populated by store
    model_config: dict = dc_field(default_factory=dict)    # Populated by store
    model_size_config: dict = dc_field(default_factory=dict)  # Populated by store
    training_config: TrainingConfig = dc_field(default_factory=AllSamplesTraining)
    distillation_config: dict = dc_field(default_factory=dict)  # Populated by store
    learning_config: dict = dc_field(default_factory=dict)  # Populated by store
    
    # Runtime settings
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    # Logging and checkpointing
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1
    save_top_k: int = 3
    checkpoint_dir: str = "${experiment_path}/checkpoints"
    log_dir: str = "${experiment_path}/logs"
    
    # MLflow settings
    mlflow_experiment_name: str = "${dataset}_${learning_type}_${model_architecture}"
    mlflow_run_name: str = "${model_size}_${distillation}_${training_mode}"
    mlflow_tracking_uri: str = "${experiment_root}/.mlruns"


# ============================================================================
# OSC/SLURM SETTINGS
# ============================================================================

@dataclass
class OSCSettings:
    """Ohio Supercomputer Center (OSC) Slurm submission settings"""
    account: str = "PAS3209"
    email: str = "frenken.2@osu.edu"
    notification_type: str = "END,FAIL"  # Also: BEGIN, REQUEUE, ALL
    project_root: str = "/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT"
    conda_env: str = "gnn-gpu"
    submit_host: str = "owens.osc.edu"
    
    # Slurm job parameters
    walltime: str = "02:00:00"
    memory: str = "32G"
    cpus_per_task: int = 8
    gpus_per_node: int = 1
    gpu_type: str = "v100"  # v100, a100, rtx6000
    
    # Optional notifications
    slack_webhook: Optional[str] = None
    notification_email: Optional[str] = None

@dataclass
class LocalSettings:
    """Local machine execution settings"""
    project_root: str = "/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT"
    conda_env: str = "gnn-gpu"
    gpus_per_node: int = 1  # Local GPU count

# ============================================================================
# BUILD THE STORE
# ============================================================================

store = Store()

# Register base configs
store(ExperimentConfig, name="base_experiment")
store(OSCSettings, name="osc_settings")
store(LocalSettings, name="local_settings")

# ============================================================================
# STORE ALL EXPERIMENT COMBINATIONS
# ============================================================================

def create_experiment_configs(store: Store):
    """
    Programmatically generate all valid experiment combinations.
    This ensures consistency and prevents configuration mismatches.
    """
    
    # Define all possible values at each level
    modalities = ["automotive"]
    datasets = {
        "automotive": ["hcrlch", "set01", "set02", "set03", "set04"]
    }
    learning_types = {
        "unsupervised": ["VGAE"],
        "classifier": ["GAT", "DQN"],
        "fusion": ["GAT", "DQN"]
    }
    model_sizes = ["teacher", "student", "intermediate", "huge", "tiny"]
    distillations = ["no", "standard", "topology_preserving"]
    training_modes = {
        "unsupervised": ["all_samples", "normals_only"],
        "classifier": ["all_samples", "normals_only", "curriculum_classifier"],
        "fusion": ["all_samples", "curriculum_fusion"]
    }
    
    # Model architecture configs
    model_configs = {
        "VGAE": VAEConfig,
        "GAT": GATConfig,
        "DQN": DQNConfig
    }
    
    # Model size configs
    size_configs = {
        "teacher": TeacherModelSize,
        "student": StudentModelSize,
        "intermediate": IntermediateModelSize,
        "huge": HugeModelSize,
        "tiny": TinyModelSize
    }
    
    # Distillation configs
    distill_configs = {
        "no": NoDistillation,
        "standard": StandardDistillation,
        "topology_preserving": TopologyPreservingDistillation
    }
    
    # Dataset configs
    dataset_configs = {
        "hcrlch": HCRLCHDataset,
        "set01": Set01Dataset,
        "set02": Set02Dataset
    }
    
    # Training mode configs
    training_configs = {
        "all_samples": AllSamplesTraining,
        "normals_only": NormalsOnlyTraining,
        "curriculum_classifier": CurriculumClassifierTraining,
        "curriculum_fusion": CurriculumFusionTraining
    }
    
    # Learning type configs
    learning_configs = {
        "unsupervised": UnsupervisedConfig,
        "classifier": ClassifierConfig,
        "fusion": FusionConfig
    }
    
    # Generate all valid combinations
    config_count = 0
    for modality in modalities:
        for dataset in datasets[modality]:
            for learning_type in learning_types:
                for model_arch in learning_types[learning_type]:
                    for model_size in model_sizes:
                        for distillation in distillations:
                            for training_mode in training_modes[learning_type]:
                                
                                config_name = (
                                    f"{modality}_{dataset}_{learning_type}_"
                                    f"{model_arch}_{model_size}_{distillation}_{training_mode}"
                                )
                                
                                # Build the config
                                cfg = ExperimentConfig(
                                    modality=modality,
                                    dataset=dataset,
                                    learning_type=learning_type,
                                    model_architecture=model_arch,
                                    model_size=model_size,
                                    distillation=distillation,
                                    training_mode=training_mode,
                                    model_config=model_configs[model_arch](),
                                    model_size_config=size_configs[model_size](),
                                    distillation_config=distill_configs[distillation](),
                                    dataset_config=dataset_configs.get(dataset, HCRLCHDataset)(),
                                    training_config=training_configs[training_mode](),
                                    learning_config=learning_configs[learning_type]()
                                )
                                
                                store(cfg, name=config_name)
                                config_count += 1
    
    print(f"✅ Generated {config_count} valid experiment configurations")
    return config_count

# Generate all configs when module is imported
create_experiment_configs(store)

# ============================================================================
# HELPER FUNCTION: Get all available configs
# ============================================================================

def get_available_configs() -> dict:
    """Return dictionary of all available configuration combinations"""
    # This would be populated from the store
    # In actual usage, iterate through store.repo
    pass

if __name__ == "__main__":
    print("Hydra-Zen Configuration Store for KD-GAT")
    print("=" * 60)
    print("Module ready for use with @hydra_zen.main()")
