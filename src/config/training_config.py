"""
Training Configuration Management

Centralized configuration for GAT, VGAE, and Fusion training pipelines.
Includes dataset paths, model parameters, and training hyperparameters.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(r"c:\Users\User1\Documents\GitHub\CAN-Graph")
DATASET_BASE = BASE_DIR / "datasets"

# Dataset paths
DATASET_PATHS = {
    'CAN_DATASET': DATASET_BASE / "can-dataset",
    'CAN_TRAIN_TEST_V15': DATASET_BASE / "can-train-and-test-v1.5",
    'CAR_HACKING': DATASET_BASE / "CAR-Hacking Dataset",
    'OTIDS': DATASET_BASE / "OTIDS"
}

# Model save paths
MODEL_PATHS = {
    'SAVED_MODELS': BASE_DIR / "saved_models",
    'OUTPUT_MODEL_1': BASE_DIR / "output_model_1",
    'SAVED_MODELS_1': BASE_DIR / "saved_models1",
    'FUSION_CHECKPOINTS': BASE_DIR / "saved_models" / "fusion_checkpoints"
}

# Output directories
OUTPUT_PATHS = {
    'OUTPUTS': BASE_DIR / "outputs",
    'PUBLICATION_FIGURES': BASE_DIR / "publication_figures",
    'VISUALS': BASE_DIR / "visuals"
}

# Training hyperparameters by model type
TRAINING_CONFIG = {
    'FUSION': {
        'num_episodes': 500,
        'replay_buffer_size': 100000,
        'batch_size': 8192,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 10,
        'fusion_weights': {
            'vgae_weight': 0.4,
            'gat_weight': 0.6,
            'combined_threshold': 0.5
        }
    },
    'GAT': {
        'num_epochs': 100,
        'batch_size': 2048,
        'learning_rate': 0.001,
        'hidden_channels': 128,
        'num_heads': 8,
        'dropout': 0.2,
        'early_stopping_patience': 10
    },
    'VGAE': {
        'num_epochs': 200,
        'batch_size': 1024,
        'learning_rate': 0.001,
        'hidden_channels': 64,
        'latent_channels': 32,
        'dropout': 0.1,
        'reconstruction_loss_weight': 1.0,
        'kl_loss_weight': 0.1
    }
}

# GPU optimization settings
GPU_CONFIG = {
    'MEMORY_FRACTION': 0.8,
    'ENABLE_BENCHMARKING': True,
    'PIN_MEMORY': True,
    'PERSISTENT_WORKERS': True,
    'PREFETCH_FACTOR': 6,
    'TARGET_GPU_UTILIZATION': 85.0,
    'TARGET_MEMORY_UTILIZATION': 75.0
}

# Dataset specific configurations
DATASET_CONFIG = {
    'GRAPH_SIZE': 100,  # Number of nodes per graph
    'SEQUENCE_LENGTH': 10,  # Number of time steps
    'FEATURE_DIM': 8,  # Number of features per node
    'NUM_CLASSES': 2,  # Binary classification (normal/attack)
    'CLASS_WEIGHTS': {
        'normal': 1.0,
        'attack': 55.0  # To handle class imbalance
    },
    'TRAIN_SPLIT': 0.8,
    'VAL_SPLIT': 0.1,
    'TEST_SPLIT': 0.1
}

# Evaluation metrics
EVALUATION_CONFIG = {
    'METRICS': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
    'THRESHOLD_TUNING': True,
    'CONFUSION_MATRIX': True,
    'CLASSIFICATION_REPORT': True,
    'SAVE_PREDICTIONS': True
}

# Logging and monitoring
LOGGING_CONFIG = {
    'LOG_LEVEL': 'INFO',
    'CONSOLE_LOGGING': True,
    'FILE_LOGGING': True,
    'TENSORBOARD_LOGGING': False,  # Set to True if using TensorBoard
    'WANDB_LOGGING': False,  # Set to True if using Weights & Biases
    'CHECKPOINT_FREQUENCY': 10,  # Save model every N episodes/epochs
    'EVALUATION_FREQUENCY': 5   # Evaluate model every N episodes/epochs
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'SAVE_PLOTS': True,
    'SHOW_PLOTS': False,  # Set to True for interactive display
    'DPI': 300,
    'FIGURE_SIZE': (12, 8),
    'COLOR_PALETTE': 'husl',
    'PLOT_FORMATS': ['png', 'pdf'],
    'PUBLICATION_READY': True
}


def get_dataset_path(dataset_name: str) -> Path:
    """Get the path for a specific dataset."""
    if dataset_name.upper() not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_PATHS[dataset_name.upper()]


def get_model_save_path(model_type: str, experiment_name: str = None) -> Path:
    """Get the save path for a specific model."""
    base_path = MODEL_PATHS['SAVED_MODELS']
    
    if experiment_name:
        return base_path / f"{model_type.lower()}_{experiment_name}"
    else:
        return base_path / f"{model_type.lower()}_model"


def get_output_path(output_type: str = 'OUTPUTS') -> Path:
    """Get the output path for results and figures."""
    if output_type.upper() not in OUTPUT_PATHS:
        raise ValueError(f"Unknown output type: {output_type}")
    return OUTPUT_PATHS[output_type.upper()]


def create_experiment_directory(model_type: str, experiment_name: str) -> Path:
    """Create and return experiment directory."""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    experiment_dir = get_output_path() / f"{model_type.lower()}_{experiment_name}_{timestamp}"
    
    # Create directory structure
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "models").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    
    return experiment_dir


def get_training_config(model_type: str) -> dict:
    """Get training configuration for a specific model type."""
    if model_type.upper() not in TRAINING_CONFIG:
        raise ValueError(f"Unknown model type: {model_type}")
    return TRAINING_CONFIG[model_type.upper()].copy()


def update_config_for_gpu(config: dict, gpu_memory_gb: float) -> dict:
    """Update configuration based on available GPU memory."""
    updated_config = config.copy()
    
    # Adjust batch sizes based on GPU memory
    if gpu_memory_gb >= 30:  # A100 class
        batch_size_multiplier = 2.0
    elif gpu_memory_gb >= 15:  # RTX 4090 class
        batch_size_multiplier = 1.5
    elif gpu_memory_gb >= 8:   # RTX 3080 class
        batch_size_multiplier = 1.0
    else:  # Lower-end GPUs
        batch_size_multiplier = 0.5
    
    if 'batch_size' in updated_config:
        updated_config['batch_size'] = int(updated_config['batch_size'] * batch_size_multiplier)
    
    if 'replay_buffer_size' in updated_config:
        updated_config['replay_buffer_size'] = int(updated_config['replay_buffer_size'] * batch_size_multiplier)
    
    return updated_config


def validate_paths():
    """Validate that all configured paths exist or can be created."""
    issues = []
    
    # Check dataset paths
    for name, path in DATASET_PATHS.items():
        if not path.exists():
            issues.append(f"Dataset path does not exist: {path}")
    
    # Create model directories if they don't exist
    for name, path in MODEL_PATHS.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create model directory {path}: {e}")
    
    # Create output directories if they don't exist
    for name, path in OUTPUT_PATHS.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create output directory {path}: {e}")
    
    if issues:
        print("Configuration validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ All configured paths validated successfully")
        return True


# Environment-specific settings
def get_slurm_config():
    """Get SLURM-specific configuration if running on cluster."""
    slurm_config = {}
    
    if 'SLURM_JOB_ID' in os.environ:
        slurm_config.update({
            'job_id': os.environ.get('SLURM_JOB_ID'),
            'node_list': os.environ.get('SLURM_JOB_NODELIST'),
            'cpus_per_task': int(os.environ.get('SLURM_CPUS_PER_TASK', 6)),
            'mem_per_cpu': os.environ.get('SLURM_MEM_PER_CPU'),
            'partition': os.environ.get('SLURM_JOB_PARTITION'),
            'time_limit': os.environ.get('SLURM_JOB_TIME_LIMIT')
        })
    
    return slurm_config


# Configuration validation and setup
if __name__ == "__main__":
    print("Validating CAN-Graph Training Configuration...")
    validate_paths()
    
    # Print configuration summary
    print(f"\nDataset Configurations:")
    for name, path in DATASET_PATHS.items():
        print(f"  {name}: {path}")
    
    print(f"\nModel Save Paths:")
    for name, path in MODEL_PATHS.items():
        print(f"  {name}: {path}")
    
    print(f"\nGPU Configuration:")
    for key, value in GPU_CONFIG.items():
        print(f"  {key}: {value}")
    
    slurm_info = get_slurm_config()
    if slurm_info:
        print(f"\nSLURM Environment Detected:")
        for key, value in slurm_info.items():
            print(f"  {key}: {value}")