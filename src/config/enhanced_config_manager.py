"""
Enhanced Configuration Management

This module provides centralized configuration management for all training
processes with GPU optimization, resource management, and model-specific settings.

Key Features:
- Unified configuration for all model types and datasets
- Dynamic configuration based on GPU capabilities
- Environment-specific optimization settings
- Configuration validation and recommendations
- Template generation for new datasets
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

import torch
import psutil
from omegaconf import DictConfig, OmegaConf

from src.utils.gpu_utils import detect_gpu_capabilities_unified

@dataclass
class ModelConfig:
    """Configuration for a specific model type."""
    model_type: str  # 'teacher', 'student', 'fusion'
    epochs: int
    learning_rate: float
    batch_size: int
    hidden_dim: int
    dropout_rate: float
    weight_decay: float
    patience: int
    grad_clip_norm: float
    use_scheduler: bool
    scheduler_type: str = 'cosine'
    optimizer_type: str = 'adamw'

@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    dataset_name: str
    dataset_path: str
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    data_size_fraction: float
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GPUConfig:
    """GPU-specific optimization configuration."""
    device_name: str
    memory_gb: float
    compute_capability: str
    optimal_batch_size: int
    max_batch_size: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    mixed_precision: bool
    gradient_accumulation_steps: int

@dataclass
class TrainingEnvironmentConfig:
    """Complete training environment configuration."""
    environment_id: str
    gpu_config: GPUConfig
    model_configs: Dict[str, ModelConfig]
    dataset_configs: Dict[str, DatasetConfig]
    global_settings: Dict[str, Any]
    resource_management: Dict[str, Any]
    created_at: datetime
    created_by: str = "enhanced_config_manager"

class EnhancedConfigManager:
    """
    Centralized configuration management for the training pipeline.
    """
    
    def __init__(self, 
                 config_dir: str = "conf",
                 output_dir: str = "outputs/generated_configs"):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU detection
        if torch.cuda.is_available():
            self.gpu_info = detect_gpu_capabilities_unified()
        else:
            self.gpu_info = {'name': 'CPU', 'memory_gb': 0}
        
        # Default configurations
        self.dataset_paths = {
            'hcrl_ch': "datasets/can-train-and-test-v1.5/hcrl-ch",
            'hcrl_sa': "datasets/can-train-and-test-v1.5/hcrl-sa",
            'set_01': "datasets/can-train-and-test-v1.5/set_01",
            'set_02': "datasets/can-train-and-test-v1.5/set_02",
            'set_03': "datasets/can-train-and-test-v1.5/set_03",
            'set_04': "datasets/can-train-and-test-v1.5/set_04"
        }
        
        print(f"✓ Enhanced Config Manager initialized")
        print(f"  GPU: {self.gpu_info.get('name', 'Unknown')}")
        print(f"  Config directory: {self.config_dir}")
        print(f"  Output directory: {self.output_dir}")
    
    def create_gpu_config(self) -> GPUConfig:
        """Create GPU configuration based on detected hardware."""
        if not torch.cuda.is_available():
            # CPU fallback configuration
            return GPUConfig(
                device_name="CPU",
                memory_gb=psutil.virtual_memory().total / (1024**3),
                compute_capability="N/A",
                optimal_batch_size=256,
                max_batch_size=1024,
                num_workers=psutil.cpu_count(logical=False),
                prefetch_factor=2,
                pin_memory=False,
                mixed_precision=False,
                gradient_accumulation_steps=1
            )
        
        # GPU configuration
        gpu_props = torch.cuda.get_device_properties(0)
        memory_gb = gpu_props.total_memory / (1024**3)
        
        # Determine optimal settings based on GPU memory
        if memory_gb >= 30:  # A100 40GB+
            config = GPUConfig(
                device_name=gpu_props.name,
                memory_gb=memory_gb,
                compute_capability=f"{gpu_props.major}.{gpu_props.minor}",
                optimal_batch_size=8192,
                max_batch_size=16384,
                num_workers=24,
                prefetch_factor=6,
                pin_memory=True,
                mixed_precision=True,
                gradient_accumulation_steps=1
            )
        elif memory_gb >= 20:  # RTX 3090/4090
            config = GPUConfig(
                device_name=gpu_props.name,
                memory_gb=memory_gb,
                compute_capability=f"{gpu_props.major}.{gpu_props.minor}",
                optimal_batch_size=4096,
                max_batch_size=8192,
                num_workers=16,
                prefetch_factor=4,
                pin_memory=True,
                mixed_precision=True,
                gradient_accumulation_steps=1
            )
        elif memory_gb >= 10:  # RTX 3080/4070
            config = GPUConfig(
                device_name=gpu_props.name,
                memory_gb=memory_gb,
                compute_capability=f"{gpu_props.major}.{gpu_props.minor}",
                optimal_batch_size=2048,
                max_batch_size=4096,
                num_workers=12,
                prefetch_factor=3,
                pin_memory=True,
                mixed_precision=True,
                gradient_accumulation_steps=2
            )
        else:  # Smaller GPUs
            config = GPUConfig(
                device_name=gpu_props.name,
                memory_gb=memory_gb,
                compute_capability=f"{gpu_props.major}.{gpu_props.minor}",
                optimal_batch_size=1024,
                max_batch_size=2048,
                num_workers=8,
                prefetch_factor=2,
                pin_memory=True,
                mixed_precision=True,
                gradient_accumulation_steps=4
            )
        
        return config
    
    def create_model_configs(self, gpu_config: GPUConfig) -> Dict[str, ModelConfig]:
        """Create optimized model configurations."""
        # Base learning rate adjusted for GPU capability
        base_lr = 0.001
        if gpu_config.memory_gb >= 20:
            base_lr = 0.0005  # Lower LR for larger batches
        
        # Teacher model configuration
        teacher_config = ModelConfig(
            model_type="teacher",
            epochs=10,
            learning_rate=base_lr,
            batch_size=gpu_config.optimal_batch_size,
            hidden_dim=256,
            dropout_rate=0.1,
            weight_decay=1e-5,
            patience=20,
            grad_clip_norm=1.0,
            use_scheduler=True,
            scheduler_type='cosine',
            optimizer_type='adamw'
        )
        
        # Student model configuration (knowledge distillation)
        student_config = ModelConfig(
            model_type="student",
            epochs=12,  # More epochs for distillation
            learning_rate=base_lr * 1.2,  # Slightly higher for KD
            batch_size=gpu_config.optimal_batch_size,
            hidden_dim=128,  # Compressed model
            dropout_rate=0.15,
            weight_decay=1e-5,
            patience=25,
            grad_clip_norm=1.0,
            use_scheduler=True,
            scheduler_type='cosine',
            optimizer_type='adamw'
        )
        
        # Fusion model configuration (DQN)
        fusion_config = ModelConfig(
            model_type="fusion",
            epochs=300,  # Episodes for DQN
            learning_rate=0.0001,
            batch_size=min(32768, gpu_config.max_batch_size),  # Large batches for DQN
            hidden_dim=512,
            dropout_rate=0.0,  # No dropout in DQN
            weight_decay=0,
            patience=50,
            grad_clip_norm=1.0,
            use_scheduler=False,
            optimizer_type='adam'  # Adam for DQN
        )
        
        return {
            'teacher': teacher_config,
            'student': student_config,
            'fusion': fusion_config
        }
    
    def create_dataset_configs(self) -> Dict[str, DatasetConfig]:
        """Create dataset configurations for all available datasets."""
        configs = {}
        
        for dataset_name, dataset_path in self.dataset_paths.items():
            config = DatasetConfig(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                train_ratio=0.8,
                validation_ratio=0.15,
                test_ratio=0.05,
                data_size_fraction=1.0,  # Use full dataset by default
                preprocessing_params={
                    'normalize': True,
                    'handle_missing': 'interpolate',
                    'feature_scaling': 'standard',
                    'graph_creation_method': 'temporal'
                }
            )
            configs[dataset_name] = config
        
        return configs
    
    def create_complete_environment_config(self, 
                                         environment_name: str = "auto") -> TrainingEnvironmentConfig:
        """Create a complete training environment configuration."""
        
        if environment_name == "auto":
            environment_name = f"auto_{self.gpu_info.get('name', 'cpu').replace(' ', '_').lower()}"
        
        # Create components
        gpu_config = self.create_gpu_config()
        model_configs = self.create_model_configs(gpu_config)
        dataset_configs = self.create_dataset_configs()
        
        # Global settings
        global_settings = {
            'project_name': 'CAN-Graph Enhanced Training',
            'version': '2.0.0',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42,
            'deterministic': True,
            'benchmark': True,
            'log_level': 'INFO',
            'save_checkpoints': True,
            'checkpoint_frequency': 5,
            'early_stopping': True,
            'validation_frequency': 1,
            'profiling_enabled': False
        }
        
        # Resource management settings
        resource_management = {
            'adaptive_batch_sizing': True,
            'memory_monitoring': True,
            'gpu_monitoring': True,
            'automatic_cleanup': True,
            'memory_cleanup_frequency': 100,  # steps
            'resource_check_frequency': 50,   # steps
            'oom_recovery_enabled': True,
            'performance_optimization': True,
            'dynamic_scheduling': True
        }
        
        return TrainingEnvironmentConfig(
            environment_id=environment_name,
            gpu_config=gpu_config,
            model_configs=model_configs,
            dataset_configs=dataset_configs,
            global_settings=global_settings,
            resource_management=resource_management,
            created_at=datetime.now()
        )
    
    def save_environment_config(self, config: TrainingEnvironmentConfig, 
                               filename: Optional[str] = None) -> Path:
        """Save environment configuration to file."""
        if filename is None:
            filename = f"{config.environment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        file_path = self.output_dir / filename
        
        # Convert to dict and save
        config_dict = asdict(config)
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"✓ Environment configuration saved to {file_path}")
        return file_path
    
    def load_environment_config(self, config_path: Union[str, Path]) -> TrainingEnvironmentConfig:
        """Load environment configuration from file."""
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct the configuration object
        # Note: This is a simplified reconstruction; in practice, you might want more robust deserialization
        return TrainingEnvironmentConfig(**config_dict)
    
    def generate_hydra_configs(self, env_config: TrainingEnvironmentConfig) -> Dict[str, Path]:
        """Generate Hydra configuration files from environment config."""
        generated_configs = {}
        
        for model_type, model_config in env_config.model_configs.items():
            for dataset_name, dataset_config in env_config.dataset_configs.items():
                # Create Hydra config
                hydra_config = {
                    # Model settings
                    'model_type': model_type,
                    'epochs': model_config.epochs,
                    'lr': model_config.learning_rate,
                    'batch_size': model_config.batch_size,
                    'hidden_dim': model_config.hidden_dim,
                    'dropout_rate': model_config.dropout_rate,
                    'weight_decay': model_config.weight_decay,
                    'patience': model_config.patience,
                    'grad_clip': model_config.grad_clip_norm,
                    
                    # Dataset settings
                    'root_folder': dataset_name,
                    'train_ratio': dataset_config.train_ratio,
                    'datasize': dataset_config.data_size_fraction,
                    
                    # GPU settings
                    'device': env_config.global_settings['device'],
                    'num_workers': env_config.gpu_config.num_workers,
                    'pin_memory': env_config.gpu_config.pin_memory,
                    'mixed_precision': env_config.gpu_config.mixed_precision,
                    
                    # Optimization settings
                    'optimizer': model_config.optimizer_type,
                    'scheduler': model_config.scheduler_type if model_config.use_scheduler else 'none',
                    
                    # Resource management
                    **env_config.resource_management,
                    
                    # Global settings
                    'seed': env_config.global_settings['seed'],
                    'deterministic': env_config.global_settings['deterministic']
                }
                
                # Additional model-specific settings
                if model_type == 'fusion':
                    hydra_config.update({
                        'fusion_episodes': model_config.epochs,
                        'fusion_lr': model_config.learning_rate,
                        'fusion_batch_size': model_config.batch_size,
                        'buffer_size': env_config.resource_management.get('buffer_size', 300000),
                        'target_update_freq': 10,
                        'epsilon_start': 0.9,
                        'epsilon_end': 0.1,
                        'epsilon_decay': 0.995
                    })
                elif model_type == 'student':
                    hydra_config.update({
                        'distillation_alpha': 0.7,
                        'temperature': 3.0,
                        'teacher_model_required': True
                    })
                
                # Save Hydra config
                config_filename = f"{model_type}_{dataset_name}.yaml"
                config_path = self.output_dir / config_filename
                
                # Convert to OmegaConf and save
                omega_config = OmegaConf.create(hydra_config)
                OmegaConf.save(omega_config, config_path)
                
                generated_configs[f"{model_type}_{dataset_name}"] = config_path
        
        print(f"✓ Generated {len(generated_configs)} Hydra configuration files")
        return generated_configs
    
    def validate_config(self, config: TrainingEnvironmentConfig) -> Tuple[bool, List[str]]:
        """Validate configuration for potential issues."""
        issues = []
        
        # Check GPU memory requirements
        for model_type, model_config in config.model_configs.items():
            estimated_memory = self._estimate_memory_usage(
                model_config.batch_size, 
                model_config.hidden_dim,
                model_type
            )
            
            if estimated_memory > config.gpu_config.memory_gb * 0.9:
                issues.append(
                    f"{model_type} model may exceed GPU memory: "
                    f"{estimated_memory:.1f}GB > {config.gpu_config.memory_gb * 0.9:.1f}GB (90% limit)"
                )
        
        # Check batch size reasonableness
        for model_type, model_config in config.model_configs.items():
            if model_config.batch_size < 32:
                issues.append(f"{model_type} batch size very small ({model_config.batch_size}) - may be inefficient")
            elif model_config.batch_size > 32768 and model_type != 'fusion':
                issues.append(f"{model_type} batch size very large ({model_config.batch_size}) - may cause OOM")
        
        # Check dataset paths
        for dataset_name, dataset_config in config.dataset_configs.items():
            if not Path(dataset_config.dataset_path).exists():
                issues.append(f"Dataset path does not exist: {dataset_config.dataset_path}")
        
        # Check learning rates
        for model_type, model_config in config.model_configs.items():
            if model_config.learning_rate > 0.01:
                issues.append(f"{model_type} learning rate very high ({model_config.learning_rate}) - may cause instability")
            elif model_config.learning_rate < 1e-6:
                issues.append(f"{model_type} learning rate very low ({model_config.learning_rate}) - may be too slow")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _estimate_memory_usage(self, batch_size: int, hidden_dim: int, model_type: str) -> float:
        """Rough estimation of GPU memory usage."""
        # Empirical estimates (GB) for different model types
        base_memory = {
            'teacher': 2.0,
            'student': 1.5,
            'fusion': 1.0
        }
        
        # Scale with batch size and hidden dimension
        batch_factor = batch_size / 1024  # Normalize to 1024 batch size
        hidden_factor = hidden_dim / 256  # Normalize to 256 hidden dim
        
        estimated = base_memory.get(model_type, 2.0) * batch_factor * hidden_factor
        return max(1.0, estimated)  # Minimum 1GB
    
    def get_optimization_recommendations(self, config: TrainingEnvironmentConfig) -> List[str]:
        """Generate optimization recommendations for the configuration."""
        recommendations = []
        
        gpu_memory_gb = config.gpu_config.memory_gb
        
        # Memory utilization recommendations
        total_estimated_memory = sum(
            self._estimate_memory_usage(
                model_config.batch_size,
                model_config.hidden_dim,
                model_type
            )
            for model_type, model_config in config.model_configs.items()
        )
        
        if total_estimated_memory < gpu_memory_gb * 0.5:
            recommendations.append(
                f"GPU memory underutilized ({total_estimated_memory:.1f}/{gpu_memory_gb:.1f}GB). "
                "Consider increasing batch sizes for better performance."
            )
        
        # Batch size recommendations
        for model_type, model_config in config.model_configs.items():
            if model_config.batch_size < config.gpu_config.optimal_batch_size // 2:
                recommendations.append(
                    f"Consider increasing {model_type} batch size from {model_config.batch_size} "
                    f"to ~{config.gpu_config.optimal_batch_size} for better GPU utilization."
                )
        
        # Mixed precision recommendations
        if config.gpu_config.compute_capability >= "7.0" and not config.gpu_config.mixed_precision:
            recommendations.append(
                "Enable mixed precision training for better performance on modern GPUs."
            )
        
        # Dataset-specific recommendations
        if len(config.dataset_configs) > 4:
            recommendations.append(
                "Consider training datasets in parallel or prioritizing based on importance."
            )
        
        return recommendations

def create_optimized_config(environment_name: str = "auto") -> TrainingEnvironmentConfig:
    """Create an optimized configuration for the current system."""
    manager = EnhancedConfigManager()
    return manager.create_complete_environment_config(environment_name)

if __name__ == "__main__":
    # Demonstration and testing
    print("Enhanced Configuration Manager Demo")
    print("="*50)
    
    # Create config manager
    manager = EnhancedConfigManager()
    
    # Create optimized configuration
    config = manager.create_complete_environment_config("demo_environment")
    
    # Validate configuration
    is_valid, issues = manager.validate_config(config)
    print(f"\
Configuration valid: {is_valid}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Get recommendations
    recommendations = manager.get_optimization_recommendations(config)
    if recommendations:
        print("\
Optimization recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    # Save configuration
    config_path = manager.save_environment_config(config)
    
    # Generate Hydra configs
    hydra_configs = manager.generate_hydra_configs(config)
    print(f"\
Generated Hydra configs: {len(hydra_configs)}")
    
    print("\
Demo completed successfully!")