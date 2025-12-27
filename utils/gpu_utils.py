"""
GPU Utilities for Fusion Training
Contains GPU detection, configuration, and data loader optimization functions.
"""
import os
import torch
from torch_geometric.loader import DataLoader


def detect_gpu_capabilities_unified(device: str = 'cuda'):
    """UNIFIED GPU configuration - single source of truth for all batch parameters.
    
    This is the authoritative function for GPU detection and configuration.
    Requires CUDA GPU - no CPU fallback.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for fusion training. No GPU detected.")
        
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        allocated_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        allocated_cpus = 6
    
    # Get GPU properties
    gpu_device = torch.device('cuda:0')
    gpu_props = torch.cuda.get_device_properties(gpu_device)
    memory_gb = gpu_props.total_memory / (1024**3)
    
    # UNIFIED GPU CONFIGURATION - Consistent batches regardless of dataset size
    if memory_gb >= 30:  # A100 40GB+
        config = {
            'batch_size': 8192,
            'num_workers': 24,
            'prefetch_factor': 6,
            'gpu_processing_batch': 65536,
            'dqn_training_batch': 32768,
            'dqn_batch_size': 32768,
            'buffer_size': 500000,
            'training_steps_per_episode': 16,
            'episode_sample_ratio': 0.5
        }
    else:  # Other GPUs
        config = {
            'batch_size': 4096,
            'num_workers': 16,
            'prefetch_factor': 4,
            'gpu_processing_batch': 32768,
            'dqn_training_batch': 16384,
            'dqn_batch_size': 16384,
            'buffer_size': 300000,
            'training_steps_per_episode': 12,
            'episode_sample_ratio': 0.4
        }
    
    # Add common parameters
    config.update({
        'name': gpu_props.name,
        'memory_gb': memory_gb,
        'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
        'pin_memory': True,
        'persistent_workers': True,
        'drop_last': False,
        'cuda_available': True
    })
    
    return config


def create_optimized_data_loaders(train_subset=None, test_dataset=None, full_train_dataset=None, 
                                 batch_size: int = 1024, device: str = 'cuda'):
    """Create optimized data loaders using unified GPU configuration."""
    
    dataset = next((d for d in [train_subset, test_dataset, full_train_dataset] if d is not None), None)
    if dataset is None:
        raise ValueError("No valid dataset provided")
    
    config = detect_gpu_capabilities_unified(device)
    torch.cuda.empty_cache()
    
    datasets = [train_subset, test_dataset, full_train_dataset]
    shuffles = [True, False, True]
    
    for dataset, shuffle in zip(datasets, shuffles):
        if dataset is not None:
            return DataLoader(
                dataset,
                batch_size=config['batch_size'],
                shuffle=shuffle,
                pin_memory=config['pin_memory'],
                num_workers=config['num_workers'],
                persistent_workers=config['persistent_workers'],
                prefetch_factor=config['prefetch_factor']
            )
    
    raise ValueError("No valid dataset provided")