"""
Legacy compatibility functions for deprecated GPU optimization code.
These are temporary placeholders - use PyTorch Lightning for new training pipelines.
"""
import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

def detect_gpu_capabilities_unified(device_str):
    """
    Legacy compatibility function.
    For new code, use PyTorch Lightning's automatic device detection.
    """
    logger.warning("Using legacy GPU detection. Consider migrating to PyTorch Lightning.")
    
    device = torch.device(device_str)
    
    if device.type == 'cuda' and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        return {
            'device': device_str,
            'memory_gb': gpu_memory / 1e9,
            'compute_capability': torch.cuda.get_device_capability(device),
            'recommended_batch_size': 32  # Conservative default
        }
    else:
        return {
            'device': 'cpu',
            'memory_gb': 4.0,  # Conservative CPU assumption
            'compute_capability': None,
            'recommended_batch_size': 16
        }

def create_optimized_data_loaders(train_subset=None, test_dataset=None, full_train_dataset=None, 
                                batch_size=32, device='cpu', num_workers=0, **kwargs):
    """
    Legacy compatibility function.
    For new code, use PyTorch Lightning DataModules with automatic optimization.
    """
    logger.warning("Using legacy data loader creation. Consider migrating to PyTorch Lightning DataModules.")
    
    loaders = []
    
    # Handle different parameter patterns from various legacy calls
    if train_subset is not None:
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=(device != 'cpu')
        )
        loaders.append(train_loader)
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=(device != 'cpu')
        )
        loaders.append(test_loader)
    
    if full_train_dataset is not None:
        full_train_loader = DataLoader(
            full_train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=(device != 'cpu')
        )
        loaders.append(full_train_loader)
    
    # Return based on what was requested
    if len(loaders) == 1:
        return loaders[0]
    elif len(loaders) == 3:
        return loaders[0], loaders[1], loaders[2]  # train, test, full_train
    else:
        return tuple(loaders) if loaders else None