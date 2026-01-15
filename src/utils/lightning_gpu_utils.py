"""
Lightning-native GPU optimization utilities.
Replaces legacy GPU optimization with Lightning best practices.
"""

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


class LightningGPUOptimizer:
    """Lightning-native GPU optimization (monitoring handled by MLflow)."""
    
    @staticmethod
    def get_trainer_args() -> dict:
        """Return optimized trainer arguments for GPU usage."""
        args = {
            'precision': '16-mixed' if torch.cuda.is_available() else 32,
            'enable_model_summary': True,
            'enable_progress_bar': True,
        }
        
        if torch.cuda.is_available():
            # GPU-specific optimizations
            args.update({
                'accelerator': 'gpu',
                'devices': 1,
                'strategy': 'auto',
                'sync_batchnorm': False,
            })
        else:
            args.update({
                'accelerator': 'cpu',
                'devices': 1,
            })
        
        return args
    
    @staticmethod
    def setup_environment():
        """Set up GPU environment optimizations."""
        if torch.cuda.is_available():
            # Memory optimization
            torch.cuda.empty_cache()
            
            # Better memory allocation strategy
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
            
            logger.info(f"✓ GPU optimization enabled for {torch.cuda.get_device_name()}")
            logger.info(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            logger.info("✓ CPU-only training configured")


class LightningDataLoader:
    """Lightning-native data loading utilities."""
    
    @staticmethod
    def get_dataloader_kwargs(batch_size: int = 256) -> dict:
        """Return optimized DataLoader kwargs."""
        kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'drop_last': False,
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': True,
            'num_workers': 4,  # Conservative default
        }
        
        return kwargs
    
    @staticmethod
    def create_dataloader(dataset, **override_kwargs) -> DataLoader:
        """Create optimized DataLoader."""
        from torch.utils.data import DataLoader
        
        kwargs = LightningDataLoader.get_dataloader_kwargs()
        kwargs.update(override_kwargs)
        
        return DataLoader(dataset, **kwargs)