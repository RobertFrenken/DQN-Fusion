"""
PyTorch Lightning Fabric utilities for dynamic batch sizing and efficient training.

This module provides utilities for:
- Dynamic batch size optimization based on GPU memory
- Memory-efficient data loading
- Automatic mixed precision setup
- SLURM integration utilities
"""

import os
import torch
import psutil
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
import gc

logger = logging.getLogger(__name__)


class DynamicBatchSizer:
    """
    Dynamically determines optimal batch size based on GPU memory and model size.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 sample_input: torch.Tensor,
                 target_memory_usage: float = 0.85,
                 min_batch_size: int = 1,
                 max_batch_size: int = 8192):
        """
        Initialize the dynamic batch sizer.
        
        Args:
            model: PyTorch model to optimize batch size for
            sample_input: Sample input tensor for memory estimation
            target_memory_usage: Target GPU memory usage (0.0-1.0)
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
        """
        self.model = model
        self.sample_input = sample_input
        self.target_memory_usage = target_memory_usage
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        # Get GPU memory info
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
            self.reserved_memory = torch.cuda.memory_reserved(self.device)
            logger.info(f"GPU {self.device}: {self.total_memory / (1024**3):.1f}GB total")
        else:
            raise RuntimeError("DynamicBatchSizer requires CUDA")
    
    def estimate_optimal_batch_size(self) -> int:
        """
        Estimate optimal batch size through binary search with memory profiling.
        
        Returns:
            Optimal batch size
        """
        logger.info("Estimating optimal batch size...")
        
        # Clear cache and get baseline memory
        torch.cuda.empty_cache()
        gc.collect()
        baseline_memory = torch.cuda.memory_allocated(self.device)
        
        # Binary search for optimal batch size
        low, high = self.min_batch_size, self.max_batch_size
        optimal_batch_size = self.min_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Test batch size
                if self._test_batch_size(mid):
                    optimal_batch_size = mid
                    low = mid + 1
                else:
                    high = mid - 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    high = mid - 1
                else:
                    raise e
            finally:
                # Cleanup after each test
                torch.cuda.empty_cache()
                gc.collect()
        
        logger.info(f"Optimal batch size determined: {optimal_batch_size}")
        return optimal_batch_size
    
    def _test_batch_size(self, batch_size: int) -> bool:
        """
        Test if a batch size fits in memory.
        
        Args:
            batch_size: Batch size to test
            
        Returns:
            True if batch size fits within memory constraints
        """
        try:
            # Create dummy batch
            dummy_batch = self._create_dummy_batch(batch_size)
            
            # Forward pass
            self.model.train()
            with torch.cuda.amp.autocast(enabled=True):
                output = self.model(dummy_batch)
                
                # Simulate loss and backward pass
                if hasattr(output, 'loss'):
                    loss = output.loss
                else:
                    # Create dummy loss
                    loss = output.sum() if hasattr(output, 'sum') else torch.tensor(0.0, device=self.device)
                
                loss.backward()
            
            # Check memory usage
            current_memory = torch.cuda.memory_allocated(self.device)
            memory_usage_ratio = current_memory / self.total_memory
            
            return memory_usage_ratio <= self.target_memory_usage
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            raise e
        finally:
            # Clear gradients
            for param in self.model.parameters():
                param.grad = None
    
    def _create_dummy_batch(self, batch_size: int):
        """
        Create a dummy batch based on the sample input.
        
        Args:
            batch_size: Size of dummy batch to create
            
        Returns:
            Dummy batch tensor
        """
        # This is a simplified implementation
        # You may need to customize this based on your data structure
        if hasattr(self.sample_input, 'batch_size'):
            # PyTorch Geometric Data object
            dummy_batch = self.sample_input.clone()
            # Scale up the batch
            dummy_batch.batch = torch.cat([
                torch.full((len(self.sample_input.x),), i, dtype=torch.long)
                for i in range(batch_size)
            ]).to(self.device)
            dummy_batch.x = dummy_batch.x.repeat(batch_size, 1)
            return dummy_batch
        else:
            # Regular tensor
            return self.sample_input.repeat(batch_size, *[1] * (len(self.sample_input.shape) - 1))


class FabricTrainerBase:
    """
    Base class for Fabric-based trainers with common functionality.
    """
    
    def __init__(self, 
                 fabric_config: Dict[str, Any] = None,
                 logger_config: Dict[str, Any] = None):
        """
        Initialize Fabric trainer base.
        
        Args:
            fabric_config: Configuration for Fabric initialization
            logger_config: Configuration for logging
        """
        # Default Fabric configuration
        default_fabric_config = {
            'accelerator': 'gpu',
            'devices': 1,
            'precision': '16-mixed',  # Use mixed precision for efficiency
            'strategy': 'auto'
        }
        
        if fabric_config:
            default_fabric_config.update(fabric_config)
        
        # Setup logger
        if logger_config is None:
            logger_config = {
                'root_dir': 'outputs/fabric_logs',
                'name': 'fabric_training'
            }
        
        csv_logger = CSVLogger(**logger_config)
        
        # Initialize Fabric
        self.fabric = Fabric(
            loggers=csv_logger,
            **default_fabric_config
        )
        
        # Track training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
    def setup_model_and_optimizer(self, model, optimizer, scheduler=None):
        """
        Setup model and optimizer with Fabric.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            scheduler: Optional learning rate scheduler
            
        Returns:
            Tuple of (model, optimizer, scheduler) wrapped by Fabric
        """
        if scheduler is not None:
            return self.fabric.setup(model, optimizer, scheduler)
        else:
            model, optimizer = self.fabric.setup(model, optimizer)
            return model, optimizer, None
    
    def setup_dataloader(self, dataloader):
        """
        Setup dataloader with Fabric.
        
        Args:
            dataloader: PyTorch DataLoader
            
        Returns:
            DataLoader wrapped by Fabric
        """
        return self.fabric.setup_dataloaders(dataloader)
    
    def save_checkpoint(self, 
                       model, 
                       optimizer, 
                       scheduler=None, 
                       filepath: str = None,
                       extra_state: Dict[str, Any] = None):
        """
        Save checkpoint using Fabric.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Optional scheduler state
            filepath: Path to save checkpoint
            extra_state: Additional state to save
        """
        if filepath is None:
            filepath = f"checkpoints/epoch_{self.current_epoch}.ckpt"
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model': model,
            'optimizer': optimizer,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric
        }
        
        if scheduler is not None:
            state['scheduler'] = scheduler
            
        if extra_state is not None:
            state.update(extra_state)
        
        self.fabric.save(filepath, state)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, 
                       filepath: str,
                       model=None, 
                       optimizer=None, 
                       scheduler=None):
        """
        Load checkpoint using Fabric.
        
        Args:
            filepath: Path to checkpoint
            model: Model to load state into
            optimizer: Optimizer to load state into  
            scheduler: Optional scheduler to load state into
            
        Returns:
            Dictionary containing loaded state
        """
        checkpoint = self.fabric.load(filepath)
        
        if model is not None and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics using Fabric logger.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if step is None:
            step = self.global_step
            
        self.fabric.log_dict(metrics, step=step)


def setup_slurm_fabric_config() -> Dict[str, Any]:
    """
    Setup Fabric configuration for SLURM environment.
    
    Returns:
        Dictionary with SLURM-optimized Fabric configuration
    """
    config = {}
    
    # Check if running in SLURM
    if 'SLURM_PROCID' in os.environ:
        logger.info("SLURM environment detected")
        
        # Multi-node configuration
        if 'SLURM_NNODES' in os.environ:
            num_nodes = int(os.environ['SLURM_NNODES'])
            if num_nodes > 1:
                config.update({
                    'num_nodes': num_nodes,
                    'strategy': 'ddp'
                })
                logger.info(f"Multi-node training: {num_nodes} nodes")
        
        # GPU configuration
        if 'SLURM_GPUS_PER_NODE' in os.environ:
            gpus_per_node = int(os.environ['SLURM_GPUS_PER_NODE'])
            config['devices'] = gpus_per_node
            logger.info(f"GPUs per node: {gpus_per_node}")
        
        # Set precision based on GPU type (if available in env)
        gpu_type = os.environ.get('SLURM_GPU_TYPE', '').lower()
        if 'a100' in gpu_type or 'v100' in gpu_type:
            config['precision'] = '16-mixed'
        elif 'h100' in gpu_type:
            config['precision'] = 'bf16-mixed'
        else:
            config['precision'] = '16-mixed'  # Safe default
            
    return config


def optimize_dataloader_for_fabric(dataloader, 
                                 num_workers: int = None,
                                 pin_memory: bool = True,
                                 persistent_workers: bool = True) -> torch.utils.data.DataLoader:
    """
    Optimize DataLoader settings for Fabric training.
    
    Args:
        dataloader: Original DataLoader
        num_workers: Number of worker processes (auto-detected if None)
        pin_memory: Whether to use pinned memory
        persistent_workers: Whether to use persistent workers
        
    Returns:
        Optimized DataLoader
    """
    if num_workers is None:
        # Auto-detect optimal number of workers
        cpu_count = os.cpu_count()
        num_workers = min(cpu_count, 8)  # Cap at 8 to avoid overhead
        
    # Create new DataLoader with optimized settings
    optimized_loader = torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=dataloader.batch_sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=dataloader.drop_last
    )
    
    logger.info(f"DataLoader optimized: {num_workers} workers, pin_memory={pin_memory}")
    return optimized_loader