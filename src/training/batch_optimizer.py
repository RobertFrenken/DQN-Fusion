"""
Batch Size Optimization for Graph Neural Networks

Handles Lightning Tuner-based batch size optimization with safety factors
for graph data memory overhead (message passing, attention, aggregations).
"""

import os
import time
import glob
import logging
from pathlib import Path
from typing import Optional

import torch
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

from src.training.datamodules import CANGraphDataModule

logger = logging.getLogger(__name__)


class BatchSizeOptimizer:
    """Optimizes batch size for graph neural network training."""
    
    def __init__(
        self,
        accelerator: str = "auto",
        devices: int = 1,
        graph_memory_safety_factor: float = 0.5,
        max_batch_size_trials: int = 10,
        batch_size_mode: str = "power"
    ):
        """
        Initialize batch size optimizer.
        
        Args:
            accelerator: PyTorch Lightning accelerator type
            devices: Number of devices to use
            graph_memory_safety_factor: Conservative factor (0.3-0.7) to account for 
                graph operation memory overhead not captured by tuner
            max_batch_size_trials: Maximum tuning trials
            batch_size_mode: Tuning mode ('power' or 'binsearch')
        """
        self.accelerator = accelerator
        self.devices = devices
        self.graph_memory_safety_factor = graph_memory_safety_factor
        self.max_batch_size_trials = max_batch_size_trials
        self.batch_size_mode = batch_size_mode
    
    def optimize_with_datamodule(self, model, datamodule) -> int:
        """
        Optimize batch size using custom datamodule (for curriculum learning).
        
        Args:
            model: Lightning module with batch_size attribute
            datamodule: Lightning DataModule
            
        Returns:
            Optimized safe batch size
        """
        logger.info("🔧 Optimizing batch size with curriculum datamodule...")

        trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            precision='32-true',
            max_steps=50,
            max_epochs=None,
            enable_checkpointing=False,
            logger=False
        )

        tuner = Tuner(trainer)
        initial_bs = model.batch_size

        try:
            # Temporarily reduce workers for tuning
            original_workers = getattr(datamodule, 'num_workers', None)
            if hasattr(datamodule, 'num_workers'):
                datamodule.num_workers = 4
            
            # Run tuner - modifies model.batch_size in place
            tuner.scale_batch_size(
                model,
                datamodule=datamodule,
                mode=self.batch_size_mode,
                steps_per_trial=50,
                init_val=datamodule.batch_size,
                max_trials=self.max_batch_size_trials
            )
            
            # Capture tuned batch size from model attribute
            tuner_batch_size = getattr(model, 'batch_size', initial_bs)
            
            logger.info(f"📊 Tuner changed batch size: {initial_bs} → {tuner_batch_size}")

            # Restore original workers
            if original_workers is not None and hasattr(datamodule, 'num_workers'):
                datamodule.num_workers = original_workers

            # Apply safety factor for graph data memory overhead
            safe_batch_size = int(tuner_batch_size * self.graph_memory_safety_factor)
            safe_batch_size = max(8, safe_batch_size)  # Minimum viable batch size
            
            # Update both model and datamodule
            model.batch_size = safe_batch_size
            datamodule.batch_size = safe_batch_size
            
            logger.info(f"✅ Batch size tuner found: {tuner_batch_size}")
            logger.info(f"🛡️ Applied {self.graph_memory_safety_factor:.0%} safety factor")
            logger.info(f"📊 Final safe batch size: {safe_batch_size}")
            
            return safe_batch_size

        except Exception as e:
            logger.warning(f"⚠️  Batch size optimization failed: {e}")
            logger.warning(f"Falling back to initial batch size: {datamodule.batch_size}")
            if hasattr(model, 'batch_size'):
                model.batch_size = datamodule.batch_size
            return datamodule.batch_size

        finally:
            self._cleanup_tuner_checkpoints()
            self._free_gpu_memory()
            
        # Sanity check with forward+backward pass
        self._validate_batch_size(model, datamodule)
        
        return model.batch_size
    
    def optimize_with_datasets(self, model, train_dataset, val_dataset, initial_batch_size: int = 64) -> int:
        """
        Optimize batch size using raw datasets.
        
        Args:
            model: Lightning module with batch_size attribute
            train_dataset: Training dataset
            val_dataset: Validation dataset
            initial_batch_size: Starting batch size
            
        Returns:
            Optimized safe batch size
        """
        logger.info("🔧 Optimizing batch size...")
        
        # Create temporary DataModule with reduced workers
        temp_datamodule = CANGraphDataModule(
            train_dataset, 
            val_dataset, 
            initial_batch_size, 
            num_workers=4
        )
        
        trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            precision='32-true',
            max_steps=200,
            max_epochs=None,
            enable_checkpointing=False,
            logger=False
        )
        
        tuner = Tuner(trainer)
        initial_bs = model.batch_size
        
        try:
            # Run tuner
            tuner.scale_batch_size(
                model,
                datamodule=temp_datamodule,
                mode=self.batch_size_mode,
                steps_per_trial=200,
                init_val=initial_batch_size,
                max_trials=self.max_batch_size_trials // 2
            )
            
            # Capture tuned batch size
            tuner_batch_size = getattr(model, 'batch_size', initial_bs)
            logger.info(f"📊 Tuner changed batch size: {initial_bs} → {tuner_batch_size}")
            
            # Apply safety factor
            safe_batch_size = int(tuner_batch_size * self.graph_memory_safety_factor)
            safe_batch_size = max(8, safe_batch_size)
            
            model.batch_size = safe_batch_size
            
            logger.info(f"✅ Batch size tuner found: {tuner_batch_size}")
            logger.info(f"🛡️ Applied {self.graph_memory_safety_factor:.0%} safety factor")
            logger.info(f"📊 Final safe batch size: {safe_batch_size}")
            
            return safe_batch_size

        except Exception as e:
            logger.warning(f"⚠️  Batch size optimization failed: {e}")
            logger.warning(f"Falling back to initial batch size: {initial_bs}")
            model.batch_size = initial_bs
            return initial_bs
        
        finally:
            self._cleanup_tuner_checkpoints()
            self._free_gpu_memory()
    
    def _cleanup_tuner_checkpoints(self):
        """Remove temporary tuner checkpoint files."""
        try:
            checkpoint_patterns = [
                '.scale_batch_size_*.ckpt',
                'scale_batch_size_*.ckpt',
                '.scale_batch_size_*',
                'lightning_logs/*/checkpoints/*.ckpt'
            ]
            
            removed_count = 0
            for pattern in checkpoint_patterns:
                for f in glob.glob(pattern, recursive=True):
                    try:
                        if os.path.exists(f):
                            # Only remove recent files (safety check)
                            if time.time() - os.path.getmtime(f) < 600:  # 10 minutes
                                os.remove(f)
                                removed_count += 1
                                logger.debug(f"Removed tuner checkpoint: {f}")
                    except Exception as e:
                        logger.debug(f"Could not remove {f}: {e}")
            
            if removed_count > 0:
                logger.info(f"🧹 Cleaned up {removed_count} temporary tuner checkpoint file(s)")
                
        except Exception as e:
            logger.debug(f"Tuner checkpoint cleanup encountered issue: {e}")
    
    def _free_gpu_memory(self):
        """Free fragmented GPU memory."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared GPU cache after batch size tuning")
        except Exception:
            pass
    
    def _validate_batch_size(self, model, datamodule):
        """
        Validate tuned batch size with forward+backward pass.
        
        Attempts to catch OOM errors that tuner might miss.
        Reduces batch size by half on failure until it works.
        """
        try:
            dl = datamodule.train_dataloader()
            it = iter(dl)
            candidate_bs = getattr(model, 'batch_size', 64)
            
            while candidate_bs >= 1:
                try:
                    datamodule.batch_size = candidate_bs
                    model.batch_size = candidate_bs
                    
                    batch = next(it)
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    batch = batch.to(device)
                    model.to(device)
                    
                    # Forward pass
                    out = model(batch) if hasattr(model, 'forward') else model(batch)
                    
                    # Compute loss (simplified)
                    if isinstance(out, (list, tuple)):
                        loss = out[0].sum()
                    else:
                        loss = out.sum()
                    
                    # Backward pass to test memory
                    loss.backward()
                    
                    # Success!
                    model.zero_grad()
                    self._free_gpu_memory()
                    logger.debug(f"✓ Validated batch size {candidate_bs}")
                    break
                    
                except StopIteration:
                    break
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        logger.warning(f"OOM at batch size {candidate_bs}, reducing...")
                        self._free_gpu_memory()
                        candidate_bs = max(candidate_bs // 2, 1)
                        it = iter(dl)  # Reset iterator
                        continue
                    else:
                        break
                        
        except Exception as e:
            logger.debug(f"Batch size validation skipped: {e}")
