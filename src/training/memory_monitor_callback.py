"""
PyTorch Lightning callback for monitoring GPU memory during training.

Tracks peak memory usage during steady-state training (excluding warmup)
and updates the adaptive safety factor database.
"""

from typing import Optional
import logging
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer

logger = logging.getLogger(__name__)


class MemoryMonitorCallback(Callback):
    """
    Monitor GPU memory usage during training and update safety factors.

    Tracks peak memory during steady-state training (after warmup) and
    updates the adaptive batch size safety factor database.
    """

    def __init__(self, config: 'CANGraphConfig', warmup_batches: int = 20,
                 check_every_n_batches: int = 50):
        """
        Initialize memory monitor.

        Args:
            config: CANGraphConfig object
            warmup_batches: Number of batches to skip before monitoring (dataset loading)
            check_every_n_batches: Check memory every N batches during training
        """
        self.config = config
        self.warmup_batches = warmup_batches
        self.check_every_n_batches = check_every_n_batches

        # Memory tracking
        self.peak_memory_pct: float = 0.0
        self.peak_memory_gb: float = 0.0
        self.total_memory_gb: float = 0.0
        self.batch_size: Optional[int] = None
        self.batches_seen: int = 0
        self.in_warmup: bool = True

        # Config key for database
        from src.training.adaptive_batch_size import get_config_key
        self.config_key = get_config_key(config)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        """Called at start of training."""
        if not torch.cuda.is_available():
            logger.warning("GPU not available - memory monitoring disabled")
            return

        # Get total GPU memory
        self.total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        logger.info(f"Memory monitor started (warmup: {self.warmup_batches} batches)")
        logger.info(f"Total GPU memory: {self.total_memory_gb:.2f} GB")
        logger.info(f"Config: {self.config_key}")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs, batch, batch_idx: int):
        """Called after each training batch."""
        if not torch.cuda.is_available():
            return

        self.batches_seen += 1

        # Skip warmup batches (dataset loading overhead)
        if self.batches_seen <= self.warmup_batches:
            if self.batches_seen == self.warmup_batches:
                logger.info(f"Warmup complete - starting memory monitoring")
                self.in_warmup = False
            return

        # Check memory periodically
        if self.batches_seen % self.check_every_n_batches == 0:
            self._check_memory()

        # Store batch size (from first batch after warmup)
        if self.batch_size is None and hasattr(batch, 'batch_size'):
            self.batch_size = batch.batch_size
        elif self.batch_size is None and hasattr(batch, 'num_graphs'):
            self.batch_size = batch.num_graphs

    def _check_memory(self):
        """Check current GPU memory usage and update peak."""
        try:
            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9   # GB

            # Calculate percentage of total memory
            memory_pct = memory_allocated / self.total_memory_gb

            # Update peak
            if memory_pct > self.peak_memory_pct:
                self.peak_memory_pct = memory_pct
                self.peak_memory_gb = memory_allocated

                logger.debug(
                    f"New peak memory: {memory_allocated:.2f} GB "
                    f"({memory_pct:.1%} of {self.total_memory_gb:.2f} GB)"
                )

        except Exception as e:
            logger.warning(f"Failed to check memory: {e}")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        """Called at end of training - update database."""
        if not torch.cuda.is_available():
            return

        # Final memory check
        self._check_memory()

        logger.info("="*70)
        logger.info("MEMORY MONITORING SUMMARY")
        logger.info("="*70)
        logger.info(f"Config: {self.config_key}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Peak memory: {self.peak_memory_gb:.2f} GB ({self.peak_memory_pct:.1%})")
        logger.info(f"Total memory: {self.total_memory_gb:.2f} GB")
        logger.info(f"Batches monitored: {self.batches_seen - self.warmup_batches}")

        # Update safety factor database
        if self.peak_memory_pct > 0 and self.batch_size is not None:
            try:
                from src.training.adaptive_batch_size import get_safety_factor_db

                db = get_safety_factor_db()

                # Get run ID from trainer if available
                run_id = getattr(trainer, 'run_id', None)
                if run_id is None:
                    run_id = f"{self.config_key}_{trainer.current_epoch}"

                # Update factor
                new_factor = db.update_factor(
                    config_key=self.config_key,
                    peak_memory_pct=self.peak_memory_pct,
                    batch_size=self.batch_size,
                    success=True,
                    run_id=run_id
                )

                logger.info(f"Updated safety factor to: {new_factor:.3f}")
                logger.info("="*70)

            except Exception as e:
                logger.error(f"Failed to update safety factor database: {e}")

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: Exception):
        """Called when training fails - record OOM if applicable."""
        if not torch.cuda.is_available():
            return

        # Check if it's an OOM error
        is_oom = (
            isinstance(exception, RuntimeError) and
            "out of memory" in str(exception).lower()
        )

        if is_oom:
            logger.error("="*70)
            logger.error("CUDA OUT OF MEMORY DETECTED")
            logger.error("="*70)
            logger.error(f"Config: {self.config_key}")
            logger.error(f"Batch size: {self.batch_size}")

            # Record failure in database
            try:
                from src.training.adaptive_batch_size import get_safety_factor_db

                db = get_safety_factor_db()

                # Use batch size if we got it, otherwise estimate from config
                batch_size = self.batch_size
                if batch_size is None:
                    batch_size = getattr(self.config.training, 'batch_size', 64)

                # Record OOM with current peak memory (if any)
                peak = self.peak_memory_pct if self.peak_memory_pct > 0 else 1.0

                # Get run ID from trainer if available
                run_id = getattr(trainer, 'run_id', None)
                if run_id is None:
                    run_id = f"{self.config_key}_OOM_{trainer.current_epoch}"

                new_factor = db.update_factor(
                    config_key=self.config_key,
                    peak_memory_pct=peak,  # Use 1.0 to indicate OOM
                    batch_size=batch_size,
                    success=False,  # OOM = failure
                    run_id=run_id
                )

                logger.error(f"Reduced safety factor to: {new_factor:.3f}")
                logger.error(f"Database updated: {db.db_path}")
                logger.error("="*70)

            except Exception as e:
                logger.error(f"Failed to update database after OOM: {e}")


# ============================================================================
# Integration with CANGraphConfig
# ============================================================================

def get_adaptive_safety_factor(config: 'CANGraphConfig') -> float:
    """
    Get adaptive safety factor for a configuration.

    Looks up factor in database, or returns experience-based initial factor.

    Args:
        config: CANGraphConfig object

    Returns:
        Safety factor (0.3 - 0.8)
    """
    try:
        from src.training.adaptive_batch_size import get_config_key, get_safety_factor_db

        config_key = get_config_key(config)
        db = get_safety_factor_db()

        factor = db.get_factor(config_key, use_experience=True)

        logger.info(f"Using adaptive safety factor: {factor:.3f} for {config_key}")

        return factor

    except Exception as e:
        logger.warning(f"Failed to get adaptive safety factor: {e}")
        logger.warning("Falling back to default 0.6")
        return 0.6
