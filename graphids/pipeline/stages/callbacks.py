"""Lightning callbacks for memory monitoring and profiling."""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
import torch

from ..tracking import get_memory_summary

log = logging.getLogger(__name__)


class MemoryMonitorCallback(pl.Callback):
    """Log memory usage at epoch boundaries."""

    def __init__(
        self,
        log_every_n_epochs: int = 5,
        predicted_peak_mb: float = 0.0,
        run_id: str = "",
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.predicted_peak_mb = predicted_peak_mb
        self._logged_first_epoch = False

    def on_train_start(self, trainer, pl_module):
        log.info("Memory at train start: %s", get_memory_summary())

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch % self.log_every_n_epochs == 0:
            log.info("Epoch %d memory: %s", epoch, get_memory_summary())

        # Log predicted vs actual peak memory after the first epoch
        if not self._logged_first_epoch and epoch == 0 and torch.cuda.is_available():
            self._logged_first_epoch = True
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            if self.predicted_peak_mb > 0:
                ratio = peak_mb / self.predicted_peak_mb
                error_pct = abs(1.0 - ratio) * 100
                log.info("Memory prediction: ratio=%.3f error=%.1f%%", ratio, error_pct)
            torch.cuda.reset_peak_memory_stats()
            log.info(
                "Epoch 0 peak memory: actual=%.1fMB predicted=%.1fMB",
                peak_mb,
                self.predicted_peak_mb,
            )

    def on_train_end(self, trainer, pl_module):
        log.info("Memory at train end: %s", get_memory_summary())


class ProfilerCallback(pl.Callback):
    """PyTorch Profiler callback that captures CPU/GPU traces for TensorBoard.

    Writes Chrome-format traces to ``output_dir`` that can be loaded via
    ``tensorboard --logdir <output_dir>`` with the PyTorch Profiler plugin.
    """

    def __init__(
        self,
        output_dir: str,
        wait_steps: int = 1,
        warmup_steps: int = 1,
        active_steps: int = 5,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self._profiler = None

    def on_train_start(self, trainer, pl_module):
        from torch.profiler import profile, schedule, tensorboard_trace_handler

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._profiler = profile(
            schedule=schedule(
                wait=self.wait_steps,
                warmup=self.warmup_steps,
                active=self.active_steps,
            ),
            on_trace_ready=tensorboard_trace_handler(self.output_dir),
            record_shapes=True,
            with_stack=True,
        )
        self._profiler.__enter__()
        log.info("Profiler started — traces will be saved to %s", self.output_dir)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._profiler is not None:
            self._profiler.step()

    def on_train_end(self, trainer, pl_module):
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            log.info("Profiler stopped — traces saved to %s", self.output_dir)
            self._profiler = None
