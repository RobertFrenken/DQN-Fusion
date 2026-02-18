"""Memory monitoring utilities for training stages.

Provides CPU and GPU memory metrics collection and human-readable summaries.
"""
from __future__ import annotations

import logging

import psutil
import torch

log = logging.getLogger(__name__)


def log_memory_metrics(step: int | None = None) -> dict[str, float]:
    """Collect current CPU and GPU memory usage.

    Args:
        step: Optional step number (included in returned dict if provided)

    Returns:
        Dictionary of memory metrics
    """
    metrics: dict[str, float] = {}

    # CPU memory
    mem = psutil.virtual_memory()
    metrics["cpu_mem_percent"] = mem.percent
    metrics["cpu_mem_used_gb"] = mem.used / (1024 ** 3)
    metrics["cpu_mem_available_gb"] = mem.available / (1024 ** 3)

    # GPU memory (if available)
    if torch.cuda.is_available():
        metrics["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        metrics["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
        metrics["gpu_mem_max_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            for i in range(num_gpus):
                metrics[f"gpu{i}_mem_allocated_gb"] = torch.cuda.memory_allocated(i) / (1024 ** 3)

    return metrics


def get_memory_summary() -> str:
    """Get a human-readable memory summary string.

    Returns:
        Formatted string with current memory usage
    """
    mem = psutil.virtual_memory()
    summary = f"CPU: {mem.percent:.1f}% ({mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB)"

    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        summary += f" | GPU: {gpu_alloc:.2f}/{gpu_reserved:.2f} GB (alloc/reserved)"

    return summary
