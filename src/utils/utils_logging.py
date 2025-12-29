import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.data import random_split, Subset
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix, classification_report
import random
import psutil
import gc
import multiprocessing as mp
from typing import Tuple, Dict, Any, Optional, List
import warnings

def setup_gpu_optimization():
    """Configure GPU memory optimization settings."""
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        torch.cuda.empty_cache()
        print("✓ GPU memory optimization enabled")

def log_memory_usage(stage: str = ""):
    """Log current CPU and GPU memory usage."""
    cpu_mem = psutil.virtual_memory()
    print(f"[{stage}] CPU Memory: {cpu_mem.used/1024**3:.1f}GB / {cpu_mem.total/1024**3:.1f}GB ({cpu_mem.percent:.1f}%)")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_cached:.2f}GB cached")

def cleanup_memory():
    """Perform comprehensive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()