"""
Fusion Training Pipeline for CAN Bus Intrusion Detection

This module implements reinforcement learning-based fusion training for combining
VGAE anomaly detection and GAT classification outputs using a DQN agent.

Key Features:
- Load pre-trained VGAE and GAT models
- Extract anomaly scores and GAT probabilities from validation data
- Train DQN fusion agent with experience replay
- Evaluate and compare fusion strategies
- Save trained fusion agent for deployment
"""

from contextlib import nullcontext
import sys
import os
from pathlib import Path
# Clean path setup - add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.data import random_split, Subset
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any
import warnings
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import psutil

# Import your existing modules
from models.models import GATWithJK, GraphAutoencoderNeighborhood
from models.adaptive_fusion import EnhancedDQNFusionAgent
from archive.preprocessing import graph_creation, build_id_mapping_from_normal
from utils.utils_logging import setup_gpu_optimization, log_memory_usage, cleanup_memory

warnings.filterwarnings('ignore', category=UserWarning)

class CudaPrefetcher:
    """Double-buffer batches to GPU to overlap H2D copy with compute."""
    def __init__(self, loader: DataLoader, device: torch.device, prefetch_batches: int = 2):
        self.loader = loader
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.prefetch_batches = max(1, prefetch_batches)
        self.iterator = iter(loader)
        self.prefetch_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self._queue = []
        self._prefetch()

    def _prefetch(self):
        # Keep up to prefetch_batches ready (already moved to device if CUDA)
        while len(self._queue) < self.prefetch_batches:
            try:
                batch = next(self.iterator)
            except StopIteration:
                break
            if self.device.type == 'cuda':
                # Pin memory is handled by DataLoader; move to device on a separate stream
                with torch.cuda.stream(self.prefetch_stream):
                    batch = batch.to(self.device, non_blocking=True)
            self._queue.append(batch)

    def next(self):
        if not self._queue:
            return None
        if self.device.type == 'cuda':
            torch.cuda.current_stream(self.device).wait_stream(self.prefetch_stream)
        batch = self._queue.pop(0)
        self._prefetch()
        return batch
class GPUMonitor:
    """Monitor GPU usage, memory, and performance metrics during training."""
    
    def __init__(self, device):
        self.device = torch.device(device)
        self.is_cuda = self.device.type == 'cuda'
        self.gpu_stats = []
        self.timing_stats = []
        
        if self.is_cuda:
            self.gpu_name = torch.cuda.get_device_properties(self.device).name
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self.gpu_name = "CPU"
            self.total_memory = psutil.virtual_memory().total
    
    def record_gpu_stats(self, episode: int):
        """Record current GPU statistics."""
        if self.is_cuda:
            torch.cuda.synchronize()  # Ensure all ops are complete
            
            # Memory stats
            memory_allocated = torch.cuda.memory_allocated(self.device)
            memory_reserved = torch.cuda.memory_reserved(self.device)
            memory_free = self.total_memory - memory_reserved
            
            # Calculate utilization percentages
            memory_util = (memory_allocated / self.total_memory) * 100
            reserved_util = (memory_reserved / self.total_memory) * 100
            
            # Try to get GPU utilization (requires nvidia-ml-py3 or pynvml)
            gpu_util = 0.0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except (ImportError, Exception):
                # Fallback: estimate based on memory usage
                gpu_util = min(95.0, memory_util * 1.2)  # Rough approximation
            
            stats = {
                'episode': episode,
                'memory_allocated_gb': memory_allocated / (1024**3),
                'memory_reserved_gb': memory_reserved / (1024**3),
                'memory_free_gb': memory_free / (1024**3),
                'memory_utilization_pct': memory_util,
                'reserved_utilization_pct': reserved_util,
                'gpu_utilization_pct': gpu_util,
                'timestamp': time.time()
            }
        else:
            # CPU stats
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            stats = {
                'episode': episode,
                'memory_allocated_gb': (memory.total - memory.available) / (1024**3),
                'memory_reserved_gb': memory.used / (1024**3),
                'memory_free_gb': memory.available / (1024**3),
                'memory_utilization_pct': memory.percent,
                'reserved_utilization_pct': memory.percent,
                'gpu_utilization_pct': cpu_percent,
                'timestamp': time.time()
            }
        
        self.gpu_stats.append(stats)
    
    def record_timing(self, episodes_completed: int, elapsed_time: float):
        """Record timing statistics."""
        self.timing_stats.append({
            'episodes': episodes_completed,
            'elapsed_time': elapsed_time,
            'episodes_per_minute': (episodes_completed / elapsed_time) * 60 if elapsed_time > 0 else 0
        })
    
    def get_performance_summary(self) -> Dict:
        """Generate performance summary with batch size recommendations."""
        if not self.gpu_stats:
            return {'error': 'No GPU stats collected'}
        
        # Calculate averages
        avg_memory_util = np.mean([s['memory_utilization_pct'] for s in self.gpu_stats])
        avg_gpu_util = np.mean([s['gpu_utilization_pct'] for s in self.gpu_stats])
        max_memory_util = max([s['memory_utilization_pct'] for s in self.gpu_stats])
        avg_memory_allocated = np.mean([s['memory_allocated_gb'] for s in self.gpu_stats])
        
        # Timing analysis
        timing_summary = {}
        if self.timing_stats:
            total_time = sum([t['elapsed_time'] for t in self.timing_stats])
            total_episodes = sum([t['episodes'] for t in self.timing_stats])
            avg_episodes_per_min = np.mean([t['episodes_per_minute'] for t in self.timing_stats])
            
            timing_summary = {
                'total_training_time_minutes': total_time / 60,
                'total_episodes_trained': total_episodes,
                'average_episodes_per_minute': avg_episodes_per_min,
                'estimated_time_per_100_episodes_minutes': (100 / avg_episodes_per_min) if avg_episodes_per_min > 0 else 0
            }
        
        # Batch size recommendations
        current_batch_util = avg_memory_util
        recommendations = self._generate_batch_recommendations(current_batch_util, max_memory_util, avg_gpu_util)
        
        return {
            'device_name': self.gpu_name,
            'total_memory_gb': self.total_memory / (1024**3),
            'average_memory_utilization_pct': avg_memory_util,
            'average_gpu_utilization_pct': avg_gpu_util,
            'peak_memory_utilization_pct': max_memory_util,
            'average_memory_allocated_gb': avg_memory_allocated,
            'timing': timing_summary,
            'recommendations': recommendations,
            'stats_collected': len(self.gpu_stats)
        }
    
    def _generate_batch_recommendations(self, avg_memory_util: float, peak_memory_util: float, avg_gpu_util: float) -> Dict:
        """Generate intelligent batch size recommendations."""
        recommendations = {
            'current_efficiency': 'unknown',
            'batch_size_recommendation': 'maintain current',
            'reasoning': [],
            'target_memory_utilization': '70-85%',
            'target_gpu_utilization': '85-95%'
        }
        
        # Memory utilization analysis
        if peak_memory_util > 90:
            recommendations['batch_size_recommendation'] = 'decrease by 25-50%'
            recommendations['reasoning'].append(f'Peak memory usage too high: {peak_memory_util:.1f}%')
            recommendations['current_efficiency'] = 'memory_constrained'
        elif avg_memory_util < 50:
            recommendations['batch_size_recommendation'] = 'increase by 50-100%'
            recommendations['reasoning'].append(f'Low memory usage: {avg_memory_util:.1f}%, can increase batch size')
            recommendations['current_efficiency'] = 'underutilized'
        elif 70 <= avg_memory_util <= 85:
            recommendations['current_efficiency'] = 'optimal'
            recommendations['reasoning'].append(f'Good memory utilization: {avg_memory_util:.1f}%')
        
        # GPU utilization analysis
        if avg_gpu_util < 70:
            if 'increase' not in recommendations['batch_size_recommendation']:
                recommendations['batch_size_recommendation'] = 'increase by 25-50%'
            recommendations['reasoning'].append(f'Low GPU utilization: {avg_gpu_util:.1f}%')
        elif avg_gpu_util > 95:
            recommendations['reasoning'].append(f'High GPU utilization: {avg_gpu_util:.1f}% - good throughput')
        
        # Combined analysis
        if avg_memory_util < 60 and avg_gpu_util < 80:
            recommendations['batch_size_recommendation'] = 'increase by 100-200%'
            recommendations['reasoning'].append('Both memory and GPU underutilized - significant batch size increase recommended')
        
        return recommendations
    
    def print_performance_summary(self):
        """Print a formatted performance summary."""
        summary = self.get_performance_summary()
        
        print(f"\n{'='*80}")
        print(f"🚀 FUSION TRAINING PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Device: {summary['device_name']} ({summary['total_memory_gb']:.1f}GB)")
        print(f"Stats collected from {summary['stats_collected']} episodes")
        
        print(f"\n📊 RESOURCE UTILIZATION:")
        print(f"  Average Memory Usage: {summary['average_memory_utilization_pct']:.1f}% ({summary['average_memory_allocated_gb']:.2f}GB)")
        print(f"  Peak Memory Usage: {summary['peak_memory_utilization_pct']:.1f}%")
        print(f"  Average {'GPU' if self.is_cuda else 'CPU'} Utilization: {summary['average_gpu_utilization_pct']:.1f}%")
        
        if 'timing' in summary and summary['timing']:
            timing = summary['timing']
            print(f"\n⏱️ TRAINING PERFORMANCE:")
            print(f"  Total Training Time: {timing['total_training_time_minutes']:.1f} minutes")
            print(f"  Episodes Completed: {timing['total_episodes_trained']:.0f}")
            print(f"  Training Speed: {timing['average_episodes_per_minute']:.2f} episodes/minute")
            print(f"  Time per 100 Episodes: {timing['estimated_time_per_100_episodes_minutes']:.1f} minutes")
        
        rec = summary['recommendations']
        print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS:")
        print(f"  Current Efficiency: {rec['current_efficiency'].upper()}")
        print(f"  Batch Size Recommendation: {rec['batch_size_recommendation']}")
        if rec['reasoning']:
            print(f"  Reasoning:")
            for reason in rec['reasoning']:
                print(f"    • {reason}")
        
        print(f"\n📈 TARGETS FOR OPTIMAL PERFORMANCE:")
        print(f"  Target Memory Utilization: {rec['target_memory_utilization']}")
        print(f"  Target {'GPU' if self.is_cuda else 'CPU'} Utilization: {rec['target_gpu_utilization']}")
        print(f"{'='*80}")

# Configuration Constants
DATASET_PATHS = {
    'hcrl_ch': r"datasets/can-train-and-test-v1.5/hcrl-ch",
    'hcrl_sa': r"datasets/can-train-and-test-v1.5/hcrl-sa",
    'set_01': r"datasets/can-train-and-test-v1.5/set_01",
    'set_02': r"datasets/can-train-and-test-v1.5/set_02",
    'set_03': r"datasets/can-train-and-test-v1.5/set_03",
    'set_04': r"datasets/can-train-and-test-v1.5/set_04",
}

# Fusion weights for composite anomaly scoring
FUSION_WEIGHTS = {
    'node_reconstruction': 0.4,
    'neighborhood_prediction': 0.35,
    'can_id_prediction': 0.25
}

# Publication-Ready Plotting Configuration (Scientific Paper Style)
PLOT_CONFIG = {
    # Font settings - clean, professional
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    
    # Figure settings - high DPI, clean background
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'none',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.facecolor': 'white',
    
    # Line and marker settings - thin, precise
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'patch.linewidth': 0.8,
    
    # Grid and axes - minimal, clean
    'axes.grid': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.axisbelow': True,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'axes.facecolor': 'white',
    
    # Ticks - clean, minimal
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    
    # Colors - scientific, muted palette
    'axes.prop_cycle': plt.cycler('color', [
        '#2E86AB',  # Blue
        '#A23B72',  # Magenta  
        '#F18F01',  # Orange
        '#C73E1D',  # Red
        '#592E83',  # Purple
        '#1B5F40',  # Green
        '#8B4513',  # Brown
        '#708090'   # Slate Gray
    ]),
    
    # LaTeX rendering
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',
    
    # Legend settings - clean, minimal
    'legend.frameon': False,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1
}

# Color schemes for different plot types (Scientific Paper Style)
COLOR_SCHEMES = {
    'training': {
        'accuracy': '#2E86AB',
        'reward': '#1B5F40', 
        'loss': '#C73E1D',
        'q_values': '#592E83'
    },
    'validation': {
        'primary': '#A23B72',
        'secondary': '#8B4513'
    },
    'fusion_analysis': {
        'normal': '#2E86AB',
        'attack': '#A23B72',
        'adaptive': '#F18F01'
    },
    'heatmap': 'RdYlBu_r',
    'contour': 'viridis'
}

def apply_publication_style():
    """Apply publication-ready matplotlib style."""
    plt.rcParams.update(PLOT_CONFIG)
    plt.style.use('default')  # Reset to default first
    for key, value in PLOT_CONFIG.items():
        if key in plt.rcParams:
            plt.rcParams[key] = value

def save_publication_figure(fig, filename, additional_formats=None):
    """Save figure in multiple publication-ready formats.
    
    Args:
        fig: matplotlib figure object
        filename: base filename with extension
        additional_formats: list of additional formats to save ['pdf', 'svg', 'eps'] or None for PNG only
    """
    base_path = filename.rsplit('.', 1)[0]
    
    # Always save PNG (default)
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    # Save additional formats only if specified
    if additional_formats:
        for fmt in additional_formats:
            try:
                fig.savefig(f"{base_path}.{fmt}", bbox_inches='tight', pad_inches=0.05)
            except Exception as e:
                print(f"Warning: Could not save {fmt} format: {e}")

class FusionDataExtractor:
    """Extract anomaly scores and GAT probabilities for fusion training - GPU OPTIMIZED."""
    
    def __init__(self, autoencoder: nn.Module, classifier: nn.Module, 
                 device: str, threshold: float = 0.0):
        self.autoencoder = autoencoder.to(device)
        self.classifier = classifier.to(device)
        self.device = torch.device(device)
        self.threshold = threshold
        
        # Set models to evaluation mode
        self.autoencoder.eval()
        self.classifier.eval()
        
        # Pre-compute fusion weights as tensor for GPU operations
        self.fusion_weights = torch.tensor([
            FUSION_WEIGHTS['node_reconstruction'],
            FUSION_WEIGHTS['neighborhood_prediction'],
            FUSION_WEIGHTS['can_id_prediction']
        ], dtype=torch.float32, device=self.device)
        
        print(f"✓ Fusion Data Extractor initialized (GPU-Optimized) with threshold: {threshold:.4f}")

    def compute_anomaly_scores(self, batch) -> torch.Tensor:
        """FIXED: Memory-efficient computation without massive tensors."""
        with torch.no_grad():
            # Forward pass (normal)
            cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                batch.x, batch.edge_index, batch.batch
            )
            
            # FIXED: Create neighborhood targets more efficiently
            neighbor_targets = self.autoencoder.create_neighborhood_targets(
                batch.x, batch.edge_index, batch.batch
            )
            
            # Node-level errors (efficient)
            node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
            neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                neighbor_logits, neighbor_targets
            ).mean(dim=1)
            canid_pred = canid_logits.argmax(dim=1)
            true_canids = batch.x[:, 0].long()
            canid_errors = (canid_pred != true_canids).float()
            
            # MEMORY OPTIMIZATION: Process graphs in smaller chunks
            num_graphs = batch.batch.max().item() + 1
            chunk_size = min(512, num_graphs)  # Process 512 graphs at a time
            
            graph_errors_list = []
            for chunk_start in range(0, num_graphs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_graphs)
                chunk_graph_errors = torch.zeros(chunk_end - chunk_start, 3, 
                                            device=self.device, dtype=node_errors.dtype)
                
                for i, graph_idx in enumerate(range(chunk_start, chunk_end)):
                    node_mask = (batch.batch == graph_idx)
                    if node_mask.any():
                        graph_node_errors = torch.stack([
                            node_errors[node_mask], 
                            neighbor_errors[node_mask], 
                            canid_errors[node_mask]
                        ], dim=1)
                        chunk_graph_errors[i] = graph_node_errors.max(dim=0)[0]
                
                graph_errors_list.append(chunk_graph_errors)
            
            # Combine chunks
            graph_errors = torch.cat(graph_errors_list, dim=0)
            
            # Weighted composite score (efficient)
            composite_scores = (graph_errors * self.fusion_weights).sum(dim=1)
            return torch.sigmoid(composite_scores * 3 - 1.5)

    def compute_gat_probabilities(self, batch) -> torch.Tensor:
        """
        GPU-accelerated computation of GAT classification probabilities.
        Already efficient - just needs to stay on GPU.
        """
        with torch.no_grad():
            logits = self.classifier(batch)
            probabilities = torch.sigmoid(logits.squeeze())
            return probabilities  # Keep on GPU!

    def extract_fusion_data(self, data_loader: DataLoader, max_samples: int = None) -> Tuple[List, List, List]:
        """FIXED: Eliminate CPU serialization bottleneck."""
        print("🚀 GPU-Optimized Fusion Data Extraction...")
        
        # Pre-allocate GPU tensors to avoid repeated allocation
        device_tensors = {
            'anomaly_scores': [],
            'gat_probs': [],  
            'labels': []
        }
        
        samples_processed = 0
        total_batches = len(data_loader)
        
        # FIXED: Process in larger chunks to reduce Python overhead
        with torch.cuda.stream(torch.cuda.Stream()) if self.device.type == 'cuda' else nullcontext() as stream:
            with tqdm(data_loader, desc="GPU Extraction", total=total_batches, 
                    miniters=max(1, total_batches//20)) as pbar:
                
                for batch_idx, batch in enumerate(pbar):
                    if self.device.type == 'cuda':
                        # Async GPU transfer with stream
                        batch = batch.to(self.device, non_blocking=True)
                        
                    # FIXED: Vectorized computation without intermediate CPU transfers
                    with torch.no_grad():
                        batch_anomaly_scores = self.compute_anomaly_scores(batch)
                        batch_gat_probs = self.compute_gat_probabilities(batch)
                        
                        # Extract labels efficiently (keep on GPU)
                        if hasattr(batch, 'y') and batch.y.shape[0] == batch.num_graphs:
                            batch_labels = batch.y
                        else:
                            # Handle per-node labels -> per-graph labels
                            num_graphs = batch.batch.max().item() + 1
                            batch_labels = torch.zeros(num_graphs, device=self.device, dtype=batch.y.dtype)
                            
                            for graph_idx in range(num_graphs):
                                node_mask = (batch.batch == graph_idx)
                                if node_mask.any():
                                    batch_labels[graph_idx] = batch.y[node_mask].max()
                    
                    # Accumulate on GPU (no CPU transfer yet)
                    device_tensors['anomaly_scores'].append(batch_anomaly_scores)
                    device_tensors['gat_probs'].append(batch_gat_probs)
                    device_tensors['labels'].append(batch_labels)
                    
                    samples_processed += batch.num_graphs
                    
                    # Update progress less frequently for speed
                    if batch_idx % 10 == 0:
                        pbar.set_postfix({
                            'samples': f"{samples_processed:,}",
                            'gpu_util': f"{torch.cuda.utilization():.0f}%" if self.device.type == 'cuda' else "N/A"
                        })
                    
                    if max_samples and samples_processed >= max_samples:
                        break
                    
                    # FIXED: Less frequent GPU cache clearing
                    if batch_idx % 100 == 0 and batch_idx > 0:
                        torch.cuda.empty_cache()
        
        # Single GPU→CPU transfer at the end (minimizes transfer overhead)
        print("📥 Transferring results from GPU to CPU...")
        anomaly_scores = torch.cat(device_tensors['anomaly_scores']).cpu().numpy().tolist()
        gat_probabilities = torch.cat(device_tensors['gat_probs']).cpu().numpy().tolist()
        labels = torch.cat(device_tensors['labels']).cpu().numpy().tolist()
        
        # Clean up GPU memory
        del device_tensors
        torch.cuda.empty_cache()
        
        return anomaly_scores, gat_probabilities, labels


class FusionTrainingPipeline:
    """Complete pipeline for training the fusion agent."""
    
    def __init__(self, num_ids: int, embedding_dim: int = 8, device: str = 'cpu'):
        self.device = torch.device(device)
        self.num_ids = num_ids
        self.embedding_dim = embedding_dim
        self.gpu_info = self._detect_gpu_capabilities()
        
        # Initialize GPU monitoring
        self.gpu_monitor = GPUMonitor(self.device)
        
        # Models (will be loaded)
        self.autoencoder = None
        self.classifier = None
        self.fusion_agent = None
        self.data_extractor = None
        
        # Training data
        self.training_data = None
        self.validation_data = None
        
        print(f"✓ Fusion Training Pipeline initialized on {device}")
        if self.gpu_info:
            print(f"  GPU: {self.gpu_info['name']} ({self.gpu_info['memory_gb']:.1f}GB)")
            print(f"  Optimized batch size: {self.gpu_info['optimal_batch_size']}")
        print(f"  Performance monitoring: {'GPU' if self.device.type == 'cuda' else 'CPU'} tracking enabled")
    
    def _detect_gpu_capabilities(self):
        """Detect GPU capabilities and optimize parameters accordingly."""
        if not torch.cuda.is_available():
            return None
        
        gpu_props = torch.cuda.get_device_properties(self.device)
        memory_gb = gpu_props.total_memory / (1024**3)

        if memory_gb >= 30:  # A100 40GB/80GB
                optimal_batch_size = 32768  # Very large batch for maximum A100 throughput
                buffer_size = 100000  # Keep smaller buffer for speed
                training_steps = 4
        else:
            optimal_batch_size = 16384   # Large batch for good throughput
            buffer_size = 75000   # Keep smaller buffer for speed
            training_steps = 3
        
        return {
            'name': gpu_props.name,
            'memory_gb': memory_gb,
            'optimal_batch_size': optimal_batch_size,
            'buffer_size': buffer_size,
            'training_steps': training_steps,
            'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
        }

    def load_pretrained_models(self, autoencoder_path: str, classifier_path: str):
        """
        Load pre-trained VGAE and GAT models.
        
        Args:
            autoencoder_path: Path to autoencoder checkpoint
            classifier_path: Path to classifier checkpoint
        """
        print(f"\n=== Loading Pre-trained Models ===")
        print(f"Autoencoder: {autoencoder_path}")
        print(f"Classifier: {classifier_path}")
        
        # Initialize models
        self.autoencoder = GraphAutoencoderNeighborhood(
            num_ids=self.num_ids,
            in_channels=11,
            embedding_dim=self.embedding_dim,
            hidden_dim=32,
            latent_dim=32,
            num_encoder_layers=3,
            num_decoder_layers=3,
            encoder_heads=4,
            decoder_heads=4
        ).to(self.device)
        
        self.classifier = GATWithJK(
            num_ids=self.num_ids,
            in_channels=11,
            hidden_channels=32,
            out_channels=1,
            num_layers=5,
            heads=8,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        # Load checkpoints
        try:
            # Load autoencoder
            ae_checkpoint = torch.load(autoencoder_path, map_location=self.device)
            if isinstance(ae_checkpoint, dict):
                ae_state_dict = ae_checkpoint.get('state_dict', ae_checkpoint)
                threshold = ae_checkpoint.get('threshold', 0.0)
            else:
                ae_state_dict = ae_checkpoint
                threshold = 0.0
            
            self.autoencoder.load_state_dict(ae_state_dict)
            
            # Load classifier
            clf_checkpoint = torch.load(classifier_path, map_location=self.device)
            if isinstance(clf_checkpoint, dict):
                clf_state_dict = clf_checkpoint.get('state_dict', clf_checkpoint)
            else:
                clf_state_dict = clf_checkpoint
                
            self.classifier.load_state_dict(clf_state_dict)
            
            print("✓ Models loaded successfully")
            
            # Create data extractor
            self.data_extractor = FusionDataExtractor(
                self.autoencoder, self.classifier, str(self.device), threshold
            )
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e


    def _compute_rewards_vectorized(self, batch_data: List, alphas: np.ndarray) -> np.ndarray:
        """
        Vectorized reward computation - eliminates threading overhead.
        
        Args:
            batch_data: List of (anomaly_score, gat_prob, true_label) tuples
            alphas: Array of alpha values for each sample
        
        Returns:
            Array of computed rewards
        """
        # Convert to numpy arrays for vectorization
        anomaly_scores = np.array([x[0] for x in batch_data])
        gat_probs = np.array([x[1] for x in batch_data])
        true_labels = np.array([x[2] for x in batch_data])
        
        # Vectorized fusion prediction
        fusion_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
        predictions = (fusion_scores > 0.5).astype(int)
        
        # Vectorized base reward
        correct = (predictions == true_labels)
        base_rewards = np.where(correct, 3.0, -3.0)
        
        # Model agreement bonus/penalty (vectorized)
        model_agreement = 1.0 - np.abs(anomaly_scores - gat_probs)
        agreement_bonus = np.where(correct, model_agreement, -(1.0 - model_agreement))
        
        # Confidence bonus (vectorized)
        confidence_bonus = np.zeros_like(base_rewards)
        for i in range(len(batch_data)):
            if correct[i]:
                if true_labels[i] == 1:  # Attack
                    confidence = max(anomaly_scores[i], gat_probs[i])
                else:  # Normal
                    confidence = 1.0 - max(anomaly_scores[i], gat_probs[i])
                confidence_bonus[i] = 0.5 * confidence
            else:
                # Overconfidence penalty
                if predictions[i] == 1:  # False positive
                    confidence_bonus[i] = -1.5 * fusion_scores[i]
                else:  # False negative
                    confidence_bonus[i] = -1.5 * (1.0 - fusion_scores[i])
        
        # Balance bonus
        balance_bonus = 0.3 * (1.0 - np.abs(alphas - 0.5) * 2)
        
        # Total reward
        total_rewards = base_rewards + agreement_bonus + confidence_bonus + balance_bonus
        
        # Normalize to [-1, 1] range
        return np.clip(total_rewards * 0.5, -1.0, 1.0)


    def _process_experience_batch_gpu(self, batch_indices: List[int], batch_data: List) -> Tuple:
        """
        GPU-accelerated batch processing of experiences.
        Eliminates CPU serialization by doing everything on GPU.
        
        Args:
            batch_indices: Indices of samples in this batch
            batch_data: List of (anomaly_score, gat_prob, true_label) tuples
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch_size = len(batch_indices)
        
        # Get pre-computed states from GPU (instant access!)
        batch_states = self.training_states_gpu[batch_indices]  # [batch_size, state_dim]
        
        # Batch action selection on GPU (vectorized)
        with torch.no_grad():
            q_values = self.fusion_agent.q_network(batch_states)  # [batch_size, num_actions]
            
            # Epsilon-greedy (vectorized)
            if np.random.rand() < self.fusion_agent.epsilon:
                # Random actions
                actions = torch.randint(0, self.fusion_agent.action_dim, (batch_size,), device=self.device)
            else:
                # Greedy actions
                actions = q_values.argmax(dim=1)
        
        # Get alphas for reward computation
        alphas = self.fusion_agent.alpha_values[actions.cpu().numpy()]
        
        # Vectorized reward computation (CPU is still faster for small Python logic)
        rewards = self._compute_rewards_vectorized(batch_data, alphas)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # Get next states
        next_indices = [min(idx + 1, len(self.training_data) - 1) for idx in batch_indices]
        batch_next_states = self.training_states_gpu[next_indices]
        
        # Determine done flags
        dones = torch.tensor(
            [idx + 1 >= len(self.training_data) for idx in batch_indices],
            dtype=torch.float32,
            device=self.device
        )
        
        return batch_states, actions, rewards_tensor, batch_next_states, dones




    def prepare_fusion_data(self, train_loader: DataLoader, val_loader: DataLoader, 
                        max_train_samples: int = 200000, max_val_samples: int = 50000):
        """
        GPU-optimized preparation of training and validation data for fusion learning.
        Pre-computes and caches all states on GPU to eliminate CPU bottleneck.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_train_samples: Maximum training samples to use
            max_val_samples: Maximum validation samples to use
        """
        print(f"\n=== Preparing Fusion Data (GPU-Optimized) ===")
        
        # Extract training data
        print("Extracting training data...")
        train_anomaly_scores, train_gat_probs, train_labels = self.data_extractor.extract_fusion_data(
            train_loader, max_train_samples
        )
        
        # Extract validation data
        print("Extracting validation data...")
        val_anomaly_scores, val_gat_probs, val_labels = self.data_extractor.extract_fusion_data(
            val_loader, max_val_samples
        )
        
        # Store as tuples for easy access
        self.training_data = list(zip(train_anomaly_scores, train_gat_probs, train_labels))
        self.validation_data = list(zip(val_anomaly_scores, val_gat_probs, val_labels))
        self.test_data = self.validation_data  # val_loader is actually test data in main()
        
        print(f"✓ Fusion data prepared:")
        print(f"  Training samples: {len(self.training_data)}")
        print(f"  Validation samples: {len(self.validation_data)}")
        
        # ===== NEW: Pre-compute all states on GPU =====
        if self.device.type == 'cuda':
            print("🚀 Pre-computing states on GPU for maximum throughput...")
            
            try:
                # Pre-compute training states
                train_states_list = []
                for anomaly_score, gat_prob, _ in self.training_data:
                    state = self.fusion_agent.normalize_state(anomaly_score, gat_prob)
                    train_states_list.append(state)
                
                self.training_states_gpu = torch.tensor(
                    np.array(train_states_list), 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                # Pre-compute validation states
                val_states_list = []
                for anomaly_score, gat_prob, _ in self.validation_data:
                    state = self.fusion_agent.normalize_state(anomaly_score, gat_prob)
                    val_states_list.append(state)
                
                self.validation_states_gpu = torch.tensor(
                    np.array(val_states_list), 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                print(f"✓ GPU state cache created: {self.training_states_gpu.shape} training, {self.validation_states_gpu.shape} validation")
                print(f"  GPU memory used: {self.training_states_gpu.element_size() * self.training_states_gpu.nelement() / (1024**2):.2f}MB")
                
            except RuntimeError as e:
                print(f"⚠️ GPU memory error during state precomputation: {e}")
                print("Falling back to CPU computation during training...")
                self.training_states_gpu = None
                self.validation_states_gpu = None
        else:
            # Ensure GPU state tensors are None for CPU mode
            self.training_states_gpu = None
            self.validation_states_gpu = None
        
        # Data quality checks
        self._analyze_fusion_data()

    def _analyze_fusion_data(self):
        """Analyze the extracted fusion data for quality and distribution."""
        if not self.training_data:
            return
        
        train_anomaly_scores = [x[0] for x in self.training_data]
        train_gat_probs = [x[1] for x in self.training_data]
        train_labels = [x[2] for x in self.training_data]
        
        print(f"\n📊 Fusion Data Analysis:")
        print(f"Anomaly Score Range: [{min(train_anomaly_scores):.3f}, {max(train_anomaly_scores):.3f}]")
        print(f"GAT Probability Range: [{min(train_gat_probs):.3f}, {max(train_gat_probs):.3f}]")
        print(f"Class Distribution: Normal={train_labels.count(0)}, Attack={train_labels.count(1)}")
        
        # Correlation analysis
        normal_data = [(a, g) for a, g, l in self.training_data if l == 0]
        attack_data = [(a, g) for a, g, l in self.training_data if l == 1]
        
        if normal_data:
            normal_anomaly_avg = np.mean([x[0] for x in normal_data])
            normal_gat_avg = np.mean([x[1] for x in normal_data])
            print(f"Normal Class Averages: Anomaly={normal_anomaly_avg:.3f}, GAT={normal_gat_avg:.3f}")
        
        if attack_data:
            attack_anomaly_avg = np.mean([x[0] for x in attack_data])
            attack_gat_avg = np.mean([x[1] for x in attack_data])
            print(f"Attack Class Averages: Anomaly={attack_anomaly_avg:.3f}, GAT={attack_gat_avg:.3f}")

    def initialize_fusion_agent(self, alpha_steps: int = 21, lr: float = 1e-3, 
                               epsilon: float = 0.3, buffer_size: int = 100000,
                               batch_size: int = 16384, target_update_freq: int = 100,
                               config_dict: dict = None):
        """Initialize fusion agent."""
        state_dim = 4  # anomaly_score, gat_prob, disagreement, avg_confidence
        
        # GPU optimization
        if self.device.type == 'cuda' and self.gpu_info:
            batch_size = max(batch_size, self.gpu_info['optimal_batch_size'])
            buffer_size = max(buffer_size, self.gpu_info['buffer_size'])
        elif self.device.type == 'cuda':
            batch_size = max(batch_size, 16384)
            buffer_size = max(buffer_size, 500000)
            
        self.fusion_agent = EnhancedDQNFusionAgent(
            alpha_steps=alpha_steps, lr=lr, 
            gamma=config_dict.get('fusion_gamma', 0.95) if config_dict else 0.95, 
            epsilon=epsilon,
            epsilon_decay=config_dict.get('fusion_epsilon_decay', 0.995) if config_dict else 0.995,
            min_epsilon=config_dict.get('fusion_min_epsilon', 0.1) if config_dict else 0.1,
            buffer_size=buffer_size, batch_size=batch_size, target_update_freq=target_update_freq,
            device=str(self.device), state_dim=state_dim
        )
        
        print(f"✓ Fusion Agent initialized with {state_dim}D state space")
        print(f"  Batch Size: {batch_size:,} samples")
        print(f"  Buffer Size: {buffer_size:,} experiences")
        if self.gpu_info:
            print(f"  GPU-Optimized: Using {self.gpu_info['name']} configuration")

    def _get_curriculum_phase(self, episode: int, total_episodes: int) -> dict:
        """Return default sampling strategy (simplified)."""
        return {'phase': 'natural', 'high_disagreement_prob': 0.2, 
               'extreme_confidence_prob': 0.2, 'balanced_prob': 0.6}



    def _sample_by_curriculum(self, episode: int, total_episodes: int, episode_size: int) -> List:
        """Sample training data (simplified without curriculum learning)."""
        # Simple random sampling without curriculum complexity
        return random.sample(self.training_data, min(episode_size, len(self.training_data)))
    

    def train_fusion_agent(self, episodes: int = 50, validation_interval: int = 25,
                        early_stopping_patience: int = 50, save_interval: int = 50,
                        dataset_key: str = 'default', config_dict: dict = None):
        """
        GPU-optimized fusion agent training with batch accumulation for high GPU utilization.
        
        Strategy: Load small batches from disk → Accumulate → Process large batches on GPU
        
        Args:
            episodes: Number of training episodes
            validation_interval: Episodes between validation checks  
            early_stopping_patience: Episodes to wait before early stopping
            save_interval: Episodes between model saves
            dataset_key: Dataset identifier for saving
            config_dict: Additional configuration parameters
        """
        print(f"\n=== Training Fusion Agent (GPU-Optimized with Batch Accumulation) ===")
        print(f"Episodes: {episodes}, Validation every {validation_interval} episodes")
        
        if not self.training_data or not self.fusion_agent:
            raise ValueError("Training data and fusion agent must be initialized first")
        
        # ===== MULTI-LEVEL BATCH SIZE CONFIGURATION =====
        # 1. DataLoader batch_size: Small (prevents OOM during disk loading/graph creation)
        # 2. GPU Processing batch_size: Large (maximizes GPU utilization)
        # 3. DQN Training batch_size: Optimal for learning stability
        
        if self.gpu_info and self.device.type == 'cuda':
            if self.gpu_info['memory_gb'] >= 30:  # A100
                # Small disk batches, large GPU processing
                disk_batch_size = 512          # Small for memory safety
                gpu_processing_batch = 8192    # Large for GPU efficiency
                dqn_training_batch = 4096      # Optimal for Q-learning
                batch_accumulation_factor = 16  # Accumulate 16 small batches
                training_steps_per_episode = 12
                episode_sample_ratio = 0.4
            elif self.gpu_info['memory_gb'] >= 15:  # RTX 3090/4090
                disk_batch_size = 256
                gpu_processing_batch = 4096
                dqn_training_batch = 2048
                batch_accumulation_factor = 16
                training_steps_per_episode = 8
                episode_sample_ratio = 0.3
            else:  # Smaller GPUs
                disk_batch_size = 128
                gpu_processing_batch = 2048
                dqn_training_batch = 1024
                batch_accumulation_factor = 16
                training_steps_per_episode = 6
                episode_sample_ratio = 0.25
        else:
            # CPU mode - keep simple
            disk_batch_size = 256
            gpu_processing_batch = 512
            dqn_training_batch = 256
            batch_accumulation_factor = 2
            training_steps_per_episode = 3
            episode_sample_ratio = 0.15
        
        print(f"🎯 Multi-Level Batch Configuration:")
        print(f"  Disk Loading Batch: {disk_batch_size:,} (prevents OOM)")
        print(f"  GPU Processing Batch: {gpu_processing_batch:,} (high GPU utilization)")
        print(f"  DQN Training Batch: {dqn_training_batch:,} (optimal learning)")
        print(f"  Accumulation Factor: {batch_accumulation_factor}x")
        print(f"  Training Steps per Episode: {training_steps_per_episode}")
        
        # Training setup
        training_start_time = time.time()
        episode_rewards = []
        episode_accuracies = []
        episode_losses = []
        episode_q_values = []
        action_distributions = []
        validation_scores = []
        best_validation_score = -float('inf')
        patience_counter = 0
        
        # Create checkpoint directory
        checkpoint_dir = f"saved_models/fusion_checkpoints/{dataset_key}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # ===== MAIN TRAINING LOOP =====
        for episode in range(episodes):
            episode_start_time = time.time()
            
            # Determine episode sample size
            episode_size = max(1000, int(len(self.training_data) * episode_sample_ratio))
            episode_indices = np.random.choice(len(self.training_data), 
                                            min(episode_size, len(self.training_data)), 
                                            replace=False)
            
            # Progress reporting
            if episode % 10 == 0:
                elapsed = time.time() - training_start_time
                eps_per_min = (episode / elapsed * 60) if elapsed > 0 else 0
                print(f"\n📊 Episode {episode}/{episodes} | {eps_per_min:.1f} eps/min | "
                    f"ε={self.fusion_agent.epsilon:.3f} | Buffer: {len(self.fusion_agent.replay_buffer):,}")
            
            # Episode statistics
            episode_reward = 0.0
            episode_correct = 0
            episode_samples = 0
            episode_losses_list = []
            episode_q_list = []
            episode_action_counts = np.zeros(len(self.fusion_agent.alpha_values))
            
            # ===== BATCH ACCUMULATION STRATEGY =====
            # Process small batches and accumulate for GPU efficiency
            
            if (self.device.type == 'cuda' and 
                hasattr(self, 'training_states_gpu') and 
                self.training_states_gpu is not None):
                
                # ===== GPU-OPTIMIZED PATH WITH BATCH ACCUMULATION =====
                print(f"🚀 GPU batch accumulation: {disk_batch_size} → {gpu_processing_batch}")
                
                # Process episode in chunks that will be accumulated
                for batch_start in range(0, len(episode_indices), gpu_processing_batch):
                    batch_end = min(batch_start + gpu_processing_batch, len(episode_indices))
                    processing_indices = episode_indices[batch_start:batch_end]
                    
                    # STRATEGY: Accumulate multiple small disk batches into one large GPU batch
                    accumulated_states = []
                    accumulated_actions = []
                    accumulated_rewards = []
                    accumulated_next_states = []
                    accumulated_dones = []
                    accumulated_data = []
                    
                    # Process in small disk-friendly chunks, accumulate for GPU
                    for small_batch_start in range(0, len(processing_indices), disk_batch_size):
                        small_batch_end = min(small_batch_start + disk_batch_size, len(processing_indices))
                        small_indices = processing_indices[small_batch_start:small_batch_end]
                        
                        # Get pre-computed states (instant GPU access!)
                        small_batch_states = self.training_states_gpu[small_indices]
                        accumulated_states.append(small_batch_states)
                        
                        # Get corresponding data
                        small_batch_data = [self.training_data[idx] for idx in small_indices]
                        accumulated_data.extend(small_batch_data)
                    
                    # Combine all accumulated small batches into one large GPU batch
                    if accumulated_states:
                        large_batch_states = torch.cat(accumulated_states, dim=0)
                        batch_size = large_batch_states.shape[0]
                        
                        # ===== VECTORIZED GPU PROCESSING (HIGH UTILIZATION) =====
                        with torch.no_grad():
                            # Process entire large batch at once on GPU
                            q_values = self.fusion_agent.q_network(large_batch_states)
                            
                            # Vectorized epsilon-greedy for entire batch
                            if np.random.rand() < self.fusion_agent.epsilon:
                                actions = torch.randint(0, self.fusion_agent.action_dim, 
                                                    (batch_size,), device=self.device)
                            else:
                                actions = q_values.argmax(dim=1)
                        
                        # ===== VECTORIZED REWARD COMPUTATION =====
                        # Extract data for vectorized computation
                        batch_anomaly_scores = np.array([x[0] for x in accumulated_data])
                        batch_gat_probs = np.array([x[1] for x in accumulated_data])
                        batch_true_labels = np.array([x[2] for x in accumulated_data])
                        batch_alphas = self.fusion_agent.alpha_values[actions.cpu().numpy()]
                        
                        # Vectorized fusion and reward computation
                        fusion_scores = (1 - batch_alphas) * batch_anomaly_scores + batch_alphas * batch_gat_probs
                        predictions = (fusion_scores > 0.5).astype(int)
                        
                        # Vectorized reward computation (much faster than loops!)
                        correct_mask = (predictions == batch_true_labels)
                        base_rewards = np.where(correct_mask, 3.0, -3.0)
                        
                        # Model agreement bonus (vectorized)
                        model_agreement = 1.0 - np.abs(batch_anomaly_scores - batch_gat_probs)
                        agreement_bonus = np.where(correct_mask, model_agreement, -(1.0 - model_agreement))
                        
                        # Confidence bonus (vectorized)
                        confidence_bonus = np.zeros_like(base_rewards)
                        for i in range(len(accumulated_data)):
                            if correct_mask[i]:
                                if batch_true_labels[i] == 1:  # Attack
                                    confidence = max(batch_anomaly_scores[i], batch_gat_probs[i])
                                else:  # Normal
                                    confidence = 1.0 - max(batch_anomaly_scores[i], batch_gat_probs[i])
                                confidence_bonus[i] = 0.5 * confidence
                            else:
                                # Overconfidence penalty
                                if predictions[i] == 1:  # False positive
                                    confidence_bonus[i] = -1.5 * fusion_scores[i]
                                else:  # False negative
                                    confidence_bonus[i] = -1.5 * (1.0 - fusion_scores[i])
                        
                        # Balance bonus (vectorized)
                        balance_bonus = 0.3 * (1.0 - np.abs(batch_alphas - 0.5) * 2)
                        
                        # Total rewards (vectorized)
                        total_rewards = base_rewards + agreement_bonus + confidence_bonus + balance_bonus
                        normalized_rewards = np.clip(total_rewards * 0.5, -1.0, 1.0)
                        
                        # ===== NEXT STATES AND EXPERIENCE STORAGE =====
                        # Get next states efficiently
                        next_indices = [min(idx + 1, len(self.training_data) - 1) 
                                    for idx in processing_indices[:len(accumulated_data)]]
                        large_batch_next_states = self.training_states_gpu[next_indices]
                        
                        # Done flags
                        dones = torch.tensor(
                            [idx + 1 >= len(self.training_data) for idx in processing_indices[:len(accumulated_data)]],
                            dtype=torch.float32, device=self.device
                        )
                        
                        # Store experiences efficiently (batch storage)
                        for i in range(len(accumulated_data)):
                            self.fusion_agent.store_experience(
                                large_batch_states[i].cpu().numpy(),
                                actions[i].item(),
                                normalized_rewards[i],
                                large_batch_next_states[i].cpu().numpy(),
                                dones[i].item()
                            )
                        
                        # Update episode statistics
                        episode_reward += np.sum(normalized_rewards)
                        episode_correct += np.sum(correct_mask)
                        episode_samples += len(accumulated_data)
                        
                        # Update action counts
                        for action_idx in actions.cpu().numpy():
                            episode_action_counts[action_idx] += 1
            
            else:
                # ===== CPU FALLBACK PATH =====
                print("🔄 CPU fallback mode - processing individually")
                
                for batch_start in range(0, len(episode_indices), disk_batch_size):
                    batch_end = min(batch_start + disk_batch_size, len(episode_indices))
                    batch_indices = episode_indices[batch_start:batch_end]
                    
                    for idx in batch_indices:
                        anomaly_score, gat_prob, true_label = self.training_data[idx]
                        
                        # Get current state
                        current_state = self.fusion_agent.normalize_state(anomaly_score, gat_prob)
                        
                        # Select action
                        alpha, action_idx, _ = self.fusion_agent.select_action(
                            anomaly_score, gat_prob, training=True
                        )
                        
                        # Compute reward
                        fusion_score = (1 - alpha) * anomaly_score + alpha * gat_prob
                        prediction = 1 if fusion_score > 0.5 else 0
                        raw_reward = self.fusion_agent.compute_fusion_reward(
                            prediction, true_label, anomaly_score, gat_prob, alpha
                        )
                        normalized_reward = np.clip(raw_reward * 0.5, -1.0, 1.0)
                        
                        # Get next state
                        next_idx = min(idx + 1, len(self.training_data) - 1)
                        next_anomaly, next_gat, _ = self.training_data[next_idx]
                        next_state = self.fusion_agent.normalize_state(next_anomaly, next_gat)
                        done = (idx + 1 >= len(self.training_data))
                        
                        # Store experience
                        self.fusion_agent.store_experience(
                            current_state, action_idx, normalized_reward, next_state, done
                        )
                        
                        # Update statistics
                        episode_reward += normalized_reward
                        episode_correct += (prediction == true_label)
                        episode_action_counts[action_idx] += 1
                    
                    episode_samples += len(batch_indices)
            
            # ===== DQN TRAINING PHASE =====
            # Train with large, stable batches for good learning
            if len(self.fusion_agent.replay_buffer) >= dqn_training_batch:
                for training_step in range(training_steps_per_episode):
                    loss = self.fusion_agent.train_step()
                    
                    if loss is not None:
                        episode_losses_list.append(loss)
                        
                        # Sample Q-values for monitoring (less frequently)
                        if training_step == 0:
                            with torch.no_grad():
                                sample_idx = np.random.randint(0, len(self.training_data))
                                if (hasattr(self, 'training_states_gpu') and 
                                    self.training_states_gpu is not None):
                                    sample_state = self.training_states_gpu[sample_idx:sample_idx+1]
                                else:
                                    anomaly_s, gat_p, _ = self.training_data[sample_idx]
                                    sample_state = torch.tensor(
                                        self.fusion_agent.normalize_state(anomaly_s, gat_p),
                                        dtype=torch.float32, device=self.device
                                    ).unsqueeze(0)
                                
                                q_vals = self.fusion_agent.q_network(sample_state)
                                episode_q_list.append(q_vals.mean().item())
            
            # ===== END EPISODE PROCESSING =====
            # Decay exploration
            self.fusion_agent.decay_epsilon()
            
            # Calculate episode metrics
            episode_accuracy = episode_correct / episode_samples if episode_samples > 0 else 0
            avg_episode_reward = episode_reward / episode_samples if episode_samples > 0 else 0
            avg_episode_loss = np.mean(episode_losses_list) if episode_losses_list else 0
            avg_q_value = np.mean(episode_q_list) if episode_q_list else 0
            
            # Store episode statistics
            episode_rewards.append(avg_episode_reward)
            episode_accuracies.append(episode_accuracy)
            episode_losses.append(avg_episode_loss)
            episode_q_values.append(avg_q_value)
            action_distributions.append(episode_action_counts / max(episode_samples, 1))
            
            # ===== VALIDATION AND CHECKPOINTING =====
            if (episode + 1) % validation_interval == 0:
                val_results = self.fusion_agent.validate_agent(
                    self.validation_data, 
                    num_samples=min(5000, len(self.validation_data))
                )
                validation_scores.append(val_results)
                
                print(f"  📈 Validation: Acc={val_results['accuracy']:.4f}, "
                    f"Reward={val_results['avg_reward']:.3f}, α={val_results['avg_alpha']:.3f}")
                
                # Early stopping check
                if val_results['accuracy'] > best_validation_score:
                    best_validation_score = val_results['accuracy']
                    patience_counter = 0
                    self.save_fusion_agent(checkpoint_dir, f"best_ep{episode}", dataset_key)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"⏹️  Early stopping at episode {episode} (patience: {early_stopping_patience})")
                    break
            
            # Periodic checkpoint save
            if (episode + 1) % save_interval == 0:
                self.save_fusion_agent(checkpoint_dir, f"checkpoint_ep{episode}", dataset_key)
            
            # GPU monitoring
            if episode % 10 == 0:
                self.gpu_monitor.record_gpu_stats(episode)
                episode_time = time.time() - episode_start_time
                if episode > 0:
                    print(f"  ⏱️  Episode time: {episode_time:.2f}s, "
                        f"Acc: {episode_accuracy:.3f}, Reward: {avg_episode_reward:.3f}")
                    
                    # GPU utilization feedback
                    if hasattr(self, 'training_states_gpu') and self.training_states_gpu is not None:
                        print(f"  🎯 Batch strategy: {disk_batch_size}×{batch_accumulation_factor} → {gpu_processing_batch} GPU batch")
        
        # ===== TRAINING COMPLETION =====
        total_training_time = time.time() - training_start_time
        print(f"\n✅ Fusion agent training completed in {total_training_time/60:.1f} minutes!")
        print(f"🏆 Best validation accuracy: {best_validation_score:.4f}")
        
        # Performance summary
        self.gpu_monitor.print_performance_summary()
        
        # Generate training plots
        self._plot_enhanced_training_progress(
            episode_accuracies, episode_rewards, episode_losses, 
            episode_q_values, action_distributions, [], validation_scores, dataset_key
        )
        
        return {
            'episode_accuracies': episode_accuracies,
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'episode_q_values': episode_q_values,
            'action_distributions': action_distributions,
            'validation_scores': validation_scores,
            'best_validation_score': best_validation_score,
            'performance_summary': self.gpu_monitor.get_performance_summary(),
            'total_training_time_minutes': total_training_time / 60
        }

    def evaluate_fusion_strategies(self, dataset_key: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation comparing different fusion strategies.
        
        Returns:
            Dictionary with evaluation results for different strategies
        """
        print(f"\n=== Evaluating Fusion Strategies ===")
        
        # Use already extracted test data (from prepare_fusion_data)
        if not hasattr(self, 'test_data') or not self.test_data:
            raise ValueError("No test data available. Run prepare_fusion_data() first.")
            
        test_anomaly_scores = [data[0] for data in self.test_data]
        test_gat_probs = [data[1] for data in self.test_data]
        test_labels = [data[2] for data in self.test_data]
        
        results = {}
        
        # 1. Individual model performance
        print("Evaluating individual models...")
        
        # VGAE only (using threshold)
        vgae_predictions = [1 if score > 0.5 else 0 for score in test_anomaly_scores]
        results['vgae_only'] = self._calculate_metrics(test_labels, vgae_predictions, "VGAE Only")
        
        # GAT only
        gat_predictions = [1 if prob > 0.5 else 0 for prob in test_gat_probs]
        results['gat_only'] = self._calculate_metrics(test_labels, gat_predictions, "GAT Only")
        
        # 2. Fixed fusion strategies
        print("Evaluating fixed fusion strategies...")
        
        fixed_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        for alpha in fixed_alphas:
            fused_scores = [(1 - alpha) * a + alpha * g for a, g in zip(test_anomaly_scores, test_gat_probs)]
            fused_predictions = [1 if score > 0.5 else 0 for score in fused_scores]
            results[f'fixed_alpha_{alpha}'] = self._calculate_metrics(
                test_labels, fused_predictions, f"Fixed α={alpha}"
            )
        
        # 3. Adaptive fusion (trained agent)
        print("Evaluating adaptive fusion...")
        
        adaptive_predictions = []
        adaptive_alphas = []
        
        for anomaly_score, gat_prob in zip(test_anomaly_scores, test_gat_probs):
            alpha, _, _ = self.fusion_agent.select_action(anomaly_score, gat_prob, training=False)
            fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
            prediction = 1 if fused_score > 0.5 else 0
            
            adaptive_predictions.append(prediction)
            adaptive_alphas.append(alpha)
        
        results['adaptive_fusion'] = self._calculate_metrics(
            test_labels, adaptive_predictions, "Adaptive Fusion"
        )
        results['adaptive_fusion']['avg_alpha'] = np.mean(adaptive_alphas)
        results['adaptive_fusion']['alpha_std'] = np.std(adaptive_alphas)
        
        # 4. Print comparison table
        self._print_comparison_table(results)
        
        # 5. Generate analysis plots
        self._plot_fusion_analysis(test_anomaly_scores, test_gat_probs, test_labels, adaptive_alphas, dataset_key)
        
        return results

    def _calculate_metrics(self, true_labels: List, predictions: List, method_name: str) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        return {
            'method': method_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }

    def _print_comparison_table(self, results: Dict):
        """Print a formatted comparison table of all methods."""
        print(f"\n📊 FUSION STRATEGY COMPARISON")
        print("=" * 80)
        print(f"{'Method':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 80)
        
        # Sort by accuracy for better presentation
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for method_key, metrics in sorted_results:
            method_name = metrics['method']
            print(f"{method_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
        
        print("=" * 80)
        
        # Highlight best performance
        best_method = sorted_results[0]
        print(f"🏆 Best Performance: {best_method[1]['method']} "
              f"(Accuracy: {best_method[1]['accuracy']:.4f})")

    def _plot_training_progress(self, accuracies: List, rewards: List, validation_scores: List):
        """Plot training progress visualization with publication-ready styling."""
        apply_publication_style()
        plt.ioff()  # Turn off interactive mode
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        episodes = range(1, len(accuracies) + 1)
        colors = COLOR_SCHEMES['training']
        
        # Training accuracy
        ax1.plot(episodes, accuracies, color=colors['accuracy'], linewidth=2.5, alpha=0.8)
        ax1.set_xlabel('Training Episode')
        ax1.set_ylabel('Training Accuracy')
        ax1.set_title('Training Accuracy Progression', fontweight='bold')
        ax1.set_ylim([0, 1.02])
        
        # Training rewards
        ax2.plot(episodes, rewards, color=colors['reward'], linewidth=2.5, alpha=0.8)
        ax2.set_xlabel('Training Episode')
        ax2.set_ylabel('Average Normalized Reward')
        ax2.set_title('Training Reward Evolution', fontweight='bold')
        
        # Validation accuracy
        if validation_scores:
            val_episodes = [i * 100 for i in range(1, len(validation_scores) + 1)]
            val_accuracies = [score['accuracy'] for score in validation_scores]
            ax3.plot(val_episodes, val_accuracies, color=COLOR_SCHEMES['validation']['primary'], 
                    linewidth=2.5, marker='o', markersize=6, alpha=0.8)
            ax3.set_xlabel('Training Episode')
            ax3.set_ylabel('Validation Accuracy')
            ax3.set_title('Validation Performance', fontweight='bold')
            # Auto-scale validation accuracy to show data range better
            if val_accuracies:
                y_min = max(0, min(val_accuracies) - 0.01)
                y_max = min(1.0, max(val_accuracies) + 0.01)
                ax3.set_ylim([y_min, y_max])
        
        # Fusion weights used
        if validation_scores:
            val_alphas = [score['avg_alpha'] for score in validation_scores]
            ax4.plot(val_episodes, val_alphas, color=colors['q_values'], 
                    linewidth=2.5, marker='s', markersize=6, alpha=0.8, label='Learned α')
            ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Balanced Fusion')
            ax4.set_xlabel('Training Episode')
            ax4.set_ylabel('Average Fusion Weight (α)')
            ax4.set_title('Adaptive Fusion Strategy', fontweight='bold')
            ax4.legend(loc='best')
        
        plt.tight_layout()
        save_publication_figure(fig, 'images/fusion_training_progress.png')
        plt.close(fig)
        plt.ion()  # Turn interactive mode back on

    def _plot_fusion_analysis(self, anomaly_scores: List, gat_probs: List, 
                            labels: List, adaptive_alphas: List,
                            dataset_key: str):
        """Add fusion analysis plots to the existing training progress figure."""
        # Check if we have the figure from training progress
        if not hasattr(self, '_current_fig') or not hasattr(self, '_current_axes'):
            # Fallback: create new figure if training plots weren't called first
            apply_publication_style()
            plt.ioff()
            fig, axes = plt.subplots(2, 4, figsize=(32, 16))
            # Hide first 5 subplots if training wasn't called
            for i in range(2):
                for j in range(4):
                    if i == 0 or (i == 1 and j == 0):
                        axes[i,j].set_visible(False)
        else:
            fig = self._current_fig
            axes = self._current_axes
        
        # Convert to numpy arrays
        anomaly_scores = np.array(anomaly_scores)
        gat_probs = np.array(gat_probs)
        labels = np.array(labels)
        adaptive_alphas = np.array(adaptive_alphas)
        
        colors = COLOR_SCHEMES['fusion_analysis']
        
        # Row 2, Col 2: State space visualization - simplified scatter plot
        x_min, x_max = max(0, anomaly_scores.min() - 0.02), min(1, anomaly_scores.max() + 0.02)
        y_min, y_max = max(0, gat_probs.min() - 0.02), min(1, gat_probs.max() + 0.02)
        
        scatter = axes[1,1].scatter(anomaly_scores, gat_probs, c=adaptive_alphas, 
                                  cmap=COLOR_SCHEMES['contour'], s=12, alpha=0.7, 
                                  edgecolors='white', linewidths=0.1)
        
        axes[1,1].set_xlabel('VGAE Anomaly Score')
        axes[1,1].set_ylabel('GAT Classification Probability')
        axes[1,1].set_title('Learned Fusion Policy', fontweight='bold')
        axes[1,1].set_xlim([x_min, x_max])
        axes[1,1].set_ylim([y_min, y_max])
        
        cbar = plt.colorbar(scatter, ax=axes[1,1], shrink=0.8)
        cbar.set_label('Fusion Weight (α)', rotation=270, labelpad=15, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        
        # Row 2, Col 3: Fusion weight distribution by class
        normal_alphas = adaptive_alphas[labels == 0]
        attack_alphas = adaptive_alphas[labels == 1]
        
        bins = 25
        n_normal, bins_normal, _ = axes[1,2].hist(normal_alphas, bins=bins, alpha=0.7, 
                                               color=colors['normal'], edgecolor='black', linewidth=0.8,
                                               label='Normal (Count)', histtype='bar')
        n_attack, bins_attack, _ = axes[1,2].hist(attack_alphas, bins=bins, alpha=0.7,
                                               color=colors['attack'], edgecolor='black', linewidth=0.8,
                                               label='Attack (Count)', histtype='bar')
        
        axes[1,2].set_xlabel('Fusion Weight (α)')
        axes[1,2].set_ylabel('Raw Sample Count', color='black')
        axes[1,2].set_xlim([0, 1])
        axes[1,2].tick_params(axis='y', labelcolor='black')
        
        # Secondary axis - proportional distribution
        ax2_twin = axes[1,2].twinx()
        
        bin_centers = (bins_normal[:-1] + bins_normal[1:]) / 2
        normal_prop = n_normal / len(normal_alphas) if len(normal_alphas) > 0 else n_normal
        attack_prop = n_attack / len(attack_alphas) if len(attack_alphas) > 0 else n_attack
        
        ax2_twin.plot(bin_centers, normal_prop, color=colors['normal'], linewidth=1.5, 
                     linestyle='-', marker='o', markersize=2, alpha=0.9, label='Normal (Proportion)')
        ax2_twin.plot(bin_centers, attack_prop, color=colors['attack'], linewidth=1.5,
                     linestyle='-', marker='s', markersize=2, alpha=0.9, label='Attack (Proportion)')
        
        ax2_twin.set_ylabel('Proportional Distribution', color='gray')
        ax2_twin.tick_params(axis='y', labelcolor='gray')
        
        lines1, labels1 = axes[1,2].get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        axes[1,2].legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        axes[1,2].set_title('Fusion Strategy Distribution', fontweight='bold')
        
        # Row 2, Col 4: Model agreement analysis - simplified
        model_diff = np.abs(anomaly_scores - gat_probs)
        jitter_amount = 0.01
        jittered_alphas = adaptive_alphas + np.random.normal(0, jitter_amount, len(adaptive_alphas))
        
        normal_mask = labels == 0
        attack_mask = labels == 1
        
        axes[1,3].scatter(model_diff[normal_mask], jittered_alphas[normal_mask], 
                         alpha=0.6, s=12, c=colors['normal'], 
                         edgecolors='white', linewidths=0.3, label='Normal Traffic')
        
        axes[1,3].scatter(model_diff[attack_mask], jittered_alphas[attack_mask],
                         alpha=0.8, s=15, c=colors['attack'], 
                         edgecolors='white', linewidths=0.3, label='Attack Traffic')
        
        # Add horizontal reference lines
        key_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        for alpha_val in key_alphas:
            axes[1,3].axhline(y=alpha_val, color='gray', alpha=0.2, linewidth=0.5, linestyle=':')
        
        axes[1,3].set_xlabel('Model Disagreement |VGAE Score - GAT Probability|')
        axes[1,3].set_ylabel('Fusion Weight (α)')
        axes[1,3].set_title('Strategy vs. Model Agreement', fontweight='bold')
        axes[1,3].set_xlim([0, 1])
        axes[1,3].set_ylim([0, 1])
        axes[1,3].legend(loc='best', fontsize=9)
        
        # Update the saved figure
        plt.tight_layout()
        filename = f'images/complete_fusion_training_analysis_{dataset_key}'
        save_publication_figure(fig, filename + '.png')
        plt.close(fig)
        plt.ion()

    def save_fusion_agent(self, save_folder: str, suffix: str = "final", dataset_key: str = "default"):
        """Save the trained fusion agent."""
        os.makedirs(save_folder, exist_ok=True)
        filepath = os.path.join(save_folder, f'fusion_agent_{dataset_key}_{suffix}.pth')
        self.fusion_agent.save_agent(filepath)
        
        # Also save configuration
        config_path = os.path.join(save_folder, f'fusion_config_{dataset_key}_{suffix}.json')
        import json
        config = {
            'num_ids': self.num_ids,
            'embedding_dim': self.embedding_dim,
            'alpha_steps': len(self.fusion_agent.alpha_values),
            'fusion_weights': FUSION_WEIGHTS,
            'device': str(self.device)
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Fusion agent and config saved to {save_folder}")
    
    def _analyze_q_values_for_sample_states(self):
        """Analyze Q-values for sample states to check learning."""
        if not self.training_data:
            return
            
        # Sample a few representative states
        sample_indices = [0, len(self.training_data)//4, len(self.training_data)//2, 
                         3*len(self.training_data)//4, len(self.training_data)-1]
        
        print(f"  Q-Value Analysis for Sample States:")
        for i, idx in enumerate(sample_indices[:3]):  # Just show first 3
            if idx < len(self.training_data):
                anomaly_score, gat_prob, true_label = self.training_data[idx]
                state = self.fusion_agent.normalize_state(anomaly_score, gat_prob)
                
                with torch.no_grad():
                    device = getattr(self.fusion_agent, 'device', self.device)
                    q_network = getattr(self.fusion_agent, 'q_network', None)
                    alpha_values = getattr(self.fusion_agent, 'alpha_values', [0.0, 0.25, 0.5, 0.75, 1.0])
                    
                    if q_network is not None:
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        q_values = q_network(state_tensor).squeeze()
                        best_action = torch.argmax(q_values).item()
                        best_alpha = alpha_values[best_action] if best_action < len(alpha_values) else 0.5
                        max_q = q_values[best_action].item()
                    else:
                        best_alpha = 0.5  # Fallback
                        max_q = 0.0
                
                print(f"    State{i+1}: VGAE={anomaly_score:.3f}, GAT={gat_prob:.3f}, Label={true_label} "
                      f"→ Best α={best_alpha:.2f} (Q={max_q:.3f})")

    def _plot_enhanced_training_progress(self, accuracies, rewards, losses, q_values, 
                                       action_distributions, reward_stats, validation_scores, dataset_key):
        """Enhanced training progress visualization with publication-ready styling."""
        apply_publication_style()
        plt.ioff()  # Turn off interactive mode
        fig, axes = plt.subplots(2, 4, figsize=(32, 16))  # Single 2x4 layout for all plots
        episodes = range(1, len(accuracies) + 1)
        colors = COLOR_SCHEMES['training']
        
        # Row 1, Col 1: Training accuracy
        axes[0,0].plot(episodes, accuracies, color=colors['accuracy'], linewidth=1.5, alpha=0.8)
        axes[0,0].set_xlabel('Training Episode')
        axes[0,0].set_ylabel('Training Accuracy')
        axes[0,0].set_title('Training Accuracy Progression', fontweight='bold')
        if accuracies:
            y_min = max(0, min(accuracies) - 0.01)
            y_max = min(1.0, max(accuracies) + 0.01)
            axes[0,0].set_ylim([y_min, y_max])
        
        # Row 1, Col 2: Training rewards
        axes[0,1].plot(episodes, rewards, color=colors['reward'], linewidth=1.5, alpha=0.8)
        axes[0,1].set_xlabel('Training Episode')
        axes[0,1].set_ylabel('Normalized Reward')
        axes[0,1].set_title('Training Reward Evolution', fontweight='bold')
        
        # Row 1, Col 3: Training losses
        if len(losses) > 1:
            loss_episodes = episodes[1:]
            loss_values = losses[1:]
            axes[0,2].plot(loss_episodes, loss_values, color=colors['loss'], linewidth=1.5, alpha=0.8)
        axes[0,2].set_xlabel('Training Episode')
        axes[0,2].set_ylabel('Average Loss (log scale)')
        axes[0,2].set_title('Training Loss Convergence', fontweight='bold')
        axes[0,2].set_yscale('log')
        
        # Row 1, Col 4: Action distribution heatmap
        if action_distributions:
            action_matrix = np.array(action_distributions).T
            bin_size = 50
            n_episodes = len(action_distributions)
            n_bins = (n_episodes + bin_size - 1) // bin_size
            
            binned_matrix = np.zeros((action_matrix.shape[0], n_bins))
            bin_labels = []
            
            for i in range(n_bins):
                start_ep = i * bin_size
                end_ep = min((i + 1) * bin_size, n_episodes)
                if end_ep > start_ep:
                    binned_matrix[:, i] = np.mean(action_matrix[:, start_ep:end_ep], axis=1)
                    bin_labels.append(f'{start_ep+1}-{end_ep}')
            
            im = axes[0,3].imshow(binned_matrix, aspect='auto', cmap=COLOR_SCHEMES['heatmap'], 
                                origin='lower', interpolation='nearest')
            axes[0,3].set_xlabel('Episode Bins')
            axes[0,3].set_ylabel('Fusion Weight (α)')
            axes[0,3].set_title(f'Action Selection Evolution', fontweight='bold')
            
            alpha_values = getattr(self.fusion_agent, 'alpha_values', [0.0, 0.25, 0.5, 0.75, 1.0])
            alpha_ticks = range(0, len(alpha_values), max(1, len(alpha_values)//5))
            alpha_labels = [f'{alpha_values[i]:.2f}' for i in alpha_ticks if i < len(alpha_values)]
            axes[0,3].set_yticks(alpha_ticks)
            axes[0,3].set_yticklabels(alpha_labels)
            
            x_tick_interval = max(1, n_bins // 6)  # Fewer labels for smaller subplot
            x_ticks = range(0, n_bins, x_tick_interval)
            x_labels = [bin_labels[i] for i in x_ticks]
            axes[0,3].set_xticks(x_ticks)
            axes[0,3].set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
            
            cbar = plt.colorbar(im, ax=axes[0,3], shrink=0.8)
            cbar.set_label('Selection Frequency', fontsize=10)
            cbar.ax.tick_params(labelsize=9)
        
        # Row 2, Col 1: Exploration-Exploitation Balance
        if validation_scores and hasattr(self.fusion_agent, 'epsilon'):
            epsilon_values = []
            action_entropies = []
            
            for action_dist in action_distributions:
                action_probs = action_dist / (action_dist.sum() + 1e-8)
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
                action_entropies.append(entropy)
            
            initial_epsilon = 0.8
            epsilon_decay = 0.99
            for ep in range(len(episodes)):
                current_epsilon = max(0.15, initial_epsilon * (epsilon_decay ** (ep // 5)))
                epsilon_values.append(current_epsilon)
            
            axes[1,0].plot(episodes, epsilon_values, color='#d62728', linewidth=1.5, 
                          alpha=0.9, label='Exploration (ε)')
            axes[1,0].set_xlabel('Training Episode')
            axes[1,0].set_ylabel('Epsilon Value', color='#d62728')
            axes[1,0].tick_params(axis='y', labelcolor='#d62728')
            axes[1,0].set_ylim([0, 1])
            
            ax_twin = axes[1,0].twinx()
            window_size = min(25, len(action_entropies)//10)
            if window_size > 0:
                smoothed_entropy = np.convolve(action_entropies, np.ones(window_size)/window_size, mode='valid')
                entropy_episodes = episodes[window_size-1:]
                ax_twin.plot(entropy_episodes, smoothed_entropy, color='#ff7f0e', 
                           linewidth=1.5, alpha=0.9, label='Action Entropy')
            
            ax_twin.set_ylabel('Action Entropy (bits)', color='#ff7f0e')
            ax_twin.tick_params(axis='y', labelcolor='#ff7f0e')
            
            lines1, labels1 = axes[1,0].get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            axes[1,0].legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
            axes[1,0].set_title('Exploration-Exploitation Balance', fontweight='bold')
        
        # Store for fusion analysis plots
        self._current_fig = fig
        self._current_axes = axes
        
        plt.tight_layout()
        filename = f'images/complete_fusion_training_analysis_{dataset_key}'
        save_publication_figure(fig, filename + '.png')
        plt.close(fig)
        plt.ion()

def calculate_dynamic_resources(dataset_size: int, device: str = 'cuda'):
    """FIXED: More aggressive worker allocation for better CPU utilization."""
    
    # Get allocated CPUs
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        allocated_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        allocated_cpus = 6  # Your SLURM allocation
    
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"🔍 System Resources: {allocated_cpus} Physical CPUs, {memory_gb:.1f}GB RAM")
    print(f"📊 Dataset size: {dataset_size:,} samples")
    
    if torch.cuda.is_available() and 'cuda' in device:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory_gb >= 30:  # A100
            target_batch = 2048
            # AGGRESSIVE: Use 2-3x more workers than CPUs for I/O bound tasks
            num_workers = min(24, allocated_cpus * 4)  # 6 CPUs × 4 = 24 workers max
            prefetch_factor = 4
        elif gpu_memory_gb >= 15:  # RTX 3090/4090
            target_batch = 1024
            num_workers = min(16, allocated_cpus * 3)  # 6 CPUs × 3 = 18 workers max
            prefetch_factor = 3
        else:
            target_batch = 512
            num_workers = min(12, allocated_cpus * 2)  # 6 CPUs × 2 = 12 workers max
            prefetch_factor = 2
        
        print(f"🎯 Optimized for I/O Throughput:")
        print(f"  Physical CPUs: {allocated_cpus}")
        print(f"  DataLoader Workers: {num_workers} ({num_workers/allocated_cpus:.1f}x CPU multiplier)")
        print(f"  Batch Size: {target_batch:,}")
        print(f"  Prefetch Factor: {prefetch_factor}")
        
        return {
            'batch_size': target_batch,
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,
            'pin_memory': True,
            'persistent_workers': True,
            'drop_last': False
        }
    else:
        # CPU mode - still use more workers than cores
        return {
            'batch_size': 512,
            'num_workers': min(8, allocated_cpus * 2),
            'prefetch_factor': 2,
            'pin_memory': False,
            'persistent_workers': False,
            'drop_last': False
        }

def create_optimized_data_loaders(train_subset=None, test_dataset=None, full_train_dataset=None, 
                                 batch_size: int = 1024, device: str = 'cuda'):
    """FIXED: Aggressive worker allocation with fallback protection."""
    
    dataset = next((d for d in [train_subset, test_dataset, full_train_dataset] if d is not None), None)
    if dataset is None:
        raise ValueError("No valid dataset provided")
    
    dataset_size = len(dataset)
    config, cuda_available = calculate_dynamic_resources(dataset_size, device)
    
    print(f"🎯 Aggressive Worker Configuration:")
    print(f"  Target Workers: {config['num_workers']} (for maximum I/O throughput)")
    print(f"  Batch Size: {config['batch_size']:,}")
    print(f"  Expected GPU Utilization Improvement: 3-5x")
    
    if cuda_available:
        torch.cuda.empty_cache()
    
    datasets = [train_subset, test_dataset, full_train_dataset]
    shuffles = [True, False, True]
    
    for dataset, shuffle in zip(datasets, shuffles):
        if dataset is not None:
            try:
                print(f"🚀 Attempting aggressive configuration: {config['num_workers']} workers...")
                
                # Add timeout protection for worker testing
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("DataLoader creation timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30-second timeout
                
                try:
                    loader = DataLoader(
                        dataset,
                        batch_size=config['batch_size'],
                        shuffle=shuffle,
                        pin_memory=config['pin_memory'],
                        num_workers=config['num_workers'],
                        persistent_workers=config['persistent_workers'],
                        prefetch_factor=config['prefetch_factor']
                    )
                    
                    # Quick test batch
                    test_batch = next(iter(loader))
                    signal.alarm(0)  # Cancel timeout
                    
                    print(f"✅ Success! {config['num_workers']} workers on {6} CPUs working efficiently")
                    return loader
                    
                except TimeoutError:
                    signal.alarm(0)
                    print(f"⏰ Timeout with {config['num_workers']} workers - trying conservative config")
                    raise
                    
            except Exception as e:
                signal.alarm(0)  # Ensure timeout is cancelled
                print(f"❌ Failed with {config['num_workers']} workers: {e}")
                
                # Progressive fallback: reduce workers but keep aggressive batching
                fallback_workers = [
                    config['num_workers'] // 2,  # Half workers first
                    config['num_workers'] // 3,  # Then 1/3
                    min(8, config['num_workers'] // 4),  # Conservative
                    4  # Minimum fallback
                ]
                
                for attempt, workers in enumerate(fallback_workers, 1):
                    try:
                        print(f"🔄 Fallback {attempt}: {workers} workers...")
                        loader = DataLoader(
                            dataset,
                            batch_size=config['batch_size'],  # Keep aggressive batch size
                            shuffle=shuffle,
                            pin_memory=config['pin_memory'],
                            num_workers=workers,
                            persistent_workers=config['persistent_workers'],
                            prefetch_factor=config['prefetch_factor']
                        )
                        test_batch = next(iter(loader))
                        print(f"✅ Fallback successful: {workers} workers")
                        return loader
                    except Exception as fe:
                        print(f"❌ Fallback {attempt} failed: {fe}")
                
                raise RuntimeError("All worker configurations failed")
    
    raise ValueError("No valid dataset provided")

@hydra.main(config_path="../conf", config_name="base", version_base=None)
def main(config: DictConfig):
    """
    Main fusion training pipeline.
    
    Args:
        config: Hydra configuration object
    """
    print(f"\n{'='*80}")
    print("FUSION TRAINING FOR CAN BUS ANOMALY DETECTION")
    print(f"{'='*80}")
    
    # Setup
    setup_gpu_optimization()
    config_dict = OmegaConf.to_container(config, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # Dataset configuration
    dataset_key = config_dict['root_folder']
    if dataset_key not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    
    root_folder = DATASET_PATHS[dataset_key]
    
    # Create directories
    for dir_name in ["images", "saved_models", "saved_models/fusion_checkpoints"]:
        os.makedirs(dir_name, exist_ok=True)
    
    # === Data Loading and Preprocessing ===
    print(f"\n=== Data Loading and Preprocessing ===")
    print(f"✓ Dataset: {dataset_key}, Path: {root_folder}")
    
    # Add memory monitoring for preprocessing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧠 Starting preprocessing - GPU memory cleared")
    
    # Timing diagnostics with progress
    print("🔄 Step 1/2: Building ID mapping...")
    io_start_time = time.time()
    id_mapping = build_id_mapping_from_normal(root_folder)
    io_mapping_time = time.time() - io_start_time
    print(f"✓ ID mapping built in {io_mapping_time:.2f}s ({len(id_mapping)} IDs)")

    print("🔄 Step 2/2: Creating graph dataset...")
    if preprocessing_time := io_mapping_time > 60:  # If ID mapping took > 1 min, likely slow I/O
        print("⚠️  Slow I/O detected - using conservative settings...")
    
    start_time = time.time()
    dataset = graph_creation(root_folder, id_mapping=id_mapping, 
                           window_size=config_dict.get('window_size', 100))
    graph_creation_time = time.time() - start_time
    total_preprocessing_time = io_mapping_time + graph_creation_time
    
    print(f"✓ Dataset: {len(dataset)} graphs, {len(id_mapping)} CAN IDs")
    print(f"✓ Graph creation time: {graph_creation_time:.2f}s")
    print(f"✓ Total preprocessing time: {total_preprocessing_time:.2f}s")
    
    if total_preprocessing_time > 300:  # > 5 minutes
        print(f"⚠️  Preprocessing took {total_preprocessing_time/60:.1f} minutes - consider data caching for future runs")
    
    # Configuration (optimized for GPU utilization and speed)
    TRAIN_RATIO = config_dict.get('train_ratio', 0.8)
    BATCH_SIZE = config_dict.get('batch_size', 1024)
    FUSION_EPISODES = config_dict.get('fusion_episodes', 1000)
    ALPHA_STEPS = 21
    FUSION_LR = 0.0005  # Much lower for stability
    FUSION_EPSILON = 0.8  # Higher exploration for better Q-learning
    # Dynamic parameters - will be set by GPU detection and resource calculation
    BUFFER_SIZE = None  # Will be calculated dynamically
    FUSION_BATCH_SIZE = None  # Will be calculated dynamically  
    TARGET_UPDATE_FREQ = None  # Will be calculated dynamically
    
    # Enhanced exploration parameters
    FUSION_EPSILON = 0.9  # Higher initial exploration (was 0.8)
    FUSION_EPSILON_DECAY = 0.995  # Slower decay for longer exploration (was 0.99)
    FUSION_MIN_EPSILON = 0.2  # Higher minimum exploration (was 0.15)

    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train/test split
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    # Create GPU-optimized data loaders
    print("🔧 Creating GPU-optimized data loaders...")
    train_loader = create_optimized_data_loaders(train_dataset, None, None, BATCH_SIZE, str(device))
    test_loader = create_optimized_data_loaders(None, test_dataset, None, BATCH_SIZE, str(device))
    
    print(f"✓ Data split: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # === Initialize Fusion Pipeline ===
    pipeline = FusionTrainingPipeline(
        num_ids=len(id_mapping), 
        embedding_dim=8, 
        device=str(device)
    )
    
    # === Load Pre-trained Models ===
    autoencoder_path = f"saved_models1/autoencoder_best_{dataset_key}.pth"
    classifier_path = f"saved_models1/classifier_{dataset_key}.pth"

    try:
        pipeline.load_pretrained_models(autoencoder_path, classifier_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run osc_training_AD.py first to train the base models!")
        return
    
    # Dynamic sampling based on dataset size and available resources
    def calculate_optimal_sampling(dataset_size: int, device_type: str) -> Dict[str, int]:
        """Calculate optimal train/val sampling based on dataset characteristics."""
        # Dynamic ratios based purely on dataset size and compute power
        if dataset_size < 50000:  # Small datasets - use most data
            train_ratio = 0.9
            val_ratio = 0.3
        elif dataset_size < 200000:  # Medium datasets
            train_ratio = 0.7
            val_ratio = 0.25
        else:  # Large datasets
            train_ratio = 0.5
            val_ratio = 0.15
            
        # Scale up for better hardware
        if device_type == 'cuda':
            train_ratio *= 1.5  # Use more data on GPU
            val_ratio *= 1.2
            
        max_train = int(dataset_size * train_ratio)
        max_val = int(dataset_size * val_ratio)
        
        # Remove artificial caps - let hardware determine limits
        max_train = max(10000, max_train)  # Only minimum bound
        max_val = max(2000, max_val)       # Only minimum bound
        
        return {'max_train': max_train, 'max_val': max_val}
    
    total_samples = len(train_dataset) + len(test_dataset)
    sampling_config = calculate_optimal_sampling(total_samples, device.type)
    
    print(f"🎯 Dynamic sampling for {total_samples:,} total samples:")
    print(f"   Training samples: {sampling_config['max_train']:,}")
    print(f"   Validation samples: {sampling_config['max_val']:,}")
    
    # GPU memory optimization for extraction phase
    if torch.cuda.is_available():
        print("🚀 Pre-allocating GPU memory for accelerated extraction...")
        torch.cuda.empty_cache()  # Clear cache before large extraction
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% instead of 99%
    
    # Initialize fusion agent BEFORE preparing fusion data (needed for normalize_state)
    pipeline.initialize_fusion_agent(
        alpha_steps=config_dict.get('alpha_steps', ALPHA_STEPS),           # Use stable config
        lr=config_dict.get('fusion_lr', FUSION_LR),                       # Use stable learning rate
        epsilon=config_dict.get('fusion_epsilon', FUSION_EPSILON),        # Use enhanced exploration
        buffer_size=config_dict.get('fusion_buffer_size', BUFFER_SIZE), # Use optimized buffer size
        batch_size=config_dict.get('fusion_batch_size', FUSION_BATCH_SIZE), # Use optimized batch size
        target_update_freq=config_dict.get('fusion_target_update', TARGET_UPDATE_FREQ), # Use optimized update freq
        config_dict={**config_dict, 'fusion_epsilon_decay': FUSION_EPSILON_DECAY, 'fusion_min_epsilon': FUSION_MIN_EPSILON}
    )
    
    pipeline.prepare_fusion_data(
        train_loader, 
        test_loader, 
        max_train_samples=sampling_config['max_train'],
        max_val_samples=sampling_config['max_val']
    )
    
    # Train the fusion agent with patience for continued learning
    training_results = pipeline.train_fusion_agent(
        episodes=config_dict.get('fusion_episodes', FUSION_EPISODES),  # Use faster default
        validation_interval=25,   # Less frequent validation for speed
        early_stopping_patience=75,   # Much longer patience - continue learning!
        save_interval=50,         # Less frequent checkpoints for speed
        dataset_key=dataset_key,
        config_dict=config_dict  # Pass config_dict here
    )
    
    # === Final Evaluation ===
    print(f"\n=== Final Evaluation and Comparison ===")
    evaluation_results = pipeline.evaluate_fusion_strategies(dataset_key)
    
    # === Save Final Model ===
    pipeline.save_fusion_agent("saved_models", "final", dataset_key)
    
    # === Summary ===
    print(f"\n🎉 FUSION TRAINING COMPLETED!")
    print(f"Best validation accuracy: {training_results['best_validation_score']:.4f}")
    
    # Find best method from evaluation
    best_method = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"Best fusion method: {best_method[1]['method']} (Acc: {best_method[1]['accuracy']:.4f})")
    
    log_memory_usage("Final")

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        end_time = time.time()
        print(f"\n⏱️  Total runtime: {end_time - start_time:.2f} seconds")
        cleanup_memory()