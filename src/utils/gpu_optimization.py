"""
GPU Optimization Utilities for CAN Bus Training

Reusable GPU monitoring, resource management, and optimization utilities
for GAT, VGAE, and Fusion training pipelines.
"""

import os
import time
import psutil
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from typing import Dict, Tuple
import signal

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
            torch.cuda.synchronize()
            
            memory_allocated = torch.cuda.memory_allocated(self.device)
            memory_reserved = torch.cuda.memory_reserved(self.device)
            memory_free = self.total_memory - memory_reserved
            
            memory_util = (memory_allocated / self.total_memory) * 100
            reserved_util = (memory_reserved / self.total_memory) * 100
            
            # Try to get GPU utilization
            gpu_util = 0.0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except (ImportError, Exception):
                gpu_util = min(95.0, memory_util * 1.2)
            
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
        
        avg_memory_util = np.mean([s['memory_utilization_pct'] for s in self.gpu_stats])
        avg_gpu_util = np.mean([s['gpu_utilization_pct'] for s in self.gpu_stats])
        max_memory_util = max([s['memory_utilization_pct'] for s in self.gpu_stats])
        avg_memory_allocated = np.mean([s['memory_allocated_gb'] for s in self.gpu_stats])
        
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
        
        recommendations = self._generate_batch_recommendations(avg_memory_util, max_memory_util, avg_gpu_util)
        
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
        
        if avg_gpu_util < 70:
            if 'increase' not in recommendations['batch_size_recommendation']:
                recommendations['batch_size_recommendation'] = 'increase by 25-50%'
            recommendations['reasoning'].append(f'Low GPU utilization: {avg_gpu_util:.1f}%')
        elif avg_gpu_util > 95:
            recommendations['reasoning'].append(f'High GPU utilization: {avg_gpu_util:.1f}% - good throughput')
        
        if avg_memory_util < 60 and avg_gpu_util < 80:
            recommendations['batch_size_recommendation'] = 'increase by 100-200%'
            recommendations['reasoning'].append('Both memory and GPU underutilized - significant batch size increase recommended')
        
        return recommendations
    
    def print_performance_summary(self):
        """Print a formatted performance summary."""
        summary = self.get_performance_summary()
        
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING PERFORMANCE SUMMARY")
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


def detect_gpu_capabilities(device):
    """Detect GPU capabilities and optimize parameters."""
    if not torch.cuda.is_available():
        return None
    
    device = torch.device(device)
    gpu_props = torch.cuda.get_device_properties(device)
    memory_gb = gpu_props.total_memory / (1024**3)

    # GPU configuration based on memory
    if memory_gb >= 30:  # A100 class
        optimal_batch_size = 32768
        buffer_size = 100000
        training_steps = 4
    else:  # Other GPUs
        optimal_batch_size = 16384
        buffer_size = 75000
        training_steps = 3
    
    return {
        'name': gpu_props.name,
        'memory_gb': memory_gb,
        'optimal_batch_size': optimal_batch_size,
        'buffer_size': buffer_size,
        'training_steps': training_steps,
        'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
    }


def calculate_dynamic_resources(dataset_size: int, device: str = 'cuda'):
    """Calculate optimal resources for maximum GPU utilization."""
    
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        allocated_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        allocated_cpus = 6
    
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    if torch.cuda.is_available() and 'cuda' in device:
        cuda_available = True
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory_gb >= 30:  # A100
            target_batch = 8192
            num_workers = min(24, allocated_cpus * 4)
            prefetch_factor = 6
            gpu_processing_batch = 16384
            dqn_training_batch = 8192
        elif gpu_memory_gb >= 15:
            target_batch = 4096
            gpu_processing_batch = 8192
            dqn_training_batch = 4096
            num_workers = min(16, allocated_cpus * 3)
            prefetch_factor = 4
        else:
            target_batch = 2048
            gpu_processing_batch = 4096
            dqn_training_batch = 2048
            num_workers = min(12, allocated_cpus * 2)
            prefetch_factor = 3
        
        config = {
            'batch_size': target_batch,
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,
            'pin_memory': True,
            'persistent_workers': True,
            'drop_last': False,
            'gpu_processing_batch': gpu_processing_batch,
            'dqn_training_batch': dqn_training_batch
        }
    else:
        cuda_available = False
        config = {
            'batch_size': 512,
            'num_workers': min(8, allocated_cpus * 2),
            'prefetch_factor': 2,
            'pin_memory': False,
            'persistent_workers': False,
            'drop_last': False,
            'gpu_processing_batch': 512,
            'dqn_training_batch': 256
        }
    
    return config, cuda_available


def create_optimized_data_loaders(train_subset=None, test_dataset=None, full_train_dataset=None, 
                                 batch_size: int = 1024, device: str = 'cuda'):
    """Create optimized data loaders with aggressive worker allocation."""
    
    dataset = next((d for d in [train_subset, test_dataset, full_train_dataset] if d is not None), None)
    if dataset is None:
        raise ValueError("No valid dataset provided")
    
    dataset_size = len(dataset)
    config, cuda_available = calculate_dynamic_resources(dataset_size, device)
    
    print(f"🎯 Optimized DataLoader Configuration:")
    print(f"  Target Workers: {config['num_workers']}")
    print(f"  Batch Size: {config['batch_size']:,}")
    print(f"  Expected GPU Utilization: High")
    
    if cuda_available:
        torch.cuda.empty_cache()
    
    datasets = [train_subset, test_dataset, full_train_dataset]
    shuffles = [True, False, True]
    
    for dataset, shuffle in zip(datasets, shuffles):
        if dataset is not None:
            try:
                print(f"🚀 Creating DataLoader with {config['num_workers']} workers...")
                
                # Add timeout protection
                def timeout_handler(signum, frame):
                    raise TimeoutError("DataLoader creation timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
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
                    
                    # Test batch
                    test_batch = next(iter(loader))
                    signal.alarm(0)
                    
                    print(f"✅ Success! DataLoader working efficiently")
                    return loader
                    
                except TimeoutError:
                    signal.alarm(0)
                    raise
                    
            except Exception as e:
                signal.alarm(0)
                print(f"❌ Failed with {config['num_workers']} workers: {e}")
                
                # Progressive fallback
                fallback_workers = [
                    config['num_workers'] // 2,
                    config['num_workers'] // 3,
                    min(8, config['num_workers'] // 4),
                    4
                ]
                
                for attempt, workers in enumerate(fallback_workers, 1):
                    try:
                        print(f"🔄 Fallback {attempt}: {workers} workers...")
                        loader = DataLoader(
                            dataset,
                            batch_size=config['batch_size'],
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


def setup_gpu_training(device_str: str = 'cuda'):
    """Setup GPU training environment with optimal settings."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - GPU training not supported")
    
    device = torch.device(device_str)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Set memory allocation strategy
    torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    gpu_info = detect_gpu_capabilities(device)
    monitor = GPUMonitor(device)
    
    print(f"✓ GPU Training Setup Complete")
    print(f"  Device: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
    print(f"  Optimal Batch Size: {gpu_info['optimal_batch_size']:,}")
    
    return device, gpu_info, monitor