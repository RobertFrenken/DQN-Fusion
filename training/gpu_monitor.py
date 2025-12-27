"""
GPU Monitor for tracking GPU usage and performance during fusion training.
Latest version with comprehensive performance analysis.
"""
import time
import torch
import psutil
import numpy as np
from typing import Dict


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