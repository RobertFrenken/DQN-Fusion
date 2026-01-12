"""
Adaptive Memory Manager

This module provides dynamic memory management and batch size optimization
based on real-time GPU utilization, memory availability, and training performance.

Key Features:
- Dynamic batch size adjustment during training
- Memory pressure detection and response
- Performance-based optimization
- Multi-model memory coordination
- Automatic fallback strategies
"""

import torch
import psutil
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

@dataclass
class MemoryProfile:
    """Current memory utilization profile."""
    gpu_allocated_gb: float
    gpu_cached_gb: float
    gpu_free_gb: float
    cpu_used_gb: float
    cpu_available_gb: float
    utilization_percentage: float
    memory_pressure: str  # 'low', 'medium', 'high', 'critical'

@dataclass
class BatchSizeRecommendation:
    """Batch size recommendation based on current conditions."""
    recommended_batch_size: int
    confidence: float  # 0.0-1.0
    reason: str
    estimated_memory_usage_gb: float
    safety_margin: float

class AdaptiveMemoryManager:
    """
    Manages memory and batch sizes dynamically during training.
    """
    
    def __init__(self, device: str = 'cuda', 
                 initial_batch_size: int = 1024,
                 safety_margin: float = 0.2):
        """
        Initialize the adaptive memory manager.
        
        Args:
            device: Target device ('cuda' or 'cpu')
            initial_batch_size: Starting batch size
            safety_margin: Memory safety margin (0.0-1.0)
        """
        self.device = torch.device(device)
        self.initial_batch_size = initial_batch_size
        self.safety_margin = safety_margin
        
        # Memory tracking
        self.memory_history = []
        self.batch_size_history = []
        self.performance_history = []
        
        # Current state
        self.current_batch_size = initial_batch_size
        self.last_optimization_time = 0
        self.optimization_interval = 60  # seconds
        
        # GPU properties
        if self.device.type == 'cuda' and torch.cuda.is_available():
            self.gpu_props = torch.cuda.get_device_properties(self.device)
            self.max_memory_gb = self.gpu_props.total_memory / (1024**3)
        else:
            self.max_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Optimization parameters
        self.min_batch_size = 64
        self.max_batch_size = min(16384, initial_batch_size * 8)
        self.batch_size_growth_factor = 1.5
        self.batch_size_reduction_factor = 0.7
        
        print(f"✓ Adaptive Memory Manager initialized")
        print(f"  Device: {self.device}")
        print(f"  Max memory: {self.max_memory_gb:.1f}GB")
        print(f"  Initial batch size: {self.initial_batch_size}")
    
    def get_memory_profile(self) -> MemoryProfile:
        """Get current memory utilization profile."""
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            cached = torch.cuda.memory_reserved(self.device) / (1024**3)
            free = self.max_memory_gb - cached
            utilization = (allocated / self.max_memory_gb) * 100
            
        else:
            # CPU memory
            mem = psutil.virtual_memory()
            allocated = (mem.total - mem.available) / (1024**3)
            cached = mem.used / (1024**3)
            free = mem.available / (1024**3)
            utilization = mem.percent
        
        # Determine memory pressure
        if utilization < 60:
            pressure = 'low'
        elif utilization < 80:
            pressure = 'medium'
        elif utilization < 95:
            pressure = 'high'
        else:
            pressure = 'critical'
        
        return MemoryProfile(
            gpu_allocated_gb=allocated,
            gpu_cached_gb=cached,
            gpu_free_gb=free,
            cpu_used_gb=psutil.virtual_memory().used / (1024**3),
            cpu_available_gb=psutil.virtual_memory().available / (1024**3),
            utilization_percentage=utilization,
            memory_pressure=pressure
        )
    
    def recommend_batch_size(self, 
                           model_complexity: float = 1.0,
                           data_complexity: float = 1.0) -> BatchSizeRecommendation:
        """
        Recommend optimal batch size based on current conditions.
        
        Args:
            model_complexity: Relative model complexity (1.0 = baseline)
            data_complexity: Relative data complexity (1.0 = baseline)
        """
        profile = self.get_memory_profile()
        
        # Base recommendation on memory pressure
        if profile.memory_pressure == 'critical':
            base_batch_size = max(self.min_batch_size, 
                                  int(self.current_batch_size * 0.5))
            confidence = 0.9
            reason = "Critical memory pressure - reducing batch size"
            
        elif profile.memory_pressure == 'high':
            base_batch_size = max(self.min_batch_size,
                                  int(self.current_batch_size * self.batch_size_reduction_factor))
            confidence = 0.8
            reason = "High memory pressure - moderate reduction"
            
        elif profile.memory_pressure == 'low':
            # Can potentially increase batch size
            potential_increase = int(self.current_batch_size * self.batch_size_growth_factor)
            base_batch_size = min(self.max_batch_size, potential_increase)
            confidence = 0.7
            reason = "Low memory pressure - can increase batch size"
            
        else:  # medium pressure
            base_batch_size = self.current_batch_size
            confidence = 0.6
            reason = "Medium memory pressure - maintaining current batch size"
        
        # Adjust for model and data complexity
        complexity_factor = model_complexity * data_complexity
        adjusted_batch_size = int(base_batch_size / complexity_factor)
        adjusted_batch_size = max(self.min_batch_size, 
                                 min(self.max_batch_size, adjusted_batch_size))
        
        # Estimate memory usage
        estimated_memory = self._estimate_memory_usage(adjusted_batch_size, 
                                                      model_complexity, 
                                                      data_complexity)
        
        return BatchSizeRecommendation(
            recommended_batch_size=adjusted_batch_size,
            confidence=confidence,
            reason=reason + f" (complexity factor: {complexity_factor:.2f})",
            estimated_memory_usage_gb=estimated_memory,
            safety_margin=self.safety_margin
        )
    
    def _estimate_memory_usage(self, batch_size: int, 
                              model_complexity: float,
                              data_complexity: float) -> float:
        """Estimate memory usage for given batch size."""
        # Base estimation (empirical formula for graph neural networks)
        base_memory_per_sample = 0.01  # GB per sample (rough estimate)
        
        # Adjust for complexity
        memory_per_sample = (base_memory_per_sample * 
                           model_complexity * 
                           data_complexity)
        
        total_estimated = batch_size * memory_per_sample
        return total_estimated
    
    def optimize_batch_size(self, 
                           current_throughput: Optional[float] = None,
                           current_loss: Optional[float] = None,
                           force_optimization: bool = False) -> Tuple[int, str]:
        """
        Optimize batch size based on current performance.
        
        Args:
            current_throughput: Current training throughput (samples/second)
            current_loss: Current training loss
            force_optimization: Force optimization regardless of timing
            
        Returns:
            Tuple of (new_batch_size, reason)
        """
        current_time = time.time()
        
        # Check if optimization is due
        if not force_optimization and (current_time - self.last_optimization_time) < self.optimization_interval:
            return self.current_batch_size, "Optimization interval not reached"
        
        self.last_optimization_time = current_time
        
        # Get recommendation
        recommendation = self.recommend_batch_size()
        
        # Consider performance history for better decisions
        if len(self.performance_history) >= 3:
            recent_performance = self.performance_history[-3:]
            avg_throughput = np.mean([p.get('throughput', 0) for p in recent_performance])
            
            # If performance is declining, be more conservative
            if current_throughput and current_throughput < avg_throughput * 0.9:
                recommendation.recommended_batch_size = int(
                    recommendation.recommended_batch_size * 0.8
                )
                recommendation.reason += " + performance decline detected"
        
        # Update current batch size
        old_batch_size = self.current_batch_size
        self.current_batch_size = recommendation.recommended_batch_size
        
        # Record the change
        self.batch_size_history.append({
            'timestamp': current_time,
            'old_batch_size': old_batch_size,
            'new_batch_size': self.current_batch_size,
            'reason': recommendation.reason,
            'confidence': recommendation.confidence
        })
        
        # Record performance data
        if current_throughput or current_loss:
            self.performance_history.append({
                'timestamp': current_time,
                'batch_size': self.current_batch_size,
                'throughput': current_throughput,
                'loss': current_loss
            })
        
        return self.current_batch_size, recommendation.reason
    
    def emergency_memory_cleanup(self) -> bool:
        """Perform emergency memory cleanup when under severe pressure."""
        try:
            profile = self.get_memory_profile()
            
            if profile.memory_pressure != 'critical':
                return False
            
            print(f"⚠️ Emergency memory cleanup triggered")
            print(f"   Memory utilization: {profile.utilization_percentage:.1f}%")
            
            # Aggressive cleanup
            import gc
            
            # Clear Python garbage
            collected = gc.collect()
            
            # Clear GPU cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force batch size reduction
            self.current_batch_size = max(self.min_batch_size,
                                         int(self.current_batch_size * 0.4))
            
            # Wait a moment for cleanup to take effect
            time.sleep(2)
            
            new_profile = self.get_memory_profile()
            improvement = profile.utilization_percentage - new_profile.utilization_percentage
            
            print(f"   Cleanup completed: {improvement:.1f}% memory freed")
            print(f"   New batch size: {self.current_batch_size}")
            
            return improvement > 5  # Consider successful if freed >5%
            
        except Exception as e:
            print(f"Emergency cleanup failed: {e}")
            return False
    
    def get_adaptive_data_loader_config(self, 
                                       base_num_workers: int = 4) -> Dict[str, Any]:
        """Get optimized data loader configuration."""
        profile = self.get_memory_profile()
        
        # Adjust num_workers based on memory and CPU
        if profile.memory_pressure in ['high', 'critical']:
            num_workers = max(1, base_num_workers // 2)
            prefetch_factor = 2
        elif profile.memory_pressure == 'low':
            num_workers = min(base_num_workers * 2, psutil.cpu_count())
            prefetch_factor = 4
        else:
            num_workers = base_num_workers
            prefetch_factor = 2
        
        return {
            'batch_size': self.current_batch_size,
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,
            'pin_memory': self.device.type == 'cuda',
            'persistent_workers': num_workers > 0,
            'drop_last': True  # Consistent batch sizes
        }
    
    def monitor_training_step(self, 
                             step_time: float,
                             memory_usage: Optional[float] = None,
                             loss: Optional[float] = None) -> Dict[str, Any]:
        """Monitor a single training step and return recommendations."""
        profile = self.get_memory_profile()
        
        # Calculate throughput
        throughput = self.current_batch_size / step_time if step_time > 0 else 0
        
        # Check if optimization is needed
        optimization_needed = False
        recommendations = []
        
        if profile.memory_pressure == 'critical':
            optimization_needed = True
            recommendations.append("Emergency memory cleanup required")
        elif profile.memory_pressure == 'high' and self.current_batch_size > self.min_batch_size:
            optimization_needed = True
            recommendations.append("Consider reducing batch size")
        elif profile.memory_pressure == 'low' and self.current_batch_size < self.max_batch_size:
            recommendations.append("Can potentially increase batch size")
        
        return {
            'memory_profile': profile,
            'current_batch_size': self.current_batch_size,
            'throughput': throughput,
            'optimization_needed': optimization_needed,
            'recommendations': recommendations,
            'step_efficiency': min(1.0, throughput / (self.current_batch_size * 10))  # Normalized efficiency
        }
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of memory management performance."""
        if not self.batch_size_history:
            return {'message': 'No optimization history available'}
        
        # Calculate statistics
        batch_sizes = [entry['new_batch_size'] for entry in self.batch_size_history]
        avg_batch_size = np.mean(batch_sizes)
        batch_size_variance = np.var(batch_sizes)
        
        optimization_count = len(self.batch_size_history)
        
        # Performance trends
        performance_trend = "stable"
        if len(self.performance_history) >= 5:
            recent_throughputs = [p.get('throughput', 0) for p in self.performance_history[-5:]]
            if len(recent_throughputs) > 1:
                trend_slope = np.polyfit(range(len(recent_throughputs)), recent_throughputs, 1)[0]
                if trend_slope > 0.1:
                    performance_trend = "improving"
                elif trend_slope < -0.1:
                    performance_trend = "declining"
        
        return {
            'optimization_count': optimization_count,
            'avg_batch_size': avg_batch_size,
            'batch_size_variance': batch_size_variance,
            'current_batch_size': self.current_batch_size,
            'performance_trend': performance_trend,
            'memory_utilization_avg': np.mean([p.utilization_percentage for p in self.memory_history]) if self.memory_history else 0,
            'max_memory_gb': self.max_memory_gb,
            'efficiency_score': self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate an efficiency score based on memory usage and performance."""
        if not self.performance_history:
            return 0.5
        
        # Score based on throughput stability and memory usage
        recent_performance = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        if not recent_performance:
            return 0.5
        
        throughputs = [p.get('throughput', 0) for p in recent_performance if p.get('throughput', 0) > 0]
        
        if not throughputs:
            return 0.5
        
        # Efficiency is based on throughput consistency and absolute performance
        throughput_consistency = 1 - (np.std(throughputs) / np.mean(throughputs)) if len(throughputs) > 1 else 1.0
        throughput_level = min(1.0, np.mean(throughputs) / (self.max_batch_size * 5))  # Normalize
        
        efficiency_score = 0.6 * throughput_consistency + 0.4 * throughput_level
        return max(0.0, min(1.0, efficiency_score))

def create_adaptive_memory_manager(device: str = 'cuda',
                                  initial_batch_size: int = 1024,
                                  safety_margin: float = 0.15) -> AdaptiveMemoryManager:
    """
    Factory function to create an adaptive memory manager with optimal settings.
    
    Args:
        device: Target device
        initial_batch_size: Starting batch size
        safety_margin: Memory safety margin
    """
    return AdaptiveMemoryManager(
        device=device,
        initial_batch_size=initial_batch_size,
        safety_margin=safety_margin
    )