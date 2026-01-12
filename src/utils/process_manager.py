"""
Process Management and Monitoring Tools

This module provides comprehensive process management, monitoring, and coordination
for distributed training across multiple models and datasets.

Key Features:
- Multi-process training coordination
- Resource monitoring and allocation
- Process health checking and recovery
- Training progress visualization
- Automated checkpoint management
- Performance analytics and reporting
"""

import os
import sys
import time
import json
import psutil
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import threading
import queue
import signal

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

@dataclass
class ProcessStatus:
    """Status information for a training process."""
    process_id: str
    pid: int
    status: str  # 'running', 'completed', 'failed', 'killed'
    start_time: datetime
    end_time: Optional[datetime] = None
    progress: float = 0.0  # 0.0-1.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: Optional[float] = None
    gpu_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    error_message: Optional[str] = None
    log_file: Optional[str] = None

@dataclass
class SystemResources:
    """Current system resource utilization."""
    cpu_percent: float
    memory_percent: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_utilization_percent: float
    disk_usage_percent: float
    network_io_mbps: float
    temperature_celsius: Optional[float] = None

class ProcessManager:
    """
    Manages multiple training processes with resource monitoring.
    """
    
    def __init__(self, log_dir: str = "outputs/process_logs",
                 max_concurrent_processes: int = 2):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent = max_concurrent_processes
        self.processes = {}  # process_id -> ProcessStatus
        self.active_pids = {}  # pid -> process_id
        self.resource_history = []
        
        # Monitoring
        self.monitor_thread = None
        self.monitoring_active = False
        self.monitor_queue = queue.Queue()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Performance tracking
        self.performance_data = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_memory': [],
            'gpu_utilization': [],
            'active_processes': []
        }
        
        self.logger.info("Process Manager initialized")
        self.logger.info(f"Max concurrent processes: {self.max_concurrent}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for process management."""
        logger = logging.getLogger('ProcessManager')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / f"process_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def start_monitoring(self):
        """Start system resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop system resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Monitor system resources in background thread."""
        while self.monitoring_active:
            try:
                resources = self._get_system_resources()
                self.resource_history.append({
                    'timestamp': time.time(),
                    'resources': resources
                })
                
                # Update performance tracking
                self.performance_data['timestamps'].append(time.time())
                self.performance_data['cpu_usage'].append(resources.cpu_percent)
                self.performance_data['memory_usage'].append(resources.memory_percent)
                self.performance_data['gpu_memory'].append(resources.gpu_memory_used_mb)
                self.performance_data['gpu_utilization'].append(resources.gpu_utilization_percent)
                self.performance_data['active_processes'].append(len(self.get_active_processes()))
                
                # Keep only last hour of data
                max_history = 3600  # 1 hour
                if len(self.resource_history) > max_history:
                    self.resource_history = self.resource_history[-max_history:]
                    for key in self.performance_data:
                        if isinstance(self.performance_data[key], list):
                            self.performance_data[key] = self.performance_data[key][-max_history:]
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _get_system_resources(self) -> SystemResources:
        """Get current system resource utilization."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        gpu_utilization = 0.0
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024**2)  # MB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
            
            # Try to get GPU utilization
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu
            except (ImportError, Exception):
                gpu_utilization = min(95.0, (gpu_memory_used / gpu_memory_total) * 100 * 1.2)
        
        # Disk
        disk = psutil.disk_usage('/')
        
        # Network (simplified)
        network_io = psutil.net_io_counters()
        network_mbps = (network_io.bytes_sent + network_io.bytes_recv) / (1024**2)  # Rough approximation
        
        return SystemResources(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_utilization_percent=gpu_utilization,
            disk_usage_percent=disk.percent,
            network_io_mbps=network_mbps
        )
    
    def can_start_process(self) -> Tuple[bool, str]:
        """Check if a new process can be started."""
        active_processes = self.get_active_processes()
        
        if len(active_processes) >= self.max_concurrent:
            return False, f"Max concurrent processes ({self.max_concurrent}) reached"
        
        # Check system resources
        resources = self._get_system_resources()
        
        if resources.cpu_percent > 90:
            return False, f"High CPU usage ({resources.cpu_percent:.1f}%)"
        
        if resources.memory_percent > 90:
            return False, f"High memory usage ({resources.memory_percent:.1f}%)"
        
        if resources.gpu_memory_used_mb / resources.gpu_memory_total_mb > 0.9:
            return False, f"High GPU memory usage ({resources.gpu_memory_used_mb/resources.gpu_memory_total_mb*100:.1f}%)"
        
        return True, "Resources available"
    
    def start_training_process(self, process_id: str, 
                              script_path: str,
                              config_path: str,
                              dataset_name: str) -> bool:
        """Start a new training process."""
        can_start, reason = self.can_start_process()
        if not can_start:
            self.logger.warning(f"Cannot start process {process_id}: {reason}")
            return False
        
        # Create log file for this process
        log_file = self.log_dir / f"{process_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Build command
        cmd = [
            sys.executable,
            script_path,
            f"--config-path={Path(config_path).parent}",
            f"--config-name={Path(config_path).stem}",
            f"root_folder={dataset_name}"
        ]
        
        try:
            # Start process
            proc = subprocess.Popen(
                cmd,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                cwd=Path(script_path).parent.parent.parent  # Project root
            )
            
            # Record process status
            status = ProcessStatus(
                process_id=process_id,
                pid=proc.pid,
                status='running',
                start_time=datetime.now(),
                log_file=str(log_file)
            )
            
            self.processes[process_id] = status
            self.active_pids[proc.pid] = process_id
            
            self.logger.info(f"Started training process {process_id} (PID: {proc.pid})")
            self.logger.info(f"Command: {' '.join(cmd)}")
            self.logger.info(f"Log file: {log_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start process {process_id}: {e}")
            return False
    
    def update_process_status(self, process_id: str, **kwargs):
        """Update process status information."""
        if process_id in self.processes:
            for key, value in kwargs.items():
                if hasattr(self.processes[process_id], key):
                    setattr(self.processes[process_id], key, value)
    
    def get_active_processes(self) -> List[str]:
        """Get list of currently active process IDs."""
        active = []
        
        for process_id, status in self.processes.items():
            if status.status == 'running':
                # Check if process is actually still running
                try:
                    proc = psutil.Process(status.pid)
                    if proc.is_running():
                        active.append(process_id)
                    else:
                        # Process finished, update status
                        self.processes[process_id].status = 'completed'
                        self.processes[process_id].end_time = datetime.now()
                        if status.pid in self.active_pids:
                            del self.active_pids[status.pid]
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process no longer exists
                    self.processes[process_id].status = 'failed'
                    self.processes[process_id].end_time = datetime.now()
                    if status.pid in self.active_pids:
                        del self.active_pids[status.pid]
        
        return active
    
    def kill_process(self, process_id: str) -> bool:
        """Kill a specific training process."""
        if process_id not in self.processes:
            return False
        
        status = self.processes[process_id]
        
        try:
            proc = psutil.Process(status.pid)
            proc.terminate()
            
            # Wait for graceful termination
            try:
                proc.wait(timeout=30)
            except psutil.TimeoutExpired:
                proc.kill()  # Force kill if not terminated
            
            # Update status
            self.processes[process_id].status = 'killed'
            self.processes[process_id].end_time = datetime.now()
            
            if status.pid in self.active_pids:
                del self.active_pids[status.pid]
            
            self.logger.info(f"Process {process_id} killed")
            return True
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.error(f"Failed to kill process {process_id}: {e}")
            return False
    
    def kill_all_processes(self):
        """Kill all active training processes."""
        active = self.get_active_processes()
        
        for process_id in active:
            self.kill_process(process_id)
        
        self.logger.info(f"Killed {len(active)} processes")
    
    def get_process_info(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a process."""
        if process_id not in self.processes:
            return None
        
        status = self.processes[process_id]
        
        # Get current resource usage if process is running
        current_resources = {}
        if status.status == 'running':
            try:
                proc = psutil.Process(status.pid)
                current_resources = {
                    'cpu_percent': proc.cpu_percent(),
                    'memory_mb': proc.memory_info().rss / (1024**2),
                    'threads': proc.num_threads()
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Calculate runtime
        if status.end_time:
            runtime = status.end_time - status.start_time
        else:
            runtime = datetime.now() - status.start_time
        
        info = {
            **asdict(status),
            'runtime_seconds': runtime.total_seconds(),
            'runtime_formatted': str(runtime),
            'current_resources': current_resources
        }
        
        # Parse log file for additional info if available
        if status.log_file and Path(status.log_file).exists():
            log_info = self._parse_log_file(status.log_file)
            info.update(log_info)
        
        return info
    
    def _parse_log_file(self, log_file: str) -> Dict[str, Any]:
        """Parse log file to extract training progress."""
        log_info = {
            'last_log_line': None,
            'error_lines': [],
            'progress_info': {}
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                if lines:
                    log_info['last_log_line'] = lines[-1].strip()
                
                # Look for error patterns
                error_patterns = ['error', 'exception', 'failed', 'cuda out of memory']
                for line in lines[-50:]:  # Check last 50 lines
                    line_lower = line.lower()
                    if any(pattern in line_lower for pattern in error_patterns):
                        log_info['error_lines'].append(line.strip())
                
                # Look for progress patterns
                for line in lines[-20:]:  # Check recent lines
                    if 'epoch' in line.lower():
                        # Try to extract epoch information
                        import re
                        epoch_match = re.search(r'epoch[:\\s]*(\\d+)', line.lower())
                        if epoch_match:
                            log_info['progress_info']['current_epoch'] = int(epoch_match.group(1))
                    
                    if 'loss' in line.lower():
                        # Try to extract loss information
                        import re
                        loss_match = re.search(r'loss[:\\s]*([0-9.]+)', line.lower())
                        if loss_match:
                            log_info['progress_info']['current_loss'] = float(loss_match.group(1))
        
        except Exception as e:
            log_info['parse_error'] = str(e)
        
        return log_info
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_resources = self._get_system_resources()
        active_processes = self.get_active_processes()
        
        # Calculate resource utilization trends
        trends = {}
        if len(self.resource_history) > 60:  # At least 1 minute of data
            recent = self.resource_history[-60:]  # Last minute
            older = self.resource_history[-120:-60]  # Previous minute
            
            if older:
                recent_avg_cpu = np.mean([r['resources'].cpu_percent for r in recent])
                older_avg_cpu = np.mean([r['resources'].cpu_percent for r in older])
                trends['cpu_trend'] = 'increasing' if recent_avg_cpu > older_avg_cpu * 1.1 else 'decreasing' if recent_avg_cpu < older_avg_cpu * 0.9 else 'stable'
        
        return {
            'current_resources': asdict(current_resources),
            'active_processes': len(active_processes),
            'total_processes': len(self.processes),
            'max_concurrent': self.max_concurrent,
            'can_start_new': self.can_start_process(),
            'resource_trends': trends,
            'monitoring_duration_minutes': len(self.resource_history) / 60,
            'process_details': {pid: self.get_process_info(pid) for pid in active_processes}
        }
    
    def generate_performance_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.resource_history:
            return {'error': 'No resource data available'}
        
        # Calculate statistics
        cpu_usage = [r['resources'].cpu_percent for r in self.resource_history]
        memory_usage = [r['resources'].memory_percent for r in self.resource_history]
        gpu_memory = [r['resources'].gpu_memory_used_mb for r in self.resource_history]
        
        report = {
            'monitoring_period': {
                'start_time': datetime.fromtimestamp(self.resource_history[0]['timestamp']),
                'end_time': datetime.fromtimestamp(self.resource_history[-1]['timestamp']),
                'duration_hours': (self.resource_history[-1]['timestamp'] - self.resource_history[0]['timestamp']) / 3600
            },
            'resource_statistics': {
                'cpu': {
                    'mean': np.mean(cpu_usage),
                    'max': np.max(cpu_usage),
                    'min': np.min(cpu_usage),
                    'std': np.std(cpu_usage)
                },
                'memory': {
                    'mean': np.mean(memory_usage),
                    'max': np.max(memory_usage),
                    'min': np.min(memory_usage),
                    'std': np.std(memory_usage)
                },
                'gpu_memory_mb': {
                    'mean': np.mean(gpu_memory),
                    'max': np.max(gpu_memory),
                    'min': np.min(gpu_memory),
                    'std': np.std(gpu_memory)
                }
            },
            'process_summary': {
                'total_processes': len(self.processes),
                'completed': len([p for p in self.processes.values() if p.status == 'completed']),
                'failed': len([p for p in self.processes.values() if p.status == 'failed']),
                'killed': len([p for p in self.processes.values() if p.status == 'killed']),
                'running': len([p for p in self.processes.values() if p.status == 'running'])
            }
        }
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Also save performance data as CSV
            csv_path = save_path.with_suffix('.csv')
            df = pd.DataFrame({
                'timestamp': [r['timestamp'] for r in self.resource_history],
                'cpu_percent': [r['resources'].cpu_percent for r in self.resource_history],
                'memory_percent': [r['resources'].memory_percent for r in self.resource_history],
                'gpu_memory_mb': [r['resources'].gpu_memory_used_mb for r in self.resource_history],
                'gpu_utilization': [r['resources'].gpu_utilization_percent for r in self.resource_history]
            })
            df.to_csv(csv_path, index=False)
            
            self.logger.info(f"Performance report saved to {save_path}")
            self.logger.info(f"Performance data saved to {csv_path}")
        
        return report
    
    def create_monitoring_dashboard(self, save_path: str = "outputs/monitoring_dashboard.png"):
        """Create a visual monitoring dashboard."""
        if not self.performance_data['timestamps']:
            print("No performance data available for dashboard")
            return
        
        # Convert timestamps to datetime
        timestamps = [datetime.fromtimestamp(ts) for ts in self.performance_data['timestamps']]
        
        # Create dashboard
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Process Monitoring Dashboard', fontsize=16)
        
        # CPU Usage
        axes[0, 0].plot(timestamps, self.performance_data['cpu_usage'], 'b-', alpha=0.7)
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory Usage
        axes[0, 1].plot(timestamps, self.performance_data['memory_usage'], 'g-', alpha=0.7)
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].grid(True, alpha=0.3)
        
        # GPU Memory
        axes[0, 2].plot(timestamps, np.array(self.performance_data['gpu_memory'])/1024, 'r-', alpha=0.7)
        axes[0, 2].set_title('GPU Memory (GB)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # GPU Utilization
        axes[1, 0].plot(timestamps, self.performance_data['gpu_utilization'], 'm-', alpha=0.7)
        axes[1, 0].set_title('GPU Utilization (%)')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Active Processes
        axes[1, 1].plot(timestamps, self.performance_data['active_processes'], 'c-', alpha=0.7, marker='o', markersize=3)
        axes[1, 1].set_title('Active Processes')
        axes[1, 1].set_ylim(0, max(self.performance_data['active_processes']) + 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Process Status Summary (bar chart)
        status_counts = {}
        for process in self.processes.values():
            status_counts[process.status] = status_counts.get(process.status, 0) + 1
        
        if status_counts:
            statuses = list(status_counts.keys())
            counts = list(status_counts.values())
            colors = {'running': 'green', 'completed': 'blue', 'failed': 'red', 'killed': 'orange'}
            bar_colors = [colors.get(status, 'gray') for status in statuses]
            
            axes[1, 2].bar(statuses, counts, color=bar_colors, alpha=0.7)
            axes[1, 2].set_title('Process Status Summary')
            axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        # Format x-axis timestamps
        for ax in axes.flat:
            if hasattr(ax, 'xaxis'):
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save dashboard
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Monitoring dashboard saved to {save_path}")
    
    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        self.stop_monitoring()
        self.kill_all_processes()
        
        self.logger.info("Process manager cleanup completed")

# Example usage and integration functions
def create_process_manager(max_concurrent: int = 2, 
                          log_dir: str = "outputs/process_logs") -> ProcessManager:
    """Factory function to create a process manager."""
    return ProcessManager(log_dir=log_dir, max_concurrent_processes=max_concurrent)

if __name__ == "__main__":
    # Example usage
    manager = ProcessManager(max_concurrent_processes=2)
    
    try:
        # Start monitoring
        manager.start_monitoring()
        
        # Simulate some processes (in real use, these would be actual training scripts)
        print("Process manager demo - monitoring for 30 seconds...")
        
        time.sleep(30)
        
        # Generate report
        report = manager.generate_performance_report("outputs/demo_report.json")
        print(f"Generated performance report with {len(manager.resource_history)} data points")
        
        # Create dashboard
        manager.create_monitoring_dashboard("outputs/demo_dashboard.png")
        
    finally:
        manager.cleanup()