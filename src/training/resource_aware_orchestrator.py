"""
Resource-Aware Training Orchestrator

This module orchestrates the retraining of all base models with optimized GPU utilization,
memory management, and process coordination. It ensures efficient resource usage across
multiple training phases and datasets.

Key Features:
- Dynamic resource allocation based on GPU capabilities
- Intelligent scheduling of training phases
- Memory-aware batch size optimization
- Process monitoring and error recovery
- Automated model checkpoint management
- Multi-dataset training coordination
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import torch
import torch.multiprocessing as mp
import psutil
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.utils.legacy_compatibility import detect_gpu_capabilities_unified
from src.training.gpu_monitor import GPUMonitor
from src.utils.utils_logging import setup_gpu_optimization, log_memory_usage, cleanup_memory

@dataclass
class TrainingPhase:
    """Configuration for a single training phase."""
    phase_name: str
    dataset_key: str
    model_type: str  # 'teacher', 'student', 'fusion'
    script_path: str
    config_override: Dict[str, Any]
    dependencies: List[str]  # List of phases that must complete first
    estimated_time_minutes: int
    memory_requirement_gb: float
    priority: int = 1  # Higher numbers = higher priority

@dataclass
class ResourceProfile:
    """Current system resource profile."""
    gpu_memory_total_gb: float
    gpu_memory_available_gb: float
    cpu_cores: int
    ram_available_gb: float
    optimal_batch_size: int
    max_workers: int
    training_capacity: float  # 0.0-1.0 scale of how much load system can handle

class ResourceAwareOrchestrator:
    """
    Orchestrates the retraining of all models with optimal resource utilization.
    """
    
    def __init__(self, base_config_path: str = "conf/base.yaml", 
                 log_dir: str = "outputs/training_logs"):
        self.base_config_path = Path(base_config_path)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize GPU monitoring
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            raise RuntimeError("Resource-aware orchestrator requires CUDA GPU")
            
        self.gpu_monitor = GPUMonitor(self.device)
        self.gpu_info = detect_gpu_capabilities_unified(str(self.device))
        
        # Initialize resource tracking
        self.resource_profile = self._get_resource_profile()
        self.training_history = []
        self.failed_phases = []
        
        # Define training phases
        self.training_phases = self._define_training_phases()
        
        self.logger.info(f"Orchestrator initialized on {self.device}")
        self.logger.info(f"GPU: {self.gpu_info.get('name', 'Unknown')} "
                        f"({self.resource_profile.gpu_memory_total_gb:.1f}GB)")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for the orchestrator."""
        logger = logging.getLogger('ResourceAwareOrchestrator')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    
    def _get_resource_profile(self) -> ResourceProfile:
        """Get current system resource profile."""
        # GPU memory
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(self.device)
            total_memory = gpu_props.total_memory / (1024**3)
            torch.cuda.empty_cache()
            available_memory = (gpu_props.total_memory - torch.cuda.memory_allocated()) / (1024**3)
        else:
            total_memory = available_memory = 0.0
        
        # CPU and RAM
        cpu_cores = psutil.cpu_count(logical=False)
        ram_info = psutil.virtual_memory()
        ram_available = ram_info.available / (1024**3)
        
        # Calculate optimal parameters based on GPU memory
        if total_memory >= 30:  # A100 or similar
            optimal_batch_size = 8192
            max_workers = min(24, cpu_cores * 2)
            training_capacity = 0.9
        elif total_memory >= 20:  # RTX 3090/4090
            optimal_batch_size = 4096
            max_workers = min(16, cpu_cores * 2)
            training_capacity = 0.8
        else:  # Smaller GPUs
            optimal_batch_size = 2048
            max_workers = min(8, cpu_cores)
            training_capacity = 0.7
        
        return ResourceProfile(
            gpu_memory_total_gb=total_memory,
            gpu_memory_available_gb=available_memory,
            cpu_cores=cpu_cores,
            ram_available_gb=ram_available,
            optimal_batch_size=optimal_batch_size,
            max_workers=max_workers,
            training_capacity=training_capacity
        )
    
    def _define_training_phases(self) -> List[TrainingPhase]:
        """Define all training phases for the complete retraining pipeline."""
        phases = []
        
        # Dataset configurations
        datasets = ['hcrl_ch', 'hcrl_sa', 'set_01', 'set_02', 'set_03', 'set_04']
        
        for dataset in datasets:
            # Phase 1: Teacher model training (base models)
            phases.append(TrainingPhase(
                phase_name=f"teacher_{dataset}",
                dataset_key=dataset,
                model_type="teacher",
                script_path="src/training/AD_KD_GPU.py",
                config_override={
                    'root_folder': dataset,
                    'epochs': 10,
                    'batch_size': self.resource_profile.optimal_batch_size,
                    'device': str(self.device),
                    'mode': 'teacher_only'
                },
                dependencies=[],
                estimated_time_minutes=45,
                memory_requirement_gb=8.0,
                priority=3
            ))
            
            # Phase 2: Student model training (knowledge distillation)
            phases.append(TrainingPhase(
                phase_name=f"student_{dataset}",
                dataset_key=dataset,
                model_type="student",
                script_path="src/training/AD_KD_GPU.py",
                config_override={
                    'root_folder': dataset,
                    'epochs': 8,
                    'batch_size': self.resource_profile.optimal_batch_size,
                    'device': str(self.device),
                    'mode': 'student_only',
                    'teacher_model_path': f'saved_models/best_teacher_model_{dataset}.pth'
                },
                dependencies=[f"teacher_{dataset}"],
                estimated_time_minutes=35,
                memory_requirement_gb=6.0,
                priority=2
            ))
            
            # Phase 3: Fusion training
            phases.append(TrainingPhase(
                phase_name=f"fusion_{dataset}",
                dataset_key=dataset,
                model_type="fusion",
                script_path="src/training/fusion_training.py",
                config_override={
                    'root_folder': dataset,
                    'fusion_episodes': 300,
                    'batch_size': self.resource_profile.optimal_batch_size,
                    'device': str(self.device),
                    'autoencoder_path': f'saved_models/final_student_model_{dataset}.pth',
                    'classifier_path': f'saved_models/classifier_{dataset}.pth'
                },
                dependencies=[f"student_{dataset}"],
                estimated_time_minutes=60,
                memory_requirement_gb=10.0,
                priority=1
            ))
        
        return sorted(phases, key=lambda x: x.priority, reverse=True)
    
    def _check_dependencies(self, phase: TrainingPhase) -> bool:
        """Check if all dependencies for a phase are completed."""
        completed_phases = [p['phase_name'] for p in self.training_history if p['status'] == 'completed']
        return all(dep in completed_phases for dep in phase.dependencies)
    
    def _can_run_phase(self, phase: TrainingPhase) -> Tuple[bool, str]:
        """Check if a phase can run given current resources."""
        current_profile = self._get_resource_profile()
        
        # Check memory requirement
        if phase.memory_requirement_gb > current_profile.gpu_memory_available_gb:
            return False, f"Insufficient GPU memory: need {phase.memory_requirement_gb}GB, have {current_profile.gpu_memory_available_gb:.1f}GB"
        
        # Check dependencies
        if not self._check_dependencies(phase):
            missing_deps = [dep for dep in phase.dependencies if dep not in [p['phase_name'] for p in self.training_history if p['status'] == 'completed']]
            return False, f"Missing dependencies: {missing_deps}"
        
        return True, "Ready to run"
    
    def _execute_training_phase(self, phase: TrainingPhase) -> Dict[str, Any]:
        """Execute a single training phase."""
        self.logger.info(f"Starting phase: {phase.phase_name}")
        start_time = time.time()
        
        try:
            # Setup GPU optimization before training
            setup_gpu_optimization()
            
            # Record initial GPU stats
            self.gpu_monitor.record_gpu_stats(0)
            log_memory_usage(f"Before {phase.phase_name}")
            
            # Create config for this phase
            config = OmegaConf.load(self.base_config_path)
            config.update(phase.config_override)
            
            # Save config for this phase
            phase_config_path = self.log_dir / f"{phase.phase_name}_config.yaml"
            OmegaConf.save(config, phase_config_path)
            
            # Import and run the training script
            if phase.script_path == "src/training/AD_KD_GPU.py":
                from src.training.AD_KD_GPU import main as kd_main
                kd_main(config)
            elif phase.script_path == "src/training/fusion_training.py":
                from src.training.fusion_training import main as fusion_main
                fusion_main(config)
            else:
                raise ValueError(f"Unknown script path: {phase.script_path}")
            
            # Record final GPU stats
            log_memory_usage(f"After {phase.phase_name}")
            elapsed_time = time.time() - start_time
            
            result = {
                'phase_name': phase.phase_name,
                'status': 'completed',
                'elapsed_time': elapsed_time,
                'dataset': phase.dataset_key,
                'model_type': phase.model_type,
                'config_path': str(phase_config_path)
            }
            
            self.logger.info(f"Phase {phase.phase_name} completed in {elapsed_time/60:.1f} minutes")
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = str(e)
            
            result = {
                'phase_name': phase.phase_name,
                'status': 'failed',
                'elapsed_time': elapsed_time,
                'error': error_msg,
                'dataset': phase.dataset_key,
                'model_type': phase.model_type
            }
            
            self.logger.error(f"Phase {phase.phase_name} failed after {elapsed_time/60:.1f} minutes: {error_msg}")
            return result
        
        finally:
            # Always cleanup memory after each phase
            cleanup_memory()
    
    def run_complete_retraining(self, max_parallel: int = 1, 
                               skip_completed: bool = True) -> Dict[str, Any]:
        """
        Run the complete retraining pipeline.
        
        Args:
            max_parallel: Maximum number of phases to run in parallel
            skip_completed: Skip phases with existing model files
            
        Returns:
            Summary of the retraining process
        """
        self.logger.info("Starting complete retraining pipeline")
        start_time = time.time()
        
        # Filter phases if skip_completed is True
        phases_to_run = []
        for phase in self.training_phases:
            if skip_completed and self._model_exists(phase):
                self.logger.info(f"Skipping {phase.phase_name} - model already exists")
                continue
            phases_to_run.append(phase)
        
        self.logger.info(f"Running {len(phases_to_run)} training phases")
        
        completed_count = 0
        failed_count = 0
        
        # Process phases in dependency order
        remaining_phases = phases_to_run.copy()
        
        while remaining_phases:
            # Find phases that can run now
            ready_phases = []
            for phase in remaining_phases:
                can_run, reason = self._can_run_phase(phase)
                if can_run:
                    ready_phases.append(phase)
            
            if not ready_phases:
                # Check if we're stuck due to failed dependencies
                if failed_count > 0:
                    self.logger.error("Cannot proceed - some phases failed and others depend on them")
                    break
                else:
                    # Wait a bit and check again (might be resource issue)
                    time.sleep(30)
                    continue
            
            # Run the highest priority ready phase
            ready_phases.sort(key=lambda x: x.priority, reverse=True)
            phase = ready_phases[0]
            
            # Execute the phase
            result = self._execute_training_phase(phase)
            self.training_history.append(result)
            
            if result['status'] == 'completed':
                completed_count += 1
            else:
                failed_count += 1
                self.failed_phases.append(phase.phase_name)
            
            # Remove completed/failed phase from remaining
            remaining_phases.remove(phase)
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = {
            'total_phases': len(phases_to_run),
            'completed': completed_count,
            'failed': failed_count,
            'total_time_hours': total_time / 3600,
            'failed_phases': self.failed_phases,
            'training_history': self.training_history
        }
        
        # Save summary
        summary_path = self.log_dir / f"retraining_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Retraining completed: {completed_count}/{len(phases_to_run)} phases successful")
        self.logger.info(f"Total time: {total_time/3600:.1f} hours")
        self.logger.info(f"Summary saved to: {summary_path}")
        
        return summary
    
    def _model_exists(self, phase: TrainingPhase) -> bool:
        """Check if a model already exists for this phase."""
        model_paths = {
            'teacher': f'saved_models/best_teacher_model_{phase.dataset_key}.pth',
            'student': f'saved_models/final_student_model_{phase.dataset_key}.pth',
            'fusion': f'saved_models/fusion_agent_{phase.dataset_key}.pth'
        }
        
        model_path = Path(model_paths.get(phase.model_type, ''))
        return model_path.exists()
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and resource usage."""
        current_profile = self._get_resource_profile()
        
        return {
            'resource_profile': asdict(current_profile),
            'phases_completed': len([p for p in self.training_history if p['status'] == 'completed']),
            'phases_failed': len([p for p in self.training_history if p['status'] == 'failed']),
            'total_phases': len(self.training_phases),
            'failed_phases': self.failed_phases,
            'gpu_info': self.gpu_info
        }

def main():
    """Main entry point for the resource-aware orchestrator."""
    orchestrator = ResourceAwareOrchestrator()
    
    # Run complete retraining
    summary = orchestrator.run_complete_retraining(
        max_parallel=1,  # Sequential for now to avoid memory conflicts
        skip_completed=True  # Skip already trained models
    )
    
    print("\n" + "="*50)
    print("RETRAINING COMPLETE")
    print("="*50)
    print(f"Completed: {summary['completed']}/{summary['total_phases']} phases")
    print(f"Failed: {summary['failed']} phases")
    print(f"Total time: {summary['total_time_hours']:.1f} hours")
    
    if summary['failed_phases']:
        print(f"Failed phases: {', '.join(summary['failed_phases'])}")

if __name__ == "__main__":
    main()