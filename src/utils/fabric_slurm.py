"""
PyTorch Lightning Fabric SLURM environment wrapper and utilities.

This module provides:
- SLURM job submission utilities
- Fabric configuration for HPC environments
- Resource detection and optimization
- Multi-node training coordination
- Job monitoring and management
"""

import os
import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import tempfile
from datetime import datetime, timedelta

from lightning.fabric import Fabric
from src.utils.fabric_utils import setup_slurm_fabric_config

logger = logging.getLogger(__name__)


@dataclass
class SlurmJobConfig:
    """Configuration for SLURM job submission."""
    job_name: str
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 8
    gpus_per_node: int = 1
    mem: str = "32G"
    time: str = "24:00:00"
    partition: str = "gpu"
    account: Optional[str] = None
    qos: Optional[str] = None
    mail_type: str = "END,FAIL"
    mail_user: Optional[str] = None
    output: str = "slurm_%j.out"
    error: str = "slurm_%j.err"
    exclusive: bool = False
    constraint: Optional[str] = None
    gres: Optional[str] = None
    
    def to_sbatch_args(self) -> List[str]:
        """Convert configuration to sbatch command arguments."""
        args = [
            f"--job-name={self.job_name}",
            f"--nodes={self.nodes}",
            f"--ntasks-per-node={self.ntasks_per_node}",
            f"--cpus-per-task={self.cpus_per_task}",
            f"--mem={self.mem}",
            f"--time={self.time}",
            f"--partition={self.partition}",
            f"--output={self.output}",
            f"--error={self.error}",
            f"--mail-type={self.mail_type}"
        ]
        
        if self.gpus_per_node > 0:
            args.append(f"--gpus-per-node={self.gpus_per_node}")
        
        if self.account:
            args.append(f"--account={self.account}")
        
        if self.qos:
            args.append(f"--qos={self.qos}")
        
        if self.mail_user:
            args.append(f"--mail-user={self.mail_user}")
        
        if self.exclusive:
            args.append("--exclusive")
        
        if self.constraint:
            args.append(f"--constraint={self.constraint}")
        
        if self.gres:
            args.append(f"--gres={self.gres}")
        
        return args


class SlurmFabricManager:
    """
    Manager for PyTorch Lightning Fabric training jobs in SLURM environments.
    """
    
    def __init__(self, 
                 slurm_config: SlurmJobConfig,
                 project_root: str = ".",
                 modules_to_load: List[str] = None,
                 conda_env: str = None):
        """
        Initialize SLURM Fabric manager.
        
        Args:
            slurm_config: SLURM job configuration
            project_root: Root directory of the project
            modules_to_load: List of modules to load (e.g., ['cuda/11.8', 'python/3.9'])
            conda_env: Name of conda environment to activate
        """
        self.slurm_config = slurm_config
        self.project_root = Path(project_root).absolute()
        self.modules_to_load = modules_to_load or []
        self.conda_env = conda_env
        
        # Create directories for logs and scripts
        self.logs_dir = self.project_root / "slurm_logs"
        self.scripts_dir = self.project_root / "slurm_scripts"
        self.logs_dir.mkdir(exist_ok=True)
        self.scripts_dir.mkdir(exist_ok=True)
        
        logger.info(f"SLURM Fabric manager initialized for {self.slurm_config.job_name}")
    
    def generate_training_script(self,
                                training_command: str,
                                fabric_config: Dict[str, Any] = None,
                                environment_vars: Dict[str, str] = None) -> str:
        """
        Generate SLURM training script with Fabric configuration.
        
        Args:
            training_command: Python training command to execute
            fabric_config: Additional Fabric configuration
            environment_vars: Environment variables to set
            
        Returns:
            Path to generated script
        """
        # Merge with SLURM-detected configuration
        slurm_fabric_config = setup_slurm_fabric_config()
        if fabric_config:
            slurm_fabric_config.update(fabric_config)
        
        # Generate script content
        script_content = self._generate_script_content(
            training_command, slurm_fabric_config, environment_vars
        )
        
        # Save script
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = f"{self.slurm_config.job_name}_{timestamp}.sh"
        script_path = self.scripts_dir / script_name
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Training script generated: {script_path}")
        return str(script_path)
    
    def _generate_script_content(self,
                                training_command: str,
                                fabric_config: Dict[str, Any],
                                environment_vars: Dict[str, str] = None) -> str:
        """Generate the content of the SLURM training script."""
        
        # SLURM directives
        sbatch_directives = "\n".join([
            f"#SBATCH {arg}" for arg in self.slurm_config.to_sbatch_args()
        ])
        
        # Module loading
        module_commands = "\n".join([
            f"module load {module}" for module in self.modules_to_load
        ])
        
        # Environment activation
        env_commands = []
        if self.conda_env:
            env_commands.append(f"conda activate {self.conda_env}")
        
        # Environment variables
        env_vars_section = ""
        if environment_vars:
            env_vars_section = "\n".join([
                f"export {key}={value}" for key, value in environment_vars.items()
            ])
        
        # Fabric configuration as environment variables
        fabric_env_vars = self._fabric_config_to_env_vars(fabric_config)
        
        # Multi-node setup
        multi_node_setup = self._generate_multi_node_setup()
        
        script_template = f"""#!/bin/bash
{sbatch_directives}

# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_NODELIST"
echo "Number of Nodes: $SLURM_NNODES"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "Start Time: $(date)"

# Change to project directory
cd {self.project_root}

# Load modules
{module_commands}

# Activate environment
{chr(10).join(env_commands)}

# Set environment variables
{env_vars_section}
{fabric_env_vars}

# Multi-node setup
{multi_node_setup}

# Training command
echo "Starting training..."
echo "Command: {training_command}"
{training_command}

echo "Training completed at: $(date)"
"""
        
        return script_template
    
    def _fabric_config_to_env_vars(self, fabric_config: Dict[str, Any]) -> str:
        """Convert Fabric configuration to environment variables."""
        env_vars = []
        
        # Set common Fabric environment variables
        if 'devices' in fabric_config:
            env_vars.append(f"export FABRIC_DEVICES={fabric_config['devices']}")
        
        if 'num_nodes' in fabric_config:
            env_vars.append(f"export FABRIC_NUM_NODES={fabric_config['num_nodes']}")
        
        if 'precision' in fabric_config:
            env_vars.append(f"export FABRIC_PRECISION={fabric_config['precision']}")
        
        if 'strategy' in fabric_config:
            env_vars.append(f"export FABRIC_STRATEGY={fabric_config['strategy']}")
        
        # GPU optimization flags
        env_vars.extend([
            "export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID",
            "export NCCL_DEBUG=INFO",
            "export NCCL_TREE_THRESHOLD=0",
            "export TORCH_DISTRIBUTED_DEBUG=DETAIL"
        ])
        
        return "\n".join(env_vars)
    
    def _generate_multi_node_setup(self) -> str:
        """Generate multi-node coordination setup."""
        if self.slurm_config.nodes <= 1:
            return "# Single node training"
        
        return """
# Multi-node training setup
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Multi-node training setup:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
"""
    
    def submit_job(self, script_path: str) -> str:
        """
        Submit SLURM job and return job ID.
        
        Args:
            script_path: Path to the training script
            
        Returns:
            SLURM job ID
        """
        try:
            result = subprocess.run(
                ['sbatch', script_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract job ID from output (format: "Submitted batch job 12345")
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"Job submitted successfully. Job ID: {job_id}")
            
            # Save job info
            self._save_job_info(job_id, script_path)
            
            return job_id
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job: {e.stderr}")
            raise
    
    def _save_job_info(self, job_id: str, script_path: str):
        """Save job information for monitoring."""
        job_info = {
            'job_id': job_id,
            'job_name': self.slurm_config.job_name,
            'script_path': script_path,
            'submit_time': datetime.now().isoformat(),
            'slurm_config': asdict(self.slurm_config)
        }
        
        job_info_path = self.logs_dir / f"job_{job_id}.json"
        with open(job_info_path, 'w') as f:
            json.dump(job_info, f, indent=2)
    
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check SLURM job status.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Dictionary with job status information
        """
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '--format=%i,%T,%M,%l,%N', '--noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                fields = result.stdout.strip().split(',')
                return {
                    'job_id': fields[0],
                    'status': fields[1],
                    'time_used': fields[2],
                    'time_limit': fields[3],
                    'nodes': fields[4]
                }
            else:
                # Job not in queue, check if completed
                return self._check_completed_job(job_id)
                
        except subprocess.CalledProcessError:
            return {'job_id': job_id, 'status': 'UNKNOWN', 'error': 'Failed to query job'}
    
    def _check_completed_job(self, job_id: str) -> Dict[str, Any]:
        """Check status of completed job."""
        try:
            result = subprocess.run(
                ['sacct', '-j', job_id, '--format=JobID,State,ExitCode,CPUTime', '--noheader', '--parsable2'],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                # Get the main job info (not step info)
                for line in lines:
                    if not '.batch' in line and not '.extern' in line:
                        fields = line.split('|')
                        return {
                            'job_id': fields[0],
                            'status': fields[1],
                            'exit_code': fields[2],
                            'cpu_time': fields[3]
                        }
            
            return {'job_id': job_id, 'status': 'COMPLETED', 'info': 'Job finished'}
            
        except subprocess.CalledProcessError:
            return {'job_id': job_id, 'status': 'UNKNOWN', 'error': 'Failed to query completed job'}
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel SLURM job.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            subprocess.run(['scancel', job_id], check=True)
            logger.info(f"Job {job_id} cancelled successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def monitor_job(self, job_id: str, poll_interval: int = 60) -> Dict[str, Any]:
        """
        Monitor job until completion.
        
        Args:
            job_id: SLURM job ID
            poll_interval: Time between status checks in seconds
            
        Returns:
            Final job status
        """
        logger.info(f"Monitoring job {job_id} (polling every {poll_interval}s)")
        
        while True:
            status = self.check_job_status(job_id)
            logger.info(f"Job {job_id} status: {status.get('status', 'UNKNOWN')}")
            
            if status.get('status') in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
                logger.info(f"Job {job_id} finished with status: {status.get('status')}")
                return status
            
            time.sleep(poll_interval)
    
    def get_job_output(self, job_id: str) -> Dict[str, str]:
        """
        Get job output and error logs.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Dictionary with stdout and stderr content
        """
        output_files = {
            'stdout': self.logs_dir / f"slurm_{job_id}.out",
            'stderr': self.logs_dir / f"slurm_{job_id}.err"
        }
        
        content = {}
        for stream, filepath in output_files.items():
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        content[stream] = f.read()
                except Exception as e:
                    content[stream] = f"Error reading {stream}: {e}"
            else:
                content[stream] = f"{stream} file not found"
        
        return content


class FabricSlurmTrainingOrchestrator:
    """
    High-level orchestrator for running Fabric training jobs on SLURM.
    """
    
    def __init__(self, 
                 project_root: str = ".",
                 default_slurm_config: Dict[str, Any] = None):
        """
        Initialize the training orchestrator.
        
        Args:
            project_root: Root directory of the project
            default_slurm_config: Default SLURM configuration
        """
        self.project_root = Path(project_root).absolute()
        self.default_slurm_config = default_slurm_config or {}
        self.active_jobs = {}
        
    def submit_gat_training(self,
                           dataset: str,
                           model_type: str = "teacher",
                           custom_config: Dict[str, Any] = None,
                           wait_for_completion: bool = False) -> str:
        """
        Submit GAT training job to SLURM.
        
        Args:
            dataset: Dataset name (e.g., 'hcrl_ch', 'set_01')
            model_type: Type of model ('teacher', 'student')
            custom_config: Custom configuration overrides
            wait_for_completion: Whether to wait for job completion
            
        Returns:
            SLURM job ID
        """
        # Create job configuration
        job_config = SlurmJobConfig(
            job_name=f"gat_{model_type}_{dataset}",
            **self.default_slurm_config
        )
        
        # Create SLURM manager
        slurm_manager = SlurmFabricManager(
            job_config, 
            self.project_root,
            modules_to_load=['cuda/11.8', 'python/3.9'],
            conda_env='can-graph'
        )
        
        # Generate training command
        training_command = self._generate_gat_command(dataset, model_type, custom_config)
        
        # Generate and submit job
        script_path = slurm_manager.generate_training_script(training_command)
        job_id = slurm_manager.submit_job(script_path)
        
        # Track job
        self.active_jobs[job_id] = {
            'type': 'gat',
            'dataset': dataset,
            'model_type': model_type,
            'manager': slurm_manager
        }
        
        if wait_for_completion:
            return slurm_manager.monitor_job(job_id)
        
        return job_id
    
    def submit_vgae_training(self,
                            dataset: str,
                            model_type: str = "teacher",
                            custom_config: Dict[str, Any] = None,
                            wait_for_completion: bool = False) -> str:
        """
        Submit VGAE training job to SLURM.
        
        Args:
            dataset: Dataset name
            model_type: Type of model ('teacher', 'student')
            custom_config: Custom configuration overrides
            wait_for_completion: Whether to wait for job completion
            
        Returns:
            SLURM job ID
        """
        # Create job configuration
        job_config = SlurmJobConfig(
            job_name=f"vgae_{model_type}_{dataset}",
            **self.default_slurm_config
        )
        
        # Create SLURM manager
        slurm_manager = SlurmFabricManager(
            job_config, 
            self.project_root,
            modules_to_load=['cuda/11.8', 'python/3.9'],
            conda_env='can-graph'
        )
        
        # Generate training command
        training_command = self._generate_vgae_command(dataset, model_type, custom_config)
        
        # Generate and submit job
        script_path = slurm_manager.generate_training_script(training_command)
        job_id = slurm_manager.submit_job(script_path)
        
        # Track job
        self.active_jobs[job_id] = {
            'type': 'vgae',
            'dataset': dataset,
            'model_type': model_type,
            'manager': slurm_manager
        }
        
        if wait_for_completion:
            return slurm_manager.monitor_job(job_id)
        
        return job_id
    
    def _generate_gat_command(self, 
                             dataset: str, 
                             model_type: str,
                             custom_config: Dict[str, Any] = None) -> str:
        """Generate GAT training command."""
        base_command = f"python train_fabric_models.py --model gat --dataset {dataset} --type {model_type}"
        
        if custom_config:
            config_str = json.dumps(custom_config).replace('"', '\\"')
            base_command += f' --config "{config_str}"'
        
        return base_command
    
    def _generate_vgae_command(self, 
                              dataset: str, 
                              model_type: str,
                              custom_config: Dict[str, Any] = None) -> str:
        """Generate VGAE training command."""
        base_command = f"python train_fabric_models.py --model vgae --dataset {dataset} --type {model_type}"
        
        if custom_config:
            config_str = json.dumps(custom_config).replace('"', '\\"')
            base_command += f' --config "{config_str}"'
        
        return base_command
    
    def check_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Check status of all active jobs."""
        statuses = {}
        
        for job_id, job_info in self.active_jobs.items():
            manager = job_info['manager']
            status = manager.check_job_status(job_id)
            statuses[job_id] = {
                **job_info,
                **status
            }
        
        return statuses
    
    def cancel_all_jobs(self) -> Dict[str, bool]:
        """Cancel all active jobs."""
        results = {}
        
        for job_id, job_info in self.active_jobs.items():
            manager = job_info['manager']
            success = manager.cancel_job(job_id)
            results[job_id] = success
        
        return results


def create_slurm_config_for_gpu_type(gpu_type: str = "a100") -> SlurmJobConfig:
    """
    Create optimized SLURM configuration for specific GPU types.
    
    Args:
        gpu_type: Type of GPU ('a100', 'v100', 'h100', etc.)
        
    Returns:
        Optimized SLURM configuration
    """
    gpu_type = gpu_type.lower()
    
    if gpu_type == "a100":
        return SlurmJobConfig(
            job_name="fabric_training_a100",
            nodes=1,
            ntasks_per_node=1,
            cpus_per_task=16,
            gpus_per_node=1,
            mem="64G",
            time="48:00:00",
            partition="gpu",
            constraint="a100"
        )
    elif gpu_type == "v100":
        return SlurmJobConfig(
            job_name="fabric_training_v100",
            nodes=1,
            ntasks_per_node=1,
            cpus_per_task=12,
            gpus_per_node=1,
            mem="48G",
            time="72:00:00",
            partition="gpu",
            constraint="v100"
        )
    elif gpu_type == "h100":
        return SlurmJobConfig(
            job_name="fabric_training_h100",
            nodes=1,
            ntasks_per_node=1,
            cpus_per_task=20,
            gpus_per_node=1,
            mem="96G",
            time="24:00:00",
            partition="gpu",
            constraint="h100"
        )
    else:
        # Generic configuration
        return SlurmJobConfig(
            job_name="fabric_training_gpu",
            nodes=1,
            ntasks_per_node=1,
            cpus_per_task=8,
            gpus_per_node=1,
            mem="32G",
            time="48:00:00",
            partition="gpu"
        )