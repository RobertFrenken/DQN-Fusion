"""
Environment detection for safe execution across login nodes and compute nodes.

Prevents accidental resource hogging on shared HPC login nodes.
"""

import os
import subprocess
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ExecutionEnvironment(Enum):
    """Where the code is executing."""
    LOCAL_DEV = "local_dev"           # Developer workstation
    LOGIN_NODE = "login_node"         # HPC login node (shared resource)
    COMPUTE_NODE = "compute_node"     # HPC compute node (allocated via SLURM)
    UNKNOWN = "unknown"               # Unable to determine


@dataclass
class EnvironmentInfo:
    """Complete environment information."""
    env_type: ExecutionEnvironment
    slurm_job_id: Optional[str] = None
    slurm_submit_host: Optional[str] = None
    hostname: Optional[str] = None
    has_gpu: bool = False
    gpu_count: int = 0
    is_safe_for_training: bool = False

    def __str__(self) -> str:
        lines = [
            f"Environment: {self.env_type.value}",
            f"Hostname: {self.hostname}",
        ]
        if self.slurm_job_id:
            lines.append(f"SLURM Job ID: {self.slurm_job_id}")
        if self.has_gpu:
            lines.append(f"GPUs: {self.gpu_count}")
        lines.append(f"Safe for training: {'✓' if self.is_safe_for_training else '✗'}")
        return "\n".join(lines)


def detect_environment() -> EnvironmentInfo:
    """
    Detect current execution environment.

    Returns:
        EnvironmentInfo with complete environment details
    """
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    slurm_submit_host = os.environ.get('SLURM_SUBMIT_HOST')
    hostname = _get_hostname()
    has_gpu, gpu_count = _detect_gpus()

    # Determine environment type
    if slurm_job_id:
        # Running inside SLURM allocation
        env_type = ExecutionEnvironment.COMPUTE_NODE
        is_safe = True
    elif slurm_submit_host or _is_known_hpc_login_node(hostname):
        # On HPC system but not in allocation
        env_type = ExecutionEnvironment.LOGIN_NODE
        is_safe = False  # NOT safe for GPU training
    elif hostname and ('local' in hostname.lower() or 'pc' in hostname.lower() or 'laptop' in hostname.lower()):
        # Local development machine
        env_type = ExecutionEnvironment.LOCAL_DEV
        is_safe = True  # User's own machine
    else:
        # Unknown environment - be conservative
        env_type = ExecutionEnvironment.UNKNOWN
        is_safe = has_gpu  # Only safe if GPUs available (assume personal if GPU present)

    return EnvironmentInfo(
        env_type=env_type,
        slurm_job_id=slurm_job_id,
        slurm_submit_host=slurm_submit_host,
        hostname=hostname,
        has_gpu=has_gpu,
        gpu_count=gpu_count,
        is_safe_for_training=is_safe,
    )


def check_execution_safety(dry_run: bool = False, smoke_test: bool = False) -> None:
    """
    Check if current environment is safe for training.

    Raises RuntimeError if attempting to run heavy training on login node.

    Args:
        dry_run: If True, skip safety check (no actual execution)
        smoke_test: If True, allow execution (quick test)

    Raises:
        RuntimeError: If environment is unsafe for training
    """
    if dry_run or smoke_test:
        # Dry runs and smoke tests are always safe
        return

    env = detect_environment()

    if not env.is_safe_for_training:
        raise RuntimeError(
            f"\n{'='*70}\n"
            f"⚠️  UNSAFE EXECUTION ENVIRONMENT DETECTED\n"
            f"{'='*70}\n\n"
            f"You are attempting to run GPU training on:\n"
            f"  Environment: {env.env_type.value}\n"
            f"  Hostname: {env.hostname}\n\n"
            f"This is a SHARED LOGIN NODE. Running heavy workloads here\n"
            f"will impact all users and may result in job termination.\n\n"
            f"Safe alternatives:\n"
            f"  1. Submit to SLURM cluster:\n"
            f"     can-train <args> --submit\n\n"
            f"  2. Run quick smoke test (safe on login node):\n"
            f"     can-train <args> --smoke\n\n"
            f"  3. Preview configuration without execution:\n"
            f"     can-train <args> --dry-run\n\n"
            f"{'='*70}\n"
        )

    logger.info(f"✓ Environment is safe for training: {env.env_type.value}")


def _get_hostname() -> Optional[str]:
    """Get current hostname."""
    try:
        return subprocess.check_output(['hostname'], text=True).strip()
    except Exception:
        return os.environ.get('HOSTNAME')


def _detect_gpus() -> tuple[bool, int]:
    """
    Detect if GPUs are available and count them.

    Returns:
        (has_gpu, gpu_count) tuple
    """
    try:
        # Try nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            gpu_count = len([line for line in result.stdout.strip().split('\n') if line])
            return (gpu_count > 0, gpu_count)
    except Exception:
        pass

    # Fallback: check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible:
        try:
            gpu_count = len(cuda_visible.split(','))
            return (gpu_count > 0, gpu_count)
        except Exception:
            pass

    return (False, 0)


def _is_known_hpc_login_node(hostname: Optional[str]) -> bool:
    """
    Check if hostname matches known HPC login node patterns.

    Add your HPC system's login node patterns here.
    """
    if not hostname:
        return False

    hostname_lower = hostname.lower()

    # Common HPC login node patterns
    login_patterns = [
        'owens',           # OSC Owens
        'pitzer',          # OSC Pitzer
        'login',           # Generic
        'head',            # Generic head node
        'submit',          # Generic submit node
        'gateway',         # Generic gateway
        'frontend',        # Generic frontend
    ]

    return any(pattern in hostname_lower for pattern in login_patterns)


def print_environment_info() -> None:
    """Print detailed environment information (for debugging)."""
    env = detect_environment()
    print(env)

    # Additional SLURM info
    if env.env_type == ExecutionEnvironment.COMPUTE_NODE:
        print("\nSLURM Allocation:")
        for var in ['SLURM_JOB_NAME', 'SLURM_NTASKS', 'SLURM_CPUS_PER_TASK',
                    'SLURM_MEM_PER_NODE', 'SLURM_GPUS']:
            value = os.environ.get(var)
            if value:
                print(f"  {var}: {value}")
