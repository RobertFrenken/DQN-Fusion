#!/usr/bin/env python3
# ============================================================================
# KD-GAT OSC Slurm Job Manager
# Submits experiments to Ohio Supercomputer Center with proper error checking
# ============================================================================

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# SLURM SCRIPT TEMPLATE
# ============================================================================

SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus-per-node={gpus}
#SBATCH --account={account}
#SBATCH --mail-type={notification_type}
#SBATCH --mail-user={email}
#SBATCH --output={log_path}
#SBATCH --error={error_path}
#SBATCH --chdir={project_root}

# ============================================================================
# KD-GAT Experiment Job
# Submitted: {timestamp}
# Config: {config_name}
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "KD-GAT Experiment Start"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "GPU Count: $(nvidia-smi --list-gpus | wc -l)"
echo ""

# Load conda environment
module load conda
conda activate {conda_env}

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: CUDA/GPU not available"
    exit 1
fi

echo "✅ GPU available"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Run training with Hydra-Zen
echo "🚀 Starting training..."
python src/training/train_with_hydra_zen.py \\
    config_store={config_name} \\
    hydra.run.dir={run_dir}

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ JOB COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "Results saved to: {run_dir}"
    echo ""
    exit 0
else
    echo ""
    echo "=========================================="
    echo "❌ JOB FAILED"
    echo "=========================================="
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "Check logs: {error_path}"
    echo ""
    exit 1
fi
"""

# ============================================================================
# JOB MANAGER CLASS
# ============================================================================

class OSCJobManager:
    """Manage KD-GAT experiment submissions to OSC Slurm"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize job manager"""
        
        self.project_root = Path(__file__).parent.absolute()
        
        # Load config
        if config_path and Path(config_path).exists():
            self.cfg = OmegaConf.load(config_path)
        else:
            # Use defaults
            self.cfg = self._get_default_config()
        
        # Job queue
        self.jobs: List[Dict] = []
    
    def _get_default_config(self) -> OmegaConf:
        """Get default OSC configuration"""
        return OmegaConf.create({
            'osc': {
                'account': 'PAS3209',
                'email': 'frenken.2@osu.edu',
                'notification_type': 'END,FAIL',
                'conda_env': 'gnn-gpu',
                'submit_host': 'owens.osc.edu',
                'walltime': '02:00:00',
                'memory': '32G',
                'cpus_per_task': 8,
                'gpus_per_node': 1,
                'gpu_type': 'v100',
            },
            'project_root': str(self.project_root),
        })
    
    def create_slurm_script(
        self,
        config_name: str,
        experiment_dir: Path,
        walltime: Optional[str] = None,
        memory: Optional[str] = None,
        job_name: Optional[str] = None,
    ) -> Path:
        """
        Create a Slurm submission script.
        
        Args:
            config_name: Name of Hydra-Zen config to use
            experiment_dir: Directory to save results
            walltime: Job walltime (default from config)
            memory: Memory allocation (default from config)
            job_name: Job name for Slurm (default: config_name)
            
        Returns:
            Path to created script
        """
        
        # Use defaults if not specified
        walltime = walltime or self.cfg.osc.walltime
        memory = memory or self.cfg.osc.memory
        job_name = job_name or config_name[:50]  # Slurm limit
        
        # Create experiment run directory
        run_dir = experiment_dir / "slurm_runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Log paths
        log_file = run_dir / f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        error_file = run_dir / f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.err"
        
        # Create script
        script_content = SLURM_SCRIPT_TEMPLATE.format(
            job_name=job_name,
            walltime=walltime,
            memory=memory,
            cpus=self.cfg.osc.cpus_per_task,
            gpus=self.cfg.osc.gpus_per_node,
            account=self.cfg.osc.account,
            notification_type=self.cfg.osc.notification_type,
            email=self.cfg.osc.email,
            log_path=str(log_file),
            error_path=str(error_file),
            project_root=self.project_root,
            conda_env=self.cfg.osc.conda_env,
            config_name=config_name,
            timestamp=datetime.now().isoformat(),
            run_dir=str(experiment_dir),
        )
        
        # Write script
        script_path = run_dir / f"{config_name}.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        logger.info(f"✅ Created Slurm script: {script_path}")
        logger.info(f"   Walltime: {walltime}")
        logger.info(f"   Memory: {memory}")
        logger.info(f"   GPUs: {self.cfg.osc.gpus_per_node}")
        logger.info(f"   Log: {log_file}")
        
        return script_path
    
    def submit_job(
        self,
        config_name: str,
        experiment_dir: Path,
        dry_run: bool = False,
        walltime: Optional[str] = None,
        memory: Optional[str] = None,
    ) -> Optional[str]:
        """
        Submit a job to Slurm.
        
        Args:
            config_name: Name of Hydra-Zen config
            experiment_dir: Directory to save results
            dry_run: If True, only create script without submitting
            walltime: Override walltime
            memory: Override memory
            
        Returns:
            Job ID if submitted, None if dry_run
        """
        
        # Create script
        script_path = self.create_slurm_script(
            config_name=config_name,
            experiment_dir=experiment_dir,
            walltime=walltime,
            memory=memory,
        )
        
        if dry_run:
            logger.info(f"🔍 [DRY RUN] Would submit: {script_path}")
            logger.info("Script content:")
            print("-" * 70)
            print(script_path.read_text())
            print("-" * 70)
            return None
        
        # Submit to Slurm
        import subprocess
        
        try:
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                # Extract job ID
                job_id = result.stdout.strip().split()[-1]
                logger.info(f"✅ Job submitted successfully")
                logger.info(f"   Job ID: {job_id}")
                logger.info(f"   Script: {script_path}")
                
                self.jobs.append({
                    'job_id': job_id,
                    'config_name': config_name,
                    'script_path': str(script_path),
                    'timestamp': datetime.now().isoformat(),
                })
                
                return job_id
            else:
                logger.error(f"❌ Failed to submit job")
                logger.error(f"   stdout: {result.stdout}")
                logger.error(f"   stderr: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error submitting job: {e}")
            return None
    
    def submit_experiment_sweep(
        self,
        modality: str = "automotive",
        dataset: str = "hcrlch",
        learning_type: str = "unsupervised",
        model_architecture: str = "VGAE",
        model_sizes: List[str] = None,
        distillations: List[str] = None,
        training_modes: List[str] = None,
        dry_run: bool = False,
    ):
        """
        Submit multiple experiment configurations.
        
        Args:
            modality: Modality (automotive, internet, watertreatment)
            dataset: Dataset name
            learning_type: unsupervised, classifier, fusion
            model_architecture: VGAE, GAT, DQN
            model_sizes: List of sizes [teacher, student, intermediate, huge, tiny]
            distillations: List of distillations [no, standard, topology_preserving]
            training_modes: List of training modes
            dry_run: If True, don't actually submit
        """
        
        model_sizes = model_sizes or ["student"]
        distillations = distillations or ["no"]
        training_modes = training_modes or ["all_samples"]
        
        experiment_dir = self.project_root / "experimentruns"
        
        job_count = 0
        for size in model_sizes:
            for distill in distillations:
                for mode in training_modes:
                    config_name = (
                        f"{modality}_{dataset}_{learning_type}_"
                        f"{model_architecture}_{size}_{distill}_{mode}"
                    )
                    
                    self.submit_job(
                        config_name=config_name,
                        experiment_dir=experiment_dir,
                        dry_run=dry_run,
                    )
                    job_count += 1
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"📊 Sweep Summary")
        logger.info(f"{'=' * 70}")
        logger.info(f"Configurations: {job_count}")
        logger.info(f"Model sizes: {len(model_sizes)}")
        logger.info(f"Distillations: {len(distillations)}")
        logger.info(f"Training modes: {len(training_modes)}")
        logger.info(f"Total jobs: {job_count}")
    
    def print_job_queue(self):
        """Print submitted jobs"""
        if not self.jobs:
            logger.info("No jobs submitted yet")
            return
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Submitted Jobs ({len(self.jobs)})")
        logger.info(f"{'=' * 70}")
        
        for i, job in enumerate(self.jobs, 1):
            logger.info(f"{i}. Job ID: {job['job_id']}")
            logger.info(f"   Config: {job['config_name']}")
            logger.info(f"   Submitted: {job['timestamp']}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface"""
    
    parser = argparse.ArgumentParser(
        description="KD-GAT OSC Slurm Job Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit single experiment
  python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_student_no_all_samples
  
  # Dry run (create script without submitting)
  python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_student_no_all_samples --dry-run
  
  # Submit sweep of experiments
  python oscjobmanager.py sweep --model-sizes student,teacher --distillations no,standard
  
  # Submit with custom resources
  python oscjobmanager.py submit config_name --walltime 04:00:00 --memory 64G
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit a single experiment')
    submit_parser.add_argument('config_name', help='Hydra-Zen config name')
    submit_parser.add_argument('--dry-run', action='store_true', help='Create script without submitting')
    submit_parser.add_argument('--walltime', help='Job walltime (e.g., 04:00:00)')
    submit_parser.add_argument('--memory', help='Memory allocation (e.g., 64G)')
    
    # Sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Submit multiple experiments')
    sweep_parser.add_argument('--modality', default='automotive')
    sweep_parser.add_argument('--dataset', default='hcrlch')
    sweep_parser.add_argument('--learning-type', default='unsupervised')
    sweep_parser.add_argument('--model-architecture', default='VGAE')
    sweep_parser.add_argument('--model-sizes', default='student', help='Comma-separated list')
    sweep_parser.add_argument('--distillations', default='no', help='Comma-separated list')
    sweep_parser.add_argument('--training-modes', default='all_samples', help='Comma-separated list')
    sweep_parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create manager
    manager = OSCJobManager()
    
    if args.command == 'submit':
        experiment_dir = manager.project_root / "experimentruns"
        manager.submit_job(
            config_name=args.config_name,
            experiment_dir=experiment_dir,
            dry_run=args.dry_run,
            walltime=args.walltime,
            memory=args.memory,
        )
        manager.print_job_queue()
    
    elif args.command == 'sweep':
        manager.submit_experiment_sweep(
            modality=args.modality,
            dataset=args.dataset,
            learning_type=args.learning_type,
            model_architecture=args.model_architecture,
            model_sizes=args.model_sizes.split(','),
            distillations=args.distillations.split(','),
            training_modes=args.training_modes.split(','),
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    main()
