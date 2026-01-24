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

# NOTE: `{data_env_exports}` may be populated by the job manager when a
# canonical local dataset path is found on the submit host. This exports
# `CAN_DATA_PATH`/`DATA_PATH` inside the job to ensure training finds the
# same dataset location on the compute node.
SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --account={account}
#SBATCH --mail-type={notification_type}
#SBATCH --mail-user={email}
#SBATCH --output={log_path}
#SBATCH --error={error_path}
#SBATCH --chdir={project_root}

# Minimal KD-GAT Slurm script (simplified for portability)
set -euo pipefail

echo "Starting KD-GAT job on $(hostname)"
module load miniconda3/24.1.2-py310 || true
source activate gnn-experiments || true
module load cuda/11.8.0 || true
# Limit OpenMP/BLAS threads to avoid oversubscription on shared nodes
# GPU optimizations and CUDA debugging
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1  # Enable synchronous CUDA calls for better debugging
export TORCH_USE_CUDA_DSA=1   # Enable device-side assertions
export OMP_NUM_THREADS={cpus}
export MKL_NUM_THREADS={cpus}
export NUMEXPR_NUM_THREADS={cpus}
export OPENBLAS_NUM_THREADS={cpus}
# Enable faulthandler to get Python tracebacks on crashes
export PYTHONFAULTHANDLER=1
# Echo python executable for quick debugging
echo "python -> $(python -c 'import sys; print(sys.executable)')" || true

# Dataset environment exports (optional)
{data_env_exports}

# Run training (Hydra-Zen based)
echo "Running: python train_with_hydra_zen.py --preset={preset_name} {data_path_flag}"
python train_with_hydra_zen.py --preset {preset_name} {data_path_flag}

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Job finished successfully"
  exit 0
else
  echo "Job failed with exit code: $EXIT_CODE"
  exit $EXIT_CODE
fi

# Run training with Hydra-Zen
echo "🚀 Starting training..."
python src/training/train_with_hydra_zen.py --preset {preset_name} {data_path_flag}

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

    def _map_legacy_to_preset(self, name: str) -> str:
        """Map legacy job identifiers to canonical preset names.

        This is intentionally strict and only handles the known naming conventions that
        our job generator historically produced. It avoids fuzzy runtime behaviour in
        the training script while keeping generated Slurm scripts compatible.
        """
        s = name.lower()
        # Normalize tokens
        s = s.replace('automotive_', '')
        s = s.replace('automotive', '')
        s = s.replace('hcrlch', 'hcrl_ch')
        s = s.replace('hcrlsa', 'hcrl_sa')

        # Dataset detection
        import re
        ds_match = re.search(r'(hcrl_ch|hcrl_sa|set_0?\d|car_hacking)', s)
        dataset = ds_match.group(0) if ds_match else None
        if dataset:
            # Normalize set_1 -> set_01
            m = re.match(r'set_(\d)$', dataset)
            if m:
                dataset = f'set_0{m.group(1)}'

        # Autoencoder / VGAE / Unsupervised
        if dataset and any(k in s for k in ('autoencoder', 'vgae', 'unsupervised')):
            return f'autoencoder_{dataset}'

        # Supervised / GAT / Normal
        if dataset and any(k in s for k in ('supervised', 'gat', 'normal')):
            return f'gat_normal_{dataset}'

        # Fusion
        if dataset and 'fusion' in s:
            return f'fusion_{dataset}'

        # Distillation
        if dataset and ('distill' in s or 'distillation' in s):
            # Prefer balanced/default 0.5 if unspecified
            for scale in ('0.5', '0.25', '0.75'):
                candidate = f'distillation_{dataset}_scale_{scale}'
                # We don't have access to preset list here; return the first plausible
                return candidate

        # Fallback to original name (caller should ensure preset exists)
        return name    
    def _get_default_config(self) -> OmegaConf:
        """Get default OSC configuration"""
        return OmegaConf.create({
            'osc': {
                'account': 'PAS3209',
                'email': 'frenken.2@osu.edu',
                'notification_type': 'END,FAIL',
                'conda_env': 'gnn-experiments',
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
        
        # Derive a canonical preset name to use for script invocations
        preset_name = self._map_legacy_to_preset(config_name)

        # Use canonical preset name for generated filenames (avoids legacy mismatches)
        job_name = job_name or preset_name[:50]
        log_file = run_dir / f"{preset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        error_file = run_dir / f"{preset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.err"

        # Try to locate a local dataset path and pass it into the job if present
        data_path_flag = ""
        data_env_exports = ""
        try:
            import re
            ds_match = re.search(r'(hcrl_ch|hcrl_sa|set_0?\d|car_hacking)', preset_name)
            if ds_match:
                ds = ds_match.group(0)
                # Normalize set_1 -> set_01
                m = re.match(r'set_(\d)$', ds)
                if m:
                    ds = f'set_0{m.group(1)}'
                candidate_path = self.project_root / 'data' / 'automotive' / ds
                if candidate_path.exists():
                    # Pass absolute path as CLI flag and export env vars inside job
                    data_path_flag = f"--data-path {str(candidate_path)}"
                    data_env_exports = (
                        f"export CAN_DATA_PATH={str(candidate_path)}\n"
                        f"export DATA_PATH={str(candidate_path)}\n"
                    )
        except Exception:
            data_path_flag = ""

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
            preset_name=preset_name,
            data_path_flag=data_path_flag,
            data_env_exports=data_env_exports,
            timestamp=datetime.now().isoformat(),
            run_dir=str(experiment_dir),
        )
        
        # Write script
        script_path = run_dir / f"{preset_name}.sh"
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
        pre_submit: bool = False,
    ) -> Optional[str]:
        """
        Submit a job to Slurm.

        Args:
            config_name: Name of Hydra-Zen config
            experiment_dir: Directory to save results
            dry_run: If True, only create script without submitting
            walltime: Override walltime
            memory: Override memory
            pre_submit: If True, run `scripts/pre_submit_check.py` before submission and abort if it fails

        Returns:
            Job ID if submitted, None if dry_run or submission failed
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

        # Optional pre-submit readiness check
        import subprocess
        if pre_submit:
            # Try to infer dataset name from config_name (expected format: modality_dataset_...)
            parts = config_name.split('_')
            dataset_guess = parts[1] if len(parts) > 1 else None
            pre_cmd = ['python', 'scripts/pre_submit_check.py']
            if dataset_guess:
                pre_cmd += ['--dataset', dataset_guess]
            # Run with reasonable defaults: validate dataset and preview
            pre_cmd += ['--run-load', '--smoke', '--smoke-synthetic', '--preview-json']

            logger.info(f"🔒 Running pre-submit checks: {' '.join(pre_cmd)}")
            try:
                pre_res = subprocess.run(pre_cmd, capture_output=True, text=True, timeout=600)
                if pre_res.returncode != 0:
                    logger.error("Pre-submit checks failed; aborting job submission.")
                    logger.error(pre_res.stdout)
                    logger.error(pre_res.stderr)
                    return None
                else:
                    logger.info("Pre-submit checks passed; proceeding to submit job.")
            except Exception as e:
                logger.error(f"Pre-submit check invocation failed: {e}; aborting submission.")
                return None

        # Submit to Slurm
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

    def submit_experiment_sweep(self, modality: str = "automotive", dataset: str = "hcrlch",
                                learning_type: str = "unsupervised", model_architecture: str = "VGAE",
                                model_sizes: List[str] = None, distillations: List[str] = None,
                                training_modes: List[str] = None, dry_run: bool = False):
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
    submit_parser.add_argument('--pre-submit', action='store_true', help='Run pre-submit readiness checks before submitting (uses scripts/pre_submit_check.py)')
    
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

    # Preview subcommand: print an easy-to-read summary of a sweep (JSON or table)
    preview_parser = subparsers.add_parser('preview', help='Preview a sweep without creating scripts')
    preview_parser.add_argument('--modality', default='automotive')
    preview_parser.add_argument('--dataset', default='hcrlch')
    preview_parser.add_argument('--learning-type', default='unsupervised')
    preview_parser.add_argument('--model-architecture', default='VGAE')
    preview_parser.add_argument('--model-sizes', default='student', help='Comma-separated list')
    preview_parser.add_argument('--distillations', default='no', help='Comma-separated list')
    preview_parser.add_argument('--training-modes', default='all_samples', help='Comma-separated list')
    preview_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON output')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Create manager
    manager = OSCJobManager()

    # Handle preview
    if args.command == 'preview':
        model_sizes = [s.strip() for s in args.model_sizes.split(',') if s.strip()]
        distillations = [s.strip() for s in args.distillations.split(',') if s.strip()]
        training_modes = [s.strip() for s in args.training_modes.split(',') if s.strip()]

        previews = []
        for size in model_sizes:
            for distill in distillations:
                for mode in training_modes:
                    config_name = (
                        f"{args.modality}_{args.dataset}_{args.learning_type}_"
                        f"{args.model_architecture}_{size}_{distill}_{mode}"
                    )
                    run_dir = Path(manager.cfg.project_root) / 'experimentruns' / args.modality / args.dataset / args.learning_type / args.model_architecture / size / distill / mode

                    preview = {
                        'config_name': config_name,
                        'run_dir': str(run_dir),
                        'expected_artifacts': []
                    }
                    # Add fusion/curriculum artifact suggestions
                    if mode in ('fusion', 'rl_fusion'):
                        preview['expected_artifacts'].append(str(run_dir / '..' / '..' / 'unsupervised' / 'vgae' / 'teacher' / 'no_distillation' / 'autoencoder' / 'vgae_autoencoder.pth'))
                        preview['expected_artifacts'].append(str(run_dir / '..' / '..' / 'supervised' / 'gat' / 'teacher' / 'no_distillation' / 'normal' / f'gat_{args.dataset}_normal.pth'))
                    if mode in ('curriculum', 'curriculum_classifier'):
                        preview['expected_artifacts'].append(str(run_dir / '..' / '..' / 'unsupervised' / 'vgae' / 'teacher' / 'no_distillation' / 'autoencoder' / 'vgae_autoencoder.pth'))

                    previews.append(preview)

        if args.json:
            import json
            print(json.dumps(previews, indent=2))
        else:
            # Print a compact table
            print('\nPreview of generated configurations:')
            print('=' * 80)
            print(f"{'CONFIG NAME':<60} | {'RUN DIR'}")
            print('-' * 80)
            for p in previews:
                print(f"{p['config_name']:<60} | {p['run_dir']}")
            print('=' * 80)
        sys.exit(0)
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
