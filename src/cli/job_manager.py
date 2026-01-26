"""
SLURM Job Manager for CAN-Graph training.

Generates and submits SLURM batch scripts for training jobs.
Handles single runs, sweeps, and pipeline mode with dependencies.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import subprocess
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SLURM Script Template
# ============================================================================

SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpu_type}:{gpus}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --mail-type={notification_type}
#SBATCH --mail-user={email}
#SBATCH --output={log_path}
#SBATCH --error={error_path}
#SBATCH --chdir={project_root}
{dependency_line}

# CAN-Graph Training Job
# Generated: {timestamp}
set -euo pipefail

echo "=================================================================="
echo "CAN-Graph Training Job"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=================================================================="

# Load environment
module load miniconda3/24.1.2-py310 || true
source activate gnn-experiments || true
module load cuda/12.3.0 || module load cuda/11.8.0 || true

# Environment configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export OMP_NUM_THREADS={cpus}
export MKL_NUM_THREADS={cpus}
export NUMEXPR_NUM_THREADS={cpus}
export OPENBLAS_NUM_THREADS={cpus}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1

echo "Python: $(which python)"
echo "=================================================================="

# Run training with bucket-based config
echo "Running: python train_with_hydra_zen.py [args...]"

python train_with_hydra_zen.py \\
{training_args}

EXIT_CODE=$?

echo "=================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ JOB COMPLETED SUCCESSFULLY"
    echo "Results: {output_dir}"
else
    echo "❌ JOB FAILED (exit code: $EXIT_CODE)"
    echo "Check error log: {error_path}"
fi
echo "End time: $(date)"
echo "=================================================================="

exit $EXIT_CODE
"""


# ============================================================================
# Job Manager
# ============================================================================

class JobManager:
    """Manages SLURM job submission for CAN-Graph training."""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize Job Manager.

        Args:
            project_root: Project root directory (defaults to CWD)
        """
        self.project_root = project_root or Path.cwd()
        self.slurm_runs_dir = self.project_root / "experimentruns" / "slurm_runs"
        self.slurm_runs_dir.mkdir(parents=True, exist_ok=True)

    def generate_job_name(self, config: 'CANGraphConfig') -> str:
        """Generate a descriptive job name for SLURM."""
        # Format: model_dataset_mode (e.g., gat_hcrl_ch_curriculum)
        model_type = config.model.type.replace('_student', '_s').replace('_teacher', '_t')
        dataset = config.dataset.name
        mode = config.training.mode[:10]  # Truncate if too long
        return f"{model_type}_{dataset}_{mode}"

    def generate_log_paths(self, job_name: str, dataset: str = None) -> Tuple[Path, Path]:
        """
        Generate log file paths for SLURM job.

        Args:
            job_name: SLURM job name
            dataset: Dataset name for subfolder organization (e.g., 'hcrl_ch', 'set_01')

        Returns:
            Tuple of (stdout_path, stderr_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"{job_name}_{timestamp}"

        # Organize logs by dataset subfolder
        if dataset:
            log_dir = self.slurm_runs_dir / dataset
            log_dir.mkdir(parents=True, exist_ok=True)
        else:
            log_dir = self.slurm_runs_dir

        stdout_path = log_dir / f"{log_name}.out"
        stderr_path = log_dir / f"{log_name}.err"

        return stdout_path, stderr_path

    def format_training_args(self, run_type: Dict, model_args: Dict,
                            slurm_args: Dict) -> str:
        """
        Format training arguments for CLI.

        Args:
            run_type: Run type bucket (model, model_size, dataset, mode)
            model_args: Model args bucket
            slurm_args: SLURM args bucket (not used in training command)

        Returns:
            Formatted argument string for train_with_hydra_zen.py
        """
        # For now, we'll use the preset-based approach until train_with_hydra_zen.py
        # is updated to accept bucket syntax directly

        # Map to preset-like args
        model = run_type['model']
        model_size = run_type['model_size']
        dataset = run_type['dataset']
        mode = run_type['mode']

        # Build model type
        if model_size == 'student':
            model_type = f"{model}_student"
        else:
            model_type = model

        # Create args list using argparse style
        args = []
        args.append(f"    --model {model_type}")
        args.append(f"    --dataset {dataset}")
        args.append(f"    --training {mode}")

        # Add model arg overrides
        for key, value in model_args.items():
            # Convert to argparse style
            if key == 'epochs':
                args.append(f"    --epochs {value}")
            elif key == 'learning_rate':
                args.append(f"    --learning_rate {value}")
            elif key == 'batch_size':
                args.append(f"    --batch_size {value}")
            # train_with_hydra_zen.py doesn't support these overrides directly
            # but we can add them for future integration

        return " \\\n".join(args)

    def generate_script(self, config: 'CANGraphConfig', run_type: Dict,
                       model_args: Dict, slurm_args: Dict,
                       dependency_job_id: Optional[str] = None) -> Tuple[str, Path]:
        """
        Generate SLURM batch script for a training job.

        Args:
            config: CANGraphConfig object
            run_type: Run type bucket
            model_args: Model args bucket
            slurm_args: SLURM args bucket
            dependency_job_id: Optional job ID to depend on

        Returns:
            Tuple of (script_content, script_path)
        """
        job_name = self.generate_job_name(config)
        dataset = config.dataset.name
        stdout_log, stderr_log = self.generate_log_paths(job_name, dataset=dataset)

        # Build dependency line if needed
        dependency_line = ""
        if dependency_job_id:
            dependency_line = f"#SBATCH --dependency=afterok:{dependency_job_id}"

        # Format training arguments
        training_args = self.format_training_args(run_type, model_args, slurm_args)

        # Fill template (use 'or' to handle explicit None values)
        script_content = SLURM_SCRIPT_TEMPLATE.format(
            job_name=job_name,
            walltime=slurm_args.get('walltime') or '06:00:00',
            memory=slurm_args.get('memory') or '64G',
            cpus=slurm_args.get('cpus') or 16,
            gpus=slurm_args.get('gpus') or 1,
            gpu_type=slurm_args.get('gpu_type') or 'v100',
            account=slurm_args.get('account') or 'PAS3209',
            partition=slurm_args.get('partition') or 'gpu',
            notification_type=slurm_args.get('notification_type') or 'END,FAIL',
            email=slurm_args.get('email') or 'frugoli.1@osu.edu',
            log_path=stdout_log,
            error_path=stderr_log,
            project_root=self.project_root,
            dependency_line=dependency_line,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            training_args=training_args,
            output_dir=config.canonical_experiment_dir(),
        )

        # Write script to dataset subfolder
        script_dir = self.slurm_runs_dir / dataset
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"{job_name}.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)  # Make executable

        logger.info(f"Generated SLURM script: {script_path}")

        return script_content, script_path

    def submit_job(self, script_path: Path, dry_run: bool = False) -> Optional[str]:
        """
        Submit a SLURM job.

        Args:
            script_path: Path to SLURM batch script
            dry_run: If True, don't actually submit (just show command)

        Returns:
            Job ID if submitted, None if dry_run

        Raises:
            RuntimeError: If submission fails
        """
        if dry_run:
            logger.info(f"[DRY RUN] Would submit: sbatch {script_path}")
            return None

        try:
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse job ID from output: "Submitted batch job 12345678"
            output = result.stdout.strip()
            if "Submitted batch job" in output:
                job_id = output.split()[-1]
                logger.info(f"✅ Submitted job {job_id}: {script_path.name}")
                return job_id
            else:
                raise RuntimeError(f"Unexpected sbatch output: {output}")

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to submit job: {e}")
            logger.error(f"   stdout: {e.stdout}")
            logger.error(f"   stderr: {e.stderr}")
            raise RuntimeError(f"Job submission failed: {e}")
        except FileNotFoundError:
            raise RuntimeError(
                "sbatch command not found. Are you on a SLURM cluster?"
            )

    def submit_single(self, config: 'CANGraphConfig', run_type: Dict,
                     model_args: Dict, slurm_args: Dict,
                     dry_run: bool = False) -> Optional[str]:
        """
        Submit a single training job.

        Args:
            config: CANGraphConfig object (can be None if run_type is provided)
            run_type: Run type bucket
            model_args: Model args bucket
            slurm_args: SLURM args bucket (can include 'dependency' for SLURM afterok)
            dry_run: If True, generate script but don't submit

        Returns:
            Job ID if submitted, None if dry_run
        """
        # Extract dependency from slurm_args if present
        dependency_job_id = slurm_args.pop('dependency', None) if slurm_args else None

        # If config is None, build it from run_type
        if config is None:
            import importlib.util
            config_builder_path = Path(__file__).parent / 'config_builder.py'
            spec = importlib.util.spec_from_file_location("config_builder", config_builder_path)
            config_builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_builder)
            create_can_graph_config = config_builder.create_can_graph_config
            config = create_can_graph_config(run_type, model_args, slurm_args)

        script_content, script_path = self.generate_script(
            config, run_type, model_args, slurm_args,
            dependency_job_id=dependency_job_id
        )

        if dry_run:
            logger.info("\n" + "="*70)
            logger.info("GENERATED SLURM SCRIPT:")
            logger.info("="*70)
            print(script_content)
            logger.info("="*70)
            logger.info(f"Script saved to: {script_path}")
            return None

        return self.submit_job(script_path, dry_run=False)

    def submit_sweep(self, configs: List[Tuple[Dict, Dict, Dict]],
                    dry_run: bool = False) -> List[Optional[str]]:
        """
        Submit multiple jobs from a parameter sweep.

        Args:
            configs: List of (run_type, model_args, slurm_args) tuples
            dry_run: If True, generate scripts but don't submit

        Returns:
            List of job IDs (or None for dry_run)
        """
        job_ids = []

        logger.info(f"Submitting sweep with {len(configs)} jobs...")

        # Import config_builder dynamically
        import importlib.util
        import sys

        config_builder_path = Path(__file__).parent / 'config_builder.py'
        spec = importlib.util.spec_from_file_location("config_builder", config_builder_path)
        config_builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_builder)
        create_can_graph_config = config_builder.create_can_graph_config

        for idx, (run_type, model_args, slurm_args) in enumerate(configs, 1):
            logger.info(f"\n[{idx}/{len(configs)}] Preparing job...")

            config = create_can_graph_config(run_type, model_args, slurm_args)

            job_id = self.submit_single(config, run_type, model_args, slurm_args, dry_run)
            job_ids.append(job_id)

        if not dry_run:
            logger.info(f"\n✅ Submitted {len(job_ids)} jobs")
            logger.info("Job IDs: " + ", ".join(str(jid) for jid in job_ids if jid))

        return job_ids

    def submit_pipeline(self, stages: List[Tuple[Dict, Dict, Dict]],
                       dry_run: bool = False) -> List[Optional[str]]:
        """
        Submit pipeline of dependent jobs.

        Each job waits for the previous to complete successfully.

        Args:
            stages: List of (run_type, model_args, slurm_args) for each stage
            dry_run: If True, generate scripts but don't submit

        Returns:
            List of job IDs (or None for dry_run)
        """
        job_ids = []
        prev_job_id = None

        logger.info(f"Submitting pipeline with {len(stages)} stages...")

        # Import config_builder dynamically
        import importlib.util
        import sys

        config_builder_path = Path(__file__).parent / 'config_builder.py'
        spec = importlib.util.spec_from_file_location("config_builder", config_builder_path)
        config_builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_builder)
        create_can_graph_config = config_builder.create_can_graph_config

        for idx, (run_type, model_args, slurm_args) in enumerate(stages, 1):
            logger.info(f"\n[Stage {idx}/{len(stages)}]")

            config = create_can_graph_config(run_type, model_args, slurm_args)

            # Generate script with dependency
            script_content, script_path = self.generate_script(
                config, run_type, model_args, slurm_args,
                dependency_job_id=prev_job_id
            )

            if dry_run:
                logger.info(f"[DRY RUN] Stage {idx} script: {script_path}")
                job_ids.append(None)
            else:
                job_id = self.submit_job(script_path)
                job_ids.append(job_id)
                prev_job_id = job_id

        if not dry_run:
            logger.info(f"\n✅ Submitted pipeline with {len(job_ids)} stages")
            logger.info("Job dependency chain: " + " → ".join(str(jid) for jid in job_ids if jid))

        return job_ids
