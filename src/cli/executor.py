"""
Execution router for CAN-Graph training.

Routes training jobs to appropriate execution backend:
- Local execution (direct training on current node)
- SLURM submission (batch job to cluster)
- Pipeline mode (multi-stage with dependencies)
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import sys

logger = logging.getLogger(__name__)


# ============================================================================
# Execution Router
# ============================================================================

class ExecutionRouter:
    """Routes training execution to appropriate backend."""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize Execution Router.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root or Path.cwd()

    def execute_single(self, config: 'CANGraphConfig', run_type: Dict,
                      model_args: Dict, slurm_args: Dict,
                      mode: str = 'slurm', dry_run: bool = False,
                      skip_validation: bool = False) -> int:
        """
        Execute a single training job.

        Args:
            config: CANGraphConfig object
            run_type: Run type bucket
            model_args: Model args bucket
            slurm_args: SLURM args bucket
            mode: Execution mode ('slurm', 'local', 'dry-run')
            dry_run: If True, only preview (don't execute)
            skip_validation: If True, skip pre-flight validation

        Returns:
            Exit code (0 for success)
        """
        # Import modules dynamically to avoid package import issues
        import importlib.util
        import sys
        from pathlib import Path

        # Import validator
        validator_path = Path(__file__).parent / 'validator.py'
        spec = importlib.util.spec_from_file_location("validator", validator_path)
        validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator)
        validate_config = validator.validate_config
        ValidationError = validator.ValidationError

        # Import job_manager
        job_manager_path = Path(__file__).parent / 'job_manager.py'
        spec = importlib.util.spec_from_file_location("job_manager", job_manager_path)
        job_manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job_manager_module)
        JobManager = job_manager_module.JobManager

        # Import config_builder
        config_builder_path = Path(__file__).parent / 'config_builder.py'
        spec = importlib.util.spec_from_file_location("config_builder", config_builder_path)
        config_builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_builder)
        format_config_summary = config_builder.format_config_summary

        # Print configuration summary
        print(format_config_summary(run_type, model_args, slurm_args))

        # Pre-flight validation (unless skipped or dry-run)
        if not skip_validation and not dry_run:
            logger.info("\n📋 Running pre-flight validation...")
            result = validate_config(
                config,
                slurm_args=slurm_args,
                skip_artifact_check=(mode == 'dry-run')
            )

            print(result.format_report())

            if not result.is_valid():
                logger.error("❌ Validation failed - cannot proceed")
                return 1

        # Route based on mode
        if mode == 'slurm':
            logger.info("\n🚀 Submitting to SLURM cluster...")
            job_manager = JobManager(self.project_root)
            job_id = job_manager.submit_single(
                config, run_type, model_args, slurm_args, dry_run=False
            )

            if job_id:
                logger.info(f"\n✅ Job submitted successfully!")
                logger.info(f"   Job ID: {job_id}")
                logger.info(f"   Monitor: squeue -j {job_id}")
                logger.info(f"   Cancel: scancel {job_id}")
                logger.info(f"   Logs: {self.project_root}/experimentruns/slurm_runs/")
                return 0
            else:
                logger.error("❌ Job submission failed")
                return 1

        elif mode == 'local':
            logger.info("\n🏃 Running locally...")
            return self._execute_local(config, run_type, model_args)

        elif mode == 'dry-run':
            logger.info("\n👁️  DRY RUN MODE - No execution")
            logger.info("✅ Configuration is valid")
            return 0

        else:
            logger.error(f"Unknown execution mode: {mode}")
            return 1

    def execute_sweep(self, configs: List[Tuple[Dict, Dict, Dict]],
                     mode: str = 'slurm', dry_run: bool = False,
                     skip_validation: bool = False) -> int:
        """
        Execute a parameter sweep (multiple independent jobs).

        Args:
            configs: List of (run_type, model_args, slurm_args) tuples
            mode: Execution mode ('slurm', 'local', 'dry-run')
            dry_run: If True, only preview
            skip_validation: If True, skip pre-flight validation

        Returns:
            Exit code (0 for success)
        """
        # Import job_manager dynamically
        import importlib.util
        import sys
        from pathlib import Path

        job_manager_path = Path(__file__).parent / 'job_manager.py'
        spec = importlib.util.spec_from_file_location("job_manager", job_manager_path)
        job_manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job_manager_module)
        JobManager = job_manager_module.JobManager

        logger.info(f"\n📊 Parameter sweep: {len(configs)} configurations")

        if mode == 'slurm':
            job_manager = JobManager(self.project_root)
            job_ids = job_manager.submit_sweep(configs, dry_run=dry_run)

            if not dry_run:
                logger.info(f"\n✅ Sweep submitted: {len(job_ids)} jobs")
                logger.info(f"   Monitor: squeue -u $USER")
            return 0

        elif mode == 'local':
            logger.info("🏃 Running sweep locally (sequential)...")

            # Import config_builder for creating configs
            config_builder_path = Path(__file__).parent / 'config_builder.py'
            spec = importlib.util.spec_from_file_location("config_builder", config_builder_path)
            config_builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_builder)
            create_can_graph_config = config_builder.create_can_graph_config

            for idx, (run_type, model_args, slurm_args) in enumerate(configs, 1):
                logger.info(f"\n[{idx}/{len(configs)}] Running configuration...")
                config = create_can_graph_config(run_type, model_args, slurm_args)

                exit_code = self._execute_local(config, run_type, model_args)
                if exit_code != 0:
                    logger.error(f"❌ Configuration {idx} failed")
                    return exit_code

            logger.info(f"\n✅ Sweep completed: {len(configs)} configs")
            return 0

        elif mode == 'dry-run':
            logger.info("👁️  DRY RUN - Would submit {len(configs)} jobs")
            return 0

        else:
            logger.error(f"Unknown execution mode: {mode}")
            return 1

    def execute_pipeline(self, stages: List[Tuple[Dict, Dict, Dict]],
                        mode: str = 'slurm', dry_run: bool = False) -> int:
        """
        Execute a multi-stage pipeline with dependencies.

        Args:
            stages: List of (run_type, model_args, slurm_args) for each stage
            mode: Execution mode (only 'slurm' supported for pipelines)
            dry_run: If True, only preview

        Returns:
            Exit code (0 for success)
        """
        # Import job_manager dynamically
        import importlib.util
        import sys
        from pathlib import Path

        job_manager_path = Path(__file__).parent / 'job_manager.py'
        spec = importlib.util.spec_from_file_location("job_manager", job_manager_path)
        job_manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job_manager_module)
        JobManager = job_manager_module.JobManager

        logger.info(f"\n🔗 Pipeline mode: {len(stages)} stages")

        if mode != 'slurm':
            logger.error("❌ Pipeline mode only supports SLURM execution")
            logger.error("   (Local execution would run stages sequentially, use sweep instead)")
            return 1

        job_manager = JobManager(self.project_root)
        job_ids = job_manager.submit_pipeline(stages, dry_run=dry_run)

        if not dry_run:
            logger.info(f"\n✅ Pipeline submitted: {len(job_ids)} stages")
            logger.info("   Each stage waits for previous to complete")
            logger.info(f"   Monitor: squeue -u $USER")

        return 0

    def _execute_local(self, config: 'CANGraphConfig', run_type: Dict,
                      model_args: Dict) -> int:
        """
        Execute training locally (on current node).

        Args:
            config: CANGraphConfig object
            run_type: Run type bucket
            model_args: Model args bucket

        Returns:
            Exit code (0 for success)
        """
        try:
            # Import here to avoid heavy dependencies at startup
            logger.info("Loading training modules...")

            # This will integrate with HydraZenTrainer
            # For now, show what would be executed
            logger.info("Training with:")
            logger.info(f"  Model: {config.model.type}")
            logger.info(f"  Dataset: {config.dataset.name}")
            logger.info(f"  Mode: {config.training.mode}")
            logger.info(f"  Output: {config.canonical_experiment_dir()}")

            logger.warning("\n⚠️  LOCAL EXECUTION NOT YET IMPLEMENTED")
            logger.warning("   Integration with HydraZenTrainer pending")
            logger.warning("   Use --submit to run on SLURM for now")

            return 1

        except Exception as e:
            logger.error(f"❌ Local execution failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


# ============================================================================
# Helper Functions
# ============================================================================

def determine_execution_mode(submit: bool, local: bool, dry_run: bool,
                            smoke: bool) -> str:
    """
    Determine execution mode from CLI flags.

    Args:
        submit: --submit flag (SLURM)
        local: --local flag (local execution)
        dry_run: --dry-run flag (preview only)
        smoke: --smoke flag (quick test)

    Returns:
        Execution mode string: 'slurm', 'local', 'dry-run', 'smoke'
    """
    if dry_run:
        return 'dry-run'
    elif smoke:
        return 'smoke'
    elif submit:
        return 'slurm'
    elif local:
        return 'local'
    else:
        # Default to SLURM on HPC environment
        return 'slurm'
