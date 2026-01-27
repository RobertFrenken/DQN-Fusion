#!/usr/bin/env python3
"""
Unified CLI entry point for KD-GAT experiments.

Philosophy: Explicit > Implicit
- Verbose, clear arguments
- No magic preset names
- Full configurability
- Lookup guide for common patterns
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Handle both relative (package) and absolute (standalone) imports
try:
    from .environment import (
        detect_environment,
        check_execution_safety,
        print_environment_info,
        ExecutionEnvironment,
    )
except ImportError:
    # Standalone mode: import sibling module directly
    import importlib.util
    env_path = Path(__file__).parent / 'environment.py'
    spec = importlib.util.spec_from_file_location("environment", env_path)
    environment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(environment)
    detect_environment = environment.detect_environment
    check_execution_safety = environment.check_execution_safety
    print_environment_info = environment.print_environment_info
    ExecutionEnvironment = environment.ExecutionEnvironment

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create flexible, verbose argument parser."""

    parser = argparse.ArgumentParser(
        prog='can-train',
        description='Unified CLI for KD-GAT graph neural network training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explicit configuration (recommended)
  can-train \\
    --model gat \\
    --model-size teacher \\
    --dataset hcrl_ch \\
    --mode normal \\
    --submit

  # VGAE autoencoder training
  can-train \\
    --model vgae \\
    --model-size teacher \\
    --dataset set_01 \\
    --mode autoencoder \\
    --epochs 50 \\
    --submit

  # Load from YAML config
  can-train --config experiments/my_config.yaml --submit

  # Dry run (preview without execution)
  can-train --model gat --dataset hcrl_ch --dry-run

  # Smoke test (quick validation)
  can-train --model gat --dataset hcrl_ch --smoke

  # Multi-stage pipeline
  can-train pipeline \\
    --stages autoencoder,curriculum,fusion \\
    --dataset hcrl_ch \\
    --submit

For more examples and patterns, see: docs/CLI_REFERENCE.md
        """)

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Main training command (default)
    train_parser = subparsers.add_parser('train', help='Train a single model', add_help=False)
    _add_training_args(train_parser)

    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run multi-stage pipeline')
    _add_pipeline_args(pipeline_parser)

    # Utility commands
    subparsers.add_parser('env', help='Show environment information')
    subparsers.add_parser('list-configs', help='List available example configurations')

    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--config', type=Path, help='Config file to validate')

    # If no subcommand, treat as 'train' (convenience)
    _add_training_args(parser)

    return parser


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add training-specific arguments."""

    # ============================================================================
    # CORE CONFIGURATION (explicit, required)
    # ============================================================================
    core = parser.add_argument_group('Core Configuration')

    core.add_argument(
        '--job-type',
        choices=['single', 'pipeline'],
        default='single',
        help='Job execution type - single job or multi-job pipeline (default: single)'
    )

    core.add_argument(
        '--modality',
        choices=['automotive', 'industrial', 'robotics'],
        default='automotive',
        help='Application domain (default: automotive)'
    )

    core.add_argument(
        '--model',
        choices=['gat', 'vgae', 'dqn', 'gcn', 'gnn', 'graphsage'],
        help='Model architecture (required unless --config specified)'
    )

    core.add_argument(
        '--model-size',
        choices=['teacher', 'student'],
        default='teacher',
        help='Model size variant - teacher (full) or student (compressed) (default: teacher)'
    )

    core.add_argument(
        '--dataset',
        help='Dataset name (e.g., hcrl_ch, hcrl_sa, set_01, set_02, etc.)'
    )

    core.add_argument(
        '--learning-type',
        choices=['supervised', 'unsupervised', 'semi_supervised', 'rl_fusion'],
        help='ML learning paradigm (required unless --config specified). Use: supervised for classifiers, unsupervised for autoencoders, rl_fusion for DQN fusion'
    )

    core.add_argument(
        '--distillation',
        choices=['with-kd', 'no-kd'],
        default='no-kd',
        help='Knowledge distillation flag (default: no-kd). Independent of --model-size; any model can learn via KD'
    )

    core.add_argument(
        '--training-strategy',
        dest='training_strategy',
        choices=['normal', 'autoencoder', 'curriculum', 'fusion', 'distillation', 'evaluation'],
        help='Training strategy (required unless --config specified). Options: normal, autoencoder, curriculum, fusion'
    )

    # ============================================================================
    # CONFIGURATION FILE (alternative to explicit args)
    # ============================================================================
    config_group = parser.add_argument_group('Configuration File')

    config_group.add_argument(
        '--config',
        type=Path,
        help='Load configuration from YAML file (alternative to explicit args)'
    )

    config_group.add_argument(
        '--preset',
        help='[DEPRECATED] Load from preset name. Use explicit args or --config instead.'
    )

    # ============================================================================
    # EXECUTION MODE
    # ============================================================================
    exec_mode = parser.add_argument_group('Execution Mode')

    exec_mode.add_argument(
        '--submit',
        action='store_true',
        help='Submit to SLURM cluster (default: run locally if safe)'
    )

    exec_mode.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview configuration without execution'
    )

    exec_mode.add_argument(
        '--smoke',
        action='store_true',
        help='Quick smoke test (reduced epochs, safe for login node)'
    )

    exec_mode.add_argument(
        '--local',
        action='store_true',
        help='Force local execution (overrides environment safety check)'
    )

    # ============================================================================
    # HYPERPARAMETERS (override defaults)
    # ============================================================================
    hyper = parser.add_argument_group('Hyperparameters')

    hyper.add_argument('--learning-rate', '--lr', type=float, help='Learning rate')
    hyper.add_argument('--batch-size', type=int, help='Batch size')
    hyper.add_argument('--epochs', type=int, help='Number of epochs')
    hyper.add_argument('--hidden-channels', type=int, help='Hidden layer size')
    hyper.add_argument('--dropout', type=float, help='Dropout rate')
    hyper.add_argument('--weight-decay', type=float, help='Weight decay (L2 regularization)')

    # ============================================================================
    # MODEL-SPECIFIC OPTIONS
    # ============================================================================
    model_opts = parser.add_argument_group('Model Options')

    # GAT-specific
    model_opts.add_argument('--num-layers', type=int, help='Number of GNN layers')
    model_opts.add_argument('--heads', type=int, help='Number of attention heads (GAT only)')

    # VGAE-specific
    model_opts.add_argument('--latent-dim', type=int, help='Latent dimension (VGAE only)')

    # DQN-specific
    model_opts.add_argument('--replay-buffer-size', type=int, help='Replay buffer size (DQN only)')
    model_opts.add_argument('--gamma', type=float, help='Discount factor (DQN only)')

    # ============================================================================
    # DISTILLATION OPTIONS (when mode=distillation or model-size=student)
    # ============================================================================
    distill = parser.add_argument_group('Knowledge Distillation')

    distill.add_argument(
        '--teacher-path',
        type=Path,
        help='Path to teacher model checkpoint (for distillation)'
    )

    distill.add_argument(
        '--distill-temperature',
        type=float,
        help='Distillation temperature'
    )

    distill.add_argument(
        '--distill-alpha',
        type=float,
        help='Distillation loss weight (0-1)'
    )

    # ============================================================================
    # FUSION OPTIONS (when mode=fusion)
    # ============================================================================
    fusion = parser.add_argument_group('Fusion Mode')

    fusion.add_argument(
        '--autoencoder-path',
        type=Path,
        help='Path to pre-trained autoencoder (for fusion)'
    )

    fusion.add_argument(
        '--classifier-path',
        type=Path,
        help='Path to pre-trained classifier (for fusion)'
    )

    fusion.add_argument(
        '--fusion-batch-size',
        type=int,
        help='Batch size for fusion training (typically larger)'
    )

    # ============================================================================
    # TRAINING OPTIONS
    # ============================================================================
    training = parser.add_argument_group('Training Options')

    training.add_argument(
        '--early-stopping',
        action='store_true',
        default=True,
        help='Enable early stopping (default: True)'
    )

    training.add_argument(
        '--no-early-stopping',
        action='store_false',
        dest='early_stopping',
        help='Disable early stopping'
    )

    training.add_argument(
        '--patience',
        type=int,
        help='Early stopping patience (epochs)'
    )

    training.add_argument(
        '--optimize-batch-size',
        action='store_true',
        help='Auto-tune batch size (uses PyTorch Lightning Tuner)'
    )

    training.add_argument(
        '--gradient-checkpointing',
        action='store_true',
        default=True,
        help='Enable gradient checkpointing (default: True, saves memory)'
    )

    training.add_argument(
        '--no-gradient-checkpointing',
        action='store_false',
        dest='gradient_checkpointing',
        help='Disable gradient checkpointing'
    )

    # ============================================================================
    # SLURM OPTIONS (when --submit used)
    # ============================================================================
    slurm = parser.add_argument_group('SLURM Options')

    slurm.add_argument('--account', help='SLURM account')
    slurm.add_argument('--partition', help='SLURM partition')
    slurm.add_argument('--walltime', help='Job walltime (HH:MM:SS)')
    slurm.add_argument('--memory', help='Memory per node (e.g., 64G)')
    slurm.add_argument('--cpus', type=int, help='CPUs per task')
    slurm.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    slurm.add_argument('--gpu-type', help='GPU type (e.g., v100, a100)')

    # ============================================================================
    # PATHS & OUTPUTS
    # ============================================================================
    paths = parser.add_argument_group('Paths & Outputs')

    paths.add_argument(
        '--data-path',
        type=Path,
        help='Override dataset path'
    )

    paths.add_argument(
        '--experiment-root',
        type=Path,
        default=Path('experimentruns'),
        help='Root directory for experiment outputs (default: experimentruns)'
    )

    paths.add_argument(
        '--output-dir',
        type=Path,
        help='Override output directory (default: auto-generated from config)'
    )

    # ============================================================================
    # LOGGING & TRACKING
    # ============================================================================
    logging_group = parser.add_argument_group('Logging & Tracking')

    logging_group.add_argument(
        '--mlflow-uri',
        help='MLflow tracking URI'
    )

    logging_group.add_argument(
        '--experiment-name',
        help='MLflow experiment name (default: auto-generated)'
    )

    logging_group.add_argument(
        '--tags',
        nargs='*',
        help='Custom tags for tracking (key=value pairs)'
    )

    logging_group.add_argument(
        '--notes',
        help='Experiment notes/description'
    )

    # ============================================================================
    # DEBUGGING & DEVELOPMENT
    # ============================================================================
    debug = parser.add_argument_group('Debugging & Development')

    debug.add_argument('--seed', type=int, help='Random seed for reproducibility')
    debug.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    debug.add_argument('--debug', action='store_true', help='Enable debug mode')
    debug.add_argument('--profile', action='store_true', help='Enable profiling')

    # ============================================================================
    # VALIDATION
    # ============================================================================
    validation = parser.add_argument_group('Validation')

    validation.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip pre-flight validation (advanced users only)'
    )

    validation.add_argument(
        '--force',
        action='store_true',
        help='Force execution even if validation fails'
    )


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add pipeline-specific arguments.

    Pipeline creates 3 jobs with SLURM dependencies: VGAE → GAT → DQN

    Multi-value parameters (comma-separated, one per job):
        --model vgae,gat,dqn
        --learning-type unsupervised,supervised,rl_fusion
        --mode autoencoder,curriculum,fusion

    Single-value parameters (duplicated for all jobs):
        --dataset hcrl_sa
        --model-size teacher
        --modality automotive
        --distillation no-kd
    """

    # ============================================================================
    # MULTI-VALUE PARAMETERS (one value per pipeline stage)
    # ============================================================================
    multi = parser.add_argument_group('Multi-Value Parameters (comma-separated, one per job)')

    multi.add_argument(
        '--model',
        required=True,
        help='Model architectures (e.g., vgae,gat,dqn). One per pipeline stage.'
    )

    multi.add_argument(
        '--learning-type',
        required=True,
        help='Learning types (e.g., unsupervised,supervised,rl_fusion). One per pipeline stage.'
    )

    multi.add_argument(
        '--training-strategy',
        dest='training_strategy',
        required=True,
        help='Training strategies (e.g., autoencoder,curriculum,fusion). One per pipeline stage.'
    )

    # ============================================================================
    # SINGLE-VALUE PARAMETERS (duplicated for all jobs)
    # ============================================================================
    single = parser.add_argument_group('Single-Value Parameters (applied to all jobs)')

    single.add_argument(
        '--dataset',
        required=True,
        help='Dataset name (applied to all pipeline stages)'
    )

    single.add_argument(
        '--model-size',
        choices=['teacher', 'student'],
        default='teacher',
        help='Model size for all stages (default: teacher)'
    )

    single.add_argument(
        '--modality',
        choices=['automotive', 'industrial', 'robotics'],
        default='automotive',
        help='Application domain for all stages (default: automotive)'
    )

    single.add_argument(
        '--distillation',
        choices=['with-kd', 'no-kd'],
        default='no-kd',
        help='Knowledge distillation flag for all stages (default: no-kd)'
    )

    single.add_argument(
        '--teacher_path',
        type=str,
        help='Path to teacher model checkpoint (required when --distillation with-kd)'
    )

    # ============================================================================
    # EXECUTION MODE
    # ============================================================================
    exec_mode = parser.add_argument_group('Execution Mode')

    exec_mode.add_argument(
        '--submit',
        action='store_true',
        help='Submit pipeline to SLURM'
    )

    exec_mode.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview pipeline without execution'
    )

    # ============================================================================
    # SLURM OPTIONS
    # ============================================================================
    slurm = parser.add_argument_group('SLURM Options')

    slurm.add_argument('--account', help='SLURM account')
    slurm.add_argument('--partition', help='SLURM partition')
    slurm.add_argument('--walltime', help='Job walltime (HH:MM:SS) for each stage')
    slurm.add_argument('--memory', help='Memory per node (e.g., 64G)')
    slurm.add_argument('--gpus', type=int, default=1, help='Number of GPUs per stage')


def main():
    """Main CLI entry point."""

    parser = create_parser()
    args = parser.parse_args()

    # Handle no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Route to appropriate handler
    try:
        if args.command == 'env':
            print_environment_info()
            return 0

        elif args.command == 'list-configs':
            _list_example_configs()
            return 0

        elif args.command == 'validate':
            return _validate_config(args)

        elif args.command == 'pipeline':
            return _run_pipeline(args)

        else:
            # Default: training
            return _run_training(args)

    except KeyboardInterrupt:
        logger.error("\n✗ Interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        if args.debug if hasattr(args, 'debug') else False:
            raise
        return 1


def _run_training(args):
    """Execute training workflow."""

    # Environment detection
    env = detect_environment()
    if args.verbose:
        logger.info(f"Environment: {env.env_type.value}")

    # Safety check (unless explicitly overridden or submitting to SLURM)
    # --submit is safe because it just creates/submits a job script
    if not args.local and not args.force and not args.submit:
        try:
            check_execution_safety(
                dry_run=args.dry_run,
                smoke_test=args.smoke
            )
        except RuntimeError as e:
            logger.error(str(e))
            return 1

    # Validate required arguments
    if not args.config and not args.preset:
        missing = []
        if not args.model:
            missing.append('--model')
        if not args.dataset:
            missing.append('--dataset')
        if not args.training_strategy:
            missing.append('--training-strategy')
        if not args.learning_type:
            missing.append('--learning-type')

        if missing:
            logger.error(
                f"✗ Missing required arguments: {', '.join(missing)}\n\n"
                f"Either specify:\n"
                f"  1. Explicit configuration: --model --dataset --training-strategy --learning-type\n"
                f"  2. Config file: --config path/to/config.yaml\n"
                f"  3. [Deprecated] Preset: --preset preset_name\n\n"
                f"Run 'can-train --help' for more details."
            )
            return 1

    # Show deprecation warning for presets
    if args.preset:
        logger.warning(
            "⚠️  WARNING: Presets are deprecated and will be removed.\n"
            "   Please migrate to explicit arguments or YAML configs.\n"
            "   See docs/CLI_REFERENCE.md for migration guide."
        )

    # Import CLI modules (lazy import to avoid loading torch on help/env commands)
    try:
        # Use dynamic imports to avoid triggering src/__init__.py
        import importlib.util
        from pathlib import Path

        # Import config_builder
        config_builder_path = Path(__file__).parent / 'config_builder.py'
        spec = importlib.util.spec_from_file_location("config_builder", config_builder_path)
        config_builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_builder)

        # Import executor
        executor_path = Path(__file__).parent / 'executor.py'
        spec = importlib.util.spec_from_file_location("executor", executor_path)
        executor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(executor_module)

        # Import Pydantic validators
        pydantic_validators_path = Path(__file__).parent / 'pydantic_validators.py'
        spec = importlib.util.spec_from_file_location("pydantic_validators", pydantic_validators_path)
        pydantic_validators = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pydantic_validators)

        build_config_from_buckets = config_builder.build_config_from_buckets
        create_can_graph_config = config_builder.create_can_graph_config
        ExecutionRouter = executor_module.ExecutionRouter
        determine_execution_mode = executor_module.determine_execution_mode
        validate_cli_config = pydantic_validators.validate_cli_config

    except Exception as e:
        logger.error(f"✗ Failed to load CLI modules: {e}")
        return 1

    # Run Pydantic validation first (fail early)
    try:
        validated_config = validate_cli_config(
            model=args.model,
            dataset=args.dataset,
            mode=args.training_strategy,
            model_size=args.model_size or 'teacher',
            modality=getattr(args, 'modality', 'automotive'),
            learning_type=getattr(args, 'learning_type', None),
            distillation=getattr(args, 'distillation', None),
            job_type=getattr(args, 'job_type', 'single'),
            teacher_path=getattr(args, 'teacher_path', None),
            autoencoder_path=getattr(args, 'autoencoder_path', None),
            classifier_path=getattr(args, 'classifier_path', None),
            experiment_root=args.experiment_root,
            skip_prerequisite_check=getattr(args, 'skip_validation', False),
        )
        logger.info("✅ Pydantic validation passed")
    except ValueError as e:
        logger.error(f"✗ Configuration validation failed:\n{e}")
        return 1

    # Build bucket strings from CLI args (using validated values)
    run_type_parts = [
        f"model={validated_config.model}",
        f"model_size={validated_config.model_size}",
        f"dataset={validated_config.dataset}",
        f"mode={validated_config.mode}",
        f"modality={validated_config.modality}",
        f"learning_type={validated_config.learning_type}",
        f"distillation={validated_config.distillation}",
    ]
    run_type_str = ",".join(run_type_parts)

    # Build model args bucket from CLI overrides
    model_args_parts = []
    if args.epochs:
        model_args_parts.append(f"epochs={args.epochs}")
    if args.learning_rate:
        model_args_parts.append(f"learning_rate={args.learning_rate}")
    if args.batch_size:
        model_args_parts.append(f"batch_size={args.batch_size}")
    if args.hidden_channels:
        model_args_parts.append(f"hidden_channels={args.hidden_channels}")
    if args.dropout is not None:
        model_args_parts.append(f"dropout={args.dropout}")
    if args.num_layers:
        model_args_parts.append(f"num_layers={args.num_layers}")
    if args.heads:
        model_args_parts.append(f"heads={args.heads}")
    if args.latent_dim:
        model_args_parts.append(f"latent_dim={args.latent_dim}")
    if args.weight_decay is not None:
        model_args_parts.append(f"weight_decay={args.weight_decay}")

    model_args_str = ",".join(model_args_parts) if model_args_parts else ""

    # Build SLURM args bucket
    slurm_args_parts = []
    if args.walltime:
        slurm_args_parts.append(f"walltime={args.walltime}")
    if args.memory:
        slurm_args_parts.append(f"memory={args.memory}")
    if args.cpus:
        slurm_args_parts.append(f"cpus={args.cpus}")
    if args.gpus:
        slurm_args_parts.append(f"gpus={args.gpus}")
    if args.account:
        slurm_args_parts.append(f"account={args.account}")
    if args.partition:
        slurm_args_parts.append(f"partition={args.partition}")
    if args.gpu_type:
        slurm_args_parts.append(f"gpu_type={args.gpu_type}")

    slurm_args_str = ",".join(slurm_args_parts) if slurm_args_parts else ""

    # Build configs from buckets
    try:
        configs = build_config_from_buckets(run_type_str, model_args_str, slurm_args_str)
    except Exception as e:
        logger.error(f"✗ Failed to build configuration: {e}")
        return 1

    # Determine execution mode
    mode = determine_execution_mode(
        submit=args.submit,
        local=args.local,
        dry_run=args.dry_run,
        smoke=args.smoke
    )

    # Execute each configuration
    router = ExecutionRouter()

    if len(configs) == 1:
        # Single run
        run_type, model_args, slurm_args = configs[0]
        try:
            config = create_can_graph_config(run_type, model_args, slurm_args)
        except Exception as e:
            logger.error(f"✗ Failed to create config: {e}")
            return 1

        return router.execute_single(
            config, run_type, model_args, slurm_args,
            mode=mode,
            dry_run=args.dry_run,
            skip_validation=args.skip_validation
        )
    else:
        # Multi-run sweep
        return router.execute_sweep(
            configs,
            mode=mode,
            dry_run=args.dry_run,
            skip_validation=args.skip_validation
        )


def _run_pipeline(args):
    """Execute pipeline workflow.

    Creates 3 SLURM jobs with dependencies:
        Job 1 (VGAE autoencoder) → Job 2 (GAT curriculum) → Job 3 (DQN fusion)

    Multi-value parameters are parsed as comma-separated lists.
    Single-value parameters are duplicated for all jobs.
    """

    # ========================================================================
    # Parse multi-value parameters
    # ========================================================================
    models = [m.strip() for m in args.model.split(',')]
    learning_types = [lt.strip() for lt in args.learning_type.split(',')]
    training_strategies = [m.strip() for m in args.training_strategy.split(',')]

    num_stages = len(models)

    # Validate: all multi-value parameters must have same length
    if len(learning_types) != num_stages:
        logger.error(
            f"✗ --learning-type has {len(learning_types)} values but --model has {num_stages}. "
            f"Multi-value parameters must have the same number of values."
        )
        return 1

    if len(training_strategies) != num_stages:
        logger.error(
            f"✗ --training-strategy has {len(training_strategies)} values but --model has {num_stages}. "
            f"Multi-value parameters must have the same number of values."
        )
        return 1

    # ========================================================================
    # Validate KD configuration
    # ========================================================================
    distillation = args.distillation or 'no-kd'

    if distillation == 'with-kd':
        # Check for fusion mode (not supported with KD)
        if 'fusion' in training_strategies:
            logger.error(
                "✗ Knowledge distillation is not compatible with fusion mode.\n"
                "   Fusion uses already-distilled VGAE and GAT models.\n"
                "   Either:\n"
                "   1. Remove 'fusion' from --training-strategy\n"
                "   2. Use '--distillation no-kd' (default)"
            )
            return 1

        # Note: model_size and distillation are independent dimensions
        # Any model size can use KD (though student models are typical)
        if args.model_size == 'teacher':
            logger.info(
                "ℹ️  Note: Using teacher-sized model with KD.\n"
                "   This is valid but atypical (KD is usually for student models)."
            )

        # Require teacher_path for KD
        if not args.teacher_path:
            logger.error(
                "✗ --teacher_path is required when using --distillation with-kd\n"
                "   Example: --teacher_path /path/to/teacher_model.pth"
            )
            return 1

        # Validate teacher_path exists
        from pathlib import Path
        teacher_path = Path(args.teacher_path)
        if not teacher_path.exists():
            logger.error(
                f"✗ Teacher model not found: {teacher_path}\n"
                f"   Please verify the path is correct and the file exists"
            )
            return 1

        logger.info(
            "✓ KD configuration validated\n"
            f"   Distillation: {distillation}\n"
            f"   Model size: {args.model_size}\n"
            f"   Teacher path: {args.teacher_path}"
        )

    # ========================================================================
    # Build job specifications
    # ========================================================================
    jobs = []
    for i in range(num_stages):
        job = {
            'stage': i + 1,
            'model': models[i],
            'learning_type': learning_types[i],
            'mode': training_strategies[i],  # Internal field still 'mode' for job_manager compatibility
            # Single-value parameters (same for all jobs)
            'dataset': args.dataset,
            'model_size': args.model_size,
            'modality': args.modality,
            'distillation': args.distillation,
        }
        jobs.append(job)

    # ========================================================================
    # Display pipeline summary
    # ========================================================================
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model Size: {args.model_size}")
    logger.info(f"Modality: {args.modality}")
    logger.info(f"Distillation: {args.distillation}")
    logger.info("-" * 60)

    for i, job in enumerate(jobs):
        dep_str = f" (depends on Job {i})" if i > 0 else " (no dependencies)"
        logger.info(
            f"Job {job['stage']}: {job['model']} / {job['learning_type']} / {job['mode']}{dep_str}"
        )

    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN] Pipeline preview complete - no jobs submitted")
        return 0

    # ========================================================================
    # Submit jobs to SLURM with dependencies
    # ========================================================================
    if not args.submit:
        logger.info("Use --submit to submit jobs to SLURM")
        return 0

    # Import job manager
    try:
        import importlib.util
        job_manager_path = Path(__file__).parent / 'job_manager.py'
        spec = importlib.util.spec_from_file_location("job_manager", job_manager_path)
        job_manager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job_manager)
        JobManager = job_manager.JobManager
    except Exception as e:
        logger.error(f"✗ Failed to load job manager: {e}")
        return 1

    # Submit each job with dependency on previous
    job_ids = []
    slurm_manager = JobManager()

    for i, job in enumerate(jobs):
        # Build run_type dict for this job
        run_type = {
            'model': job['model'],
            'model_size': job['model_size'],
            'dataset': job['dataset'],
            'mode': job['mode'],
            'modality': job['modality'],
            'learning_type': job['learning_type'],
            'distillation': job['distillation'],
        }

        # Build SLURM args
        slurm_args = {
            'account': args.account,
            'partition': getattr(args, 'partition', None),
            'walltime': args.walltime or '06:00:00',
            'memory': getattr(args, 'memory', None) or '64G',
            'gpus': getattr(args, 'gpus', 1),
        }

        # Add dependency if not first job (just the job ID - job_manager adds "afterok:")
        if i > 0 and len(job_ids) > 0:
            slurm_args['dependency'] = job_ids[-1]

        # Build model_args (including teacher_path if KD is enabled)
        model_args = {}
        if distillation == 'with-kd' and hasattr(args, 'teacher_path') and args.teacher_path:
            model_args['teacher_path'] = args.teacher_path

        try:
            job_id = slurm_manager.submit_single(
                config=None,  # Will be built from run_type
                run_type=run_type,
                model_args=model_args,
                slurm_args=slurm_args,
                dry_run=False
            )
            job_ids.append(job_id)
            dep_msg = f" (afterok:{job_ids[-2]})" if i > 0 else ""
            logger.info(f"✓ Submitted Job {i+1}: {job_id}{dep_msg}")
        except Exception as e:
            logger.error(f"✗ Failed to submit Job {i+1}: {e}")
            return 1

    logger.info("=" * 60)
    logger.info(f"✓ Pipeline submitted: {len(job_ids)} jobs")
    logger.info(f"  Job IDs: {', '.join(job_ids)}")
    logger.info("=" * 60)

    return 0


def _validate_config(args):
    """Validate configuration file."""
    if not args.config:
        logger.error("✗ --config required for validation")
        return 1

    logger.info(f"Validating: {args.config}")
    # TODO: Implement validation
    logger.info("✓ Configuration validation (implementation in progress)")
    return 0


def _list_example_configs():
    """List available example configurations."""
    print("\n📋 Example Configurations:\n")
    print("VGAE Autoencoder (Teacher):")
    print("  can-train --model vgae --model-size teacher --dataset hcrl_ch --mode autoencoder --submit")
    print()
    print("GAT Normal (Teacher):")
    print("  can-train --model gat --model-size teacher --dataset hcrl_ch --mode normal --submit")
    print()
    print("GAT Curriculum (Teacher):")
    print("  can-train --model gat --model-size teacher --dataset set_01 --mode curriculum --submit")
    print()
    print("DQN Fusion (Teacher):")
    print("  can-train --model dqn --model-size teacher --dataset hcrl_sa --mode fusion --submit")
    print()
    print("GAT Student with Distillation:")
    print("  can-train --model gat --model-size student --dataset hcrl_ch --mode distillation \\")
    print("    --teacher-path experimentruns/.../models/teacher.pth --submit")
    print()
    print("For more examples, see: docs/CLI_REFERENCE.md")
    print()


if __name__ == '__main__':
    sys.exit(main())
