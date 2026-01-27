"""
Pre-flight validation for CAN-Graph training configurations.

Validates configurations before training to catch issues early:
- Dataset paths exist and are accessible
- Required artifacts exist for modes that need them (fusion, curriculum, distillation)
- SLURM resources are reasonable
- Output directories can be created
"""

from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Validation Classes
# ============================================================================

class ValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ValidationWarning:
    """Represents a non-fatal validation warning."""
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return f"⚠️  {self.message}"


class ValidationResult:
    """Result of validation check."""
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[ValidationWarning] = []
        self.checks_passed: List[str] = []

    def add_error(self, message: str):
        """Add a fatal error."""
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add a non-fatal warning."""
        self.warnings.append(ValidationWarning(message))

    def add_check(self, message: str):
        """Record a passed check."""
        self.checks_passed.append(message)

    def is_valid(self) -> bool:
        """Returns True if no errors (warnings are OK)."""
        return len(self.errors) == 0

    def format_report(self) -> str:
        """Format validation report for display."""
        lines = []
        lines.append("\n" + "="*70)
        lines.append("PRE-FLIGHT VALIDATION REPORT")
        lines.append("="*70)

        if self.checks_passed:
            lines.append("\n✓ Passed Checks:")
            for check in self.checks_passed:
                lines.append(f"  ✓ {check}")

        if self.warnings:
            lines.append(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  {warning}")

        if self.errors:
            lines.append(f"\n✗ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  ✗ {error}")
            lines.append("\n" + "="*70)
            lines.append("VALIDATION FAILED - Cannot proceed with training")
            lines.append("="*70)
        else:
            lines.append("\n" + "="*70)
            lines.append("✅ VALIDATION PASSED")
            lines.append("="*70)

        return "\n".join(lines)


# ============================================================================
# Validators
# ============================================================================

def validate_dataset_path(config: 'CANGraphConfig', result: ValidationResult):
    """Validate dataset path exists and is accessible."""
    data_path = Path(config.dataset.data_path)

    if not data_path.exists():
        result.add_error(
            f"Dataset path does not exist: {data_path}\n"
            f"    Expected for dataset '{config.dataset.name}'\n"
            f"    Create the dataset or fix the path"
        )
    elif not data_path.is_dir():
        result.add_error(f"Dataset path is not a directory: {data_path}")
    else:
        # Check if directory is readable
        try:
            list(data_path.iterdir())
            result.add_check(f"Dataset path exists: {data_path}")
        except PermissionError:
            result.add_error(f"Dataset path is not readable: {data_path}")


def validate_required_artifacts(config: 'CANGraphConfig', result: ValidationResult):
    """Validate required pre-trained artifacts exist for modes that need them."""
    training_mode = config.training.mode

    # Get required artifacts from config
    required_artifacts = config.required_artifacts()

    if not required_artifacts:
        result.add_check(f"No pre-trained artifacts required for mode '{training_mode}'")
        return

    # Check each required artifact
    for artifact_name, artifact_path in required_artifacts.items():
        if not artifact_path.exists():
            result.add_error(
                f"Required artifact missing for '{training_mode}' mode:\n"
                f"    {artifact_name}: {artifact_path}\n"
                f"    Train the prerequisite model first"
            )
        else:
            result.add_check(f"Required artifact exists: {artifact_name} at {artifact_path}")


def validate_output_directory(config: 'CANGraphConfig', result: ValidationResult):
    """Validate output directory can be created."""
    output_dir = config.canonical_experiment_dir()

    # Check if parent exists and is writable
    if output_dir.exists():
        result.add_warning(
            f"Output directory already exists: {output_dir}\n"
            f"    Existing files may be overwritten"
        )
    else:
        # Check if parent is writable
        parent = output_dir.parent
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
                result.add_check(f"Created parent directory: {parent}")
            except Exception as e:
                result.add_error(f"Cannot create parent directory {parent}: {e}")
        elif not parent.is_dir():
            result.add_error(f"Parent path exists but is not a directory: {parent}")
        else:
            result.add_check(f"Output directory can be created at: {output_dir}")


def validate_slurm_resources(slurm_args: Dict, result: ValidationResult):
    """Validate SLURM resource requests are reasonable."""
    # Parse walltime
    walltime_str = slurm_args.get('walltime', '06:00:00')
    try:
        # Parse HH:MM:SS format
        parts = walltime_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            total_hours = hours + minutes/60 + seconds/3600

            if total_hours > 48:
                result.add_warning(
                    f"Walltime is very long: {walltime_str} ({total_hours:.1f} hours)\n"
                    f"    Consider if you really need more than 48 hours"
                )
            elif total_hours < 0.5:
                result.add_warning(
                    f"Walltime is very short: {walltime_str} ({total_hours:.1f} hours)\n"
                    f"    Training may not complete in time"
                )
            else:
                result.add_check(f"Walltime is reasonable: {walltime_str}")
        else:
            result.add_warning(f"Walltime format unclear: {walltime_str}")
    except Exception as e:
        result.add_warning(f"Could not parse walltime '{walltime_str}': {e}")

    # Check memory
    memory_str = slurm_args.get('memory', '64G')
    try:
        # Parse memory (e.g., "64G", "128G", "256G")
        if memory_str.endswith('G'):
            memory_gb = int(memory_str[:-1])
            if memory_gb > 512:
                result.add_warning(
                    f"Memory request is very high: {memory_str}\n"
                    f"    May be difficult to schedule"
                )
            elif memory_gb < 16:
                result.add_warning(
                    f"Memory request is low: {memory_str}\n"
                    f"    May cause OOM errors for large datasets"
                )
            else:
                result.add_check(f"Memory request is reasonable: {memory_str}")
        else:
            result.add_warning(f"Memory format unclear: {memory_str}")
    except Exception as e:
        result.add_warning(f"Could not parse memory '{memory_str}': {e}")

    # Check GPU count
    gpus = slurm_args.get('gpus', 1)
    if isinstance(gpus, int):
        if gpus > 4:
            result.add_warning(
                f"Requesting many GPUs: {gpus}\n"
                f"    May be difficult to schedule, consider if you need this many"
            )
        elif gpus == 0:
            result.add_warning(
                "Requesting 0 GPUs\n"
                "    Training will be very slow on CPU"
            )
        else:
            result.add_check(f"GPU count is reasonable: {gpus}")


def validate_mode_specific(config: 'CANGraphConfig', result: ValidationResult):
    """Mode-specific validation checks.

    NOTE: This is now the SINGLE source of truth for mode-specific validation.
    Validation logic was consolidated here from hydra_zen_configs.py to avoid
    duplicate validation across modules.
    """
    mode = config.training.mode

    # Check knowledge distillation requirements
    # (both legacy mode="knowledge_distillation" and new toggle use_knowledge_distillation)
    use_kd = getattr(config.training, "use_knowledge_distillation", False)
    is_legacy_kd_mode = mode == "knowledge_distillation"

    if use_kd or is_legacy_kd_mode:
        # Reject fusion + KD (not supported - DQN uses already-distilled models)
        if mode == "fusion":
            result.add_error(
                "Knowledge distillation is not supported for fusion mode.\n"
                "    Reason: The DQN agent uses already-distilled VGAE and GAT models"
            )

        teacher_path = getattr(config.training, 'teacher_model_path', None)
        if not teacher_path:
            result.add_error(
                "Knowledge distillation requires 'teacher_model_path'\n"
                "    Specify the path to a trained teacher model (.pth or .ckpt file)"
            )
        elif not Path(teacher_path).exists():
            result.add_error(
                f"Teacher model not found at: {teacher_path}\n"
                "    Please ensure the teacher model exists under experiment_runs"
            )

    if mode == "fusion":
        # Check fusion-specific configs
        if hasattr(config.training, 'autoencoder_path'):
            ae_path = config.training.autoencoder_path
            if ae_path and not Path(ae_path).exists():
                result.add_error(f"Fusion autoencoder path does not exist: {ae_path}")

        if hasattr(config.training, 'classifier_path'):
            clf_path = config.training.classifier_path
            if clf_path and not Path(clf_path).exists():
                result.add_error(f"Fusion classifier path does not exist: {clf_path}")

    if mode == "curriculum":
        # Check VGAE model for curriculum
        if hasattr(config.training, 'vgae_model_path'):
            vgae_path = config.training.vgae_model_path
            if vgae_path and not Path(vgae_path).exists():
                result.add_warning(
                    f"Curriculum VGAE path does not exist yet: {vgae_path}\n"
                    f"    It will be auto-discovered if available, or you may need to train VGAE first"
                )


def validate_config_consistency(config: 'CANGraphConfig', result: ValidationResult):
    """Validate config consistency (precision, modality, etc.).

    NOTE: This validation was moved from hydra_zen_configs.py to consolidate
    all validation in one place.
    """
    # Check modality is set
    if getattr(config, 'modality', None) is None:
        result.add_error("Modality must be set (e.g., 'automotive')")

    # Check precision compatibility
    training_precision = getattr(config.training, 'precision', '32-true')
    trainer_precision = getattr(config.trainer, 'precision', '32-true')

    if training_precision == "16-mixed" and trainer_precision != "16-mixed":
        result.add_error(
            "Precision mismatch: training.precision='16-mixed' but trainer.precision='{trainer_precision}'\n"
            "    Both should be '16-mixed' when using mixed precision training"
        )


# ============================================================================
# Main Validation Function
# ============================================================================

def validate_config(config: 'CANGraphConfig', slurm_args: Optional[Dict] = None,
                   skip_artifact_check: bool = False) -> ValidationResult:
    """
    Run all validation checks on a configuration.

    This is the SINGLE source of truth for pre-flight validation.
    All validation logic is consolidated here to avoid duplication across modules.

    Validation responsibilities:
        - Dataset path existence and accessibility
        - Required artifacts for modes (fusion, curriculum, distillation)
        - Output directory creation
        - SLURM resource reasonableness
        - Mode-specific checks (KD requirements, fusion artifacts)
        - Config consistency (precision, modality)

    Args:
        config: CANGraphConfig to validate
        slurm_args: Optional SLURM arguments dictionary
        skip_artifact_check: If True, skip checking for required artifacts
                            (useful for dry-run mode)

    Returns:
        ValidationResult with errors, warnings, and passed checks

    Example:
        >>> result = validate_config(config, slurm_args)
        >>> if not result.is_valid():
        ...     print(result.format_report())
        ...     raise ValidationError("Configuration invalid")
    """
    result = ValidationResult()

    logger.info("Running pre-flight validation...")

    # 1. Validate dataset path
    validate_dataset_path(config, result)

    # 2. Validate required artifacts (unless skipped)
    if not skip_artifact_check:
        validate_required_artifacts(config, result)

    # 3. Validate output directory
    validate_output_directory(config, result)

    # 4. Validate SLURM resources (if provided)
    if slurm_args:
        validate_slurm_resources(slurm_args, result)

    # 5. Mode-specific validation
    validate_mode_specific(config, result)

    # 6. Config consistency validation (precision, modality)
    validate_config_consistency(config, result)

    # Log summary
    if result.is_valid():
        logger.info(f"✅ Validation passed ({len(result.checks_passed)} checks)")
        if result.warnings:
            logger.info(f"⚠️  {len(result.warnings)} warnings (non-fatal)")
    else:
        logger.error(f"✗ Validation failed with {len(result.errors)} errors")

    return result
