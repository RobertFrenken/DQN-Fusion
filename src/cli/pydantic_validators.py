"""
Pydantic validators for CAN-Graph CLI configuration.

Implements design principles from DESIGN_PRINCIPLES.md:
- PRINCIPLE 1: All folder structure parameters must be explicit in CLI
- PRINCIPLE 8: Dependency schema with fail-early prerequisite checking

Enforces P→Q validation rules from parameters/required_cli.yaml and
parameters/dependencies.yaml.
"""

from pathlib import Path
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field
import logging
import yaml

logger = logging.getLogger(__name__)


# ============================================================================
# Core Configuration Model
# ============================================================================

class CANGraphCLIConfig(BaseModel):
    """
    Pydantic model for CAN-Graph CLI configuration.

    Mirrors the parameter bible in parameters/required_cli.yaml.
    All folder structure parameters are REQUIRED and EXPLICIT.

    Folder structure:
        {modality}/{dataset}/{model_size}/{learning_type}/{model}/{distillation}/{mode}/
    """

    # Level 0: Job Type
    job_type: Literal["single", "pipeline"] = Field(
        default="single",
        description="Job execution type - single job or multi-job pipeline"
    )

    # Level 1: Modality
    modality: Literal["automotive", "industrial", "robotics"] = Field(
        ...,
        description="Application domain (REQUIRED - DESIGN PRINCIPLE 1)"
    )

    # Level 2: Dataset
    dataset: Literal["hcrl_ch", "hcrl_sa", "set_01", "set_02", "set_03", "set_04"] = Field(
        ...,
        description="Dataset name"
    )

    # Level 3: Model Size
    model_size: Literal["teacher", "student"] = Field(
        ...,
        description="Model capacity - teacher (full) or student (compressed) (REQUIRED - DESIGN PRINCIPLE 1)"
    )

    # Level 4: Learning Type
    learning_type: Literal["supervised", "unsupervised", "semi_supervised", "rl_fusion"] = Field(
        ...,
        description="ML learning paradigm"
    )

    # Level 5: Model Architecture
    model: Literal["vgae", "gat", "dqn", "gcn", "gnn", "graphsage"] = Field(
        ...,
        description="Model architecture"
    )

    # Level 6: Distillation
    distillation: Literal["with-kd", "no-kd"] = Field(
        default="no-kd",
        description="Knowledge distillation enabled"
    )

    # Level 7: Training Mode
    mode: Literal["normal", "curriculum", "autoencoder", "fusion", "distillation", "evaluation"] = Field(
        ...,
        description="Training strategy/mode"
    )

    # Optional paths for prerequisites (will be auto-discovered if not provided)
    teacher_path: Optional[Path] = Field(
        default=None,
        description="Path to teacher model checkpoint (for distillation mode)"
    )

    autoencoder_path: Optional[Path] = Field(
        default=None,
        description="Path to pretrained autoencoder (for fusion/curriculum modes)"
    )

    classifier_path: Optional[Path] = Field(
        default=None,
        description="Path to pretrained classifier (for fusion mode)"
    )

    # Skip prerequisite check flag (for SLURM job submission where prerequisites
    # will be trained in earlier jobs within the same pipeline)
    skip_prerequisite_check: bool = Field(
        default=False,
        description="Skip prerequisite file existence checks (for batch job submission)"
    )

    # Experiment root for path resolution
    experiment_root: Path = Field(
        default=Path("experimentruns"),
        description="Root directory for experiment outputs"
    )

    # ========================================================================
    # VALIDATION RULES (P→Q Logic)
    # ========================================================================

    @field_validator("learning_type")
    @classmethod
    def validate_learning_type_model_consistency(cls, v, info):
        """Validate that learning_type is consistent with model architecture."""
        model = info.data.get("model")

        if not model:
            return v

        # P→Q: model=vgae → learning_type=unsupervised
        if model == "vgae" and v != "unsupervised":
            raise ValueError(
                f"VGAE model requires learning_type='unsupervised', got '{v}'\n"
                f"    Reason: VGAE is an autoencoder (unsupervised architecture)"
            )

        # P→Q: model=dqn → learning_type=rl_fusion
        if model == "dqn" and v != "rl_fusion":
            raise ValueError(
                f"DQN model requires learning_type='rl_fusion', got '{v}'\n"
                f"    Reason: DQN is a reinforcement learning agent for fusion"
            )

        # P→Q: model in [gat, gcn, gnn, graphsage] → learning_type=supervised
        supervised_models = ["gat", "gcn", "gnn", "graphsage"]
        if model in supervised_models and v != "supervised":
            raise ValueError(
                f"{model.upper()} model requires learning_type='supervised', got '{v}'\n"
                f"    Reason: {model.upper()} is a supervised classifier architecture"
            )

        return v

    @field_validator("mode")
    @classmethod
    def validate_mode_learning_type_consistency(cls, v, info):
        """Validate that mode is consistent with learning_type."""
        learning_type = info.data.get("learning_type")

        if not learning_type:
            return v

        # P→Q: learning_type=unsupervised → mode=autoencoder
        if learning_type == "unsupervised" and v != "autoencoder":
            raise ValueError(
                f"Unsupervised learning requires mode='autoencoder', got '{v}'\n"
                f"    Reason: Unsupervised training uses autoencoder reconstruction"
            )

        # P→Q: learning_type=rl_fusion → mode=fusion
        if learning_type == "rl_fusion" and v != "fusion":
            raise ValueError(
                f"RL fusion learning requires mode='fusion', got '{v}'\n"
                f"    Reason: RL agent learns to fuse pretrained models"
            )

        # P→Q: learning_type=supervised → mode in [normal, curriculum, distillation]
        if learning_type == "supervised" and v not in ["normal", "curriculum", "distillation"]:
            raise ValueError(
                f"Supervised learning requires mode in ['normal', 'curriculum', 'distillation'], got '{v}'\n"
                f"    Reason: These are the valid supervised training strategies"
            )

        return v

    @field_validator("distillation")
    @classmethod
    def validate_distillation_consistency(cls, v, info):
        """Validate distillation is consistent with mode and model_size."""
        mode = info.data.get("mode")
        model_size = info.data.get("model_size")

        # Note: model_size and distillation are INDEPENDENT dimensions
        # Any model_size can use with-kd (though student is typical)
        # KD can combine with modes: autoencoder, curriculum, normal, distillation
        if v == "with-kd":
            # Only validate that fusion mode doesn't use KD
            if mode == "fusion":
                raise ValueError(
                    f"Knowledge distillation (with-kd) cannot be used with fusion mode.\n"
                    f"    Reason: Fusion uses already-distilled models"
                )

        # P→Q: mode=distillation → distillation=with-kd
        if mode == "distillation" and v != "with-kd":
            raise ValueError(
                f"Distillation mode requires distillation='with-kd', got '{v}'\n"
                f"    Reason: Distillation mode means knowledge distillation is enabled"
            )

        return v

    @model_validator(mode='after')
    def validate_prerequisites(self) -> 'CANGraphCLIConfig':
        """
        Validate that required prerequisites exist for the training configuration.

        Implements fail-early checking from parameters/dependencies.yaml.
        """
        # Skip prerequisite check if explicitly requested (for batch job submission)
        if self.skip_prerequisite_check:
            logger.info("⚠️  Skipping prerequisite checks (--skip-validation flag)")
            return self

        if self.job_type == "pipeline":
            # Pipeline mode handles dependencies automatically
            logger.info("Pipeline mode: dependency checking will be handled by pipeline manager")
            return self

        # Single job mode: check prerequisites NOW (fail early)
        errors = []

        # Check mode-specific prerequisites
        if self.mode == "curriculum":
            # Curriculum requires VGAE checkpoint
            if not self.autoencoder_path:
                # Auto-discover VGAE checkpoint
                vgae_path = self._find_prerequisite_checkpoint(
                    model="vgae",
                    learning_type="unsupervised",
                    mode="autoencoder"
                )
                if vgae_path:
                    self.autoencoder_path = vgae_path
                    logger.info(f"Auto-discovered VGAE checkpoint: {vgae_path}")
                else:
                    errors.append(
                        f"Curriculum mode requires pretrained VGAE autoencoder\n"
                        f"    Expected path: {self.modality}/{self.dataset}/{self.model_size}/unsupervised/vgae/no-kd/autoencoder\n"
                        f"    Reason: VGAE needed for hard sample mining\n"
                        f"    Solution: Train VGAE first or provide --autoencoder-path"
                    )

        if self.mode == "fusion":
            # Fusion requires autoencoder AND classifier
            if not self.autoencoder_path:
                vgae_path = self._find_prerequisite_checkpoint(
                    model="vgae",
                    learning_type="unsupervised",
                    mode="autoencoder"
                )
                if vgae_path:
                    self.autoencoder_path = vgae_path
                    logger.info(f"Auto-discovered VGAE checkpoint: {vgae_path}")
                else:
                    errors.append(
                        f"Fusion mode requires pretrained autoencoder\n"
                        f"    Expected path: {self.modality}/{self.dataset}/{self.model_size}/unsupervised/vgae/no-kd/autoencoder\n"
                        f"    Reason: Autoencoder provides feature extraction for fusion\n"
                        f"    Solution: Train VGAE first or provide --autoencoder-path"
                    )

            if not self.classifier_path:
                # Try to find any supervised classifier
                classifier_path = self._find_prerequisite_checkpoint(
                    model=None,  # Any supervised model
                    learning_type="supervised",
                    mode=None,  # Any supervised mode
                    flexible=True
                )
                if classifier_path:
                    self.classifier_path = classifier_path
                    logger.info(f"Auto-discovered classifier checkpoint: {classifier_path}")
                else:
                    errors.append(
                        f"Fusion mode requires pretrained classifier\n"
                        f"    Expected path: {self.modality}/{self.dataset}/{self.model_size}/supervised/{{model}}/no-kd/{{mode}}\n"
                        f"    Reason: Classifier provides decision making for fusion\n"
                        f"    Solution: Train GAT/GCN/GNN first or provide --classifier-path"
                    )

        if self.mode == "distillation":
            # Distillation requires teacher model
            if not self.teacher_path:
                # Auto-discover teacher checkpoint
                teacher_path = self._find_prerequisite_checkpoint(
                    model=self.model,
                    learning_type=self.learning_type,
                    mode="normal",  # Teachers are usually trained in normal mode
                    model_size="teacher"
                )
                if teacher_path:
                    self.teacher_path = teacher_path
                    logger.info(f"Auto-discovered teacher checkpoint: {teacher_path}")
                else:
                    errors.append(
                        f"Distillation mode requires pretrained teacher model\n"
                        f"    Expected path: {self.modality}/{self.dataset}/teacher/{self.learning_type}/{self.model}/no-kd/normal\n"
                        f"    Reason: Student learns from teacher via knowledge distillation\n"
                        f"    Solution: Train teacher model first or provide --teacher-path"
                    )

        if errors:
            error_msg = "\n\n".join([f"✗ {e}" for e in errors])
            raise ValueError(
                f"\n{'='*70}\n"
                f"PREREQUISITE CHECK FAILED\n"
                f"{'='*70}\n\n"
                f"{error_msg}\n\n"
                f"{'='*70}\n"
                f"DESIGN PRINCIPLE 8: Fail-early dependency checking\n"
                f"See parameters/dependencies.yaml for prerequisite schema\n"
                f"{'='*70}\n"
            )

        return self

    def _find_prerequisite_checkpoint(
        self,
        model: Optional[str] = None,
        learning_type: Optional[str] = None,
        mode: Optional[str] = None,
        model_size: Optional[str] = None,
        flexible: bool = False
    ) -> Optional[Path]:
        """
        Find a prerequisite checkpoint using flexible path matching.

        Args:
            model: Model type (or None for wildcard)
            learning_type: Learning type (or None for wildcard)
            mode: Training mode (or None for wildcard)
            model_size: Model size (or None to use current config)
            flexible: If True, use glob matching for wildcards

        Returns:
            Path to checkpoint if found, None otherwise
        """
        # Use current model_size if not specified
        if model_size is None:
            model_size = self.model_size

        # Build expected path pattern
        path_parts = [
            self.experiment_root,
            self.modality,
            self.dataset,
            model_size,
            learning_type or "*",
            model or "*",
            "no-kd",  # Prerequisite models don't use distillation
            mode or "*"
        ]

        if flexible:
            # Use glob to find any matching checkpoint
            from glob import glob
            pattern = str(Path(*path_parts) / "checkpoints" / "best.ckpt")
            matches = glob(pattern)
            if matches:
                return Path(matches[0])
        else:
            # Check exact path
            checkpoint_dir = Path(*path_parts) / "checkpoints"
            checkpoint_path = checkpoint_dir / "best.ckpt"
            if checkpoint_path.exists():
                return checkpoint_path

        return None

    @computed_field
    @property
    def canonical_save_path(self) -> Path:
        """
        Compute canonical save path based on folder structure.

        Target folder structure (must match canonical_experiment_dir in hydra_zen_configs.py):
            {modality}/{dataset}/{learning_type}/{model}/{model_size}/{distillation}/{mode}/

        Note: distillation values are mapped for path consistency:
            - "no-kd" -> "no_distillation"
            - "with-kd" -> "distilled"
        """
        # Map CLI distillation values to path-friendly format
        distillation_path = "distilled" if self.distillation == "with-kd" else "no_distillation"

        return (
            self.experiment_root /
            self.modality /
            self.dataset /
            self.learning_type /
            self.model /
            self.model_size /
            distillation_path /
            self.mode
        )

    @computed_field
    @property
    def sample_strategy(self) -> Literal["all_samples", "normal_only"]:
        """
        Determine sample strategy based on mode.

        - autoencoder: normal_only (no attacks)
        - all others: all_samples (normal + attacks)
        """
        if self.mode == "autoencoder":
            return "normal_only"
        return "all_samples"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for config builder."""
        return self.model_dump(exclude_none=True)

    def validate_all(self) -> List[str]:
        """
        Run all validation checks and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        try:
            # Pydantic validators run automatically on construction
            # This method is for additional runtime checks

            # Check that save path can be created
            save_path = self.canonical_save_path
            if not save_path.parent.exists():
                try:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create save path parent: {e}")

        except Exception as e:
            errors.append(str(e))

        return errors


# ============================================================================
# Utility Functions
# ============================================================================

def load_parameter_bible() -> Dict[str, Any]:
    """Load parameter bible from YAML file."""
    bible_path = Path(__file__).parent.parent.parent / "parameters" / "required_cli.yaml"

    if not bible_path.exists():
        logger.warning(f"Parameter bible not found at {bible_path}")
        return {}

    with open(bible_path, 'r') as f:
        return yaml.safe_load(f)


def load_dependency_schema() -> Dict[str, Any]:
    """Load dependency schema from YAML file."""
    schema_path = Path(__file__).parent.parent.parent / "parameters" / "dependencies.yaml"

    if not schema_path.exists():
        logger.warning(f"Dependency schema not found at {schema_path}")
        return {}

    with open(schema_path, 'r') as f:
        return yaml.safe_load(f)


def validate_cli_config(
    model: str,
    dataset: str,
    mode: str,
    model_size: str,
    modality: str,
    learning_type: Optional[str] = None,
    distillation: Optional[str] = None,
    job_type: str = "single",
    **kwargs
) -> CANGraphCLIConfig:
    """
    Validate CLI configuration and return Pydantic model.

    This is the main entry point for CLI validation.

    Args:
        model: Model architecture
        dataset: Dataset name
        mode: Training mode
        model_size: Model size (teacher/student)
        modality: Application domain
        learning_type: Learning paradigm (auto-computed if not provided)
        distillation: Distillation flag (auto-computed if not provided)
        job_type: Job type (single/pipeline)
        **kwargs: Additional arguments (teacher_path, autoencoder_path, etc.)

    Returns:
        Validated CANGraphCLIConfig

    Raises:
        ValueError: If validation fails

    Example:
        >>> config = validate_cli_config(
        ...     model="gat",
        ...     dataset="hcrl_ch",
        ...     mode="normal",
        ...     learning_type="supervised",
        ...     distillation="no-kd"
        ... )
    """
    # Auto-compute learning_type if not provided (for backward compatibility)
    if learning_type is None:
        if mode == "autoencoder":
            learning_type = "unsupervised"
        elif mode == "fusion":
            learning_type = "rl_fusion"
        elif mode in ["normal", "curriculum", "distillation"]:
            learning_type = "supervised"
        else:
            raise ValueError(
                f"Cannot auto-compute learning_type for mode='{mode}'\n"
                f"    Please provide --learning-type explicitly"
            )
        logger.warning(
            f"⚠️  Auto-computed learning_type='{learning_type}' from mode='{mode}'\n"
            f"    This violates DESIGN PRINCIPLE 1 (explicit parameters)\n"
            f"    Please provide --learning-type explicitly in future"
        )

    # Auto-compute distillation if not provided (for backward compatibility)
    if distillation is None:
        if mode == "distillation":
            distillation = "with-kd"
        else:
            distillation = "no-kd"
        logger.warning(
            f"⚠️  Auto-computed distillation='{distillation}' from mode='{mode}'\n"
            f"    This violates DESIGN PRINCIPLE 1 (explicit parameters)\n"
            f"    Please provide --distillation explicitly in future"
        )

    # Create and validate config
    try:
        config = CANGraphCLIConfig(
            job_type=job_type,
            modality=modality,
            dataset=dataset,
            model_size=model_size,
            learning_type=learning_type,
            model=model,
            distillation=distillation,
            mode=mode,
            **kwargs
        )

        logger.info(f"✅ Configuration validated successfully")
        logger.info(f"    Save path: {config.canonical_save_path}")
        logger.info(f"    Sample strategy: {config.sample_strategy}")

        return config

    except ValueError as e:
        logger.error(f"❌ Configuration validation failed:\n{e}")
        raise
