"""
Frozen Configuration Serialization Utilities.

This module provides serialization and deserialization for CANGraphConfig objects,
enabling the "Frozen Config Pattern" where configurations are resolved once at
job submission time and saved as JSON files.

Benefits:
- Eliminates CLI → SLURM → Config resolution chain issues
- Configs are frozen at submission time (reproducibility)
- Training scripts just load pre-validated JSON (no re-resolution)
- Easy to reproduce: `python train.py --frozen-config /path/to/config.json`

Usage:
    # Save frozen config at job submission
    from src.config.frozen_config import save_frozen_config, load_frozen_config

    config = store.create_config(...)
    save_frozen_config(config, "/path/to/frozen_config.json")

    # Load in training script
    config = load_frozen_config("/path/to/frozen_config.json")
"""

import json
from pathlib import Path
from typing import Dict, Any, Union
from dataclasses import fields, is_dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Serialization
# ============================================================================

def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value to JSON-compatible format."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        # Dataclass instance - convert to dict
        result = {'_type': type(value).__name__}
        for field in fields(value):
            field_value = getattr(value, field.name)
            result[field.name] = _serialize_value(field_value)
        return result
    # Fallback: convert to string
    return str(value)


def config_to_dict(config: 'CANGraphConfig') -> Dict[str, Any]:
    """
    Convert CANGraphConfig to a JSON-serializable dictionary.

    Preserves type information for nested dataclasses to enable reconstruction.

    Args:
        config: CANGraphConfig instance

    Returns:
        Dictionary suitable for JSON serialization
    """
    result = {
        '_frozen_config_version': '1.0',
        '_frozen_at': datetime.now().isoformat(),
        '_type': type(config).__name__,
    }

    for field in fields(config):
        field_value = getattr(config, field.name)
        result[field.name] = _serialize_value(field_value)

    return result


def save_frozen_config(config: 'CANGraphConfig', path: Union[str, Path]) -> Path:
    """
    Save a CANGraphConfig as a frozen JSON file.

    Args:
        config: CANGraphConfig instance to freeze
        path: Output path for the JSON file

    Returns:
        Path to the saved config file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config_to_dict(config)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Saved frozen config to {path}")
    return path


# ============================================================================
# Deserialization
# ============================================================================

def _get_config_class(type_name: str):
    """Get the config class by name."""
    # Import config classes
    from src.config.hydra_zen_configs import (
        CANGraphConfig,
        GATConfig, StudentGATConfig,
        VGAEConfig, StudentVGAEConfig,
        DQNConfig, StudentDQNConfig,
        CANDatasetConfig,
        NormalTrainingConfig, AutoencoderTrainingConfig,
        KnowledgeDistillationConfig, StudentBaselineTrainingConfig,
        FusionTrainingConfig, CurriculumTrainingConfig, EvaluationTrainingConfig,
        TrainerConfig,
        # Nested config types
        OptimizerConfig, SchedulerConfig, MemoryOptimizationConfig,
        FusionAgentConfig,
    )

    class_map = {
        'CANGraphConfig': CANGraphConfig,
        'GATConfig': GATConfig,
        'StudentGATConfig': StudentGATConfig,
        'VGAEConfig': VGAEConfig,
        'StudentVGAEConfig': StudentVGAEConfig,
        'DQNConfig': DQNConfig,
        'StudentDQNConfig': StudentDQNConfig,
        'CANDatasetConfig': CANDatasetConfig,
        'NormalTrainingConfig': NormalTrainingConfig,
        'AutoencoderTrainingConfig': AutoencoderTrainingConfig,
        'KnowledgeDistillationConfig': KnowledgeDistillationConfig,
        'StudentBaselineTrainingConfig': StudentBaselineTrainingConfig,
        'FusionTrainingConfig': FusionTrainingConfig,
        'CurriculumTrainingConfig': CurriculumTrainingConfig,
        'EvaluationTrainingConfig': EvaluationTrainingConfig,
        'TrainerConfig': TrainerConfig,
        # Nested config types
        'OptimizerConfig': OptimizerConfig,
        'SchedulerConfig': SchedulerConfig,
        'MemoryOptimizationConfig': MemoryOptimizationConfig,
        'FusionAgentConfig': FusionAgentConfig,
    }

    return class_map.get(type_name)


def _deserialize_value(value: Any) -> Any:
    """Recursively deserialize a value from JSON format."""
    if value is None:
        return None

    # Check if it's a serialized dataclass
    if isinstance(value, dict) and '_type' in value:
        type_name = value['_type']
        config_class = _get_config_class(type_name)

        if config_class is not None:
            # Reconstruct the dataclass
            kwargs = {}
            for field in fields(config_class):
                if field.name in value:
                    kwargs[field.name] = _deserialize_value(value[field.name])
            return config_class(**kwargs)

    # Regular dict (like logging config)
    if isinstance(value, dict):
        return {k: _deserialize_value(v) for k, v in value.items()
                if not k.startswith('_')}

    # List
    if isinstance(value, list):
        return [_deserialize_value(v) for v in value]

    # Primitives
    return value


def dict_to_config(config_dict: Dict[str, Any]) -> 'CANGraphConfig':
    """
    Reconstruct a CANGraphConfig from a serialized dictionary.

    Args:
        config_dict: Dictionary from config_to_dict or JSON load

    Returns:
        Reconstructed CANGraphConfig instance
    """
    from src.config.hydra_zen_configs import CANGraphConfig

    # Filter out metadata fields
    config_data = {k: v for k, v in config_dict.items()
                   if not k.startswith('_')}

    # Reconstruct nested dataclasses
    # Skip fields with init=False (like experiment_name) - they're computed in __post_init__
    kwargs = {}
    for field_info in fields(CANGraphConfig):
        if field_info.name in config_data and field_info.init:
            kwargs[field_info.name] = _deserialize_value(config_data[field_info.name])

    return CANGraphConfig(**kwargs)


def load_frozen_config(path: Union[str, Path]) -> 'CANGraphConfig':
    """
    Load a CANGraphConfig from a frozen JSON file.

    Args:
        path: Path to the frozen config JSON file

    Returns:
        Reconstructed CANGraphConfig instance

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config format is invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Frozen config not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # Validate version
    version = config_dict.get('_frozen_config_version', '0.0')
    if not version.startswith('1.'):
        logger.warning(f"Frozen config version {version} may not be compatible")

    frozen_at = config_dict.get('_frozen_at', 'unknown')
    logger.info(f"📖 Loading frozen config from {path} (frozen at: {frozen_at})")

    return dict_to_config(config_dict)


# ============================================================================
# Utility Functions
# ============================================================================

def get_frozen_config_path(experiment_dir: Union[str, Path]) -> Path:
    """
    Get the canonical path for a frozen config within an experiment directory.

    Args:
        experiment_dir: Experiment directory path

    Returns:
        Path to frozen_config.json
    """
    return Path(experiment_dir) / "frozen_config.json"


def validate_frozen_config(path: Union[str, Path]) -> bool:
    """
    Validate that a frozen config file exists and is loadable.

    Args:
        path: Path to the frozen config file

    Returns:
        True if valid, False otherwise
    """
    try:
        load_frozen_config(path)  # Raises exception if invalid
        return True
    except Exception as e:
        logger.error(f"Invalid frozen config {path}: {e}")
        return False
