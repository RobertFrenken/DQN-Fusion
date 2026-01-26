"""
Configuration builder for bucket-based CLI arguments.

Parses bucket syntax (key=value,key=[val1,val2]) and builds CANGraphConfig objects.
Supports parameter sweeps and multi-run expansion.
"""

import re
from typing import Dict, List, Union, Any, Tuple
from pathlib import Path
from itertools import product
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Default Values
# ============================================================================

DEFAULT_MODEL_ARGS = {
    'epochs': 50,
    'learning_rate': 0.003,
    'batch_size': 64,
    'hidden_channels': 128,
    'dropout': 0.2,
    'weight_decay': 0.0,
    'early_stopping': True,
    'patience': 10,
    'gradient_checkpointing': True,
    'optimize_batch_size': False,

    # GAT-specific
    'num_layers': 3,
    'heads': 4,

    # VGAE-specific
    'latent_dim': 16,

    # DQN-specific
    'gamma': 0.95,
    'replay_buffer_size': 10000,
}

DEFAULT_SLURM_ARGS = {
    'walltime': '06:00:00',
    'memory': '64G',
    'cpus': 16,
    'gpus': 1,
    'gpu_type': 'v100',
    'account': 'PAS3209',
    'partition': 'gpu',
}


# ============================================================================
# Bucket Parsing
# ============================================================================

def parse_bucket(bucket_str: str) -> Dict[str, Union[str, List[str]]]:
    """
    Parse bucket string with comma-separated key=value pairs.

    Supports:
    - Simple values: key=value
    - List values: key=[val1,val2,val3]
    - Mixed: key1=val1,key2=[val2a,val2b],key3=val3

    Args:
        bucket_str: Comma-separated key=value string

    Returns:
        Dictionary with parsed key-value pairs (values can be strings or lists)

    Examples:
        >>> parse_bucket("model=gat,dataset=hcrl_ch")
        {'model': 'gat', 'dataset': 'hcrl_ch'}

        >>> parse_bucket("model=gat,dataset=[hcrl_ch,hcrl_sa]")
        {'model': 'gat', 'dataset': ['hcrl_ch', 'hcrl_sa']}
    """
    result = {}

    if not bucket_str:
        return result

    # Split by commas, but not commas inside brackets
    # Pattern: split on commas not inside []
    parts = re.split(r',(?![^\[]*\])', bucket_str)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if '=' not in part:
            raise ValueError(f"Invalid bucket syntax: '{part}' (expected key=value)")

        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Check if value is a list: [val1,val2,val3]
        if value.startswith('[') and value.endswith(']'):
            # Parse list
            list_content = value[1:-1]  # Remove brackets
            if list_content:
                values = [v.strip() for v in list_content.split(',')]
                result[key] = values
            else:
                result[key] = []
        else:
            result[key] = value

    return result


def expand_sweep(config_dict: Dict[str, Union[str, List[str]]]) -> List[Dict[str, str]]:
    """
    Expand dictionary with list values into Cartesian product of all combinations.

    Args:
        config_dict: Dictionary with string or list values

    Returns:
        List of dictionaries, each with only string values (all combinations)

    Examples:
        >>> expand_sweep({'model': 'gat', 'dataset': ['hcrl_ch', 'hcrl_sa']})
        [
            {'model': 'gat', 'dataset': 'hcrl_ch'},
            {'model': 'gat', 'dataset': 'hcrl_sa'}
        ]

        >>> expand_sweep({'lr': ['0.001', '0.003'], 'batch': ['64', '128']})
        [
            {'lr': '0.001', 'batch': '64'},
            {'lr': '0.001', 'batch': '128'},
            {'lr': '0.003', 'batch': '64'},
            {'lr': '0.003', 'batch': '128'}
        ]
    """
    # Separate keys with lists from keys with single values
    list_keys = []
    list_values = []
    single_items = {}

    for key, value in config_dict.items():
        if isinstance(value, list):
            list_keys.append(key)
            list_values.append(value)
        else:
            single_items[key] = value

    # If no lists, return single config
    if not list_keys:
        return [config_dict.copy()]

    # Generate Cartesian product of all list values
    combinations = list(product(*list_values))

    # Create a config dict for each combination
    results = []
    for combo in combinations:
        config = single_items.copy()
        for key, value in zip(list_keys, combo):
            config[key] = value
        results.append(config)

    return results


# ============================================================================
# Configuration Building
# ============================================================================

def validate_run_type(run_type: Dict[str, str]) -> None:
    """
    Validate that run_type has all required fields.

    Args:
        run_type: Parsed run_type bucket (should have no lists at this point)

    Raises:
        ValueError: If required fields are missing
    """
    # DESIGN PRINCIPLE 1: All folder structure parameters must be explicit
    required_fields = [
        'model',
        'model_size',
        'dataset',
        'mode',
        'modality',
        'learning_type',
        'distillation'
    ]
    missing = [field for field in required_fields if field not in run_type]

    if missing:
        raise ValueError(
            f"Missing required fields in --run-type: {', '.join(missing)}\n"
            f"Required: {', '.join(required_fields)}\n"
            f"Provided: {', '.join(run_type.keys())}\n\n"
            f"DESIGN PRINCIPLE 1: All folder structure parameters must be explicit\n"
            f"See parameters/required_cli.yaml for parameter bible"
        )

    # Validate choices
    valid_models = ['gat', 'vgae', 'dqn', 'gcn', 'gnn', 'graphsage']
    if run_type['model'] not in valid_models:
        raise ValueError(
            f"Invalid model: {run_type['model']}\n"
            f"Valid options: {', '.join(valid_models)}"
        )

    valid_sizes = ['teacher', 'student']
    if run_type['model_size'] not in valid_sizes:
        raise ValueError(
            f"Invalid model_size: {run_type['model_size']}\n"
            f"Valid options: {', '.join(valid_sizes)}"
        )

    valid_modalities = ['automotive', 'industrial', 'robotics']
    if run_type['modality'] not in valid_modalities:
        raise ValueError(
            f"Invalid modality: {run_type['modality']}\n"
            f"Valid options: {', '.join(valid_modalities)}"
        )

    valid_learning_types = ['supervised', 'unsupervised', 'semi_supervised', 'rl_fusion']
    if run_type['learning_type'] not in valid_learning_types:
        raise ValueError(
            f"Invalid learning_type: {run_type['learning_type']}\n"
            f"Valid options: {', '.join(valid_learning_types)}"
        )

    valid_distillation = ['with-kd', 'no-kd']
    if run_type['distillation'] not in valid_distillation:
        raise ValueError(
            f"Invalid distillation: {run_type['distillation']}\n"
            f"Valid options: {', '.join(valid_distillation)}"
        )

    valid_modes = ['normal', 'autoencoder', 'curriculum', 'fusion', 'distillation', 'evaluation']
    if run_type['mode'] not in valid_modes:
        raise ValueError(
            f"Invalid mode: {run_type['mode']}\n"
            f"Valid options: {', '.join(valid_modes)}"
        )


def merge_with_defaults(user_args: Dict[str, str], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user-provided arguments with default values.

    Args:
        user_args: User-provided arguments (strings)
        defaults: Default values (with correct types)

    Returns:
        Merged dictionary with type-converted user values
    """
    result = defaults.copy()

    for key, value in user_args.items():
        if key in defaults:
            # Convert to same type as default
            default_type = type(defaults[key])
            try:
                if default_type == bool:
                    # Handle boolean conversion
                    result[key] = value.lower() in ('true', 'yes', '1', 'on')
                elif default_type == int:
                    result[key] = int(value)
                elif default_type == float:
                    result[key] = float(value)
                else:
                    result[key] = value
            except (ValueError, AttributeError) as e:
                logger.warning(f"Could not convert {key}={value} to {default_type.__name__}, using as string")
                result[key] = value
        else:
            # Unknown key, keep as string
            logger.warning(f"Unknown argument: {key}={value} (not in defaults)")
            result[key] = value

    return result


def build_config_from_buckets(
    run_type_str: str,
    model_args_str: str = "",
    slurm_args_str: str = "",
) -> List[Tuple[Dict, Dict, Dict]]:
    """
    Build configuration(s) from bucket strings.

    Args:
        run_type_str: Run type bucket (model=gat,dataset=hcrl_ch,...)
        model_args_str: Model args bucket (epochs=100,lr=0.001,...)
        slurm_args_str: SLURM args bucket (walltime=12:00:00,...)

    Returns:
        List of (run_type, model_args, slurm_args) tuples, one per sweep combination

    Raises:
        ValueError: If parsing fails or validation fails
    """
    # Parse buckets
    run_type_dict = parse_bucket(run_type_str)
    model_args_dict = parse_bucket(model_args_str)
    slurm_args_dict = parse_bucket(slurm_args_str)

    # Expand sweeps
    run_type_configs = expand_sweep(run_type_dict)
    model_args_configs = expand_sweep(model_args_dict)
    slurm_args_configs = expand_sweep(slurm_args_dict)

    # Validate each run_type config
    for rt in run_type_configs:
        validate_run_type(rt)

    # Build all combinations (run_type × model_args × slurm_args)
    results = []
    for rt in run_type_configs:
        for ma in model_args_configs:
            for sa in slurm_args_configs:
                # Merge with defaults
                model_args_merged = merge_with_defaults(ma, DEFAULT_MODEL_ARGS)
                slurm_args_merged = merge_with_defaults(sa, DEFAULT_SLURM_ARGS)

                results.append((rt, model_args_merged, slurm_args_merged))

    logger.info(f"Generated {len(results)} configuration(s) from buckets")

    return results


# ============================================================================
# CANGraphConfig Creation
# ============================================================================

def create_can_graph_config(
    run_type: Dict[str, str],
    model_args: Dict[str, Any],
    slurm_args: Dict[str, Any],
) -> 'CANGraphConfig':
    """
    Create CANGraphConfig from bucket parameters.

    Args:
        run_type: Run type configuration with ALL explicit parameters:
                  - model, model_size, dataset, mode
                  - modality, learning_type, distillation (NEW - explicit)
        model_args: Model arguments (epochs, lr, etc.)
        slurm_args: SLURM arguments (walltime, memory, etc.)

    Returns:
        CANGraphConfig object ready for training

    Examples:
        >>> run_type = {
        ...     'model': 'gat',
        ...     'model_size': 'teacher',
        ...     'dataset': 'hcrl_ch',
        ...     'mode': 'normal',
        ...     'modality': 'automotive',
        ...     'learning_type': 'supervised',
        ...     'distillation': 'no-kd'
        ... }
        >>> model_args = {'epochs': 50, 'learning_rate': 0.001}
        >>> slurm_args = {'walltime': '06:00:00', 'memory': '64G'}
        >>> config = create_can_graph_config(run_type, model_args, slurm_args)
    """
    # Import hydra_zen_configs directly to avoid triggering src/__init__.py
    # which imports torch (not available without conda environment active)
    import importlib.util
    import sys

    config_path = Path(__file__).parent.parent / 'config' / 'hydra_zen_configs.py'
    spec = importlib.util.spec_from_file_location("hydra_zen_configs", config_path)
    hydra_zen_configs = importlib.util.module_from_spec(spec)
    sys.modules['hydra_zen_configs'] = hydra_zen_configs
    spec.loader.exec_module(hydra_zen_configs)

    CANGraphConfigStore = hydra_zen_configs.CANGraphConfigStore

    # Extract all explicit parameters (DESIGN PRINCIPLE 1)
    model = run_type['model']
    model_size = run_type['model_size']
    dataset_name = run_type['dataset']
    training_mode = run_type['mode']
    modality = run_type['modality']
    learning_type = run_type['learning_type']
    distillation = run_type['distillation']

    # Log that we're using explicit parameters (not computing them)
    logger.info(
        f"Building config with EXPLICIT parameters:\n"
        f"    modality={modality}\n"
        f"    learning_type={learning_type}\n"
        f"    distillation={distillation}\n"
        f"    (DESIGN PRINCIPLE 1: No implicit computation)"
    )

    # Build model_type string from model + model_size
    if model_size == 'student':
        model_type = f"{model}_student"
    else:
        model_type = model

    # Map mode names to training config names
    mode_mapping = {
        'normal': 'normal',
        'autoencoder': 'autoencoder',
        'curriculum': 'curriculum',
        'fusion': 'fusion',
        'distillation': 'knowledge_distillation',
    }

    config_mode = mode_mapping.get(training_mode, training_mode)

    # Build overrides from model_args
    overrides = {}

    # Map common model_args to training config fields
    arg_mapping = {
        'epochs': 'max_epochs',
        'learning_rate': 'learning_rate',
        'batch_size': 'batch_size',
        'weight_decay': 'weight_decay',
        'early_stopping': 'early_stopping_patience',  # Special case: bool → int
        'patience': 'early_stopping_patience',
        'gradient_checkpointing': None,  # Handled via MemoryOptimizationConfig
        'optimize_batch_size': 'optimize_batch_size',
    }

    for key, value in model_args.items():
        # Skip defaults (only apply non-default values)
        if key in DEFAULT_MODEL_ARGS and value == DEFAULT_MODEL_ARGS[key]:
            continue

        # Map to training config field name
        if key in arg_mapping:
            config_key = arg_mapping[key]
            if config_key:
                # Special handling for early_stopping (bool → patience value)
                if key == 'early_stopping' and isinstance(value, bool):
                    if not value:  # If early_stopping is False, set huge patience
                        overrides['early_stopping_patience'] = 999999
                else:
                    overrides[config_key] = value
        else:
            # Pass through unknown args (might be mode-specific)
            overrides[key] = value

    # Create config via CANGraphConfigStore
    store = CANGraphConfigStore()
    config = store.create_config(
        model_type=model_type,
        dataset_name=dataset_name,
        training_mode=config_mode,
        **overrides
    )

    # Apply model-specific overrides to model config
    model_specific_overrides = {}

    # GAT-specific
    if 'num_layers' in model_args and model_args['num_layers'] != DEFAULT_MODEL_ARGS['num_layers']:
        model_specific_overrides['num_layers'] = model_args['num_layers']
    if 'heads' in model_args and model_args['heads'] != DEFAULT_MODEL_ARGS['heads']:
        model_specific_overrides['heads'] = model_args['heads']
    if 'hidden_channels' in model_args and model_args['hidden_channels'] != DEFAULT_MODEL_ARGS['hidden_channels']:
        model_specific_overrides['hidden_channels'] = model_args['hidden_channels']
    if 'dropout' in model_args and model_args['dropout'] != DEFAULT_MODEL_ARGS['dropout']:
        model_specific_overrides['dropout'] = model_args['dropout']

    # VGAE-specific
    if 'latent_dim' in model_args and model_args['latent_dim'] != DEFAULT_MODEL_ARGS['latent_dim']:
        model_specific_overrides['latent_dim'] = model_args['latent_dim']

    # DQN-specific
    if 'gamma' in model_args and model_args['gamma'] != DEFAULT_MODEL_ARGS['gamma']:
        model_specific_overrides['gamma'] = model_args['gamma']
    if 'replay_buffer_size' in model_args and model_args['replay_buffer_size'] != DEFAULT_MODEL_ARGS['replay_buffer_size']:
        model_specific_overrides['buffer_size'] = model_args['replay_buffer_size']

    # Apply model-specific overrides
    for key, value in model_specific_overrides.items():
        if hasattr(config.model, key):
            setattr(config.model, key, value)
        else:
            logger.warning(f"Model config does not have attribute '{key}', skipping")

    # Handle gradient checkpointing (applied to memory optimization config if available)
    if 'gradient_checkpointing' in model_args:
        if hasattr(config.training, 'memory_optimization'):
            config.training.memory_optimization.gradient_checkpointing = model_args['gradient_checkpointing']
        else:
            logger.warning("Training config does not support memory_optimization.gradient_checkpointing")

    # Store SLURM args for later use by JobManager (attach as metadata)
    # We'll add this as a custom attribute for JobManager to read
    config._slurm_args = slurm_args

    # Store explicit parameters as metadata (for tracking and validation)
    config._modality = modality
    config._learning_type = learning_type
    config._distillation = distillation
    config._canonical_path = f"{modality}/{dataset_name}/{model_size}/{learning_type}/{model}/{distillation}/{training_mode}"

    logger.info(
        f"Created CANGraphConfig: {model_type} on {dataset_name} with mode {config_mode}\n"
        f"    Canonical path: {config._canonical_path}"
    )

    return config


# ============================================================================
# Pretty Printing
# ============================================================================

def format_config_summary(
    run_type: Dict[str, str],
    model_args: Dict[str, Any],
    slurm_args: Dict[str, Any],
    index: int = 0,
    total: int = 1,
) -> str:
    """Format configuration summary for display."""
    lines = []

    if total > 1:
        lines.append(f"\n{'='*70}")
        lines.append(f"Configuration {index + 1} of {total}")
        lines.append(f"{'='*70}")
    else:
        lines.append(f"\n{'='*70}")
        lines.append("Configuration Summary")
        lines.append(f"{'='*70}")

    lines.append("\n[Run Type]")
    for key, value in run_type.items():
        lines.append(f"  {key}: {value}")

    # Show only non-default model args
    non_default_model = {k: v for k, v in model_args.items()
                         if k not in DEFAULT_MODEL_ARGS or v != DEFAULT_MODEL_ARGS[k]}
    if non_default_model:
        lines.append("\n[Model Args] (non-default)")
        for key, value in non_default_model.items():
            lines.append(f"  {key}: {value}")

    # Show only non-default slurm args
    non_default_slurm = {k: v for k, v in slurm_args.items()
                         if k not in DEFAULT_SLURM_ARGS or v != DEFAULT_SLURM_ARGS[k]}
    if non_default_slurm:
        lines.append("\n[SLURM Args] (non-default)")
        for key, value in non_default_slurm.items():
            lines.append(f"  {key}: {value}")

    lines.append(f"{'='*70}\n")

    return '\n'.join(lines)
