"""
CAN-Graph Training with Hydra-Zen Configurations

This script replaces the YAML-based configuration system with hydra-zen,
providing type-safe, programmatic configuration management.

Usage:
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training knowledge_distillation --teacher_path saved_models/teacher.pth
python train_with_hydra_zen.py --config-preset distillation_hcrl_sa_half
"""

import os
import sys
from pathlib import Path
import logging
import argparse
import warnings
from importlib import import_module
import pandas as pd
import lightning.pytorch as pl

# Ensure minimal Lightning attributes exist when running under test shims
if not hasattr(pl, 'LightningModule'):
    pl.LightningModule = object
if not hasattr(pl, 'Callback'):
    pl.Callback = object
if not hasattr(pl, 'LightningDataModule'):
    pl.LightningDataModule = object
if not hasattr(pl, 'Trainer'):
    import types
    pl.Trainer = lambda *a, **k: types.SimpleNamespace(_kwargs=k, logger=k.get('logger', None))

# Import unified training orchestrator
from src.training.trainer import HydraZenTrainer

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules (robust to path resolution issues in some SLURM environments)
# Import config module and optional factory functions gracefully so tests can inject them

_cfg_mod = import_module('src.config.hydra_zen_configs')
CANGraphConfig = _cfg_mod.CANGraphConfig
CANGraphConfigStore = getattr(_cfg_mod, 'CANGraphConfigStore', None)
create_gat_normal_config = getattr(_cfg_mod, 'create_gat_normal_config', None)
create_distillation_config = getattr(_cfg_mod, 'create_distillation_config', None)
create_autoencoder_config = getattr(_cfg_mod, 'create_autoencoder_config', None)
create_fusion_config = getattr(_cfg_mod, 'create_fusion_config', None)
validate_config = getattr(_cfg_mod, 'validate_config', lambda cfg: True)
# import lighting loader modules

# Suppress warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch-scatter.*")
warnings.filterwarnings("ignore", message=".*GLIBCXX.*")
warnings.filterwarnings("ignore", message=".*Trying to infer.*batch_size.*")
warnings.filterwarnings("ignore", message=".*Checkpoint directory.*exists.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)  # Keep logs clean
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_preset_configs():
    """Return available preset configurations used by CLI helpers."""
    presets = {}
    
    # All datasets we support
    ALL_DATASETS = ["hcrl_sa", "hcrl_ch", "set_01", "set_02", "set_03", "set_04"]
    
    # Normal training presets
    for dataset in ALL_DATASETS:
        presets[f"gat_normal_{dataset}"] = create_gat_normal_config(dataset)
        presets[f"autoencoder_{dataset}"] = create_autoencoder_config(dataset)
    
    # Knowledge distillation presets (requires teacher path to be set)
    for dataset in ["hcrl_sa", "hcrl_ch"]:
        for scale in [0.25, 0.5, 0.75]:
            name = f"distillation_{dataset}_scale_{scale}"
            presets[name] = create_distillation_config(
                dataset=dataset, 
                student_scale=scale,
                teacher_model_path=str(Path(__file__).parent.resolve() / "experimentruns" / "automotive" / dataset / "unsupervised" / "vgae" / "teacher" / "no_distillation" / "autoencoder" / f"best_teacher_model_{dataset}.pth")
            )
    
    # Fusion presets - now for ALL datasets
    for dataset in ALL_DATASETS:
        presets[f"fusion_{dataset}"] = create_fusion_config(dataset)
    
    # Curriculum learning presets - now for ALL datasets
    from src.config.hydra_zen_configs import create_curriculum_config
    for dataset in ALL_DATASETS:
        presets[f"curriculum_{dataset}"] = create_curriculum_config(dataset)
    
    return presets


def list_presets():
    """List available preset configurations."""
    presets = get_preset_configs()
    
    print("📋 Available preset configurations:")
    print("=" * 50)
    
    categories = {
        "Normal Training": [k for k in presets.keys() if k.startswith("gat_normal")],
        "Autoencoder Training": [k for k in presets.keys() if k.startswith("autoencoder")],
        "Curriculum Learning": [k for k in presets.keys() if k.startswith("curriculum")],
        "Knowledge Distillation": [k for k in presets.keys() if k.startswith("distillation")],
        "Fusion Training": [k for k in presets.keys() if k.startswith("fusion")]
    }
    
    for category, preset_names in categories.items():
        if preset_names:
            print(f"\n{category}:")
            for name in sorted(preset_names):
                print(f"  - {name}")


# Helper: apply a dependency manifest to a config
def apply_manifest_to_config(config, manifest_path: str):
    """Load and validate a dependency manifest, apply paths to config.training for fusion jobs.

    This helper keeps validation strict and fails early with informative errors.
    """
    try:
        # First try a regular import (works when package is installed)
        from src.utils.dependency_manifest import load_manifest, validate_manifest_for_config, ManifestValidationError
    except Exception:
        # Fallback: load by file path (tests and some environments use direct file execution)
        try:
            import importlib.util
            dm_path = Path(__file__).parent / 'src' / 'utils' / 'dependency_manifest.py'
            spec = importlib.util.spec_from_file_location('dependency_manifest', str(dm_path))
            dm_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dm_mod)
            load_manifest = dm_mod.load_manifest
            validate_manifest_for_config = dm_mod.validate_manifest_for_config
            ManifestValidationError = dm_mod.ManifestValidationError
        except Exception as e:
            raise RuntimeError(f"Failed to import dependency manifest utilities: {e}") from e

    manifest = load_manifest(manifest_path)
    try:
        ok, msg = validate_manifest_for_config(manifest, config)
    except ManifestValidationError:
        raise
    # If fusion job, apply explicit paths into the training config to be used later
    if getattr(config.training, 'mode', None) == 'fusion':
        autoencoder = manifest['autoencoder']['path']
        classifier = manifest['classifier']['path']
        config.training.autoencoder_path = autoencoder
        config.training.classifier_path = classifier
        print(f"🔒 Dependency manifest applied: autoencoder={autoencoder}, classifier={classifier}")

    return config


# ==========================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CAN-Graph Training with Hydra-Zen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal training
  python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal
  
  # Knowledge distillation
  python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training knowledge_distillation \\
      --teacher_path saved_models/teacher.pth --student_scale 0.5
  
  # Fusion training (uses pre-cached predictions)
  python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training fusion
  
  # Using preset
  python train_with_hydra_zen.py --preset gat_normal_hcrl_sa
  
  # Fusion preset
  python train_with_hydra_zen.py --preset fusion_hcrl_sa
  
  # List presets
  python train_with_hydra_zen.py --list-presets
  
  # Alternatively, use dedicated fusion training script:
  python train_fusion_lightning.py --dataset hcrl_sa --max-epochs 50
        """
    )
    
    # Preset mode
    parser.add_argument('--preset', type=str,
                      help='Use a preset configuration')
    parser.add_argument('--list-presets', action='store_true',
                      help='List available preset configurations')
    
    # Manual configuration
    parser.add_argument('--model', type=str, choices=['gat', 'vgae', 'dqn'], default='gat',
                      help='Model type')
    parser.add_argument('--dataset', type=str, 
                      choices=['hcrl_sa', 'hcrl_ch', 'set_01', 'set_02', 'set_03', 'set_04', 'car_hacking'],
                      default='hcrl_sa', help='Dataset name')
    parser.add_argument('--training', type=str, 
                      choices=['normal', 'autoencoder', 'knowledge_distillation', 'curriculum', 'fusion'],
                      default='normal', help='Training mode')
    
    # Knowledge distillation (toggle - can be used with any mode except fusion)
    parser.add_argument('--use-kd', action='store_true',
                      help='Enable knowledge distillation (requires --teacher_path)')
    parser.add_argument('--teacher_path', type=str,
                      help='Path to teacher model (required for --use-kd or mode=knowledge_distillation)')
    parser.add_argument('--student_scale', type=float, default=1.0,
                      help='Student model scale factor')
    parser.add_argument('--distillation_alpha', type=float, default=0.7,
                      help='KD loss weight: alpha*KD_loss + (1-alpha)*task_loss')
    parser.add_argument('--temperature', type=float, default=4.0,
                      help='KD temperature for soft labels (higher = softer)')
    
    # Curriculum learning specific
    parser.add_argument('--vgae_path', type=str,
                      help='Path to VGAE model (required for curriculum learning)')
    
    # Fusion training specific
    parser.add_argument('--autoencoder_path', type=str,
                      help='Path to autoencoder model (required for fusion training)')
    parser.add_argument('--classifier_path', type=str,
                      help='Path to classifier model (required for fusion training)')

    # Optional: override dataset data path when running on machines where dataset lives elsewhere
    parser.add_argument('--data-path', dest='data_path', type=str, default=None,
                      help='Override dataset data_path if your local dataset lives elsewhere')
    
    # Training overrides
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--tensorboard', action='store_true',
                      help='Enable TensorBoard logging')
    parser.add_argument('--force-rebuild-cache', action='store_true',
                      help='Force rebuild of cached processed data')
    # Removed --debug-graph-count flag as it was unused and causing segfaults
    parser.add_argument('--early-stopping-patience', type=int,
                      help='Early stopping patience (default: 25 for normal, 30 for autoencoder)')
    parser.add_argument('--dependency-manifest', type=str,
                      help='Path to a JSON dependency manifest to validate and apply for fusion training')
    
    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return

    # Validate --use-kd flag
    if getattr(args, 'use_kd', False):
        if not args.teacher_path:
            print("❌ --use-kd requires --teacher_path to specify the teacher model")
            return
        if args.training == 'fusion':
            print("❌ --use-kd is not compatible with fusion mode (DQN uses already-distilled models)")
            return

    # Create configuration
    if args.preset:
        presets = get_preset_configs()
        if args.preset not in presets:
            print(f"❌ Unknown preset: {args.preset}")
            print("Available presets:")
            list_presets()
            return
        config = presets[args.preset]
        
        # Apply teacher path if provided
        if args.teacher_path and hasattr(config.training, 'teacher_model_path'):
            config.training.teacher_model_path = args.teacher_path

        # Apply optional dataset path override
        if args.data_path:
            if hasattr(config, 'dataset') and hasattr(config.dataset, 'data_path'):
                config.dataset.data_path = args.data_path
            else:
                print(f"Warning: --data-path provided but config has no dataset.data_path attribute: {args.data_path}")
    
    else:
        # Manual configuration
        store_manager = CANGraphConfigStore()
        
        # Prepare overrides
        overrides = {}
        # Knowledge distillation toggle (new orthogonal approach)
        if getattr(args, 'use_kd', False):
            overrides['use_knowledge_distillation'] = True
        if args.teacher_path:
            overrides['teacher_model_path'] = args.teacher_path
        if args.student_scale != 1.0:
            overrides['student_model_scale'] = args.student_scale
        if args.distillation_alpha != 0.7:
            overrides['distillation_alpha'] = args.distillation_alpha
        if args.temperature != 4.0:
            overrides['distillation_temperature'] = args.temperature
        if args.epochs:
            overrides['max_epochs'] = args.epochs
        if args.batch_size:
            overrides['batch_size'] = args.batch_size
        if args.learning_rate:
            overrides['learning_rate'] = args.learning_rate
        if args.early_stopping_patience:
            overrides['early_stopping_patience'] = args.early_stopping_patience
        
        config = store_manager.create_config(args.model, args.dataset, args.training, **overrides)
    
    # Apply global overrides
    if args.tensorboard:
        config.logging["enable_tensorboard"] = True

    # If a dependency manifest is provided, validate and apply it (strict, fail-fast)
    if getattr(args, 'dependency_manifest', None):
        try:
            config = apply_manifest_to_config(config, args.dependency_manifest)
        except Exception as e:
            print(f"❌ Dependency manifest validation failed: {e}")
            raise
    
    # Apply environment override for experiment root if provided (useful for job managers)
    env_exp_root = os.environ.get('CAN_EXPERIMENT_ROOT') or os.environ.get('EXPERIMENT_ROOT')
    if env_exp_root:
        try:
            config.experiment_root = str(Path(env_exp_root).resolve())
            print(f"🔧 Overriding experiment root from env: {config.experiment_root}")
        except Exception:
            print(f"⚠️  Invalid experiment root provided in CAN_EXPERIMENT_ROOT: {env_exp_root}")

    # Print configuration summary
    print(f"📋 Configuration Summary:")
    print(f"   Model: {config.model.type}")
    print(f"   Dataset: {config.dataset.name}")
    print(f"   Training: {config.training.mode}")
    print(f"   Experiment: {config.experiment_name}")
    
    if hasattr(config.training, 'teacher_model_path') and config.training.teacher_model_path:
        print(f"   Teacher: {config.training.teacher_model_path}")
        print(f"   Student scale: {config.training.student_model_scale}")
    
    # Create trainer and run
    try:
        trainer = HydraZenTrainer(config)
        model, lightning_trainer = trainer.train()
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()