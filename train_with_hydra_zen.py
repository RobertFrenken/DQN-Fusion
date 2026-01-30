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

import torch.multiprocessing as mp
# Must be called before any CUDA or multiprocessing usage.
# Prevents "Cannot re-initialize CUDA in forked subprocess" errors.
mp.set_start_method('spawn', force=True)

import pandas as pd
import lightning.pytorch as pl

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


def create_snakemake_parser():
    """Simple parser for Snakemake invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['vgae', 'gat', 'dqn'])
    parser.add_argument('--model-size', required=True, choices=['teacher', 'student'])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--modality', required=True)
    parser.add_argument('--training', required=True,
                       choices=['autoencoder', 'curriculum', 'fusion'])
    parser.add_argument('--learning-type', required=True)
    parser.add_argument('--teacher-path', default=None)
    parser.add_argument('--vgae-path', default=None)
    parser.add_argument('--gat-path', default=None)
    parser.add_argument('--output-dir', required=True)
    return parser

def main():
    # Detect if called from Snakemake (simpler CLI)
    # vs. from can-train CLI (frozen config)

    if '--frozen-config' in sys.argv:
        # Legacy path: use frozen config
        use_frozen_config_path()
    else:
        # New path: Snakemake direct invocation
        parser = create_snakemake_parser()
        args = parser.parse_args()

        # Build config from simple arguments
        config = build_config_from_args(args)

        # Run training
        train(config)

# ==========================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CAN-Graph Training with Hydra-Zen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Frozen config (recommended for SLURM jobs)
  python train_with_hydra_zen.py --frozen-config /path/to/frozen_config.json
        """
    )

    # =========================================================================
    # Frozen Config Mode (Preferred for SLURM jobs)
    # =========================================================================
    parser.add_argument('--frozen-config', dest='frozen_config', type=str,
                      help='Path to frozen config JSON file (bypasses all other config options)')

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

    # Model size (independent from distillation - affects architecture, not learning strategy)
    parser.add_argument('--model-size', dest='model_size', type=str,
                      choices=['teacher', 'student'],
                      default='teacher',
                      help='Model size: teacher (larger arch) or student (smaller arch)')

    # Modality (application domain - affects paths and data loading)
    parser.add_argument('--modality', type=str,
                      choices=['automotive', 'industrial', 'robotics'],
                      default='automotive',
                      help='Application domain/modality')

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


    # =========================================================================
    # Frozen Config Mode: Load pre-validated config from JSON
    # =========================================================================
    if getattr(args, 'frozen_config', None):
        try:
            from src.config.frozen_config import load_frozen_config

            print(f"📖 Loading frozen config: {args.frozen_config}")
            config = load_frozen_config(args.frozen_config)

            # Print configuration summary
            print(f"📋 Configuration Summary (from frozen config):")
            print(f"   Model: {config.model.type}")
            print(f"   Dataset: {config.dataset.name}")
            print(f"   Training: {config.training.mode}")
            print(f"   Experiment: {config.experiment_name}")

            if hasattr(config.training, 'teacher_model_path') and config.training.teacher_model_path:
                print(f"   Teacher: {config.training.teacher_model_path}")
                print(f"   Student scale: {getattr(config.training, 'student_model_scale', 'N/A')}")

            # Create trainer and run
            try:
                trainer = HydraZenTrainer(config)
                model, lightning_trainer = trainer.train()
            except KeyboardInterrupt:
                print("\n🛑 Training interrupted by user")
            except Exception as e:
                print(f"❌ Training failed: {e}")
                raise

            return

        except FileNotFoundError as e:
            print(f"❌ Frozen config not found: {args.frozen_config}")
            print(f"   Error: {e}")
            return
        except Exception as e:
            print(f"❌ Failed to load frozen config: {e}")
            raise

   
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