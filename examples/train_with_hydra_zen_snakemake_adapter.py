"""
Example adapter code for train_with_hydra_zen.py to support Snakemake.

This shows how to modify your training script to accept simple CLI arguments
from Snakemake while maintaining backward compatibility with the frozen config
system used by can-train.

Add this code to your train_with_hydra_zen.py file.
"""

import argparse
import sys
from pathlib import Path


def create_snakemake_parser():
    """
    Create simple argument parser for Snakemake invocation.

    This parser accepts straightforward CLI arguments that Snakemake can easily
    pass from the Snakefile, avoiding the complex bucket-based config system.
    """
    parser = argparse.ArgumentParser(
        description='Train GNN models (Snakemake-compatible interface)'
    )

    # Core configuration
    parser.add_argument(
        '--model',
        required=True,
        choices=['vgae', 'gat', 'dqn'],
        help='Model architecture'
    )
    parser.add_argument(
        '--model-size',
        required=True,
        choices=['teacher', 'student'],
        help='Model size variant'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset name (e.g., hcrl_sa, hcrl_ch)'
    )
    parser.add_argument(
        '--modality',
        required=True,
        choices=['automotive', 'industrial', 'robotics'],
        help='Application domain'
    )
    parser.add_argument(
        '--training',
        required=True,
        choices=['autoencoder', 'curriculum', 'fusion', 'normal'],
        help='Training strategy/mode'
    )
    parser.add_argument(
        '--learning-type',
        required=True,
        choices=['supervised', 'unsupervised', 'rl_fusion'],
        help='Learning paradigm'
    )

    # Paths for special modes
    parser.add_argument(
        '--teacher-path',
        type=Path,
        default=None,
        help='Path to teacher model for knowledge distillation'
    )
    parser.add_argument(
        '--vgae-path',
        type=Path,
        default=None,
        help='Path to VGAE model for fusion'
    )
    parser.add_argument(
        '--gat-path',
        type=Path,
        default=None,
        help='Path to GAT model for fusion'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for models and logs'
    )

    # Optional hyperparameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')

    return parser


def build_config_from_simple_args(args):
    """
    Build full config object from simple Snakemake arguments.

    This function translates the simple CLI arguments into whatever config
    structure your training code expects. Adapt this to match your actual
    config structure.

    Args:
        args: Parsed arguments from create_snakemake_parser()

    Returns:
        Config object suitable for train() function
    """
    # Example structure - adapt to your actual config format
    config = {
        'model': {
            'type': args.model,
            'size': args.model_size,
        },
        'dataset': {
            'name': args.dataset,
            'modality': args.modality,
        },
        'training': {
            'mode': args.training,
            'learning_type': args.learning_type,
        },
        'paths': {
            'output_dir': args.output_dir,
        },
    }

    # Add distillation config if teacher path provided
    if args.teacher_path:
        config['distillation'] = {
            'enabled': True,
            'teacher_path': args.teacher_path,
        }

    # Add fusion paths if provided
    if args.vgae_path and args.gat_path:
        config['fusion'] = {
            'vgae_path': args.vgae_path,
            'gat_path': args.gat_path,
        }

    # Add optional hyperparameters
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    return config


def main():
    """
    Main entry point with dual-mode support.

    Detects whether called from:
    1. can-train CLI (legacy) → use frozen config
    2. Snakemake (new) → use simple args
    """

    # Mode 1: Legacy frozen config path (from can-train)
    if '--frozen-config' in sys.argv:
        print("Running in LEGACY mode (frozen config from can-train)")
        # Your existing frozen config loading code
        # Example:
        # from src.config.frozen_config import load_frozen_config
        # config_path = sys.argv[sys.argv.index('--frozen-config') + 1]
        # config = load_frozen_config(config_path)
        # train(config)
        pass

    # Mode 2: New Snakemake mode (simple CLI args)
    else:
        print("Running in SNAKEMAKE mode (simple CLI arguments)")
        parser = create_snakemake_parser()
        args = parser.parse_args()

        # Convert simple args to config
        config = build_config_from_simple_args(args)

        # Run training with config
        # train(config)  # Your actual training function
        print(f"Would train with config: {config}")


if __name__ == '__main__':
    main()


# ============================================================================
# Integration Instructions
# ============================================================================
"""
To integrate this into your existing train_with_hydra_zen.py:

1. Copy create_snakemake_parser() function
2. Copy build_config_from_simple_args() function (adapt to your config format!)
3. Modify your main() function to add the dual-mode detection:

    def main():
        if '--frozen-config' in sys.argv:
            # Existing frozen config path
            run_with_frozen_config()
        else:
            # New Snakemake path
            parser = create_snakemake_parser()
            args = parser.parse_args()
            config = build_config_from_simple_args(args)
            train(config)

4. Test both modes:

   # Old way (should still work)
   python train_with_hydra_zen.py --frozen-config /path/to/config.json

   # New way (for Snakemake)
   python train_with_hydra_zen.py \
       --model vgae \
       --model-size teacher \
       --dataset hcrl_sa \
       --modality automotive \
       --training autoencoder \
       --learning-type unsupervised \
       --output-dir test_output

5. Verify model outputs:
   - Models saved to: {output_dir}/models/{model}_{size}_{distillation}_best.pth
   - Checkpoints saved to: {output_dir}/checkpoints/last.ckpt
   - Logs saved to: {output_dir}/logs/training.log

That's it! Your training script now works with both systems.
"""
