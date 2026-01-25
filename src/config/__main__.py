#!/usr/bin/env python3
"""
Configuration CLI Helpers

Provides command-line utilities for managing KD-GAT configurations.

Usage:
    python -m src.config list-models
    python -m src.config list-datasets
    python -m src.config list-training-modes
    python -m src.config validate <config_file>
    python -m src.config show <preset_name>
    python -m src.config create --model gat --dataset hcrl_sa --training normal
"""

import argparse
from pathlib import Path

from src.config.hydra_zen_configs import (
    CANGraphConfigStore,
    CANGraphConfig,
    validate_config,
    create_gat_normal_config,
    create_autoencoder_config,
    create_distillation_config,
    create_fusion_config
)


def list_models():
    """List all available model types."""
    models = {
        "gat": "Graph Attention Network (Teacher, ~1.1M params)",
        "gat_student": "GAT Student (Onboard, ~55K params)",
        "vgae": "Variational Graph AutoEncoder (Teacher, ~1.74M params)",
        "vgae_student": "VGAE Student (Onboard, ~87K params)",
        "dqn": "Deep Q-Network for Fusion (Teacher, ~687K params)",
        "dqn_student": "DQN Student (Onboard, ~32K params)"
    }
    
    print("\n📦 Available Models\n" + "="*60)
    for model, desc in models.items():
        print(f"  {model:15s} - {desc}")
    print()


def list_datasets():
    """List all available datasets."""
    datasets = {
        "hcrl_sa": "HCRL Speed Acceleration dataset",
        "hcrl_ch": "HCRL Car Hacking dataset",
        "set_01": "SET 01 - Known attacks, known vehicles",
        "set_02": "SET 02 - Known attacks, unknown vehicles",
        "set_03": "SET 03 - Unknown attacks, known vehicles",
        "set_04": "SET 04 - Unknown attacks, unknown vehicles"
    }
    
    print("\n📊 Available Datasets\n" + "="*60)
    for dataset, desc in datasets.items():
        print(f"  {dataset:15s} - {desc}")
    print()


def list_training_modes():
    """List all available training modes."""
    modes = {
        "normal": "Standard supervised training",
        "autoencoder": "Unsupervised VGAE reconstruction",
        "curriculum": "Hard sample mining with VGAE guidance",
        "knowledge_distillation": "Teacher→Student compression",
        "student_baseline": "Student-only training (no KD)",
        "fusion": "Multi-model ensemble with DQN"
    }
    
    print("\n🎯 Available Training Modes\n" + "="*60)
    for mode, desc in modes.items():
        print(f"  {mode:22s} - {desc}")
    print()


def validate_config_file(config_path: str):
    """Validate a configuration file or object."""
    try:
        # Try to load config if it's a file
        if Path(config_path).exists():
            # Load Python file and extract config
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", config_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'config'):
                config = module.config
            elif hasattr(module, 'main'):
                print("Config file has main() function - run it directly")
                return
            else:
                print(f"❌ No 'config' object found in {config_path}")
                return
        else:
            print(f"❌ File not found: {config_path}")
            return
        
        # Validate
        print(f"\n🔍 Validating configuration from {config_path}...")
        is_valid = validate_config(config)
        
        if is_valid:
            print("\n✅ Configuration is valid!")
            print(f"\nConfig details:")
            print(f"  Model: {config.model.type}")
            print(f"  Dataset: {config.dataset.name}")
            print(f"  Training mode: {config.training.mode}")
            print(f"  Experiment: {config.experiment_name}")
            print(f"  Output: {config.canonical_experiment_dir()}")
        else:
            print("\n❌ Configuration validation failed")
            
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()


def show_preset(preset_name: str):
    """Show details of a configuration preset."""
    presets = {
        "gat_normal": lambda ds: create_gat_normal_config(ds),
        "vgae_autoencoder": lambda ds: create_autoencoder_config(ds),
        "fusion": lambda ds: create_fusion_config(ds),
    }
    
    # Extract preset type and dataset
    parts = preset_name.split("_")
    if len(parts) < 2:
        print(f"❌ Invalid preset format. Expected: <type>_<dataset>")
        print(f"   Examples: gat_normal_hcrl_sa, vgae_autoencoder_hcrl_ch")
        return
    
    preset_type = "_".join(parts[:-1])
    dataset = parts[-1]
    
    if preset_type not in presets:
        print(f"❌ Unknown preset type: {preset_type}")
        print(f"   Available: {list(presets.keys())}")
        return
    
    try:
        config = presets[preset_type](dataset)
        
        print(f"\n📋 Preset: {preset_name}\n" + "="*60)
        print(f"Model: {config.model.type}")
        print(f"Dataset: {config.dataset.name}")
        print(f"Training mode: {config.training.mode}")
        print(f"\nModel parameters:")
        print(f"  Hidden dim: {getattr(config.model, 'hidden_channels', 'N/A')}")
        print(f"  Layers: {getattr(config.model, 'num_layers', 'N/A')}")
        print(f"  Dropout: {getattr(config.model, 'dropout', 'N/A')}")
        print(f"\nTraining parameters:")
        print(f"  Epochs: {config.training.max_epochs}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Learning rate: {config.training.learning_rate}")
        print(f"\nOutput:")
        print(f"  {config.canonical_experiment_dir()}")
        
    except Exception as e:
        print(f"❌ Error creating preset: {e}")


def create_config(model: str, dataset: str, training: str, output: str = None):
    """Create a new configuration."""
    try:
        store = CANGraphConfigStore()
        config = store.create_config(
            model_type=model,
            dataset_name=dataset,
            training_mode=training
        )
        
        print(f"\n✅ Configuration created!")
        print(f"\nModel: {config.model.type}")
        print(f"Dataset: {config.dataset.name}")
        print(f"Training: {config.training.mode}")
        print(f"Experiment: {config.experiment_name}")
        
        if output:
            # Save to file
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create Python file with config
            code = f'''"""
Auto-generated configuration

Model: {config.model.type}
Dataset: {config.dataset.name}
Training: {config.training.mode}
"""

from src.config.hydra_zen_configs import CANGraphConfigStore

store = CANGraphConfigStore()
config = store.create_config(
    model_type="{model}",
    dataset_name="{dataset}",
    training_mode="{training}"
)

if __name__ == "__main__":
    from src.config.hydra_zen_configs import validate_config
    from src.training.trainer import HydraZenTrainer
    
    # Validate
    validate_config(config)
    
    # Train
    trainer = HydraZenTrainer(config)
    results = trainer.train()
    
    print(f"Training complete!")
    print(f"Results: {{results}}")
'''
            
            output_path.write_text(code)
            print(f"\n💾 Saved to: {output_path}")
            print(f"\nTo use:")
            print(f"  python {output_path}")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        print(f"\nValid options:")
        print(f"  Models: gat, gat_student, vgae, vgae_student, dqn, dqn_student")
        print(f"  Datasets: hcrl_sa, hcrl_ch, set_01, set_02, set_03, set_04")
        print(f"  Training: normal, autoencoder, curriculum, knowledge_distillation, fusion")


def main():
    parser = argparse.ArgumentParser(
        description="KD-GAT Configuration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.config list-models
  python -m src.config list-datasets
  python -m src.config show gat_normal_hcrl_sa
  python -m src.config create --model gat --dataset hcrl_sa --training normal
  python -m src.config validate my_config.py
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # list-models
    subparsers.add_parser('list-models', help='List available model types')
    
    # list-datasets
    subparsers.add_parser('list-datasets', help='List available datasets')
    
    # list-training-modes
    subparsers.add_parser('list-training-modes', help='List available training modes')
    
    # validate
    validate_parser = subparsers.add_parser('validate', help='Validate a configuration')
    validate_parser.add_argument('config', help='Configuration file to validate')
    
    # show
    show_parser = subparsers.add_parser('show', help='Show preset details')
    show_parser.add_argument('preset', help='Preset name (e.g., gat_normal_hcrl_sa)')
    
    # create
    create_parser = subparsers.add_parser('create', help='Create a new configuration')
    create_parser.add_argument('--model', required=True, help='Model type')
    create_parser.add_argument('--dataset', required=True, help='Dataset name')
    create_parser.add_argument('--training', required=True, help='Training mode')
    create_parser.add_argument('--output', '-o', help='Output file (optional)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Dispatch commands
    if args.command == 'list-models':
        list_models()
    elif args.command == 'list-datasets':
        list_datasets()
    elif args.command == 'list-training-modes':
        list_training_modes()
    elif args.command == 'validate':
        validate_config_file(args.config)
    elif args.command == 'show':
        show_preset(args.preset)
    elif args.command == 'create':
        create_config(args.model, args.dataset, args.training, args.output)


if __name__ == "__main__":
    main()
