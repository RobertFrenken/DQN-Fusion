#!/usr/bin/env python3
"""
Multi-Model Fusion Training Example

Train a DQN agent to fuse predictions from VGAE and GAT models.
"""

from pathlib import Path
from src.config.hydra_zen_configs import create_fusion_config, validate_config
from src.training.trainer import HydraZenTrainer

def main():
    # Create fusion configuration
    config = create_fusion_config("hcrl_sa")
    
    # Check required artifacts
    print("Checking required artifacts for fusion...")
    artifacts = config.required_artifacts()
    
    missing = []
    for name, path in artifacts.items():
        if path.exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name} MISSING: {path}")
            missing.append(name)
    
    if missing:
        print(f"\nERROR: Missing required models: {', '.join(missing)}")
        print("\nYou must train these models first:")
        print("  1. VGAE autoencoder: python examples/vgae_autoencoder_training.py")
        print("  2. GAT classifier: python examples/simple_gat_training.py")
        return
    
    # Configure fusion training
    config.training.fusion_episodes = 500
    config.training.max_train_samples = 150000
    config.training.max_val_samples = 30000
    
    # DQN agent parameters
    config.training.fusion_agent_config.fusion_lr = 0.001
    config.training.fusion_agent_config.gamma = 0.9
    config.training.fusion_agent_config.fusion_epsilon = 0.9
    
    # Validate
    print("\nValidating configuration...")
    validate_config(config)
    
    # Initialize trainer
    print(f"\nFusion Training Setup")
    print(f"Model: DQN Fusion Agent")
    print(f"Episodes: {config.training.fusion_episodes}")
    print(f"Training samples: {config.training.max_train_samples:,}")
    print(f"Output: {config.canonical_experiment_dir()}")
    
    trainer = HydraZenTrainer(config)
    
    # Train fusion agent
    print("\nStarting fusion training...")
    results = trainer.train()
    
    # Results
    print("\n" + "="*60)
    print("Fusion Training Complete!")
    print("="*60)
    print(f"Best validation accuracy: {results.get('best_val_acc', 'N/A'):.4f}")
    print(f"Model saved to: {results.get('checkpoint_path', 'N/A')}")
    print(f"\nFusion agent learned to optimally combine:")
    print("  - VGAE anomaly scores")
    print("  - GAT classification probabilities")

if __name__ == "__main__":
    main()
