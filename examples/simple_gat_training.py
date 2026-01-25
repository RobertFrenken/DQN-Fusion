#!/usr/bin/env python3
"""
Simple GAT Training Example

Train a teacher GAT model for supervised classification.
"""

from src.config.hydra_zen_configs import create_gat_normal_config, validate_config
from src.training.trainer import HydraZenTrainer

def main():
    # Create configuration
    config = create_gat_normal_config("hcrl_sa")
    
    # Optional: Override defaults
    config.training.max_epochs = 100
    config.training.batch_size = 64
    config.training.learning_rate = 0.001
    
    # Validate configuration
    print("Validating configuration...")
    validate_config(config)
    
    # Initialize trainer
    print(f"\nInitializing trainer for {config.experiment_name}")
    print(f"Model: {config.model.type}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Training mode: {config.training.mode}")
    print(f"Output: {config.canonical_experiment_dir()}")
    
    trainer = HydraZenTrainer(config)
    
    # Train model
    print("\nStarting training...")
    results = trainer.train()
    
    # Print results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation accuracy: {results.get('best_val_acc', 'N/A'):.4f}")
    print(f"Best validation loss: {results.get('best_val_loss', 'N/A'):.4f}")
    print(f"Model saved to: {results.get('checkpoint_path', 'N/A')}")
    print(f"Experiment directory: {config.canonical_experiment_dir()}")

if __name__ == "__main__":
    main()
