#!/usr/bin/env python3
"""
VGAE Autoencoder Training Example

Train an unsupervised VGAE model for anomaly detection.
"""

from src.config.hydra_zen_configs import create_autoencoder_config, validate_config
from src.training.trainer import HydraZenTrainer

def main():
    # Create configuration for VGAE autoencoder
    config = create_autoencoder_config("hcrl_sa")
    
    # Override defaults if needed
    config.training.max_epochs = 200
    config.training.batch_size = 64
    config.training.learning_rate = 0.0005
    config.training.use_normal_samples_only = True  # Unsupervised on normal data
    
    # Validate
    print("Validating configuration...")
    validate_config(config)
    
    # Initialize trainer
    print(f"\nInitializing VGAE trainer")
    print(f"Model: {config.model.type}")
    print(f"Training mode: {config.training.mode}")
    print(f"Using normal samples only: {config.training.use_normal_samples_only}")
    print(f"Output: {config.canonical_experiment_dir()}")
    
    trainer = HydraZenTrainer(config)
    
    # Train
    print("\nStarting VGAE training...")
    results = trainer.train()
    
    # Results
    print("\n" + "="*60)
    print("VGAE Training Complete!")
    print("="*60)
    print(f"Best reconstruction loss: {results.get('best_val_loss', 'N/A'):.4f}")
    print(f"Model saved to: {results.get('checkpoint_path', 'N/A')}")
    print(f"\nUse this model for:")
    print("  - Anomaly detection")
    print("  - Curriculum learning (as VGAE guidance)")
    print("  - Fusion training (as autoencoder component)")

if __name__ == "__main__":
    main()
