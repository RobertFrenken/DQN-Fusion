#!/usr/bin/env python3
"""
Curriculum Learning Example

Train GAT with VGAE-guided hard sample mining for improved performance.
"""

from pathlib import Path
from src.config.hydra_zen_configs import (
    CANGraphConfigStore,
    validate_config
)
from src.training.trainer import HydraZenTrainer

def main():
    # Create curriculum configuration
    store = CANGraphConfigStore()
    config = store.create_config(
        model_type="gat",
        dataset_name="hcrl_sa",
        training_mode="curriculum"
    )
    
    # Check for required VGAE model
    print("Checking for VGAE model (required for curriculum learning)...")
    artifacts = config.required_artifacts()
    vgae_path = artifacts.get("vgae")
    
    if not vgae_path or not vgae_path.exists():
        print(f"ERROR: VGAE model not found at: {vgae_path}")
        print("\nYou must train a VGAE model first:")
        print("  python examples/vgae_autoencoder_training.py")
        return
    
    print(f"  ✓ VGAE found: {vgae_path}")
    
    # Curriculum parameters
    config.training.vgae_model_path = str(vgae_path)
    config.training.start_ratio = 1.0   # Start with 1:1 normal:attack
    config.training.end_ratio = 10.0    # End with 10:1 (realistic imbalance)
    config.training.difficulty_percentile = 75.0  # Use top 75% difficult samples
    
    # Training parameters
    config.training.max_epochs = 400
    config.training.batch_size = 32
    config.training.learning_rate = 0.001
    config.training.early_stopping_patience = 150
    
    # Batch size optimization
    config.training.optimize_batch_size = True
    config.training.max_batch_size_trials = 15
    
    # Validate
    print("\nValidating configuration...")
    validate_config(config)
    
    # Initialize trainer
    print(f"\nCurriculum Learning Setup")
    print(f"Model: {config.model.type}")
    print(f"VGAE guidance: {vgae_path}")
    print(f"Difficulty schedule: {config.training.start_ratio}:1 → {config.training.end_ratio}:1")
    print(f"Hard mining: Top {config.training.difficulty_percentile}% difficult samples")
    print(f"Output: {config.canonical_experiment_dir()}")
    
    trainer = HydraZenTrainer(config)
    
    # Train with curriculum
    print("\nStarting curriculum learning...")
    print("The model will progressively learn:")
    print("  1. Easy samples (balanced dataset)")
    print("  2. Hard samples (VGAE reconstruction errors)")
    print("  3. Realistic imbalance (10:1 normal:attack)")
    
    results = trainer.train()
    
    # Results
    print("\n" + "="*60)
    print("Curriculum Learning Complete!")
    print("="*60)
    print(f"Best validation accuracy: {results.get('best_val_acc', 'N/A'):.4f}")
    print(f"Model saved to: {results.get('checkpoint_path', 'N/A')}")
    print(f"\nCurriculum learning typically achieves:")
    print("  - Better generalization")
    print("  - Improved rare attack detection")
    print("  - More robust to class imbalance")

if __name__ == "__main__":
    main()
