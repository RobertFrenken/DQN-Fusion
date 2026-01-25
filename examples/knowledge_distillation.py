#!/usr/bin/env python3
"""
Knowledge Distillation Example

Train a lightweight student model using knowledge from a trained teacher.
"""

from pathlib import Path
from src.config.hydra_zen_configs import (
    CANGraphConfigStore, 
    validate_config
)
from src.training.trainer import HydraZenTrainer

def main():
    # Path to trained teacher model
    # Update this to your actual teacher model path
    teacher_path = "experiment_runs/automotive/hcrl_sa/supervised/gat/teacher/no_distillation/normal/best_teacher_model.pth"
    
    # Check if teacher exists
    if not Path(teacher_path).exists():
        print(f"ERROR: Teacher model not found at: {teacher_path}")
        print("\nYou must train a teacher model first:")
        print("  python examples/simple_gat_training.py")
        return
    
    # Create student config with knowledge distillation
    store = CANGraphConfigStore()
    config = store.create_config(
        model_type="gat_student",
        dataset_name="hcrl_sa",
        training_mode="knowledge_distillation"
    )
    
    # Set teacher path and distillation parameters
    config.training.teacher_model_path = teacher_path
    config.training.distillation_temperature = 4.0
    config.training.distillation_alpha = 0.7  # Balance between KD and task loss
    config.training.student_model_scale = 1.0
    
    # Training parameters
    config.training.max_epochs = 200
    config.training.batch_size = 64
    config.training.learning_rate = 0.001
    
    # Validate
    print("Validating configuration...")
    validate_config(config)
    
    # Initialize trainer
    print(f"\nKnowledge Distillation Setup")
    print(f"Teacher: {teacher_path}")
    print(f"Student: {config.model.type} ({config.model.target_parameters:,} params)")
    print(f"Temperature: {config.training.distillation_temperature}")
    print(f"Alpha (KD weight): {config.training.distillation_alpha}")
    print(f"Output: {config.canonical_experiment_dir()}")
    
    trainer = HydraZenTrainer(config)
    
    # Train student
    print("\nStarting knowledge distillation...")
    results = trainer.train()
    
    # Results
    print("\n" + "="*60)
    print("Knowledge Distillation Complete!")
    print("="*60)
    print(f"Student validation accuracy: {results.get('best_val_acc', 'N/A'):.4f}")
    print(f"Model saved to: {results.get('checkpoint_path', 'N/A')}")
    print(f"\nStudent model is ready for edge deployment!")
    print(f"  Parameters: {config.model.target_parameters:,}")
    print(f"  Memory budget: ~{config.model.target_parameters * 4 / 1024:.1f} KB")

if __name__ == "__main__":
    main()
