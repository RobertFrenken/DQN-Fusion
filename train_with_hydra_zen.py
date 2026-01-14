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
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.tuner import Tuner

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from src.config.hydra_zen_configs import (
    CANGraphConfig, CANGraphConfigStore, 
    create_gat_normal_config, create_distillation_config,
    create_autoencoder_config, create_fusion_config,
    validate_config
)
from train_models import CANGraphLightningModule, load_dataset, create_dataloaders, CANGraphDataModule

# Suppress warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HydraZenTrainer:
    """Training manager using hydra-zen configurations."""
    
    def __init__(self, config: CANGraphConfig):
        self.config = config
        self.validate_config()
    
    def validate_config(self):
        """Validate the configuration."""
        if not validate_config(self.config):
            raise ValueError("Configuration validation failed")
    
    def setup_model(self, num_ids: int) -> CANGraphLightningModule:
        """Create the Lightning module from config."""
        model = CANGraphLightningModule(
            model_config=self.config.model,
            training_config=self.config.training,
            model_type=self.config.model.type,
            training_mode=self.config.training.mode,
            num_ids=num_ids
        )
        return model
    
    def setup_trainer(self) -> pl.Trainer:
        """Create the Lightning trainer from config."""
        # Setup callbacks
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'{self.config.model_save_dir}/lightning_checkpoints',
            filename=f'{self.config.experiment_name}_{{epoch:02d}}_{{val_loss:.3f}}',
            save_top_k=self.config.logging.get("save_top_k", 3),
            monitor=self.config.logging.get("monitor_metric", "val_loss"),
            mode=self.config.logging.get("monitor_mode", "min"),
            save_last=True,
            auto_insert_metric_name=False
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if hasattr(self.config.training, 'early_stopping_patience'):
            early_stop_callback = EarlyStopping(
                monitor=self.config.logging.get("monitor_metric", "val_loss"),
                patience=self.config.training.early_stopping_patience,
                mode=self.config.logging.get("monitor_mode", "min"),
                verbose=True
            )
            callbacks.append(early_stop_callback)
        
        # Setup loggers
        loggers = []
        
        # CSV logger
        csv_logger = CSVLogger(
            save_dir=self.config.log_dir,
            name=self.config.experiment_name
        )
        loggers.append(csv_logger)
        
        # TensorBoard logger
        if self.config.logging.get("enable_tensorboard", False):
            tb_logger = TensorBoardLogger(
                save_dir=self.config.log_dir,
                name=self.config.experiment_name
            )
            loggers.append(tb_logger)
        
        # Create trainer
        trainer = pl.Trainer(
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            precision=self.config.training.precision,
            max_epochs=self.config.training.max_epochs,
            gradient_clip_val=self.config.training.gradient_clip_val,
            accumulate_grad_batches=self.config.training.accumulate_grad_batches,
            logger=loggers,
            callbacks=callbacks,
            enable_checkpointing=self.config.trainer.enable_checkpointing,
            log_every_n_steps=self.config.training.log_every_n_steps,
            enable_progress_bar=self.config.trainer.enable_progress_bar,
            num_sanity_val_steps=self.config.trainer.num_sanity_val_steps
        )
        
        return trainer
    
    def train(self):
        """Execute the complete training pipeline."""
        logger.info(f"🚀 Starting training with hydra-zen config")
        logger.info(f"Experiment: {self.config.experiment_name}")
        logger.info(f"Mode: {self.config.training.mode}")
        
        # Load dataset
        train_dataset, val_dataset, num_ids = load_dataset(self.config.dataset.name, self.config)
        
        # Setup model
        model = self.setup_model(num_ids)
        
        # Optimize batch size if requested
        if self.config.training.optimize_batch_size:
            model = self._optimize_batch_size(model, train_dataset, val_dataset)
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, model.batch_size
        )
        
        # Setup trainer
        trainer = self.setup_trainer()
        
        # Train
        logger.info("Starting training...")
        trainer.fit(model, train_loader, val_loader)
        
        # Test
        if self.config.training.run_test:
            test_results = trainer.test(model, val_loader)
            logger.info(f"Test results: {test_results}")
        
        # Save final model
        model_name = f"{self.config.experiment_name}.pth"
        save_path = Path(self.config.model_save_dir) / model_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to: {save_path}")
        
        logger.info("✅ Training completed successfully!")
        return model, trainer
    
    def _optimize_batch_size(self, model: CANGraphLightningModule, 
                           train_dataset, val_dataset) -> CANGraphLightningModule:
        """Optimize batch size using Lightning's Tuner."""
        logger.info("🔧 Optimizing batch size...")
        
        # Create temporary DataModule
        temp_datamodule = CANGraphDataModule(train_dataset, val_dataset, model.batch_size)
        
        trainer = pl.Trainer(
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            precision='32-true',
            max_epochs=1,
            enable_checkpointing=False,
            logger=False
        )
        
        tuner = Tuner(trainer)
        try:
            tuner.scale_batch_size(
                model,
                datamodule=temp_datamodule,
                mode=self.config.training.batch_size_mode,
                steps_per_trial=3,
                init_val=self.config.training.batch_size,
                max_trials=self.config.training.max_batch_size_trials
            )
            
            logger.info(f"Batch size optimized to: {model.batch_size}")
            
        except Exception as e:
            logger.warning(f"Batch size optimization failed: {e}. Using original size.")
        
        return model


# ============================================================================
# Preset Configurations
# ============================================================================

def get_preset_configs() -> Dict[str, CANGraphConfig]:
    """Get predefined preset configurations."""
    presets = {}
    
    # Normal training presets
    for dataset in ["hcrl_sa", "hcrl_ch", "set_01", "set_02", "set_03", "set_04"]:
        presets[f"gat_normal_{dataset}"] = create_gat_normal_config(dataset)
        presets[f"autoencoder_{dataset}"] = create_autoencoder_config(dataset)
    
    # Knowledge distillation presets (requires teacher path to be set)
    for dataset in ["hcrl_sa", "hcrl_ch"]:
        for scale in [0.25, 0.5, 0.75]:
            name = f"distillation_{dataset}_scale_{scale}"
            presets[name] = create_distillation_config(
                dataset=dataset, 
                student_scale=scale,
                teacher_model_path=f"saved_models/best_teacher_model_{dataset}.pth"
            )
    
    # Fusion presets
    for dataset in ["hcrl_sa", "hcrl_ch"]:
        presets[f"fusion_{dataset}"] = create_fusion_config(dataset)
    
    return presets


def list_presets():
    """List available preset configurations."""
    presets = get_preset_configs()
    
    print("📋 Available preset configurations:")
    print("=" * 50)
    
    categories = {
        "Normal Training": [k for k in presets.keys() if k.startswith("gat_normal")],
        "Autoencoder Training": [k for k in presets.keys() if k.startswith("autoencoder")],
        "Knowledge Distillation": [k for k in presets.keys() if k.startswith("distillation")],
        "Fusion Training": [k for k in presets.keys() if k.startswith("fusion")]
    }
    
    for category, preset_names in categories.items():
        if preset_names:
            print(f"\n{category}:")
            for name in preset_names:
                print(f"  - {name}")


# ============================================================================
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
  
  # Using preset
  python train_with_hydra_zen.py --preset gat_normal_hcrl_sa
  
  # List presets
  python train_with_hydra_zen.py --list-presets
        """
    )
    
    # Preset mode
    parser.add_argument('--preset', type=str,
                      help='Use a preset configuration')
    parser.add_argument('--list-presets', action='store_true',
                      help='List available preset configurations')
    
    # Manual configuration
    parser.add_argument('--model', type=str, choices=['gat', 'vgae'], default='gat',
                      help='Model type')
    parser.add_argument('--dataset', type=str, 
                      choices=['hcrl_sa', 'hcrl_ch', 'set_01', 'set_02', 'set_03', 'set_04', 'car_hacking'],
                      default='hcrl_sa', help='Dataset name')
    parser.add_argument('--training', type=str, 
                      choices=['normal', 'autoencoder', 'knowledge_distillation', 'fusion'],
                      default='normal', help='Training mode')
    
    # Knowledge distillation specific
    parser.add_argument('--teacher_path', type=str,
                      help='Path to teacher model (required for knowledge distillation)')
    parser.add_argument('--student_scale', type=float, default=1.0,
                      help='Student model scale factor')
    parser.add_argument('--distillation_alpha', type=float, default=0.7,
                      help='Distillation alpha weight')
    parser.add_argument('--temperature', type=float, default=4.0,
                      help='Distillation temperature')
    
    # Training overrides
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--tensorboard', action='store_true',
                      help='Enable TensorBoard logging')
    
    args = parser.parse_args()
    
    if args.list_presets:
        list_presets()
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
    
    else:
        # Manual configuration
        store_manager = CANGraphConfigStore()
        
        # Prepare overrides
        overrides = {}
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
        
        config = store_manager.create_config(args.model, args.dataset, args.training, **overrides)
    
    # Apply global overrides
    if args.tensorboard:
        config.logging["enable_tensorboard"] = True
    
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