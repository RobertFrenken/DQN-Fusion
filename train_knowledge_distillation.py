"""
CAN-Graph Knowledge Distillation Training Script

This script provides a streamlined interface for knowledge distillation experiments.
It automatically handles:
1. Teacher model loading and validation
2. Student model scaling  
3. Memory optimization for distillation
4. Comprehensive teacher-student performance comparison
5. Proper checkpointing and model saving

Usage:
python train_knowledge_distillation.py \
    --teacher_path saved_models/best_teacher_model_hcrl_sa.pth \
    --dataset hcrl_sa \
    --student_scale 0.5 \
    --distillation_alpha 0.7 \
    --temperature 4.0

Or with Hydra config:
python train_knowledge_distillation.py \
    model=gat dataset=hcrl_sa training=knowledge_distillation \
    training.teacher_model_path=saved_models/best_teacher_model_hcrl_sa.pth \
    training.student_model_scale=0.5
"""

import os
import sys
from pathlib import Path
import logging
import argparse
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf

# Suppress warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from train_models import CANGraphLightningModule, load_dataset, create_dataloaders, CANGraphDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistillationTrainingManager:
    """Manages knowledge distillation training with enhanced features."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.teacher_model = None
        self.student_model = None
        self.teacher_performance = {}
        
    def validate_teacher_model(self, teacher_path: str, num_ids: int) -> dict:
        """Validate and analyze teacher model performance."""
        logger.info(f"Validating teacher model: {teacher_path}")
        
        if not Path(teacher_path).exists():
            raise FileNotFoundError(f"Teacher model not found: {teacher_path}")
        
        # Create teacher model for validation
        teacher_config = self.config.model.copy()
        teacher_model = CANGraphLightningModule(
            model_config=teacher_config,
            training_config=self.config.training,
            model_type=self.config.model.type,
            training_mode="normal",  # Teacher always in normal mode
            num_ids=num_ids
        )
        
        # Load teacher weights
        teacher_state = torch.load(teacher_path, map_location='cpu')
        if 'state_dict' in teacher_state:
            state_dict = teacher_state['state_dict']
            state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v 
                         for k, v in state_dict.items()}
        else:
            state_dict = teacher_state
            
        teacher_model.load_state_dict(state_dict)
        
        # Get model info
        total_params = sum(p.numel() for p in teacher_model.parameters())
        trainable_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
        
        teacher_info = {
            'path': teacher_path,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # Assuming float32
            'architecture': self.config.model.type
        }
        
        logger.info(f"Teacher model validation complete:")
        logger.info(f"  - Parameters: {total_params:,} ({trainable_params:,} trainable)")
        logger.info(f"  - Model size: {teacher_info['model_size_mb']:.2f} MB")
        
        return teacher_info
    
    def setup_student_model(self, num_ids: int, teacher_info: dict) -> CANGraphLightningModule:
        """Setup student model with optional scaling."""
        student_scale = self.config.training.get('student_model_scale', 1.0)
        
        logger.info(f"Setting up student model (scale: {student_scale})")
        
        student_model = CANGraphLightningModule(
            model_config=self.config.model,
            training_config=self.config.training,
            model_type=self.config.model.type,
            training_mode="knowledge_distillation",
            num_ids=num_ids
        )
        
        # Get student model info
        student_params = sum(p.numel() for p in student_model.parameters())
        student_size_mb = student_params * 4 / 1024 / 1024
        
        compression_ratio = teacher_info['total_parameters'] / student_params
        
        logger.info(f"Student model created:")
        logger.info(f"  - Parameters: {student_params:,}")
        logger.info(f"  - Model size: {student_size_mb:.2f} MB")
        logger.info(f"  - Compression ratio: {compression_ratio:.2f}x")
        
        return student_model
    
    def create_distillation_trainer(self) -> pl.Trainer:
        """Create Lightning trainer optimized for knowledge distillation."""
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath='saved_models/distillation_checkpoints',
                filename=f'student_{self.config.model.type}_{self.config.dataset.name}_{{epoch:02d}}_{{val_loss:.3f}}',
                save_top_k=3,
                monitor='val_loss',
                mode='min',
                save_last=True,
                auto_insert_metric_name=False
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.training.get('early_stopping_patience', 12),
                mode='min',
                verbose=True
            )
        ]
        
        # Setup loggers
        loggers = [
            CSVLogger(
                save_dir='outputs/distillation_logs',
                name=f"distillation_{self.config.model.type}_{self.config.dataset.name}"
            )
        ]
        
        if self.config.get('logging', {}).get('enable_tensorboard', False):
            loggers.append(
                TensorBoardLogger(
                    save_dir='outputs/distillation_logs',
                    name=f"distillation_{self.config.model.type}_{self.config.dataset.name}"
                )
            )
        
        # Create trainer with distillation-optimized settings
        trainer = pl.Trainer(
            accelerator='auto',
            devices='auto',
            precision=self.config.training.get('precision', '16-mixed'),  # Mixed precision for memory
            max_epochs=self.config.training.max_epochs,
            gradient_clip_val=self.config.training.get('gradient_clip_val', 1.0),
            accumulate_grad_batches=self.config.training.get('accumulate_grad_batches', 2),
            logger=loggers,
            callbacks=callbacks,
            enable_checkpointing=True,
            log_every_n_steps=self.config.training.get('log_every_n_steps', 50),
            enable_progress_bar=True,
            num_sanity_val_steps=2
        )
        
        return trainer
    
    def run_distillation(self):
        """Run complete knowledge distillation pipeline."""
        logger.info("Starting knowledge distillation training")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(self.config)}")
        
        # Load dataset
        train_dataset, val_dataset, num_ids = load_dataset(self.config.dataset.name, self.config)
        
        # Validate teacher model
        teacher_info = self.validate_teacher_model(
            self.config.training.teacher_model_path, num_ids
        )
        
        # Setup student model
        student_model = self.setup_student_model(num_ids, teacher_info)
        
        # Optimize batch size for distillation (both teacher and student in memory)
        initial_batch_size = self.config.training.batch_size
        if self.config.training.get('optimize_batch_size', True):
            # Conservative batch size for distillation
            optimized_batch_size = max(initial_batch_size // 2, 8)
            logger.info(f"Adjusted batch size for distillation: {initial_batch_size} -> {optimized_batch_size}")
            student_model.batch_size = optimized_batch_size
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, student_model.batch_size
        )
        
        # Create trainer
        trainer = self.create_distillation_trainer()
        
        # Train student model
        logger.info("Starting distillation training...")
        trainer.fit(student_model, train_loader, val_loader)
        
        # Test performance
        if self.config.training.get('run_test', True):
            test_results = trainer.test(student_model, val_loader)
            logger.info(f"Student test results: {test_results}")
        
        # Save final student model
        final_model_name = f"final_student_model_{self.config.dataset.name}.pth"
        save_path = Path("saved_models") / final_model_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(student_model.state_dict(), save_path)
        logger.info(f"Final student model saved to: {save_path}")
        
        # Performance summary
        self.print_distillation_summary(teacher_info, student_model, test_results if 'test_results' in locals() else None)
        
    def print_distillation_summary(self, teacher_info: dict, student_model: CANGraphLightningModule, test_results=None):
        """Print comprehensive distillation results summary."""
        student_params = sum(p.numel() for p in student_model.parameters())
        compression_ratio = teacher_info['total_parameters'] / student_params
        
        logger.info("\n" + "="*60)
        logger.info("KNOWLEDGE DISTILLATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Teacher Model:")
        logger.info(f"  - Path: {teacher_info['path']}")
        logger.info(f"  - Parameters: {teacher_info['total_parameters']:,}")
        logger.info(f"  - Size: {teacher_info['model_size_mb']:.2f} MB")
        logger.info(f"")
        logger.info(f"Student Model:")
        logger.info(f"  - Parameters: {student_params:,}")
        logger.info(f"  - Size: {student_params * 4 / 1024 / 1024:.2f} MB")
        logger.info(f"  - Compression: {compression_ratio:.2f}x smaller")
        logger.info(f"")
        logger.info(f"Training Configuration:")
        logger.info(f"  - Temperature: {self.config.training.get('distillation_temperature', 4.0)}")
        logger.info(f"  - Alpha: {self.config.training.get('distillation_alpha', 0.7)}")
        logger.info(f"  - Student scale: {self.config.training.get('student_model_scale', 1.0)}")
        logger.info(f"  - Max epochs: {self.config.training.max_epochs}")
        
        if test_results:
            logger.info(f"")
            logger.info(f"Final Performance:")
            logger.info(f"  - Test loss: {test_results[0].get('test_loss', 'N/A')}")
        
        logger.info("="*60)


def create_distillation_config(args) -> DictConfig:
    """Create configuration for distillation from command line arguments."""
    config = OmegaConf.create({
        'model': {
            'type': args.model_type,
            'gat': {
                'input_dim': 8,
                'hidden_channels': 64,
                'output_dim': 2,
                'num_layers': 3,
                'heads': 4,
                'dropout': 0.2
            }
        },
        'dataset': {
            'name': args.dataset,
            'data_path': f"datasets/{args.dataset}"
        },
        'training': {
            'mode': 'knowledge_distillation',
            'max_epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'teacher_model_path': args.teacher_path,
            'distillation_temperature': args.temperature,
            'distillation_alpha': args.distillation_alpha,
            'student_model_scale': args.student_scale,
            'optimize_batch_size': True,
            'early_stopping_patience': 15,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 2,
            'precision': '16-mixed',
            'log_every_n_steps': 50,
            'run_test': True,
            'log_teacher_student_comparison': True,
            'use_scheduler': True,
            'scheduler_type': 'cosine',
            'scheduler_params': {
                'T_max': args.epochs
            },
            'memory_optimization': {
                'use_teacher_cache': True,
                'clear_cache_every_n_steps': 100
            }
        },
        'logging': {
            'enable_tensorboard': args.tensorboard
        }
    })
    
    return config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_hydra(config: DictConfig):
    """Main function using Hydra configuration."""
    if config.training.get('mode') != 'knowledge_distillation':
        logger.warning("Training mode is not 'knowledge_distillation'. Setting it now.")
        config.training.mode = 'knowledge_distillation'
    
    distillation_manager = DistillationTrainingManager(config)
    distillation_manager.run_distillation()


def main_cli():
    """Main function using command line arguments."""
    parser = argparse.ArgumentParser(description='CAN-Graph Knowledge Distillation Training')
    parser.add_argument('--teacher_path', type=str, required=True,
                      help='Path to trained teacher model')
    parser.add_argument('--dataset', type=str, default='hcrl_sa',
                      help='Dataset name (default: hcrl_sa)')
    parser.add_argument('--model_type', type=str, default='gat',
                      help='Model type (default: gat)')
    parser.add_argument('--student_scale', type=float, default=1.0,
                      help='Student model scale factor (default: 1.0)')
    parser.add_argument('--distillation_alpha', type=float, default=0.7,
                      help='Distillation alpha weight (default: 0.7)')
    parser.add_argument('--temperature', type=float, default=4.0,
                      help='Distillation temperature (default: 4.0)')
    parser.add_argument('--epochs', type=int, default=80,
                      help='Number of training epochs (default: 80)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                      help='Learning rate (default: 0.002)')
    parser.add_argument('--tensorboard', action='store_true',
                      help='Enable TensorBoard logging')
    
    args = parser.parse_args()
    
    config = create_distillation_config(args)
    
    distillation_manager = DistillationTrainingManager(config)
    distillation_manager.run_distillation()


if __name__ == "__main__":
    # Check if running with Hydra or command line
    if len(sys.argv) > 1 and any(arg.startswith('--teacher_path') for arg in sys.argv):
        main_cli()
    else:
        main_hydra()