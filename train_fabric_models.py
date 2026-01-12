"""
PyTorch Lightning Fabric Individual Model Training Script

A comprehensive training script that replaces the original PyTorch training with
PyTorch Lightning Fabric for better performance, scalability, and HPC integration.

Features:
- Fabric-based training with mixed precision
- Dynamic batch sizing
- Optimized data loading
- SLURM integration
- Knowledge distillation support
- Advanced logging and checkpointing

Usage:
    python train_fabric_models.py --model gat --dataset hcrl_ch --type teacher
    python train_fabric_models.py --model vgae --dataset set_01 --type student --teacher saved_models/best_vgae_model_hcrl_sa.pth
    python train_fabric_models.py --list-configs  # Show available configurations
    python train_fabric_models.py --optimize-batch-size --dataset hcrl_sa --model gat
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Fabric trainers
from src.training.fabric_gat_trainer import FabricGATTrainer, FabricGATKnowledgeDistillationTrainer
from src.training.fabric_vgae_trainer import FabricVGAETrainer, FabricVGAEKnowledgeDistillationTrainer

# Utilities
from src.utils.fabric_utils import setup_slurm_fabric_config
from src.utils.fabric_dataloader import FabricDataLoaderFactory, optimize_dataloader_for_hardware
from src.utils.fabric_slurm import FabricSlurmTrainingOrchestrator

# Data preprocessing
from src.preprocessing.preprocessing import graph_creation, GraphDataset
from torch.utils.data import random_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fabric_training.log')
    ]
)
logger = logging.getLogger(__name__)


class FabricModelTrainer:
    """
    Main trainer class that orchestrates Fabric-based model training.
    """
    
    def __init__(self):
        self.available_datasets = ['hcrl_ch', 'hcrl_sa', 'set_01', 'set_02', 'set_03', 'set_04']
        self.available_models = ['gat', 'vgae']
        self.available_types = ['teacher', 'student']
        
        # Results tracking
        self.results_dir = Path("outputs/fabric_training_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base configurations
        self.base_config = self._load_base_config()
        self.hardware_config = optimize_dataloader_for_hardware()
        
        logger.info("FabricModelTrainer initialized")
    
    def _load_base_config(self) -> DictConfig:
        """Load base configuration from YAML."""
        config_path = Path("conf/base.yaml")
        if config_path.exists():
            return OmegaConf.load(config_path)
        else:
            # Default configuration
            return OmegaConf.create({
                'epochs': 100,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'batch_size': 32,
                'train_ratio': 0.8,
                'optimizer': 'adamw',
                'scheduler': {
                    'type': 'cosine',
                    'T_max': 100
                },
                'loss': {
                    'type': 'bce'
                }
            })
    
    def list_available_options(self):
        """Display all available training options."""
        print("🎯 Available Fabric Training Options:")
        print(f"\n📊 Datasets ({len(self.available_datasets)}):")
        for dataset in self.available_datasets:
            print(f"  - {dataset}")
        
        print(f"\n🤖 Models ({len(self.available_models)}):")
        for model in self.available_models:
            print(f"  - {model}")
        
        print(f"\n🎓 Types ({len(self.available_types)}):")
        for model_type in self.available_types:
            print(f"  - {model_type}")
        
        # Show hardware configuration
        print(f"\n💻 Hardware Configuration:")
        for key, value in self.hardware_config.items():
            print(f"  - {key}: {value}")
    
    def get_model_config(self, model: str, model_type: str, dataset: str) -> Dict[str, Any]:
        """
        Get model configuration based on model type and dataset.
        
        Args:
            model: Model name ('gat' or 'vgae')
            model_type: Model type ('teacher' or 'student')
            dataset: Dataset name
            
        Returns:
            Model configuration dictionary
        """
        # Determine number of unique CAN IDs based on dataset
        num_ids_map = {
            'hcrl_ch': 2000,
            'hcrl_sa': 2000,
            'set_01': 2000,
            'set_02': 2000,
            'set_03': 2000,
            'set_04': 2000
        }
        
        num_ids = num_ids_map.get(dataset, 2000)
        in_channels = 11  # Standard for CAN data
        
        if model == 'gat':
            if model_type == 'teacher':
                return {
                    'num_ids': num_ids,
                    'in_channels': in_channels,
                    'hidden_channels': 64,
                    'out_channels': 1,
                    'num_layers': 5,
                    'heads': 8,
                    'dropout': 0.2,
                    'num_fc_layers': 3,
                    'embedding_dim': 16
                }
            else:  # student
                return {
                    'num_ids': num_ids,
                    'in_channels': in_channels,
                    'hidden_channels': 32,
                    'out_channels': 1,
                    'num_layers': 3,
                    'heads': 4,
                    'dropout': 0.2,
                    'num_fc_layers': 2,
                    'embedding_dim': 8
                }
        
        elif model == 'vgae':
            if model_type == 'teacher':
                return {
                    'num_ids': num_ids,
                    'in_channels': in_channels,
                    'hidden_dim': 64,
                    'latent_dim': 32,
                    'num_encoder_layers': 4,
                    'num_decoder_layers': 4,
                    'encoder_heads': 8,
                    'decoder_heads': 4,
                    'embedding_dim': 16,
                    'dropout': 0.3
                }
            else:  # student
                return {
                    'num_ids': num_ids,
                    'in_channels': in_channels,
                    'hidden_dim': 32,
                    'latent_dim': 16,
                    'num_encoder_layers': 2,
                    'num_decoder_layers': 2,
                    'encoder_heads': 4,
                    'decoder_heads': 2,
                    'embedding_dim': 8,
                    'dropout': 0.25
                }
        
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def get_training_config(self, model: str, model_type: str) -> Dict[str, Any]:
        """
        Get training configuration.
        
        Args:
            model: Model name
            model_type: Model type
            
        Returns:
            Training configuration dictionary
        """
        base_config = OmegaConf.to_container(self.base_config, resolve=True)
        
        # Adjust learning rate based on model type
        if model_type == 'student':
            base_config['learning_rate'] = base_config.get('learning_rate', 1e-3) * 0.5
            base_config['epochs'] = min(base_config.get('epochs', 100), 80)
        
        # Model-specific adjustments
        if model == 'vgae':
            # VGAE-specific loss configuration
            base_config['loss'] = {
                'reconstruction_weight': 1.0,
                'kl_weight': 0.1,
                'canid_weight': 1.0,
                'neighborhood_weight': 0.5
            }
            base_config['batch_size'] = min(base_config.get('batch_size', 32), 64)
        
        # Add hardware-optimized settings
        base_config.update({
            'gradient_clip_val': 1.0,
            'accumulation_steps': 1,
            'checkpoint_interval': 10,
            'target_memory_usage': 0.85,
            'min_batch_size': 1,
            'max_batch_size': 8192 if model == 'gat' else 4096
        })
        
        return base_config
    
    def load_dataset(self, dataset_name: str) -> Tuple[GraphDataset, GraphDataset]:
        """
        Load and split dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset using existing preprocessing
        dataset_path = f"datasets/can-train-and-test-v1.5/{dataset_name}"
        if not Path(dataset_path).exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")
        
        # Create graph dataset
        graphs = graph_creation(
            root_folder=dataset_name,
            datasize=self.base_config.get('datasize', 1.0)
        )
        
        full_dataset = GraphDataset(graphs)
        
        # Split dataset
        train_ratio = self.base_config.get('train_ratio', 0.8)
        train_size = int(len(full_dataset) * train_ratio)
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
        return train_dataset, val_dataset
    
    def optimize_batch_size(self, dataset_name: str, model: str) -> int:
        """
        Optimize batch size for the given dataset and model.
        
        Args:
            dataset_name: Dataset name
            model: Model name
            
        Returns:
            Optimal batch size
        """
        logger.info(f"Optimizing batch size for {model} on {dataset_name}")
        
        # Load small sample of dataset
        train_dataset, _ = self.load_dataset(dataset_name)
        sample_data = train_dataset[0] if hasattr(train_dataset, '__getitem__') else train_dataset.dataset[0]
        
        # Get model configuration
        model_config = self.get_model_config(model, 'teacher', dataset_name)
        training_config = self.get_training_config(model, 'teacher')
        
        # Create trainer for batch size estimation
        if model == 'gat':
            trainer = FabricGATTrainer(
                model_config=model_config,
                training_config=training_config,
                use_dynamic_batching=True
            )
        elif model == 'vgae':
            trainer = FabricVGAETrainer(
                model_config=model_config,
                training_config=training_config,
                use_dynamic_batching=True
            )
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Determine optimal batch size
        optimal_size = trainer.determine_optimal_batch_size(sample_data)
        
        logger.info(f"Optimal batch size determined: {optimal_size}")
        return optimal_size
    
    def train_model(self,
                   model: str,
                   dataset_name: str,
                   model_type: str,
                   teacher_model_path: str = None,
                   custom_config: Dict[str, Any] = None,
                   use_slurm: bool = False) -> Dict[str, Any]:
        """
        Train a model with Fabric.
        
        Args:
            model: Model name ('gat' or 'vgae')
            dataset_name: Dataset name
            model_type: Model type ('teacher' or 'student')
            teacher_model_path: Path to teacher model (for knowledge distillation)
            custom_config: Custom configuration overrides
            use_slurm: Whether to submit to SLURM
            
        Returns:
            Training results
        """
        logger.info(f"Starting {model} {model_type} training on {dataset_name}")
        
        # Check if this is knowledge distillation
        is_distillation = (model_type == 'student' and teacher_model_path)
        
        # Get configurations
        model_config = self.get_model_config(model, model_type, dataset_name)
        training_config = self.get_training_config(model, model_type)
        
        # Apply custom configuration
        if custom_config:
            training_config.update(custom_config)
        
        # Setup Fabric configuration
        fabric_config = setup_slurm_fabric_config() if use_slurm else {
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'precision': '16-mixed' if torch.cuda.is_available() else '32'
        }
        
        # Load dataset
        train_dataset, val_dataset = self.load_dataset(dataset_name)
        
        # Create optimized data loaders
        batch_size = training_config.get('batch_size', 32)
        
        if model == 'gat':
            train_loader = FabricDataLoaderFactory.create_gat_dataloader(
                train_dataset, 
                batch_size=batch_size,
                split='train',
                **self.hardware_config
            )
            val_loader = FabricDataLoaderFactory.create_gat_dataloader(
                val_dataset,
                batch_size=batch_size,
                split='val',
                **self.hardware_config
            )
        else:  # vgae
            train_loader = FabricDataLoaderFactory.create_vgae_dataloader(
                train_dataset,
                batch_size=batch_size,
                split='train',
                **self.hardware_config
            )
            val_loader = FabricDataLoaderFactory.create_vgae_dataloader(
                val_dataset,
                batch_size=batch_size,
                split='val',
                **self.hardware_config
            )
        
        # Create trainer
        if is_distillation:
            # Knowledge distillation trainer
            distillation_config = {
                'alpha': 0.7,
                'temperature': 3.0,
                'latent_weight': 0.5,
                'output_weight': 0.3
            }
            
            if model == 'gat':
                trainer = FabricGATKnowledgeDistillationTrainer(
                    student_config=model_config,
                    teacher_model_path=teacher_model_path,
                    training_config=training_config,
                    distillation_config=distillation_config,
                    fabric_config=fabric_config
                )
            else:  # vgae
                trainer = FabricVGAEKnowledgeDistillationTrainer(
                    student_config=model_config,
                    teacher_model_path=teacher_model_path,
                    training_config=training_config,
                    distillation_config=distillation_config,
                    fabric_config=fabric_config
                )
        else:
            # Standard trainer
            if model == 'gat':
                trainer = FabricGATTrainer(
                    model_config=model_config,
                    training_config=training_config,
                    fabric_config=fabric_config
                )
            else:  # vgae
                trainer = FabricVGAETrainer(
                    model_config=model_config,
                    training_config=training_config,
                    fabric_config=fabric_config
                )
        
        # Setup checkpoint directory
        checkpoint_dir = self.results_dir / f"{model}_{model_type}_{dataset_name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Train model
        start_time = time.time()
        
        try:
            training_history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=training_config.get('epochs', 100),
                save_best=True,
                checkpoint_dir=str(checkpoint_dir)
            )
            
            training_time = time.time() - start_time
            
            # Save results
            results = {
                'model': model,
                'dataset': dataset_name,
                'model_type': model_type,
                'is_distillation': is_distillation,
                'training_time': training_time,
                'model_config': model_config,
                'training_config': training_config,
                'training_history': training_history,
                'checkpoint_dir': str(checkpoint_dir)
            }
            
            results_file = checkpoint_dir / "training_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Training completed successfully in {training_time:.2f} seconds")
            logger.info(f"Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning Fabric Individual Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', 
                      choices=['gat', 'vgae'], 
                      help='Model type to train')
    
    parser.add_argument('--dataset',
                      choices=['hcrl_ch', 'hcrl_sa', 'set_01', 'set_02', 'set_03', 'set_04'],
                      help='Dataset to use for training')
    
    parser.add_argument('--type',
                      choices=['teacher', 'student'],
                      help='Model type (teacher or student)')
    
    parser.add_argument('--teacher',
                      type=str,
                      help='Path to teacher model for knowledge distillation')
    
    parser.add_argument('--config',
                      type=str,
                      help='JSON string with custom configuration overrides')
    
    parser.add_argument('--use-slurm',
                      action='store_true',
                      help='Configure for SLURM environment')
    
    parser.add_argument('--list-configs',
                      action='store_true',
                      help='List available configurations and exit')
    
    parser.add_argument('--optimize-batch-size',
                      action='store_true',
                      help='Optimize batch size and exit')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FabricModelTrainer()
    
    # Handle special commands
    if args.list_configs:
        trainer.list_available_options()
        return
    
    if args.optimize_batch_size:
        if not args.dataset or not args.model:
            print("Error: --dataset and --model are required for batch size optimization")
            return
        
        optimal_size = trainer.optimize_batch_size(args.dataset, args.model)
        print(f"Optimal batch size for {args.model} on {args.dataset}: {optimal_size}")
        return
    
    # Validate required arguments
    if not all([args.model, args.dataset, args.type]):
        parser.error("--model, --dataset, and --type are required")
    
    # Validate teacher model for student training
    if args.type == 'student' and not args.teacher:
        print("Warning: Training student model without teacher (no knowledge distillation)")
    
    # Parse custom configuration
    custom_config = {}
    if args.config:
        try:
            custom_config = json.loads(args.config)
        except json.JSONDecodeError as e:
            parser.error(f"Invalid JSON in --config: {e}")
    
    try:
        # Train model
        results = trainer.train_model(
            model=args.model,
            dataset_name=args.dataset,
            model_type=args.type,
            teacher_model_path=args.teacher,
            custom_config=custom_config,
            use_slurm=args.use_slurm
        )
        
        # Print summary
        print("\n🎉 Training Summary:")
        print(f"Model: {results['model']} {results['model_type']}")
        print(f"Dataset: {results['dataset']}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Knowledge distillation: {results['is_distillation']}")
        print(f"Results saved to: {results['checkpoint_dir']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()