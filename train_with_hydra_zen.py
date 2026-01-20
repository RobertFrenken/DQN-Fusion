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
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, DeviceStatsMonitor
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
from src.training.fusion_lightning import FusionLightningModule
from src.training.prediction_cache import create_fusion_prediction_cache

# Suppress warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch-scatter.*")
warnings.filterwarnings("ignore", message=".*GLIBCXX.*")
warnings.filterwarnings("ignore", message=".*Trying to infer.*batch_size.*")
warnings.filterwarnings("ignore", message=".*Checkpoint directory.*exists.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

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
    
    def get_hierarchical_paths(self):
        """Create hierarchical directory structure: osc_jobs/dataset/model/mode"""
        # Use absolute paths to avoid any path resolution issues
        base_dir = Path.cwd() / "osc_jobs"
        dataset_name = self.config.dataset.name
        model_type = self.config.model.type  # 'gat' or 'vgae'
        training_mode = self.config.training.mode  # 'autoencoder', 'normal', 'curriculum', etc.
        
        # Debug logging
        logger.info(f"Creating paths - CWD: {Path.cwd()}")
        logger.info(f"Dataset: {dataset_name}, Model: {model_type}, Mode: {training_mode}")
        
        # Create the hierarchical path (absolute)
        experiment_dir = base_dir / dataset_name / model_type / training_mode
        logger.info(f"Experiment directory: {experiment_dir}")
        
        # Create directories
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "lightning_checkpoints").mkdir(exist_ok=True)
        (experiment_dir / "mlruns").mkdir(exist_ok=True)
        
        paths = {
            'experiment_dir': experiment_dir.resolve(),
            'model_save_dir': experiment_dir.resolve(),
            'log_dir': experiment_dir.resolve(),
            'checkpoint_dir': (experiment_dir / "lightning_checkpoints").resolve(),
            'mlruns_dir': (experiment_dir / "mlruns").resolve()
        }
        
        # Debug log all paths
        logger.info("Generated hierarchical paths:")
        for key, path in paths.items():
            logger.info(f"  {key}: {path}")
            
        return paths
    
    def setup_model(self, num_ids: int) -> pl.LightningModule:
        """Create the Lightning module from config."""
        if self.config.training.mode == "fusion":
            # Create fusion model with DQN agent
            fusion_config = dict(self.config.training)
            model = FusionLightningModule(fusion_config, num_ids)
            return model
        else:
            # Standard GAT/VGAE models
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
        # Get hierarchical paths
        paths = self.get_hierarchical_paths()
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(paths['checkpoint_dir']),
            filename=f'{self.config.experiment_name}_{{epoch:02d}}_{{val_loss:.3f}}',
            save_top_k=self.config.logging.get("save_top_k", 3),
            monitor=self.config.logging.get("monitor_metric", "val_loss"),
            mode=self.config.logging.get("monitor_mode", "min"),
            save_last=True,
            auto_insert_metric_name=False
        )
        callbacks.append(checkpoint_callback)
        
        # GPU/CPU monitoring - logs essential resource metrics
        # Note: DeviceStatsMonitor automatically detects and logs available metrics
        device_stats = DeviceStatsMonitor()
        callbacks.append(device_stats)
        
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
            save_dir=str(paths['log_dir']),
            name=self.config.experiment_name
        )
        loggers.append(csv_logger)
        
        # MLflow logger with reduced logging
        # Ensure we have an absolute path and log it for debugging
        mlruns_path = paths['mlruns_dir'].resolve()  # resolve() makes it fully absolute
        logger.info(f"Setting up MLflow with path: {mlruns_path}")
        
        mlflow_logger = MLFlowLogger(
            experiment_name=f"CAN-Graph-{self.config.dataset.name}",
            tracking_uri=mlruns_path.as_uri(),  # Use as_uri() for proper file URI format
            log_model=False,  # Disable model artifacts to reduce logging
            # Only log essential metrics: train/val loss, accuracy, epoch time
        )
        loggers.append(mlflow_logger)
        
        # TensorBoard logger
        if self.config.logging.get("enable_tensorboard", False):
            tb_logger = TensorBoardLogger(
                save_dir=str(paths['log_dir']),
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
            enable_progress_bar=False,  # Disable progress bar for cleaner SLURM logs
            num_sanity_val_steps=self.config.trainer.num_sanity_val_steps
        )
        
        return trainer
    
    def train(self):
        """Execute the complete training pipeline."""
        logger.info(f"🚀 Starting training with hydra-zen config")
        logger.info(f"Experiment: {self.config.experiment_name}")
        logger.info(f"Mode: {self.config.training.mode}")
        
        # Check if fusion mode
        if self.config.training.mode == "fusion":
            return self._train_fusion()
        elif self.config.training.mode == "curriculum":
            return self.train_with_curriculum()
        
        # Standard training - Load dataset
        force_rebuild = hasattr(self.config, 'force_rebuild_cache') and self.config.force_rebuild_cache
        train_dataset, val_dataset, num_ids = load_dataset(self.config.dataset.name, self.config, force_rebuild_cache=force_rebuild)
        
        logger.info(f"📊 Dataset loaded: {len(train_dataset)} training + {len(val_dataset)} validation = {len(train_dataset) + len(val_dataset)} total graphs")
        
        # Setup model
        model = self.setup_model(num_ids)
        
        # Optimize batch size if requested (disabled by default to avoid warnings)
        if getattr(self.config.training, 'optimize_batch_size', False):
            logger.info("Running batch size optimization...")
            model = self._optimize_batch_size(model, train_dataset, val_dataset)
        else:
            logger.info(f"Using fixed batch size: {model.batch_size}")
        
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
        paths = self.get_hierarchical_paths()
        model_name = f"{self.config.experiment_name}.pth"
        save_path = paths['model_save_dir'] / model_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to: {save_path}")
        
        logger.info("✅ Training completed successfully!")
        return model, trainer
    
    def _train_fusion(self):
        """Fusion training using cached VGAE and GAT predictions."""
        logger.info("🔀 Training fusion agent with cached predictions")
        
        from src.training.fusion_lightning import FusionLightningModule, FusionPredictionCache
        from src.training.prediction_cache import create_fusion_prediction_cache
        
        # Validate fusion config
        if not hasattr(self.config.training, 'fusion_agent_config'):
            raise ValueError("Fusion config missing. Use create_fusion_config() or set fusion_agent_config")
        
        fusion_cfg = self.config.training.fusion_agent_config
        
        # Load pre-trained models for prediction caching
        logger.info("📦 Loading pre-trained models for prediction caching")
        
        paths = self.get_hierarchical_paths()
        ae_path = getattr(self.config.training, 'autoencoder_path', 
                         paths['model_save_dir'] / f'autoencoder_{self.config.dataset.name}.pth')
        classifier_path = getattr(self.config.training, 'classifier_path',
                                 paths['model_save_dir'] / f'best_teacher_model_{self.config.dataset.name}.pth')
        
        # Check if paths exist
        if not Path(ae_path).exists():
            raise FileNotFoundError(f"Autoencoder not found: {ae_path}")
        if not Path(classifier_path).exists():
            raise FileNotFoundError(f"Classifier not found: {classifier_path}")
        
        logger.info(f"  Autoencoder: {ae_path}")
        logger.info(f"  Classifier: {classifier_path}")
        
        # Load dataset for inference
        train_dataset, val_dataset, num_ids = load_dataset(self.config.dataset.name, self.config)
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, batch_size=64  # Smaller batch for extraction
        )
        
        # Load model checkpoints
        ae_ckpt = torch.load(ae_path, map_location='cpu')
        classifier_ckpt = torch.load(classifier_path, map_location='cpu')
        
        # Create models for extraction
        ae_model = self.setup_model(num_ids).model
        classifier_model = self.setup_model(num_ids).model
        
        # Load weights
        if isinstance(ae_ckpt, dict) and 'state_dict' in ae_ckpt:
            ae_model.load_state_dict(ae_ckpt['state_dict'])
        else:
            ae_model.load_state_dict(ae_ckpt)
        
        if isinstance(classifier_ckpt, dict) and 'state_dict' in classifier_ckpt:
            classifier_model.load_state_dict(classifier_ckpt['state_dict'])
        else:
            classifier_model.load_state_dict(classifier_ckpt)
        
        logger.info("✓ Models loaded")
        
        # Build prediction caches
        logger.info("🔄 Building prediction caches")
        train_anomaly, train_gat, train_labels, val_anomaly, val_gat, val_labels = \
            create_fusion_prediction_cache(
                autoencoder=ae_model,
                classifier=classifier_model,
                train_loader=train_loader,
                val_loader=val_loader,
                dataset_name=self.config.dataset.name,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                cache_dir='cache/fusion'
            )
        
        logger.info(f"✓ Caches built: {len(train_anomaly)} train, {len(val_anomaly)} val samples")
        
        # Create fusion datasets
        train_fusion_dataset = FusionPredictionCache(
            anomaly_scores=train_anomaly,
            gat_probs=train_gat,
            labels=train_labels
        )
        
        val_fusion_dataset = FusionPredictionCache(
            anomaly_scores=val_anomaly,
            gat_probs=val_gat,
            labels=val_labels
        )
        
        # Create fusion dataloaders
        from torch.utils.data import DataLoader
        
        # Use SLURM allocation or default to 8 workers
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_CPUS_ON_NODE')
        num_workers = int(slurm_cpus) if slurm_cpus else min(os.cpu_count() or 1, 8)
        
        fusion_train_loader = DataLoader(
            train_fusion_dataset,
            batch_size=fusion_cfg.fusion_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        
        fusion_val_loader = DataLoader(
            val_fusion_dataset,
            batch_size=fusion_cfg.fusion_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        
        logger.info(f"✓ Fusion dataloaders created (batch size: {fusion_cfg.fusion_batch_size})")
        
        # Create fusion Lightning module
        logger.info("⚙️  Creating fusion Lightning module")
        fusion_model = FusionLightningModule(
            fusion_config={
                'alpha_steps': fusion_cfg.alpha_steps,
                'fusion_lr': fusion_cfg.fusion_lr,
                'gamma': fusion_cfg.gamma,
                'fusion_epsilon': fusion_cfg.fusion_epsilon,
                'fusion_epsilon_decay': fusion_cfg.fusion_epsilon_decay,
                'fusion_min_epsilon': fusion_cfg.fusion_min_epsilon,
                'fusion_buffer_size': fusion_cfg.fusion_buffer_size,
                'fusion_batch_size': fusion_cfg.fusion_batch_size,
                'target_update_freq': fusion_cfg.target_update_freq
            },
            num_ids=num_ids
        )
        
        # Setup trainer for fusion
        logger.info("🏋️  Setting up Lightning trainer for fusion")
        
        # Get hierarchical paths
        paths = self.get_hierarchical_paths()
        
        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(paths['checkpoint_dir']),
            filename=f'fusion_{self.config.dataset.name}_{{epoch:02d}}_{{val_accuracy:.3f}}',
            save_top_k=3,
            monitor='val_accuracy',
            mode='max',
            auto_insert_metric_name=False
        )
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            mode='max',
            verbose=True
        )
        
        # CSV Logger
        csv_logger = CSVLogger(
            save_dir=str(paths['log_dir']),
            name=f'fusion_{self.config.dataset.name}'
        )
        
        # Create trainer
        trainer = pl.Trainer(
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            max_epochs=self.config.training.max_epochs,
            callbacks=[checkpoint_callback, early_stop],
            logger=csv_logger,
            log_every_n_steps=10,
            enable_progress_bar=True
        )
        
        # Train fusion agent
        logger.info("🚀 Starting fusion training")
        trainer.fit(fusion_model, fusion_train_loader, fusion_val_loader)
        
        # Validate
        logger.info("📊 Running validation")
        val_results = trainer.validate(fusion_model, fusion_val_loader, verbose=True)
        logger.info(f"Validation results: {val_results}")
        
        # Save fusion agent
        paths = self.get_hierarchical_paths()
        agent_path = paths['model_save_dir'] / f'fusion_agent_{self.config.dataset.name}.pth'
        fusion_model.fusion_agent.save_agent(str(agent_path))
        logger.info(f"✓ Fusion agent saved to {agent_path}")
        
        logger.info("✅ Fusion training completed successfully!")
        return fusion_model, trainer

    def train_with_curriculum(self):
        """Train GAT with curriculum learning and dynamic hard mining."""
        logger.info("🎓 Starting GAT training with curriculum learning + hard mining")
        
        # Load dataset - keep separate normal/attack for sampling
        force_rebuild = hasattr(self.config, 'force_rebuild_cache') and self.config.force_rebuild_cache
        full_dataset, val_dataset, num_ids = load_dataset(self.config.dataset.name, self.config, force_rebuild_cache=force_rebuild)
        
        # Separate normal and attack graphs
        train_normal = [g for g in full_dataset if g.y.item() == 0]  
        train_attack = [g for g in full_dataset if g.y.item() == 1]
        val_normal = [g for g in val_dataset if g.y.item() == 0]
        val_attack = [g for g in val_dataset if g.y.item() == 1]
        
        logger.info(f"📊 Separated dataset: {len(train_normal)} normal + {len(train_attack)} attack training")
        logger.info(f"📊 Validation: {len(val_normal)} normal + {len(val_attack)} attack")
        
        # Load trained VGAE for hard mining (if available)
        vgae_model = None
        # Look for VGAE model in the hierarchical structure
        vgae_dir = Path("osc_jobs") / self.config.dataset.name / "vgae" / "autoencoder"
        vgae_path = vgae_dir / f"vgae_{self.config.dataset.name}_autoencoder.pth"
        if vgae_path.exists():
            logger.info(f"🔄 Loading trained VGAE from {vgae_path}")
            try:
                from train_models import CANGraphLightningModule
                vgae_checkpoint = torch.load(str(vgae_path), map_location='cpu')
                vgae_model = CANGraphLightningModule.load_from_checkpoint(
                    str(vgae_path), map_location='cpu'
                )
                vgae_model.eval()
                logger.info("✅ VGAE loaded for hard mining")
            except Exception as e:
                logger.warning(f"Could not load VGAE: {e}. Using random sampling.")
        else:
            logger.warning("No trained VGAE found. Using random sampling instead of hard mining.")
        
        # Create enhanced datamodule
        from src.training.enhanced_datamodule import EnhancedCANGraphDataModule, CurriculumCallback
        
        datamodule = EnhancedCANGraphDataModule(
            train_normal=train_normal,
            train_attack=train_attack, 
            val_normal=val_normal,
            val_attack=val_attack,
            vgae_model=vgae_model,
            batch_size=64,  # Starting batch size for Lightning tuner
            num_workers=min(8, os.cpu_count() or 1),
            total_epochs=self.config.training.max_epochs
        )
        
        # Setup GAT model
        model = self.setup_model(num_ids)
        
        # Optimize batch size if requested
        if getattr(self.config.training, 'optimize_batch_size', False):
            logger.info("🔧 Running batch size optimization...")
            model = self._optimize_batch_size_with_datamodule(model, datamodule)
        
        # Setup trainer with curriculum callback
        curriculum_callback = CurriculumCallback()
        trainer = self.setup_curriculum_trainer(extra_callbacks=[curriculum_callback])
        
        # Train model
        logger.info("🚀 Starting curriculum-enhanced training...")
        trainer.fit(model, datamodule=datamodule)
        
        # Save final model
        paths = self.get_hierarchical_paths()
        model_path = paths['model_save_dir'] / f"gat_{self.config.dataset.name}_curriculum.pth"
        trainer.save_checkpoint(model_path)
        logger.info(f"💾 Model saved to {model_path}")
        
        return model, trainer

    def _optimize_batch_size_with_datamodule(self, model, datamodule):
        """Optimize batch size using custom datamodule."""
        logger.info("🔧 Optimizing batch size with curriculum datamodule...")
        
        trainer = pl.Trainer(
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            precision='32-true',
            max_epochs=1,
            enable_checkpointing=False,
            logger=False
        )
        
        from lightning.pytorch.tuner import Tuner
        tuner = Tuner(trainer)
        
        try:
            tuner.scale_batch_size(
                model,
                datamodule=datamodule,
                mode=self.config.training.batch_size_mode,
                steps_per_trial=3,
                init_val=datamodule.batch_size,
                max_trials=getattr(self.config.training, 'max_batch_size_trials', 10)
            )
            
            # Update datamodule batch size
            datamodule.batch_size = model.batch_size
            logger.info(f"✅ Batch size optimized to: {model.batch_size}")
            
        except Exception as e:
            logger.warning(f"Batch size optimization failed: {e}. Using default.")
        
        return model
        
    def setup_curriculum_trainer(self, extra_callbacks=None):
        """Setup trainer with optional extra callbacks for curriculum learning."""
        # Get hierarchical paths
        paths = self.get_hierarchical_paths()
        
        callbacks = [
            ModelCheckpoint(
                dirpath=str(paths['checkpoint_dir']),
                monitor='val_loss',
                filename='gat_curriculum-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=getattr(self.config.training, 'early_stopping_patience', 25),
                mode='min'
            )
        ]
        
        # Add device stats monitor if requested
        if getattr(self.config.trainer, 'enable_device_stats', False):
            callbacks.append(DeviceStatsMonitor())
        
        # Add extra callbacks
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        
        # Setup loggers
        loggers = []
        
        # CSV logger
        csv_logger = CSVLogger(
            save_dir=str(paths['log_dir']),
            name=self.config.experiment_name
        )
        loggers.append(csv_logger)
        
        # MLflow logger
        # Ensure we have an absolute path and log it for debugging
        mlruns_path = paths['mlruns_dir'].resolve()
        logger.info(f"Setting up curriculum MLflow with path: {mlruns_path}")
        
        mlflow_logger = MLFlowLogger(
            experiment_name=f"CAN-Graph-{self.config.dataset.name}",
            tracking_uri=mlruns_path.as_uri(),  # Use as_uri() for proper file URI format
            log_model=False,
        )
        loggers.append(mlflow_logger)
        
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
            enable_progress_bar=False,
            num_sanity_val_steps=self.config.trainer.num_sanity_val_steps
        )
        
        return trainer
    
    def _optimize_batch_size(self, model, train_dataset, val_dataset) -> CANGraphLightningModule:
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
                teacher_model_path=f"osc_jobs/{dataset}/vgae/autoencoder/best_teacher_model_{dataset}.pth"
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
  
  # Fusion training (uses pre-cached predictions)
  python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training fusion
  
  # Using preset
  python train_with_hydra_zen.py --preset gat_normal_hcrl_sa
  
  # Fusion preset
  python train_with_hydra_zen.py --preset fusion_hcrl_sa
  
  # List presets
  python train_with_hydra_zen.py --list-presets
  
  # Alternatively, use dedicated fusion training script:
  python train_fusion_lightning.py --dataset hcrl_sa --max-epochs 50
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
    parser.add_argument('--force-rebuild-cache', action='store_true',
                      help='Force rebuild of cached processed data')
    parser.add_argument('--debug-graph-count', action='store_true',
                      help='Show detailed graph count diagnostics')
    parser.add_argument('--early-stopping-patience', type=int,
                      help='Early stopping patience (default: 25 for normal, 30 for autoencoder)')
    
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
        if args.early_stopping_patience:
            overrides['early_stopping_patience'] = args.early_stopping_patience
        
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