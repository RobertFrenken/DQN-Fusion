"""
Fusion Training Mode - DQN Agent for Multi-Model Decision Fusion

Combines VGAE anomaly detection and GAT classification using
reinforcement learning (DQN) for adaptive decision making.
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from src.training.lightning_modules import FusionLightningModule, FusionPredictionCache, CANGraphLightningModule
from src.training.prediction_cache import create_fusion_prediction_cache
from src.training.datamodules import load_dataset, create_dataloaders
from src.training.model_manager import ModelManager
from src.paths import PathResolver

logger = logging.getLogger(__name__)


class FusionTrainer:
    """Handles fusion agent training with cached VGAE + GAT predictions."""
    
    def __init__(self, config, paths: dict):
        """
        Initialize fusion trainer.
        
        Args:
            config: CANGraphConfig with fusion_agent_config
            paths: Dict with experiment directories
        """
        self.config = config
        self.paths = paths
        self.model_manager = ModelManager()
    
    def train(self) -> Tuple[FusionLightningModule, pl.Trainer]:
        """
        Execute fusion training pipeline.
        
        Returns:
            Tuple of (fusion_model, trainer)
        """
        logger.info("🔀 Training fusion agent with cached predictions")
        
        # Validate fusion config
        if not hasattr(self.config.training, 'fusion_agent_config'):
            raise ValueError(
                "Fusion config missing. Use create_fusion_config() "
                "or set fusion_agent_config"
            )
        
        fusion_cfg = self.config.training.fusion_agent_config
        
        # Load pre-trained models
        ae_path, classifier_path = self._resolve_pretrained_paths()
        
        # Load dataset
        train_dataset, val_dataset, num_ids = load_dataset(
            self.config.dataset.name, 
            self.config
        )
        
        train_loader, val_loader = create_dataloaders(
            train_dataset, 
            val_dataset, 
            batch_size=64  # Smaller batch for prediction extraction
        )
        
        # Load model weights and build prediction caches
        train_fusion_dataset, val_fusion_dataset = self._build_prediction_caches(
            ae_path, 
            classifier_path,
            train_loader,
            val_loader,
            num_ids
        )
        
        # Create fusion dataloaders
        fusion_train_loader, fusion_val_loader = self._create_fusion_dataloaders(
            train_fusion_dataset,
            val_fusion_dataset,
            fusion_cfg
        )
        
        # Create fusion Lightning module
        fusion_model = self._create_fusion_model(fusion_cfg, num_ids)
        
        # Setup trainer
        trainer = self._create_trainer()
        
        # Train
        logger.info("🚀 Starting fusion training")
        trainer.fit(fusion_model, fusion_train_loader, fusion_val_loader)
        
        # Validate
        logger.info("📊 Running validation")
        val_results = trainer.validate(fusion_model, fusion_val_loader, verbose=True)
        logger.info(f"Validation results: {val_results}")
        
        # Save fusion agent
        self._save_fusion_agent(fusion_model)
        
        logger.info("✅ Fusion training completed successfully!")
        return fusion_model, trainer
    
    def _resolve_pretrained_paths(self) -> Tuple[Path, Path]:
        """Resolve paths to pre-trained VGAE and GAT models."""
        logger.info("📦 Loading pre-trained models for prediction caching")
        
        # Use PathResolver for unified artifact resolution
        path_resolver = PathResolver(self.config)
        ae_path = path_resolver.resolve_autoencoder_path()
        classifier_path = path_resolver.resolve_classifier_path()
        
        # Fallback to config artifacts if resolver returns None
        if not ae_path or not classifier_path:
            artifacts = self.config.required_artifacts()
            ae_path = ae_path or artifacts.get('autoencoder')
            classifier_path = classifier_path or artifacts.get('classifier')

        # Validate existence
        missing = []
        if not ae_path or not Path(ae_path).exists():
            missing.append(f"autoencoder missing at {ae_path}")
        if not classifier_path or not Path(classifier_path).exists():
            missing.append(f"classifier missing at {classifier_path}")
            
        if missing:
            raise FileNotFoundError(
                "Fusion training requires pre-trained artifacts:\n" + 
                "\n".join(missing) + 
                "\nPlease ensure the artifacts are available at the canonical paths."
            )

        logger.info(f"  Autoencoder: {ae_path}")
        logger.info(f"  Classifier: {classifier_path}")
        
        return Path(ae_path), Path(classifier_path)
    
    def _build_prediction_caches(
        self,
        ae_path: Path,
        classifier_path: Path,
        train_loader,
        val_loader,
        num_ids: int
    ) -> Tuple[FusionPredictionCache, FusionPredictionCache]:
        """Build prediction caches from pre-trained models."""
        
        # Load checkpoints
        ae_ckpt = self.model_manager.load_state_dict(ae_path)
        classifier_ckpt = self.model_manager.load_state_dict(classifier_path)
        
        # Import config module for model instantiation
        from importlib import import_module
        _cfg_mod = import_module('src.config.hydra_zen_configs')
        
        # Create model architectures
        ae_module = CANGraphLightningModule(
            model_config=_cfg_mod.VGAEConfig(),
            training_config=self.config.training,
            model_type='vgae',
            training_mode='autoencoder',
            num_ids=num_ids
        )
        ae_model = ae_module.model
        
        classifier_module = CANGraphLightningModule(
            model_config=_cfg_mod.GATConfig(),
            training_config=self.config.training,
            model_type='gat',
            training_mode='normal',
            num_ids=num_ids
        )
        classifier_model = classifier_module.model
        
        # Load weights
        ae_model.load_state_dict(ae_ckpt)
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
        
        return train_fusion_dataset, val_fusion_dataset
    
    def _create_fusion_dataloaders(
        self,
        train_dataset: FusionPredictionCache,
        val_dataset: FusionPredictionCache,
        fusion_cfg
    ) -> Tuple[DataLoader, DataLoader]:
        """Create dataloaders for fusion training."""
        
        # Use SLURM allocation or default to 8 workers
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_CPUS_ON_NODE')
        num_workers = int(slurm_cpus) if slurm_cpus else min(os.cpu_count() or 1, 8)
        
        fusion_train_loader = DataLoader(
            train_dataset,
            batch_size=fusion_cfg.fusion_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        
        fusion_val_loader = DataLoader(
            val_dataset,
            batch_size=fusion_cfg.fusion_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        
        logger.info(f"✓ Fusion dataloaders created (batch size: {fusion_cfg.fusion_batch_size})")
        
        return fusion_train_loader, fusion_val_loader
    
    def _create_fusion_model(self, fusion_cfg, num_ids: int) -> FusionLightningModule:
        """Create fusion Lightning module."""
        logger.info("⚙️  Creating fusion Lightning module")
        
        return FusionLightningModule(
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
    
    def _create_trainer(self) -> pl.Trainer:
        """Setup Lightning trainer for fusion."""
        logger.info("🏋️  Setting up Lightning trainer for fusion")
        
        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.paths['checkpoint_dir']),
            filename=f'{self.config.model.type}_{self.config.training.mode}_{{epoch:02d}}_{{val_accuracy:.3f}}',
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
            save_dir=str(self.paths['log_dir']),
            name=f'fusion_{self.config.dataset.name}'
        )
        
        # Create trainer
        return pl.Trainer(
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            max_epochs=self.config.training.max_epochs,
            callbacks=[checkpoint_callback, early_stop],
            logger=csv_logger,
            log_every_n_steps=10,
            enable_progress_bar=True
        )
    
    def _save_fusion_agent(self, fusion_model: FusionLightningModule):
        """Save fusion agent with fallback to compact state dict."""
        agent_path = self.paths['model_save_dir'] / f'fusion_agent_{self.config.dataset.name}.pth'
        
        # Save using agent's built-in method
        fusion_model.fusion_agent.save_agent(str(agent_path))
        logger.info(f"✓ Fusion agent saved to {agent_path}")

        # Also save compact DQN state dict for compatibility
        try:
            agent_state_name = f"dqn_agent_{self.config.dataset.name}.pth"
            self.model_manager.save_state_dict(
                fusion_model, 
                self.paths['model_save_dir'], 
                agent_state_name
            )
        except Exception as e:
            logger.warning(f"Could not save compact DQN agent state dict: {e}")
