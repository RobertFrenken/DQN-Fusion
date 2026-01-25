"""
Curriculum Learning Mode with Hard Sample Mining

Trains GAT classifier with curriculum learning that progressively
increases difficulty using VGAE-guided hard sample selection.

Updated to use new specialized Lightning modules (VAELightningModule, GATLightningModule).
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import torch
import lightning.pytorch as pl
from omegaconf import DictConfig

from src.training.datamodules import EnhancedCANGraphDataModule, CurriculumCallback, load_dataset
from src.training.lightning_modules import GATLightningModule, VAELightningModule
from src.training.batch_optimizer import BatchSizeOptimizer
from src.training.model_manager import ModelManager
from src.paths import PathResolver

logger = logging.getLogger(__name__)


class CurriculumTrainer:
    """Handles curriculum learning with dynamic hard mining using new module structure."""
    
    def __init__(self, config: DictConfig, paths: dict):
        """
        Initialize curriculum trainer.
        
        Args:
            config: Hydra config with curriculum settings
            paths: Dict with experiment directories
        """
        self.config = config
        self.paths = paths
        self.model_manager = ModelManager()
        self.path_resolver = PathResolver(config)
    
    def train(self, num_ids: int) -> Tuple[GATLightningModule, pl.Trainer]:
        """
        Execute curriculum learning training pipeline.
        
        Args:
            num_ids: Number of unique CAN IDs
            
        Returns:
            Tuple of (trained_model, trainer)
        """
        logger.info("🎓 Starting GAT training with curriculum learning + hard mining")
        
        # Load and separate dataset
        datamodule, vgae_model = self._setup_curriculum_datamodule(num_ids)
        
        # Create GAT model
        gat_model = self._create_gat_model(num_ids)
        
        # Optimize batch size
        gat_model = self._optimize_batch_size_for_curriculum(gat_model, datamodule)
        
        # Setup trainer
        trainer = self._create_curriculum_trainer()
        
        # Train
        logger.info("🚀 Starting curriculum-enhanced training...")
        trainer.fit(gat_model, datamodule=datamodule)
        
        # Save model
        self._save_curriculum_model(gat_model, trainer)
        
        return gat_model, trainer
    
    def _create_gat_model(self, num_ids: int) -> GATLightningModule:
        """Create GAT Lightning module."""
        return GATLightningModule(
            cfg=self.config,
            num_ids=num_ids
        )
    
    def _setup_curriculum_datamodule(self, num_ids: int) -> Tuple[EnhancedCANGraphDataModule, torch.nn.Module]:
        """Setup curriculum datamodule with VGAE for hard mining."""
        
        # Load dataset
        force_rebuild = getattr(self.config, 'force_rebuild_cache', False)
        full_dataset, val_dataset, _ = load_dataset(
            self.config.dataset.name,
            self.config,
            force_rebuild_cache=force_rebuild
        )
        
        # Separate normal and attack graphs
        train_normal = [g for g in full_dataset if g.y.item() == 0]  
        train_attack = [g for g in full_dataset if g.y.item() == 1]
        val_normal = [g for g in val_dataset if g.y.item() == 0]
        val_attack = [g for g in val_dataset if g.y.item() == 1]
        
        logger.info(f"📊 Separated dataset: {len(train_normal)} normal + {len(train_attack)} attack training")
        logger.info(f"📊 Validation: {len(val_normal)} normal + {len(val_attack)} attack")
        
        # Load trained VGAE for hard mining
        vgae_path = self._resolve_vgae_path()
        logger.info(f"Loading VGAE from: {vgae_path}")
        
        # Load VGAE checkpoint and extract model
        vgae_module = VAELightningModule.load_from_checkpoint(
            str(vgae_path),
            cfg=self.config,
            num_ids=num_ids,
            map_location='cpu'
        )
        vgae_model = vgae_module.model
        vgae_model.eval()
        
        # Create enhanced datamodule
        initial_batch_size = getattr(self.config.training_config, 'batch_size', 64)
        
        datamodule = EnhancedCANGraphDataModule(
            train_normal=train_normal,
            train_attack=train_attack, 
            val_normal=val_normal,
            val_attack=val_attack,
            vgae_model=vgae_model,
            batch_size=initial_batch_size,
            num_workers=min(8, os.cpu_count() or 1),
            total_epochs=self.config.training_config.max_epochs
        )
        
        # Set dynamic batch recalculation threshold
        recalc_threshold = getattr(
            self.config.training_config,
            'dynamic_batch_recalc_threshold',
            2.0
        )
        datamodule.train_dataset.recalc_threshold = recalc_threshold
        logger.info(
            f"🔧 Dynamic batch recalculation enabled "
            f"(threshold: {recalc_threshold}x dataset growth)"
        )
        
        return datamodule, vgae_model
    
    def _resolve_vgae_path(self) -> Path:
        """Resolve path to trained VGAE model."""
        # Try PathResolver first
        vgae_path = self.path_resolver.resolve_autoencoder_path()
        
        # Fallback to config artifacts if resolver returns None
        if not vgae_path:
            if hasattr(self.config, 'required_artifacts'):
                artifacts = self.config.required_artifacts()
                vgae_path = artifacts.get('vgae')
            elif hasattr(self.config.training_config, 'vgae_model_path'):
                vgae_path = self.config.training_config.vgae_model_path
        
        if not vgae_path or not Path(vgae_path).exists():
            raise FileNotFoundError(
                f"Curriculum training requires VGAE model at {vgae_path}. "
                "Train a VGAE model first using training=autoencoder mode."
            )
        
        return Path(vgae_path)
        
        datamodule = EnhancedCANGraphDataModule(
            train_normal=train_normal,
            train_attack=train_attack, 
            val_normal=val_normal,
            val_attack=val_attack,
            vgae_model=vgae_model,
            batch_size=initial_batch_size,
            num_workers=min(8, os.cpu_count() or 1),
            total_epochs=self.config.training.max_epochs
        )
        
        # Set dynamic batch recalculation threshold
        recalc_threshold = getattr(
            self.config.training,
            'dynamic_batch_recalc_threshold',
            2.0
        )
        datamodule.train_dataset.recalc_threshold = recalc_threshold
        logger.info(
            f"🔧 Dynamic batch recalculation enabled "
            f"(threshold: {recalc_threshold}x dataset growth)"
        )
        
        return datamodule, vgae_model
    
    def _resolve_vgae_path(self) -> Path:
        """Resolve path to trained VGAE model."""
        # Use PathResolver for unified resolution
        path_resolver = PathResolver(self.config)
        vgae_path = path_resolver.resolve_autoencoder_path()
        
        # Fallback to config artifacts if resolver returns None
        if not vgae_path:
            artifacts = self.config.required_artifacts()
            vgae_path = getattr(
                self.config.training,
                'vgae_model_path',
                None
            ) or artifacts.get('vgae')
        
        if not vgae_path or not Path(vgae_path).exists():
            raise FileNotFoundError(
                f"Curriculum training requires VGAE model at {vgae_path}. "
                "Please ensure it's available under experiment_runs."
            )
        
        return Path(vgae_path)
    
    def _optimize_batch_size_for_curriculum(
        self,
        model: GATLightningModule,
        datamodule: EnhancedCANGraphDataModule
    ) -> GATLightningModule:
        """Optimize batch size for curriculum learning."""
        
        optimize_batch = getattr(self.config.training_config, 'optimize_batch_size', True)
        
        if not optimize_batch:
            # Use conservative batch size if optimization disabled
            conservative_batch_size = datamodule.get_conservative_batch_size(
                datamodule.batch_size
            )
            datamodule.batch_size = conservative_batch_size
            logger.info(
                f"📊 Using conservative batch size: {conservative_batch_size} "
                "(optimization disabled)"
            )
            return model
        
        logger.info("🔧 Optimizing batch size using maximum curriculum dataset size...")
        
        # Temporarily set dataset to maximum size for accurate optimization
        original_state = datamodule.create_max_size_dataset_for_tuning()
        
        try:
            optimizer = BatchSizeOptimizer(
                accelerator=self.config.trainer_config.accelerator,
                devices=self.config.trainer_config.devices,
                graph_memory_safety_factor=getattr(
                    self.config.training_config,
                    'graph_memory_safety_factor',
                    0.5
                ),
                max_batch_size_trials=getattr(
                    self.config.training_config,
                    'max_batch_size_trials',
                    10
                ),
                batch_size_mode=getattr(
                    self.config.training_config,
                    'batch_size_mode',
                    'power'
                )
            )
            
            safe_batch_size = optimizer.optimize_with_datamodule(model, datamodule)
            logger.info(f"✅ Batch size optimized for curriculum learning: {safe_batch_size}")
            
        except Exception as e:
            logger.warning(
                f"Batch size optimization failed: {e}. "
                "Using default batch size."
            )
            # Use conservative fallback
            safe_batch_size = datamodule.get_conservative_batch_size(
                getattr(self.config.training_config, 'batch_size', 64)
            )
            datamodule.batch_size = safe_batch_size
        
        finally:
            # Restore dataset to original state
            if original_state:
                datamodule.restore_dataset_after_tuning(original_state)
        
        return model
    
    def _create_curriculum_trainer(self) -> pl.Trainer:
        """Create trainer with curriculum callback."""
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
        from lightning.pytorch.loggers import CSVLogger
        
        curriculum_callback = CurriculumCallback()
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.paths['checkpoint_dir']),
            filename=f'gat_curriculum_{{epoch:02d}}_{{val_loss:.3f}}',
            save_top_k=getattr(self.config.training_config, 'save_top_k', 3),
            monitor=getattr(self.config.training_config, 'monitor_metric', 'val_loss'),
            mode=getattr(self.config.training_config, 'monitor_mode', 'min'),
            auto_insert_metric_name=False
        )
        
        early_stop = EarlyStopping(
            monitor=getattr(self.config.training_config, 'monitor_metric', 'val_loss'),
            patience=getattr(
                self.config.training_config,
                'early_stopping_patience',
                50
            ),
            mode=getattr(self.config.training_config, 'monitor_mode', 'min'),
            verbose=True
        )
        
        csv_logger = CSVLogger(
            save_dir=str(self.paths['log_dir']),
            name=f'curriculum_{self.config.dataset.name}'
        )
        
        return pl.Trainer(
            accelerator=self.config.trainer_config.accelerator,
            devices=self.config.trainer_config.devices,
            precision=getattr(self.config.training_config, 'precision', '32-true'),
            max_epochs=self.config.training_config.max_epochs,
            gradient_clip_val=getattr(
                self.config.training_config,
                'gradient_clip_val',
                1.0
            ),
            callbacks=[checkpoint_callback, early_stop, curriculum_callback],
            logger=csv_logger,
            enable_checkpointing=True,
            log_every_n_steps=getattr(self.config.training_config, 'log_every_n_steps', 50),
            enable_progress_bar=True
        )
    
    def _save_curriculum_model(
        self,
        model: GATLightningModule,
        trainer: pl.Trainer
    ):
        """Save final curriculum-trained model as state dict."""
        model_path = self.paths['model_save_dir'] / f"gat_{self.config.dataset.name}_curriculum.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Extract state dict from GAT model
            state = model.model.state_dict()
            torch.save(state, model_path)
            logger.info(f"💾 State-dict model saved to {model_path}")
            
        except Exception as e:
            # Attempt Lightning checkpoint as fallback
            try:
                ckpt_path = model_path.with_suffix('.ckpt')
                trainer.save_checkpoint(ckpt_path)
                logger.info(f"💾 Lightning checkpoint saved to {ckpt_path}")
            except Exception as e2:
                logger.error(f"Failed to save Lightning checkpoint: {e2}")
            
            logger.error(f"Failed to save curriculum model: {e}")
