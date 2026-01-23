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

import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
# Ensure minimal Lightning attributes exist when running under test shims
if not hasattr(pl, 'LightningModule'):
    pl.LightningModule = object
if not hasattr(pl, 'Callback'):
    pl.Callback = object
if not hasattr(pl, 'LightningDataModule'):
    pl.LightningDataModule = object
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, DeviceStatsMonitor
from lightning.pytorch.tuner import Tuner
import types
# Provide a safe Trainer fallback in case test shims replaced the module without full attributes
if not hasattr(pl, 'Trainer'):
    pl.Trainer = lambda *a, **k: types.SimpleNamespace(_kwargs=k, logger=k.get('logger', None))
# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules (robust to path resolution issues in some SLURM environments)
# Import config module and optional factory functions gracefully so tests can inject them
from importlib import import_module
_cfg_mod = import_module('src.config.hydra_zen_configs')
CANGraphConfig = _cfg_mod.CANGraphConfig
CANGraphConfigStore = getattr(_cfg_mod, 'CANGraphConfigStore', None)
create_gat_normal_config = getattr(_cfg_mod, 'create_gat_normal_config', None)
create_distillation_config = getattr(_cfg_mod, 'create_distillation_config', None)
create_autoencoder_config = getattr(_cfg_mod, 'create_autoencoder_config', None)
create_fusion_config = getattr(_cfg_mod, 'create_fusion_config', None)
validate_config = getattr(_cfg_mod, 'validate_config', lambda cfg: True)
# import lighting loader modules
from src.training.can_graph_data import CANGraphDataModule, load_dataset, create_dataloaders
from src.training.can_graph_module import CANGraphLightningModule
from src.training.fusion_lightning import FusionLightningModule
from src.training.prediction_cache import create_fusion_prediction_cache
from src.training.enhanced_datamodule import EnhancedCANGraphDataModule, CurriculumCallback
# Suppress warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch-scatter.*")
warnings.filterwarnings("ignore", message=".*GLIBCXX.*")
warnings.filterwarnings("ignore", message=".*Trying to infer.*batch_size.*")
warnings.filterwarnings("ignore", message=".*Checkpoint directory.*exists.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)  # Keep logs clean
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
        """Create hierarchical directory structure using the config's canonical path.

        Returns a dict with keys:
            - experiment_dir
            - checkpoint_dir
            - model_save_dir
            - log_dir
            - mlruns_dir

        This is strict and derives all paths from `CANGraphConfig.canonical_experiment_dir()`.
        """
        # Use canonical experiment directory from config
        exp_dir = self.config.canonical_experiment_dir()

        checkpoint_dir = exp_dir / "checkpoints"
        model_save_dir = exp_dir / "models"
        log_dir = exp_dir / "logs"
        mlruns_dir = exp_dir / "mlruns"

        # Ensure all directories exist
        for d in (exp_dir, checkpoint_dir, model_save_dir, log_dir, mlruns_dir):
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment directory: {exp_dir}")

        return {
            "experiment_dir": exp_dir,
            "checkpoint_dir": checkpoint_dir,
            "model_save_dir": model_save_dir,
            "log_dir": log_dir,
            "mlruns_dir": mlruns_dir
        }

    
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
        """Create the Lightning trainer from config.

        Builds callbacks and loggers derived from canonical hierarchical paths and training config.
        Returns a `pl.Trainer` (or compatible stub in tests).
        """
        # Get canonical hierarchical paths
        paths = self.get_hierarchical_paths()

        # Callbacks
        callbacks = []

        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(paths['checkpoint_dir']),
            filename=f'{self.config.model.type}_{self.config.training.mode}_{{epoch:02d}}_{{val_loss:.3f}}',
            save_top_k=self.config.logging.get("save_top_k", 3),
            monitor=self.config.logging.get("monitor_metric", "val_loss"),
            mode=self.config.logging.get("monitor_mode", "min"),
            save_last=True,
            auto_insert_metric_name=False
        )
        callbacks.append(checkpoint_callback)

        # Device stats monitor
        callbacks.append(DeviceStatsMonitor())

        # Early stopping
        if hasattr(self.config.training, 'early_stopping_patience'):
            early_stop_callback = EarlyStopping(
                monitor=self.config.logging.get("monitor_metric", "val_loss"),
                patience=self.config.training.early_stopping_patience,
                mode=self.config.logging.get("monitor_mode", "min"),
                verbose=True
            )
            callbacks.append(early_stop_callback)

        # Loggers
        loggers = []

        # CSV logger
        csv_logger = CSVLogger(
            save_dir=str(paths['log_dir']),
            name=f"{self.config.model.type}_{self.config.training.mode}"
        )
        loggers.append(csv_logger)

        # MLflow logger - use canonical mlruns dir if available
        try:
            mlruns_path = paths['mlruns_dir'].resolve()
            logger.info(f"Setting up MLflow with path: {mlruns_path}")
            mlflow_logger = MLFlowLogger(
                experiment_name=f"CAN-Graph-{self.config.dataset.name}",
                tracking_uri=mlruns_path.as_uri(),
                log_model=False
            )
            loggers.append(mlflow_logger)
        except Exception as e:
            logger.warning(f"Could not create MLFlowLogger (will continue without it): {e}")

        # Optional TensorBoard
        if self.config.logging.get("enable_tensorboard", False):
            tb_logger = TensorBoardLogger(
                save_dir=str(paths['log_dir']),
                name=f"{self.config.model.type}_{self.config.training.mode}"
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
            logger=loggers if loggers else None,
            callbacks=callbacks,
            enable_checkpointing=self.config.trainer.enable_checkpointing,
            log_every_n_steps=self.config.training.log_every_n_steps,
            enable_progress_bar=False,
            num_sanity_val_steps=self.config.trainer.num_sanity_val_steps
        )

        return trainer

    def _save_state_dict(self, model_obj, save_dir: Path, filename: str):
        """Save a model's state_dict safely and verify it can be reloaded as a pure state_dict.

        - Backs up existing file if present
        - Supports plain nn.Module, Lightning modules (with .model), and fusion agents (dicts containing state_dicts)
        - Attempts robust loading/conversion when legacy checkpoint formats are encountered
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        # Backup existing file
        if save_path.exists():
            backup = save_path.with_suffix(save_path.suffix + '.bak')
            if not backup.exists():
                import shutil
                shutil.copy2(save_path, backup)
                logger.info(f"Backed up existing model: {save_path} -> {backup}")

        # Determine state to save
        state_to_save = None
        try:
            # Fusion agents may expose explicit dict save methods
            if hasattr(model_obj, 'fusion_agent') and hasattr(model_obj.fusion_agent, 'q_network'):
                # Save essential agent networks rather than pickling the whole object
                state_to_save = {
                    'q_network_state_dict': model_obj.fusion_agent.q_network.state_dict(),
                    'target_network_state_dict': model_obj.fusion_agent.target_network.state_dict(),
                }
            elif hasattr(model_obj, 'state_dict') and callable(model_obj.state_dict):
                # Lightning modules often wrap models as .model
                if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'state_dict'):
                    state_to_save = model_obj.model.state_dict()
                else:
                    state_to_save = model_obj.state_dict()
            elif isinstance(model_obj, dict):
                # Already a dict - try to extract nested state_dicts
                state_to_save = model_obj
            else:
                raise RuntimeError("Unable to determine state_dict from model object")

            # First attempt: torch.save if available
            wrote_file = False
            try:
                torch.save(state_to_save, str(save_path))
                logger.info(f"Attempted torch.save to {save_path}")
                wrote_file = save_path.exists()
            except Exception:
                wrote_file = False

            # If torch.save didn't create a file (e.g., stub in test env), fallback to pickle
            if not wrote_file:
                import pickle
                with open(save_path, 'wb') as f:
                    pickle.dump(state_to_save, f)
                logger.info(f"Saved state-dict to {save_path} (via pickle fallback)")

            # Ensure file exists and try validation load
            if not save_path.exists():
                raise RuntimeError(f"Failed to create model save file at {save_path}")

            # Try to load with torch if available, else pickle
            try:
                loaded = torch.load(str(save_path), map_location='cpu')
            except Exception:
                import pickle
                with open(save_path, 'rb') as f:
                    loaded = pickle.load(f)

            if not isinstance(loaded, dict):
                logger.warning(f"Saved checkpoint at {save_path} did not load as a dict. Review format.")
            return save_path

        except Exception as e:
            logger.warning(f"State-dict save/load failed for {save_path}: {e}")
            # Legacy checkpoint conversion disabled: require explicit state_dict inputs
            raise RuntimeError(
                f"Could not save/load state-dict to {save_path}: {e}.\n"
                "Legacy checkpoint conversion is disabled to avoid unsafe unpickling.\n"
                "Please re-export the original checkpoint as a state_dict using: "
                "`ckpt = torch.load(old_checkpoint, map_location='cpu'); torch.save(ckpt['state_dict'], new_path)` "
                "or re-run the training to produce a state_dict directly via `torch.save(model.state_dict(), path)`."
            ) from e

            # Final fallback: try trainer.save_checkpoint if available (handled by caller)
            raise

    def _log_model_path_to_mlflow(self, model_path: Path, mlruns_dir: Path) -> None:
        """Best-effort: log saved model path to MLflow tracking server (file-based URI accepted).

        If MLflow isn't installed or logging fails, this method logs a warning but does not raise.
        """
        try:
            import mlflow
            tracking_uri = mlruns_dir.resolve().as_uri()
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.log_param("model_path", str(model_path))
            logger.info(f"Logged model_path to MLflow: {model_path}")
        except Exception as e:
            logger.info(f"MLflow not available or failed to log model_path: {e}")

    
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
        
        # Save final model as state_dict (standardized across all training types)
        paths = self.get_hierarchical_paths()
        model_name = f"{self.config.model.type}_{self.config.training.mode}.pth"
        try:
            saved_path = self._save_state_dict(model, paths['model_save_dir'], model_name)
            # Best-effort: log model path to MLflow tracking (if available)
            try:
                self._log_model_path_to_mlflow(saved_path, paths['mlruns_dir'])
            except Exception as e:
                logger.warning(f"Logging model path to MLflow failed: {e}")
        except Exception as e:
            logger.error(f"Failed to save model state_dict for {model_name}: {e}")
            # Fallback: let Lightning checkpoint be saved for diagnostic purposes
            try:
                trainer.save_checkpoint(paths['model_save_dir'] / f"{model_name}.ckpt")
                logger.info(f"Fallback: Lightning checkpoint saved to {paths['model_save_dir'] / f'{model_name}.ckpt'}")
            except Exception as e2:
                logger.error(f"Fallback save also failed: {e2}")

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
        
        # Use required_artifacts() to get canonical artifact locations; allow explicit overrides but fail fast if missing.
        artifacts = self.config.required_artifacts()
        ae_path = getattr(self.config.training, 'autoencoder_path', None) or artifacts.get('autoencoder')
        classifier_path = getattr(self.config.training, 'classifier_path', None) or artifacts.get('classifier')

        missing = []
        if not ae_path or not Path(ae_path).exists():
            missing.append(f"autoencoder missing at {ae_path}")
        if not classifier_path or not Path(classifier_path).exists():
            missing.append(f"classifier missing at {classifier_path}")
        if missing:
            raise FileNotFoundError("Fusion training requires pre-trained artifacts:\n" + "\n".join(missing) + "\nPlease ensure the artifacts are available at the canonical paths or set them in the training config.")

        logger.info(f"  Autoencoder: {ae_path}")
        logger.info(f"  Classifier: {classifier_path}")
        
        # Load dataset for inference
        train_dataset, val_dataset, num_ids = load_dataset(self.config.dataset.name, self.config)
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, batch_size=64  # Smaller batch for extraction
        )
        
        # Load model checkpoints STRICTLY as state_dicts (no unpickling of custom classes)
        def _strict_torch_load(path):
            try:
                return torch.load(path, map_location='cpu')
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load checkpoint {path}: {e}.\n"
                    "This pipeline requires checkpoints to be saved as pure state_dicts (" \
                    "i.e., use `torch.save(model.state_dict(), path)`).\n"
                    "If this file was saved as a Python-pickled checkpoint, please re-export it as a state_dict or re-run the training that produced it."
                ) from e

        ae_ckpt = _strict_torch_load(ae_path)
        classifier_ckpt = _strict_torch_load(classifier_path)
        
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
        
        # Save fusion agent (agent.save_agent saves networks as state_dicts internally)
        paths = self.get_hierarchical_paths()
        agent_path = paths['model_save_dir'] / f'fusion_agent_{self.config.dataset.name}.pth'
        fusion_model.fusion_agent.save_agent(str(agent_path))
        logger.info(f"✓ Fusion agent saved to {agent_path}")

        # Also save compact agent network state_dict for compatibility and fast loading
        try:
            agent_state_name = f"dqn_agent_{self.config.dataset.name}.pth"
            self._save_state_dict(fusion_model, paths['model_save_dir'], agent_state_name)
        except Exception as e:
            logger.warning(f"Could not save compact DQN agent state dict: {e}")
        
        logger.info("✅ Fusion training completed successfully!")
        # Return fusion model and trainer for consistency with other training branches
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
        # Resolve VGAE artifact using canonical required_artifacts(); allow explicit override via training.vgae_model_path
        artifacts = self.config.required_artifacts()
        vgae_path = getattr(self.config.training, 'vgae_model_path', None) or artifacts.get('vgae')
        if not vgae_path or not Path(vgae_path).exists():
            raise FileNotFoundError(f"Curriculum training requires VGAE model at {vgae_path}. Please ensure it's available under experiment_runs.")
        vgae_model = CANGraphLightningModule.load_from_checkpoint(str(vgae_path), map_location='cpu')
        vgae_model.eval()
        # Initial batch size - will be adjusted conservatively
        initial_batch_size = getattr(self.config.training, 'batch_size', 64)
        
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
        recalc_threshold = getattr(self.config.training, 'dynamic_batch_recalc_threshold', 2.0)
        datamodule.train_dataset.recalc_threshold = recalc_threshold
        logger.info(f"🔧 Dynamic batch recalculation enabled (threshold: {recalc_threshold}x dataset growth)")
        
        # Setup GAT model
        model = self.setup_model(num_ids)
        
        # Optimize batch size using maximum expected dataset size for curriculum learning
        if getattr(self.config.training, 'optimize_batch_size', True):  # Default to True
            logger.info("🔧 Optimizing batch size using maximum curriculum dataset size...")
            
            # Temporarily set dataset to maximum size for accurate batch size optimization
            original_state = datamodule.create_max_size_dataset_for_tuning()
            
            try:
                model = self._optimize_batch_size_with_datamodule(model, datamodule)
                logger.info(f"✅ Batch size optimized for curriculum learning: {model.batch_size}")
            except Exception as e:
                # Fail loudly - do not silently fallback to a conservative batch size
                raise RuntimeError(
                    f"Batch size optimization failed: {e}. "
                    "Set `training.batch_size` explicitly in your config or disable `optimize_batch_size` to proceed."
                ) from e
            
            # Restore dataset to original state
            if original_state:
                datamodule.restore_dataset_after_tuning(original_state)
        else:
            # Use conservative batch size if optimization is disabled
            conservative_batch_size = datamodule.get_conservative_batch_size(initial_batch_size)
            datamodule.batch_size = conservative_batch_size
            logger.info(f"📊 Using conservative batch size: {conservative_batch_size} (optimization disabled)")
        
        # Setup trainer with curriculum callback
        curriculum_callback = CurriculumCallback()
        trainer = self.setup_curriculum_trainer(extra_callbacks=[curriculum_callback])
        
        # Train model
        logger.info("🚀 Starting curriculum-enhanced training...")
        trainer.fit(model, datamodule=datamodule)
        
        # Save final model as a pure state_dict to avoid pickling custom objects
        paths = self.get_hierarchical_paths()
        model_path = paths['model_save_dir'] / f"gat_{self.config.dataset.name}_curriculum.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract underlying torch.nn.Module state_dict when available
        try:
            state = model.model.state_dict() if hasattr(model, 'model') else model.state_dict()
            torch.save(state, model_path)
            logger.info(f"💾 State-dict model saved to {model_path}")
        except Exception as e:
            # Attempt a last-resort Lightning checkpoint for diagnostics, but still fail loudly
            try:
                trainer.save_checkpoint(model_path.with_suffix(model_path.suffix + '.ckpt'))
                logger.info(f"💾 Lightning checkpoint saved to {model_path.with_suffix(model_path.suffix + '.ckpt')}")
            except Exception as e2:
                logger.error(f"Also failed to save Lightning checkpoint: {e2}")
            raise RuntimeError(f"Failed to save final curriculum model state_dict: {e}. A Lightning checkpoint may be available for debugging.") from e

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
            # Fail loudly to enforce strictness; caller should decide how to proceed
            raise RuntimeError(f"Batch size optimization failed: {e}. Set 'training.batch_size' or disable optimization.") from e
        
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
            name=f"{self.config.model.type}_{self.config.training.mode}"
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
                teacher_model_path=str(Path(__file__).parent.resolve() / "experiment_runs" / "automotive" / dataset / "unsupervised" / "vgae" / "teacher" / "no_distillation" / "autoencoder" / f"best_teacher_model_{dataset}.pth")
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


# Helper: apply a dependency manifest to a config
def apply_manifest_to_config(config, manifest_path: str):
    """Load and validate a dependency manifest, apply paths to config.training for fusion jobs.

    This helper keeps validation strict and fails early with informative errors.
    """
    try:
        # First try a regular import (works when package is installed)
        from src.utils.dependency_manifest import load_manifest, validate_manifest_for_config, ManifestValidationError
    except Exception:
        # Fallback: load by file path (tests and some environments use direct file execution)
        try:
            import importlib.util
            dm_path = Path(__file__).parent / 'src' / 'utils' / 'dependency_manifest.py'
            spec = importlib.util.spec_from_file_location('dependency_manifest', str(dm_path))
            dm_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dm_mod)
            load_manifest = dm_mod.load_manifest
            validate_manifest_for_config = dm_mod.validate_manifest_for_config
            ManifestValidationError = dm_mod.ManifestValidationError
        except Exception as e:
            raise RuntimeError(f"Failed to import dependency manifest utilities: {e}") from e

    manifest = load_manifest(manifest_path)
    try:
        ok, msg = validate_manifest_for_config(manifest, config)
    except ManifestValidationError:
        raise
    # If fusion job, apply explicit paths into the training config to be used later
    if getattr(config.training, 'mode', None) == 'fusion':
        autoencoder = manifest['autoencoder']['path']
        classifier = manifest['classifier']['path']
        config.training.autoencoder_path = autoencoder
        config.training.classifier_path = classifier
        print(f"🔒 Dependency manifest applied: autoencoder={autoencoder}, classifier={classifier}")

    return config


# ==========================================================================
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
    parser.add_argument('--model', type=str, choices=['gat', 'vgae', 'dqn'], default='gat',
                      help='Model type')
    parser.add_argument('--dataset', type=str, 
                      choices=['hcrl_sa', 'hcrl_ch', 'set_01', 'set_02', 'set_03', 'set_04', 'car_hacking'],
                      default='hcrl_sa', help='Dataset name')
    parser.add_argument('--training', type=str, 
                      choices=['normal', 'autoencoder', 'knowledge_distillation', 'curriculum', 'fusion'],
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
    
    # Curriculum learning specific
    parser.add_argument('--vgae_path', type=str,
                      help='Path to VGAE model (required for curriculum learning)')
    
    # Fusion training specific
    parser.add_argument('--autoencoder_path', type=str,
                      help='Path to autoencoder model (required for fusion training)')
    parser.add_argument('--classifier_path', type=str,
                      help='Path to classifier model (required for fusion training)')
    
    # Training overrides
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--tensorboard', action='store_true',
                      help='Enable TensorBoard logging')
    parser.add_argument('--force-rebuild-cache', action='store_true',
                      help='Force rebuild of cached processed data')
    # Removed --debug-graph-count flag as it was unused and causing segfaults
    parser.add_argument('--early-stopping-patience', type=int,
                      help='Early stopping patience (default: 25 for normal, 30 for autoencoder)')
    parser.add_argument('--dependency-manifest', type=str,
                      help='Path to a JSON dependency manifest to validate and apply for fusion training')
    
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

    # If a dependency manifest is provided, validate and apply it (strict, fail-fast)
    if getattr(args, 'dependency_manifest', None):
        try:
            config = apply_manifest_to_config(config, args.dependency_manifest)
        except Exception as e:
            print(f"❌ Dependency manifest validation failed: {e}")
            raise
    
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