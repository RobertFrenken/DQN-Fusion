"""
Unified Training Orchestrator for CAN-Graph

Consolidates training logic from train_with_hydra_zen.py into a clean,
maintainable class structure that delegates to mode-specific trainers.

This module provides:
- HydraZenTrainer: Main orchestrator class
- Mode dispatching (normal, curriculum, fusion, autoencoder)
- Common setup (models, trainers, callbacks, loggers)
- Batch size optimization
- Configuration validation
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import asdict

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, DeviceStatsMonitor
from lightning.pytorch.tuner import Tuner

from src.paths import PathResolver
from src.training.datamodules import load_dataset, create_dataloaders, CANGraphDataModule
from src.training.lightning_modules import (
    VAELightningModule, 
    GATLightningModule, 
    DQNLightningModule, 
    FusionLightningModule
)
from src.training.modes import FusionTrainer, CurriculumTrainer

logger = logging.getLogger(__name__)


class HydraZenTrainer:
    """
    Main training orchestrator using hydra-zen configurations.
    
    Delegates to mode-specific trainers:
    - Normal/KD: Standard supervised training
    - Curriculum: Hard mining with VGAE guidance  
    - Fusion: DQN-based multi-model fusion
    - Autoencoder: VGAE unsupervised training
    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration.
        
        Args:
            config: CANGraphConfig instance
        """
        self.config = config
        self.path_resolver = PathResolver(config)
        self.validate_config()
    
    def validate_config(self):
        """Validate the configuration."""
        try:
            from src.config.hydra_zen_configs import validate_config
            if not validate_config(self.config):
                raise ValueError("Configuration validation failed")
        except ImportError:
            # Validation function not available - skip
            pass

    def get_safety_factor_from_json(self, dataset_name: str) -> float:
        """
        Read safety factor from config/batch_size_factors.json.

        Checks for KD-specific keys first when knowledge distillation is enabled
        (keys with '_kd' suffix, e.g., 'hcrl_sa_kd').

        Args:
            dataset_name: Name of the dataset (e.g., 'hcrl_ch', 'set_02')

        Returns:
            Safety factor from JSON file, or None if not found
        """
        json_path = Path(__file__).parent.parent.parent / "config" / "batch_size_factors.json"

        if not json_path.exists():
            return None

        try:
            with open(json_path, 'r') as f:
                factors = json.load(f)

            # Check if KD is enabled - use KD-specific factors (~25% more conservative)
            use_kd = getattr(self.config.training, 'use_knowledge_distillation', False)

            if use_kd:
                # First try KD-specific key (e.g., 'hcrl_sa_kd')
                kd_key = f"{dataset_name}_kd"
                if kd_key in factors:
                    logger.info(f"Using KD-specific safety factor: {kd_key}")
                    return factors[kd_key]

                # Fall back to KD default
                if "_default_kd" in factors:
                    logger.info(f"Using KD default safety factor")
                    return factors["_default_kd"]

            # Check for exact dataset match (non-KD)
            if dataset_name in factors:
                return factors[dataset_name]

            # Check for default
            if "_default" in factors:
                return factors["_default"]

            return None
        except Exception as e:
            logger.warning(f"Failed to read batch_size_factors.json: {e}")
            return None
    
    # ========================================================================
    # Path Management
    # ========================================================================
    
    def get_hierarchical_paths(self) -> Dict[str, Path]:
        """
        Get all experiment directories using PathResolver.
        
        Returns:
            Dictionary with keys: experiment_dir, checkpoint_dir, model_save_dir,
            log_dir, mlruns_dir
        """
        paths = self.path_resolver.get_all_experiment_dirs(create=True)
        logger.info(f"Experiment directory: {paths['experiment_dir']}")
        return paths
    
    # ========================================================================
    # Model Setup
    # ========================================================================
    
    def setup_model(self, num_ids: int) -> pl.LightningModule:
        """
        Create Lightning module from config.
        
        Args:
            num_ids: Number of unique CAN IDs
            
        Returns:
            Lightning module (VAELightningModule, GATLightningModule, DQNLightningModule, or FusionLightningModule)
        """
        if self.config.training.mode == "fusion":
            # Create fusion model with DQN agent
            fusion_config = asdict(self.config.training)
            model = FusionLightningModule(fusion_config, num_ids)
            return model
        elif self.config.model.type in ["vgae", "vgae_student"]:
            # VGAE autoencoder
            model = VAELightningModule(cfg=self.config, num_ids=num_ids)
            return model
        elif self.config.model.type.startswith("gat"):
            # GAT classifier (teacher or student)
            model = GATLightningModule(cfg=self.config, num_ids=num_ids)
            return model
        elif self.config.model.type.startswith("dqn"):
            # DQN agent (teacher or student)
            model = DQNLightningModule(cfg=self.config, num_ids=num_ids)
            return model
        else:
            raise ValueError(f"Unsupported model type: {self.config.model.type}")
    
    # ========================================================================
    # Trainer Setup
    # ========================================================================
    
    def setup_trainer(self) -> pl.Trainer:
        """
        Create Lightning trainer with callbacks and loggers.
        
        Returns:
            Configured pl.Trainer instance
        """
        paths = self.get_hierarchical_paths()
        
        # Build callbacks
        callbacks = self._create_callbacks(paths)
        
        # Build loggers
        loggers = self._create_loggers(paths)
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            precision=self.config.trainer.precision,
            gradient_clip_val=getattr(self.config.training, 'gradient_clip_val', 0.0),
            accumulate_grad_batches=getattr(self.config.training, 'accumulate_grad_batches', 1),
            callbacks=callbacks,
            logger=loggers if loggers else False,
            enable_checkpointing=self.config.trainer.enable_checkpointing,
            log_every_n_steps=self.config.logging.get("log_interval", 50),
            deterministic=getattr(self.config.training, 'deterministic_training', True),
            check_val_every_n_epoch=self.config.logging.get("val_check_interval", 1)
        )
        
        return trainer
    
    def _create_callbacks(self, paths: Dict[str, Path]) -> list:
        """Create Lightning callbacks."""
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
        
        # Early stopping (if enabled)
        if getattr(self.config.training, 'early_stopping', False):
            early_stop = EarlyStopping(
                monitor=self.config.logging.get("monitor_metric", "val_loss"),
                patience=self.config.training.get("early_stopping_patience", 10),
                mode=self.config.logging.get("monitor_mode", "min"),
                verbose=True
            )
            callbacks.append(early_stop)
        
        # Device stats monitoring (if enabled)
        if getattr(self.config.logging, 'monitor_device_stats', False):
            callbacks.append(DeviceStatsMonitor())
        
        return callbacks
    
    def _create_loggers(self, paths: Dict[str, Path]) -> list:
        """Create Lightning loggers."""
        loggers = []
        
        # CSV Logger
        csv_logger = CSVLogger(
            save_dir=str(paths['log_dir']),
            name='csv_logs'
        )
        loggers.append(csv_logger)
        
        # TensorBoard Logger (if enabled)
        if getattr(self.config.logging, 'use_tensorboard', True):
            try:
                tb_logger = TensorBoardLogger(
                    save_dir=str(paths['log_dir']),
                    name='tensorboard_logs'
                )
                loggers.append(tb_logger)
            except ModuleNotFoundError:
                logger.warning("TensorBoard not installed. Install with: pip install tensorboard. Skipping TensorBoard logging.")
        
        # MLflow Logger (if enabled)
        if getattr(self.config.logging, 'use_mlflow', True):
            try:
                mlruns_path = paths.get('mlruns_dir')
                if mlruns_path:
                    logger.info(f"Setting up MLflow with path: {mlruns_path}")
                    mlflow_logger = MLFlowLogger(
                        experiment_name=self.config.experiment_name,
                        tracking_uri=f"file:{mlruns_path}",
                        save_dir=str(mlruns_path.parent)
                    )
                    loggers.append(mlflow_logger)
            except Exception as e:
                logger.warning(f"Could not create MLFlowLogger: {e}")
        
        return loggers
    
    # ========================================================================
    # Batch Size Optimization
    # ========================================================================
    
    def _optimize_batch_size(
        self, 
        model: pl.LightningModule, 
        train_dataset, 
        val_dataset
    ) -> pl.LightningModule:
        """
        Optimize batch size using Lightning's Tuner.
        
        Args:
            model: Model to optimize
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Model with optimized batch_size attribute
        """
        logger.info("🔧 Optimizing batch size...")
        
        # Create temporary DataModule
        temp_datamodule = CANGraphDataModule(
            train_dataset, 
            val_dataset, 
            model.batch_size, 
            num_workers=4
        )
        
        # Create temporary directory for tuner checkpoints
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='tuner_', dir='.')
        
        trainer = pl.Trainer(
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            precision='32-true',
            max_steps=200,
            max_epochs=None,
            enable_checkpointing=False,
            logger=False,
            default_root_dir=temp_dir
        )

        # Dataset-size-aware safety factor
        # Priority: 1) JSON file (easy to edit), 2) config, 3) auto-select by dataset size
        dataset_size = len(train_dataset) + len(val_dataset)
        dataset_name = self.config.dataset.name

        # First, try to get factor from JSON file (config/batch_size_factors.json)
        json_factor = self.get_safety_factor_from_json(dataset_name)

        if json_factor is not None:
            graph_memory_safety_factor = json_factor
            logger.info(f"Using safety factor from batch_size_factors.json: {graph_memory_safety_factor} (dataset: {dataset_name})")
        elif hasattr(self.config.training, 'graph_memory_safety_factor') and self.config.training.graph_memory_safety_factor is not None:
            # Use user-specified factor from config
            graph_memory_safety_factor = self.config.training.graph_memory_safety_factor
            logger.info(f"Using config-specified safety factor: {graph_memory_safety_factor}")
        else:
            # Auto-determine based on dataset size
            if dataset_size < 50000:
                graph_memory_safety_factor = 0.6  # Small datasets
            elif dataset_size < 100000:
                graph_memory_safety_factor = 0.5  # Medium datasets
            else:
                graph_memory_safety_factor = 0.35  # Large datasets (set_02: 203K graphs)
            logger.info(
                f"Auto-selected safety factor {graph_memory_safety_factor} "
                f"for dataset size {dataset_size:,} graphs"
            )
        
        tuner = Tuner(trainer)
        initial_bs = model.batch_size

        # Get batch size mode from config (default to 'binsearch' for more accuracy)
        batch_size_mode = getattr(self.config.training, 'batch_size_mode', 'binsearch')

        try:
            # Run tuner - modifies model.batch_size in place
            tuner.scale_batch_size(
                model,
                datamodule=temp_datamodule,
                mode=batch_size_mode,
                steps_per_trial=3,
                init_val=initial_bs
            )
            
            # Apply safety factor for graph data
            if graph_memory_safety_factor < 1.0:
                optimized_bs = model.batch_size
                adjusted_bs = max(int(optimized_bs * graph_memory_safety_factor), 1)
                model.batch_size = adjusted_bs
                logger.info(
                    f"📊 Batch size: {initial_bs} → {optimized_bs} → {adjusted_bs} "
                    f"(safety factor {graph_memory_safety_factor})"
                )
            else:
                logger.info(f"📊 Batch size optimized: {initial_bs} → {model.batch_size}")
                
        except Exception as e:
            logger.warning(f"Batch size optimization failed: {e}. Using initial batch size.")
            model.batch_size = initial_bs
        finally:
            # Clean up temporary files and directory
            import shutil
            import glob
            try:
                # Clean up tuner checkpoints in root directory
                for ckpt in glob.glob('.scale_batch_size_*.ckpt'):
                    Path(ckpt).unlink(missing_ok=True)
                # Clean up temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as cleanup_error:
                logger.debug(f"Cleanup warning: {cleanup_error}")
        
        return model
    
    # ========================================================================
    # Training Execution
    # ========================================================================
    
    def train(self):
        """
        Execute complete training pipeline.
        
        Dispatches to appropriate training mode:
        - fusion: FusionTrainer
        - curriculum: CurriculumTrainer
        - normal/kd/autoencoder: Standard training
        """
        logger.info("🚀 Starting training with hydra-zen config")
        logger.info(f"Experiment: {self.config.experiment_name}")
        logger.info(f"Mode: {self.config.training.mode}")
        
        # Set global seeds if configured
        self._set_global_seeds()
        
        # Dispatch to mode-specific trainer
        if self.config.training.mode == "fusion":
            return self._train_fusion()
        elif self.config.training.mode == "curriculum":
            return self._train_curriculum()
        else:
            return self._train_standard()
    
    def _set_global_seeds(self):
        """Set global RNG seeds for reproducibility."""
        try:
            from src.utils.seeding import set_global_seeds
            seed = getattr(self.config.training, 'seed', None)
            deterministic = getattr(self.config.training, 'deterministic_training', True)
            cudnn_benchmark = getattr(self.config.training, 'cudnn_benchmark', False)
            if seed is not None:
                set_global_seeds(seed, deterministic=deterministic, cudnn_benchmark=cudnn_benchmark)
                logger.info(
                    f"Global RNG seed set: {seed} "
                    f"(deterministic={deterministic}, cudnn_benchmark={cudnn_benchmark})"
                )
        except Exception as e:
            logger.warning(f"Failed to set global seeds: {e}")
    
    def _train_fusion(self):
        """Train fusion agent using FusionTrainer."""
        run_num = self.path_resolver.get_run_counter()
        logger.info(f"🔢 Run number: {run_num:03d}")
        logger.info("🔀 Dispatching to FusionTrainer")

        paths = self.get_hierarchical_paths()
        fusion_trainer = FusionTrainer(self.config, paths)
        model, trainer = fusion_trainer.train()

        # Post-training tasks (FusionTrainer already saves dqn_fusion.pth, config snapshot is extra)
        model_name = self._generate_model_filename("dqn", "fusion", run_num=run_num)
        self._save_final_model(model, model_name)
        self._save_config_snapshot(paths)

        return model, trainer
    
    def _train_curriculum(self):
        """Train with curriculum learning using CurriculumTrainer."""
        run_num = self.path_resolver.get_run_counter()
        logger.info(f"🔢 Run number: {run_num:03d}")
        logger.info("🎓 Dispatching to CurriculumTrainer")

        # Load dataset and setup model first
        force_rebuild = getattr(self.config, 'force_rebuild_cache', False)
        _, _, num_ids = load_dataset(
            self.config.dataset.name,
            self.config,
            force_rebuild_cache=force_rebuild
        )

        paths = self.get_hierarchical_paths()
        curriculum_trainer = CurriculumTrainer(self.config, paths)
        model, trainer = curriculum_trainer.train(num_ids)

        # Post-training tasks
        model_name = self._generate_model_filename("gat", "curriculum", run_num=run_num)
        self._save_final_model(model, model_name)
        self._save_config_snapshot(paths)

        return model, trainer
    
    def _train_standard(self):
        """Standard training (normal, KD, autoencoder)."""
        # Get run number for model filename versioning
        run_num = self.path_resolver.get_run_counter()
        logger.info(f"🔢 Run number: {run_num:03d}")
        logger.info(f"📚 Standard training mode: {self.config.training.mode}")

        # Load dataset
        force_rebuild = getattr(self.config, 'force_rebuild_cache', False)
        train_dataset, val_dataset, num_ids = load_dataset(
            self.config.dataset.name,
            self.config,
            force_rebuild_cache=force_rebuild
        )

        logger.info(
            f"📊 Dataset loaded: {len(train_dataset)} training + "
            f"{len(val_dataset)} validation = {len(train_dataset) + len(val_dataset)} total"
        )

        # Setup model
        model = self.setup_model(num_ids)

        # ===== BATCH SIZE TUNING SECTION =====
        bsc = self.config.batch_size_config

        if bsc.optimize_batch_size:
            logger.info("🔧 Running batch size optimization...")
            model = self._optimize_batch_size(model, train_dataset, val_dataset)
            logger.info(f"📊 Tuner found max safe batch size: {model.batch_size}")
            final_batch_size = int(model.batch_size * bsc.safety_factor)
            logger.info(
                f"🎯 Applied safety_factor {bsc.safety_factor}: "
                f"{model.batch_size} × {bsc.safety_factor} = {final_batch_size}"
            )
            model.batch_size = final_batch_size
        else:
            fallback_size = bsc.tuned_batch_size or bsc.default_batch_size
            logger.info(f"📊 Using batch size from config: {fallback_size}")
            model.batch_size = fallback_size

        logger.info(f"✅ Training batch size: {model.batch_size}")
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, model.batch_size
        )
        
        # Setup trainer
        trainer = self.setup_trainer()
        
        # Save config snapshot
        paths = self.get_hierarchical_paths()
        self._save_config_snapshot(paths)
        
        # Train
        logger.info("Starting training...")
        trainer.fit(model, train_loader, val_loader)
        
        # Test (if enabled)
        if self.config.training.run_test:
            test_results = trainer.test(model, val_loader)
            logger.info(f"Test results: {test_results}")

        # Update frozen config with tuned batch size after successful training
        if model.batch_size and bsc.optimize_batch_size:
            self.config.batch_size_config.tuned_batch_size = model.batch_size
            logger.info(f"🔄 Updated batch_size_config: tuned_batch_size = {model.batch_size}")

        # Save final model with model_size and run counter in filename
        model_name = self._generate_model_filename(
            self.config.model.type,
            self.config.training.mode,
            run_num=run_num
        )
        self._save_final_model(model, model_name)

        return model, trainer
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _save_config_snapshot(self, paths: Dict[str, Path]):
        """Save configuration snapshot to log directory."""
        try:
            cfg_snapshot = asdict(self.config)
            cfg_file = paths['log_dir'] / 'config.json'
            with open(cfg_file, 'w', encoding='utf-8') as f:
                json.dump(cfg_snapshot, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote config snapshot to {cfg_file}")
        except Exception as e:
            logger.warning(f"Failed to write config snapshot: {e}")
    
    def _generate_model_filename(self, model_type: str, mode: str, run_num: Optional[int] = None) -> str:
        """
        Generate consistent model filename including model_size and run counter.

        Format: {model_type}_{model_size}_{mode}_run_{run_num:03d}.pth
        Examples:
            - vgae_teacher_autoencoder_run_001.pth
            - vgae_student_autoencoder_run_002.pth
            - gat_teacher_curriculum_run_001.pth
            - dqn_teacher_fusion_run_003.pth

        Args:
            model_type: Model type (vgae, gat, dqn)
            mode: Training mode (autoencoder, curriculum, fusion, normal)
            run_num: Run counter (1, 2, 3, ...). If None, omitted from filename.

        Returns:
            Filename string with .pth extension
        """
        model_size = getattr(self.config, 'model_size', 'teacher')
        base_name = f"{model_type}_{model_size}_{mode}"
        if run_num is not None:
            return f"{base_name}_run_{run_num:03d}.pth"
        else:
            return f"{base_name}.pth"

    def _save_final_model(self, model: pl.LightningModule, filename: str):
        """Save final model state dict."""
        paths = self.get_hierarchical_paths()
        model_path = paths['model_save_dir'] / filename
        
        try:
            # Extract state dict from Lightning module
            if hasattr(model, 'model'):
                state_dict = model.model.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save(state_dict, model_path)
            logger.info(f"💾 Saved final model to {model_path}")
        except Exception as e:
            logger.warning(f"Failed to save final model: {e}")
