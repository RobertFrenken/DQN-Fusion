# ============================================================================
# KD-GAT Training with Hydra-Zen and PyTorch Lightning
# Clean integration: Hydra-Zen configs -> Lightning modules -> Slurm jobs
# ============================================================================

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
)
from pytorch_lightning.loggers import MLFlowLogger, CSVLogger
from omegaconf import DictConfig, OmegaConf
from hydra_zen import instantiate
import hydra

# Local imports
from src.utils.experiment_paths import ExperimentPathManager
from src.training.lightning_modules import (
    VAELightningModule, GATLightningModule, DQNLightningModule
)

logger = logging.getLogger(__name__)

# ============================================================================
# LIGHTNING MODULE MAPPING
# ============================================================================

ARCHITECTURE_TO_MODULE = {
    "VGAE": VAELightningModule,
    "GAT": GATLightningModule,
    "DQN": DQNLightningModule,
}

LEARNING_TYPE_TO_LOSS = {
    "unsupervised": "reconstruction",
    "classifier": "classification",
    "fusion": "fusion",
}

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_experiment(cfg: DictConfig) -> dict:
    """
    Execute complete training pipeline for KD-GAT experiment.
    
    Args:
        cfg: OmegaConf configuration from Hydra-Zen
        
    Returns:
        dict: Results including best_val_loss, checkpoint_path, etc.
    """
    
    logger.info("=" * 70)
    logger.info("🚀 Starting KD-GAT Training Pipeline")
    logger.info("=" * 70)
    
    # =====================================================================
    # 1. SETUP PATHS (Deterministic, NO FALLBACKS)
    # =====================================================================
    try:
        path_manager = ExperimentPathManager(cfg)
        run_dir = path_manager.get_run_dir_safe()
        checkpoint_dir = path_manager.get_checkpoint_dir()
        eval_dir = path_manager.get_evaluation_dir()
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        path_manager.print_structure()
        
    except Exception as e:
        logger.error(f"❌ Path initialization failed: {e}")
        raise
    
    # =====================================================================
    # 2. SAVE EXPERIMENT CONFIG (For reproducibility)
    # =====================================================================
    config_path = path_manager.get_config_path()
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
    logger.info(f"✅ Saved config to {config_path}")
    
    # =====================================================================
    # 3. PREPARE DATA
    # =====================================================================
    logger.info("\n📊 Loading dataset...")
    try:
        # Instantiate dataset from config
        dataset_config = cfg.dataset_config
        # You'll need to implement this based on your data loading
        train_loader, val_loader, test_loader = load_data_loaders(cfg)
        logger.info(f"✅ Loaded {cfg.dataset} dataset")
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        raise
    
    # =====================================================================
    # 4. BUILD LIGHTNING MODULE
    # =====================================================================
    logger.info("\n🏗️  Building model...")
    try:
        # Select architecture-specific module
        module_class = ARCHITECTURE_TO_MODULE[cfg.model_architecture]
        
        lightning_module = module_class(
            cfg=cfg,
            path_manager=path_manager,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        
        logger.info(f"✅ Created {cfg.model_architecture} Lightning module")
        logger.info(f"   Model size: {cfg.model_size}")
        logger.info(f"   Distillation: {cfg.distillation}")
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        raise
    
    # =====================================================================
    # 5. SETUP CALLBACKS
    # =====================================================================
    callbacks = [
        # Checkpoint best models
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best_model_{epoch:02d}_{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=cfg.save_top_k,
            verbose=True,
            save_weights_only=False,
            auto_insert_metric_filename=False,
        ),
        
        # Early stopping
        EarlyStopping(
            monitor=cfg.training_config.early_stopping_metric,
            mode="min",
            patience=cfg.training_config.early_stopping_patience,
            verbose=True,
            check_finite=True,
        ),
        
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="epoch"),
        
        # Progress bar
        RichProgressBar(),
    ]
    
    # =====================================================================
    # 6. SETUP LOGGERS
    # =====================================================================
    loggers = [
        # MLflow logger
        MLFlowLogger(
            experiment_name=cfg.mlflow_experiment_name,
            run_name=cfg.mlflow_run_name,
            tracking_uri=cfg.mlflow_tracking_uri,
            save_dir=str(run_dir),
            log_model=True,
        ),
        
        # CSV logger (backup)
        CSVLogger(
            save_dir=str(run_dir),
            name="lightning_logs",
        ),
    ]
    
    # =====================================================================
    # 7. SETUP TRAINER
    # =====================================================================
    logger.info("\n⚙️  Configuring trainer...")
    
    trainer = pl.Trainer(
        # Training settings
        max_epochs=cfg.training_config.epochs,
        gradient_clip_val=cfg.training_config.gradient_clip,
        enable_progress_bar=True,
        
        # GPU settings
        accelerator="gpu" if cfg.device == "cuda" else "cpu",
        devices=1 if cfg.device == "cuda" else 0,
        
        # Logging
        logger=loggers,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_every_n_epochs,
        
        # Checkpointing
        callbacks=callbacks,
        enable_checkpointing=True,
        default_root_dir=str(run_dir),
        
        # Optimization
        mixed_precision=("16-mixed" if cfg.mixed_precision else None),
        
        # Reproducibility
        deterministic=True,
        seed=cfg.seed,
    )
    
    logger.info("✅ Trainer configured")
    
    # =====================================================================
    # 8. TRAIN MODEL
    # =====================================================================
    logger.info("\n🎯 Starting training...")
    logger.info(f"   Epochs: {cfg.training_config.epochs}")
    logger.info(f"   Learning rate: {cfg.training_config.learning_rate}")
    logger.info(f"   Batch size: {cfg.training_config.batch_size}")
    
    try:
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        logger.info("✅ Training completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    # =====================================================================
    # 9. EVALUATE ON TEST SET
    # =====================================================================
    logger.info("\n📈 Running test evaluation...")
    
    try:
        test_results = trainer.test(
            dataloaders=test_loader,
            ckpt_path="best",
        )
        logger.info(f"✅ Test evaluation complete")
        
        # Save test results
        test_results_path = eval_dir / "test_results.json"
        import json
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"   Results saved to {test_results_path}")
        
    except Exception as e:
        logger.error(f"❌ Test evaluation failed: {e}")
        raise
    
    # =====================================================================
    # 10. SAVE FINAL MODEL & METRICS
    # =====================================================================
    logger.info("\n💾 Saving final model...")
    
    try:
        model_path = path_manager.get_model_path()
        
        torch.save({
            'model_state_dict': lightning_module.state_dict(),
            'hyperparameters': OmegaConf.to_container(cfg, resolve=True),
            'best_val_loss': trainer.callback_metrics.get('val_loss', float('inf')),
        }, model_path)
        
        logger.info(f"✅ Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"❌ Model saving failed: {e}")
        raise
    
    # =====================================================================
    # 11. FINAL REPORT
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Model path:      {model_path}")
    logger.info(f"Results saved to: {run_dir}")
    logger.info(f"Run ID:           {run_dir.name}")
    logger.info("=" * 70 + "\n")
    
    return {
        'status': 'success',
        'model_path': str(model_path),
        'run_dir': str(run_dir),
        'best_val_loss': float(trainer.callback_metrics.get('val_loss', float('inf'))),
        'test_results': test_results,
    }


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def load_data_loaders(cfg: DictConfig) -> Tuple:
    """
    Load data according to dataset config.
    Implement based on your data loading logic.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # This is a placeholder - implement based on your datasets
    # Should respect cfg.batch_size, cfg.num_workers, etc.
    raise NotImplementedError("Implement data loading for your datasets")


# ============================================================================
# HYDRA ENTRY POINT
# ============================================================================

@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="base_experiment"
)
def main(cfg: DictConfig) -> dict:
    """
    Main entry point for Hydra-Zen training.
    
    Usage:
        # Single experiment
        python train_with_hydra_zen.py config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples
        
        # Hyperparameter sweep
        python train_with_hydra_zen.py -m hidden_dim=64,128,256 learning_rate=1e-3,1e-4
        
        # Slurm submission
        python train_with_hydra_zen.py hydra/launcher=submitit_slurm
    """
    
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    
    # Run training
    results = train_experiment(cfg)
    
    return results


if __name__ == "__main__":
    main()
