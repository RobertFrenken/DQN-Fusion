"""
Fusion Training with PyTorch Lightning and Hydra-Zen

This script trains a DQN-based fusion agent using pre-cached predictions
from VGAE and GAT models. Much simpler than the original custom implementation.

Usage:
python train_fusion_lightning.py --dataset hcrl_sa --model-path saved_models/
python train_fusion_lightning.py --dataset set_04 --max-epochs 100
"""

import os
import sys
from pathlib import Path
import logging
import argparse

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.fusion_config import DATASET_PATHS, FusionTrainingConfig
from src.training.fusion_lightning import FusionLightningModule, FusionPredictionCache
from src.training.prediction_cache import create_fusion_prediction_cache
from train_models import CANGraphLightningModule, load_dataset, create_dataloaders

logger = logging.getLogger(__name__)


def train_fusion(
    dataset_name: str,
    autoencoder_path: str,
    classifier_path: str,
    max_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    device: str = 'cuda'
):
    """
    Train fusion agent using Lightning.
    
    Args:
        dataset_name: Name of dataset (e.g., 'set_04', 'hcrl_sa')
        autoencoder_path: Path to trained VGAE model
        classifier_path: Path to trained GAT model
        max_epochs: Number of training epochs
        batch_size: Batch size for fusion training
        learning_rate: Learning rate for Q-network
        device: Device to train on ('cuda' or 'cpu')
    """
    
    # Validate dataset
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_PATHS.keys())}")
    
    logger.info(f"🚀 Starting fusion training for dataset: {dataset_name}")
    logger.info(f"  Autoencoder: {autoencoder_path}")
    logger.info(f"  Classifier: {classifier_path}")
    
    # === Load pre-trained models ===
    logger.info("\n=== Loading Pre-trained Models ===")
    
    # Load dataset info for model initialization
    train_dataset, val_dataset, num_ids = load_dataset(dataset_name, {'batch_size': 32})
    
    # Create Lightning modules to load model weights
    ae_module = CANGraphLightningModule(
        model_config=None,  # Will be set from checkpoint
        training_config=None,
        model_type='vgae',
        num_ids=num_ids
    )
    
    classifier_module = CANGraphLightningModule(
        model_config=None,
        training_config=None,
        model_type='gat',
        num_ids=num_ids
    )
    
    # Load checkpoints
    ae_ckpt = torch.load(autoencoder_path, map_location=device)
    classifier_ckpt = torch.load(classifier_path, map_location=device)
    
    ae_module.load_state_dict(ae_ckpt['state_dict'] if 'state_dict' in ae_ckpt else ae_ckpt)
    classifier_module.load_state_dict(classifier_ckpt['state_dict'] if 'state_dict' in classifier_ckpt else classifier_ckpt)
    
    logger.info("✓ Models loaded successfully")
    
    # === Create data loaders ===
    logger.info("\n=== Creating Data Loaders ===")
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=64)
    
    # === Build prediction caches ===
    logger.info("\n=== Building Prediction Caches ===")
    train_anomaly, train_gat, train_labels, val_anomaly, val_gat, val_labels = \
        create_fusion_prediction_cache(
            autoencoder=ae_module.model,
            classifier=classifier_module.model,
            train_loader=train_loader,
            val_loader=val_loader,
            dataset_name=dataset_name,
            device=device,
            cache_dir='cache/fusion'
        )
    
    logger.info(f"✓ Caches created: {len(train_anomaly)} training, {len(val_anomaly)} validation")
    
    # === Create fusion datasets from cached predictions ===
    train_cache_dataset = FusionPredictionCache(
        anomaly_scores=train_anomaly,
        gat_probs=train_gat,
        labels=train_labels
    )
    
    val_cache_dataset = FusionPredictionCache(
        anomaly_scores=val_anomaly,
        gat_probs=val_gat,
        labels=val_labels
    )
    
    # Create data loaders for fusion training
    fusion_train_loader = DataLoader(
        train_cache_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # No workers needed for cached predictions
    )
    
    fusion_val_loader = DataLoader(
        val_cache_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"✓ Fusion data loaders created")
    
    # === Create Lightning module ===
    logger.info("\n=== Initializing Fusion Lightning Module ===")
    
    fusion_config = {
        'alpha_steps': 21,
        'fusion_lr': learning_rate,
        'gamma': 0.9,
        'fusion_epsilon': 0.9,
        'fusion_epsilon_decay': 0.995,
        'fusion_min_epsilon': 0.2,
        'fusion_buffer_size': 100000,
        'fusion_batch_size': min(batch_size, 256),
        'target_update_freq': 100
    }
    
    model = FusionLightningModule(
        fusion_config=fusion_config,
        num_ids=num_ids
    )
    
    logger.info("✓ Fusion module created")
    
    # === Setup Lightning trainer ===
    logger.info("\n=== Setting Up Lightning Trainer ===")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'saved_models/fusion_checkpoints/{dataset_name}',
        filename=f'fusion_{{epoch:02d}}_{{val_accuracy:.3f}}',
        save_top_k=3,
        monitor='val_accuracy',
        mode='max',
        auto_insert_metric_name=False
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        mode='max',
        verbose=True
    )
    
    # Logger
    csv_logger = CSVLogger(
        save_dir='logs',
        name=f'fusion_{dataset_name}'
    )
    
    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=csv_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        precision='32-true'
    )
    
    logger.info("✓ Trainer initialized")
    
    # === Train ===
    logger.info("\n=== Starting Training ===")
    trainer.fit(model, fusion_train_loader, fusion_val_loader)
    
    # === Evaluate ===
    logger.info("\n=== Final Evaluation ===")
    test_results = trainer.validate(model, fusion_val_loader, verbose=True)
    logger.info(f"Test accuracy: {test_results[0]['val_accuracy']:.4f}")
    
    # === Save agent ===
    logger.info("\n=== Saving Fusion Agent ===")
    agent_path = f'saved_models/fusion_agent_{dataset_name}.pth'
    model.fusion_agent.save_agent(agent_path)
    logger.info(f"✓ Agent saved to {agent_path}")
    
    # === Get and save policy ===
    logger.info("\n=== Extracting Learned Policy ===")
    policy = model.get_policy()
    policy_path = f'saved_models/fusion_policy_{dataset_name}.npy'
    np.save(policy_path, policy)
    logger.info(f"✓ Policy saved to {policy_path}")
    
    logger.info("\n✅ Fusion training completed!")
    
    return model, trainer, test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN fusion agent with Lightning')
    parser.add_argument('--dataset', type=str, default='set_04', help='Dataset name')
    parser.add_argument('--autoencoder', type=str, default='saved_models/autoencoder_best.pth')
    parser.add_argument('--classifier', type=str, default='saved_models/classifier_best.pth')
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    train_fusion(
        dataset_name=args.dataset,
        autoencoder_path=args.autoencoder,
        classifier_path=args.classifier,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
