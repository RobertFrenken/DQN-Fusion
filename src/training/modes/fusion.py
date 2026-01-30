"""
Fusion Training Mode - DQN Agent for Multi-Model Decision Fusion

Combines VGAE anomaly detection and GAT classification using
reinforcement learning (DQN) for adaptive decision making.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from src.training.lightning_modules import (
    FusionLightningModule, 
    FusionPredictionCache
)
from src.training.prediction_cache import create_fusion_prediction_cache
from src.training.datamodules import load_dataset, create_dataloaders
from src.training.model_manager import ModelManager
from src.paths import PathResolver

logger = logging.getLogger(__name__)


def infer_vgae_architecture(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Infer VGAE model architecture from checkpoint state dict.
    
    This makes loading robust - we determine architecture from saved weights
    rather than relying on config defaults that may not match.
    """
    # Infer from embedding layer
    inferred_num_ids, embedding_dim = state_dict['id_embedding.weight'].shape
    latent_dim = state_dict['z_mean.weight'].shape[0]
    
    # Infer hidden_dims from encoder batch norm layers
    hidden_dims = []
    for i in range(10):
        key = f'encoder_bns.{i}.weight'
        if key in state_dict:
            hidden_dims.append(state_dict[key].shape[0])
        else:
            break
    
    # Infer heads from attention layers
    encoder_heads = state_dict['encoder_layers.0.att_src'].shape[1]
    decoder_heads = state_dict['decoder_layers.0.att_src'].shape[1]
    
    # Infer in_channels: gat_in_dim = embedding_dim + (in_channels - 1)
    first_layer_in = state_dict['encoder_layers.0.lin.weight'].shape[1]
    in_channels = first_layer_in - embedding_dim + 1
    
    # Infer mlp_hidden from neighborhood_decoder.0.weight shape
    # neighborhood_decoder.0 is Linear(latent_dim, mlp_hidden)
    mlp_hidden = state_dict['neighborhood_decoder.0.weight'].shape[0]
    
    logger.info(f"Inferred VGAE: num_ids={inferred_num_ids}, embedding_dim={embedding_dim}, "
               f"hidden_dims={hidden_dims}, latent_dim={latent_dim}, heads={encoder_heads}, "
               f"mlp_hidden={mlp_hidden}")
    
    return {
        'num_ids': inferred_num_ids,
        'in_channels': in_channels,
        'hidden_dims': hidden_dims,
        'latent_dim': latent_dim,
        'encoder_heads': encoder_heads,
        'decoder_heads': decoder_heads,
        'embedding_dim': embedding_dim,
        'mlp_hidden': mlp_hidden,
    }


def infer_gat_architecture(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Infer GATWithJK model architecture from checkpoint state dict.
    
    GATWithJK architecture:
    - id_embedding: (num_ids, embedding_dim)
    - convs.0.lin.weight: (hidden_channels * heads, gat_in_dim) where gat_in_dim = embedding_dim + in_channels - 1
    - convs.{i}.att_src: (1, heads, hidden_channels)
    - fc_layers: series of linear layers ending in output dim
    """
    # Infer from embedding layer
    inferred_num_ids, embedding_dim = state_dict['id_embedding.weight'].shape
    
    # Count conv layers (num_layers)
    num_layers = 0
    for i in range(20):
        if f'convs.{i}.lin.weight' in state_dict:
            num_layers += 1
        else:
            break
    
    # Infer heads and hidden_channels from first conv attention
    heads, hidden_channels = state_dict['convs.0.att_src'].shape[1:3]
    
    # Infer in_channels from first conv input dim
    # gat_in_dim = embedding_dim + (in_channels - 1)
    first_conv_in = state_dict['convs.0.lin.weight'].shape[1]
    in_channels = first_conv_in - embedding_dim + 1
    
    # Infer output dim from last fc layer
    # Find the last fc layer
    num_fc_layers = 0
    out_channels = 2  # default
    for i in range(20):
        key = f'fc_layers.{i}.weight'
        if key in state_dict:
            out_channels = state_dict[key].shape[0]
            num_fc_layers += 1
    
    # num_fc_layers in GATWithJK counts actual Linear layers (bias too)
    # The Sequential has: Linear, ReLU, Dropout, Linear, ReLU, Dropout, ...
    # So we count by looking at weight keys
    fc_layer_count = len([k for k in state_dict if k.startswith('fc_layers.') and k.endswith('.weight')])
    
    logger.info(f"Inferred GAT: num_ids={inferred_num_ids}, embedding_dim={embedding_dim}, "
               f"hidden_channels={hidden_channels}, num_layers={num_layers}, heads={heads}, "
               f"out_channels={out_channels}, num_fc_layers={fc_layer_count}")
    
    return {
        'num_ids': inferred_num_ids,
        'in_channels': in_channels,
        'hidden_channels': hidden_channels,
        'out_channels': out_channels,
        'num_layers': num_layers,
        'heads': heads,
        'embedding_dim': embedding_dim,
        'num_fc_layers': fc_layer_count,
    }


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
        
        # Use config.required_artifacts() first - it has the correct canonical paths
        artifacts = self.config.required_artifacts()
        ae_path = artifacts.get('autoencoder')
        classifier_path = artifacts.get('classifier')
        
        # Fallback to PathResolver if artifacts don't exist
        if not ae_path or not Path(ae_path).exists():
            path_resolver = PathResolver(self.config)
            ae_path = path_resolver.resolve_autoencoder_path() or ae_path
        if not classifier_path or not Path(classifier_path).exists():
            path_resolver = PathResolver(self.config)
            classifier_path = path_resolver.resolve_classifier_path() or classifier_path

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
        """Build prediction caches from pre-trained models.
        
        Architecture is INFERRED from checkpoint shapes, not config defaults.
        This ensures we load models correctly regardless of config mismatches.
        """
        
        # Load checkpoints
        ae_ckpt = self.model_manager.load_state_dict(ae_path)
        classifier_ckpt = self.model_manager.load_state_dict(classifier_path)
        
        # Import model classes directly
        from src.models.vgae import GraphAutoencoderNeighborhood
        from src.models.models import GATWithJK
        
        # INFER VGAE architecture from checkpoint (not config)
        vgae_arch = infer_vgae_architecture(ae_ckpt)
        ae_model = GraphAutoencoderNeighborhood(
            num_ids=vgae_arch['num_ids'],
            in_channels=vgae_arch['in_channels'],
            hidden_dims=vgae_arch['hidden_dims'],
            latent_dim=vgae_arch['latent_dim'],
            encoder_heads=vgae_arch['encoder_heads'],
            decoder_heads=vgae_arch['decoder_heads'],
            embedding_dim=vgae_arch['embedding_dim'],
            mlp_hidden=vgae_arch['mlp_hidden'],
        )
        
        # INFER GAT architecture from checkpoint (not config)
        gat_arch = infer_gat_architecture(classifier_ckpt)
        classifier_model = GATWithJK(
            num_ids=gat_arch['num_ids'],
            in_channels=gat_arch['in_channels'],
            hidden_channels=gat_arch['hidden_channels'],
            out_channels=gat_arch['out_channels'],
            num_layers=gat_arch['num_layers'],
            heads=gat_arch['heads'],
            embedding_dim=gat_arch['embedding_dim'],
            num_fc_layers=gat_arch['num_fc_layers'],
        )
        
        # Load weights
        ae_model.load_state_dict(ae_ckpt)
        classifier_model.load_state_dict(classifier_ckpt)
        
        logger.info("✓ Models loaded")
        
        # Build prediction caches (returns dicts with 15D features)
        logger.info("🔄 Building prediction caches with 15D state space")
        train_features, val_features = create_fusion_prediction_cache(
                autoencoder=ae_model,
                classifier=classifier_model,
                train_loader=train_loader,
                val_loader=val_loader,
                dataset_name=self.config.dataset.name,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                cache_dir='cache/fusion'
            )

        logger.info(f"✓ 15D Caches built: {len(train_features['labels'])} train, {len(val_features['labels'])} val samples")

        # Create fusion datasets with 15D features
        train_fusion_dataset = FusionPredictionCache(
            vgae_errors=train_features['vgae_errors'],
            vgae_latent=train_features['vgae_latent'],
            vgae_confidence=train_features['vgae_confidence'],
            gat_logits=train_features['gat_logits'],
            gat_embeddings=train_features['gat_embeddings'],
            gat_confidence=train_features['gat_confidence'],
            labels=train_features['labels']
        )

        val_fusion_dataset = FusionPredictionCache(
            vgae_errors=val_features['vgae_errors'],
            vgae_latent=val_features['vgae_latent'],
            vgae_confidence=val_features['vgae_confidence'],
            gat_logits=val_features['gat_logits'],
            gat_embeddings=val_features['gat_embeddings'],
            gat_confidence=val_features['gat_confidence'],
            labels=val_features['labels']
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
        mp_ctx = "spawn" if num_workers > 0 else None

        fusion_train_loader = DataLoader(
            train_dataset,
            batch_size=fusion_cfg.fusion_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            multiprocessing_context=mp_ctx,
        )

        fusion_val_loader = DataLoader(
            val_dataset,
            batch_size=fusion_cfg.fusion_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            multiprocessing_context=mp_ctx,
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
        # Use consistent filename: dqn_fusion.pth (matches trainer.py model_name pattern)
        agent_path = self.paths['model_save_dir'] / 'dqn_fusion.pth'
        
        # Save using agent's built-in method
        fusion_model.fusion_agent.save_agent(str(agent_path))
        logger.info(f"✓ Fusion agent saved to {agent_path}")

        # Also save compact DQN state dict for compatibility with legacy names
        try:
            agent_state_name = f"fusion_agent_{self.config.dataset.name}.pth"
            self.model_manager.save_state_dict(
                fusion_model, 
                self.paths['model_save_dir'], 
                agent_state_name
            )
        except Exception as e:
            logger.warning(f"Could not save compact DQN agent state dict: {e}")
