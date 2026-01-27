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
        logger.info(f"📊 Datamodule created with batch_size={datamodule.batch_size}")

        # Create GAT model
        gat_model = self._create_gat_model(num_ids)
        logger.info(f"🧠 GAT model created")

        # Optimize batch size
        logger.info(f"🔧 About to optimize batch size (current: {datamodule.batch_size})")
        gat_model = self._optimize_batch_size_for_curriculum(gat_model, datamodule)
        logger.info(f"✅ Batch size optimization complete (final: {datamodule.batch_size})")
        
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
        logger.info("📊 Separating normal and attack graphs...")
        train_normal = [g for g in full_dataset if g.y.item() == 0]
        train_attack = [g for g in full_dataset if g.y.item() == 1]
        val_normal = [g for g in val_dataset if g.y.item() == 0]
        val_attack = [g for g in val_dataset if g.y.item() == 1]

        logger.info(f"📊 Separated dataset: {len(train_normal)} normal + {len(train_attack)} attack training")
        logger.info(f"📊 Validation: {len(val_normal)} normal + {len(val_attack)} attack")
        logger.info(f"📊 Total graphs: {len(train_normal) + len(train_attack) + len(val_normal) + len(val_attack)}")
        
        # Load trained VGAE for hard mining
        vgae_path = self._resolve_vgae_path()
        logger.info(f"Loading VGAE from: {vgae_path}")
        
        # Load VGAE - handle both state dict and Lightning checkpoint formats
        vgae_model = self._load_vgae_model(vgae_path, num_ids)
        vgae_model.eval()
        
        # Create enhanced datamodule with memory-aware batch sizing
        base_batch_size = getattr(self.config.training, 'batch_size', 64)

        # Apply curriculum memory multiplier for high-memory datasets
        # Curriculum mode has 2x memory overhead (double dataset load + VGAE model)
        memory_multiplier = getattr(self.config.training, 'curriculum_memory_multiplier', 1.0)
        curriculum_batch_size = int(base_batch_size * memory_multiplier)

        logger.info(
            f"📊 Curriculum batch size: {curriculum_batch_size} "
            f"(base: {base_batch_size}, multiplier: {memory_multiplier})"
        )

        datamodule = EnhancedCANGraphDataModule(
            train_normal=train_normal,
            train_attack=train_attack,
            val_normal=val_normal,
            val_attack=val_attack,
            vgae_model=vgae_model,
            batch_size=curriculum_batch_size,
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
    
    def _load_vgae_model(self, vgae_path: Path, num_ids: int):
        """Load VGAE model from checkpoint, handling both state dict and Lightning formats.
        
        This method infers model architecture from the checkpoint shapes when needed,
        making it robust to different training configurations.
        
        Args:
            vgae_path: Path to VGAE checkpoint
            num_ids: Number of unique CAN IDs for model initialization
            
        Returns:
            VGAE model ready for inference
        """
        from src.models.vgae import GraphAutoencoderNeighborhood
        
        # Load checkpoint
        checkpoint = torch.load(str(vgae_path), map_location='cpu', weights_only=False)
        
        # Determine format and extract state dict
        if isinstance(checkpoint, dict):
            if 'pytorch-lightning_version' in checkpoint:
                # Full Lightning checkpoint - use load_from_checkpoint
                logger.info("Loading VGAE from Lightning checkpoint format")
                vgae_module = VAELightningModule.load_from_checkpoint(
                    str(vgae_path),
                    cfg=self.config,
                    num_ids=num_ids,
                    map_location='cpu'
                )
                return vgae_module.model
            elif 'state_dict' in checkpoint:
                # State dict wrapped in dict
                logger.info("Loading VGAE from wrapped state dict format")
                state_dict = checkpoint['state_dict']
            elif any(k.startswith(('encoder', 'decoder', 'id_embedding')) for k in checkpoint.keys()):
                # Raw state dict (OrderedDict with model parameter names)
                logger.info("Loading VGAE from raw state dict format")
                state_dict = checkpoint
            else:
                # Unknown dict format, try as state dict
                logger.warning(f"Unknown checkpoint dict keys: {list(checkpoint.keys())[:5]}, trying as state dict")
                state_dict = checkpoint
        else:
            raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
        
        # Infer complete model architecture from checkpoint shapes
        # This makes loading robust to different training configurations
        inferred_num_ids, embedding_dim = state_dict['id_embedding.weight'].shape
        latent_dim = state_dict['z_mean.weight'].shape[0]  # Output dim of z_mean layer
        
        # Infer hidden_dims from encoder batch norm layers (NOT including latent_dim)
        # The model expects hidden_dims to be the schedule BEFORE latent projection
        hidden_dims = []
        for i in range(10):  # Check up to 10 layers
            key = f'encoder_bns.{i}.weight'
            if key in state_dict:
                hidden_dims.append(state_dict[key].shape[0])
            else:
                break
        
        # Infer encoder_heads from first attention layer shape
        encoder_heads = state_dict['encoder_layers.0.att_src'].shape[1]
        
        # Infer decoder_heads from first decoder attention layer shape
        decoder_heads = state_dict['decoder_layers.0.att_src'].shape[1]
        
        # Infer in_channels from first encoder layer input dimension
        # Model uses: gat_in_dim = embedding_dim + (in_channels - 1)
        # So: in_channels = gat_in_dim - embedding_dim + 1
        first_layer_in = state_dict['encoder_layers.0.lin.weight'].shape[1]
        in_channels = first_layer_in - embedding_dim + 1
        
        logger.info(f"Inferred VGAE architecture: num_ids={inferred_num_ids}, "
                   f"embedding_dim={embedding_dim}, hidden_dims={hidden_dims}, "
                   f"latent_dim={latent_dim}, encoder_heads={encoder_heads}, "
                   f"decoder_heads={decoder_heads}, in_channels={in_channels}")
        
        # Create VGAE model with inferred architecture
        vgae_model = GraphAutoencoderNeighborhood(
            num_ids=inferred_num_ids,
            in_channels=in_channels,
            hidden_dims=hidden_dims,  # Just the encoder schedule, not including latent
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            encoder_heads=encoder_heads,
            decoder_heads=decoder_heads,
        )
        
        # Handle potential key mismatches (e.g., 'model.' prefix from Lightning)
        clean_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace('model.', '') if key.startswith('model.') else key
            clean_state_dict[clean_key] = value
        
        vgae_model.load_state_dict(clean_state_dict, strict=True)
        logger.info(f"✅ Loaded VGAE model with {sum(p.numel() for p in vgae_model.parameters())} parameters")
        
        return vgae_model
    
    def _resolve_vgae_path(self) -> Path:
        """Resolve path to trained VGAE model using flexible discovery.
        
        Resolution priority:
        1. Explicit path from config.training.vgae_model_path (if it exists)
        2. Path from config.required_artifacts() (canonical location)
        3. Flexible glob-based discovery via PathResolver.discover_artifact()
        
        This approach is more robust than hardcoded paths - it will find
        any VGAE model (vgae*.pth) in the canonical autoencoder directory.
        """
        # Priority 1: Check explicit config path first (if file exists)
        explicit_path = getattr(self.config.training, 'vgae_model_path', None)
        if explicit_path and Path(explicit_path).exists():
            logger.info(f"Using explicitly configured VGAE path: {explicit_path}")
            return Path(explicit_path)
        
        # Priority 2: Use required_artifacts() canonical path
        artifacts = self.config.required_artifacts()
        artifact_path = artifacts.get('vgae')
        if artifact_path and Path(artifact_path).exists():
            logger.info(f"Using VGAE from required_artifacts: {artifact_path}")
            return Path(artifact_path)
        
        # Priority 3: Flexible discovery - glob for any vgae*.pth in canonical dir
        try:
            discovered = self.path_resolver.discover_artifact('vgae', require_exists=True)
            if discovered:
                logger.info(f"Discovered VGAE via glob search: {discovered}")
                return discovered
        except FileNotFoundError:
            pass
        
        # Provide helpful error with canonical expected location
        expected_path = artifacts.get('vgae', 'unknown')
        raise FileNotFoundError(
            f"Curriculum training requires VGAE model.\n"
            f"Expected location: {expected_path}\n"
            f"Please run autoencoder training first to create this model."
        )
    
    def _optimize_batch_size_for_curriculum(
        self,
        model: GATLightningModule,
        datamodule: EnhancedCANGraphDataModule
    ) -> GATLightningModule:
        """Optimize batch size for curriculum learning."""

        logger.info("📊 _optimize_batch_size_for_curriculum() called")
        logger.info(f"   Current datamodule.batch_size: {datamodule.batch_size}")
        logger.info(f"   Current dataset size: {len(datamodule.train_dataset)}")

        optimize_batch = getattr(self.config.training, 'optimize_batch_size', True)
        logger.info(f"   optimize_batch_size setting: {optimize_batch}")

        if not optimize_batch:
            # Use conservative batch size if optimization disabled
            logger.info("   Optimization disabled, using conservative batch size...")
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
        logger.info("   Creating max-size dataset for tuning...")
        original_state = datamodule.create_max_size_dataset_for_tuning()
        logger.info(f"   Max dataset size: {len(datamodule.train_dataset)}")
        
        try:
            logger.info("   Creating BatchSizeOptimizer...")
            safety_factor = getattr(self.config.training, 'graph_memory_safety_factor', 0.5)
            max_trials = getattr(self.config.training, 'max_batch_size_trials', 10)
            mode = getattr(self.config.training, 'batch_size_mode', 'power')

            logger.info(f"   Optimizer config: safety_factor={safety_factor}, max_trials={max_trials}, mode={mode}")

            optimizer = BatchSizeOptimizer(
                accelerator=self.config.trainer.accelerator,
                devices=self.config.trainer.devices,
                graph_memory_safety_factor=safety_factor,
                max_batch_size_trials=max_trials,
                batch_size_mode=mode
            )

            logger.info(f"   Calling optimizer.optimize_with_datamodule()...")
            logger.info(f"   Pre-optimization: datamodule.batch_size={datamodule.batch_size}")

            safe_batch_size = optimizer.optimize_with_datamodule(model, datamodule)

            logger.info(f"   Returned safe_batch_size: {safe_batch_size}")
            logger.info(f"   Assigning to datamodule.batch_size...")
            datamodule.batch_size = safe_batch_size
            logger.info(f"   Post-assignment: datamodule.batch_size={datamodule.batch_size}")
            logger.info(f"✅ Batch size optimized for curriculum learning: {safe_batch_size}")

        except Exception as e:
            logger.warning(
                f"❌ Batch size optimization failed: {e}. "
                "Using default batch size."
            )
            import traceback
            logger.warning(f"Exception traceback:\n{traceback.format_exc()}")

            # Use conservative fallback
            safe_batch_size = datamodule.get_conservative_batch_size(
                getattr(self.config.training, 'batch_size', 64)
            )
            logger.info(f"   Conservative fallback batch size: {safe_batch_size}")
            datamodule.batch_size = safe_batch_size

        finally:
            # Restore dataset to original state
            logger.info("   Restoring dataset to original state...")
            if original_state:
                datamodule.restore_dataset_after_tuning(original_state)
                logger.info(f"   Restored dataset size: {len(datamodule.train_dataset)}")

        logger.info(f"📊 Final state after optimization:")
        logger.info(f"   datamodule.batch_size: {datamodule.batch_size}")
        logger.info(f"   dataset size: {len(datamodule.train_dataset)}")

        return model
    
    def _create_curriculum_trainer(self) -> pl.Trainer:
        """Create trainer with curriculum callback."""
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
        from lightning.pytorch.loggers import CSVLogger
        
        curriculum_callback = CurriculumCallback()
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.paths['checkpoint_dir']),
            filename=f'gat_curriculum_{{epoch:02d}}_{{val_loss:.3f}}',
            save_top_k=getattr(self.config.training, 'save_top_k', 3),
            monitor=getattr(self.config.training, 'monitor_metric', 'val_loss'),
            mode=getattr(self.config.training, 'monitor_mode', 'min'),
            auto_insert_metric_name=False
        )
        
        early_stop = EarlyStopping(
            monitor=getattr(self.config.training, 'monitor_metric', 'val_loss'),
            patience=getattr(
                self.config.training,
                'early_stopping_patience',
                50
            ),
            mode=getattr(self.config.training, 'monitor_mode', 'min'),
            verbose=True
        )
        
        csv_logger = CSVLogger(
            save_dir=str(self.paths['log_dir']),
            name=f'curriculum_{self.config.dataset.name}'
        )
        
        return pl.Trainer(
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            precision=getattr(self.config.training, 'precision', '32-true'),
            max_epochs=self.config.training.max_epochs,
            gradient_clip_val=getattr(
                self.config.training,
                'gradient_clip_val',
                1.0
            ),
            callbacks=[checkpoint_callback, early_stop, curriculum_callback],
            logger=csv_logger,
            enable_checkpointing=True,
            log_every_n_steps=getattr(self.config.training, 'log_every_n_steps', 50),
            enable_progress_bar=True
        )
    
    def _save_curriculum_model(
        self,
        model: GATLightningModule,
        trainer: pl.Trainer
    ):
        """Save final curriculum-trained model as state dict."""
        # Use consistent filename: gat_curriculum.pth (matches trainer.py and fusion expectations)
        model_path = self.paths['model_save_dir'] / "gat_curriculum.pth"
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
