"""
Model loading utilities for visualization scripts.

Uses config-driven approach to load trained models from checkpoints.
Pattern: Checkpoint → Frozen Config → LightningModule → Model

This ensures visualizations use the exact same model architecture as training/evaluation.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch.nn as nn

from src.config.frozen_config import load_frozen_config
from src.training.lightning_modules import (
    VAELightningModule,
    GATLightningModule,
    FusionLightningModule
)
from src.models.vgae import GraphAutoencoderNeighborhood
from src.models.models import GATWithJK

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Utility class for loading trained models for visualization.

    Applies config-driven loading pattern to ensure consistency with training/evaluation.

    Example:
        loader = ModelLoader()
        model, config = loader.load_model(
            checkpoint_path="experimentruns/.../models/vgae_teacher.pth"
        )
        embeddings = loader.extract_embeddings(model, data)
    """

    def __init__(self, device: str = 'auto'):
        """
        Initialize model loader.

        Args:
            device: 'auto', 'cuda', 'cpu', or specific device like 'cuda:0'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"ModelLoader initialized with device: {self.device}")

    def _discover_frozen_config(self, checkpoint_path: str) -> Optional[str]:
        """
        Auto-discover frozen_config.json relative to checkpoint.

        Standard experiment structure:
        experiment_dir/
        ├── configs/
        │   └── frozen_config_TIMESTAMP.json  ← Config here
        └── models/
            └── model.pth  ← Checkpoint here

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            Path to frozen config JSON, or None if not found
        """
        checkpoint_dir = Path(checkpoint_path).parent

        # Pattern 1: ../configs/frozen_config*.json (most common)
        configs_dir = checkpoint_dir.parent / "configs"
        if configs_dir.exists():
            configs = sorted(configs_dir.glob("frozen_config*.json"))
            if configs:
                logger.info(f"Found frozen config: {configs[-1]}")
                return str(configs[-1])

        # Pattern 2: Same directory as checkpoint (legacy)
        config_same_dir = checkpoint_dir / "frozen_config.json"
        if config_same_dir.exists():
            return str(config_same_dir)

        # Pattern 3: ../../configs/ (if checkpoint nested deeper)
        configs_dir_up2 = checkpoint_dir.parent.parent / "configs"
        if configs_dir_up2.exists():
            configs = sorted(configs_dir_up2.glob("frozen_config*.json"))
            if configs:
                return str(configs[-1])

        logger.warning(f"No frozen config found for checkpoint: {checkpoint_path}")
        return None

    def load_model(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        num_ids: Optional[int] = None,
        return_lightning_module: bool = False
    ) -> Tuple[nn.Module, object]:
        """
        Load model using frozen config + LightningModule.

        This approach:
        1. Loads frozen config (has all parameters)
        2. Infers num_ids from checkpoint (handles cross-dataset evaluation)
        3. Instantiates appropriate LightningModule (handles model building)
        4. Loads checkpoint weights
        5. Returns correctly configured model

        Args:
            checkpoint_path: Path to model checkpoint (.pth)
            config_path: Optional explicit path to frozen config (auto-discovered if None)
            num_ids: Optional num_ids override (inferred from checkpoint if None)
            return_lightning_module: If True, return full LightningModule instead of just model

        Returns:
            Tuple of (model, config):
            - model: The loaded PyTorch model (or LightningModule if flag set)
            - config: The loaded frozen configuration object

        Example:
            model, config = loader.load_model(
                checkpoint_path="experimentruns/.../models/vgae_teacher.pth"
            )
        """
        # Step 1: Load checkpoint and infer num_ids
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Strip 'model.' prefix if present
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # Infer num_ids from checkpoint
        if num_ids is None and 'id_embedding.weight' in state_dict:
            num_ids = state_dict['id_embedding.weight'].shape[0]
            logger.info(f"Inferred num_ids from checkpoint: {num_ids}")
        elif num_ids is None:
            logger.warning("Could not infer num_ids from checkpoint, using default 1000")
            num_ids = 1000

        # Step 2: Load frozen config
        if config_path is None:
            config_path = self._discover_frozen_config(checkpoint_path)

        if config_path is None:
            raise ValueError(
                f"Could not find frozen config for checkpoint: {checkpoint_path}\n"
                "Please provide config_path explicitly or ensure frozen_config*.json exists in ../configs/"
            )

        logger.info(f"Loading frozen config: {config_path}")
        config = load_frozen_config(config_path)

        # Step 3: Instantiate appropriate LightningModule
        model_type = config.model.type
        training_mode = config.training.mode

        logger.info(f"Building {model_type} model (mode: {training_mode}, size: {config.model_size})")

        if model_type in ["vgae", "vgae_student"]:
            lightning_module = VAELightningModule(cfg=config, num_ids=num_ids)
        elif model_type in ["gat", "gat_student"]:
            lightning_module = GATLightningModule(cfg=config, num_ids=num_ids)
        elif training_mode == "fusion":
            # For fusion, need to load sub-models separately
            logger.warning("Fusion models require special handling - use load_fusion_model() instead")
            raise ValueError("Use load_fusion_model() for fusion models")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Step 4: Load checkpoint weights
        lightning_module.model.load_state_dict(state_dict, strict=False)

        # Step 5: Set to eval mode and move to device
        lightning_module.eval()
        lightning_module.to(self.device)

        logger.info(f"Successfully loaded {model_type} model with config-driven architecture")

        if return_lightning_module:
            return lightning_module, config
        else:
            return lightning_module.model, config

    def load_fusion_model(
        self,
        checkpoint_path: str,
        vgae_checkpoint_path: str,
        gat_checkpoint_path: str,
        config_path: Optional[str] = None,
        num_ids: Optional[int] = None
    ) -> Tuple[nn.Module, nn.Module, nn.Module, object]:
        """
        Load fusion model with its sub-models.

        Args:
            checkpoint_path: Path to DQN fusion checkpoint
            vgae_checkpoint_path: Path to VGAE checkpoint
            gat_checkpoint_path: Path to GAT checkpoint
            config_path: Optional path to fusion config
            num_ids: Optional num_ids override

        Returns:
            Tuple of (vgae_model, gat_model, dqn_model, config)
        """
        # Load VGAE
        logger.info("Loading VGAE sub-model...")
        vgae_model, vgae_config = self.load_model(vgae_checkpoint_path, num_ids=num_ids)

        # Load GAT
        logger.info("Loading GAT sub-model...")
        gat_model, gat_config = self.load_model(gat_checkpoint_path, num_ids=num_ids)

        # Load DQN fusion config
        if config_path is None:
            config_path = self._discover_frozen_config(checkpoint_path)

        config = load_frozen_config(config_path)

        # Load DQN fusion model
        logger.info("Loading DQN fusion model...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Instantiate fusion module
        lightning_module = FusionLightningModule(
            cfg=config,
            num_ids=num_ids,
            autoencoder_model=vgae_model,
            classifier_model=gat_model
        )

        lightning_module.load_state_dict(state_dict, strict=False)
        lightning_module.eval()
        lightning_module.to(self.device)

        dqn_model = lightning_module.fusion_agent

        logger.info("Successfully loaded fusion model with all sub-models")

        return vgae_model, gat_model, dqn_model, config

    @torch.no_grad()
    def extract_vgae_embeddings(
        self,
        model: nn.Module,
        data_list: list,
        batch_size: int = 64
    ) -> Dict[str, torch.Tensor]:
        """
        Extract VGAE latent embeddings (z) from data.

        Args:
            model: VGAE model
            data_list: List of PyG Data objects
            batch_size: Batch size for processing

        Returns:
            Dictionary with keys:
            - 'z_mean': Latent mean vectors [N, latent_dim]
            - 'z_log_var': Latent log variance [N, latent_dim]
            - 'z': Sampled latent vectors [N, latent_dim]
            - 'reconstruction': Reconstructed features [N, num_nodes, feature_dim]
            - 'labels': Graph labels [N]
        """
        from torch_geometric.loader import DataLoader

        model.eval()

        z_means = []
        z_log_vars = []
        zs = []
        labels = []

        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

        for batch in loader:
            batch = batch.to(self.device)

            # Forward pass through encoder
            z_mean, z_log_var = model.encode(batch.x, batch.edge_index, batch.batch)

            # Sample z
            z = model.reparameterize(z_mean, z_log_var)

            z_means.append(z_mean.cpu())
            z_log_vars.append(z_log_var.cpu())
            zs.append(z.cpu())
            labels.append(batch.y.cpu())

        embeddings = {
            'z_mean': torch.cat(z_means, dim=0),
            'z_log_var': torch.cat(z_log_vars, dim=0),
            'z': torch.cat(zs, dim=0),
            'labels': torch.cat(labels, dim=0)
        }

        logger.info(f"Extracted VGAE embeddings: z shape = {embeddings['z'].shape}")

        return embeddings

    @torch.no_grad()
    def extract_gat_embeddings(
        self,
        model: nn.Module,
        data_list: list,
        batch_size: int = 64
    ) -> Dict[str, torch.Tensor]:
        """
        Extract GAT pre-pooling embeddings from data.

        Args:
            model: GAT model
            data_list: List of PyG Data objects
            batch_size: Batch size for processing

        Returns:
            Dictionary with keys:
            - 'pre_pooling': Pre-pooling embeddings [N, hidden_dim]
            - 'post_pooling': Post-pooling embeddings [N, hidden_dim]
            - 'logits': Classification logits [N, num_classes]
            - 'labels': Graph labels [N]
        """
        from torch_geometric.loader import DataLoader

        model.eval()

        pre_pooling_embs = []
        post_pooling_embs = []
        logits_list = []
        labels = []

        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

        for batch in loader:
            batch = batch.to(self.device)

            # Forward pass
            logits = model(batch.x, batch.edge_index, batch.batch)

            # Extract embeddings (assuming model stores them)
            if hasattr(model, 'pre_pooling_embeddings'):
                pre_pooling_embs.append(model.pre_pooling_embeddings.cpu())

            if hasattr(model, 'post_pooling_embeddings'):
                post_pooling_embs.append(model.post_pooling_embeddings.cpu())

            logits_list.append(logits.cpu())
            labels.append(batch.y.cpu())

        embeddings = {
            'logits': torch.cat(logits_list, dim=0),
            'labels': torch.cat(labels, dim=0)
        }

        if pre_pooling_embs:
            embeddings['pre_pooling'] = torch.cat(pre_pooling_embs, dim=0)

        if post_pooling_embs:
            embeddings['post_pooling'] = torch.cat(post_pooling_embs, dim=0)

        logger.info(f"Extracted GAT embeddings: logits shape = {embeddings['logits'].shape}")

        return embeddings

    @torch.no_grad()
    def compute_vgae_reconstruction_errors(
        self,
        model: nn.Module,
        data_list: list,
        batch_size: int = 64
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VGAE reconstruction errors (for visualization).

        Args:
            model: VGAE model
            data_list: List of PyG Data objects
            batch_size: Batch size for processing

        Returns:
            Dictionary with keys:
            - 'node_error': Node reconstruction error [N]
            - 'neighbor_error': Neighbor reconstruction error [N]
            - 'id_error': CAN ID reconstruction error [N]
            - 'combined_error': Weighted combination [N]
            - 'labels': Graph labels [N]
        """
        from torch_geometric.loader import DataLoader

        model.eval()

        node_errors = []
        neighbor_errors = []
        id_errors = []
        combined_errors = []
        labels = []

        # Weights used in DQN state (from fusion.py)
        weights = [0.4, 0.35, 0.25]

        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

        for batch in loader:
            batch = batch.to(self.device)

            # Forward pass
            z_mean, z_log_var = model.encode(batch.x, batch.edge_index, batch.batch)
            z = model.reparameterize(z_mean, z_log_var)

            # Reconstruction
            x_recon = model.decode_features(z, batch.batch)

            # Compute errors (per graph)
            # Node reconstruction error
            node_error = torch.nn.functional.mse_loss(
                x_recon, batch.x, reduction='none'
            ).mean(dim=1)  # Average over features

            # Aggregate to graph-level (simple mean per graph)
            # TODO: Implement proper graph-level aggregation using batch index

            # For now, simplified version
            node_errors.append(node_error.mean().cpu())
            neighbor_errors.append(node_error.mean().cpu())  # Placeholder
            id_errors.append(node_error.mean().cpu())  # Placeholder

            # Combined error
            combined = (
                weights[0] * node_error.mean() +
                weights[1] * node_error.mean() +
                weights[2] * node_error.mean()
            )
            combined_errors.append(combined.cpu())

            labels.append(batch.y.cpu())

        errors = {
            'node_error': torch.tensor(node_errors),
            'neighbor_error': torch.tensor(neighbor_errors),
            'id_error': torch.tensor(id_errors),
            'combined_error': torch.tensor(combined_errors),
            'labels': torch.cat(labels, dim=0)
        }

        logger.info(f"Computed VGAE reconstruction errors for {len(data_list)} graphs")

        return errors


# Convenience function
def load_model_for_visualization(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = 'auto'
) -> Tuple[nn.Module, object]:
    """
    Convenience function to load a model for visualization.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to frozen config (auto-discovered if None)
        device: Device to load model on ('auto', 'cuda', 'cpu')

    Returns:
        Tuple of (model, config)

    Example:
        model, config = load_model_for_visualization(
            checkpoint_path="experimentruns/.../models/vgae_teacher.pth"
        )
    """
    loader = ModelLoader(device=device)
    return loader.load_model(checkpoint_path, config_path)
