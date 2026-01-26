"""
Knowledge Distillation Helper Module

Provides KDHelper class for knowledge distillation in VAE and GAT lightning modules.
Handles:
- Teacher model loading and freezing
- Projection layers for dimension mismatch (student → teacher)
- VGAE dual-signal KD (latent + reconstruction)
- GAT soft label KD with temperature scaling
- Memory-efficient training with AMP support

Usage:
    kd_helper = KDHelper(cfg, student_model, model_type="vgae")

    # In training_step:
    if kd_helper.enabled:
        teacher_outputs = kd_helper.get_teacher_outputs(batch)
        kd_loss = kd_helper.compute_vgae_kd_loss(student_z, student_recon, teacher_outputs)
        loss = kd_helper.combine_losses(task_loss, kd_loss)
"""

import gc
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def cleanup_memory():
    """Release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class KDHelper:
    """
    Helper for knowledge distillation in VAE and GAT lightning modules.

    Supports two model types:
    - "vgae": Dual-signal distillation (latent space + reconstruction)
    - "gat": Standard soft label distillation with temperature scaling

    Handles dimension mismatch between teacher (larger) and student (smaller)
    models via learned projection layers.
    """

    def __init__(
        self,
        cfg: DictConfig,
        student_model: nn.Module,
        model_type: str = "vgae"
    ):
        """
        Initialize KD helper.

        Args:
            cfg: Full config with training.use_knowledge_distillation, etc.
            student_model: The student model being trained
            model_type: "vgae" or "gat"
        """
        self.cfg = cfg
        self.model_type = model_type

        # Check if KD is enabled
        self.enabled = getattr(cfg.training, 'use_knowledge_distillation', False)
        if not self.enabled:
            logger.info("Knowledge distillation DISABLED")
            self.teacher = None
            self.projection_layer = None
            return

        # KD hyperparameters
        self.temperature = getattr(cfg.training, 'distillation_temperature', 4.0)
        self.alpha = getattr(cfg.training, 'distillation_alpha', 0.7)

        # Get device from student model
        self.device = next(student_model.parameters()).device

        # Load and freeze teacher model
        teacher_path = getattr(cfg.training, 'teacher_model_path', None)
        if not teacher_path:
            raise ValueError("KD enabled but no teacher_model_path specified in config")

        self.teacher = self._load_teacher(teacher_path)
        self._freeze_teacher()

        # Setup projection layer for dimension mismatch
        self.projection_layer = self._setup_projection_layer(student_model)

        logger.info(f"✅ Knowledge distillation ENABLED")
        logger.info(f"   Model type: {model_type}")
        logger.info(f"   Temperature: {self.temperature}")
        logger.info(f"   Alpha (KD weight): {self.alpha}")
        logger.info(f"   Teacher path: {teacher_path}")

    def _load_teacher(self, path: str) -> nn.Module:
        """
        Load teacher model from checkpoint.

        Args:
            path: Path to teacher .pth or .ckpt file

        Returns:
            Loaded teacher model (on correct device)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Teacher model not found: {path}")

        logger.info(f"Loading teacher model from {path}")

        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Lightning checkpoint format
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'model.' prefix if present
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()
                             if k.startswith('model.')}
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            # Pure state dict
            state_dict = checkpoint

        # Build teacher model based on type
        if self.model_type == "vgae":
            teacher = self._build_vgae_teacher(state_dict)
        elif self.model_type == "gat":
            teacher = self._build_gat_teacher(state_dict)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        teacher.load_state_dict(state_dict, strict=False)
        teacher = teacher.to(self.device)

        logger.info(f"✅ Teacher model loaded: {sum(p.numel() for p in teacher.parameters())} params")

        return teacher

    def _build_vgae_teacher(self, state_dict: Dict[str, Any]) -> nn.Module:
        """Build VGAE teacher model matching checkpoint."""
        from src.models.vgae import GraphAutoencoderNeighborhood

        # Infer dimensions from state dict
        # encoder.conv1.lin.weight shape: [hidden_dim * heads, input_dim]
        first_weight = state_dict.get('encoder.conv1.lin.weight',
                                      state_dict.get('encoder.conv_layers.0.lin.weight'))
        if first_weight is not None:
            hidden_heads = first_weight.shape[0]
            input_dim = first_weight.shape[1]
            # Assume 4 heads for teacher
            hidden_dim = hidden_heads // 4
        else:
            # Fallback to config
            input_dim = self.cfg.model.input_dim
            hidden_dim = 128  # Teacher default

        # Infer latent dim from mu weight
        mu_weight = state_dict.get('encoder.conv_mu.lin.weight')
        if mu_weight is not None:
            latent_dim = mu_weight.shape[0]
        else:
            latent_dim = 32  # Teacher default

        # Get num_ids from embedding
        id_emb = state_dict.get('encoder.id_embedding.weight')
        if id_emb is not None:
            num_ids = id_emb.shape[0]
        else:
            num_ids = 1000  # Default

        logger.info(f"   Teacher VGAE: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"latent_dim={latent_dim}, num_ids={num_ids}")

        return GraphAutoencoderNeighborhood(
            num_ids=num_ids,
            in_channels=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],  # Teacher uses larger dims
            latent_dim=latent_dim,
            encoder_heads=4,
            decoder_heads=4,
            embedding_dim=64,  # Teacher default
            dropout=0.15,
            batch_norm=True,
            use_checkpointing=False  # No need for checkpointing on frozen model
        )

    def _build_gat_teacher(self, state_dict: Dict[str, Any]) -> nn.Module:
        """Build GAT teacher model matching checkpoint."""
        from src.models.models import GATWithJK

        # Infer dimensions from state dict
        first_weight = state_dict.get('conv_layers.0.lin.weight')
        if first_weight is not None:
            hidden_heads = first_weight.shape[0]
            input_dim = first_weight.shape[1]
            hidden_channels = hidden_heads // 4  # Assume 4 heads
        else:
            input_dim = self.cfg.model.input_dim
            hidden_channels = 128  # Teacher default

        # Get num_ids from embedding
        id_emb = state_dict.get('id_embedding.weight')
        if id_emb is not None:
            num_ids = id_emb.shape[0]
        else:
            num_ids = 1000

        logger.info(f"   Teacher GAT: input_dim={input_dim}, hidden_channels={hidden_channels}, "
                   f"num_ids={num_ids}")

        return GATWithJK(
            in_channels=input_dim,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=4,
            heads=4,
            dropout=0.15,
            num_fc_layers=2,
            embedding_dim=64,
            num_ids=num_ids,
            use_checkpointing=False
        )

    def _freeze_teacher(self):
        """Freeze teacher model (no gradient computation)."""
        if self.teacher is None:
            return

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        logger.info("   Teacher frozen (no gradients)")

    def _setup_projection_layer(self, student_model: nn.Module) -> Optional[nn.Module]:
        """
        Setup projection layer if teacher and student have different latent dimensions.

        Projects student embeddings UP to teacher dimension for comparison.

        Args:
            student_model: The student model being trained

        Returns:
            Projection layer (nn.Linear) or None if dimensions match
        """
        if self.teacher is None:
            return None

        # Get latent dimensions
        teacher_latent_dim = getattr(self.teacher, 'latent_dim',
                                     getattr(self.teacher, 'hidden_channels', 32))
        student_latent_dim = getattr(student_model, 'latent_dim',
                                     getattr(student_model, 'hidden_channels', 16))

        if self.model_type == "vgae":
            # For VGAE, check encoder latent dim
            if hasattr(self.teacher, 'encoder'):
                teacher_latent_dim = getattr(self.teacher.encoder, 'latent_dim',
                                            getattr(self.teacher, 'latent_dim', 32))
            if hasattr(student_model, 'encoder'):
                student_latent_dim = getattr(student_model.encoder, 'latent_dim',
                                            getattr(student_model, 'latent_dim', 16))

        if teacher_latent_dim != student_latent_dim:
            projection_layer = nn.Linear(student_latent_dim, teacher_latent_dim).to(self.device)
            logger.info(f"   Projection layer: {student_latent_dim} → {teacher_latent_dim}")
            return projection_layer

        logger.info("   No projection needed (dimensions match)")
        return None

    def get_projection_parameters(self):
        """Get projection layer parameters for optimizer (if exists)."""
        if self.projection_layer is not None:
            return self.projection_layer.parameters()
        return []

    @torch.no_grad()
    def get_teacher_outputs(self, batch) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get teacher outputs for KD (no gradients).

        Args:
            batch: PyG Data batch

        Returns:
            Dict with model-specific outputs, or None if KD disabled
        """
        if not self.enabled or self.teacher is None:
            return None

        if self.model_type == "vgae":
            # VGAE returns: cont_out, canid_logits, neighbor_logits, z, kl_loss
            x = batch.x
            edge_index = batch.edge_index
            b = batch.batch

            cont_out, canid_logits, neighbor_logits, z, kl_loss = self.teacher(x, edge_index, b)

            return {
                'z': z,
                'cont_out': cont_out,
                'canid_logits': canid_logits,
                'neighbor_logits': neighbor_logits
            }

        elif self.model_type == "gat":
            # GAT returns logits
            logits = self.teacher(batch)
            return {'logits': logits}

        return None

    def compute_vgae_kd_loss(
        self,
        student_z: torch.Tensor,
        student_recon: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        VGAE-specific KD loss: distill BOTH latent space AND reconstruction.

        The dual-signal approach provides richer knowledge transfer:
        1. Latent space: MSE between projected student z and teacher z
        2. Reconstruction: MSE between student and teacher continuous outputs

        Args:
            student_z: Student latent vectors [batch_nodes, latent_dim]
            student_recon: Dict with 'cont_out' reconstruction
            teacher_outputs: Dict from get_teacher_outputs()

        Returns:
            Combined KD loss
        """
        if not self.enabled or teacher_outputs is None:
            return torch.tensor(0.0, device=self.device)

        # 1. Latent space distillation (with projection if needed)
        if self.projection_layer is not None:
            projected_student_z = self.projection_layer(student_z)
        else:
            projected_student_z = student_z

        # Handle potential shape mismatch from different graph sizes
        teacher_z = teacher_outputs['z']
        min_nodes = min(projected_student_z.shape[0], teacher_z.shape[0])

        latent_loss = F.mse_loss(
            projected_student_z[:min_nodes],
            teacher_z[:min_nodes]
        )

        # 2. Reconstruction distillation
        teacher_cont = teacher_outputs['cont_out']
        student_cont = student_recon['cont_out']
        min_nodes = min(student_cont.shape[0], teacher_cont.shape[0])

        recon_loss = F.mse_loss(
            student_cont[:min_nodes],
            teacher_cont[:min_nodes]
        )

        # Combine with equal weighting (can be tuned)
        kd_loss = 0.5 * latent_loss + 0.5 * recon_loss

        return kd_loss

    def compute_gat_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        GAT-specific KD loss: standard soft label distillation.

        Uses KL divergence with temperature-scaled softmax outputs.

        Args:
            student_logits: Student classification logits [batch, num_classes]
            teacher_logits: Teacher classification logits [batch, num_classes]

        Returns:
            KL divergence loss scaled by temperature^2
        """
        if not self.enabled or teacher_logits is None:
            return torch.tensor(0.0, device=self.device)

        # Soft labels with temperature scaling
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence (scaled by T^2 to match gradient magnitudes)
        kd_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return kd_loss

    def combine_losses(
        self,
        task_loss: torch.Tensor,
        kd_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Weighted combination of task loss and KD loss.

        Formula: alpha * kd_loss + (1 - alpha) * task_loss

        Args:
            task_loss: Original task loss (reconstruction, classification, etc.)
            kd_loss: Knowledge distillation loss

        Returns:
            Combined loss
        """
        if not self.enabled:
            return task_loss

        return self.alpha * kd_loss + (1 - self.alpha) * task_loss


def get_kd_safety_factor_key(dataset_name: str) -> str:
    """
    Get the key for KD-specific safety factor in batch_size_factors.json.

    Args:
        dataset_name: Dataset name (e.g., 'hcrl_sa')

    Returns:
        Key string with '_kd' suffix (e.g., 'hcrl_sa_kd')
    """
    return f"{dataset_name}_kd"
