"""Lightning modules for VGAE and GAT training."""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from ..config import PipelineConfig
from .utils import build_optimizer_dict, effective_batch_size, make_dataloader

log = logging.getLogger(__name__)


class VGAEModule(pl.LightningModule):
    """VGAE training: reconstruct node features + CAN IDs + neighborhood.

    When teacher is provided, adds dual-signal KD loss:
      kd_loss = latent_w * MSE(project(z_s), z_t) + recon_w * MSE(recon_s, recon_t)
      total = alpha * kd_loss + (1-alpha) * task_loss

    Memory optimization: When cfg.offload_teacher_to_cpu is True, the teacher
    model is moved to CPU after each forward pass to free GPU memory.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        num_ids: int,
        in_channels: int,
        teacher: Optional[nn.Module] = None,
        projection: Optional[nn.Linear] = None,
    ):
        super().__init__()
        from src.models.vgae import GraphAutoencoderNeighborhood

        self.cfg = cfg
        self.model = GraphAutoencoderNeighborhood(
            num_ids=num_ids,
            in_channels=in_channels,
            hidden_dims=list(cfg.vgae_hidden_dims),
            latent_dim=cfg.vgae_latent_dim,
            encoder_heads=cfg.vgae_heads,
            embedding_dim=cfg.vgae_embedding_dim,
            dropout=cfg.vgae_dropout,
            use_checkpointing=cfg.gradient_checkpointing,
        )
        self.teacher = teacher
        self.projection = projection
        self._teacher_on_cpu = False

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.batch)

    def _task_loss(self, batch):
        cont_out, canid_logits, nbr_logits, z, kl_loss = self(batch)
        recon = F.mse_loss(cont_out, batch.x[:, 1:])
        canid = F.cross_entropy(canid_logits, batch.x[:, 0].long())
        nbr_targets = self.model.create_neighborhood_targets(
            batch.x, batch.edge_index, batch.batch)
        nbr_loss = F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets)
        task_loss = recon + 0.1 * canid + 0.05 * nbr_loss + 0.01 * kl_loss
        return task_loss, cont_out, z

    def _step(self, batch):
        task_loss, cont_out, z = self._task_loss(batch)

        if self.teacher is not None:
            if self.cfg.offload_teacher_to_cpu and self._teacher_on_cpu:
                self.teacher.to(batch.x.device)
                self._teacher_on_cpu = False

            with torch.no_grad():
                batch_idx = batch.batch if batch.batch is not None else \
                    torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)
                t_cont, _, _, t_z, _ = self.teacher(batch.x, batch.edge_index, batch_idx)

            if self.cfg.offload_teacher_to_cpu:
                self.teacher.to('cpu')
                torch.cuda.empty_cache()
                self._teacher_on_cpu = True

            z_s = self.projection(z) if self.projection is not None else z
            min_n = min(z_s.size(0), t_z.size(0))
            latent_kd = F.mse_loss(z_s[:min_n], t_z[:min_n])

            min_r = min(cont_out.size(0), t_cont.size(0))
            recon_kd = F.mse_loss(cont_out[:min_r], t_cont[:min_r])

            kd_loss = (self.cfg.kd_vgae_latent_weight * latent_kd
                       + self.cfg.kd_vgae_recon_weight * recon_kd)
            return self.cfg.kd_alpha * kd_loss + (1 - self.cfg.kd_alpha) * task_loss

        return task_loss

    def training_step(self, batch, _idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, _idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        params = list(self.model.parameters())
        if self.projection is not None:
            params += list(self.projection.parameters())
        opt = torch.optim.Adam(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return build_optimizer_dict(opt, self.cfg)


class GATModule(pl.LightningModule):
    """GAT supervised classification (normal vs attack).

    When teacher is provided, adds soft-label KD:
      kd_loss = KL_div(student_logits/T, teacher_logits/T) * T^2
      total = alpha * kd_loss + (1-alpha) * task_loss

    Memory optimization: When cfg.offload_teacher_to_cpu is True, the teacher
    model is moved to CPU after each forward pass to free GPU memory.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        num_ids: int,
        in_channels: int,
        num_classes: int = 2,
        teacher: Optional[nn.Module] = None,
    ):
        super().__init__()
        from src.models.models import GATWithJK

        self.cfg = cfg
        self.model = GATWithJK(
            num_ids=num_ids,
            in_channels=in_channels,
            hidden_channels=cfg.gat_hidden,
            out_channels=num_classes,
            num_layers=cfg.gat_layers,
            heads=cfg.gat_heads,
            dropout=cfg.gat_dropout,
            num_fc_layers=getattr(cfg, 'gat_fc_layers', 3),
            embedding_dim=cfg.gat_embedding_dim,
            use_checkpointing=cfg.gradient_checkpointing,
        )
        self.teacher = teacher
        self._teacher_on_cpu = False

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch):
        logits = self(batch)
        task_loss = F.cross_entropy(logits, batch.y)
        acc = (logits.argmax(1) == batch.y).float().mean()

        if self.teacher is not None:
            if self.cfg.offload_teacher_to_cpu and self._teacher_on_cpu:
                self.teacher.to(batch.x.device)
                self._teacher_on_cpu = False

            with torch.no_grad():
                t_logits = self.teacher(batch)

            if self.cfg.offload_teacher_to_cpu:
                self.teacher.to('cpu')
                torch.cuda.empty_cache()
                self._teacher_on_cpu = True

            T = self.cfg.kd_temperature
            kd_loss = F.kl_div(
                F.log_softmax(logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T ** 2)
            loss = self.cfg.kd_alpha * kd_loss + (1 - self.cfg.kd_alpha) * task_loss
        else:
            loss = task_loss

        return loss, acc

    def training_step(self, batch, _idx):
        loss, acc = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("train_acc", acc, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, _idx):
        loss, acc = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("val_acc", acc, prog_bar=True, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
        )
        return build_optimizer_dict(opt, self.cfg)


class CurriculumDataModule(pl.LightningDataModule):
    """Resamples training data each epoch with increasing difficulty."""

    def __init__(self, normals, attacks, scores, val_data, cfg: PipelineConfig):
        super().__init__()
        self.normals = normals
        self.attacks = attacks
        self.scores = scores
        self.val_data = val_data
        self.cfg = cfg
        self._current_epoch = 0

    def train_dataloader(self):
        sampled = _curriculum_sample(
            self.normals, self.attacks, self.scores,
            self._current_epoch, self.cfg,
        )
        self._current_epoch += 1
        bs = effective_batch_size(self.cfg)
        return make_dataloader(sampled, self.cfg, bs, shuffle=True)

    def val_dataloader(self):
        bs = effective_batch_size(self.cfg)
        return make_dataloader(self.val_data, self.cfg, bs, shuffle=False)


def _curriculum_sample(normals, attacks, scores, epoch, cfg: PipelineConfig):
    """Sample training batch with curriculum ratio and difficulty-based selection."""
    progress = min(epoch / max(cfg.max_epochs, 1), 1.0)
    ratio = cfg.curriculum_start_ratio + progress * (
        cfg.curriculum_end_ratio - cfg.curriculum_start_ratio
    )
    percentile = cfg.difficulty_percentile + progress * (95 - cfg.difficulty_percentile)

    if scores:
        threshold = sorted(scores)[int(len(scores) * percentile / 100)]
        hard_normals = [n for n, s in zip(normals, scores) if s >= threshold]
        if not hard_normals:
            hard_normals = normals
    else:
        hard_normals = normals

    n_normals = min(int(len(attacks) * ratio), len(hard_normals))
    sampled_normals = hard_normals[:n_normals] if n_normals else hard_normals
    return sampled_normals + attacks
