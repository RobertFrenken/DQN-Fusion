"""Training stages: autoencoder, curriculum, normal."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from ..config import PipelineConfig
from ..paths import checkpoint_path, config_path
from ..memory import log_memory_state
from .utils import (
    load_data,
    load_teacher,
    make_projection,
    make_trainer,
    make_dataloader,
    compute_optimal_batch_size,
    effective_batch_size,
    load_vgae,
    load_frozen_cfg,
    cleanup,
    graph_label,
)
from .modules import VGAEModule, GATModule, CurriculumDataModule

log = logging.getLogger(__name__)


def train_autoencoder(cfg: PipelineConfig) -> Path:
    """Train VGAE on graph reconstruction. Returns checkpoint path."""
    log.info("=== AUTOENCODER: %s / %s ===", cfg.dataset, cfg.model_size)
    pl.seed_everything(cfg.seed)

    train_data, val_data, num_ids, in_ch = load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Optional KD: load teacher
    teacher, projection = None, None
    if cfg.use_kd and cfg.teacher_path:
        teacher = load_teacher(cfg.teacher_path, "vgae", cfg, num_ids, in_ch, device)
        from src.models.vgae import GraphAutoencoderNeighborhood
        _tmp_student = GraphAutoencoderNeighborhood(
            num_ids=num_ids, in_channels=in_ch,
            hidden_dims=list(cfg.vgae.hidden_dims), latent_dim=cfg.vgae.latent_dim,
            encoder_heads=cfg.vgae.heads, embedding_dim=cfg.vgae.embedding_dim,
            dropout=cfg.vgae.dropout,
        )
        projection = make_projection(_tmp_student, teacher, "vgae", device)
        del _tmp_student

    module = VGAEModule(cfg, num_ids, in_ch, teacher=teacher, projection=projection)

    if cfg.optimize_batch_size:
        bs = compute_optimal_batch_size(module.model, train_data, cfg, teacher=teacher)
    else:
        bs = effective_batch_size(cfg)

    log_memory_state("pre-dataloader")
    train_dl = make_dataloader(train_data, cfg, bs, shuffle=True)
    val_dl = make_dataloader(val_data, cfg, bs, shuffle=False)

    trainer = make_trainer(cfg, "autoencoder")
    trainer.fit(module, train_dl, val_dl)

    ckpt = checkpoint_path(cfg, "autoencoder")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), ckpt)
    cfg.save(config_path(cfg, "autoencoder"))
    log.info("Saved VGAE: %s", ckpt)
    cleanup()
    return ckpt


def train_curriculum(cfg: PipelineConfig) -> Path:
    """Train GAT with VGAE-guided curriculum learning. Returns checkpoint path."""
    log.info("=== CURRICULUM: %s / %s ===", cfg.dataset, cfg.model_size)
    pl.seed_everything(cfg.seed)

    train_data, val_data, num_ids, in_ch = load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load VGAE for difficulty scoring
    vgae = load_vgae(cfg, num_ids, in_ch, device, stage="autoencoder")

    # Split and score
    normals = [g for g in train_data if graph_label(g) == 0]
    attacks = [g for g in train_data if graph_label(g) == 1]
    scores = _score_difficulty(vgae, normals, device)
    del vgae
    cleanup()

    # Optional KD: load teacher GAT
    teacher = None
    if cfg.use_kd and cfg.teacher_path:
        teacher = load_teacher(cfg.teacher_path, "gat", cfg, num_ids, in_ch, device)

    module = GATModule(cfg, num_ids, in_ch, teacher=teacher)
    trainer = make_trainer(cfg, "curriculum")

    dm = CurriculumDataModule(normals, attacks, scores, val_data, cfg)
    trainer.fit(module, datamodule=dm)

    ckpt = checkpoint_path(cfg, "curriculum")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), ckpt)
    cfg.save(config_path(cfg, "curriculum"))
    log.info("Saved GAT: %s", ckpt)
    cleanup()
    return ckpt


def train_normal(cfg: PipelineConfig) -> Path:
    """Train GAT with standard cross-entropy (no curriculum). Returns checkpoint path."""
    log.info("=== NORMAL: %s / %s ===", cfg.dataset, cfg.model_size)
    pl.seed_everything(cfg.seed)

    train_data, val_data, num_ids, in_ch = load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Optional KD: load teacher GAT
    teacher = None
    if cfg.use_kd and cfg.teacher_path:
        teacher = load_teacher(cfg.teacher_path, "gat", cfg, num_ids, in_ch, device)

    module = GATModule(cfg, num_ids, in_ch, teacher=teacher)

    if cfg.optimize_batch_size:
        bs = compute_optimal_batch_size(module.model, train_data, cfg, teacher=teacher)
    else:
        bs = effective_batch_size(cfg)

    log_memory_state("pre-dataloader")
    train_dl = make_dataloader(train_data, cfg, bs, shuffle=True)
    val_dl = make_dataloader(val_data, cfg, bs, shuffle=False)

    trainer = make_trainer(cfg, "normal")
    trainer.fit(module, train_dl, val_dl)

    ckpt = checkpoint_path(cfg, "normal")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), ckpt)
    cfg.save(config_path(cfg, "normal"))
    log.info("Saved GAT (normal): %s", ckpt)
    cleanup()
    return ckpt


def _score_difficulty(vgae_model, graphs, device, chunk_size: int = 500) -> list[float]:
    """Score each graph's reconstruction difficulty using trained VGAE.

    Memory optimization: Processes graphs in chunks and clears GPU cache between
    chunks to prevent memory accumulation on large datasets.
    """
    scores = []
    vgae_model.eval()
    total_chunks = (len(graphs) + chunk_size - 1) // chunk_size

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(graphs))
        chunk_graphs = graphs[start:end]

        with torch.no_grad():
            for g in chunk_graphs:
                g = g.clone().to(device)
                batch_idx = (g.batch if hasattr(g, "batch") and g.batch is not None
                             else torch.zeros(g.x.size(0), dtype=torch.long, device=device))
                cont, canid_logits, _, _, _ = vgae_model(g.x, g.edge_index, batch_idx)
                recon = F.mse_loss(cont, g.x[:, 1:]).item()
                canid = F.cross_entropy(canid_logits, g.x[:, 0].long()).item()
                scores.append(recon + 0.1 * canid)
                del g

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (chunk_idx + 1) % 10 == 0:
            log.info("Difficulty scoring: %d/%d chunks complete", chunk_idx + 1, total_chunks)

    return scores
