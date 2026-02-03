"""Training stages. Each function: PipelineConfig -> checkpoint path.

Imports model architectures from src/models/.
Imports data loading from src/training/.
Everything else (Lightning wrappers, trainer setup, KD) is self-contained here.
"""
from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from .config import PipelineConfig
from .paths import stage_dir, checkpoint_path, config_path, data_dir, cache_dir
from .tracking import log_memory_metrics, get_memory_summary

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory monitoring callback
# ---------------------------------------------------------------------------

class MemoryMonitorCallback(pl.Callback):
    """Log memory usage to MLflow at epoch boundaries."""

    def __init__(self, log_every_n_epochs: int = 5):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_start(self, trainer, pl_module):
        log.info("Memory at train start: %s", get_memory_summary())
        log_memory_metrics(step=0)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs == 0:
            log_memory_metrics(step=epoch)

    def on_train_end(self, trainer, pl_module):
        log.info("Memory at train end: %s", get_memory_summary())
        log_memory_metrics(step=trainer.current_epoch)

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Data loading (bridges to existing src/ data pipeline)
# ---------------------------------------------------------------------------

def _load_data(cfg: PipelineConfig):
    """Load graph dataset. Returns (train_graphs, val_graphs, num_ids, in_channels)."""
    from src.training.datamodules import load_dataset

    train_data, val_data, num_ids = load_dataset(
        cfg.dataset,
        dataset_path=data_dir(cfg),
        cache_dir_path=cache_dir(cfg),
    )
    in_channels = train_data[0].x.shape[1] if train_data else 11
    return train_data, val_data, num_ids, in_channels


def _effective_batch_size(cfg: PipelineConfig) -> int:
    """Apply safety factor to batch size."""
    return max(8, int(cfg.batch_size * cfg.safety_factor))


def _auto_tune_batch_size(
    module: pl.LightningModule,
    train_data, val_data,
    cfg: PipelineConfig,
) -> int:
    """Use Lightning Tuner to find optimal batch size, then apply safety factor."""
    from src.training.datamodules import CANGraphDataModule
    from lightning.pytorch.tuner import Tuner
    import tempfile, shutil

    module.batch_size = cfg.batch_size
    dm = CANGraphDataModule(train_data, val_data, cfg.batch_size, num_workers=cfg.num_workers)
    tmp = tempfile.mkdtemp(prefix="tuner_", dir=".")

    trainer = pl.Trainer(
        accelerator="gpu", devices=1,
        precision=cfg.precision,
        max_steps=200, max_epochs=None,
        enable_checkpointing=False, logger=False,
        default_root_dir=tmp,
    )

    try:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(module, datamodule=dm,
                               mode=cfg.batch_size_mode,
                               steps_per_trial=3,
                               init_val=cfg.batch_size)
        tuned = module.batch_size
        safe = max(8, int(tuned * cfg.safety_factor))
        log.info("Batch size: %d -> tuned %d -> safe %d (sf=%.2f)",
                 cfg.batch_size, tuned, safe, cfg.safety_factor)
        return safe
    except Exception as e:
        log.warning("Batch size tuning failed: %s. Falling back.", e)
        return _effective_batch_size(cfg)
    finally:
        import glob as _glob
        for ckpt in _glob.glob(".scale_batch_size_*.ckpt"):
            Path(ckpt).unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Teacher loading & KD helpers
# ---------------------------------------------------------------------------

def _extract_state_dict(checkpoint) -> dict:
    """Handle different checkpoint formats, return clean state dict."""
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
            return {k.replace("model.", ""): v for k, v in sd.items()
                    if k.startswith("model.")} or sd
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        return checkpoint
    return checkpoint


def _load_teacher(teacher_path: str, model_type: str,
                  cfg: PipelineConfig, num_ids: int, in_channels: int,
                  device: torch.device) -> nn.Module:
    """Load teacher model using its frozen config.

    Loads the config.json saved alongside the teacher checkpoint so dimensions
    always match, regardless of what preset the current (student) run uses.
    Falls back to state-dict inference if no frozen config is found.
    """
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
    sd = _extract_state_dict(checkpoint)

    # Try to load teacher's own frozen config
    teacher_cfg_path = Path(teacher_path).parent / "config.json"
    if teacher_cfg_path.exists():
        try:
            tcfg = PipelineConfig.load(teacher_cfg_path)
        except Exception as e:
            log.warning("Could not load teacher config %s: %s", teacher_cfg_path, e)
            tcfg = cfg
    else:
        log.warning("No teacher config at %s, falling back to current cfg", teacher_cfg_path)
        tcfg = cfg

    if model_type == "vgae":
        from src.models.vgae import GraphAutoencoderNeighborhood

        id_emb = sd.get("encoder.id_embedding.weight",
                        sd.get("id_embedding.weight"))
        t_num_ids = id_emb.shape[0] if id_emb is not None else num_ids

        teacher = GraphAutoencoderNeighborhood(
            num_ids=t_num_ids, in_channels=in_channels,
            hidden_dims=list(tcfg.vgae_hidden_dims), latent_dim=tcfg.vgae_latent_dim,
            encoder_heads=tcfg.vgae_heads, embedding_dim=tcfg.vgae_embedding_dim,
            dropout=tcfg.vgae_dropout,
        )
        teacher.load_state_dict(sd, strict=False)
        log.info("Loaded VGAE teacher: latent_dim=%d, num_ids=%d",
                 tcfg.vgae_latent_dim, t_num_ids)

    elif model_type == "gat":
        from src.models.models import GATWithJK

        id_emb = sd.get("id_embedding.weight")
        t_num_ids = id_emb.shape[0] if id_emb is not None else num_ids

        teacher = GATWithJK(
            num_ids=t_num_ids, in_channels=in_channels,
            hidden_channels=tcfg.gat_hidden, out_channels=2,
            num_layers=tcfg.gat_layers, heads=tcfg.gat_heads,
            dropout=tcfg.gat_dropout, embedding_dim=tcfg.gat_embedding_dim,
        )
        teacher.load_state_dict(sd, strict=False)
        log.info("Loaded GAT teacher: hidden=%d, layers=%d, num_ids=%d",
                 tcfg.gat_hidden, tcfg.gat_layers, t_num_ids)

    elif model_type == "dqn":
        from src.models.dqn import QNetwork

        state_dim = 15
        action_dim = cfg.alpha_steps
        teacher = QNetwork(state_dim, action_dim)

        if "q_network" in sd:
            teacher.load_state_dict(sd["q_network"])
        elif "q_network_state_dict" in sd:
            teacher.load_state_dict(sd["q_network_state_dict"])
        else:
            teacher.load_state_dict(sd, strict=False)
        log.info("Loaded DQN teacher (QNetwork): state_dim=%d, action_dim=%d",
                 state_dim, action_dim)

    else:
        raise ValueError(f"Cannot load teacher for model_type={model_type}")

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _make_projection(student_model: nn.Module, teacher: nn.Module,
                     model_type: str, device: torch.device) -> Optional[nn.Linear]:
    """Create projection layer if teacher/student latent dims differ."""
    if model_type == "vgae":
        s_dim = getattr(student_model, "latent_dim",
                        getattr(student_model, "_latent_dim", 16))
        t_dim = getattr(teacher, "latent_dim",
                        getattr(teacher, "_latent_dim", 96))
    elif model_type == "gat":
        s_dim = getattr(student_model, "hidden_channels",
                        getattr(student_model, "out_channels", 2))
        t_dim = getattr(teacher, "hidden_channels",
                        getattr(teacher, "out_channels", 2))
    else:
        return None

    if s_dim != t_dim:
        proj = nn.Linear(s_dim, t_dim).to(device)
        log.info("Projection layer: %d -> %d", s_dim, t_dim)
        return proj
    return None


# ---------------------------------------------------------------------------
# Lightning modules (with optional KD support)
# ---------------------------------------------------------------------------

class VGAEModule(pl.LightningModule):
    """VGAE training: reconstruct node features + CAN IDs + neighborhood.

    When teacher is provided, adds dual-signal KD loss:
      kd_loss = latent_w * MSE(project(z_s), z_t) + recon_w * MSE(recon_s, recon_t)
      total = alpha * kd_loss + (1-alpha) * task_loss
    """

    def __init__(self, cfg: PipelineConfig, num_ids: int, in_channels: int,
                 teacher: Optional[nn.Module] = None,
                 projection: Optional[nn.Linear] = None):
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

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.batch)

    def _task_loss(self, batch):
        cont_out, canid_logits, nbr_logits, z, kl_loss = self(batch)
        recon = F.mse_loss(cont_out, batch.x[:, 1:])
        canid = F.cross_entropy(canid_logits, batch.x[:, 0].long())
        # Neighborhood prediction loss
        nbr_targets = self.model.create_neighborhood_targets(
            batch.x, batch.edge_index, batch.batch)
        nbr_loss = F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets)
        task_loss = recon + 0.1 * canid + 0.05 * nbr_loss + 0.01 * kl_loss
        return task_loss, cont_out, z

    def _step(self, batch):
        task_loss, cont_out, z = self._task_loss(batch)

        if self.teacher is not None:
            with torch.no_grad():
                batch_idx = batch.batch if batch.batch is not None else \
                    torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)
                t_cont, _, _, t_z, _ = self.teacher(batch.x, batch.edge_index, batch_idx)

            # Latent KD (project student z up to teacher dim)
            z_s = self.projection(z) if self.projection is not None else z
            min_n = min(z_s.size(0), t_z.size(0))
            latent_kd = F.mse_loss(z_s[:min_n], t_z[:min_n])

            # Reconstruction KD
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
        return _build_optimizer_dict(opt, self.cfg)


class GATModule(pl.LightningModule):
    """GAT supervised classification (normal vs attack).

    When teacher is provided, adds soft-label KD:
      kd_loss = KL_div(student_logits/T, teacher_logits/T) * T^2
      total = alpha * kd_loss + (1-alpha) * task_loss
    """

    def __init__(self, cfg: PipelineConfig, num_ids: int, in_channels: int,
                 num_classes: int = 2,
                 teacher: Optional[nn.Module] = None):
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
            embedding_dim=cfg.gat_embedding_dim,
            use_checkpointing=cfg.gradient_checkpointing,
        )
        self.teacher = teacher

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch):
        logits = self(batch)
        task_loss = F.cross_entropy(logits, batch.y)
        acc = (logits.argmax(1) == batch.y).float().mean()

        if self.teacher is not None:
            with torch.no_grad():
                t_logits = self.teacher(batch)
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
        return _build_optimizer_dict(opt, self.cfg)


# ---------------------------------------------------------------------------
# LR scheduler helper
# ---------------------------------------------------------------------------

def _build_optimizer_dict(optimizer, cfg: PipelineConfig):
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    if not cfg.use_scheduler:
        return optimizer

    t_max = cfg.scheduler_t_max if cfg.scheduler_t_max > 0 else cfg.max_epochs

    if cfg.scheduler_type == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif cfg.scheduler_type == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma,
        )
    elif cfg.scheduler_type == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=cfg.monitor_mode, patience=cfg.patience // 2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "monitor": cfg.monitor_metric},
        }
    else:
        log.warning("Unknown scheduler_type=%s, skipping", cfg.scheduler_type)
        return optimizer

    return {"optimizer": optimizer, "lr_scheduler": sched}


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def _make_trainer(cfg: PipelineConfig, stage: str) -> pl.Trainer:
    out = stage_dir(cfg, stage)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    # Route Lightning self.log() calls to the active MLflow run.
    # cli.py has already started the run; autolog patches Trainer.fit()
    # to forward epoch-level metrics (train_loss, val_loss, etc.).
    mlflow.pytorch.autolog(log_models=False)

    return pl.Trainer(
        default_root_dir=str(out),
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=cfg.precision,
        gradient_clip_val=cfg.gradient_clip,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[
            ModelCheckpoint(
                dirpath=str(out), filename="best_model",
                monitor=cfg.monitor_metric, mode=cfg.monitor_mode,
                save_top_k=cfg.save_top_k,
            ),
            EarlyStopping(
                monitor=cfg.monitor_metric, patience=cfg.patience,
                mode=cfg.monitor_mode,
            ),
            MemoryMonitorCallback(log_every_n_epochs=cfg.test_every_n_epochs),
        ],
        log_every_n_steps=cfg.log_every_n_steps,
        enable_progress_bar=True,
        deterministic=cfg.deterministic,
    )


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Curriculum DataModule (per-epoch resampling)
# ---------------------------------------------------------------------------

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
        bs = _effective_batch_size(self.cfg)
        nw = self.cfg.num_workers
        return DataLoader(sampled, batch_size=bs, shuffle=True,
                          num_workers=nw, pin_memory=True,
                          persistent_workers=nw > 0,
                          multiprocessing_context=self.cfg.mp_start_method if nw > 0 else None)

    def val_dataloader(self):
        bs = _effective_batch_size(self.cfg)
        nw = self.cfg.num_workers
        return DataLoader(self.val_data, batch_size=bs,
                          num_workers=nw, pin_memory=True,
                          persistent_workers=nw > 0,
                          multiprocessing_context=self.cfg.mp_start_method if nw > 0 else None)


# ---------------------------------------------------------------------------
# Stage 1: Autoencoder (VGAE)
# ---------------------------------------------------------------------------

def train_autoencoder(cfg: PipelineConfig) -> Path:
    """Train VGAE on graph reconstruction. Returns checkpoint path."""
    log.info("=== AUTOENCODER: %s / %s ===", cfg.dataset, cfg.model_size)
    pl.seed_everything(cfg.seed)

    train_data, val_data, num_ids, in_ch = _load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Optional KD: load teacher
    teacher, projection = None, None
    if cfg.use_kd and cfg.teacher_path:
        teacher = _load_teacher(cfg.teacher_path, "vgae", cfg, num_ids, in_ch, device)
        # Build student to get projection
        from src.models.vgae import GraphAutoencoderNeighborhood
        _tmp_student = GraphAutoencoderNeighborhood(
            num_ids=num_ids, in_channels=in_ch,
            hidden_dims=list(cfg.vgae_hidden_dims), latent_dim=cfg.vgae_latent_dim,
            encoder_heads=cfg.vgae_heads, embedding_dim=cfg.vgae_embedding_dim,
            dropout=cfg.vgae_dropout,
        )
        projection = _make_projection(_tmp_student, teacher, "vgae", device)
        del _tmp_student

    module = VGAEModule(cfg, num_ids, in_ch, teacher=teacher, projection=projection)

    if cfg.optimize_batch_size:
        bs = _auto_tune_batch_size(module, train_data, val_data, cfg)
    else:
        bs = _effective_batch_size(cfg)
    nw = cfg.num_workers
    train_dl = DataLoader(train_data, batch_size=bs,
                          shuffle=True, num_workers=nw,
                          pin_memory=True, persistent_workers=nw > 0,
                          multiprocessing_context=cfg.mp_start_method if nw > 0 else None)
    val_dl = DataLoader(val_data, batch_size=bs,
                        num_workers=nw, pin_memory=True,
                        persistent_workers=nw > 0,
                        multiprocessing_context=cfg.mp_start_method if nw > 0 else None)

    trainer = _make_trainer(cfg, "autoencoder")
    trainer.fit(module, train_dl, val_dl)

    # Save bare state dict (no Lightning wrapper needed for downstream loading)
    ckpt = checkpoint_path(cfg, "autoencoder")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), ckpt)
    cfg.save(config_path(cfg, "autoencoder"))
    log.info("Saved VGAE: %s", ckpt)
    _cleanup()
    return ckpt


# ---------------------------------------------------------------------------
# Stage 2: Curriculum (GAT with VGAE-guided hard mining)
# ---------------------------------------------------------------------------

def _score_difficulty(vgae_model, graphs, device) -> list[float]:
    """Score each graph's reconstruction difficulty using trained VGAE."""
    scores = []
    vgae_model.eval()
    with torch.no_grad():
        for g in graphs:
            g = g.clone().to(device)  # clone to avoid mutating original (PyG .to() is in-place)
            batch_idx = (g.batch if hasattr(g, "batch") and g.batch is not None
                         else torch.zeros(g.x.size(0), dtype=torch.long, device=device))
            cont, canid_logits, _, _, _ = vgae_model(g.x, g.edge_index, batch_idx)
            recon = F.mse_loss(cont, g.x[:, 1:]).item()
            canid = F.cross_entropy(canid_logits, g.x[:, 0].long()).item()
            scores.append(recon + 0.1 * canid)
    return scores


def _curriculum_sample(normals, attacks, scores, epoch, cfg):
    """Sample training batch with curriculum ratio and difficulty-based selection."""
    progress = min(epoch / max(cfg.max_epochs, 1), 1.0)
    ratio = cfg.curriculum_start_ratio + progress * (
        cfg.curriculum_end_ratio - cfg.curriculum_start_ratio
    )
    # Select hard normals (high reconstruction error)
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


def train_curriculum(cfg: PipelineConfig) -> Path:
    """Train GAT with VGAE-guided curriculum learning. Returns checkpoint path."""
    log.info("=== CURRICULUM: %s / %s ===", cfg.dataset, cfg.model_size)
    pl.seed_everything(cfg.seed)

    train_data, val_data, num_ids, in_ch = _load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load VGAE for difficulty scoring (use frozen config from autoencoder stage)
    vgae_ckpt = checkpoint_path(cfg, "autoencoder")
    vgae_cfg = _load_frozen_cfg(cfg, "autoencoder")
    from src.models.vgae import GraphAutoencoderNeighborhood
    vgae = GraphAutoencoderNeighborhood(
        num_ids=num_ids, in_channels=in_ch,
        hidden_dims=list(vgae_cfg.vgae_hidden_dims), latent_dim=vgae_cfg.vgae_latent_dim,
        encoder_heads=vgae_cfg.vgae_heads, embedding_dim=vgae_cfg.vgae_embedding_dim,
        dropout=vgae_cfg.vgae_dropout,
    )
    vgae.load_state_dict(torch.load(vgae_ckpt, map_location="cpu", weights_only=True))
    vgae.to(device)

    # Split and score
    normals = [g for g in train_data if g.y.item() == 0]
    attacks = [g for g in train_data if g.y.item() == 1]
    scores = _score_difficulty(vgae, normals, device)
    del vgae
    _cleanup()

    # Optional KD: load teacher GAT
    teacher = None
    if cfg.use_kd and cfg.teacher_path:
        teacher = _load_teacher(cfg.teacher_path, "gat", cfg, num_ids, in_ch, device)

    module = GATModule(cfg, num_ids, in_ch, teacher=teacher)
    trainer = _make_trainer(cfg, "curriculum")

    # Use CurriculumDataModule for per-epoch resampling
    dm = CurriculumDataModule(normals, attacks, scores, val_data, cfg)
    trainer.fit(module, datamodule=dm)

    ckpt = checkpoint_path(cfg, "curriculum")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), ckpt)
    cfg.save(config_path(cfg, "curriculum"))
    log.info("Saved GAT: %s", ckpt)
    _cleanup()
    return ckpt


# ---------------------------------------------------------------------------
# Stage 2b: Normal (GAT without curriculum — plain cross-entropy)
# ---------------------------------------------------------------------------

def train_normal(cfg: PipelineConfig) -> Path:
    """Train GAT with standard cross-entropy (no curriculum). Returns checkpoint path."""
    log.info("=== NORMAL: %s / %s ===", cfg.dataset, cfg.model_size)
    pl.seed_everything(cfg.seed)

    train_data, val_data, num_ids, in_ch = _load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Optional KD: load teacher GAT
    teacher = None
    if cfg.use_kd and cfg.teacher_path:
        teacher = _load_teacher(cfg.teacher_path, "gat", cfg, num_ids, in_ch, device)

    module = GATModule(cfg, num_ids, in_ch, teacher=teacher)

    if cfg.optimize_batch_size:
        bs = _auto_tune_batch_size(module, train_data, val_data, cfg)
    else:
        bs = _effective_batch_size(cfg)
    nw = cfg.num_workers
    train_dl = DataLoader(train_data, batch_size=bs,
                          shuffle=True, num_workers=nw,
                          pin_memory=True, persistent_workers=nw > 0,
                          multiprocessing_context=cfg.mp_start_method if nw > 0 else None)
    val_dl = DataLoader(val_data, batch_size=bs,
                        num_workers=nw, pin_memory=True,
                        persistent_workers=nw > 0,
                        multiprocessing_context=cfg.mp_start_method if nw > 0 else None)

    trainer = _make_trainer(cfg, "normal")
    trainer.fit(module, train_dl, val_dl)

    ckpt = checkpoint_path(cfg, "normal")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), ckpt)
    cfg.save(config_path(cfg, "normal"))
    log.info("Saved GAT (normal): %s", ckpt)
    _cleanup()
    return ckpt


# ---------------------------------------------------------------------------
# Stage 3: Fusion (DQN combining VGAE + GAT)
# ---------------------------------------------------------------------------

def _cache_predictions(vgae, gat, data, device, max_samples=150_000):
    """Run VGAE + GAT inference, produce 15-D state vectors for DQN.

    State layout (matches EnhancedDQNFusionAgent.normalize_state):
        [0:3]  VGAE errors  (node recon, neighbor, canid)
        [3:7]  VGAE latent stats  (mean, std, max, min)
        [7]    VGAE confidence
        [8:10] GAT logits  (class 0 prob, class 1 prob)
        [10:14] GAT embedding stats  (mean, std, max, min)
        [14]   GAT confidence
    """
    states, labels = [], []
    vgae.eval()
    gat.eval()
    n_samples = min(len(data), max_samples)
    with torch.no_grad():
        for i in range(n_samples):
            g = data[i]
            g = g.clone().to(device)  # clone to avoid mutating original (PyG .to() is in-place)
            batch_idx = (g.batch if hasattr(g, "batch") and g.batch is not None
                         else torch.zeros(g.x.size(0), dtype=torch.long, device=device))

            # VGAE features
            cont, canid_logits, nbr_logits, z, _ = vgae(g.x, g.edge_index, batch_idx)
            recon_err = F.mse_loss(cont, g.x[:, 1:], reduction="none").mean().item()
            canid_err = F.cross_entropy(canid_logits, g.x[:, 0].long()).item()
            # Neighborhood prediction error
            nbr_targets = vgae.create_neighborhood_targets(g.x, g.edge_index, batch_idx)
            nbr_err = F.binary_cross_entropy_with_logits(
                nbr_logits, nbr_targets, reduction="mean").item()
            z_mean = z.mean().item()
            z_std = z.std().item()
            z_max = z.max().item()
            z_min = z.min().item()
            vgae_conf = 1.0 / (1.0 + recon_err)

            # GAT features: get intermediate representations for embedding stats
            xs = gat(g, return_intermediate=True)
            jk_out = gat.jk(xs)
            pooled = global_mean_pool(jk_out, batch_idx)
            # Embedding stats from pooled JK representation
            emb_mean = pooled.mean().item()
            emb_std = pooled.std().item() if pooled.numel() > 1 else 0.0
            emb_max = pooled.max().item()
            emb_min = pooled.min().item()
            # Compute logits through FC layers
            x = pooled
            for layer in gat.fc_layers:
                x = layer(x)
            logits = x
            probs = F.softmax(logits, dim=1)
            p0 = probs[0, 0].item()
            p1 = probs[0, 1].item()
            entropy = -(probs * (probs + 1e-8).log()).sum().item()
            gat_conf = max(0.0, min(1.0, 1.0 - entropy / 0.693))

            state = torch.tensor([
                recon_err, nbr_err, canid_err,         # VGAE errors [3]
                z_mean, z_std, z_max, z_min,            # latent stats [4]
                vgae_conf,                               # VGAE confidence [1]
                p0, p1,                                  # GAT logits [2]
                emb_mean, emb_std, emb_max, emb_min,    # GAT embedding stats [4]
                gat_conf,                                # GAT confidence [1]
            ])
            states.append(state)
            labels.append(g.y[0] if g.y.dim() > 0 else g.y)

    return {"states": torch.stack(states), "labels": torch.tensor(labels)}


def train_fusion(cfg: PipelineConfig) -> Path:
    """Train DQN fusion agent on cached VGAE+GAT predictions. Returns checkpoint path."""
    log.info("=== FUSION: %s / %s ===", cfg.dataset, cfg.model_size)
    pl.seed_everything(cfg.seed)

    train_data, val_data, num_ids, in_ch = _load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load frozen VGAE + GAT (use frozen configs from their training stages)
    from src.models.vgae import GraphAutoencoderNeighborhood
    from src.models.models import GATWithJK
    from src.models.dqn import EnhancedDQNFusionAgent

    # Determine prerequisite stage names based on use_kd flag
    vgae_stage = "autoencoder_kd" if cfg.use_kd else "autoencoder"
    gat_stage = "curriculum_kd" if cfg.use_kd else "curriculum"

    vgae_cfg = _load_frozen_cfg(cfg, vgae_stage)
    vgae = GraphAutoencoderNeighborhood(
        num_ids=num_ids, in_channels=in_ch,
        hidden_dims=list(vgae_cfg.vgae_hidden_dims), latent_dim=vgae_cfg.vgae_latent_dim,
        encoder_heads=vgae_cfg.vgae_heads, embedding_dim=vgae_cfg.vgae_embedding_dim,
        dropout=vgae_cfg.vgae_dropout,
    )
    vgae.load_state_dict(torch.load(
        checkpoint_path(cfg, vgae_stage), map_location="cpu", weights_only=True,
    ))
    vgae.to(device)

    gat_cfg = _load_frozen_cfg(cfg, gat_stage)
    gat = GATWithJK(
        num_ids=num_ids, in_channels=in_ch,
        hidden_channels=gat_cfg.gat_hidden, out_channels=2,
        num_layers=gat_cfg.gat_layers, heads=gat_cfg.gat_heads,
        dropout=gat_cfg.gat_dropout, embedding_dim=gat_cfg.gat_embedding_dim,
    )
    gat.load_state_dict(torch.load(
        checkpoint_path(cfg, gat_stage), map_location="cpu", weights_only=True,
    ))
    gat.to(device)

    # Cache predictions
    log.info("Caching VGAE + GAT predictions ...")
    train_cache = _cache_predictions(vgae, gat, train_data, device, cfg.fusion_max_samples)
    val_cache = _cache_predictions(vgae, gat, val_data, device, cfg.max_val_samples)
    del vgae, gat
    _cleanup()

    # DQN agent
    agent = EnhancedDQNFusionAgent(
        lr=cfg.fusion_lr, gamma=cfg.dqn_gamma,
        epsilon=cfg.dqn_epsilon, epsilon_decay=cfg.dqn_epsilon_decay,
        min_epsilon=cfg.dqn_min_epsilon,
        buffer_size=cfg.dqn_buffer_size, batch_size=cfg.dqn_batch_size,
        target_update_freq=cfg.dqn_target_update, device=str(device),
    )

    out = stage_dir(cfg, "fusion")
    out.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for ep in range(cfg.fusion_episodes):
        # Sample a batch of cached states
        idx = torch.randperm(len(train_cache["states"]))[:cfg.episode_sample_size]
        batch_states = train_cache["states"][idx]
        batch_labels = train_cache["labels"][idx]

        total_reward = 0.0
        for i in range(len(batch_states)):
            state_np = batch_states[i].numpy()
            alpha, action_idx, proc_state = agent.select_action(state_np, training=True)
            pred = 1 if alpha > 0.5 else 0
            reward = 3.0 if pred == batch_labels[i].item() else -3.0
            agent.store_experience(proc_state, action_idx, reward, proc_state, False)
            total_reward += reward

        # DQN training steps
        if len(agent.replay_buffer) >= cfg.dqn_batch_size:
            for _ in range(cfg.gpu_training_steps):
                agent.train_step()

        # Periodic validation
        if (ep + 1) % 50 == 0:
            val_pairs = [
                (val_cache["states"][i].numpy(), val_cache["labels"][i].item())
                for i in range(min(5000, len(val_cache["states"])))
            ]
            metrics = agent.validate_agent(val_pairs, num_samples=len(val_pairs))
            acc = metrics.get("accuracy", 0)
            log.info("Episode %d/%d  reward=%.1f  val_acc=%.4f",
                     ep + 1, cfg.fusion_episodes, total_reward, acc)

            # Manual MLflow logging for DQN (no Lightning Trainer)
            mlflow.log_metrics({
                "total_reward": total_reward,
                "val_accuracy": acc,
                "epsilon": agent.epsilon,
                "best_accuracy": best_acc,
            }, step=ep + 1)

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "q_network": agent.q_network.state_dict(),
                    "target_network": agent.target_network.state_dict(),
                    "epsilon": agent.epsilon,
                }, checkpoint_path(cfg, "fusion"))

    # Ensure we always save something
    ckpt = checkpoint_path(cfg, "fusion")
    if not ckpt.exists():
        torch.save({
            "q_network": agent.q_network.state_dict(),
            "target_network": agent.target_network.state_dict(),
            "epsilon": agent.epsilon,
        }, ckpt)

    cfg.save(config_path(cfg, "fusion"))
    log.info("Saved DQN: %s (best_acc=%.4f)", ckpt, best_acc)
    _cleanup()
    return ckpt


# ---------------------------------------------------------------------------
# Stage 4: Evaluation (supports VGAE, GAT, and DQN fusion)
# ---------------------------------------------------------------------------

def _compute_metrics(labels, preds, scores=None) -> dict:
    """Compute comprehensive classification metrics.

    Returns a dict with two sections:
      - "core": accuracy, precision, recall, f1, specificity, balanced_accuracy, mcc, fpr, fnr, auc
      - "additional": kappa, tpr, tnr, detection_rate, miss_rate, pr_auc, detection_at_fpr
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
        confusion_matrix, cohen_kappa_score, precision_recall_curve,
        auc as sk_auc, roc_curve,
    )

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    core = {
        "accuracy":          float(accuracy_score(labels, preds)),
        "precision":         float(precision_score(labels, preds, zero_division=0)),
        "recall":            float(recall_score(labels, preds, zero_division=0)),
        "f1":                float(f1_score(labels, preds, zero_division=0)),
        "specificity":       specificity,
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "mcc":               float(matthews_corrcoef(labels, preds)),
        "fpr":               fpr,
        "fnr":               fnr,
        "n_samples":         int(len(labels)),
    }

    additional = {
        "kappa":          float(cohen_kappa_score(labels, preds)),
        "tpr":            tpr,
        "tnr":            specificity,
        "detection_rate": tpr,
        "miss_rate":      fnr,
    }

    if scores is not None and len(set(labels)) > 1:
        core["auc"] = float(roc_auc_score(labels, scores))

        # PR-AUC
        try:
            prec_vals, rec_vals, _ = precision_recall_curve(labels, scores)
            additional["pr_auc"] = float(sk_auc(rec_vals, prec_vals))
        except ValueError:
            additional["pr_auc"] = 0.0

        # Detection rate at key FPR thresholds
        try:
            fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
            det_at_fpr = {}
            for fpr_target in [0.05, 0.01, 0.001]:
                idx = np.argmin(np.abs(fpr_curve - fpr_target))
                det_at_fpr[str(fpr_target)] = float(tpr_curve[idx])
            additional["detection_at_fpr"] = det_at_fpr
        except ValueError:
            additional["detection_at_fpr"] = {}

    return {"core": core, "additional": additional}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _graph_label(g) -> int:
    """Extract scalar graph-level label consistently."""
    return g.y.item() if g.y.dim() == 0 else int(g.y[0].item())


def _load_frozen_cfg(cfg: PipelineConfig, stage: str) -> PipelineConfig:
    """Load the frozen config.json saved during training for *stage*.

    Falls back to *cfg* if the frozen config doesn't exist (e.g. first run).
    """
    p = config_path(cfg, stage)
    if p.exists():
        try:
            return PipelineConfig.load(p)
        except Exception as e:
            log.warning("Could not load frozen config %s: %s", p, e)
    return cfg


def _load_test_data(cfg: PipelineConfig) -> dict:
    """Load held-out test graphs per scenario.

    Returns ``{scenario_name: [Data, ...]}``.  Uses the training id_mapping
    from cache so CAN-ID indices stay consistent.
    """
    import pickle
    from src.preprocessing.preprocessing import graph_creation

    mapping_file = cache_dir(cfg) / "id_mapping.pkl"
    if not mapping_file.exists():
        log.warning("No id_mapping at %s — skipping test data", mapping_file)
        return {}

    with open(mapping_file, "rb") as f:
        id_mapping = pickle.load(f)

    ds_path = data_dir(cfg)
    if not ds_path.exists():
        log.warning("Dataset path %s not found — skipping test data", ds_path)
        return {}

    scenarios: dict[str, list] = {}
    for folder in sorted(ds_path.iterdir()):
        if folder.is_dir() and folder.name.startswith("test_"):
            name = folder.name
            log.info("Loading test scenario: %s", name)
            graphs = graph_creation(
                str(ds_path), folder_type=name,
                id_mapping=id_mapping, return_id_mapping=False,
            )
            if graphs:
                scenarios[name] = graphs
                log.info("  %s: %d graphs", name, len(graphs))

    return scenarios


def _run_gat_inference(gat, data, device):
    """Run GAT inference on a list of graphs.

    Returns (preds, labels, scores) as numpy arrays — all 1-D, one entry
    per graph.
    """
    preds, labels, scores = [], [], []
    with torch.no_grad():
        for g in data:
            g = g.clone().to(device)  # clone to avoid mutating original (PyG .to() is in-place)
            logits = gat(g)
            probs = F.softmax(logits, dim=1)
            preds.append(logits.argmax(1)[0].item())
            labels.append(_graph_label(g))
            scores.append(probs[0, 1].item())
    return np.array(preds), np.array(labels), np.array(scores)


def _run_vgae_inference(vgae, data, device):
    """Run VGAE reconstruction-error inference.

    Returns (errors, labels) as numpy arrays — one scalar error per graph.
    """
    errors, labels = [], []
    with torch.no_grad():
        for g in data:
            g = g.clone().to(device)  # clone to avoid mutating original (PyG .to() is in-place)
            batch_idx = (g.batch if hasattr(g, "batch") and g.batch is not None
                         else torch.zeros(g.x.size(0), dtype=torch.long, device=device))
            cont, canid_logits, _, _, _ = vgae(g.x, g.edge_index, batch_idx)
            err = F.mse_loss(cont, g.x[:, 1:]).item()
            errors.append(err)
            labels.append(_graph_label(g))
    return np.array(errors), np.array(labels)


def _run_fusion_inference(agent, cache):
    """Run DQN fusion inference on cached state vectors.

    Returns (preds, labels, scores) as numpy arrays.
    """
    preds, labels, scores = [], [], []
    for i in range(len(cache["states"])):
        state_np = cache["states"][i].numpy()
        alpha, _, _ = agent.select_action(state_np, training=False)
        preds.append(1 if alpha > 0.5 else 0)
        labels.append(cache["labels"][i].item())
        scores.append(float(alpha))
    return np.array(preds), np.array(labels), np.array(scores)


def _vgae_threshold(labels, errors):
    """Find optimal anomaly-detection threshold via Youden's J statistic.

    Returns (threshold, youden_j, preds).
    """
    from sklearn.metrics import roc_curve as _roc_curve

    fpr_v, tpr_v, thresholds_v = _roc_curve(labels, errors)
    j_scores = tpr_v - fpr_v
    best_idx = np.argmax(j_scores)
    best_thresh = (float(thresholds_v[best_idx])
                   if best_idx < len(thresholds_v)
                   else float(np.median(errors)))
    preds = (errors > best_thresh).astype(int)
    return best_thresh, float(j_scores[best_idx]), preds


def evaluate(cfg: PipelineConfig) -> dict:
    """Evaluate trained model(s) on validation and held-out test data.

    Fixes applied vs. original implementation:
    - Seeds RNG before data loading so val split matches training.
    - Loads frozen config.json from each training stage so model
      architecture matches what was actually trained (critical for students).
    - Uses consistent scalar label extraction across all model types.
    - Evaluates on held-out test scenarios (test_01_*, test_02_*, ...) in
      addition to the training-val split.

    Output metrics.json layout (backward-compatible top level):
        {
            "gat":    {"core": {...}, "additional": {...}},
            "vgae":   {"core": {...}, "additional": {...}},
            "fusion": {"core": {...}, "additional": {...}},
            "test": {
                "gat":    {"test_01_...": {"core": ...}, ...},
                "vgae":   {"test_01_...": {"core": ...}, ...},
                "fusion": {"test_01_...": {"core": ...}, ...}
            }
        }
    """
    log.info("=== EVALUATION: %s / %s ===", cfg.dataset, cfg.model_size)
    pl.seed_everything(cfg.seed)

    train_data, val_data, num_ids, in_ch = _load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load held-out test scenarios
    test_scenarios = _load_test_data(cfg)

    all_metrics: dict = {}
    test_metrics: dict = {}

    # Determine stage names based on use_kd flag
    gat_stage = "curriculum_kd" if cfg.use_kd else "curriculum"
    vgae_stage = "autoencoder_kd" if cfg.use_kd else "autoencoder"

    # ---- GAT evaluation (primary classifier) ----
    gat_ckpt = checkpoint_path(cfg, gat_stage)
    if gat_ckpt.exists():
        gat_cfg = _load_frozen_cfg(cfg, gat_stage)
        from src.models.models import GATWithJK

        gat = GATWithJK(
            num_ids=num_ids, in_channels=in_ch,
            hidden_channels=gat_cfg.gat_hidden, out_channels=2,
            num_layers=gat_cfg.gat_layers, heads=gat_cfg.gat_heads,
            dropout=gat_cfg.gat_dropout, embedding_dim=gat_cfg.gat_embedding_dim,
        )
        gat.load_state_dict(torch.load(gat_ckpt, map_location="cpu", weights_only=True))
        gat.to(device)
        gat.eval()

        # Val
        p, l, s = _run_gat_inference(gat, val_data, device)
        all_metrics["gat"] = _compute_metrics(l, p, s)
        log.info("GAT val metrics: %s",
                 {k: f"{v:.4f}" for k, v in all_metrics["gat"]["core"].items()
                  if isinstance(v, float)})

        # Test scenarios
        if test_scenarios:
            test_metrics["gat"] = {}
            for scenario, tdata in test_scenarios.items():
                tp, tl, ts = _run_gat_inference(gat, tdata, device)
                test_metrics["gat"][scenario] = _compute_metrics(tl, tp, ts)
                log.info("GAT %s  acc=%.4f f1=%.4f",
                         scenario,
                         test_metrics["gat"][scenario]["core"]["accuracy"],
                         test_metrics["gat"][scenario]["core"]["f1"])

        del gat
        _cleanup()

    # ---- VGAE evaluation (anomaly detection via reconstruction error) ----
    vgae_ckpt = checkpoint_path(cfg, vgae_stage)
    if vgae_ckpt.exists():
        vgae_cfg = _load_frozen_cfg(cfg, vgae_stage)
        from src.models.vgae import GraphAutoencoderNeighborhood

        vgae = GraphAutoencoderNeighborhood(
            num_ids=num_ids, in_channels=in_ch,
            hidden_dims=list(vgae_cfg.vgae_hidden_dims), latent_dim=vgae_cfg.vgae_latent_dim,
            encoder_heads=vgae_cfg.vgae_heads, embedding_dim=vgae_cfg.vgae_embedding_dim,
            dropout=vgae_cfg.vgae_dropout,
        )
        vgae.load_state_dict(torch.load(vgae_ckpt, map_location="cpu", weights_only=True))
        vgae.to(device)
        vgae.eval()

        # Val — find threshold on val, apply to val
        errors_np, labels_np = _run_vgae_inference(vgae, val_data, device)
        best_thresh, youden_j, vgae_preds = _vgae_threshold(labels_np, errors_np)
        all_metrics["vgae"] = _compute_metrics(labels_np, vgae_preds, errors_np)
        all_metrics["vgae"]["core"]["optimal_threshold"] = best_thresh
        all_metrics["vgae"]["core"]["youden_j"] = youden_j
        log.info("VGAE val metrics: %s",
                 {k: f"{v:.4f}" for k, v in all_metrics["vgae"]["core"].items()
                  if isinstance(v, float)})

        # Test scenarios — apply val threshold to test data
        if test_scenarios:
            test_metrics["vgae"] = {}
            for scenario, tdata in test_scenarios.items():
                te, tl = _run_vgae_inference(vgae, tdata, device)
                tp = (te > best_thresh).astype(int)
                test_metrics["vgae"][scenario] = _compute_metrics(tl, tp, te)
                test_metrics["vgae"][scenario]["core"]["threshold_from_val"] = best_thresh
                log.info("VGAE %s  acc=%.4f f1=%.4f",
                         scenario,
                         test_metrics["vgae"][scenario]["core"]["accuracy"],
                         test_metrics["vgae"][scenario]["core"]["f1"])

        del vgae
        _cleanup()

    # ---- DQN Fusion evaluation ----
    fusion_ckpt = checkpoint_path(cfg, "fusion")
    if fusion_ckpt.exists() and vgae_ckpt.exists() and gat_ckpt.exists():
        vgae_cfg = _load_frozen_cfg(cfg, vgae_stage)
        gat_cfg = _load_frozen_cfg(cfg, gat_stage)
        fusion_cfg = _load_frozen_cfg(cfg, "fusion")
        from src.models.vgae import GraphAutoencoderNeighborhood
        from src.models.models import GATWithJK
        from src.models.dqn import EnhancedDQNFusionAgent

        # Reload VGAE + GAT for fusion state caching
        vgae = GraphAutoencoderNeighborhood(
            num_ids=num_ids, in_channels=in_ch,
            hidden_dims=list(vgae_cfg.vgae_hidden_dims), latent_dim=vgae_cfg.vgae_latent_dim,
            encoder_heads=vgae_cfg.vgae_heads, embedding_dim=vgae_cfg.vgae_embedding_dim,
            dropout=vgae_cfg.vgae_dropout,
        )
        vgae.load_state_dict(torch.load(vgae_ckpt, map_location="cpu", weights_only=True))
        vgae.to(device)

        gat = GATWithJK(
            num_ids=num_ids, in_channels=in_ch,
            hidden_channels=gat_cfg.gat_hidden, out_channels=2,
            num_layers=gat_cfg.gat_layers, heads=gat_cfg.gat_heads,
            dropout=gat_cfg.gat_dropout, embedding_dim=gat_cfg.gat_embedding_dim,
        )
        gat.load_state_dict(torch.load(gat_ckpt, map_location="cpu", weights_only=True))
        gat.to(device)

        # Val
        val_cache = _cache_predictions(vgae, gat, val_data, device, cfg.max_val_samples)

        agent = EnhancedDQNFusionAgent(
            lr=fusion_cfg.fusion_lr, gamma=fusion_cfg.dqn_gamma,
            epsilon=0.0, epsilon_decay=1.0, min_epsilon=0.0,
            buffer_size=fusion_cfg.dqn_buffer_size,
            batch_size=fusion_cfg.dqn_batch_size,
            target_update_freq=fusion_cfg.dqn_target_update,
            device=str(device),
        )
        fusion_sd = torch.load(fusion_ckpt, map_location="cpu", weights_only=True)
        agent.q_network.load_state_dict(fusion_sd["q_network"])
        agent.target_network.load_state_dict(fusion_sd["target_network"])

        fp, fl, fs = _run_fusion_inference(agent, val_cache)
        all_metrics["fusion"] = _compute_metrics(fl, fp, fs)
        log.info("Fusion val metrics: %s",
                 {k: f"{v:.4f}" for k, v in all_metrics["fusion"]["core"].items()
                  if isinstance(v, float)})

        # Test scenarios
        if test_scenarios:
            test_metrics["fusion"] = {}
            for scenario, tdata in test_scenarios.items():
                tc = _cache_predictions(vgae, gat, tdata, device, cfg.max_val_samples)
                tp, tl, ts = _run_fusion_inference(agent, tc)
                test_metrics["fusion"][scenario] = _compute_metrics(tl, tp, ts)
                log.info("Fusion %s  acc=%.4f f1=%.4f",
                         scenario,
                         test_metrics["fusion"][scenario]["core"]["accuracy"],
                         test_metrics["fusion"][scenario]["core"]["f1"])

        del vgae, gat
        _cleanup()

    if test_metrics:
        all_metrics["test"] = test_metrics

    # Log core metrics to MLflow (flattened as model_metric)
    for model_name, model_metrics in all_metrics.items():
        if model_name == "test":
            continue
        if "core" in model_metrics:
            for k, v in model_metrics["core"].items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{model_name}_{k}", v)

    # Save all metrics
    out = stage_dir(cfg, "evaluation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    cfg.save(config_path(cfg, "evaluation"))
    log.info("All metrics saved to %s/metrics.json", out)
    _cleanup()
    return all_metrics


# ---------------------------------------------------------------------------
# Dispatch table (imported by cli.py)
# ---------------------------------------------------------------------------

STAGE_FNS = {
    "autoencoder": train_autoencoder,
    "curriculum":  train_curriculum,
    "normal":      train_normal,
    "fusion":      train_fusion,
    "evaluation":  evaluate,
}
