"""Shared utilities for training stages."""
from __future__ import annotations

import gc
import glob
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from ..config import PipelineConfig
from ..paths import stage_dir, checkpoint_path, config_path, data_dir, cache_dir
from ..tracking import log_memory_metrics, get_memory_summary
from ..memory import compute_batch_size, log_memory_state

if TYPE_CHECKING:
    from torch_geometric.data import Data

log = logging.getLogger(__name__)

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cfg: PipelineConfig):
    """Load graph dataset. Returns (train_graphs, val_graphs, num_ids, in_channels)."""
    from src.training.datamodules import load_dataset

    train_data, val_data, num_ids = load_dataset(
        cfg.dataset,
        dataset_path=data_dir(cfg),
        cache_dir_path=cache_dir(cfg),
    )
    in_channels = train_data[0].x.shape[1] if train_data else 11
    return train_data, val_data, num_ids, in_channels


# ---------------------------------------------------------------------------
# Batch size computation
# ---------------------------------------------------------------------------

def effective_batch_size(cfg: PipelineConfig) -> int:
    """Apply safety factor to batch size (legacy fallback)."""
    return max(8, int(cfg.batch_size * cfg.safety_factor))


def compute_optimal_batch_size(
    model: nn.Module,
    train_data,
    cfg: PipelineConfig,
    teacher: Optional[nn.Module] = None,
) -> int:
    """Compute optimal batch size using memory analysis.

    Uses cfg.memory_estimation: "static" (fast) or "measured" (accurate).
    Falls back to Lightning Tuner or safety_factor if estimation fails.
    """
    if len(train_data) == 0:
        log.warning("Empty training data, using fallback batch size")
        return effective_batch_size(cfg)

    sample_graph = train_data[0]
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    mode = cfg.memory_estimation if cfg.memory_estimation in ("static", "measured") else "measured"

    try:
        target_utilization = min(0.85, cfg.safety_factor + 0.15)

        budget = compute_batch_size(
            model=model,
            sample_graph=sample_graph,
            device=device,
            teacher=teacher,
            precision=cfg.precision,
            target_utilization=target_utilization,
            min_batch_size=8,
            max_batch_size=cfg.batch_size,
            mode=mode,
        )

        log.info(
            "Batch size: %d (mode=%s, max=%d, KD=%s)",
            budget.recommended_batch_size, mode, cfg.batch_size, teacher is not None
        )
        return budget.recommended_batch_size

    except Exception as e:
        log.warning("Memory estimation failed: %s", e)

    # Fallback to Lightning Tuner if configured
    if cfg.batch_size_mode in ("binsearch", "power"):
        log.info("Falling back to Lightning Tuner (mode=%s)", cfg.batch_size_mode)
        return _lightning_tune_batch_size(model, train_data, cfg)

    return effective_batch_size(cfg)


def _lightning_tune_batch_size(
    module: nn.Module,
    train_data,
    cfg: PipelineConfig,
) -> int:
    """Use Lightning Tuner for batch size (trial-based, slower but accurate)."""
    from src.training.datamodules import CANGraphDataModule
    from lightning.pytorch.tuner import Tuner

    val_size = max(1, len(train_data) // 10)
    val_data = train_data[:val_size]

    if not isinstance(module, pl.LightningModule):
        log.warning("Lightning Tuner requires LightningModule, using fallback")
        return effective_batch_size(cfg)

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
        log.info("Lightning Tuner: %d -> tuned %d -> safe %d (sf=%.2f)",
                 cfg.batch_size, tuned, safe, cfg.safety_factor)
        return safe
    except Exception as e:
        log.warning("Lightning Tuner failed: %s. Using safety factor.", e)
        return effective_batch_size(cfg)
    finally:
        for ckpt in glob.glob(".scale_batch_size_*.ckpt"):
            Path(ckpt).unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    data,
    cfg: PipelineConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader with consistent settings."""
    nw = cfg.num_workers
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        multiprocessing_context=cfg.mp_start_method if nw > 0 else None,
    )


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


def load_teacher(
    teacher_path: str,
    model_type: str,
    cfg: PipelineConfig,
    num_ids: int,
    in_channels: int,
    device: torch.device,
) -> nn.Module:
    """Load teacher model using its frozen config."""
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
    sd = _extract_state_dict(checkpoint)

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

        id_emb = sd.get("encoder.id_embedding.weight", sd.get("id_embedding.weight"))
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
        log.info("Loaded DQN teacher: state_dim=%d, action_dim=%d", state_dim, action_dim)

    else:
        raise ValueError(f"Cannot load teacher for model_type={model_type}")

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def make_projection(
    student_model: nn.Module,
    teacher: nn.Module,
    model_type: str,
    device: torch.device,
) -> Optional[nn.Linear]:
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
# Model loading factory
# ---------------------------------------------------------------------------

def load_frozen_cfg(cfg: PipelineConfig, stage: str) -> PipelineConfig:
    """Load the frozen config.json saved during training for *stage*."""
    p = config_path(cfg, stage)
    if p.exists():
        try:
            return PipelineConfig.load(p)
        except Exception as e:
            log.warning("Could not load frozen config %s: %s", p, e)
    return cfg


def load_vgae(
    cfg: PipelineConfig,
    num_ids: int,
    in_channels: int,
    device: torch.device,
    stage: str = "autoencoder",
) -> nn.Module:
    """Load trained VGAE model using its frozen config."""
    from src.models.vgae import GraphAutoencoderNeighborhood

    vgae_cfg = load_frozen_cfg(cfg, stage)
    vgae = GraphAutoencoderNeighborhood(
        num_ids=num_ids, in_channels=in_channels,
        hidden_dims=list(vgae_cfg.vgae_hidden_dims), latent_dim=vgae_cfg.vgae_latent_dim,
        encoder_heads=vgae_cfg.vgae_heads, embedding_dim=vgae_cfg.vgae_embedding_dim,
        dropout=vgae_cfg.vgae_dropout,
    )
    vgae.load_state_dict(torch.load(
        checkpoint_path(cfg, stage), map_location="cpu", weights_only=True,
    ))
    vgae.to(device)
    vgae.eval()
    return vgae


def load_gat(
    cfg: PipelineConfig,
    num_ids: int,
    in_channels: int,
    device: torch.device,
    stage: str = "curriculum",
) -> nn.Module:
    """Load trained GAT model using its frozen config."""
    from src.models.models import GATWithJK

    gat_cfg = load_frozen_cfg(cfg, stage)
    gat = GATWithJK(
        num_ids=num_ids, in_channels=in_channels,
        hidden_channels=gat_cfg.gat_hidden, out_channels=2,
        num_layers=gat_cfg.gat_layers, heads=gat_cfg.gat_heads,
        dropout=gat_cfg.gat_dropout, embedding_dim=gat_cfg.gat_embedding_dim,
    )
    gat.load_state_dict(torch.load(
        checkpoint_path(cfg, stage), map_location="cpu", weights_only=True,
    ))
    gat.to(device)
    gat.eval()
    return gat


# ---------------------------------------------------------------------------
# LR scheduler helper
# ---------------------------------------------------------------------------

def build_optimizer_dict(optimizer, cfg: PipelineConfig):
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

def make_trainer(cfg: PipelineConfig, stage: str) -> pl.Trainer:
    """Create a Lightning Trainer with standard callbacks."""
    out = stage_dir(cfg, stage)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

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


# ---------------------------------------------------------------------------
# Cleanup & caching
# ---------------------------------------------------------------------------

def cleanup():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cache_predictions(vgae, gat, data, device, max_samples: int = 150_000):
    """Run VGAE + GAT inference, produce 15-D state vectors for DQN.

    State layout:
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
            g = data[i].clone().to(device)
            batch_idx = (g.batch if hasattr(g, "batch") and g.batch is not None
                         else torch.zeros(g.x.size(0), dtype=torch.long, device=device))

            # VGAE features
            cont, canid_logits, nbr_logits, z, _ = vgae(g.x, g.edge_index, batch_idx)
            recon_err = F.mse_loss(cont, g.x[:, 1:], reduction="none").mean().item()
            canid_err = F.cross_entropy(canid_logits, g.x[:, 0].long()).item()
            nbr_targets = vgae.create_neighborhood_targets(g.x, g.edge_index, batch_idx)
            nbr_err = F.binary_cross_entropy_with_logits(
                nbr_logits, nbr_targets, reduction="mean").item()
            z_mean, z_std = z.mean().item(), z.std().item()
            z_max, z_min = z.max().item(), z.min().item()
            vgae_conf = 1.0 / (1.0 + recon_err)

            # GAT features
            xs = gat(g, return_intermediate=True)
            jk_out = gat.jk(xs)
            pooled = global_mean_pool(jk_out, batch_idx)
            emb_mean, emb_std = pooled.mean().item(), (pooled.std().item() if pooled.numel() > 1 else 0.0)
            emb_max, emb_min = pooled.max().item(), pooled.min().item()
            x = pooled
            for layer in gat.fc_layers:
                x = layer(x)
            probs = F.softmax(x, dim=1)
            p0, p1 = probs[0, 0].item(), probs[0, 1].item()
            entropy = -(probs * (probs + 1e-8).log()).sum().item()
            gat_conf = max(0.0, min(1.0, 1.0 - entropy / 0.693))

            state = torch.tensor([
                recon_err, nbr_err, canid_err,
                z_mean, z_std, z_max, z_min,
                vgae_conf,
                p0, p1,
                emb_mean, emb_std, emb_max, emb_min,
                gat_conf,
            ])
            states.append(state)
            labels.append(g.y[0] if g.y.dim() > 0 else g.y)

    return {"states": torch.stack(states), "labels": torch.tensor(labels)}
