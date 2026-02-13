"""Shared utilities for training stages."""
from __future__ import annotations

import gc
import logging
import math
import sys
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

from config import PipelineConfig, stage_dir, checkpoint_path, config_path, data_dir, cache_dir
from config.constants import MMAP_TENSOR_LIMIT
from ..tracking import log_memory_metrics, get_memory_summary
from ..memory import compute_batch_size, log_memory_state

if TYPE_CHECKING:
    from torch_geometric.data import Data

log = logging.getLogger(__name__)


def graph_label(g) -> int:
    """Extract scalar graph-level label consistently."""
    return g.y.item() if g.y.dim() == 0 else int(g.y[0].item())


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
        seed=cfg.seed,
    )
    in_channels = train_data[0].x.shape[1] if train_data else 11
    return train_data, val_data, num_ids, in_channels


# ---------------------------------------------------------------------------
# Batch size computation
# ---------------------------------------------------------------------------

def effective_batch_size(cfg: PipelineConfig) -> int:
    """Apply safety factor to batch size (legacy fallback)."""
    return max(8, int(cfg.training.batch_size * cfg.training.safety_factor))


def compute_optimal_batch_size(
    model: nn.Module,
    train_data,
    cfg: PipelineConfig,
    teacher: Optional[nn.Module] = None,
) -> int:
    """Compute optimal batch size using memory analysis.

    Uses cfg.training.memory_estimation: "static" (fast) or "measured" (accurate).
    Falls back to Lightning Tuner or safety_factor if estimation fails.
    """
    if len(train_data) == 0:
        log.warning("Empty training data, using fallback batch size")
        return effective_batch_size(cfg)

    sample_graph = train_data[0]
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    mode = cfg.training.memory_estimation if cfg.training.memory_estimation in ("static", "measured") else "measured"

    try:
        target_utilization = min(0.85, cfg.training.safety_factor + 0.15)

        budget = compute_batch_size(
            model=model,
            sample_graph=sample_graph,
            device=device,
            teacher=teacher,
            precision=cfg.training.precision,
            target_utilization=target_utilization,
            min_batch_size=8,
            max_batch_size=cfg.training.batch_size,
            mode=mode,
        )

        log.info(
            "Batch size: %d (mode=%s, max=%d, KD=%s)",
            budget.recommended_batch_size, mode, cfg.training.batch_size, teacher is not None
        )
        return budget.recommended_batch_size

    except Exception as e:
        log.warning("Memory estimation failed: %s", e)

    return effective_batch_size(cfg)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _estimate_tensor_count(data) -> int:
    """Estimate number of tensor storages in a graph dataset."""
    if not data:
        return 0
    sample = data[0]
    tensors_per_graph = sum(
        1 for attr in ['x', 'edge_index', 'y', 'edge_attr', 'batch']
        if hasattr(sample, attr) and getattr(sample, attr) is not None
    )
    return len(data) * tensors_per_graph


def _safe_num_workers(data, cfg: PipelineConfig) -> int:
    """Return num_workers, falling back to 0 if dataset exceeds mmap limits.

    With spawn multiprocessing, every tensor storage needs a separate mmap
    entry.  Calling share_memory_() does NOT help -- it also creates one mmap
    per tensor.  The only safe option for large datasets is num_workers=0.
    """
    nw = cfg.num_workers
    if nw > 0 and cfg.mp_start_method == "spawn":
        tensor_count = _estimate_tensor_count(data)
        if tensor_count > MMAP_TENSOR_LIMIT:
            log.warning(
                "Dataset has %d tensor storages (limit %d for vm.max_map_count). "
                "Falling back to num_workers=0 to avoid mmap OOM.",
                tensor_count, MMAP_TENSOR_LIMIT
            )
            return 0
    return nw


def make_dataloader(
    data,
    cfg: PipelineConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader with consistent settings.

    Falls back to single-process loading (num_workers=0) when the dataset has
    too many tensor storages for the kernel mmap limit.
    """
    nw = _safe_num_workers(data, cfg)

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
    if not teacher_cfg_path.exists():
        raise FileNotFoundError(
            f"Teacher config not found: {teacher_cfg_path}. "
            f"Cannot load teacher without its frozen config (risk of dimension mismatch)."
        )
    tcfg = PipelineConfig.load(teacher_cfg_path)

    if model_type == "vgae":
        from src.models.vgae import GraphAutoencoderNeighborhood

        id_emb = sd.get("encoder.id_embedding.weight", sd.get("id_embedding.weight"))
        t_num_ids = id_emb.shape[0] if id_emb is not None else num_ids

        teacher = GraphAutoencoderNeighborhood(
            num_ids=t_num_ids, in_channels=in_channels,
            hidden_dims=list(tcfg.vgae.hidden_dims), latent_dim=tcfg.vgae.latent_dim,
            encoder_heads=tcfg.vgae.heads, embedding_dim=tcfg.vgae.embedding_dim,
            dropout=tcfg.vgae.dropout,
        )
        teacher.load_state_dict(sd)
        log.info("Loaded VGAE teacher: latent_dim=%d, num_ids=%d",
                 tcfg.vgae.latent_dim, t_num_ids)

    elif model_type == "gat":
        from src.models.gat import GATWithJK

        id_emb = sd.get("id_embedding.weight")
        t_num_ids = id_emb.shape[0] if id_emb is not None else num_ids

        teacher = GATWithJK(
            num_ids=t_num_ids, in_channels=in_channels,
            hidden_channels=tcfg.gat.hidden, out_channels=2,
            num_layers=tcfg.gat.layers, heads=tcfg.gat.heads,
            dropout=tcfg.gat.dropout,
            num_fc_layers=tcfg.gat.fc_layers,
            embedding_dim=tcfg.gat.embedding_dim,
        )
        teacher.load_state_dict(sd)
        log.info("Loaded GAT teacher: hidden=%d, layers=%d, num_ids=%d",
                 tcfg.gat.hidden, tcfg.gat.layers, t_num_ids)

    elif model_type == "dqn":
        from src.models.dqn import QNetwork

        state_dim = 15
        action_dim = tcfg.fusion.alpha_steps
        teacher = QNetwork(state_dim, action_dim,
                           hidden_dim=tcfg.dqn.hidden, num_layers=tcfg.dqn.layers)

        if "q_network" in sd:
            teacher.load_state_dict(sd["q_network"])
        elif "q_network_state_dict" in sd:
            teacher.load_state_dict(sd["q_network_state_dict"])
        else:
            teacher.load_state_dict(sd)
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

def _cross_model_path(cfg: PipelineConfig, model_type: str, stage: str, filename: str) -> Path:
    """Build a path for a specific model_type (may differ from cfg.model_type).

    Used when loading another model's artifacts (e.g. loading VGAE checkpoint from GAT config).
    """
    aux_suffix = f"_{cfg.auxiliaries[0].type}" if cfg.auxiliaries else ""
    return (Path(cfg.experiment_root) / cfg.dataset
            / f"{model_type}_{cfg.scale}_{stage}{aux_suffix}" / filename)


# Stage → canonical model_type that owns that stage's artifacts.
_STAGE_MODEL_TYPE = {
    "autoencoder": "vgae",
    "curriculum": "gat",
    "normal": "gat",
    "fusion": "dqn",
}


def load_frozen_cfg(cfg: PipelineConfig, stage: str, model_type: str | None = None) -> PipelineConfig:
    """Load the frozen config.json saved during training for *stage*.

    model_type defaults to the canonical owner of the stage (e.g. "autoencoder" → "vgae").
    When cfg.model_type already matches the stage owner, this is equivalent to config_path(cfg, stage).

    Raises FileNotFoundError if the frozen config doesn't exist.
    """
    mt = model_type or _STAGE_MODEL_TYPE.get(stage, cfg.model_type)
    if mt == cfg.model_type:
        p = config_path(cfg, stage)
    else:
        p = _cross_model_path(cfg, mt, stage, "config.json")
    if not p.exists():
        raise FileNotFoundError(
            f"Frozen config not found: {p}. "
            f"The '{stage}' stage must be trained first (with config saved) "
            f"before dependent stages can load it."
        )
    try:
        return PipelineConfig.load(p)
    except Exception as e:
        raise RuntimeError(f"Could not load frozen config {p}: {e}") from e


def load_vgae(
    cfg: PipelineConfig,
    num_ids: int,
    in_channels: int,
    device: torch.device,
    stage: str = "autoencoder",
) -> nn.Module:
    """Load trained VGAE model using its frozen config."""
    from src.models.vgae import GraphAutoencoderNeighborhood

    vgae_cfg = load_frozen_cfg(cfg, stage, model_type="vgae")
    ckpt = _cross_model_path(cfg, "vgae", stage, "best_model.pt")
    vgae = GraphAutoencoderNeighborhood(
        num_ids=num_ids, in_channels=in_channels,
        hidden_dims=list(vgae_cfg.vgae.hidden_dims), latent_dim=vgae_cfg.vgae.latent_dim,
        encoder_heads=vgae_cfg.vgae.heads, embedding_dim=vgae_cfg.vgae.embedding_dim,
        dropout=vgae_cfg.vgae.dropout,
    )
    vgae.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
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
    from src.models.gat import GATWithJK

    gat_cfg = load_frozen_cfg(cfg, stage, model_type="gat")
    ckpt = _cross_model_path(cfg, "gat", stage, "best_model.pt")
    gat = GATWithJK(
        num_ids=num_ids, in_channels=in_channels,
        hidden_channels=gat_cfg.gat.hidden, out_channels=2,
        num_layers=gat_cfg.gat.layers, heads=gat_cfg.gat.heads,
        dropout=gat_cfg.gat.dropout,
        num_fc_layers=gat_cfg.gat.fc_layers,
        embedding_dim=gat_cfg.gat.embedding_dim,
    )
    gat.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    gat.to(device)
    gat.eval()
    return gat


# ---------------------------------------------------------------------------
# LR scheduler helper
# ---------------------------------------------------------------------------

def build_optimizer_dict(optimizer, cfg: PipelineConfig):
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    t = cfg.training
    if not t.use_scheduler:
        return optimizer

    t_max = t.scheduler_t_max if t.scheduler_t_max > 0 else t.max_epochs

    if t.scheduler_type == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif t.scheduler_type == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=t.scheduler_step_size, gamma=t.scheduler_gamma,
        )
    elif t.scheduler_type == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=t.monitor_mode, patience=t.patience // 2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "monitor": t.monitor_metric},
        }
    else:
        log.warning("Unknown scheduler_type=%s, skipping", t.scheduler_type)
        return optimizer

    return {"optimizer": optimizer, "lr_scheduler": sched}


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def make_trainer(cfg: PipelineConfig, stage: str) -> pl.Trainer:
    """Create a Lightning Trainer with standard callbacks."""
    t = cfg.training
    out = stage_dir(cfg, stage)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = t.cudnn_benchmark

    mlflow.pytorch.autolog(log_models=False)

    return pl.Trainer(
        default_root_dir=str(out),
        max_epochs=t.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=t.precision,
        gradient_clip_val=t.gradient_clip,
        accumulate_grad_batches=t.accumulate_grad_batches,
        callbacks=[
            ModelCheckpoint(
                dirpath=str(out), filename="best_model",
                monitor=t.monitor_metric, mode=t.monitor_mode,
                save_top_k=t.save_top_k,
            ),
            EarlyStopping(
                monitor=t.monitor_metric, patience=t.patience,
                mode=t.monitor_mode,
            ),
            MemoryMonitorCallback(log_every_n_epochs=t.test_every_n_epochs),
        ],
        log_every_n_steps=t.log_every_n_steps,
        enable_progress_bar=True,
        deterministic=t.deterministic,
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
            gat_conf = max(0.0, min(1.0, 1.0 - entropy / math.log(2)))

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
