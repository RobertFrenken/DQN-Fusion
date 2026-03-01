"""Shared utilities for training stages."""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch_geometric.loader import DataLoader, DynamicBatchSampler

from graphids.config import (
    PipelineConfig,
    stage_dir,
    config_path,
    data_dir,
    cache_dir,
)
from graphids.config.constants import MMAP_TENSOR_LIMIT
from ..tracking import get_memory_summary
from ..memory import (
    MemoryBudget,
    compute_batch_size,
    save_budget_cache,
    load_budget_cache,
    _get_gpu_memory_mb,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def graph_label(g) -> int:
    """Extract scalar graph-level label consistently."""
    return g.y.item() if g.y.dim() == 0 else int(g.y[0].item())


# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Memory monitoring callback
# ---------------------------------------------------------------------------


class MemoryMonitorCallback(pl.Callback):
    """Log memory usage at epoch boundaries."""

    def __init__(
        self,
        log_every_n_epochs: int = 5,
        predicted_peak_mb: float = 0.0,
        run_id: str = "",
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.predicted_peak_mb = predicted_peak_mb
        self._logged_first_epoch = False

    def on_train_start(self, trainer, pl_module):
        log.info("Memory at train start: %s", get_memory_summary())

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch % self.log_every_n_epochs == 0:
            log.info("Epoch %d memory: %s", epoch, get_memory_summary())

        # Log predicted vs actual peak memory after the first epoch
        if not self._logged_first_epoch and epoch == 0 and torch.cuda.is_available():
            self._logged_first_epoch = True
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            if self.predicted_peak_mb > 0:
                ratio = peak_mb / self.predicted_peak_mb
                error_pct = abs(1.0 - ratio) * 100
                log.info("Memory prediction: ratio=%.3f error=%.1f%%", ratio, error_pct)
            torch.cuda.reset_peak_memory_stats()
            log.info(
                "Epoch 0 peak memory: actual=%.1fMB predicted=%.1fMB",
                peak_mb,
                self.predicted_peak_mb,
            )

    def on_train_end(self, trainer, pl_module):
        log.info("Memory at train end: %s", get_memory_summary())


# ---------------------------------------------------------------------------
# Profiler callback
# ---------------------------------------------------------------------------


class ProfilerCallback(pl.Callback):
    """PyTorch Profiler callback that captures CPU/GPU traces for TensorBoard.

    Writes Chrome-format traces to ``output_dir`` that can be loaded via
    ``tensorboard --logdir <output_dir>`` with the PyTorch Profiler plugin.
    """

    def __init__(
        self,
        output_dir: str,
        wait_steps: int = 1,
        warmup_steps: int = 1,
        active_steps: int = 5,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self._profiler = None

    def on_train_start(self, trainer, pl_module):
        from torch.profiler import profile, schedule, tensorboard_trace_handler

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._profiler = profile(
            schedule=schedule(
                wait=self.wait_steps,
                warmup=self.warmup_steps,
                active=self.active_steps,
            ),
            on_trace_ready=tensorboard_trace_handler(self.output_dir),
            record_shapes=True,
            with_stack=True,
        )
        self._profiler.__enter__()
        log.info("Profiler started — traces will be saved to %s", self.output_dir)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._profiler is not None:
            self._profiler.step()

    def on_train_end(self, trainer, pl_module):
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            log.info("Profiler stopped — traces saved to %s", self.output_dir)
            self._profiler = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(cfg: PipelineConfig):
    """Load graph dataset. Returns (train_graphs, val_graphs, num_ids, in_channels)."""
    from graphids.core.training.datamodules import load_dataset

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


def _get_representative_graph(train_data, cfg: PipelineConfig):
    """Get the p95 graph by node count for conservative batch sizing.

    Falls back to ``train_data[0]`` when cache metadata is unavailable.
    """
    import json as _json

    metadata_path = cache_dir(cfg) / "cache_metadata.json"
    if metadata_path.exists():
        try:
            meta = _json.loads(metadata_path.read_text())
            p95_nodes = meta.get("graph_stats", {}).get("node_count", {}).get("p95")
            if p95_nodes:
                candidates = [train_data[i] for i in range(min(1000, len(train_data)))]
                best = min(candidates, key=lambda g: abs(g.x.size(0) - p95_nodes))
                log.info(
                    "Representative graph: p95=%d nodes, selected=%d nodes",
                    p95_nodes,
                    best.x.size(0),
                )
                return best
        except Exception as e:
            log.warning("Failed to read graph stats: %s", e)

    return train_data[0]


def compute_optimal_batch_size(
    model: nn.Module,
    train_data,
    cfg: PipelineConfig,
    teacher: Optional[nn.Module] = None,
    run_dir: Optional[Path] = None,
) -> int:
    """Compute optimal batch size using memory analysis.

    Uses the p95 graph from cache metadata for conservative sizing.
    Falls back to safety_factor if estimation fails.  Results are cached
    to ``memory_cache.json`` in *run_dir* (if provided) for faster
    subsequent runs with the same config.
    """
    if len(train_data) == 0:
        log.warning("Empty training data, using fallback batch size")
        return effective_batch_size(cfg)

    # Check cache first
    if run_dir is not None:
        cached = load_budget_cache(run_dir, cfg)
        if cached is not None:
            log.info("Using cached batch size: %d", cached.recommended_batch_size)
            return cached.recommended_batch_size

    sample_graph = _get_representative_graph(train_data, cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Trial mode: binary search with actual forward+backward passes
    if cfg.training.memory_estimation == "trial":
        from ..memory import _trial_batch_size

        try:
            trial_bs = _trial_batch_size(
                model,
                train_data,
                device,
                min_bs=8,
                max_bs=cfg.training.batch_size,
                precision=cfg.training.precision,
            )
            log.info("Trial batch size: %d (max=%d)", trial_bs, cfg.training.batch_size)
            # Cache result using existing mechanism
            if run_dir is not None:
                budget = MemoryBudget(
                    total_gpu_mb=_get_gpu_memory_mb(device),
                    recommended_batch_size=trial_bs,
                    estimation_mode="trial",
                )
                save_budget_cache(budget, run_dir, cfg)
            return trial_bs
        except Exception as e:
            log.warning("Trial batch size failed: %s, falling back to measured", e)

    mode = (
        cfg.training.memory_estimation
        if cfg.training.memory_estimation in ("static", "measured")
        else "measured"
    )

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
            budget.recommended_batch_size,
            mode,
            cfg.training.batch_size,
            teacher is not None,
        )

        # Save to cache for next run
        if run_dir is not None:
            save_budget_cache(budget, run_dir, cfg)

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
        1
        for attr in ["x", "edge_index", "y", "edge_attr", "batch"]
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
                tensor_count,
                MMAP_TENSOR_LIMIT,
            )
            return 0
    return nw


def compute_node_budget(batch_size: int, cfg: PipelineConfig) -> int | None:
    """Derive max_num_nodes from batch_size * p95 graph node count.

    Returns None when cache metadata is unavailable (falls back to static batching).
    """
    import json as _json

    metadata_path = cache_dir(cfg) / "cache_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        meta = _json.loads(metadata_path.read_text())
        p95 = meta.get("graph_stats", {}).get("node_count", {}).get("p95")
        if not p95:
            return None
        return int(batch_size * p95)
    except Exception as e:
        log.warning("Failed to read graph stats for node budget: %s", e)
        return None


def make_dataloader(
    data,
    cfg: PipelineConfig,
    batch_size: int,
    shuffle: bool = True,
    max_num_nodes: int | None = None,
) -> DataLoader:
    """Create a DataLoader with consistent settings.

    When *max_num_nodes* is provided, uses ``DynamicBatchSampler`` to pack
    variable-size graphs up to a node budget per batch.  Falls back to
    single-process loading (num_workers=0) when the dataset has too many
    tensor storages for the kernel mmap limit.
    """
    nw = _safe_num_workers(data, cfg)

    if max_num_nodes is not None:
        # num_steps required so Lightning can call len(dataloader)
        num_steps = max(1, len(data) // max(1, batch_size))
        sampler = DynamicBatchSampler(
            data,
            max_num=max_num_nodes,
            mode="node",
            shuffle=shuffle,
            num_steps=num_steps,
        )
        return DataLoader(
            data,
            batch_sampler=sampler,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
            multiprocessing_context=cfg.mp_start_method if nw > 0 else None,
        )

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
            return {
                k.replace("model.", ""): v for k, v in sd.items() if k.startswith("model.")
            } or sd
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
    """Load a teacher model from its checkpoint for knowledge distillation.

    Uses the model registry (``registry.get(model_type).factory()``) to
    construct the architecture, then loads weights from *teacher_path*.
    Dimensions come from the **frozen config.json** saved alongside the
    checkpoint — never from the student config — preventing shape mismatches
    when teacher and student have different hidden sizes.

    The returned model is moved to *device*, set to eval mode, and has all
    parameters frozen (``requires_grad=False``).
    """
    from graphids.core.models.registry import get as registry_get

    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
    sd = _extract_state_dict(checkpoint)

    teacher_cfg_path = Path(teacher_path).parent / "config.json"
    if not teacher_cfg_path.exists():
        raise FileNotFoundError(
            f"Teacher config not found: {teacher_cfg_path}. "
            f"Cannot load teacher without its frozen config (risk of dimension mismatch)."
        )
    tcfg = PipelineConfig.load(teacher_cfg_path)

    # Infer num_ids from checkpoint embedding if present
    t_num_ids = num_ids
    for key in sd:
        if key.endswith("id_embedding.weight"):
            t_num_ids = sd[key].shape[0]
            break

    teacher = registry_get(model_type).factory(tcfg, t_num_ids, in_channels)

    # DQN checkpoints have nested state dict
    if model_type == "dqn":
        if "q_network" in sd:
            teacher.load_state_dict(sd["q_network"])
        elif "q_network_state_dict" in sd:
            teacher.load_state_dict(sd["q_network_state_dict"])
        else:
            teacher.load_state_dict(sd)
    else:
        teacher.load_state_dict(sd)

    log.info("Loaded %s teacher from %s (num_ids=%d)", model_type, teacher_path, t_num_ids)

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
        s_dim = getattr(student_model, "latent_dim", getattr(student_model, "_latent_dim", 16))
        t_dim = getattr(teacher, "latent_dim", getattr(teacher, "_latent_dim", 96))
    elif model_type == "gat":
        s_dim = getattr(student_model, "hidden_channels", getattr(student_model, "out_channels", 2))
        t_dim = getattr(teacher, "hidden_channels", getattr(teacher, "out_channels", 2))
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
    return (
        Path(cfg.experiment_root)
        / cfg.dataset
        / f"{model_type}_{cfg.scale}_{stage}{aux_suffix}"
        / filename
    )


# Stage → canonical model_type that owns that stage's artifacts.
_STAGE_MODEL_TYPE = {
    "autoencoder": "vgae",
    "curriculum": "gat",
    "normal": "gat",
    "fusion": "dqn",
}


def load_frozen_cfg(
    cfg: PipelineConfig, stage: str, model_type: str | None = None
) -> PipelineConfig:
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


def load_model(
    cfg: PipelineConfig,
    model_type: str,
    stage: str,
    num_ids: int,
    in_channels: int,
    device: torch.device,
) -> nn.Module:
    """Load a trained model using its frozen config and the registry.

    Replaces the old ``load_vgae`` / ``load_gat`` helpers with a single
    generic loader that works for any registered model type.
    """
    from graphids.core.models.registry import get as registry_get

    frozen_cfg = load_frozen_cfg(cfg, stage, model_type=model_type)
    ckpt = _cross_model_path(cfg, model_type, stage, "best_model.pt")
    model = registry_get(model_type).factory(frozen_cfg, num_ids, in_channels)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return model


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
            optimizer,
            step_size=t.scheduler_step_size,
            gamma=t.scheduler_gamma,
        )
    elif t.scheduler_type == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=t.monitor_mode,
            patience=t.patience // 2,
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


def _make_loggers(
    cfg: PipelineConfig,
    stage: str,
    out: Path,
    run_id_str: str,
) -> list:
    """Build Lightning loggers: CSV (always) + W&B (when available).

    If a W&B run is already active (initialized by cli.py), the WandbLogger
    attaches to it instead of creating a new run.
    """
    loggers: list = [CSVLogger(save_dir=str(out), name="csv_logs")]

    try:
        import wandb

        if wandb.run is not None:
            # Attach to existing run started by cli.py
            loggers.append(WandbLogger(experiment=wandb.run))
        else:
            log.debug("No active wandb run — WandbLogger skipped in trainer")
    except ImportError:
        log.debug("wandb not installed — skipping WandbLogger")

    return loggers


def make_trainer(
    cfg: PipelineConfig,
    stage: str,
    predicted_peak_mb: float = 0.0,
    run_id_str: str = "",
) -> pl.Trainer:
    """Create a Lightning Trainer with standard callbacks."""
    from graphids.config import run_id as _run_id

    t = cfg.training
    out = stage_dir(cfg, stage)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = t.cudnn_benchmark

    rid = run_id_str or _run_id(cfg, stage)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(out),
            filename="best_model",
            monitor=t.monitor_metric,
            mode=t.monitor_mode,
            save_top_k=t.save_top_k,
        ),
        EarlyStopping(
            monitor=t.monitor_metric,
            patience=t.patience,
            mode=t.monitor_mode,
        ),
        MemoryMonitorCallback(
            log_every_n_epochs=t.test_every_n_epochs,
            predicted_peak_mb=predicted_peak_mb,
            run_id=rid,
        ),
    ]

    if t.profile:
        profiler_dir = str(out / "profiler_traces")
        callbacks.append(
            ProfilerCallback(
                output_dir=profiler_dir,
                active_steps=t.profile_steps,
            )
        )

    return pl.Trainer(
        default_root_dir=str(out),
        max_epochs=t.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=t.precision,
        gradient_clip_val=t.gradient_clip,
        accumulate_grad_batches=t.accumulate_grad_batches,
        callbacks=callbacks,
        logger=_make_loggers(cfg, stage, out, rid),
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


def cache_predictions(models: dict[str, nn.Module], data, device, max_samples: int = 150_000):
    """Run registered extractors over data, produce N-D state vectors for DQN.

    ``models`` maps model_type name to loaded model (e.g. ``{"vgae": vgae, "gat": gat}``).
    Feature concatenation order follows registry registration order (VGAE then GAT)
    to preserve the existing 15-D layout.
    """
    from graphids.core.models.registry import extractors as registry_extractors

    registered = registry_extractors()
    active = [(name, ext) for name, ext in registered if name in models]

    states, labels = [], []
    for model in models.values():
        model.eval()
    n_samples = min(len(data), max_samples)

    with torch.no_grad():
        for i in range(n_samples):
            g = data[i].clone().to(device)
            batch_idx = (
                g.batch
                if hasattr(g, "batch") and g.batch is not None
                else torch.zeros(g.x.size(0), dtype=torch.long, device=device)
            )

            features = [ext.extract(models[name], g, batch_idx, device) for name, ext in active]
            states.append(torch.cat(features))
            labels.append(g.y[0] if g.y.dim() > 0 else g.y)

    return {"states": torch.stack(states), "labels": torch.tensor(labels)}
