"""Single source of truth for all pipeline parameters.

One frozen dataclass. Two preset dicts (architecture + training). JSON save/load.
No Hydra, no Pydantic, no duplicate DEFAULT_MODEL_ARGS.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, asdict, replace
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Architecture presets — determines weight shapes (must match checkpoint).
# ---------------------------------------------------------------------------

ARCHITECTURES: dict[tuple[str, str], dict] = {
    ("vgae", "teacher"): dict(
        vgae_hidden_dims=(480, 240, 48), vgae_latent_dim=48,
        vgae_heads=4, vgae_embedding_dim=32, vgae_dropout=0.15,
    ),
    ("vgae", "student"): dict(
        vgae_hidden_dims=(80, 40, 16), vgae_latent_dim=16,
        vgae_heads=1, vgae_embedding_dim=4, vgae_dropout=0.1,
    ),
    ("gat", "teacher"): dict(
        gat_hidden=48, gat_layers=3, gat_heads=8,
        gat_embedding_dim=16, gat_fc_layers=2,
    ),
    ("gat", "student"): dict(
        gat_hidden=24, gat_layers=2, gat_heads=4,
        gat_embedding_dim=8, gat_fc_layers=3,
    ),
    ("dqn", "teacher"): dict(
        dqn_hidden=576, dqn_layers=3,
        dqn_buffer_size=100_000, dqn_batch_size=128, dqn_target_update=100,
    ),
    ("dqn", "student"): dict(
        dqn_hidden=160, dqn_layers=2,
        dqn_buffer_size=50_000, dqn_batch_size=64, dqn_target_update=50,
    ),
}

# ---------------------------------------------------------------------------
# Training hyperparameters — can be tuned without affecting checkpoint compat.
# ---------------------------------------------------------------------------

TRAINING: dict[str, dict] = {
    "vgae": dict(lr=0.002, max_epochs=300, patience=100, safety_factor=0.55),
    "gat":  dict(lr=0.003, max_epochs=300, patience=100, gat_dropout=0.2, safety_factor=0.5),
    "dqn":  dict(dqn_gamma=0.99, fusion_episodes=500, safety_factor=0.45),
}

# Size-specific training overrides
TRAINING_SIZE: dict[tuple[str, str], dict] = {
    ("vgae", "student"): dict(use_kd=True, safety_factor=0.6),
    ("gat", "student"):  dict(lr=0.001, patience=50, gat_dropout=0.1, use_kd=True, safety_factor=0.55),
    ("dqn", "student"):  dict(safety_factor=0.5),
}

# Backwards-compatible PRESETS view (merged arch + training)
PRESETS: dict[tuple[str, str], dict] = {}
for _key in ARCHITECTURES:
    _model, _size = _key
    _merged = {}
    _merged.update(ARCHITECTURES.get(_key, {}))
    _merged.update(TRAINING.get(_model, {}))
    _merged.update(TRAINING_SIZE.get(_key, {}))
    PRESETS[_key] = _merged
del _key, _model, _size, _merged


@dataclass(frozen=True)
class VGAEConfig:
    hidden_dims: Tuple[int, ...] = (480, 240, 48)
    latent_dim: int = 48
    heads: int = 4
    embedding_dim: int = 32
    dropout: float = 0.15


@dataclass(frozen=True)
class GATConfig:
    hidden: int = 48
    layers: int = 3
    heads: int = 8
    dropout: float = 0.2
    embedding_dim: int = 16
    fc_layers: int = 3


@dataclass(frozen=True)
class DQNConfig:
    hidden: int = 576
    layers: int = 3
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    buffer_size: int = 100_000
    batch_size: int = 128
    target_update: int = 100


@dataclass(frozen=True)
class KDConfig:
    enabled: bool = False
    teacher_path: str = ""
    temperature: float = 4.0
    alpha: float = 0.7
    vgae_latent_weight: float = 0.5
    vgae_recon_weight: float = 0.5


@dataclass(frozen=True)
class FusionConfig:
    episodes: int = 500
    max_samples: int = 150_000
    max_val_samples: int = 30_000
    episode_sample_size: int = 20_000
    training_step_interval: int = 32
    gpu_training_steps: int = 16
    lr: float = 0.001
    alpha_steps: int = 21


@dataclass(frozen=True)
class PipelineConfig:
    """Every tunable parameter lives here. Nowhere else."""

    # --- Identity ---
    dataset:    str = "hcrl_ch"
    modality:   str = "automotive"
    model_size: str = "teacher"          # teacher | student
    seed:       int = 42

    # --- Training ---
    lr:             float = 0.003
    max_epochs:     int   = 300
    batch_size:     int   = 4096
    patience:       int   = 100
    weight_decay:   float = 1e-4
    gradient_clip:  float = 1.0
    precision:      str   = "16-mixed"

    # --- LR scheduling ---
    use_scheduler:       bool  = False
    scheduler_type:      str   = "cosine"  # cosine | step | plateau
    scheduler_t_max:     int   = -1        # -1 = use max_epochs
    scheduler_step_size: int   = 50
    scheduler_gamma:     float = 0.1

    # --- Memory optimization ---
    gradient_checkpointing: bool  = True
    use_teacher_cache:      bool  = True
    clear_cache_every_n:    int   = 100
    offload_teacher_to_cpu: bool  = False

    # --- Batch size ---
    optimize_batch_size:   bool  = True
    safety_factor:         float = 0.5
    memory_estimation:     str   = "measured"  # static | measured

    # --- Training dynamics ---
    accumulate_grad_batches: int  = 1

    # --- Checkpointing & monitoring ---
    save_top_k:          int = 1
    monitor_metric:      str = "val_loss"
    monitor_mode:        str = "min"       # min | max
    log_every_n_steps:   int = 50
    test_every_n_epochs: int = 5

    # --- Reproducibility ---
    deterministic:   bool = False
    cudnn_benchmark: bool = True

    # --- GAT architecture ---
    gat_hidden:        int   = 48
    gat_layers:        int   = 3
    gat_heads:         int   = 8
    gat_dropout:       float = 0.2
    gat_embedding_dim: int   = 16
    gat_fc_layers:     int   = 3

    # --- VGAE architecture ---
    vgae_hidden_dims:  Tuple[int, ...] = (480, 240, 48)
    vgae_latent_dim:   int   = 48
    vgae_heads:        int   = 4
    vgae_embedding_dim:int   = 32
    vgae_dropout:      float = 0.15

    # --- DQN architecture ---
    dqn_hidden:        int   = 576
    dqn_layers:        int   = 3
    dqn_gamma:         float = 0.99
    dqn_epsilon:       float = 0.1
    dqn_epsilon_decay: float = 0.995
    dqn_min_epsilon:   float = 0.01
    dqn_buffer_size:   int   = 100_000
    dqn_batch_size:    int   = 128
    dqn_target_update: int   = 100

    # --- Knowledge distillation ---
    use_kd:                bool  = False
    teacher_path:          str   = ""
    kd_temperature:        float = 4.0
    kd_alpha:              float = 0.7
    kd_vgae_latent_weight: float = 0.5
    kd_vgae_recon_weight:  float = 0.5
    log_teacher_student_comparison: bool = True

    # --- Curriculum ---
    curriculum_start_ratio:       float = 1.0
    curriculum_end_ratio:         float = 10.0
    difficulty_percentile:        float = 75.0
    use_vgae_mining:              bool  = True
    difficulty_cache_update:      int   = 10
    curriculum_memory_multiplier: float = 1.0

    # --- Fusion ---
    fusion_episodes:        int   = 500
    fusion_max_samples:     int   = 150_000
    max_val_samples:        int   = 30_000
    episode_sample_size:    int   = 20_000
    training_step_interval: int   = 32
    gpu_training_steps:     int   = 16
    fusion_lr:              float = 0.001
    alpha_steps:            int   = 21

    # --- Inference ---
    run_test: bool = True

    # --- Infrastructure ---
    experiment_root:  str = "experimentruns"
    device:           str = "cuda"
    num_workers:      int = 8
    mp_start_method:  str = "spawn"   # multiprocessing start method (spawn required for CUDA + DataLoader workers)

    # ----- serialization -----

    def save(self, path: str | Path) -> None:
        """Write config as JSON. This is the frozen record of the run."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """Reconstruct config from a frozen JSON file."""
        raw = json.loads(Path(path).read_text())
        valid_names = {f.name for f in fields(cls)}
        # JSON turns tuples into lists; convert back
        for f in fields(cls):
            type_str = str(f.type)
            if f.name in raw and ("Tuple" in type_str or "tuple" in type_str):
                raw[f.name] = tuple(raw[f.name])
        # Drop unknown keys, use defaults for missing fields (forward compat)
        filtered = {k: v for k, v in raw.items() if k in valid_names}
        return cls(**filtered)

    # ----- factory -----

    @classmethod
    def from_preset(cls, model: str, model_size: str, **overrides) -> PipelineConfig:
        """Build config from a (model, model_size) preset plus overrides."""
        arch = dict(ARCHITECTURES.get((model, model_size), {}))
        train_base = dict(TRAINING.get(model, {}))
        train_size = dict(TRAINING_SIZE.get((model, model_size), {}))
        defaults = {**arch, **train_base, **train_size, "model_size": model_size}
        defaults.update(overrides)
        return cls(**defaults)

    def with_overrides(self, **kw) -> PipelineConfig:
        """Return a new config with selected fields changed."""
        return replace(self, **kw)

    # ----- typed sub-config views -----

    @property
    def vgae(self) -> VGAEConfig:
        return VGAEConfig(
            hidden_dims=self.vgae_hidden_dims,
            latent_dim=self.vgae_latent_dim,
            heads=self.vgae_heads,
            embedding_dim=self.vgae_embedding_dim,
            dropout=self.vgae_dropout,
        )

    @property
    def gat(self) -> GATConfig:
        return GATConfig(
            hidden=self.gat_hidden,
            layers=self.gat_layers,
            heads=self.gat_heads,
            dropout=self.gat_dropout,
            embedding_dim=self.gat_embedding_dim,
            fc_layers=self.gat_fc_layers,
        )

    @property
    def dqn(self) -> DQNConfig:
        return DQNConfig(
            hidden=self.dqn_hidden,
            layers=self.dqn_layers,
            gamma=self.dqn_gamma,
            epsilon=self.dqn_epsilon,
            epsilon_decay=self.dqn_epsilon_decay,
            min_epsilon=self.dqn_min_epsilon,
            buffer_size=self.dqn_buffer_size,
            batch_size=self.dqn_batch_size,
            target_update=self.dqn_target_update,
        )

    @property
    def kd(self) -> KDConfig:
        return KDConfig(
            enabled=self.use_kd,
            teacher_path=self.teacher_path,
            temperature=self.kd_temperature,
            alpha=self.kd_alpha,
            vgae_latent_weight=self.kd_vgae_latent_weight,
            vgae_recon_weight=self.kd_vgae_recon_weight,
        )

    @property
    def fusion(self) -> FusionConfig:
        return FusionConfig(
            episodes=self.fusion_episodes,
            max_samples=self.fusion_max_samples,
            max_val_samples=self.max_val_samples,
            episode_sample_size=self.episode_sample_size,
            training_step_interval=self.training_step_interval,
            gpu_training_steps=self.gpu_training_steps,
            lr=self.fusion_lr,
            alpha_steps=self.alpha_steps,
        )
