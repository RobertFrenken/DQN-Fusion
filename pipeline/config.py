"""Single source of truth for all pipeline parameters.

One frozen dataclass. One dict of presets. JSON save/load.
No Hydra, no Pydantic, no duplicate DEFAULT_MODEL_ARGS.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, asdict, replace
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Preset defaults for (model_type, model_size) combinations.
# Any field can be overridden at construction time.
# ---------------------------------------------------------------------------

PRESETS: dict[tuple[str, str], dict] = {
    # ---- VGAE (autoencoder) ----
    ("vgae", "teacher"): dict(
        lr=0.002, max_epochs=300, patience=100,
        vgae_hidden_dims=(1024, 512, 96), vgae_latent_dim=96,
        vgae_heads=4, vgae_embedding_dim=64, vgae_dropout=0.15,
        safety_factor=0.55,
    ),
    ("vgae", "student"): dict(
        lr=0.002, max_epochs=300, patience=100,
        vgae_hidden_dims=(80, 40, 16), vgae_latent_dim=16,
        vgae_heads=1, vgae_embedding_dim=4, vgae_dropout=0.1,
        use_kd=True,
        safety_factor=0.6,
    ),
    # ---- GAT (curriculum) ----
    ("gat", "teacher"): dict(
        lr=0.003, max_epochs=300, patience=100,
        gat_hidden=64, gat_layers=5, gat_heads=8,
        gat_dropout=0.2, gat_embedding_dim=32,
        safety_factor=0.5,
    ),
    ("gat", "student"): dict(
        lr=0.001, max_epochs=300, patience=50,
        gat_hidden=24, gat_layers=2, gat_heads=4,
        gat_dropout=0.1, gat_embedding_dim=8,
        use_kd=True,
        safety_factor=0.55,
    ),
    # ---- DQN (fusion) ----
    ("dqn", "teacher"): dict(
        dqn_hidden=576, dqn_layers=3,
        dqn_gamma=0.99, dqn_buffer_size=100_000,
        dqn_batch_size=128, dqn_target_update=100,
        fusion_episodes=500,
        safety_factor=0.45,
    ),
    ("dqn", "student"): dict(
        dqn_hidden=160, dqn_layers=2,
        dqn_gamma=0.99, dqn_buffer_size=50_000,
        dqn_batch_size=64, dqn_target_update=50,
        fusion_episodes=500,
        safety_factor=0.5,
    ),
}


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
    batch_size_mode:       str   = "binsearch"
    max_batch_size_trials: int   = 10

    # --- Training dynamics ---
    accumulate_grad_batches: int  = 1
    find_unused_parameters:  bool = False

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
    gat_hidden:        int   = 64
    gat_layers:        int   = 5
    gat_heads:         int   = 8
    gat_dropout:       float = 0.2
    gat_embedding_dim: int   = 32

    # --- VGAE architecture ---
    vgae_hidden_dims:  Tuple[int, ...] = (1024, 512, 96)
    vgae_latent_dim:   int   = 96
    vgae_heads:        int   = 4
    vgae_embedding_dim:int   = 64
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
        # JSON turns tuples into lists; convert back
        for f in fields(cls):
            if f.name in raw and "Tuple" in str(f.type):
                raw[f.name] = tuple(raw[f.name])
        return cls(**raw)

    # ----- factory -----

    @classmethod
    def from_preset(cls, model: str, model_size: str, **overrides) -> PipelineConfig:
        """Build config from a (model, model_size) preset plus overrides."""
        defaults = dict(PRESETS.get((model, model_size), {}))
        defaults["model_size"] = model_size
        defaults.update(overrides)
        return cls(**defaults)

    def with_overrides(self, **kw) -> PipelineConfig:
        """Return a new config with selected fields changed."""
        return replace(self, **kw)
