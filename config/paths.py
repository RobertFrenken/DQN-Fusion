"""All paths derived from PipelineConfig. One function, one truth.

Every file location in the entire system comes from stage_dir().
The Snakefile, the CLI, the stages -- they all call these functions.
No second implementation. No disagreement possible.

Path layout: {root}/{dataset}/{model_type}_{scale}_{stage}[_{aux}]

Two interfaces:
  - PipelineConfig-based (stage_dir, checkpoint_path, etc.) -- used by Python stages
  - String-based (_str variants) -- used by Snakefile (wildcard strings, no config object)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import PipelineConfig

EXPERIMENT_ROOT = "experimentruns"

# stage_name -> (learning_type, model_arch, training_mode)
STAGES = {
    "autoencoder": ("unsupervised", "vgae", "autoencoder"),
    "curriculum":  ("supervised",   "gat",  "curriculum"),
    "normal":      ("supervised",   "gat",  "normal"),
    "fusion":      ("rl_fusion",    "dqn",  "fusion"),
    "evaluation":  ("evaluation",   "eval", "evaluation"),
}

from .constants import CATALOG_PATH

_datasets_cache: list[str] | None = None


def get_datasets() -> list[str]:
    """Read dataset names from config/datasets.yaml (cached after first call)."""
    global _datasets_cache
    if _datasets_cache is None:
        import yaml
        with open(CATALOG_PATH) as f:
            catalog = yaml.safe_load(f)
        _datasets_cache = list(catalog.keys())
    return _datasets_cache


# Backwards-compatible module-level name (lazy property via __getattr__)
def __getattr__(name: str):
    if name == "DATASETS":
        return get_datasets()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_id(cfg: PipelineConfig, stage: str) -> str:
    """Deterministic run ID from config and stage.

    Format: {dataset}/{model_type}_{scale}_{stage}[_{aux}]
    Examples:
        - "hcrl_sa/vgae_large_autoencoder"
        - "hcrl_sa/gat_small_curriculum_kd"
        - "set_01/dqn_large_fusion"

    This ID is used for:
    - Filesystem directory names (via stage_dir)
    - MLflow run names (for tracking)
    - Snakemake target paths (deterministic at DAG construction time)
    """
    aux_suffix = f"_{cfg.auxiliaries[0].type}" if cfg.auxiliaries else ""
    return f"{cfg.dataset}/{cfg.model_type}_{cfg.scale}_{stage}{aux_suffix}"


def stage_dir(cfg: PipelineConfig, stage: str) -> Path:
    """Canonical experiment directory.

    Layout: {root}/{dataset}/{model_type}_{scale}_{stage}[_{aux}]
    """
    return Path(cfg.experiment_root) / run_id(cfg, stage)


def checkpoint_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the best model checkpoint is saved."""
    return stage_dir(cfg, stage) / "best_model.pt"


def config_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the frozen config JSON is saved alongside the model."""
    return stage_dir(cfg, stage) / "config.json"


def log_dir(cfg: PipelineConfig, stage: str) -> Path:
    """Lightning / CSV log directory for a stage."""
    return stage_dir(cfg, stage) / "logs"


def data_dir(cfg: PipelineConfig) -> Path:
    """Raw data directory for a dataset."""
    return Path("data") / "automotive" / cfg.dataset


def metrics_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the evaluation metrics JSON is saved."""
    return stage_dir(cfg, stage) / "metrics.json"


def cache_dir(cfg: PipelineConfig) -> Path:
    """Processed-graph cache directory."""
    return Path("data") / "cache" / cfg.dataset


# ---------------------------------------------------------------------------
# String-based path functions (for Snakefile -- no PipelineConfig needed)
# ---------------------------------------------------------------------------

def run_id_str(dataset: str, model_type: str, scale: str, stage: str, aux: str = "") -> str:
    """Deterministic run ID from raw strings (Snakefile companion to run_id)."""
    suffix = f"_{aux}" if aux else ""
    return f"{dataset}/{model_type}_{scale}_{stage}{suffix}"


def checkpoint_path_str(dataset: str, model_type: str, scale: str, stage: str, aux: str = "") -> str:
    """Checkpoint path from raw strings (Snakefile companion to checkpoint_path)."""
    return f"{EXPERIMENT_ROOT}/{run_id_str(dataset, model_type, scale, stage, aux)}/best_model.pt"


def metrics_path_str(dataset: str, model_type: str, scale: str, stage: str, aux: str = "") -> str:
    """Metrics JSON path from raw strings."""
    return f"{EXPERIMENT_ROOT}/{run_id_str(dataset, model_type, scale, stage, aux)}/metrics.json"


def benchmark_path_str(dataset: str, model_type: str, scale: str, stage: str, aux: str = "") -> str:
    """Snakemake benchmark TSV path from raw strings."""
    return f"{EXPERIMENT_ROOT}/{run_id_str(dataset, model_type, scale, stage, aux)}/benchmark.tsv"


def log_path_str(dataset: str, model_type: str, scale: str, stage: str, aux: str = "", stream: str = "out") -> str:
    """SLURM log path from raw strings."""
    return f"{EXPERIMENT_ROOT}/{run_id_str(dataset, model_type, scale, stage, aux)}/slurm.{stream}"
