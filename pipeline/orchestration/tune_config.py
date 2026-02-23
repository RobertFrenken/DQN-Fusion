"""Ray Tune HPO configuration for KD-GAT stages.

Replaces scripts/generate_sweep.py parallel-command approach with
Ray Tune + OptunaSearch + ASHAScheduler for efficient hyperparameter search.

Usage:
    from pipeline.orchestration.tune_config import run_tune
    run_tune("autoencoder", dataset="hcrl_sa", num_samples=20)
"""
from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search spaces per stage
# ---------------------------------------------------------------------------

def vgae_search_space() -> dict[str, Any]:
    """VGAE autoencoder hyperparameter search space."""
    from ray import tune

    return {
        "training.lr": tune.loguniform(1e-4, 1e-2),
        "training.weight_decay": tune.loguniform(1e-6, 1e-3),
        "vgae.latent_dim": tune.choice([16, 32, 48, 64]),
        "vgae.dropout": tune.uniform(0.05, 0.4),
        "vgae.heads": tune.choice([1, 2, 4, 8]),
        "vgae.embedding_dim": tune.choice([8, 16, 32]),
    }


def gat_search_space() -> dict[str, Any]:
    """GAT classifier hyperparameter search space."""
    from ray import tune

    return {
        "training.lr": tune.loguniform(1e-4, 1e-2),
        "training.weight_decay": tune.loguniform(1e-6, 1e-3),
        "gat.hidden": tune.choice([32, 48, 64, 96]),
        "gat.layers": tune.choice([2, 3, 4]),
        "gat.heads": tune.choice([4, 8]),
        "gat.dropout": tune.uniform(0.1, 0.4),
        "gat.embedding_dim": tune.choice([8, 16, 32]),
        "gat.fc_layers": tune.choice([2, 3, 4]),
    }


def dqn_search_space() -> dict[str, Any]:
    """DQN fusion agent hyperparameter search space."""
    from ray import tune

    return {
        "fusion.lr": tune.loguniform(1e-4, 1e-2),
        "dqn.hidden": tune.choice([256, 512, 576, 768]),
        "dqn.layers": tune.choice([2, 3, 4]),
        "dqn.gamma": tune.uniform(0.95, 0.999),
        "dqn.epsilon": tune.uniform(0.05, 0.2),
        "dqn.epsilon_decay": tune.uniform(0.99, 0.999),
        "fusion.episodes": tune.choice([300, 500, 750]),
    }


_STAGE_SEARCH_SPACES = {
    "autoencoder": vgae_search_space,
    "curriculum": gat_search_space,
    "normal": gat_search_space,
    "fusion": dqn_search_space,
}

_STAGE_MODEL = {
    "autoencoder": "vgae",
    "curriculum": "gat",
    "normal": "gat",
    "fusion": "dqn",
}


# ---------------------------------------------------------------------------
# Trainable function (subprocess-based, like the pipeline)
# ---------------------------------------------------------------------------

def _trainable(config: dict, stage: str, dataset: str, scale: str) -> None:
    """Ray Tune trainable that runs a pipeline stage as subprocess.

    Reports val_loss from the stage's metrics.json.
    """
    import json
    from pathlib import Path

    from ray import train as ray_train

    model = _STAGE_MODEL[stage]

    # Build CLI overrides from tune config
    cmd = [
        sys.executable, "-m", "pipeline.cli", stage,
        "--model", model,
        "--scale", scale,
        "--dataset", dataset,
    ]
    for key, value in config.items():
        cmd.extend(["-O", key, str(value)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.warning("Trial failed: %s", result.stderr[-500:] if result.stderr else "unknown")
        ray_train.report({"val_loss": float("inf")})
        return

    # Read metrics from the stage output
    from config.resolver import resolve
    from config import metrics_path, stage_dir

    overrides = {"dataset": dataset}
    cfg = resolve(model, scale, **overrides)
    mpath = stage_dir(cfg, stage) / "metrics.json"

    if mpath.exists():
        metrics = json.loads(mpath.read_text())
        val_loss = metrics.get("val_loss", metrics.get("best_val_loss", float("inf")))
        ray_train.report({"val_loss": val_loss})
    else:
        ray_train.report({"val_loss": float("inf")})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tune(
    stage: str,
    dataset: str = "hcrl_sa",
    scale: str = "large",
    num_samples: int = 20,
    max_concurrent: int = 1,
    metric: str = "val_loss",
    mode: str = "min",
    grace_period: int = 10,
    local: bool = False,
) -> Any:
    """Run Ray Tune HPO for a pipeline stage.

    Parameters
    ----------
    stage : str
        Pipeline stage (autoencoder, curriculum, normal, fusion).
    dataset : str
        Dataset name.
    scale : str
        Model scale (large, small).
    num_samples : int
        Number of HPO trials.
    max_concurrent : int
        Max concurrent trials (limited by GPU count).
    metric : str
        Metric to optimize.
    mode : str
        "min" or "max".
    grace_period : int
        ASHA grace period (epochs before early stopping a trial).
    local : bool
        Use Ray local mode.

    Returns
    -------
    ray.tune.ResultGrid
        Tune results with best config accessible via result.get_best_result().
    """
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    from .ray_slurm import ray_init_kwargs

    if stage not in _STAGE_SEARCH_SPACES:
        raise ValueError(f"No search space defined for stage '{stage}'")

    if not ray.is_initialized():
        kwargs = ray_init_kwargs()
        if local:
            kwargs["num_gpus"] = 0
        ray.init(**kwargs)

    search_space = _STAGE_SEARCH_SPACES[stage]()

    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        grace_period=grace_period,
        reduction_factor=3,
    )

    search_alg = OptunaSearch(metric=metric, mode=mode)

    # WandbLoggerCallback if wandb is available
    callbacks = []
    try:
        from ray.tune.logger import TBXLoggerCallback
        callbacks.append(TBXLoggerCallback())
    except ImportError:
        pass

    try:
        from ray.air.integrations.wandb import WandbLoggerCallback
        callbacks.append(WandbLoggerCallback(
            project="kd-gat-tune",
            group=f"{stage}_{dataset}_{scale}",
        ))
    except ImportError:
        pass

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                _trainable,
                stage=stage,
                dataset=dataset,
                scale=scale,
            ),
            resources={"gpu": 1},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=ray.train.RunConfig(
            name=f"tune_{stage}_{dataset}_{scale}",
            callbacks=callbacks,
        ),
    )

    results = tuner.fit()

    best = results.get_best_result(metric=metric, mode=mode)
    log.info("Best config for %s: %s (val_loss=%.6f)",
             stage, best.config, best.metrics.get(metric, float("inf")))

    return results
