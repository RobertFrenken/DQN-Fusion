"""Ray Tune HPO configuration for KD-GAT stages.

Replaces scripts/generate_sweep.py parallel-command approach with
Ray Tune + OptunaSearch + ASHAScheduler for efficient hyperparameter search.

Usage:
    from graphids.pipeline.orchestration.tune_config import run_tune
    run_tune("autoencoder", dataset="hcrl_sa", num_samples=20)
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search spaces per model (declarative: type + args → Ray Tune sampler)
# ---------------------------------------------------------------------------

_SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "vgae": {
        "training.lr": ("loguniform", 1e-4, 1e-2),
        "training.weight_decay": ("loguniform", 1e-6, 1e-3),
        "vgae.latent_dim": ("choice", [16, 32, 48, 64]),
        "vgae.dropout": ("uniform", 0.05, 0.4),
        "vgae.heads": ("choice", [1, 2, 4, 8]),
        "vgae.embedding_dim": ("choice", [8, 16, 32]),
    },
    "gat": {
        "training.lr": ("loguniform", 1e-4, 1e-2),
        "training.weight_decay": ("loguniform", 1e-6, 1e-3),
        "gat.hidden": ("choice", [32, 48, 64, 96]),
        "gat.layers": ("choice", [2, 3, 4]),
        "gat.heads": ("choice", [4, 8]),
        "gat.dropout": ("uniform", 0.1, 0.4),
        "gat.embedding_dim": ("choice", [8, 16, 32]),
        "gat.fc_layers": ("choice", [2, 3, 4]),
    },
    "dqn": {
        "fusion.lr": ("loguniform", 1e-4, 1e-2),
        "dqn.hidden": ("choice", [256, 512, 576, 768]),
        "dqn.layers": ("choice", [2, 3, 4]),
        "dqn.gamma": ("uniform", 0.95, 0.999),
        "dqn.epsilon": ("uniform", 0.05, 0.2),
        "dqn.epsilon_decay": ("uniform", 0.99, 0.999),
        "fusion.episodes": ("choice", [300, 500, 750]),
    },
}

_STAGE_MODEL = {
    "autoencoder": "vgae",
    "curriculum": "gat",
    "normal": "gat",
    "fusion": "dqn",
}


def _build_search_space(stage: str) -> dict[str, Any]:
    """Build Ray Tune search space from declarative spec."""
    from ray import tune

    _BUILDERS = {"loguniform": tune.loguniform, "choice": tune.choice, "uniform": tune.uniform}
    return {k: _BUILDERS[t](*args) for k, (t, *args) in _SEARCH_SPACES[_STAGE_MODEL[stage]].items()}


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
        sys.executable,
        "-m",
        "graphids.pipeline.cli",
        stage,
        "--model",
        model,
        "--scale",
        scale,
        "--dataset",
        dataset,
    ]
    for key, value in config.items():
        cmd.extend(["-O", key, str(value)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.warning("Trial failed: %s", result.stderr[-500:] if result.stderr else "unknown")
        ray_train.report({"val_loss": float("inf")})
        return

    # Read metrics from the stage output
    from graphids.config import metrics_path, stage_dir
    from graphids.config.resolver import resolve

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

    if stage not in _STAGE_MODEL:
        raise ValueError(f"No search space defined for stage '{stage}'")

    if not ray.is_initialized():
        kwargs = ray_init_kwargs()
        if local:
            kwargs["num_gpus"] = 0
        ray.init(**kwargs)

    search_space = _build_search_space(stage)

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

        callbacks.append(
            WandbLoggerCallback(
                project="kd-gat-tune",
                group=f"{stage}_{dataset}_{scale}",
            )
        )
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
    log.info(
        "Best config for %s: %s (val_loss=%.6f)",
        stage,
        best.config,
        best.metrics.get(metric, float("inf")),
    )

    return results
