"""Single CLI entry point. Replaces train_with_hydra_zen.py + src/cli/main.py.

Usage:
    python -m pipeline.cli autoencoder --dataset hcrl_ch --model-size teacher
    python -m pipeline.cli curriculum  --config path/to/config.json
    python -m pipeline.cli fusion      --preset dqn,teacher --dataset hcrl_ch
"""
from __future__ import annotations

import torch.multiprocessing as mp
# Must be called before any CUDA or multiprocessing usage.
# Prevents "Cannot re-initialize CUDA in forked subprocess" errors
# when DataLoader workers collate tensors after CUDA has been initialized
# in the main process (e.g. by _score_difficulty in the curriculum stage).
# Default matches PipelineConfig.mp_start_method; override via --mp-start-method.
mp.set_start_method('spawn', force=True)

import argparse
import logging
from dataclasses import fields as dc_fields, replace
from pathlib import Path

from .config import PipelineConfig, PRESETS
from .paths import STAGES, config_path, run_id
from .validate import validate
from .tracking import start_run, end_run, log_failure, log_run_artifacts


def _parse_bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="KD-GAT training pipeline",
    )
    p.add_argument(
        "stage", choices=list(STAGES.keys()),
        help="Training stage to run",
    )

    # Config source
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--config", type=Path, default=None,
        help="Load a frozen config JSON (e.g. from a previous run)",
    )
    src.add_argument(
        "--preset", type=str, default=None,
        help="Preset key as 'model,size' (e.g. gat,teacher)",
    )

    # Every PipelineConfig field becomes an optional override
    for f in dc_fields(PipelineConfig):
        flag = f"--{f.name.replace('_', '-')}"
        default_val = f.default

        if isinstance(default_val, bool):
            p.add_argument(flag, type=_parse_bool, default=None, metavar="BOOL")
        elif isinstance(default_val, tuple):
            p.add_argument(flag, type=int, nargs="+", default=None)
        elif isinstance(default_val, int):
            p.add_argument(flag, type=int, default=None)
        elif isinstance(default_val, float):
            p.add_argument(flag, type=float, default=None)
        else:
            p.add_argument(flag, type=str, default=None)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )
    log = logging.getLogger("pipeline")

    # ---- Build config ----
    if args.config:
        cfg = PipelineConfig.load(args.config)
        log.info("Loaded frozen config: %s", args.config)
    elif args.preset:
        model, size = [s.strip() for s in args.preset.split(",")]
        cfg = PipelineConfig.from_preset(model, size)
        log.info("Using preset: %s, %s", model, size)
    else:
        cfg = PipelineConfig()

    # ---- Apply CLI overrides ----
    overrides: dict = {}
    for f in dc_fields(PipelineConfig):
        # argparse stores with underscores
        val = getattr(args, f.name, None)
        if val is not None:
            overrides[f.name] = tuple(val) if isinstance(val, list) else val

    if overrides:
        cfg = replace(cfg, **overrides)
        log.info("Applied %d CLI overrides: %s", len(overrides), list(overrides.keys()))

    # ---- Validate ----
    validate(cfg, args.stage)

    # ---- Save frozen config ----
    cfg_out = config_path(cfg, args.stage)
    cfg.save(cfg_out)
    log.info("Frozen config: %s", cfg_out)

    # ---- Start MLflow tracking ----
    run_name = run_id(cfg, args.stage)
    start_run(cfg, args.stage, run_name)
    log.info("MLflow run started: %s", run_name)

    # ---- Dispatch ----
    try:
        from .stages import STAGE_FNS
        result = STAGE_FNS[args.stage](cfg)
        log.info("Stage '%s' complete. Result: %s", args.stage, result)

        # ---- Log artifacts and results to MLflow ----
        from .paths import stage_dir
        log_run_artifacts(stage_dir(cfg, args.stage))
        end_run(result if isinstance(result, dict) else None, success=True)
        log.info("MLflow run completed successfully")

    except Exception as e:
        # ---- Log failure to MLflow ----
        log_failure(str(e))
        log.error("MLflow run failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
