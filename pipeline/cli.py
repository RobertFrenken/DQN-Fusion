"""Single CLI entry point.

Usage:
    python -m pipeline.cli autoencoder --model vgae --scale large --dataset hcrl_ch
    python -m pipeline.cli curriculum  --model gat --scale small --auxiliaries kd_standard --dataset hcrl_sa
    python -m pipeline.cli fusion      --config path/to/config.json
"""
from __future__ import annotations

import torch.multiprocessing as mp
# Must be called before any CUDA or multiprocessing usage.
# Prevents "Cannot re-initialize CUDA in forked subprocess" errors
# when DataLoader workers collate tensors after CUDA has been initialized
# in the main process (e.g. by _score_difficulty in the curriculum stage).
mp.set_start_method('spawn', force=True)

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from config import PipelineConfig, STAGES, config_path, run_id, stage_dir
from config.resolver import resolve
from .validate import validate

_ON_COMPUTE_NODE = bool(os.environ.get("SLURM_JOB_ID"))


def _parse_bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="KD-GAT training pipeline",
    )
    p.add_argument(
        "stage", choices=list(STAGES.keys()) + ["flow"],
        help="Training stage to run, or 'flow' to run full pipeline via Ray",
    )

    # Config source
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--config", type=Path, default=None,
        help="Load a frozen config JSON (e.g. from a previous run)",
    )

    # Identity flags (used by resolver)
    p.add_argument("--model", type=str, default="vgae", help="Model type: vgae, gat, dqn")
    p.add_argument("--scale", type=str, default="large", help="Model scale: large, small")
    p.add_argument("--auxiliaries", type=str, default="none", help="Auxiliary config: none, kd_standard")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    # Infrastructure overrides
    p.add_argument("--experiment-root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--mp-start-method", type=str, default=None)
    p.add_argument("--run-test", type=_parse_bool, default=None)

    # KD shorthand: --teacher-path sets auxiliaries + model_path
    p.add_argument("--teacher-path", type=str, default=None,
                    help="Shorthand: implies kd_standard aux with given model_path")

    # Flow subcommand options
    p.add_argument("--eval-only", action="store_true", default=False,
                    help="(flow) Re-run evaluation only, skip training")
    p.add_argument("--local", action="store_true", default=False,
                    help="(flow) Use Ray local mode instead of cluster")

    # Nested overrides via dot-path: --training.lr 0.001, --vgae.latent-dim 16
    p.add_argument("--override", "-O", nargs=2, action="append", default=[],
                    metavar=("KEY", "VALUE"),
                    help="Nested override as 'section.field value' (e.g. -O training.lr 0.001)")

    return p


def _parse_dot_overrides(pairs: list[list[str]]) -> dict:
    """Parse -O key value pairs into a nested dict."""
    result: dict = {}
    for key, value in pairs:
        parts = key.replace("-", "_").split(".")
        # Auto-coerce types
        try:
            typed_value: object = int(value)
        except ValueError:
            try:
                typed_value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    typed_value = value.lower() == "true"
                else:
                    typed_value = value

        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = typed_value
    return result


def _run_flow(args: argparse.Namespace, log: logging.Logger) -> None:
    """Dispatch pipeline flow via Ray.

    --scale filters to a single variant (large, small_kd, small_nokd).
    Without --scale, all variants run.  The argparse default "large" is
    for single-stage dispatch; for flows, we treat it as "run all" unless
    the user explicitly passes a flow-relevant scale value.
    """
    datasets = [args.dataset] if args.dataset else None

    # Detect if --scale was explicitly provided on the CLI
    # (argparse default is "large", which we ignore for flow mode)
    _flow_scales = ("large", "small_kd", "small_nokd")
    scale = args.scale if args.scale in _flow_scales else None
    # Check if user actually passed --scale or if it's the default
    import sys
    if "--scale" not in sys.argv:
        scale = None

    from .orchestration.ray_pipeline import train_pipeline, eval_pipeline

    if args.eval_only:
        log.info("Starting Ray evaluation flow (datasets=%s, scale=%s)", datasets, scale)
        eval_pipeline(datasets=datasets, scale=scale, local=args.local)
    else:
        log.info("Starting Ray training flow (datasets=%s, scale=%s)", datasets, scale)
        train_pipeline(datasets=datasets, scale=scale, local=args.local)


def _init_wandb(cfg: PipelineConfig, stage: str, run_name: str):
    """Initialize a W&B run. Returns the run object, or None on failure."""
    try:
        import wandb
    except ImportError:
        return None

    # Offline mode on SLURM compute nodes (no internet); sync later via onsuccess
    if _ON_COMPUTE_NODE and not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = "offline"

    try:
        return wandb.init(
            project="kd-gat",
            name=run_name,
            config=cfg.model_dump(),
            tags=[cfg.dataset, cfg.model_type, cfg.scale, stage],
            reinit=True,
        )
    except Exception as e:
        logging.getLogger("pipeline").warning("wandb.init() failed: %s", e)
        return None


def _wandb_log_metrics(result: dict) -> None:
    """Log final result metrics to the active W&B run."""
    try:
        import wandb
        if wandb.run is None:
            return
        flat: dict[str, float] = {}
        for model_key, model_metrics in result.items():
            if model_key == "test":
                continue  # test metrics are nested differently
            if isinstance(model_metrics, dict) and "core" in model_metrics:
                for k, v in model_metrics["core"].items():
                    if isinstance(v, (int, float)):
                        flat[f"{model_key}/{k}"] = v
        if flat:
            wandb.log(flat)
    except Exception:
        pass


def _sync_lakehouse(
    cfg: PipelineConfig, stage: str, run_name: str,
    result: object = None, success: bool = True,
    failure_reason: str | None = None,
) -> None:
    """Fire-and-forget sync to datalake (Parquet)."""
    try:
        from .lakehouse import sync_to_lakehouse
        sync_to_lakehouse(
            run_id=run_name,
            dataset=cfg.dataset,
            model_type=cfg.model_type,
            scale=cfg.scale,
            stage=stage,
            has_kd=cfg.has_kd,
            metrics=result if isinstance(result, dict) else None,
            success=success,
            failure_reason=failure_reason,
        )
    except Exception as e:
        logging.getLogger("pipeline").debug("Lakehouse sync skipped: %s", e)


def _finish_wandb() -> None:
    """Finish the active W&B run if one exists."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )
    log = logging.getLogger("pipeline")

    # ---- Handle non-training subcommands ----
    if args.stage == "flow":
        _run_flow(args, log)
        return

    # ---- Build config ----
    if args.config:
        cfg = PipelineConfig.load(args.config)
        log.info("Loaded frozen config: %s", args.config)
    else:
        # Build overrides dict
        _OVERRIDE_FIELDS = ("dataset", "seed", "experiment_root", "device",
                            "num_workers", "mp_start_method", "run_test")
        overrides = {f: getattr(args, f) for f in _OVERRIDE_FIELDS if getattr(args, f) is not None}

        # Handle --teacher-path shorthand
        aux_name = args.auxiliaries
        if args.teacher_path:
            if aux_name == "none":
                aux_name = "kd_standard"
            overrides.setdefault("auxiliaries", [{"type": "kd", "model_path": args.teacher_path}])

        # Parse dot-path overrides
        dot_overrides = _parse_dot_overrides(args.override)
        if dot_overrides:
            from config.resolver import _deep_merge
            _deep_merge(overrides, dot_overrides)

        cfg = resolve(args.model, args.scale, auxiliaries=aux_name, **overrides)
        log.info("Resolved config: model=%s, scale=%s, aux=%s", args.model, args.scale, aux_name)

    # ---- Validate ----
    validate(cfg, args.stage)

    # ---- Archive completed run if re-running same config ----
    sdir = stage_dir(cfg, args.stage)
    archive = None
    if (sdir / "metrics.json").exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = sdir.parent / f"{sdir.name}.archive_{ts}"
        sdir.rename(archive)
        log.warning("Archived completed run → %s", archive)

    # ---- Save frozen config ----
    cfg_out = config_path(cfg, args.stage)
    cfg.save(cfg_out)
    log.info("Frozen config: %s", cfg_out)

    # ---- Run ID ----
    run_name = run_id(cfg, args.stage)
    log.info("Run started: %s", run_name)

    # ---- W&B init ----
    _wandb_run = _init_wandb(cfg, args.stage, run_name)

    # ---- Dispatch ----
    try:
        from .stages import STAGE_FNS
        result = STAGE_FNS[args.stage](cfg)
        log.info("Stage '%s' complete. Result: %s", args.stage, result)

        # Log final metrics to W&B
        if _wandb_run is not None and isinstance(result, dict):
            _wandb_log_metrics(result)

        # Sync to datalake (fire-and-forget)
        _sync_lakehouse(cfg, args.stage, run_name, result)

        # Register artifacts in datalake (fire-and-forget)
        try:
            from .lakehouse import register_artifacts
            register_artifacts(run_name, sdir)
        except Exception as e:
            log.debug("Artifact registration skipped: %s", e)

        # Success → delete archive
        if archive and archive.exists():
            import shutil
            shutil.rmtree(archive, ignore_errors=True)

        log.info("Run completed successfully")

    except Exception as e:
        # Failure → restore archive
        if archive and archive.exists():
            if sdir.exists():
                import shutil
                shutil.rmtree(sdir, ignore_errors=True)
            archive.rename(sdir)
            log.warning("Restored archive after failure: %s", sdir)
        _sync_lakehouse(cfg, args.stage, run_name, None, success=False, failure_reason=str(e))
        log.error("Run failed: %s", str(e))
        raise

    finally:
        _finish_wandb()


if __name__ == "__main__":
    main()
