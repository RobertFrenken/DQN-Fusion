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
import json
import logging
from datetime import datetime
from pathlib import Path

from config import PipelineConfig, STAGES, config_path, run_id, stage_dir
from config.resolver import resolve
from .validate import validate
from .db import record_run_start, record_run_end, get_connection


def _parse_bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="KD-GAT training pipeline",
    )
    p.add_argument(
        "stage", choices=list(STAGES.keys()) + ["state"],
        help="Training stage to run (or 'state' to update STATE.md)",
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


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )
    log = logging.getLogger("pipeline")

    # ---- Handle non-training subcommands ----
    if args.stage == "state":
        from .state_sync import update_state
        update_state(preview=False)
        log.info("STATE.md updated")
        return

    # ---- Build config ----
    if args.config:
        cfg = PipelineConfig.load(args.config)
        log.info("Loaded frozen config: %s", args.config)
    else:
        # Build overrides dict
        overrides: dict = {}
        if args.dataset:
            overrides["dataset"] = args.dataset
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.experiment_root:
            overrides["experiment_root"] = args.experiment_root
        if args.device:
            overrides["device"] = args.device
        if args.num_workers is not None:
            overrides["num_workers"] = args.num_workers
        if args.mp_start_method:
            overrides["mp_start_method"] = args.mp_start_method
        if args.run_test is not None:
            overrides["run_test"] = args.run_test

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
    if (sdir / "metrics.json").exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = sdir.parent / f"{sdir.name}.archive_{ts}"
        sdir.rename(archive)
        log.warning("Archived completed run → %s", archive)

    # ---- Save frozen config ----
    cfg_out = config_path(cfg, args.stage)
    cfg.save(cfg_out)
    log.info("Frozen config: %s", cfg_out)

    # ---- Record run start in project DB ----
    run_name = run_id(cfg, args.stage)

    # Propagate teacher_run for KD evaluation runs from their training counterpart
    teacher_run = ""
    if cfg.has_kd and args.stage == "evaluation":
        conn = get_connection()
        try:
            row = conn.execute(
                """SELECT teacher_run FROM runs
                   WHERE dataset = ? AND has_kd = 1 AND teacher_run != ''
                     AND stage IN ('curriculum', 'fusion')
                   LIMIT 1""",
                (cfg.dataset,),
            ).fetchone()
            if row and row[0]:
                teacher_run = row[0]
                log.info("Propagated teacher_run=%s for eval KD run", teacher_run)
        finally:
            conn.close()

    record_run_start(
        run_id=run_name, dataset=cfg.dataset, model_type=cfg.model_type,
        scale=cfg.scale, stage=args.stage, has_kd=cfg.has_kd,
        config_json=cfg.model_dump_json(indent=2),
        teacher_run=teacher_run,
    )
    log.info("Run started: %s", run_name)

    # ---- Dispatch ----
    try:
        from .stages import STAGE_FNS
        result = STAGE_FNS[args.stage](cfg)
        log.info("Stage '%s' complete. Result: %s", args.stage, result)

        record_run_end(run_name, success=True,
                       metrics=result if isinstance(result, dict) else None)
        log.info("Run completed successfully")

    except Exception as e:
        record_run_end(run_name, success=False)
        log.error("Run failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
