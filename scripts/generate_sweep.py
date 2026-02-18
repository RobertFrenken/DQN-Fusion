#!/usr/bin/env python
"""Generate cartesian product of CLI commands for hyperparameter sweeps.

Usage:
    python scripts/generate_sweep.py \
        --stage autoencoder --model vgae --scale large --dataset hcrl_sa \
        --sweep "training.lr=0.001,0.0005" "vgae.latent_dim=8,16,32" \
        --output /tmp/sweep_commands.txt

Generates one `python -m pipeline.cli ...` command per parameter combination,
suitable for parallel-command-processor or xargs.
"""
from __future__ import annotations

import argparse
import itertools
import sys


def parse_sweep_spec(spec: str) -> tuple[str, list[str]]:
    """Parse 'key=val1,val2,val3' into (key, [val1, val2, val3])."""
    key, _, values = spec.partition("=")
    if not key or not values:
        raise ValueError(f"Invalid sweep spec: {spec!r}. Expected 'key=val1,val2,...'")
    return key, values.split(",")


def main():
    p = argparse.ArgumentParser(description="Generate sweep commands")
    p.add_argument("--stage", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--scale", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--auxiliaries", default="none")
    p.add_argument("--teacher-path", default=None)
    p.add_argument("--sweep", nargs="+", required=True,
                    help="Sweep specs: 'key=val1,val2' (cartesian product)")
    p.add_argument("--output", default=None,
                    help="Output file (default: stdout)")
    args = p.parse_args()

    # Parse sweep specs
    keys = []
    value_lists = []
    for spec in args.sweep:
        key, vals = parse_sweep_spec(spec)
        keys.append(key)
        value_lists.append(vals)

    # Base command
    base = [
        sys.executable, "-m", "pipeline.cli", args.stage,
        "--model", args.model,
        "--scale", args.scale,
        "--dataset", args.dataset,
        "--auxiliaries", args.auxiliaries,
    ]
    if args.teacher_path:
        base.extend(["--teacher-path", args.teacher_path])

    # Generate cartesian product
    lines = []
    for combo in itertools.product(*value_lists):
        cmd = list(base)
        for key, val in zip(keys, combo):
            cmd.extend(["-O", key, val])
        lines.append(" ".join(cmd))

    output = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Generated {len(lines)} commands → {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)
        print(f"# {len(lines)} commands generated", file=sys.stderr)


if __name__ == "__main__":
    main()
