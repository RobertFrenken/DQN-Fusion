#!/usr/bin/env python3
"""Aggregate evaluation metrics from all 18 experiment runs.

Reads experimentruns/*/teacher_evaluation/metrics.json, etc.
Produces a markdown table and results/kd_comparison.json.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py --exp-root experimentruns
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = ["hcrl_ch", "hcrl_sa", "set_01", "set_02", "set_03", "set_04"]
VARIANTS = ["teacher", "student_kd", "student_nokd"]
VARIANT_DIRS = {
    "teacher":      "teacher_evaluation",
    "student_kd":   "student_evaluation_kd",
    "student_nokd": "student_evaluation",
}
CORE_METRICS = ["accuracy", "f1", "precision", "recall", "auc", "mcc"]


def load_metrics(exp_root: Path) -> dict:
    """Load all metrics.json files into a nested dict."""
    results = {}
    for ds in DATASETS:
        results[ds] = {}
        for variant, dirname in VARIANT_DIRS.items():
            path = exp_root / ds / dirname / "metrics.json"
            if path.exists():
                data = json.loads(path.read_text())
                results[ds][variant] = data
            else:
                results[ds][variant] = None
    return results


def extract_core(metrics: dict | None, model: str) -> dict:
    """Extract core metrics for a specific model (gat/vgae/fusion)."""
    if metrics is None or model not in metrics:
        return {}
    return metrics[model].get("core", {})


def build_table(results: dict) -> str:
    """Build a markdown comparison table."""
    lines = []

    for model in ["gat", "vgae", "fusion"]:
        lines.append(f"\n## {model.upper()} Results\n")
        header = "| Dataset | Variant | " + " | ".join(CORE_METRICS) + " |"
        sep = "|---------|---------|" + "|".join(["--------"] * len(CORE_METRICS)) + "|"
        lines.append(header)
        lines.append(sep)

        for ds in DATASETS:
            for variant in VARIANTS:
                core = extract_core(results[ds].get(variant), model)
                if not core:
                    vals = " | ".join(["---"] * len(CORE_METRICS))
                else:
                    vals = " | ".join(
                        f"{core.get(m, 0):.4f}" if isinstance(core.get(m), (int, float)) else "---"
                        for m in CORE_METRICS
                    )
                lines.append(f"| {ds} | {variant} | {vals} |")

    # KD lift table
    lines.append("\n## KD Lift (student_kd - student_nokd)\n")
    for model in ["gat", "vgae", "fusion"]:
        lines.append(f"\n### {model.upper()}\n")
        header = "| Dataset | " + " | ".join(CORE_METRICS) + " |"
        sep = "|---------|" + "|".join(["--------"] * len(CORE_METRICS)) + "|"
        lines.append(header)
        lines.append(sep)

        for ds in DATASETS:
            kd = extract_core(results[ds].get("student_kd"), model)
            nokd = extract_core(results[ds].get("student_nokd"), model)
            if kd and nokd:
                vals = []
                for m in CORE_METRICS:
                    kv = kd.get(m)
                    nv = nokd.get(m)
                    if isinstance(kv, (int, float)) and isinstance(nv, (int, float)):
                        diff = kv - nv
                        sign = "+" if diff >= 0 else ""
                        vals.append(f"{sign}{diff:.4f}")
                    else:
                        vals.append("---")
                lines.append(f"| {ds} | {' | '.join(vals)} |")
            else:
                vals = " | ".join(["---"] * len(CORE_METRICS))
                lines.append(f"| {ds} | {vals} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare KD-GAT evaluation results")
    parser.add_argument("--exp-root", default="experimentruns", help="Experiment root dir")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    results = load_metrics(exp_root)

    # Count how many evaluations completed
    total = sum(1 for ds in results for v in results[ds] if results[ds][v] is not None)
    print(f"Found {total}/18 completed evaluations\n")

    if total == 0:
        print("No evaluation results found. Run the pipeline first.")
        return

    table = build_table(results)
    print(table)

    # Save JSON
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    flat = {}
    for ds in DATASETS:
        for variant in VARIANTS:
            for model in ["gat", "vgae", "fusion"]:
                core = extract_core(results[ds].get(variant), model)
                if core:
                    flat[f"{ds}/{variant}/{model}"] = core

    (out_dir / "kd_comparison.json").write_text(json.dumps(flat, indent=2))
    print(f"\nResults saved to {out_dir / 'kd_comparison.json'}")


if __name__ == "__main__":
    main()
