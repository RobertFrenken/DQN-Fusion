#!/usr/bin/env python3
"""Plot per-component and per-layer parameter breakdowns for candidate models.

Inputs: CSV output from `scripts/hyperparam_search.py` or a single candidate JSON.
Generates PNGs (one per candidate) showing bar charts of per-component and per-layer counts.

Usage:
  scripts/plot_candidate_breakdown.py --csv /tmp/results.csv --top 5 --outdir plots
  scripts/plot_candidate_breakdown.py --json experiment_results/budget_proposal_M3_k20.json --which student_candidates --outdir plots
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ensure project importability
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def extract_parts(candidate_row):
    # candidate_row is a dict-like with keys 'part_*' and 'total_params'
    parts = {k.replace('part_', ''): int(v) for k, v in candidate_row.items() if k.startswith('part_')}
    # Sort parts by size descending for easier reading
    parts_sorted = dict(sorted(parts.items(), key=lambda x: x[1], reverse=True))
    return parts_sorted


def plot_parts(parts: dict, title: str, outpath: str):
    labels = list(parts.keys())
    values = [parts[k] for k in labels]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color='C0')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Parameter count')
    plt.title(title)
    # annotate
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def read_csv(csv_path):
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, help='CSV produced by hyperparam_search')
    p.add_argument('--json', type=str, help='JSON summary produced by parameter_budget')
    p.add_argument('--which', type=str, default='student_candidates', help='Which list in JSON to use (student_candidates/teacher_candidates/ta_candidates)')
    p.add_argument('--top', type=int, default=5)
    p.add_argument('--outdir', type=str, default='plots')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    candidates = []
    if args.csv:
        candidates = read_csv(args.csv)
    elif args.json:
        with open(args.json, 'r') as f:
            data = json.load(f)
        candidates = data.get(args.which, [])
    else:
        raise ValueError('Provide --csv or --json')

    for i, cand in enumerate(candidates[: args.top]):
        parts = extract_parts(cand)
        meta = cand.get('hidden_dims') or cand.get('encoder_dims') or cand.get('hidden_channels') or cand.get('hidden_units') or str(i)
        title = f"Candidate {i+1}: {meta} total={cand.get('total_params', cand.get('part_total', ''))}"
        outpath = outdir / f"candidate_{i+1}_{meta}.png"
        plot_parts(parts, title, str(outpath))
        print('Wrote', outpath)


if __name__ == '__main__':
    main()
