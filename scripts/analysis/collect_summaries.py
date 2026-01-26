"""
Collect and present `summary.json` files across experiment runs.

Usage:
    python scripts/collect_summaries.py --experiment-root experiment_runs --json
"""
import json
from pathlib import Path
import argparse


def find_summaries(experiment_root: Path):
    out = []
    for p in experiment_root.rglob('summary.json'):
        try:
            data = json.loads(p.read_text())
            data['_path'] = str(p)
            out.append(data)
        except Exception:
            continue
    return out


def main():
    parser = argparse.ArgumentParser(description='Collect summary.json files across runs')
    parser.add_argument('--experiment-root', default='experimentruns')
    parser.add_argument('--json', action='store_true', help='Print JSON output')
    args = parser.parse_args()

    root = Path(args.experiment_root)
    summaries = find_summaries(root)

    if args.json:
        print(json.dumps(summaries, indent=2))
        return

    # Human table
    if not summaries:
        print('No summaries found under', root)
        return

    print('\nFound summaries:')
    print('=' * 80)
    for s in summaries:
        print(f"Model: {s.get('model')} | Dataset: {s.get('dataset')} | Training: {s.get('training_mode')} | Path: {s.get('_path')}")
    print('=' * 80)


if __name__ == '__main__':
    main()
