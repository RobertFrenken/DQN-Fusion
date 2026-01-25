#!/usr/bin/env python3
"""Generate per-dataset dependency manifests with absolute canonical paths.

Usage:
  # Use default experiment root (project_root/experiment_runs)
  python scripts/generate_manifests.py --datasets hcrl_sa hcrl_ch

  # Provide a custom experiment root
  python scripts/generate_manifests.py --experiment-root /path/to/experiment_runs --datasets hcrl_sa

Outputs are written to: docs/examples/manifests/generated/{dataset}_manifest.json
"""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from osc_job_manager import OSCJobManager


def make_manifest(experiment_root: Path, dataset: str) -> dict:
    base = Path(experiment_root)
    autoencoder_path = base / 'automotive' / dataset / 'unsupervised' / 'vgae' / 'teacher' / 'no_distillation' / 'autoencoder' / 'vgae_autoencoder.pth'
    classifier_path = base / 'automotive' / dataset / 'supervised' / 'gat' / 'teacher' / 'no_distillation' / 'normal' / f'gat_{dataset}_normal.pth'
    return {
        'autoencoder': {
            'path': str(autoencoder_path),
            'model_size': 'teacher',
            'training_mode': 'autoencoder',
            'distillation': 'no_distillation'
        },
        'classifier': {
            'path': str(classifier_path),
            'model_size': 'teacher',
            'training_mode': 'normal',
            'distillation': 'no_distillation'
        }
    }


def write_manifest(manifest: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate per-dataset manifests with absolute canonical paths')
    parser.add_argument('--experiment-root', type=Path, help='Root path for experiments (defaults to project/experiment_runs)')
    parser.add_argument('--datasets', nargs='+', help='Datasets to generate manifests for (default: all automotive datasets)')
    parser.add_argument('--out-dir', type=Path, default=PROJECT_ROOT / 'docs' / 'examples' / 'manifests' / 'generated', help='Output directory for generated manifests')

    args = parser.parse_args()

    manager = OSCJobManager()
    default_datasets = manager.training_configurations['datasets']['automotive']

    datasets = args.datasets or default_datasets

    exp_root = args.experiment_root if args.experiment_root else (Path(os.environ.get('CAN_EXPERIMENT_ROOT')) if os.environ.get('CAN_EXPERIMENT_ROOT') else PROJECT_ROOT / 'experiment_runs')

    for ds in datasets:
        m = make_manifest(exp_root, ds)
        out_path = args.out_dir / f"{ds}_manifest.json"
        write_manifest(m, out_path)

    print('Done. Generated manifests in', args.out_dir)


if __name__ == '__main__':
    import os
    main()
