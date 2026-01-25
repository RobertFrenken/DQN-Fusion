"""
Pre-submit readiness check script

Performs the following steps:
 1. Environment check (runs `scripts/check_environment.py`)
 2. Dataset existence check (runs `scripts/check_datasets.py` and optionally `--run-load`)
 3. Optional short smoke run (synthetic by default) using `scripts/local_smoke_experiment.py`
 4. Preview SLURM sweep (runs `python oscjobmanager.py preview --json ...`)

Usage:
  python scripts/pre_submit_check.py --dataset hcrl_ch --run-load --smoke --smoke-synthetic --preview-json

The script prints a short summary and exits with code 0 (ready) or 2 (not ready).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, timeout=300):
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as e:
        return 124, '', f'Timeout after {timeout}s'
    except Exception as e:
        return 1, '', str(e)


def main():
    parser = argparse.ArgumentParser(description='Pre-submit readiness check')
    parser.add_argument('--dataset', help='Dataset key (e.g., hcrl_ch)')
    parser.add_argument('--run-load', action='store_true', help='Attempt to run dataset loader to validate CSV discovery/cache')
    parser.add_argument('--smoke', action='store_true', help='Run a short smoke run (synthetic by default)')
    parser.add_argument('--smoke-synthetic', action='store_true', help='Use synthetic data for smoke (default when --smoke)')
    parser.add_argument('--smoke-epochs', type=int, default=1, help='Epochs for smoke run')
    parser.add_argument('--smoke-size', type=int, default=100, help='Synthetic dataset size (rows)')
    parser.add_argument('--preview-json', action='store_true', help='Run oscjobmanager preview and parse JSON')
    parser.add_argument('--experiment-root', default=str(Path.cwd() / 'experimentruns_presubmit'), help='Where to write smoke outputs for presubmit checks')
    args = parser.parse_args()

    summary = {
        'env_check': {'ok': False, 'stdout': '', 'stderr': ''},
        'dataset_check': {'ok': None, 'stdout': '', 'stderr': ''},
        'smoke_run': {'ok': None, 'stdout': '', 'stderr': ''},
        'preview': {'ok': None, 'json': None, 'stdout': '', 'stderr': ''}
    }

    print('\n1) Environment check:')
    rc, out, err = run_cmd(['python', 'scripts/check_environment.py'], timeout=60)
    summary['env_check']['ok'] = (rc == 0)
    summary['env_check']['stdout'] = out
    summary['env_check']['stderr'] = err
    print(out)

    if args.dataset:
        print('\n2) Dataset check:')
        cmd = ['python', 'scripts/check_datasets.py', '--dataset', args.dataset]
        if args.run_load:
            cmd += ['--run-load']
        rc, out, err = run_cmd(cmd, timeout=300)
        summary['dataset_check']['ok'] = (rc == 0)
        summary['dataset_check']['stdout'] = out
        summary['dataset_check']['stderr'] = err
        print(out)

    if args.smoke:
        print('\n3) Smoke run:')
        smoke_cmd = ['python', 'scripts/local_smoke_experiment.py', '--run', '--epochs', str(args.smoke_epochs), '--experiment-root', args.experiment_root]
        if args.smoke_synthetic:
            smoke_cmd += ['--use-synthetic-data', '--synthetic-size', str(args.smoke_size)]
        rc, out, err = run_cmd(smoke_cmd, timeout=600)
        summary['smoke_run']['ok'] = (rc == 0)
        summary['smoke_run']['stdout'] = out
        summary['smoke_run']['stderr'] = err
        print(out)

    if args.preview_json:
        print('\n4) Preview SLURM sweep:')
        cmd = ['python', 'oscjobmanager.py', 'preview', '--dataset', args.dataset or 'hcrl_ch', '--json']
        rc, out, err = run_cmd(cmd, timeout=30)
        summary['preview']['ok'] = (rc == 0)
        summary['preview']['stdout'] = out
        summary['preview']['stderr'] = err
        if rc == 0:
            try:
                parsed = json.loads(out)
                summary['preview']['json'] = parsed
                print(json.dumps(parsed, indent=2)[:1000])
            except Exception as e:
                summary['preview']['json'] = None
                print('Preview JSON parse failed:', e)
        else:
            print(err)

    # Final concise report
    print('\n' + '=' * 60)
    print('Pre-submit readiness summary:')
    print('=' * 60)
    print(f"Env check: {'OK' if summary['env_check']['ok'] else 'FAIL'}")
    if args.dataset:
        ds_ok = summary['dataset_check']['ok']
        print(f"Dataset check ({args.dataset}): {'OK' if ds_ok else 'FAIL'}")
    if args.smoke:
        smoke_ok = summary['smoke_run']['ok']
        print(f"Smoke run: {'OK' if smoke_ok else 'FAIL'}")
    if args.preview_json:
        print(f"Preview: {'OK' if summary['preview']['ok'] else 'FAIL'}")

    all_ok = summary['env_check']['ok'] and (summary['dataset_check']['ok'] in (True, None)) and (summary['smoke_run']['ok'] in (True, None)) and (summary['preview']['ok'] in (True, None))

    print('\nDetailed logs are available in the summary object if you need more info.')

    if not all_ok:
        print('\nNOT READY: Address the failing items above before submission')
        sys.exit(2)

    print('\nREADY: Pre-submit checks passed. You can proceed to submit your jobs using oscjobmanager.')
    sys.exit(0)


if __name__ == '__main__':
    main()
