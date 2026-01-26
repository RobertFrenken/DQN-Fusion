#!/usr/bin/env python3
"""Driver script to run job specs with OSCJobManager.

Usage examples:
  # Generate SBATCH scripts only for example vgae jobs
  python scripts/drive_jobs.py --job jobs/example_vgae.json --generate-only

  # Submit vgae jobs for all automotive datasets
  python scripts/drive_jobs.py --job jobs/example_vgae.json --submit

  # Submit the full pipeline (autoencoder -> curriculum -> fusion) with automatic chaining
  python scripts/drive_jobs.py --job jobs/example_pipeline.json --submit --chain

Notes:
- Job spec files are JSON with fields: name, datasets (list or "all"), training_types (list), extra_args (dict), dependency_manifests (dict optional per dataset)
- When --chain is used, the driver will submit jobs in order and use sbatch dependencies (afterok) to chain them.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import sys

# Ensure repo root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osc_job_manager import OSCJobManager


def load_job_spec(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def resolve_datasets(spec: Dict[str, Any], manager: OSCJobManager) -> List[str]:
    datasets = spec.get('datasets', 'all')
    if datasets == 'all':
        return manager.training_configurations['datasets']['automotive']
    return datasets


def submit_or_generate(spec_path: Path, submit: bool = False, chain: bool = False, generate_only: bool = False):
    manager = OSCJobManager()
    spec = load_job_spec(spec_path)
    datasets = resolve_datasets(spec, manager)

    training_types = spec.get('training_types', [])
    extra_args_template = spec.get('extra_args', {})
    dep_manifests_template = spec.get('dependency_manifests', {})  # dict mapping dataset->path or template

    created_scripts = []
    submitted_jobs = []

    # Validate all jobs before generating/submitting; collect errors to fail-fast
    validation_errors = []
    for ds in datasets:
        for t in training_types:
            extra_args = {k: (v.format(dataset=ds) if isinstance(v, str) else v) for k, v in extra_args_template.items()}
            if isinstance(dep_manifests_template, dict):
                dm = dep_manifests_template.get(ds) or dep_manifests_template.get('default')
                if dm:
                    extra_args['dependency_manifest'] = dm.format(dataset=ds) if isinstance(dm, str) else dm

            ok, errors = manager.validate_job_spec(t, ds, extra_args)
            if not ok:
                validation_errors.append({'dataset': ds, 'training_type': t, 'errors': errors})

    if validation_errors:
        print('❌ Validation failed for one or more jobs:')
        for v in validation_errors:
            print(f" - dataset={v['dataset']}, training_type={v['training_type']}: {v['errors']}")
        raise SystemExit(1)

    # All validations passed — proceed to generate scripts
    for ds in datasets:
        for t in training_types:
            # instantiate any template placeholders (e.g., {dataset})
            extra_args = {k: (v.format(dataset=ds) if isinstance(v, str) else v) for k, v in extra_args_template.items()}
            # per-dataset dependency manifest
            if isinstance(dep_manifests_template, dict):
                dm = dep_manifests_template.get(ds) or dep_manifests_template.get('default')
                if dm:
                    extra_args['dependency_manifest'] = dm.format(dataset=ds) if isinstance(dm, str) else dm

            script = manager.generate_slurm_script(job_name=f"can_{t}", training_type=t, dataset=ds, extra_args=extra_args)
            created_scripts.append(script)
            print(f"Created script: {script}")

    if generate_only and not submit:
        return created_scripts

    if submit:
        # If chaining requested, submit grouped by dataset in pipeline order
        if chain:
            for ds in datasets:
                previous_job_id = None
                for t in training_types:
                    # Build per-job extra args again
                    extra_args = {k: (v.format(dataset=ds) if isinstance(v, str) else v) for k, v in extra_args_template.items()}
                    dm = dep_manifests_template.get(ds) or dep_manifests_template.get('default') if isinstance(dep_manifests_template, dict) else None
                    if dm:
                        extra_args['dependency_manifest'] = dm.format(dataset=ds) if isinstance(dm, str) else dm

                    script = manager.generate_slurm_script(job_name=f"can_{t}", training_type=t, dataset=ds, extra_args=extra_args)
                    job_id = manager._submit_slurm_job(Path(script), dependency=previous_job_id)  # afterok chaining
                    print(f"Submitted job for dataset={ds}, type={t}. job_id: {job_id}, dependency: {previous_job_id}")
                    submitted_jobs.append(job_id)
                    previous_job_id = job_id
        else:
            # Submit all scripts without chaining
            for script in created_scripts:
                job_id = manager._submit_slurm_job(Path(script))
                print(f"Submitted script {script} -> job {job_id}")
                submitted_jobs.append(job_id)

    return {'scripts': created_scripts, 'jobs': submitted_jobs}


def main():
    parser = argparse.ArgumentParser(description='Drive job specs and submit/generate SLURM scripts using OSCJobManager')
    parser.add_argument('--job', required=True, type=Path, help='Path to job spec JSON file')
    parser.add_argument('--generate-only', action='store_true', help='Generate scripts and exit without submitting')
    parser.add_argument('--submit', action='store_true', help='Submit generated scripts to SLURM (uses sbatch)')
    parser.add_argument('--chain', action='store_true', help='When submitting, chain jobs per dataset using sbatch dependencies (afterok)')

    args = parser.parse_args()
    res = submit_or_generate(args.job, submit=args.submit, chain=args.chain, generate_only=args.generate_only)
    print('\n=== Result ===')
    print(res)


if __name__ == '__main__':
    main()
