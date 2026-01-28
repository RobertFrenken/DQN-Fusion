#!/usr/bin/env python3
"""
Script to find teacher DQN job IDs from squeue and slurm logs
"""
import subprocess
import re
import os
from pathlib import Path

# Datasets in order
DATASETS = ["hcrl_sa", "hcrl_ch", "set_01", "set_02", "set_03", "set_04"]

def get_job_from_output_file(dataset):
    """Extract job ID from existing output file"""
    base_path = Path("experimentruns/automotive")
    pattern = f"{base_path}/{dataset}/rl_fusion/dqn/teacher/no_distillation/fusion/slurm_logs/*.out"

    # Find most recent .out file
    import glob
    files = sorted(glob.glob(str(pattern)), key=os.path.getmtime, reverse=True)

    if files:
        with open(files[0], 'r') as f:
            for line in f:
                if line.startswith("Job ID:"):
                    job_id = line.split(":")[-1].strip()
                    return job_id
    return None

def get_jobs_from_squeue():
    """Get jobs from squeue output"""
    try:
        # Run squeue with proper formatting
        result = subprocess.run(
            ['squeue', '-u', 'rf15', '-o', '%.18i %.100j %.8T'],
            capture_output=True,
            text=True
        )

        jobs = {}
        for line in result.stdout.split('\n'):
            # Look for teacher DQN jobs
            if 'teacher' in line.lower() and 'dqn' in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    job_id = parts[0]
                    job_name = parts[1]
                    # Try to extract dataset name from job name
                    for dataset in DATASETS:
                        dataset_clean = dataset.replace('_', '')
                        if dataset_clean in job_name.lower() or dataset in job_name.lower():
                            jobs[dataset] = job_id
                            break

        return jobs
    except Exception as e:
        print(f"Error running squeue: {e}")
        return {}

def main():
    print("Finding teacher DQN job IDs...")
    print("=" * 60)

    job_ids = {}

    # First, try to get job IDs from output files (completed/running jobs)
    for dataset in DATASETS:
        job_id = get_job_from_output_file(dataset)
        if job_id:
            job_ids[dataset] = job_id
            print(f"{dataset}: {job_id} (from output file)")

    # Then, check squeue for pending/running jobs not yet in output files
    squeue_jobs = get_jobs_from_squeue()
    for dataset in DATASETS:
        if dataset not in job_ids and dataset in squeue_jobs:
            job_ids[dataset] = squeue_jobs[dataset]
            print(f"{dataset}: {squeue_jobs[dataset]} (from squeue)")

    print("=" * 60)
    print("\nJob IDs in order:")
    for dataset in DATASETS:
        if dataset in job_ids:
            print(f"  {dataset}: {job_ids[dataset]}")
        else:
            print(f"  {dataset}: NOT FOUND")

    print("\n" + "=" * 60)
    print("For student_kd.sh TEACHER_DQN_JOBS array:")
    print("TEACHER_DQN_JOBS=(")
    for dataset in DATASETS:
        job_id = job_ids.get(dataset, "UNKNOWN")
        print(f'  "{job_id}"  # {dataset}')
    print(")")

if __name__ == "__main__":
    main()
