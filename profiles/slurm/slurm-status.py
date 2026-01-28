#!/usr/bin/env python3
"""
SLURM status checker for Snakemake.

This script checks the status of SLURM jobs for Snakemake's executor.
It should return:
- "running" if the job is running or pending
- "success" if the job completed successfully
- "failed" if the job failed

Usage:
    slurm-status.py <job_id>
"""

import subprocess
import sys


def get_slurm_status(job_id):
    """
    Query SLURM for job status.

    Args:
        job_id: SLURM job ID

    Returns:
        Status string: "running", "success", or "failed"
    """
    try:
        # Query sacct for job status
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            # If sacct fails, try squeue (for very recent jobs)
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0 or not result.stdout.strip():
                # Job not found in squeue or sacct - assume completed
                return "success"

        status = result.stdout.strip().split("\n")[0].strip()

        # Map SLURM states to Snakemake states
        if status in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]:
            return "running"
        elif status in ["COMPLETED"]:
            return "success"
        elif status in ["FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"]:
            return "failed"
        else:
            # Unknown state - assume running
            return "running"

    except subprocess.TimeoutExpired:
        print("failed", file=sys.stderr)
        return "failed"
    except Exception as e:
        print(f"Error checking job status: {e}", file=sys.stderr)
        return "failed"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: slurm-status.py <job_id>", file=sys.stderr)
        sys.exit(1)

    job_id = sys.argv[1]
    status = get_slurm_status(job_id)
    print(status)
