#!/bin/bash
# Snakemake cluster-status script for SLURM.
# Called with one argument: the job ID returned by sbatch --parsable.
# Must print exactly one of: running, success, failed

jobid="$1"

if [ -z "$jobid" ]; then
    echo "failed"
    exit 0
fi

# sacct is more reliable than squeue for completed jobs
state=$(sacct -j "$jobid" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')

case "$state" in
    COMPLETED)    echo "success" ;;
    RUNNING|PENDING|REQUEUED|SUSPENDED|CONFIGURING) echo "running" ;;
    "")           echo "running" ;;  # sacct may lag behind squeue
    *)            echo "failed" ;;
esac
