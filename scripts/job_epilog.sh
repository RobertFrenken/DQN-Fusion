#!/usr/bin/env bash
# Post-job GPU utilization report.
# Source this at the end of SLURM job scripts:
#   source scripts/job_epilog.sh
#
# Reads nvidia-smi accounting data to report peak GPU utilization and memory.

echo ""
echo "=== GPU Utilization Report ==="
echo "Job ID:    ${SLURM_JOB_ID:-unknown}"
echo "Job Name:  ${SLURM_JOB_NAME:-unknown}"
echo "End Time:  $(date '+%Y-%m-%d %H:%M:%S')"

if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits | while IFS=',' read -r idx name gpu_util mem_util mem_used mem_total temp; do
        echo "GPU ${idx} (${name}):"
        echo "  Compute utilization: ${gpu_util}%"
        echo "  Memory utilization:  ${mem_util}%"
        echo "  Memory used:         ${mem_used} / ${mem_total} MiB"
        echo "  Temperature:         ${temp}°C"
    done

    # Peak memory from accounting (if available)
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo ""
        echo "SLURM accounting (sacct):"
        sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,TRESUsageInTot%80 \
            --noheader --parsable2 2>/dev/null | head -5 || true
    fi
else
    echo "nvidia-smi not available (CPU-only job?)"
fi

echo "=== End Report ==="
