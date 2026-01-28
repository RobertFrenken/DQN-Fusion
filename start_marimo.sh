#!/bin/bash
# ============================================================================
# Helper script to launch Marimo visualization server on SLURM
# ============================================================================
#
# Usage:
#   ./start_marimo.sh           # Launch with default settings (4 hours, GPU)
#   ./start_marimo.sh 8         # Launch with 8 hours
#   ./start_marimo.sh 2 cpu     # Launch with 2 hours, no GPU
#
# After job starts, follow the connection instructions to access the notebook.
# ============================================================================

set -euo pipefail

# Default settings
TIME_HOURS=${1:-4}
USE_GPU=${2:-gpu}

echo "=================================================================="
echo "🎨 Starting Marimo Visualization Server"
echo "=================================================================="
echo "Time requested: ${TIME_HOURS} hours"
echo "GPU: ${USE_GPU}"
echo "=================================================================="
echo ""

# Create logs directory
mkdir -p slurm_logs

# Submit job
if [ "$USE_GPU" = "gpu" ]; then
    JOB_ID=$(sbatch --parsable \
        --time="${TIME_HOURS}:00:00" \
        launch_marimo_visualization.sh)
else
    # CPU-only version
    JOB_ID=$(sbatch --parsable \
        --time="${TIME_HOURS}:00:00" \
        --partition=serial \
        --gres="" \
        launch_marimo_visualization.sh)
fi

echo "✅ Job submitted: $JOB_ID"
echo ""
echo "Waiting for job to start (checking every 5 seconds)..."
echo "Press Ctrl+C to stop waiting (job will continue running)"
echo ""

# Wait for job to start and connection file to be created
INSTRUCTIONS_FILE="slurm_logs/marimo_${JOB_ID}_connection.txt"
TIMEOUT=300  # 5 minutes
ELAPSED=0

while [ $ELAPSED -lt $TIMEOUT ]; do
    # Check if job is running
    JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "NOT_FOUND")

    if [ "$JOB_STATE" = "RUNNING" ] && [ -f "$INSTRUCTIONS_FILE" ]; then
        echo ""
        echo "=================================================================="
        echo "✅ Job is running! Connection instructions below:"
        echo "=================================================================="
        echo ""
        cat "$INSTRUCTIONS_FILE"
        echo ""
        echo "=================================================================="
        echo "📝 Quick Reference"
        echo "=================================================================="
        echo ""
        echo "View output log:"
        echo "  tail -f slurm_logs/marimo_${JOB_ID}.out"
        echo ""
        echo "View error log:"
        echo "  tail -f slurm_logs/marimo_${JOB_ID}.err"
        echo ""
        echo "Check job status:"
        echo "  squeue -j $JOB_ID"
        echo ""
        echo "Stop the server:"
        echo "  scancel $JOB_ID"
        echo ""
        echo "=================================================================="
        exit 0
    elif [ "$JOB_STATE" = "PENDING" ]; then
        echo -n "⏳ Job pending... (${ELAPSED}s) "
        squeue -j "$JOB_ID" -h -o "Reason: %R"
    elif [ "$JOB_STATE" = "NOT_FOUND" ]; then
        echo "❌ Job $JOB_ID not found. It may have failed to start."
        echo ""
        echo "Check the error log:"
        echo "  cat slurm_logs/marimo_${JOB_ID}.err"
        exit 1
    elif [ "$JOB_STATE" = "RUNNING" ]; then
        echo "⏳ Job running, waiting for connection instructions... (${ELAPSED}s)"
    else
        echo "❌ Unexpected job state: $JOB_STATE"
        exit 1
    fi

    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

echo ""
echo "⚠️  Timeout waiting for job to start (${TIMEOUT}s)"
echo ""
echo "The job may still be starting. Check manually:"
echo "  squeue -j $JOB_ID"
echo "  cat $INSTRUCTIONS_FILE"
echo ""
