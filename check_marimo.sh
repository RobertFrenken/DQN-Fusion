#!/bin/bash
# ============================================================================
# Check status of running Marimo visualization sessions
# ============================================================================
#
# Usage:
#   ./check_marimo.sh           # List all Marimo jobs
#   ./check_marimo.sh <JOB_ID>  # Show connection info for specific job
#
# ============================================================================

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

if [ $# -eq 0 ]; then
    # List all Marimo jobs
    echo "=================================================================="
    echo "🎨 Active Marimo Visualization Sessions"
    echo "=================================================================="
    echo ""

    JOBS=$(squeue -u $USER -n marimo_viz -h -o "%i %T %M %R" 2>/dev/null || echo "")

    if [ -z "$JOBS" ]; then
        echo "No active Marimo sessions found."
        echo ""
        echo "To start a new session:"
        echo "  ./start_marimo.sh"
        echo ""
    else
        printf "%-12s %-12s %-12s %s\n" "JOB_ID" "STATE" "TIME" "REASON"
        echo "----------------------------------------------------------------"
        echo "$JOBS"
        echo ""
        echo "=================================================================="
        echo "To view connection instructions for a job:"
        echo "  ./check_marimo.sh <JOB_ID>"
        echo ""
        echo "To stop a job:"
        echo "  scancel <JOB_ID>"
        echo "=================================================================="
    fi

else
    # Show connection info for specific job
    JOB_ID=$1

    echo "=================================================================="
    echo "🎨 Marimo Session Info - Job $JOB_ID"
    echo "=================================================================="
    echo ""

    # Check if job exists
    JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "NOT_FOUND")

    if [ "$JOB_STATE" = "NOT_FOUND" ]; then
        echo -e "${RED}❌ Job $JOB_ID not found${NC}"
        echo ""
        echo "It may have completed or been cancelled."
        echo ""
        echo "Check recent logs:"
        echo "  ls -lht slurm_logs/marimo_*.out | head"
        exit 1
    fi

    echo -e "${GREEN}Status: $JOB_STATE${NC}"
    echo ""

    # Show full job info
    squeue -j "$JOB_ID" -o "  Node: %N" 2>/dev/null
    squeue -j "$JOB_ID" -o "  Time: %M / %l" 2>/dev/null
    squeue -j "$JOB_ID" -o "  Memory: %m" 2>/dev/null
    echo ""

    # Connection instructions
    INSTRUCTIONS_FILE="slurm_logs/marimo_${JOB_ID}_connection.txt"

    if [ -f "$INSTRUCTIONS_FILE" ]; then
        echo "=================================================================="
        echo "📋 Connection Instructions"
        echo "=================================================================="
        echo ""
        cat "$INSTRUCTIONS_FILE"
    else
        echo -e "${YELLOW}⚠️  Connection instructions not yet available${NC}"
        echo ""
        if [ "$JOB_STATE" = "PENDING" ]; then
            echo "Job is still pending. Waiting for resources..."
            squeue -j "$JOB_ID" -o "  Reason: %R"
        elif [ "$JOB_STATE" = "RUNNING" ]; then
            echo "Job is running but connection file not created yet."
            echo "Wait a few seconds and try again."
        fi
        echo ""
        echo "Connection instructions will be saved to:"
        echo "  $INSTRUCTIONS_FILE"
    fi

    echo ""
    echo "=================================================================="
    echo "📝 Logs"
    echo "=================================================================="
    echo ""

    OUTPUT_LOG="slurm_logs/marimo_${JOB_ID}.out"
    ERROR_LOG="slurm_logs/marimo_${JOB_ID}.err"

    if [ -f "$OUTPUT_LOG" ]; then
        echo -e "${BLUE}Output log:${NC} $OUTPUT_LOG"
        echo "  View: tail -f $OUTPUT_LOG"
        echo ""
        echo "Last 5 lines:"
        tail -5 "$OUTPUT_LOG" | sed 's/^/  /'
    else
        echo "Output log not yet created: $OUTPUT_LOG"
    fi

    echo ""

    if [ -f "$ERROR_LOG" ]; then
        ERROR_SIZE=$(wc -l < "$ERROR_LOG")
        if [ "$ERROR_SIZE" -gt 0 ]; then
            echo -e "${RED}Error log:${NC} $ERROR_LOG (${ERROR_SIZE} lines)"
            echo "  View: cat $ERROR_LOG"
        else
            echo -e "${GREEN}Error log:${NC} $ERROR_LOG (empty - no errors)"
        fi
    else
        echo "Error log not yet created: $ERROR_LOG"
    fi

    echo ""
    echo "=================================================================="
    echo "🛠️  Actions"
    echo "=================================================================="
    echo ""
    echo "Stop server:"
    echo "  scancel $JOB_ID"
    echo ""
    echo "View live output:"
    echo "  tail -f $OUTPUT_LOG"
    echo ""
    echo "SSH tunnel command (from local machine):"
    if [ -f "$INSTRUCTIONS_FILE" ]; then
        grep "ssh -L" "$INSTRUCTIONS_FILE" | sed 's/^/  /'
    else
        echo "  (Available once job starts)"
    fi
    echo ""
    echo "=================================================================="
fi
