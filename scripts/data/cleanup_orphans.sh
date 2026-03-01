#!/usr/bin/env bash
# Scan experimentruns/ for orphaned/failed run directories.
#
# An "orphan" is a run directory that has no .done sentinel file,
# indicating the stage never completed successfully.
#
# Modes:
#   --dry-run  (default)  List orphaned dirs and failed lakehouse records
#   --archive             Move orphaned dirs to $KD_GAT_DATA_ROOT/archive/
#   --delete              Permanently remove orphaned dirs (destructive!)
#
# Usage:
#   bash scripts/data/cleanup_orphans.sh                # dry-run
#   bash scripts/data/cleanup_orphans.sh --archive      # archive orphans
#   bash scripts/data/cleanup_orphans.sh --delete       # delete orphans

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${PROJECT_ROOT}/experimentruns"
DATA_ROOT="${KD_GAT_DATA_ROOT:-}"
LAKEHOUSE_DIR="${DATA_ROOT:+${DATA_ROOT}/lakehouse/runs}"
LAKEHOUSE_DIR="${LAKEHOUSE_DIR:-${EXPERIMENT_ROOT}/lakehouse}"
ARCHIVE_DIR="${DATA_ROOT:+${DATA_ROOT}/archive}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${EXPERIMENT_ROOT}/archive}"

MODE="dry-run"
if [[ "${1:-}" == "--archive" ]]; then
    MODE="archive"
elif [[ "${1:-}" == "--delete" ]]; then
    MODE="delete"
fi

echo "=== Cleanup Orphaned Runs ==="
echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Lakehouse dir:   ${LAKEHOUSE_DIR}"
echo "Mode:            ${MODE}"
echo ""

# --- Find orphaned run directories (no .done sentinel) ---
orphans=()
if [[ -d "$EXPERIMENT_ROOT" ]]; then
    # Run dirs are two levels deep: experimentruns/{dataset}/{run_name}/
    for dataset_dir in "$EXPERIMENT_ROOT"/*/; do
        [[ -d "$dataset_dir" ]] || continue
        dataset_name="$(basename "$dataset_dir")"
        # Skip non-run directories (lakehouse, archive)
        [[ "$dataset_name" == "lakehouse" || "$dataset_name" == "archive" ]] && continue

        for run_dir in "$dataset_dir"*/; do
            [[ -d "$run_dir" ]] || continue
            # Skip archived runs
            [[ "$(basename "$run_dir")" == *.archive_* ]] && continue
            # Check for .done sentinel
            if [[ ! -f "${run_dir}.done" ]]; then
                orphans+=("$run_dir")
            fi
        done
    done
fi

echo "Orphaned directories (no .done sentinel): ${#orphans[@]}"
for d in "${orphans[@]:-}"; do
    [[ -z "$d" ]] && continue
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "  ${d} (${size})"
done
echo ""

# --- Cross-reference lakehouse JSON for failed runs ---
failed_runs=()
if [[ -d "$LAKEHOUSE_DIR" ]]; then
    for json_file in "$LAKEHOUSE_DIR"/*.json; do
        [[ -f "$json_file" ]] || continue
        # Use python for reliable JSON parsing
        is_failed=$(python3 -c "
import json, sys
with open('$json_file') as f:
    d = json.load(f)
if not d.get('success', True):
    reason = d.get('failure_reason', 'unknown')
    print(f\"{d['run_id']}|{reason}\")
" 2>/dev/null || true)
        if [[ -n "$is_failed" ]]; then
            failed_runs+=("$is_failed")
        fi
    done
fi

echo "Failed runs in lakehouse: ${#failed_runs[@]}"
for entry in "${failed_runs[@]:-}"; do
    [[ -z "$entry" ]] && continue
    run_id="${entry%%|*}"
    reason="${entry#*|}"
    echo "  ${run_id}: ${reason}"
done
echo ""

# --- Take action ---
if [[ "$MODE" == "dry-run" ]]; then
    echo "Dry-run complete. Use --archive or --delete to take action."
    exit 0
fi

if [[ ${#orphans[@]} -eq 0 ]]; then
    echo "No orphans to process."
    exit 0
fi

if [[ "$MODE" == "archive" ]]; then
    mkdir -p "$ARCHIVE_DIR"
    for d in "${orphans[@]}"; do
        dest="${ARCHIVE_DIR}/$(basename "$(dirname "$d")")_$(basename "$d")"
        echo "Archiving: $d → $dest"
        mv "$d" "$dest"
    done
    echo "Archived ${#orphans[@]} directories to ${ARCHIVE_DIR}"
elif [[ "$MODE" == "delete" ]]; then
    for d in "${orphans[@]}"; do
        echo "Deleting: $d"
        rm -rf "$d"
    done
    echo "Deleted ${#orphans[@]} orphaned directories."
fi
