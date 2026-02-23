#!/usr/bin/env bash
# Stage data from permanent storage to scratch / $TMPDIR for fast job I/O.
#
# Data flow:  KD_GAT_DATA_ROOT (home/project NFS)
#         →   KD_GAT_SCRATCH/kd-gat-data/ (GPFS scratch, persistent across jobs)
#         →   $TMPDIR/kd-gat-data/ (local SSD, per-job, fastest)
#
# Usage:
#   source scripts/stage_data.sh           # default: stage raw + cache
#   source scripts/stage_data.sh --cache   # cache only (for training jobs)
#   source scripts/stage_data.sh --raw     # raw only (for preprocessing jobs)
#
# After sourcing, KD_GAT_DATA_ROOT and KD_GAT_CACHE_ROOT point to the fastest
# available copy. The calling SLURM script can use these env vars directly.

set -euo pipefail

# --- Config ---
DATA_ROOT="${KD_GAT_DATA_ROOT:-/users/PAS2022/rf15/kd-gat-data}"
SCRATCH="${KD_GAT_SCRATCH:-/fs/scratch/PAS1266}"
SCRATCH_DATA="${SCRATCH}/kd-gat-data"

STAGE_RAW=true
STAGE_CACHE=true

for arg in "$@"; do
    case "$arg" in
        --cache) STAGE_RAW=false ;;
        --raw)   STAGE_CACHE=false ;;
    esac
done

echo "=== Data staging ==="
echo "Source:  ${DATA_ROOT}"
echo "Scratch: ${SCRATCH_DATA}"
echo "TMPDIR:  ${TMPDIR:-<not set>}"

# --- Step 1: NFS → Scratch (rsync, incremental) ---
if $STAGE_RAW && [[ -d "${DATA_ROOT}/raw" ]]; then
    echo "Staging raw data to scratch..."
    mkdir -p "${SCRATCH_DATA}/raw"
    rsync -a --info=progress2 "${DATA_ROOT}/raw/" "${SCRATCH_DATA}/raw/"
fi

if $STAGE_CACHE && [[ -d "${DATA_ROOT}/cache" ]]; then
    echo "Staging cache to scratch..."
    mkdir -p "${SCRATCH_DATA}/cache"
    rsync -a --info=progress2 "${DATA_ROOT}/cache/" "${SCRATCH_DATA}/cache/"
fi

# --- Step 2: Scratch → $TMPDIR (cp, per-job local SSD) ---
if [[ -n "${TMPDIR:-}" ]]; then
    TMPDIR_DATA="${TMPDIR}/kd-gat-data"
    mkdir -p "${TMPDIR_DATA}"

    if $STAGE_CACHE && [[ -d "${SCRATCH_DATA}/cache" ]]; then
        echo "Staging cache to TMPDIR..."
        cp -r "${SCRATCH_DATA}/cache" "${TMPDIR_DATA}/"
        export KD_GAT_CACHE_ROOT="${TMPDIR_DATA}/cache"
        echo "KD_GAT_CACHE_ROOT=${KD_GAT_CACHE_ROOT}"
    fi

    if $STAGE_RAW && [[ -d "${SCRATCH_DATA}/raw" ]]; then
        echo "Staging raw data to TMPDIR..."
        cp -r "${SCRATCH_DATA}/raw" "${TMPDIR_DATA}/"
        export KD_GAT_DATA_ROOT="${TMPDIR_DATA}"
        echo "KD_GAT_DATA_ROOT=${KD_GAT_DATA_ROOT}"
    fi
else
    # No TMPDIR (login node or non-SLURM) — use scratch as fastest tier
    if [[ -d "${SCRATCH_DATA}" ]]; then
        export KD_GAT_DATA_ROOT="${SCRATCH_DATA}"
        export KD_GAT_CACHE_ROOT="${SCRATCH_DATA}/cache"
    fi
fi

echo "=== Staging complete ==="
echo "KD_GAT_DATA_ROOT=${KD_GAT_DATA_ROOT}"
echo "KD_GAT_CACHE_ROOT=${KD_GAT_CACHE_ROOT:-<using default>}"
