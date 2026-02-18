#!/usr/bin/env bash
# Export experiment data → dashboard JSON and sync to S3.
# Dashboard data is served from S3, not committed to git.
#
# Fast exports (leaderboard, runs, metrics, etc.) run on login node in ~2s.
# Heavy exports (embeddings, graph_samples, attention, recon_errors) should
# be submitted to SLURM via: sbatch scripts/export_dashboard_slurm.sh --only-heavy
#
# Usage:
#   bash scripts/export_dashboard.sh                  # fast exports + S3 sync
#   bash scripts/export_dashboard.sh --skip-heavy     # same (explicit)
#   bash scripts/export_dashboard.sh --all            # all exports (use on SLURM)
#   bash scripts/export_dashboard.sh --only-heavy     # heavy exports only (SLURM)
#   bash scripts/export_dashboard.sh --dry-run        # export only (no S3 sync)
set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT"
DASHBOARD_DATA="docs/dashboard/data"
S3_BUCKET="${KD_GAT_S3_BUCKET:-kd-gat}"

# --- Parse flags ---
DRY_RUN=false
EXPORT_FLAG="--skip-heavy"
for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=true ;;
        --skip-heavy)  EXPORT_FLAG="--skip-heavy" ;;
        --only-heavy)  EXPORT_FLAG="--only-heavy" ;;
        --all)         EXPORT_FLAG="" ;;
        *)             echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

# --- Ensure Python env is available ---
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "gnn-experiments" ]]; then
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null)"
        conda activate gnn-experiments
    else
        # Fallback: add conda env bin directly (works in non-interactive shells)
        export PATH="$HOME/.conda/envs/gnn-experiments/bin:$PATH"
    fi
fi

# --- Export experiment data → JSON ---
WORKERS_FLAG=""
if [[ "$EXPORT_FLAG" == "--only-heavy" ]] || [[ -z "$EXPORT_FLAG" ]]; then
    WORKERS_FLAG="--workers 4"
fi

echo "Exporting experiment data → ${DASHBOARD_DATA}/ ${EXPORT_FLAG:+(${EXPORT_FLAG})} ${WORKERS_FLAG}"
python -m pipeline.export --output-dir "$DASHBOARD_DATA" $EXPORT_FLAG $WORKERS_FLAG
echo "Export complete."

if $DRY_RUN; then
    echo "Dry run — skipping S3 sync."
    exit 0
fi

# --- Sync dashboard data to S3 (public read) ---
echo "Syncing dashboard data to s3://${S3_BUCKET}/dashboard/..."
aws s3 sync "$DASHBOARD_DATA/" "s3://${S3_BUCKET}/dashboard/" --delete \
    || echo "WARNING: S3 dashboard sync failed (non-fatal)"

# --- Push DVC-tracked data to S3 remote (if configured) ---
if dvc remote list 2>/dev/null | grep -q "s3"; then
    echo "Pushing DVC data to S3 remote..."
    dvc push -r s3 2>/dev/null || echo "WARNING: DVC push to S3 failed (non-fatal)"
fi

echo "Dashboard export + S3 sync complete."
