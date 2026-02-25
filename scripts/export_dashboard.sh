#!/usr/bin/env bash
# Export experiment data → dashboard JSON and sync to S3.
# All exports are lightweight (~2s, login node safe).
#
# Usage:
#   bash scripts/export_dashboard.sh                  # export + S3 sync
#   bash scripts/export_dashboard.sh --dry-run        # export only (no S3 sync)
set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
DASHBOARD_DATA="docs/dashboard/data"
S3_BUCKET="${KD_GAT_S3_BUCKET:-kd-gat}"

# --- Parse flags ---
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=true ;;
        *)             echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

# --- Ensure Python env is available ---
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source ~/KD-GAT/.venv/bin/activate
fi

# --- Export experiment data → JSON ---
echo "Exporting experiment data → ${DASHBOARD_DATA}/"
python -m pipeline.export --output-dir "$DASHBOARD_DATA"
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
