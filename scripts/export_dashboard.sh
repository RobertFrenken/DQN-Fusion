#!/usr/bin/env bash
# Export project DB → dashboard JSON and push to GitHub Pages.
# No GPU/SLURM needed — runs on login node in ~2 seconds.
#
# Usage:
#   bash scripts/export_dashboard.sh              # export + commit + push
#   bash scripts/export_dashboard.sh --no-push    # export + commit only
#   bash scripts/export_dashboard.sh --dry-run    # export only (no git)
set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT"
DASHBOARD_DATA="docs/dashboard/data"
BRANCH="main"
S3_BUCKET="${KD_GAT_S3_BUCKET:-kd-gat}"

# --- Parse flags ---
DRY_RUN=false
NO_PUSH=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --no-push)  NO_PUSH=true ;;
        *)          echo "Unknown flag: $arg"; exit 1 ;;
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

# --- Export DB → JSON ---
echo "Exporting project DB → ${DASHBOARD_DATA}/"
python -m pipeline.export --output-dir "$DASHBOARD_DATA"
echo "Export complete."

# --- Sync dashboard data to S3 (public read) ---
echo "Syncing dashboard data to s3://${S3_BUCKET}/dashboard/..."
aws s3 sync "$DASHBOARD_DATA/" "s3://${S3_BUCKET}/dashboard/" --delete \
    || echo "WARNING: S3 dashboard sync failed (non-fatal)"

# --- Push DVC-tracked data to S3 remote (if configured) ---
if dvc remote list 2>/dev/null | grep -q "s3"; then
    echo "Pushing DVC data to S3 remote..."
    dvc push -r s3 2>/dev/null || echo "WARNING: DVC push to S3 failed (non-fatal)"
fi

if $DRY_RUN; then
    echo "Dry run — skipping git operations."
    exit 0
fi

# --- Check for changes ---
if git diff --quiet "$DASHBOARD_DATA/"; then
    echo "No changes to dashboard data — nothing to commit."
    exit 0
fi

# --- Commit ---
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M UTC")
git add "$DASHBOARD_DATA/"
git commit -m "dashboard: update exported metrics ${TIMESTAMP}"
echo "Committed dashboard update."

if $NO_PUSH; then
    echo "No-push mode — skipping git push."
    exit 0
fi

# --- Push with retry ---
MAX_RETRIES=3
for i in $(seq 1 $MAX_RETRIES); do
    git pull --rebase origin "$BRANCH" 2>/dev/null || true
    if git push origin "$BRANCH" 2>/dev/null; then
        echo "Pushed to ${BRANCH}."
        exit 0
    fi
    echo "Push attempt $i/$MAX_RETRIES failed. Retrying in ${i}s..."
    sleep "$i"
done
echo "ERROR: Dashboard push failed after $MAX_RETRIES attempts." >&2
exit 1
