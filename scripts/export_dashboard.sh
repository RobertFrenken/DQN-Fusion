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

# --- Activate conda if not already active ---
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "gnn-experiments" ]]; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate gnn-experiments
fi

# --- Export DB → JSON ---
echo "Exporting project DB → ${DASHBOARD_DATA}/"
python -m pipeline.export --output-dir "$DASHBOARD_DATA"
echo "Export complete."

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
