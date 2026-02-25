#!/usr/bin/env bash
# Sync dashboard export data into docs-site for Astro consumption.
# Run after: python -m pipeline.export
#
# Two tiers:
#   1. Catalog JSON → src/data/ (for Astro Content Collections, build-time)
#   2. All data → public/data/ symlink (for client-side fetch, runtime)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_SOURCE="$(cd "$SITE_DIR/../docs/dashboard/data" && pwd)"

# Tier 1: Copy catalog files into src/data/ for content collections
CATALOG_DIR="$SITE_DIR/src/data"
mkdir -p "$CATALOG_DIR"

CATALOG_FILES=(
  leaderboard.json
  runs.json
  datasets.json
  kd_transfer.json
  model_sizes.json
)

echo "Syncing catalog data → src/data/"
for f in "${CATALOG_FILES[@]}"; do
  if [[ -f "$DATA_SOURCE/$f" ]]; then
    cp "$DATA_SOURCE/$f" "$CATALOG_DIR/$f"
    echo "  ✓ $f"
  else
    echo "  ✗ $f (not found)"
  fi
done

# Also copy metric_catalog from metrics/ subdirectory
if [[ -f "$DATA_SOURCE/metrics/metric_catalog.json" ]]; then
  cp "$DATA_SOURCE/metrics/metric_catalog.json" "$CATALOG_DIR/metric_catalog.json"
  echo "  ✓ metric_catalog.json"
fi

# Tier 2: Symlink public/data/ → dashboard data for runtime fetch
PUBLIC_DATA="$SITE_DIR/public/data"
if [[ -L "$PUBLIC_DATA" ]]; then
  echo "Symlink public/data/ already exists → $(readlink "$PUBLIC_DATA")"
elif [[ -d "$PUBLIC_DATA" ]]; then
  echo "WARNING: public/data/ is a real directory, not a symlink. Skipping."
else
  mkdir -p "$SITE_DIR/public"
  ln -s "$DATA_SOURCE" "$PUBLIC_DATA"
  echo "Created symlink: public/data/ → $DATA_SOURCE"
fi

echo "Done."
