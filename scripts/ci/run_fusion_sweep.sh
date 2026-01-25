#!/usr/bin/env bash
# Example: run a tiny fusion sweep via oscjobmanager (preview + submit)
# Usage: bash scripts/run_fusion_sweep.sh --dry-run

MANIFEST=jobs/smoke_fusion_sweep.json
DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

echo "Manifest: $MANIFEST"
if [ $DRY_RUN -eq 1 ]; then
  echo "Previewing sweep (no submission)"
  python oscjobmanager.py preview --manifest $MANIFEST --json
else
  echo "Submitting sweep (small smoke sizes)"
  python oscjobmanager.py sweep --manifest $MANIFEST || echo "sweep command failed; inspect manifest"
fi
