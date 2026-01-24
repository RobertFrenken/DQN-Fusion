#!/usr/bin/env bash
set -euo pipefail
URL=${1:-http://localhost:5000}
if command -v curl >/dev/null 2>&1; then
  if curl --silent --fail $URL >/dev/null; then
    echo "MLflow UI reachable at $URL"
    exit 0
  else
    echo "MLflow UI not reachable at $URL"
    exit 2
  fi
else
  echo "curl not found; cannot check MLflow UI"
  exit 3
fi
