#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GITHUB_REPO="<owner/repo>" RUNNER_TOKEN="<token>" ./scripts/setup_selfhosted_runner.sh
# The RUNNER_TOKEN can be created in your repository Settings -> Actions -> Runners -> New self-hosted runner -> Generate token

GITHUB_REPO=${GITHUB_REPO:-}
RUNNER_TOKEN=${RUNNER_TOKEN:-}
RUNNER_NAME=${RUNNER_NAME:-"gpu-runner-$(hostname)"}
RUNNER_LABELS=${RUNNER_LABELS:-"self-hosted,gpu"}
RUNNER_DIR=${RUNNER_DIR:-"/opt/github-runner"}

if [ -z "${GITHUB_REPO}" ] || [ -z "${RUNNER_TOKEN}" ]; then
  echo "ERROR: Please set GITHUB_REPO and RUNNER_TOKEN environment variables before running." >&2
  echo "Example: GITHUB_REPO=RobertFrenken/KD-GAT RUNNER_TOKEN=xxx ./scripts/setup_selfhosted_runner.sh" >&2
  exit 1
fi

# Ensure prerequisites: curl, tar
if ! command -v curl >/dev/null 2>&1; then
  echo "Please install curl" >&2; exit 1
fi

mkdir -p "${RUNNER_DIR}"
cd "${RUNNER_DIR}"

ARCHIVE_URL="https://github.com/actions/runner/releases/download/v2.308.0/actions-runner-linux-x64-2.308.0.tar.gz"

echo "Downloading GitHub Actions runner to ${RUNNER_DIR}..."
curl -L -o actions-runner.tar.gz "${ARCHIVE_URL}"
rm -rf actions-runner
tar xzf actions-runner.tar.gz

echo "Configuring runner name=${RUNNER_NAME} labels=${RUNNER_LABELS}"
./config.sh --unattended --url https://github.com/${GITHUB_REPO} --token "${RUNNER_TOKEN}" --name "${RUNNER_NAME}" --labels "${RUNNER_LABELS}"

# Install as a service for automatic start
./svc.sh install
./svc.sh start

cat <<'EOF'

Runner installed and started.
To reconfigure or remove the runner:
  cd ${RUNNER_DIR}
  ./svc.sh stop
  ./svc.sh uninstall
  ./config.sh remove

Notes:
 - Ensure NVIDIA drivers and CUDA are installed on this host
 - Install Docker or required runtime if you plan to use containerized tasks
 - Secure this host: limit SSH access, keep runner token secret

EOF