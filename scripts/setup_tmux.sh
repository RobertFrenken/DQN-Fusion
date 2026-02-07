#!/usr/bin/env bash
# Set up tmux sessions for KD-GAT development.
# Creates: claude, terminal, pipeline (with squeue watch pane).
set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT="/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT"
CONDA_ENV="/users/PAS2022/rf15/.conda/envs/gnn-experiments"
SESSIONS=(claude terminal pipeline)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
${BOLD}Usage:${NC} $(basename "$0") [OPTIONS]

Set up tmux sessions for KD-GAT development.

Sessions created:
  claude    - Claude Code / AI assistant
  terminal  - General terminal work
  pipeline  - Pipeline monitoring (split pane with squeue watch)

${BOLD}Options:${NC}
  -a, --attach <name>     Attach to a specific session
  -l, --list              List existing tmux sessions and exit
  -h, --help              Show this help message

${BOLD}Notes:${NC}
  tmux sessions are node-specific. You must be on the same login node
  to reattach. Current node: $(hostname)
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
ATTACH=""
LIST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--attach) ATTACH="$2"; shift 2 ;;
        -l|--list)   LIST=true;   shift ;;
        -h|--help)   usage ;;
        *)           error "Unknown option: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Check tmux
# ---------------------------------------------------------------------------
if ! command -v tmux &>/dev/null; then
    error "tmux is not installed or not in PATH."
    exit 1
fi

# ---------------------------------------------------------------------------
# List mode
# ---------------------------------------------------------------------------
if $LIST; then
    echo -e "${BOLD}tmux sessions on $(hostname):${NC}"
    tmux list-sessions 2>/dev/null || echo "  (no sessions)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Attach mode
# ---------------------------------------------------------------------------
if [[ -n "$ATTACH" ]]; then
    if tmux has-session -t "$ATTACH" 2>/dev/null; then
        exec tmux attach-session -t "$ATTACH"
    else
        error "Session '$ATTACH' does not exist."
        echo "Available sessions:"
        tmux list-sessions 2>/dev/null || echo "  (none)"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Create sessions
# ---------------------------------------------------------------------------
echo -e "${BOLD}Setting up tmux sessions on $(hostname)${NC}"
echo ""

for sess in "${SESSIONS[@]}"; do
    if tmux has-session -t "$sess" 2>/dev/null; then
        ok "$sess — already exists, skipping"
    else
        tmux new-session -d -s "$sess" -c "$PROJECT_ROOT"
        tmux send-keys -t "$sess" "conda activate gnn-experiments" Enter
        ok "$sess — created"
    fi
done

# Pipeline session: add squeue watch pane
if ! tmux list-panes -t pipeline 2>/dev/null | grep -q "^1:"; then
    # Only split if there's just one pane
    PANE_COUNT=$(tmux list-panes -t pipeline 2>/dev/null | wc -l)
    if [[ "$PANE_COUNT" -eq 1 ]]; then
        tmux split-window -h -t pipeline -c "$PROJECT_ROOT"
        tmux send-keys -t pipeline:0.1 "watch -n 30 squeue -u \$USER" Enter
        ok "pipeline — added squeue watch pane"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Session summary:${NC}"
printf "  %-12s %s\n" "Session" "Status"
printf "  %-12s %s\n" "-------" "------"
for sess in "${SESSIONS[@]}"; do
    if tmux has-session -t "$sess" 2>/dev/null; then
        printf "  %-12s %s\n" "$sess" "ready"
    else
        printf "  %-12s %s\n" "$sess" "FAILED"
    fi
done

echo ""
echo -e "${BOLD}Attach commands:${NC}"
for sess in "${SESSIONS[@]}"; do
    echo "  tmux attach -t $sess"
done
echo ""

# Auto-attach to claude
info "Attaching to 'claude' session..."
exec tmux attach-session -t claude
