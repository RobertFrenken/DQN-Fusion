#!/usr/bin/env bash
# Set up tmux sessions for KD-GAT development.
# Each session = one WaveTerm tab. No splits — WaveTerm handles layout.
#
# Usage:
#   bash scripts/dev/setup_tmux.sh              # Create all sessions
#   bash scripts/dev/setup_tmux.sh -a claude    # Attach to a session
#   bash scripts/dev/setup_tmux.sh -l           # List sessions
set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
SESSION_ORDER=(claude terminal pipeline)

# ---------------------------------------------------------------------------
GREEN='\033[0;32m'; BLUE='\033[0;34m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
ATTACH=""
LIST=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--attach) ATTACH="$2"; shift 2 ;;
        -l|--list)   LIST=true; shift ;;
        -h|--help)
            echo "Usage: $(basename "$0") [-a SESSION | -l | -h]"
            echo "Sessions: ${SESSION_ORDER[*]}"
            echo "Node: $(hostname)"
            exit 0 ;;
        *) error "Unknown: $1"; exit 1 ;;
    esac
done

command -v tmux &>/dev/null || { error "tmux not found"; exit 1; }

if $LIST; then
    tmux list-sessions 2>/dev/null || echo "(no sessions)"
    exit 0
fi

if [[ -n "$ATTACH" ]]; then
    exec tmux attach-session -t "$ATTACH"
fi

# ---------------------------------------------------------------------------
# Create sessions (no splits — one session per WaveTerm tab)
# ---------------------------------------------------------------------------
echo -e "${BOLD}Setting up tmux sessions on $(hostname)${NC}"

for sess in "${SESSION_ORDER[@]}"; do
    if tmux has-session -t "$sess" 2>/dev/null; then
        ok "$sess — exists"
    else
        tmux new-session -d -s "$sess" -c "$PROJECT_ROOT"
        tmux send-keys -t "$sess" "source ~/.bashrc" Enter
        # Pipeline session auto-starts squeue watch
        if [[ "$sess" == "pipeline" ]]; then
            tmux send-keys -t "$sess" "watch -n 30 squeue -u \$USER" Enter
        fi
        ok "$sess — created"
    fi
done

echo ""
echo -e "${BOLD}Attach from WaveTerm tabs:${NC}"
for sess in "${SESSION_ORDER[@]}"; do
    echo "  tmux attach -t $sess"
done
