#!/usr/bin/env bash
# Unified pipeline runner for KD-GAT via Snakemake + SLURM.
#
# Usage:
#   scripts/run.sh [train|evaluate|all] [OPTIONS]
#
# Examples:
#   scripts/run.sh train -d hcrl_sa -t teachers -n   # Dry run teachers only
#   scripts/run.sh evaluate -d hcrl_sa,set_01 -y      # Evaluate two datasets
#   scripts/run.sh all -n                              # Dry run full pipeline
set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT"
SNAKEMAKE="/users/PAS2022/rf15/.conda/envs/gnn-experiments/bin/snakemake"
SNAKEFILE="pipeline/Snakefile"
SLURM_PROFILE="profiles/slurm"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }

usage() {
    cat <<EOF
${BOLD}Usage:${NC} $(basename "$0") <command> [OPTIONS]

${BOLD}Commands:${NC}
  train         Run training pipeline (targets: all, teachers, students, students_nokd)
  evaluate      Run evaluation on trained models
  all           Full pipeline: training + evaluation

${BOLD}Options:${NC}
  -d, --datasets <csv>    Comma-separated datasets (default: all from datasets.yaml)
  -t, --target <name>     Training target: all, teachers, students, students_nokd (train only)
  -n, --dry-run           Show what would run without executing
  -y, --yes               Skip confirmation prompt
  -h, --help              Show this help message
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
[[ $# -eq 0 ]] && usage
COMMAND="$1"; shift

DATASETS=""
TARGET="all"
DRY_RUN=false
YES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--datasets) DATASETS="$2"; shift 2 ;;
        -t|--target)   TARGET="$2";   shift 2 ;;
        -n|--dry-run)  DRY_RUN=true;  shift ;;
        -y|--yes)      YES=true;      shift ;;
        -h|--help)     usage ;;
        *)             error "Unknown option: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve Snakemake target
# ---------------------------------------------------------------------------
case "$COMMAND" in
    train)
        case "$TARGET" in
            all)            SNAKE_TARGET="" ;;
            teachers|students|students_nokd) SNAKE_TARGET="$TARGET" ;;
            *) error "Invalid target: $TARGET"; exit 1 ;;
        esac
        ;;
    evaluate)   SNAKE_TARGET="evaluate_all" ;;
    all)        SNAKE_TARGET="evaluate_all" ;;
    *)          error "Unknown command: $COMMAND"; usage ;;
esac

# ---------------------------------------------------------------------------
# Build Snakemake command
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"

SNAKE_ARGS=(-s "$SNAKEFILE" --profile "$SLURM_PROFILE")
[[ -n "$SNAKE_TARGET" ]] && SNAKE_ARGS+=("$SNAKE_TARGET")

if [[ -n "$DATASETS" ]]; then
    IFS=',' read -ra DS_ARRAY <<< "$DATASETS"
    DS_JSON=$(printf ',"%s"' "${DS_ARRAY[@]}")
    DS_JSON="[${DS_JSON:1}]"
    SNAKE_ARGS+=(--config "datasets=$DS_JSON")
    info "Datasets: ${DS_ARRAY[*]}"
else
    info "Datasets: all (from datasets.yaml)"
fi
info "Command: $COMMAND (target: ${SNAKE_TARGET:-default})"

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
echo ""
info "Running dry run..."
echo ""
"$SNAKEMAKE" "${SNAKE_ARGS[@]}" -n 2>&1 || {
    error "Dry run failed. Check Snakefile and configuration."
    exit 1
}

if $DRY_RUN; then
    echo ""
    ok "Dry run complete (no jobs submitted)."
    exit 0
fi

# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------
if ! $YES; then
    echo ""
    echo -e "${YELLOW}Submit these jobs to SLURM?${NC} [y/N] "
    read -r REPLY
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        info "Aborted."
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
echo ""
info "Submitting to SLURM..."
echo ""
"$SNAKEMAKE" "${SNAKE_ARGS[@]}" 2>&1

echo ""
ok "Jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 squeue -u \$USER"
