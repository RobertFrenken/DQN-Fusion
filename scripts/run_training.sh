#!/usr/bin/env bash
# Run the KD-GAT training pipeline via Snakemake + SLURM.
# Targets: teachers, students (with KD), students_nokd (ablation), or all.
set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT="/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT"
CONDA_ENV="/users/PAS2022/rf15/.conda/envs/gnn-experiments"
SNAKEFILE="pipeline/Snakefile"
SLURM_PROFILE="profiles/slurm"
ALL_DATASETS=(hcrl_ch hcrl_sa set_01 set_02 set_03 set_04)
SNAKEMAKE="$CONDA_ENV/bin/snakemake"

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

Run the KD-GAT training pipeline (teachers, students, or both).

${BOLD}Options:${NC}
  -d, --datasets <csv>    Comma-separated datasets (default: all 6)
                          Available: ${ALL_DATASETS[*]}
  -t, --target <name>     Snakemake target rule:
                            teachers      - teacher models only
                            students      - student models with KD
                            students_nokd - student models without KD (ablation)
                            all           - all three (default)
  -n, --dry-run           Show what would run without executing
  -y, --yes               Skip confirmation prompt
  -h, --help              Show this help message

${BOLD}Examples:${NC}
  $(basename "$0") -d hcrl_sa -t teachers -n     # Dry run, one dataset, teachers only
  $(basename "$0") -d hcrl_sa,set_01 -y          # Train all variants, two datasets
  $(basename "$0") -n                             # Dry run, all datasets, all targets
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
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

# Validate target
case "$TARGET" in
    all|teachers|students|students_nokd) ;;
    *) error "Invalid target: $TARGET"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Build Snakemake command
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

SNAKE_ARGS=(-s "$SNAKEFILE" --profile "$SLURM_PROFILE")

# Target rule (must come before --config for Snakemake)
if [[ "$TARGET" != "all" ]]; then
    SNAKE_ARGS+=("$TARGET")
    info "Target: $TARGET"
else
    info "Target: all (teachers + students + students_nokd)"
fi

# Dataset config
if [[ -n "$DATASETS" ]]; then
    # Convert comma-separated to JSON list
    IFS=',' read -ra DS_ARRAY <<< "$DATASETS"
    DS_JSON=$(printf ',"%s"' "${DS_ARRAY[@]}")
    DS_JSON="[${DS_JSON:1}]"
    SNAKE_ARGS+=(--config "datasets=$DS_JSON")
    info "Datasets: ${DS_ARRAY[*]}"
else
    info "Datasets: ${ALL_DATASETS[*]} (all)"
fi

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
info "Submitting jobs to SLURM..."
echo ""
"$SNAKEMAKE" "${SNAKE_ARGS[@]}" 2>&1

echo ""
ok "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 squeue -u \$USER"
echo "  $SNAKEMAKE -s $SNAKEFILE --profile $SLURM_PROFILE --summary"
