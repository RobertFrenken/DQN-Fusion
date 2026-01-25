#!/bin/bash
# Sequential Pipeline Submission for OSC SLURM
# Submits VGAE → Curriculum GAT → Fusion DQN with job dependencies

set -euo pipefail

DATASET=${1:-hcrl_sa}
DRY_RUN=${2:-}

echo "================================================"
echo "🚀 Sequential Training Pipeline Submission"
echo "================================================"
echo "Dataset: $DATASET"
echo ""

# Step 1: Submit VGAE Autoencoder
echo "📦 Step 1: Submitting VGAE Autoencoder..."
if [ "$DRY_RUN" == "--dry-run" ]; then
    python oscjobmanager.py submit "autoencoder_${DATASET}" --dry-run
    VGAE_JOB_ID="12345"  # Fake ID for dry run
else
    VGAE_OUTPUT=$(python oscjobmanager.py submit "autoencoder_${DATASET}")
    VGAE_JOB_ID=$(echo "$VGAE_OUTPUT" | grep "Job ID:" | awk '{print $NF}')
fi
echo "   ✅ VGAE Job ID: $VGAE_JOB_ID"
echo ""

# Step 2: Submit Curriculum GAT (depends on VGAE)
echo "🎓 Step 2: Submitting Curriculum GAT (after VGAE)..."
if [ "$DRY_RUN" == "--dry-run" ]; then
    python oscjobmanager.py submit "curriculum_${DATASET}" --dry-run
    CURRICULUM_JOB_ID="12346"
else
    # Use SLURM dependency to wait for VGAE
    CURRICULUM_OUTPUT=$(sbatch --dependency=afterok:${VGAE_JOB_ID} \
        "$(python oscjobmanager.py submit curriculum_${DATASET} --dry-run | grep 'Script:' | awk '{print $NF}')")
    CURRICULUM_JOB_ID=$(echo "$CURRICULUM_OUTPUT" | awk '{print $NF}')
fi
echo "   ✅ Curriculum GAT Job ID: $CURRICULUM_JOB_ID (depends on $VGAE_JOB_ID)"
echo ""

# Step 3: Submit Fusion DQN (depends on both VGAE and Curriculum GAT)
echo "🔀 Step 3: Submitting Fusion DQN (after VGAE + Curriculum GAT)..."
if [ "$DRY_RUN" == "--dry-run" ]; then
    python oscjobmanager.py submit "fusion_${DATASET}" --dry-run
    FUSION_JOB_ID="12347"
else
    # Fusion needs both VGAE and GAT, so wait for Curriculum GAT (which already waits for VGAE)
    FUSION_OUTPUT=$(sbatch --dependency=afterok:${CURRICULUM_JOB_ID} \
        "$(python oscjobmanager.py submit fusion_${DATASET} --dry-run | grep 'Script:' | awk '{print $NF}')")
    FUSION_JOB_ID=$(echo "$FUSION_OUTPUT" | awk '{print $NF}')
fi
echo "   ✅ Fusion DQN Job ID: $FUSION_JOB_ID (depends on $CURRICULUM_JOB_ID)"
echo ""

echo "================================================"
echo "✅ Pipeline Submitted Successfully!"
echo "================================================"
echo "Job Chain:"
echo "  1. VGAE:       $VGAE_JOB_ID"
echo "  2. Curriculum: $CURRICULUM_JOB_ID → (after $VGAE_JOB_ID)"
echo "  3. Fusion:     $FUSION_JOB_ID → (after $CURRICULUM_JOB_ID)"
echo ""
echo "Monitor with: squeue -u $USER"
echo "================================================"
