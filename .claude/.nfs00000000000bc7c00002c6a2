# MAYBE - Useful Historical Context & Reference

**Last Updated**: 2026-01-27
**Purpose**: Information that might be useful later, but not essential right now

---

## Evaluation Framework v1.0 Documentation

### Overview
Complete rehaul completed on 2026-01-27 with real inference implementation.

**Key Components**:
- 13 metric functions (classification, security, threshold-independent)
- Flexible CLI with 11 arguments
- Multiple output formats (CSV, JSON, console)
- 4 built-in ablation studies
- Support for 3 pipelines (teacher, student no-KD, student with-KD)

### Metrics Computed (60+ per model)
- **Classification (8)**: Accuracy, Precision, Recall, F1, Specificity, Balanced Accuracy, MCC, Kappa
- **Security (6)**: FPR, FNR, TPR, TNR, Detection Rate @ FPR thresholds, Confusion Matrix
- **Threshold-Independent (2)**: ROC-AUC, PR-AUC
- **Class Distribution (4)**: Normal/Attack counts, percentages, imbalance ratio

### Usage Examples
```bash
# Single model evaluation
python -m src.evaluation.evaluation \
  --dataset hcrl_sa \
  --model-path saved_models/gat_student.pth \
  --training-mode normal \
  --mode standard \
  --csv-output results.csv \
  --json-output results.json

# KD comparison ablation
python -m src.evaluation.ablation \
  --study kd \
  --model-list models_kd.json \
  --output-dir results/
```

### Files
- [src/evaluation/metrics.py](../src/evaluation/metrics.py) - 560 lines, 13 functions
- [src/evaluation/evaluation.py](../src/evaluation/evaluation.py) - 527 lines, modular design
- [src/evaluation/ablation.py](../src/evaluation/ablation.py) - 400+ lines, 4 study types

**Reference**: session_notes/EVALUATION_1_0_RELEASE.md

---

## DQN State Space Evolution

### Historical Context: 2D → 15D Journey

**Original 2D State** (Limited):
```python
state = [anomaly_score, gat_prob]  # Only 2 dimensions
```

**Enhancement Analysis**:
| Option | Dimensions | Effort | Impact | Status |
|--------|------------|--------|--------|--------|
| Separate VGAE errors | +2 | Low | Medium | ✅ DONE |
| GAT full logits | +1 | Low | High | ✅ DONE |
| Confidence indicators | +2 | Low | High | ✅ DONE |
| VGAE latent summary | +4 | Medium | High | ✅ DONE |
| GAT embeddings | +4 | Medium | High | ✅ DONE |

**Result**: 2D → 7D → 11D → 15D progressive enhancement

### Why 15D is Better
- **Richer decision-making**: DQN has access to error breakdowns, latent space patterns, and confidence metrics
- **Attack-type awareness**: Different attacks show different error patterns
- **Context-dependent fusion**: Can learn when to trust each model
- **Expected improvement**: +6-10% accuracy vs 2D baseline

**Reference**: session_notes/DQN_STATE_SPACE_ANALYSIS.md

---

## GPU Monitoring Best Practices

### Setup in SLURM Scripts
```bash
# Start monitoring in background
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 2 > gpu_monitor_${SLURM_JOB_ID}.csv &
MONITOR_PID=$!

# ... run training ...

# Stop monitoring
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null
```

### Analysis Script Usage
```bash
python analyze_gpu_monitor.py gpu_monitor_43977817.csv
```

**Output Interpretation**:
- **Peak Memory < 50%**: Underutilized, can increase batch size
- **GPU Util < 50% for >30% of time**: Data loading bound, increase num_workers
- **Memory growth >50 MiB/step**: Potential memory leak

### What to Look For
1. **Memory Usage**: Should plateau after model loads, not grow each epoch
2. **GPU Utilization**: Target 70-85% for compute-bound workload
3. **Memory vs Compute**: Balanced utilization indicates good tuning

**Reference**: session_notes/GPU_MONITORING_GUIDE.md

---

## Inference Implementation Details

### VGAE Inference (Autoencoder Mode)
```
Input: PyTorch Geometric graph batch
  ↓
Forward pass: VGAE network
  ↓
Compute MSE reconstruction error for continuous features
  ↓
Threshold at median error:
  - error > median → prediction=1 (attack)
  - error ≤ median → prediction=0 (normal)
  ↓
Output: (binary predictions, confidence scores)
```

### GAT Inference (Normal/Curriculum Mode)
```
Input: PyTorch Geometric graph batch
  ↓
Forward pass: GAT network → logits [N, 2]
  ↓
Apply softmax to get probabilities
  ↓
Predictions: argmax(logits, dim=1)
Confidence: max(softmax(logits), dim=1)
  ↓
Output: (class predictions {0,1}, softmax confidence [0,1])
```

### DQN Fusion Inference (Simplified)
Currently simplified approach. Full implementation requires:
1. Load VGAE model separately
2. Load GAT model separately
3. Extract 15D state features from both
4. Use DQN to select fusion weight alpha
5. Return fused predictions

**Reference**: session_notes/INFERENCE_IMPLEMENTATION.md

---

## Batch Size Optimization History

### Bug: Double Safety Factor Application
**Discovered**: 2026-01-27 in test job 43977817
**Issue**: Safety factor (0.55) applied twice:
1. First in `_optimize_batch_size()`: tuned_bs × 0.55
2. Again in `_train_standard()`: adjusted_bs × 0.55
**Result**: Final batch size was 0.55² = 0.3025 (~30%) of intended

**Fix**: Removed redundant application in trainer.py

**Impact**:
- Before: Batch size 2,354 (GPU memory 13.9%)
- After: Batch size 4,280 (GPU memory 22.9%)
- Improvement: 1.82x larger batches, 3x faster training

**Reference**: session_notes/BUGFIX_SUMMARY.md (run counter bug), MASTER_TASK_LIST.md (double safety factor)

---

## Run Counter Implementation

### Bug: AttributeError on self.paths
**Issue**: trainer.py referenced `self.paths` but initialized `self.path_resolver`
**Fix**: Changed all 3 references from `self.paths` to `self.path_resolver`
**Impact**: Critical - prevented any training from starting
**Status**: ✅ FIXED

**Reference**: session_notes/BUGFIX_SUMMARY.md

---

## Knowledge Distillation Implementation Notes

### Model Sizing Strategy
- **Teacher models**: Full-size (baseline, no KD)
- **Student models**: Compressed (10% of teacher parameters)
- **With-KD**: Student trained with teacher supervision
- **No-KD**: Student trained without teacher

### KD Safety Factors
Knowledge distillation requires more memory:
- Base safety factor: 0.55
- KD multiplier: 0.75
- Effective KD safety: 0.55 × 0.75 = 0.4125

**Location**: [config/batch_size_factors.json](../config/batch_size_factors.json)

---

## Evaluation Test Plans

### Phase 2: Model Loading Validation
- Verify teacher model files exist
- Test model loading robustness (no corruption)
- Test dataset loading for set_01
- Test data preprocessing pipeline
- Ensure no OOM errors

### Phase 3: End-to-End Evaluation
**Datasets**: hcrl_sa, hcrl_ch, set_01, set_02, set_03, set_04
**Models**: VGAE teacher, GAT teacher, DQN teacher
**Expected Runtime**: 45 minutes (3 runs × 15 min each)
**Outputs**: CSV (wide format), JSON (nested structure), console (formatted)

**Reference**: session_notes/EVALUATION_TEST_PLAN.md, session_notes/EVALUATION_PHASE1_REVIEW.md

---

## Frozen Config Pattern (Legacy)

**Note**: This refers to an older CLI workflow (can-train) that may not match current pipeline.

### Concept
Configuration resolved once at job submission, saved as JSON for reproducibility.

### Structure
```
experimentruns/{modality}/{dataset}/{learning_type}/{model}/{model_size}/{distillation}/{mode}/
└── configs/
    └── frozen_config_{timestamp}.json
```

### Usage
```bash
python train_with_hydra_zen.py --frozen-config /path/to/config.json
```

**Status**: Infrastructure exists but may need updates for current workflow

**Reference**: .claude/FROZEN_CONFIG_ROADMAP.md, .claude/PIPELINE_INVESTIGATION.md

---

## Lessons Learned

### Always Verify Attribute Names
The run counter bug (self.paths vs self.path_resolver) showed the importance of checking attribute consistency between initialization and usage.

### Safety Factors Can Stack
The double safety factor bug revealed that applying conservative safety margins multiple times can compound unexpectedly. Always check the full chain.

### Per-Graph Statistics > Global Statistics
For DQN state space, per-graph aggregations provide much more information than fixed global statistics.

### Visualization Infrastructure First
Building reusable utilities (paper_style.mplstyle, utils.py) before implementing individual figures saves time and ensures consistency.

---

## Useful Code Patterns

### Loading Evaluation Results
```python
from visualizations.utils import load_evaluation_results

results = load_evaluation_results('test_results/')
vgae_df = results['vgae_teacher_set01']
```

### Creating Publication Figures
```python
from visualizations.utils import setup_figure, save_figure, get_color_palette

fig, ax = setup_figure(width=7, height=4.5)
colors = get_color_palette('colorblind')

# ... plotting code ...

save_figure(fig, 'my_figure', output_dir='figures/', formats=['pdf', 'png'])
```

### Computing Confidence Intervals
```python
from visualizations.utils import compute_confidence_intervals

mean, lower, upper = compute_confidence_intervals(data, confidence=0.95)
```

---

## Where Things Live

### Experiment Outputs
```
experimentruns/automotive/{dataset}/
├── unsupervised/vgae/teacher/no_distillation/autoencoder/
│   ├── configs/
│   ├── slurm_logs/
│   ├── checkpoints/
│   └── models/
├── supervised/gat/teacher/no_distillation/curriculum/
│   └── ...
└── rl_fusion/dqn/teacher/no_distillation/fusion/
    └── ...
```

### Evaluation Results
```
test_results/
├── vgae_teacher_set01.csv
├── vgae_teacher_set01.json
├── gat_teacher_set01.csv
└── ...
```

### Visualizations
```
figures/
├── fig1_architecture.pdf
├── fig2_embeddings.pdf
├── ...
└── supplementary/
    └── ...
```

---

**Usage**: Reference this file when you need historical context, implementation details, or examples of completed work.
