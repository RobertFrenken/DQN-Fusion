# Evaluation Framework Implementation Plan

**Date**: 2026-01-27
**Status**: PLANNING
**Scope**: Complete rehaul of evaluation.py and creation of ablation.py

---

## 1. Design Overview

### File Structure
```
src/evaluation/
├── evaluation.py          # Main evaluation pipeline (complete rehaul)
├── ablation.py           # Ablation study framework (NEW)
├── metrics.py            # Shared metrics computation (NEW)
└── __init__.py
```

### Three Primary Pipelines Supported
1. **Teacher**: Full-sized models trained without distillation
2. **Student No-KD**: Compressed models without knowledge distillation
3. **Student With-KD**: Compressed models trained with teacher knowledge distillation

### Data Handling Strategy
```
train/: Combine all train_* folders
    ├── train_01_attack_free/
    ├── train_02_with_attacks/
    └── ...
    → Create train/val split (80/20)

test/: Dataset-specific
    - hcrl_sa: test_01_known_vehicle_known_attack
    - hcrl_ch: glob all test_*.csv files
    - set_0x: test_01_known_vehicle_known_attack (standard)
```

---

## 2. CLI Arguments

### Main Command
```bash
python evaluate.py \
    --dataset {hcrl_sa|hcrl_ch|set_01|set_02|set_03|set_04} \
    --model-path <path_to_model> \
    --teacher-path <path_to_teacher> [optional] \
    --training-mode {normal|autoencoder|curriculum|fusion} \
    --mode {standard|with-kd} \
    --batch-size 512 \
    --device cuda \
    --csv-output results.csv \
    --json-output results.json \
    --plots-dir plots/ \
    --threshold-optimization true \
    --verbose true
```

### Arguments Details

| Argument | Type | Required | Description | Example |
|----------|------|----------|-------------|---------|
| `--dataset` | str | Yes | Dataset to evaluate on | `hcrl_sa` |
| `--model-path` | str | Yes | Path to primary model (student or teacher) | `saved_models/gat_student_hcrl_sa.pth` |
| `--teacher-path` | str | No | Path to teacher model (for KD evaluation or teacher baseline) | `saved_models/gat_teacher_hcrl_sa.pth` |
| `--training-mode` | str | Yes | Mode the model was trained with | `autoencoder`, `fusion`, `curriculum`, `normal` |
| `--mode` | str | Yes | KD status | `standard` (no-kd) or `with-kd` |
| `--batch-size` | int | No (default: 512) | Evaluation batch size | `256`, `1024` |
| `--device` | str | No (default: auto) | Device to use | `cuda`, `cpu` |
| `--csv-output` | str | No | CSV output path | `eval_results.csv` |
| `--json-output` | str | No | JSON output path | `eval_results.json` |
| `--plots-dir` | str | No | Directory to save metric plots | `plots/` |
| `--threshold-optimization` | bool | No (default: true) | Whether to optimize detection threshold | `true` |
| `--verbose` | bool | No (default: false) | Print detailed progress | `true` |

---

## 3. Metrics Computation

### Phase 1: Per-Subset Metrics (Train/Val/Test)

For each subset, compute:

**Classification Metrics:**
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- Balanced Accuracy = (Recall + Specificity) / 2
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa

**Threshold-Dependent Metrics:**
- AUC-ROC
- AUC-PR
- Confusion Matrix (TP, TN, FP, FN)

**Security-Focused Metrics:**
- Detection Rate at False Positive Rate (FPR) thresholds:
  - @ FPR < 5% (TPR when FPR = 0.05)
  - @ FPR < 1% (TPR when FPR = 0.01)
  - @ FPR < 0.1% (TPR when FPR = 0.001)
- False Positive Rate (FPR) = FP / (FP + TN)
- False Negative Rate (FNR) = FN / (FN + TP)
- Miss Rate = FNR (% of attacks missed)

**Dataset Characteristics:**
- Total samples
- Normal samples (class 0) count and %
- Attack samples (class 1) count and %
- Class imbalance ratio (normal:attack)

### Phase 2: Threshold Optimization

For each model/subset combination:
1. Compute anomaly scores on validation set
2. Search for optimal threshold that maximizes F1-score
3. Store optimal_threshold for later test set evaluation
4. Report: original threshold vs optimized threshold metrics

### Phase 3: Multi-Method Evaluation (Training Mode Dependent)

**If training_mode = "normal":**
- Direct classification predictions from model

**If training_mode = "autoencoder":**
- Anomaly score-based detection (reconstruction error)
- Optional classifier fusion (if classifier available)
- Two-stage predictions (anomaly filter + classifier)

**If training_mode = "curriculum":**
- Same as normal (curriculum affects training, not inference)

**If training_mode = "fusion":**
- DQN-optimized fusion (learned alpha blending)
- Static fusion (0.5 weight each)
- Individual component metrics (anomaly-only, classifier-only)

---

## 4. Output Formats

### A. Wide CSV (One Row Per Model)
```csv
model,dataset,subset,accuracy,precision,recall,f1,auc_roc,auc_pr,specificity,balanced_accuracy,mcc,kappa,fpr_at_5pct,fnr,samples,normal_count,attack_count,class_imbalance_ratio
gat_student_no_kd,hcrl_sa,train,0.9823,0.9645,0.8932,0.9273,0.9812,0.9456,0.9912,0.9422,0.8734,0.8321,0.0234,0.0168,4595,4234,361,11.7:1
gat_student_no_kd,hcrl_sa,val,0.9734,0.9512,0.8834,0.9156,0.9723,0.9324,0.9845,0.9340,0.8423,0.7998,0.0456,0.0234,920,850,70,12.1:1
gat_student_no_kd,hcrl_sa,test,0.9801,0.9567,0.9012,0.9283,0.9834,0.9534,0.9876,0.9444,0.8856,0.8567,0.0312,0.0145,780,720,60,12.0:1
```

### B. Long CSV (One Row Per Metric)
```csv
model,dataset,subset,metric,value,metric_type
gat_student_no_kd,hcrl_sa,train,accuracy,0.9823,classification
gat_student_no_kd,hcrl_sa,train,precision,0.9645,classification
gat_student_no_kd,hcrl_sa,train,recall,0.8932,classification
gat_student_no_kd,hcrl_sa,train,f1,0.9273,classification
gat_student_no_kd,hcrl_sa,train,auc_roc,0.9812,threshold_independent
gat_student_no_kd,hcrl_sa,train,specificity,0.9912,classification
gat_student_no_kd,hcrl_sa,train,balanced_accuracy,0.9422,classification
...
```

### C. Summary JSON
```json
{
  "evaluation_metadata": {
    "timestamp": "2026-01-27T15:30:00",
    "dataset": "hcrl_sa",
    "model_path": "saved_models/gat_student_no_kd.pth",
    "training_mode": "normal",
    "kd_mode": "standard",
    "total_samples": {
      "train": 4595,
      "val": 920,
      "test": 780
    }
  },
  "results": {
    "train": {
      "accuracy": 0.9823,
      "precision": 0.9645,
      ...
    },
    "val": {...},
    "test": {...}
  },
  "threshold_optimization": {
    "optimal_threshold": 0.4234,
    "original_threshold": 0.5,
    "improvement_f1": 0.0234
  },
  "class_distribution": {
    "train": {"normal": 4234, "attack": 361, "imbalance_ratio": "11.7:1"},
    "val": {"normal": 850, "attack": 70, "imbalance_ratio": "12.1:1"},
    "test": {"normal": 720, "attack": 60, "imbalance_ratio": "12.0:1"}
  }
}
```

### D. Plots (Optional)
- ROC curves (per subset)
- PR curves (per subset)
- Confusion matrices (heatmaps)
- Metric comparison bar charts (train/val/test)
- Threshold optimization visualization

---

## 5. Ablation Framework (ablation.py)

### Purpose
Compare multiple trained models across dimensions to isolate the impact of specific design choices.

### Ablation Studies

#### **Study 1: Knowledge Distillation Impact**
Compares KD vs No-KD
```
Model A: Student No-KD    (--model-path student_no_kd.pth --mode standard)
Model B: Student With-KD  (--model-path student_with_kd.pth --mode with-kd)
Compute: Δ = metrics(B) - metrics(A)
Report: Which metrics improve with KD? By how much?
```

#### **Study 2: Curriculum Learning Impact**
Compares Curriculum vs All-Samples
```
Model A: Standard (normal training on 1:1 normal:attack mix)
Model B: Curriculum (progressive 1:1 → 10:1 ratio)
Compute: Δ = metrics(B) - metrics(A)
Report: Does curriculum help? Which metrics improve?
```

#### **Study 3: Fusion Strategy Impact (DQN vs Static)**
Compares learned fusion vs static blending
```
Model A: Static Fusion (0.5 weight anomaly + 0.5 weight classifier)
Model B: DQN Fusion (learned alpha values per sample)
Compute: Δ = metrics(B) - metrics(A)
Report: Does learned fusion beat static? By how much?
```

#### **Study 4: Training Mode Impact**
Compares different training modes on same model architecture
```
Model A: Normal (supervised classification)
Model B: Autoencoder (unsupervised VGAE)
Model C: Curriculum (hard sample mining)
Compute: metrics for each
Report: Which mode best?
```

### Ablation CSV Output
```csv
study,model_a,model_b,dataset,subset,metric,value_a,value_b,delta,percent_change,winner
kd_impact,student_no_kd,student_with_kd,hcrl_sa,test,f1,0.9156,0.9324,+0.0168,+1.83%,with_kd
kd_impact,student_no_kd,student_with_kd,hcrl_sa,test,auc_roc,0.9723,0.9834,+0.0111,+1.14%,with_kd
curriculum_impact,normal,curriculum,hcrl_sa,test,f1,0.9234,0.9283,+0.0049,+0.53%,curriculum
curriculum_impact,normal,curriculum,hcrl_sa,test,recall,0.8923,0.9012,+0.0089,+0.99%,curriculum
fusion_impact,static_fusion,dqn_fusion,hcrl_sa,test,f1,0.9234,0.9456,+0.0222,+2.40%,dqn_fusion
```

### Ablation JSON Summary
```json
{
  "ablation_studies": {
    "kd_impact": {
      "description": "Knowledge Distillation: Student With-KD vs Student No-KD",
      "winner": "with_kd",
      "avg_improvement": {
        "f1": "+1.83%",
        "auc_roc": "+1.14%",
        "balanced_accuracy": "+1.56%"
      },
      "datasets_tested": ["hcrl_sa", "set_01"]
    },
    "curriculum_impact": {...},
    "fusion_impact": {...}
  },
  "summary_table": {
    "best_overall_model": "student_with_kd_curriculum",
    "metric_leaders": {
      "f1": {"model": "dqn_fusion", "value": 0.9456},
      "auc_roc": {"model": "student_with_kd_curriculum", "value": 0.9867},
      "specificity": {"model": "student_no_kd", "value": 0.9912}
    }
  }
}
```

---

## 6. Implementation Steps

### Phase 1: Core Metrics Module (metrics.py)
- [ ] `compute_classification_metrics()` - standard metrics
- [ ] `compute_security_metrics()` - FPR, FNR, detection rates
- [ ] `compute_threshold_independent_metrics()` - AUC-ROC, AUC-PR
- [ ] `optimize_threshold()` - search optimal decision boundary
- [ ] `format_confusion_matrix()` - pretty print/visualize

### Phase 2: Main Evaluation Pipeline (evaluation.py)
- [ ] Parse CLI arguments with argparse
- [ ] Load dataset (train + val + test)
- [ ] Load model(s) with correct architecture
- [ ] Inference on all three subsets
- [ ] Compute metrics per subset
- [ ] Export wide CSV, long CSV, JSON
- [ ] Generate plots (optional)
- [ ] Print summary to console

### Phase 3: Ablation Framework (ablation.py)
- [ ] Define ablation studies (KD, curriculum, fusion, mode)
- [ ] Run multiple models
- [ ] Compute delta metrics
- [ ] Generate ablation CSV + JSON
- [ ] Statistical significance testing (optional)

### Phase 4: Polish & Documentation
- [ ] Add docstrings
- [ ] Add error handling
- [ ] Create evaluation workflow README
- [ ] Example usage scripts

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Explicit model paths (Option C)** | Flexibility: user controls exactly which models to load, no assumptions about naming |
| **Manual training-mode flag** | Some models may be trained with one mode but need to be evaluated as another (edge cases) |
| **Separate ablation.py** | Clean separation of concerns; evaluation.py handles single-model eval, ablation.py handles comparisons |
| **Both CSV formats** | Wide CSV for LaTeX tables, long CSV for plotting and downstream analysis |
| **Threshold optimization included** | Critical for intrusion detection: optimal threshold often differs from 0.5 |
| **Class distribution in output** | Context for metrics: high accuracy on imbalanced data is misleading without imbalance ratio |
| **Security-focused metrics** | FPR/FNR/detection-at-low-FPR are key for security applications |

---

## 8. Integration with Existing Code

### Reuses from Current evaluation.py
- `ComprehensiveEvaluationPipeline` class (refactor into new architecture)
- `compute_comprehensive_metrics()` function (migrate to metrics.py)
- Data loading patterns from `graph_creation()` (reuse)
- Model creation helpers (reuse)

### New Dependencies (if any)
- `scikit-learn`: Already imported, extended metrics
- `matplotlib`: For plot generation
- `pandas`: For CSV export
- `json`: Standard library

---

## 9. Testing Strategy

### Unit Tests
- [ ] Metrics computation correctness
- [ ] CSV export formatting
- [ ] CLI argument parsing

### Integration Tests
- [ ] Full evaluation pipeline on small dataset
- [ ] Ablation study execution
- [ ] LaTeX table generation from CSV

### Manual Tests
- [ ] Run on hcrl_sa + hcrl_ch
- [ ] Compare results with current evaluation.py (sanity check)
- [ ] Verify CSV/JSON quality for LaTeX integration

---

## 10. Timeline Estimate

| Phase | Status | Notes |
|-------|--------|-------|
| Metrics module | Not started | ~1-2 hours |
| Evaluation pipeline | Not started | ~2-3 hours |
| Ablation framework | Not started | ~1-2 hours |
| Testing & polish | Not started | ~1-2 hours |
| **Total** | | **~5-9 hours** |

---

## Ready for Implementation?

✅ **Decisions finalized:**
- CLI structure: Explicit paths (--model-path, --teacher-path, --mode, --training-mode)
- Metrics: All standard + security-focused + threshold optimization
- Outputs: Wide CSV, long CSV, JSON
- Ablations: Separate ablation.py file

**Next step:** Begin implementation starting with metrics.py
