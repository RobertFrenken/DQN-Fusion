# Evaluation Framework for CAN-Graph Models

Complete evaluation framework for comprehensive model assessment with LaTeX paper integration support.

**Status**: ✅ Version 1.0 - Complete Rehaul Ready

---

## Quick Start

### Single Model Evaluation

```bash
# Evaluate student model without KD on hcrl_sa dataset
python -m src.evaluation.evaluation \
  --dataset hcrl_sa \
  --model-path saved_models/gat_student.pth \
  --training-mode normal \
  --mode standard \
  --csv-output results.csv \
  --json-output results.json
```

### Ablation Study (Compare Models)

```bash
# Create model configuration file (models_kd.json)
# Then run KD impact ablation study
python -m src.evaluation.ablation \
  --study kd \
  --model-list models_kd.json \
  --output-dir ablation_results/
```

---

## Framework Architecture

### Three Files

#### 1. **src/evaluation/metrics.py** (Core Metrics Module)

Computes comprehensive metrics for binary classification (normal vs attack):

**Functions:**
- `compute_classification_metrics()` - Accuracy, Precision, Recall, F1, Specificity, MCC, Kappa
- `compute_security_metrics()` - FPR, FNR, TPR, TNR, Detection Rate, Miss Rate
- `compute_threshold_independent_metrics()` - ROC-AUC, PR-AUC
- `detect_optimal_threshold()` - Find optimal decision boundary that maximizes F1/Balanced Acc/Youden
- `compute_detection_rate_at_fpr()` - Security-focused: TPR at FPR thresholds (5%, 1%, 0.1%)
- `compute_all_metrics()` - One-call comprehensive metric computation
- `flatten_metrics()` - Flatten nested dict for CSV export
- `compute_class_distribution()` - Class balance statistics

**Usage Example:**
```python
from src.evaluation.metrics import compute_all_metrics

# Compute all metrics
metrics = compute_all_metrics(y_true, y_pred, y_scores)
# Returns: {
#   'classification': {...},
#   'security': {...},
#   'confusion': {...},
#   'class_distribution': {...},
#   'threshold_independent': {...},
#   'detection_at_fpr': {...}
# }
```

#### 2. **src/evaluation/evaluation.py** (Main Evaluation Pipeline)

Complete evaluation with flexible CLI arguments and multiple output formats.

**Core Classes:**

- `EvaluationConfig` - Configuration validation and logging
- `ModelLoader` - Load PyTorch models and Lightning checkpoints
- `DatasetHandler` - Load train/val/test splits (dataset-specific logic)
- `Evaluator` - Main pipeline: inference → metrics → export

**CLI Arguments:**

```
REQUIRED:
  --dataset {hcrl_sa|hcrl_ch|set_01|set_02|set_03|set_04}
  --model-path <path_to_model.pth>
  --training-mode {normal|autoencoder|curriculum|fusion}
  --mode {standard|with-kd}

OPTIONAL:
  --teacher-path <path_to_teacher.pth>
  --batch-size 512
  --device {cuda|cpu|auto}
  --csv-output evaluation_results.csv
  --json-output evaluation_results.json
  --plots-dir evaluation_plots/
  --threshold-optimization true
  --verbose
```

**Output Formats:**

1. **Wide CSV** (for LaTeX tables)
   ```
   dataset,subset,model,training_mode,kd_mode,classification_accuracy,classification_precision,...
   hcrl_sa,train,gat_student,normal,standard,0.9823,0.9645,...
   ```

2. **JSON Summary**
   ```json
   {
     "metadata": {...},
     "results": {
       "train": {...},
       "val": {...},
       "test": {...}
     },
     "threshold_optimization": {...}
   }
   ```

#### 3. **src/evaluation/ablation.py** (Ablation Study Framework)

Compare multiple models to isolate impact of design choices.

**Available Studies:**

| Study | Purpose | Compares |
|-------|---------|----------|
| `kd` | KD Impact | With-KD vs Without-KD |
| `curriculum` | Curriculum Learning | Curriculum vs Standard |
| `fusion` | Fusion Strategy | DQN vs Static |
| `training_mode` | Training Mode | Normal vs Autoencoder vs Curriculum |

**Core Classes:**
- `AblationStudy` - Base class (run multiple evaluations, compute deltas)
- `KDImpactStudy` - Knowledge distillation comparison
- `CurriculumImpactStudy` - Curriculum learning comparison
- `FusionImpactStudy` - Fusion strategy comparison
- `TrainingModeImpactStudy` - Training mode comparison

---

## Usage Examples

### Example 1: Evaluate Student Model (No-KD)

```bash
python -m src.evaluation.evaluation \
  --dataset hcrl_sa \
  --model-path saved_models/gat_student_hcrl_sa.pth \
  --training-mode normal \
  --mode standard \
  --csv-output student_no_kd_results.csv \
  --json-output student_no_kd_results.json
```

**Output:**
- `student_no_kd_results.csv` - Wide format for LaTeX
- `student_no_kd_results.json` - Full metrics summary
- Console output with train/val/test metrics

---

### Example 2: Evaluate Student Model (With-KD)

```bash
python -m src.evaluation.evaluation \
  --dataset hcrl_sa \
  --model-path saved_models/gat_student_with_kd_hcrl_sa.pth \
  --teacher-path saved_models/gat_teacher_hcrl_sa.pth \
  --training-mode normal \
  --mode with-kd \
  --csv-output student_with_kd_results.csv \
  --json-output student_with_kd_results.json
```

---

### Example 3: Run KD Impact Ablation Study

**Step 1: Create model config file (models_kd.json):**
```json
[
  {
    "name": "gat_student_no_kd",
    "dataset": "hcrl_sa",
    "model_path": "saved_models/gat_student_hcrl_sa.pth",
    "teacher_path": null,
    "training_mode": "normal",
    "kd_mode": "standard",
    "description": "Student model without knowledge distillation"
  },
  {
    "name": "gat_student_with_kd",
    "dataset": "hcrl_sa",
    "model_path": "saved_models/gat_student_with_kd_hcrl_sa.pth",
    "teacher_path": "saved_models/gat_teacher_hcrl_sa.pth",
    "training_mode": "normal",
    "kd_mode": "with-kd",
    "description": "Student model with knowledge distillation"
  }
]
```

**Step 2: Run ablation:**
```bash
python -m src.evaluation.ablation \
  --study kd \
  --model-list models_kd.json \
  --output-dir ablation_results/
```

**Output:**
- `ablation_results/kd_impact_ablation.csv` - Ablation results
- `ablation_results/kd_impact_gat_student_no_kd_results.json` - Individual results
- `ablation_results/kd_impact_gat_student_with_kd_results.json` - Individual results

**Ablation CSV format:**
```
study,model_a,model_b,metric,value_a,value_b,delta,pct_change,winner
kd_impact,gat_student_no_kd,gat_student_with_kd,f1,0.9156,0.9324,+0.0168,+1.83%,gat_student_with_kd
```

---

### Example 4: Run Curriculum Impact Ablation Study

**Step 1: Create model config (models_curriculum.json):**
```json
[
  {
    "name": "gat_standard",
    "dataset": "hcrl_sa",
    "model_path": "saved_models/gat_normal_hcrl_sa.pth",
    "training_mode": "normal",
    "kd_mode": "standard"
  },
  {
    "name": "gat_curriculum",
    "dataset": "hcrl_sa",
    "model_path": "saved_models/gat_curriculum_hcrl_sa.pth",
    "training_mode": "curriculum",
    "kd_mode": "standard"
  }
]
```

**Step 2: Run:**
```bash
python -m src.evaluation.ablation \
  --study curriculum \
  --model-list models_curriculum.json \
  --output-dir ablation_results/
```

---

## Metrics Explained

### Classification Metrics (Binary)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN) - % of attacks caught
- **Specificity**: TN / (TN + FP) - % of normal samples correctly identified
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Balanced Accuracy**: (Recall + Specificity) / 2
- **MCC** (Matthews Correlation Coefficient): Correlation between predicted and actual
- **Kappa** (Cohen's Kappa): Agreement accounting for chance

### Security Metrics (Intrusion Detection Focus)
- **FPR** (False Positive Rate): FP / (FP + TN) - % false alarms
- **FNR** (False Negative Rate): FN / (FN + TP) - % missed attacks
- **TPR** (True Positive Rate): TP / (TP + FN) - % detected attacks (= Recall)
- **TNR** (True Negative Rate): TN / (TN + FP) - % normal samples correctly identified (= Specificity)
- **Detection Rate at FPR**: How many attacks detected while keeping false alarms < 5% / 1% / 0.1%

### Threshold-Independent Metrics
- **ROC-AUC**: Area under Receiver Operating Characteristic curve (threshold-independent)
- **PR-AUC**: Area under Precision-Recall curve

### Class Distribution
- **Imbalance Ratio**: Normal : Attack (e.g., 11.7:1 means heavily imbalanced)
- **Normal Count**: Number of normal samples
- **Attack Count**: Number of attack samples

---

## Data Splits

**Train/Val/Test Strategy:**

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| hcrl_sa | train_01_attack_free + train_02_with_attacks | 20% from train | test_01_known_vehicle_known_attack |
| hcrl_ch | All train_* folders | 20% from train | All test_*.csv |
| set_0x | train_* folders | 20% from train | test_01_known_vehicle_known_attack |

- **Train**: Combined from all train_* folders
- **Val**: 20% random split from training (stratified by class)
- **Test**: Dataset-specific test folder (no training data leak)

---

## CSV Export for LaTeX

### Wide Format (for Tables)

Perfect for LaTeX `tabular` environment:

```latex
\begin{table}
\begin{tabular}{lllll}
\hline
Dataset & Subset & Model & Accuracy & F1 \\
\hline
hcrl_sa & test & student_no_kd & 0.9801 & 0.9283 \\
hcrl_sa & test & student_with_kd & 0.9867 & 0.9412 \\
\hline
\end{tabular}
\end{table}
```

**Python to generate from CSV:**
```python
import pandas as pd

df = pd.read_csv('evaluation_results.csv')

# Filter to test set only
test_df = df[df['subset'] == 'test'][['model', 'accuracy', 'precision', 'recall', 'f1']]

# Export to LaTeX
print(test_df.to_latex(index=False))
```

---

## Integration with Training

### After Training, Before Evaluation

1. **Save trained model:**
   ```python
   torch.save(model.state_dict(), 'saved_models/gat_student.pth')
   ```

2. **Run evaluation:**
   ```bash
   python -m src.evaluation.evaluation \
     --dataset hcrl_sa \
     --model-path saved_models/gat_student.pth \
     --training-mode normal \
     --mode standard \
     --csv-output results.csv
   ```

3. **Import results into paper:**
   ```latex
   \input{results.csv}  % After converting to LaTeX table
   ```

---

## Performance Considerations

- **Batch Size**: Default 512 for inference (adjust down if OOM)
- **Device**: Auto-selects GPU if available
- **Memory**: ~2-4 GB for full evaluation on large datasets
- **Time**: ~5-15 minutes per model depending on dataset size

---

## Troubleshooting

### Issue: "Model not found"
**Solution**: Verify model path is correct and file exists
```bash
ls -la saved_models/gat_student.pth
```

### Issue: "Dataset folder not found"
**Solution**: Check data/automotive/{dataset} exists and has correct structure
```bash
ls data/automotive/hcrl_sa/
```

### Issue: "Out of memory"
**Solution**: Reduce batch size
```bash
python -m src.evaluation.evaluation ... --batch-size 256
```

### Issue: No metrics in CSV
**Solution**: Check that inference is returning valid predictions/scores
- Verify model loads correctly
- Check dataset loading for errors

---

## Next Steps

### 1. For Paper Integration

```bash
# Evaluate all model variants
python -m src.evaluation.evaluation --dataset hcrl_sa --model-path student_no_kd.pth --mode standard --training-mode normal --csv-output paper_results.csv

# Run ablation studies
python -m src.evaluation.ablation --study kd --model-list models_kd.json --output-dir results/
python -m src.evaluation.ablation --study curriculum --model-list models_curriculum.json --output-dir results/

# Collect all CSVs → Create paper tables
```

### 2. For Statistical Analysis

```python
import pandas as pd

results_csv = pd.read_csv('evaluation_results.csv')

# Compare models
comparison = results_csv.pivot_table(
    index='model',
    columns='metric',
    values='value'
)

# Print improvement from KD
kd_improvement = (
    results_csv[results_csv['model'] == 'with_kd']['f1'].values[0] -
    results_csv[results_csv['model'] == 'no_kd']['f1'].values[0]
)
print(f"KD Improvement: {kd_improvement:.4f}")
```

### 3. For Visualization

```python
import matplotlib.pyplot as plt

# Plot ROC curves
from sklearn.metrics import roc_curve, auc

# Load results and plot comparison across models
```

---

## File Locations

| Component | File |
|-----------|------|
| Metrics | `src/evaluation/metrics.py` |
| Evaluation | `src/evaluation/evaluation.py` |
| Ablation | `src/evaluation/ablation.py` |
| Plan | `EVALUATION_IMPLEMENTATION_PLAN.md` |
| This Guide | `EVALUATION_FRAMEWORK_README.md` |

---

## Version History

- **v1.0** (2026-01-27): Complete rehaul with comprehensive CLI, metrics module, and ablation framework
- **v0.x** (old): Original evaluation.py (archived)

---

**Happy evaluating! 🚀**
