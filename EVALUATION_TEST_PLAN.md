# Evaluation Framework Test Plan

**Date**: 2026-01-27
**Purpose**: Validate evaluation.py before using it for paper integration
**Test Models**: 3 teacher models from set_01 dataset

---

## Phase 1: Protocol & Robustness Review

### Checklist: Compare evaluation.py with train_with_hydra_zen.py

**Reference Protocol from train_with_hydra_zen.py**:
- ✓ Robust argument parsing with help messages
- ✓ Graceful error handling (try/except with informative messages)
- ✓ Path validation before execution
- ✓ Clear logging and status messages
- ✓ Warning suppression for known non-critical issues
- ✓ Fallback/default options for optional parameters

**Review evaluation.py for**:
- [ ] EvaluationConfig validates all required arguments
- [ ] Informative error messages if model not found
- [ ] Informative error messages if dataset not found
- [ ] Graceful handling of missing optional arguments
- [ ] Clear logging throughout pipeline
- [ ] Proper exception handling (try/except in critical sections)
- [ ] Path handling using Path() for cross-platform compatibility

### Critical Code Sections to Verify

**In EvaluationConfig.__init__():**
```python
# Should check:
✓ dataset is valid
✓ model_path exists and is readable
✓ teacher_path exists (if provided)
✓ device is valid (cuda/cpu/auto)
✓ All required args are provided
```

**In ModelLoader.load_model():**
```python
# Should check:
✓ File exists before loading
✓ Proper error if weights_only fails
✓ Clear error message on corruption/format issue
```

**In DatasetHandler.load_datasets():**
```python
# Should check:
✓ Data folder exists
✓ Train/Val split works
✓ Test folder structure is correct
✓ ID mapping built successfully
```

**In Evaluator.evaluate():**
```python
# Should check:
✓ No OOM during inference
✓ Metrics computation doesn't crash
✓ CSV/JSON export works
```

---

## Phase 2: Model Loading & Data Robustness

### Test Data
**Teacher Models for set_01**:

1. **VGAE Teacher (Unsupervised)**
   - Path: `experimentruns/automotive/set_01/unsupervised/vgae/teacher/no_distillation/autoencoder/models/vgae_teacher_autoencoder.pth`
   - Training Mode: `autoencoder`
   - Expected Size: ~1.7 MB (teacher)

2. **GAT Teacher (Supervised with Curriculum)**
   - Path: `experimentruns/automotive/set_01/supervised/gat/teacher/no_distillation/curriculum/models/gat_teacher_curriculum.pth`
   - Training Mode: `curriculum`
   - Expected Size: ~1.1 MB (teacher)

3. **DQN Teacher (RL Fusion)**
   - Path: `experimentruns/automotive/set_01/rl_fusion/dqn/teacher/no_distillation/fusion/models/dqn_teacher_fusion.pth`
   - Training Mode: `fusion`
   - Expected Size: ~687 KB (teacher DQN agent)

### Test Scenarios

#### Test 2.1: Model Loading Robustness
```bash
# Verify each model loads without corruption
for model in vgae_teacher_autoencoder.pth gat_teacher_curriculum.pth dqn_teacher_fusion.pth; do
  python -c "import torch; torch.load('path/to/$model')"
done
```
**Expected**: All models load successfully

#### Test 2.2: Data Loading for set_01
```python
# Check data loading works
from src.preprocessing.preprocessing import build_id_mapping_from_normal, graph_creation

id_mapping = build_id_mapping_from_normal("data/automotive/set_01")
# Expected: Non-empty ID mapping
```
**Expected**: ID mapping with >0 IDs

#### Test 2.3: Train/Val/Test Split
```python
# Verify data loader creates correct splits
train_dataset, val_dataset = ...
# Check ratios, class balance
```
**Expected**: ~80% train, ~20% val, balanced class distribution

---

## Phase 3: End-to-End Evaluation Tests

### Test 3.1: VGAE Teacher Evaluation
```bash
python -m src.evaluation.evaluation \
  --dataset set_01 \
  --model-path experimentruns/automotive/set_01/unsupervised/vgae/teacher/no_distillation/autoencoder/models/vgae_teacher_autoencoder.pth \
  --training-mode autoencoder \
  --mode standard \
  --csv-output test_vgae_teacher.csv \
  --json-output test_vgae_teacher.json \
  --verbose
```

**Expected Results**:
- ✓ No errors during inference
- ✓ Metrics computed successfully
- ✓ CSV file created with proper format
- ✓ JSON file with structured results
- ✓ Console output shows train/val/test metrics

**Validation Criteria**:
- Accuracy in range [0, 1]
- Precision, Recall, F1 all in range [0, 1]
- AUC-ROC in range [0, 1]
- Metrics consistent across train/val/test (val/test should be similar if representative)

### Test 3.2: GAT Teacher Evaluation
```bash
python -m src.evaluation.evaluation \
  --dataset set_01 \
  --model-path experimentruns/automotive/set_01/supervised/gat/teacher/no_distillation/curriculum/models/gat_teacher_curriculum.pth \
  --training-mode curriculum \
  --mode standard \
  --csv-output test_gat_teacher.csv \
  --json-output test_gat_teacher.json
```

**Expected Results**: Same as Test 3.1

### Test 3.3: DQN Teacher Evaluation
```bash
python -m src.evaluation.evaluation \
  --dataset set_01 \
  --model-path experimentruns/automotive/set_01/rl_fusion/dqn/teacher/no_distillation/fusion/models/dqn_teacher_fusion.pth \
  --training-mode fusion \
  --mode standard \
  --csv-output test_dqn_teacher.csv \
  --json-output test_dqn_teacher.json
```

**Expected Results**: Same as Test 3.1

### Test 3.4: CSV Export Format Validation
```python
import pandas as pd

# Load CSVs
vgae_csv = pd.read_csv('test_vgae_teacher.csv')
gat_csv = pd.read_csv('test_gat_teacher.csv')
dqn_csv = pd.read_csv('test_dqn_teacher.csv')

# Check structure
for df, name in [(vgae_csv, 'VGAE'), (gat_csv, 'GAT'), (dqn_csv, 'DQN')]:
    print(f"{name} CSV columns: {df.columns.tolist()}")
    print(f"{name} CSV shape: {df.shape}")
    print(f"{name} Rows: {len(df)} (train + val + test)")
```

**Expected**:
- Wide format: 1 row per subset (3 rows total)
- Columns include: dataset, subset, model, training_mode, kd_mode, all metrics
- No NaN values in numeric columns

### Test 3.5: JSON Export Format Validation
```python
import json

with open('test_vgae_teacher.json') as f:
    data = json.load(f)

# Check structure
print("JSON keys:", data.keys())
print("Metadata:", data['metadata'])
print("Results subsets:", data['results'].keys())
print("Train metrics keys:", data['results']['train'].keys())
```

**Expected**:
- Top-level keys: metadata, results, threshold_optimization
- metadata contains: dataset, model_path, training_mode, kd_mode, timestamp
- results contains: train, val, test
- Each subset has: classification, security, confusion, class_distribution, threshold_independent, detection_at_fpr

---

## Phase 4: Ablation Study Test (Optional)

### Test 4.1: Create Model Config
```json
[
  {
    "name": "vgae_teacher",
    "dataset": "set_01",
    "model_path": "experimentruns/automotive/set_01/unsupervised/vgae/teacher/no_distillation/autoencoder/models/vgae_teacher_autoencoder.pth",
    "training_mode": "autoencoder",
    "kd_mode": "standard",
    "description": "VGAE teacher model for set_01"
  },
  {
    "name": "gat_teacher",
    "dataset": "set_01",
    "model_path": "experimentruns/automotive/set_01/supervised/gat/teacher/no_distillation/curriculum/models/gat_teacher_curriculum.pth",
    "training_mode": "curriculum",
    "kd_mode": "standard",
    "description": "GAT teacher model for set_01"
  }
]
```

### Test 4.2: Run Ablation (Training Mode Comparison)
```bash
python -m src.evaluation.ablation \
  --study training_mode \
  --model-list set_01_teachers.json \
  --output-dir ablation_test/
```

**Expected Results**:
- ablation_test/training_mode_impact_ablation.csv
- Delta metrics comparing autoencoder vs curriculum modes
- Summary showing which mode performs better

---

## Success Criteria

### Critical (Must Pass)
- [ ] All 3 models load without errors
- [ ] Inference completes without OOM
- [ ] Metrics are numerically valid (0-1 range, no NaN)
- [ ] CSV exports with correct structure
- [ ] JSON exports with correct structure
- [ ] Console output is readable and informative

### Important (Should Pass)
- [ ] Threshold optimization completes
- [ ] Metrics are consistent across runs (deterministic)
- [ ] Ablation studies generate meaningful deltas
- [ ] No warnings in output (except known suppressible ones)

### Nice to Have
- [ ] Performance metrics (inference time <5 min for set_01)
- [ ] Memory usage stays under 4 GB
- [ ] Plots generated (optional for this phase)

---

## Execution Plan

### Step 1: Code Review (15 min)
```
1. Read evaluation.py _init_, load, validate methods
2. Check error handling patterns match train_with_hydra_zen.py
3. Identify any potential issues
4. Create list of improvements if needed
```

### Step 2: Test Model Loading (10 min)
```
1. Verify all 3 teacher models exist
2. Test torch.load() on each
3. Check file sizes are reasonable
```

### Step 3: Test Data Loading (10 min)
```
1. Check set_01 dataset exists
2. Test ID mapping generation
3. Verify train/val/test split creation
```

### Step 4: Run Individual Model Evaluations (30 min)
```
1. Test 3.1: VGAE Teacher (should take ~5-8 min)
2. Test 3.2: GAT Teacher (should take ~5-8 min)
3. Test 3.3: DQN Teacher (should take ~5-8 min)
4. Validate outputs
```

### Step 5: Validate CSV/JSON Exports (15 min)
```
1. Load and inspect CSVs
2. Load and inspect JSONs
3. Verify structure matches expectations
4. Check LaTeX table generation from CSV
```

### Step 6: Test Ablation (Optional, 15 min)
```
1. Create model config
2. Run training_mode comparison
3. Inspect ablation CSV
```

**Total Time**: ~1.5 hours for full test suite

---

## Known Issues to Watch For

1. **Model Architecture Mismatch**: If model was saved with different input dims, loading will fail
2. **GPU OOM**: set_01 is large; may need to reduce batch size
3. **Inference Placeholder**: evaluation.py currently has placeholder inference (returns random predictions)
   - **This is CRITICAL and must be fixed before using for papers**
4. **Missing Dependencies**: Ensure graph_creation works for set_01
5. **Path Issues**: Windows vs Linux path separators (should be handled by Path())

---

## Critical Fix Needed

**The _infer_subset() method in evaluation.py is currently a placeholder**:

```python
def _infer_subset(self, dataset: List) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run inference on dataset subset."""
    logger.warning("Using placeholder inference (actual inference depends on model architecture)")
    predictions = np.random.randint(0, 2, len(dataset))
    scores = np.random.rand(len(dataset))
    return predictions, scores
```

**Action Required**:
1. Implement actual model inference based on model type (VGAE, GAT, DQN)
2. Return real predictions and confidence scores
3. Remove placeholder warning

---

## Test Outputs Location

All test outputs will be saved to:
```
test_vgae_teacher.csv
test_vgae_teacher.json
test_gat_teacher.csv
test_gat_teacher.json
test_dqn_teacher.csv
test_dqn_teacher.json
ablation_test/  (if Phase 4 runs)
```

---

## Next Steps After Testing

1. **If all tests pass**: Evaluation framework is ready for paper integration
2. **If placeholder inference is issue**: Fix _infer_subset() with real inference logic
3. **If other issues found**: Fix in priority order, re-test
4. **Then run full evaluation pipeline** on all model variants for paper

---

**Status**: READY FOR PHASE 1 (Code Review)
