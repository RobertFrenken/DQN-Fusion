# CAN-Graph Master Task List

**Last Updated**: 2026-01-27 17:30 UTC
**Project Goal**: Complete evaluation framework validation and generate paper-ready results
**Current Phase**: Phase 2 - Evaluation Framework Testing

---

## 🎯 CRITICAL PATH (Required for Paper)

These tasks MUST be completed in order. Do not proceed to next phase until current phase passes all acceptance criteria.

### ✅ PHASE 1: Evaluation Framework Code Review & Inference Implementation
**Status**: COMPLETE
**Completion Date**: 2026-01-27

- [x] Protocol compliance review vs train_with_hydra_zen.py
- [x] Implement real inference for VGAE (reconstruction error-based)
- [x] Implement real inference for GAT (softmax-based)
- [x] Implement simplified inference for DQN fusion
- [x] Add model loading and instantiation logic
- [x] Verify syntax and imports
- [x] Commit changes to git

**Acceptance**: All 11 protocol checks pass, inference returns real predictions

---

### 🔄 PHASE 2: Model Loading & Data Robustness Validation
**Status**: READY TO START
**Estimated Time**: 1 hour
**Dependencies**: Phase 1 complete

**Objective**: Verify evaluation framework can load teacher models and datasets without errors

#### Tasks:

- [ ] **2.1**: Verify teacher model files exist
  - [ ] Check VGAE teacher: `experimentruns/automotive/set_01/unsupervised/vgae/teacher/no_distillation/autoencoder/models/vgae_teacher_autoencoder.pth`
  - [ ] Check GAT teacher: `experimentruns/automotive/set_01/supervised/gat/teacher/no_distillation/curriculum/models/gat_teacher_curriculum.pth`
  - [ ] Check DQN teacher: `experimentruns/automotive/set_01/rl_fusion/dqn/teacher/no_distillation/fusion/models/dqn_teacher_fusion.pth`
  - [ ] Verify file sizes are reasonable (~1-2 MB range)

- [ ] **2.2**: Test model loading robustness
  - [ ] Test VGAE model loads without corruption
  - [ ] Test GAT model loads without corruption
  - [ ] Test DQN model loads without corruption
  - [ ] Verify state dict keys match expected model architecture

- [ ] **2.3**: Test dataset loading for set_01
  - [ ] Verify data folder exists: `data/automotive/set_01/`
  - [ ] Build ID mapping from normal samples
  - [ ] Confirm ID mapping has >0 IDs
  - [ ] Test train/val split (expect ~80/20 ratio)
  - [ ] Test class balance in splits
  - [ ] Load test data successfully

- [ ] **2.4**: Test data preprocessing pipeline
  - [ ] Verify graph_creation works on set_01
  - [ ] Check window_size=100 processing
  - [ ] Confirm batch loading works (batch_size=32)

**Acceptance Criteria**:
- ✅ All 3 teacher models load without FileNotFoundError
- ✅ No model corruption errors
- ✅ ID mapping builds successfully with >0 IDs
- ✅ Train/val split creates non-empty datasets
- ✅ Test data loads without errors
- ✅ No OOM errors during data loading

**Gate**: Do not proceed to Phase 3 until all acceptance criteria pass

---

### 📊 PHASE 3: End-to-End Evaluation on Teacher Models (set_01)
**Status**: BLOCKED (waiting for Phase 2)
**Estimated Time**: 45 minutes (3 runs × 15 min each)
**Dependencies**: Phase 2 complete

**Objective**: Generate evaluation metrics for set_01 teacher models to validate framework

#### Tasks:

- [ ] **3.1**: VGAE Teacher Evaluation
  ```bash
  python -m src.evaluation.evaluation \
    --dataset set_01 \
    --model-path experimentruns/automotive/set_01/unsupervised/vgae/teacher/no_distillation/autoencoder/models/vgae_teacher_autoencoder.pth \
    --training-mode autoencoder \
    --mode standard \
    --csv-output test_results/vgae_teacher_set01.csv \
    --json-output test_results/vgae_teacher_set01.json \
    --verbose
  ```
  - [ ] Run completes without errors
  - [ ] Metrics computed for train/val/test
  - [ ] CSV file created with correct structure
  - [ ] JSON file created with correct structure
  - [ ] All metrics in [0, 1] range
  - [ ] No NaN values in results

- [ ] **3.2**: GAT Teacher Evaluation
  ```bash
  python -m src.evaluation.evaluation \
    --dataset set_01 \
    --model-path experimentruns/automotive/set_01/supervised/gat/teacher/no_distillation/curriculum/models/gat_teacher_curriculum.pth \
    --training-mode curriculum \
    --mode standard \
    --csv-output test_results/gat_teacher_set01.csv \
    --json-output test_results/gat_teacher_set01.json \
    --verbose
  ```
  - [ ] Run completes without errors
  - [ ] Metrics computed for train/val/test
  - [ ] CSV and JSON created
  - [ ] Metrics in valid range

- [ ] **3.3**: DQN Teacher Evaluation
  ```bash
  python -m src.evaluation.evaluation \
    --dataset set_01 \
    --model-path experimentruns/automotive/set_01/rl_fusion/dqn/teacher/no_distillation/fusion/models/dqn_teacher_fusion.pth \
    --training-mode fusion \
    --mode standard \
    --csv-output test_results/dqn_teacher_set01.csv \
    --json-output test_results/dqn_teacher_set01.json \
    --verbose
  ```
  - [ ] Run completes (note: simplified fusion inference)
  - [ ] CSV and JSON created
  - [ ] No crashes or exceptions

- [ ] **3.4**: CSV Export Validation
  - [ ] Load all 3 CSVs with pandas
  - [ ] Verify wide format: 1 row per subset (3 rows total)
  - [ ] Check columns include: dataset, subset, model, training_mode, kd_mode, all metrics
  - [ ] Confirm no NaN in numeric columns
  - [ ] Verify CSV is LaTeX-table ready

- [ ] **3.5**: JSON Export Validation
  - [ ] Load all 3 JSONs
  - [ ] Verify structure: metadata, results (train/val/test), threshold_optimization
  - [ ] Check all required metric categories present
  - [ ] Confirm timestamps are correct

**Acceptance Criteria**:
- ✅ All 3 models run inference successfully
- ✅ No OOM errors during inference
- ✅ All metrics numerically valid (0-1 range, no NaN)
- ✅ CSV files have correct wide format structure
- ✅ JSON files have complete nested structure
- ✅ Console output is informative and clear

**Gate**: Do not proceed to Phase 4 until metrics are validated

---

### 🔬 PHASE 4: Evaluation on All Datasets (hcrl_sa, hcrl_ch, set_02-04)
**Status**: BLOCKED (waiting for Phase 3)
**Estimated Time**: 4-6 hours
**Dependencies**: Phase 3 complete, teacher models validated

**Objective**: Generate comprehensive evaluation results for all datasets for paper

#### Tasks:

- [ ] **4.1**: Run VGAE teacher evaluation on all datasets
  - [ ] hcrl_sa (main dataset)
  - [ ] hcrl_ch (alternative dataset)
  - [ ] set_02
  - [ ] set_03
  - [ ] set_04

- [ ] **4.2**: Run GAT teacher evaluation on all datasets
  - [ ] hcrl_sa
  - [ ] hcrl_ch
  - [ ] set_02
  - [ ] set_03
  - [ ] set_04

- [ ] **4.3**: Run DQN teacher evaluation on all datasets
  - [ ] hcrl_sa
  - [ ] hcrl_ch
  - [ ] set_02
  - [ ] set_03
  - [ ] set_04

- [ ] **4.4**: Consolidate results
  - [ ] Merge all CSV files into master results CSV
  - [ ] Organize JSON files by dataset and model
  - [ ] Create summary statistics table

**Acceptance Criteria**:
- ✅ All dataset evaluations complete
- ✅ Consolidated results file created
- ✅ No missing data in results

---

### 📈 PHASE 5: Student Model Evaluation (With & Without KD)
**Status**: BLOCKED (waiting for Phase 4)
**Estimated Time**: 8-12 hours
**Dependencies**: Phase 4 complete, student models trained

**Objective**: Compare student model performance with and without knowledge distillation

#### Tasks:

- [ ] **5.1**: Evaluate Student No-KD models
  - [ ] VGAE student (no-KD) on all datasets
  - [ ] GAT student (no-KD) on all datasets
  - [ ] DQN student (no-KD) on all datasets

- [ ] **5.2**: Evaluate Student With-KD models
  - [ ] VGAE student (with-KD) on all datasets
  - [ ] GAT student (with-KD) on all datasets
  - [ ] DQN student (with-KD) on all datasets

- [ ] **5.3**: Generate comparison tables
  - [ ] Teacher vs Student No-KD
  - [ ] Student No-KD vs Student With-KD
  - [ ] Performance degradation analysis

**Acceptance Criteria**:
- ✅ All student models evaluated
- ✅ Comparison tables generated
- ✅ KD impact quantified

---

### 🧪 PHASE 6: Ablation Studies
**Status**: BLOCKED (waiting for Phase 5)
**Estimated Time**: 2-3 hours
**Dependencies**: Phase 5 complete

**Objective**: Generate ablation study results for paper

#### Tasks:

- [ ] **6.1**: Knowledge Distillation Impact Study
  ```bash
  python -m src.evaluation.ablation \
    --study kd \
    --model-list configs/kd_ablation_models.json \
    --output-dir ablation_results/kd_impact/
  ```
  - [ ] Compare Teacher vs Student No-KD vs Student With-KD
  - [ ] Generate delta metrics table
  - [ ] Compute statistical significance

- [ ] **6.2**: Curriculum Learning Impact Study
  ```bash
  python -m src.evaluation.ablation \
    --study curriculum \
    --model-list configs/curriculum_ablation_models.json \
    --output-dir ablation_results/curriculum_impact/
  ```
  - [ ] Compare Normal vs Curriculum training
  - [ ] Generate delta metrics table

- [ ] **6.3**: Fusion Strategy Impact Study
  ```bash
  python -m src.evaluation.ablation \
    --study fusion \
    --model-list configs/fusion_ablation_models.json \
    --output-dir ablation_results/fusion_impact/
  ```
  - [ ] Compare individual models vs fusion
  - [ ] Generate delta metrics table

- [ ] **6.4**: Training Mode Comparison Study
  ```bash
  python -m src.evaluation.ablation \
    --study training_mode \
    --model-list configs/training_mode_ablation_models.json \
    --output-dir ablation_results/training_mode/
  ```
  - [ ] Compare Autoencoder vs Normal vs Curriculum vs Fusion
  - [ ] Generate comprehensive comparison table

**Acceptance Criteria**:
- ✅ All 4 ablation studies complete
- ✅ Delta metrics computed correctly
- ✅ Results organized for paper integration

---

### 📝 PHASE 7: LaTeX Table Generation & Paper Integration
**Status**: BLOCKED (waiting for Phase 6)
**Estimated Time**: 2 hours
**Dependencies**: Phase 6 complete

**Objective**: Convert evaluation results to LaTeX tables for paper

#### Tasks:

- [ ] **7.1**: Generate main results tables
  - [ ] Teacher model performance table (all datasets)
  - [ ] Student model comparison table
  - [ ] KD impact summary table

- [ ] **7.2**: Generate ablation tables
  - [ ] KD ablation table
  - [ ] Curriculum ablation table
  - [ ] Fusion ablation table
  - [ ] Training mode comparison table

- [ ] **7.3**: Generate supplementary tables
  - [ ] Threshold optimization results
  - [ ] Per-dataset detailed metrics
  - [ ] Confusion matrices

- [ ] **7.4**: Paper integration
  - [ ] Insert tables into paper sections
  - [ ] Update results text with final numbers
  - [ ] Cross-reference table numbers
  - [ ] Verify all table captions

**Acceptance Criteria**:
- ✅ All LaTeX tables generated
- ✅ Tables formatted correctly
- ✅ Paper results section complete
- ✅ Ready for submission

---

## 🔧 NON-CRITICAL TASKS (Can Do Anytime)

These tasks improve the project but are not blocking for paper completion.

### Analysis & Documentation

- [ ] Analyze GPU monitoring from test job 43977477
  - [ ] Run: `python analyze_gpu_monitor.py gpu_monitor_43977477.csv`
  - [ ] Check for memory leaks
  - [ ] Identify bottlenecks (compute/memory/data-bound)
  - [ ] Generate utilization plots

- [ ] Document batch size optimization results
  - [ ] Summarize tuning logs
  - [ ] Create batch size recommendations table
  - [ ] Document memory usage patterns

- [ ] Verify run counter functionality
  - [ ] Check run_counter.txt increments correctly
  - [ ] Verify model filenames: run_001.pth, run_002.pth, etc.
  - [ ] Confirm no filename collisions

### Code Quality & Testing

- [ ] Add unit tests for evaluation framework
  - [ ] Test metrics.py functions
  - [ ] Test dataset loading edge cases
  - [ ] Test model instantiation

- [ ] Performance profiling
  - [ ] Profile inference time per sample
  - [ ] Identify slow operations
  - [ ] Optimize if needed

- [ ] Add plotting functionality
  - [ ] ROC curves
  - [ ] Precision-Recall curves
  - [ ] Confusion matrices
  - [ ] Threshold sweep plots

### Enhancements

- [ ] **DQN State Space Enrichment** (See `DQN_STATE_SPACE_ANALYSIS.md` for details)
  - [ ] Phase 1 Quick Wins (30 min total): 2D → 7D state
    - [ ] Separate VGAE error components (node/neighbor/canid)
    - [ ] GAT full logits instead of single probability
    - [ ] Add model confidence indicators (VGAE variance + GAT entropy)
    - [ ] Update fusion training loop to handle 7D state
    - [ ] Expected improvement: +2-5% accuracy
  - [ ] Phase 2 Medium Wins (30 min): 7D → 11D state
    - [ ] Add VGAE latent space summary (mean/std/max/min of z)
    - [ ] Update state stacking logic
    - [ ] Expected improvement: +4-8% accuracy

- [ ] Implement full DQN fusion inference for evaluation
  - [ ] Load both VGAE and GAT simultaneously
  - [ ] Properly combine outputs using DQN weights
  - [ ] Return fused predictions
  - [ ] Estimated time: 30 minutes

- [ ] Add support for additional datasets
  - [ ] Test on external CAN datasets
  - [ ] Generalize preprocessing pipeline

- [ ] Create visualization dashboard
  - [ ] Streamlit or Plotly dashboard
  - [ ] Interactive metric exploration
  - [ ] Model comparison visualizations

---

## ⏸️ BLOCKED / WAITING

### Currently Waiting For:

- **Test Job 43977817 Results** (RESUBMITTED WITH FIX)
  - Status: RUNNING in queue
  - Purpose: Verify run counter and batch size implementation
  - Previous failure (43977477): AttributeError - `'dict' object has no attribute 'optimize_batch_size'`
  - **Fix Applied**: Added `BatchSizeConfig` to frozen_config.py class_map (missing from deserialization)
  - Expected completion: Within 1 hour of submission
  - Action when complete: Check logs, verify run counter, analyze GPU monitor CSV

### Recently Completed:

- ✅ **Test Job 43977477 Failure Analysis** (2026-01-27 18:00 UTC)
  - Root cause: BatchSizeConfig not in frozen config deserialization class_map
  - Fix: Added to imports and class_map in `src/config/frozen_config.py`
  - GPU monitoring: No GPU utilization (failure before training started)

- ✅ **DQN State Space Research** (2026-01-27 18:15 UTC)
  - Documented current 2D state: [anomaly_score, gat_prob]
  - Identified quick wins: 7D state with separated error components + confidence
  - Created comprehensive analysis: `DQN_STATE_SPACE_ANALYSIS.md`

### Currently Blocked:

- **Phase 3+ Evaluation** (Blocked on: Phase 2 not started yet)
- **Student Model Evaluation** (Blocked on: Teacher validation not complete)
- **Ablation Studies** (Blocked on: Student evaluations not complete)
- **Paper Integration** (Blocked on: All evaluations not complete)

### Dependencies to Resolve:

- **set_01 Dataset Availability**: Must verify exists at `data/automotive/set_01/`
- **Teacher Model Availability**: Must verify all 3 teacher models exist
- **GPU Availability**: Need GPU access for faster inference (optional, can use CPU)

---

## 🧹 CHECK & CLEANUP PROTOCOL

Run this protocol at the end of each phase or before finalizing work.

### Phase Completion Checklist

After completing any phase:

- [ ] Verify all acceptance criteria met
- [ ] Run syntax check: `python -m py_compile src/**/*.py`
- [ ] Check for uncommitted changes: `git status`
- [ ] Review test outputs for errors
- [ ] Update MASTER_TASK_LIST.md with completion status
- [ ] Update SESSION_SUMMARY.md with progress

### File Organization Checklist

- [ ] **Code Files**
  - [ ] All .py files have docstrings
  - [ ] No commented-out code blocks
  - [ ] No debug print statements

- [ ] **Documentation Files**
  - [ ] Consolidate related docs into folders (e.g., evaluation_docs/)
  - [ ] Remove duplicate or outdated files
  - [ ] Update README.md with current status

- [ ] **Test Outputs**
  - [ ] Move test CSVs to `test_results/` folder
  - [ ] Move test JSONs to `test_results/` folder
  - [ ] Archive old test runs to `test_results/archive/`
  - [ ] Remove temporary test files

- [ ] **Experiment Outputs**
  - [ ] Organize model checkpoints by experiment name
  - [ ] Clean up failed run directories
  - [ ] Archive completed experiments

### Git Cleanup Checklist

- [ ] Stage relevant changes: `git add <files>`
- [ ] Verify staged changes: `git diff --staged`
- [ ] Write descriptive commit message
- [ ] Include Co-Authored-By line
- [ ] Commit: `git commit -m "message"`
- [ ] Verify clean working tree: `git status`

### Pre-Submission Cleanup (Before Paper Submission)

- [ ] **Remove All Test/Debug Files**
  - [ ] test_*.csv
  - [ ] test_*.json
  - [ ] gpu_monitor_*.csv
  - [ ] slurm-*.out (archive to separate folder)

- [ ] **Consolidate Documentation**
  - [ ] Merge redundant documentation files
  - [ ] Create single comprehensive README
  - [ ] Move implementation details to docs/ folder

- [ ] **Archive Old Experiments**
  - [ ] Move failed runs to archive/
  - [ ] Keep only final trained models
  - [ ] Document model locations in README

- [ ] **Code Quality Final Pass**
  - [ ] Run linter: `pylint src/`
  - [ ] Remove unused imports
  - [ ] Remove unused functions
  - [ ] Check all docstrings are complete

- [ ] **Final Git Commit**
  - [ ] Tag final version: `git tag v1.0-paper-submission`
  - [ ] Push to remote: `git push origin main --tags`

---

## 📊 PROGRESS TRACKER

| Phase | Status | Completion | Blocking Issues |
|-------|--------|------------|-----------------|
| Phase 1: Code Review & Inference | ✅ COMPLETE | 100% | None |
| Phase 2: Model Loading Validation | 🔄 READY | 0% | None |
| Phase 3: Teacher Evaluation (set_01) | ⏸️ BLOCKED | 0% | Phase 2 not started |
| Phase 4: All Datasets Evaluation | ⏸️ BLOCKED | 0% | Phase 3 not complete |
| Phase 5: Student Model Evaluation | ⏸️ BLOCKED | 0% | Phase 4 not complete |
| Phase 6: Ablation Studies | ⏸️ BLOCKED | 0% | Phase 5 not complete |
| Phase 7: LaTeX & Paper Integration | ⏸️ BLOCKED | 0% | Phase 6 not complete |

**Overall Progress**: 14% (1/7 phases complete)

---

## 🎯 CURRENT FOCUS

**Next Action**: Start Phase 2 - Model Loading & Data Robustness Validation

**Command to Start**:
```
"Start Phase 2 from MASTER_TASK_LIST.md"
```

**Expected Duration**: ~1 hour
**Expected Completion**: All teacher models load, dataset loads, ready for Phase 3

---

## 📞 QUICK REFERENCE

### Key Files
- This file: `MASTER_TASK_LIST.md` - Single source of truth for all tasks
- Progress: `SESSION_SUMMARY.md` - Session history
- Inference docs: `INFERENCE_IMPLEMENTATION.md` - Technical details
- Test plan: `EVALUATION_TEST_PLAN.md` - Detailed test procedures
- Phase 1 review: `EVALUATION_PHASE1_REVIEW.md` - Code review results
- DQN analysis: `DQN_STATE_SPACE_ANALYSIS.md` - State space enhancement options

### Key Commands
- Start phase: "Start Phase N from MASTER_TASK_LIST.md"
- Check status: "What's the status of MASTER_TASK_LIST.md?"
- Update progress: "Mark task X.Y as complete"
- Skip to task: "Work on task X.Y from MASTER_TASK_LIST.md"

### Key Paths
- Models: `experimentruns/automotive/{dataset}/{pipeline}/{model}/...`
- Data: `data/automotive/{dataset}/`
- Evaluation: `src/evaluation/evaluation.py`
- Results: `test_results/` (to be created)

---

**Last Updated**: 2026-01-27 18:20 UTC
**Maintainer**: Claude Code Agent
**Status**: Phase 1 complete, test job resubmitted, Phase 2 ready to start
**Recent Work**: Fixed BatchSizeConfig deserialization bug, researched DQN state space enhancements

