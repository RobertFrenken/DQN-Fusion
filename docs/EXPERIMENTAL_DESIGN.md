# Experimental Design: Testing Your VGAE + Curriculum + Memory Approach

## Overview

This document outlines a comprehensive experimental strategy to validate your approach and compare it against alternatives. The goal is to isolate the contribution of each component (VGAE hard mining, momentum curriculum, memory buffer) and demonstrate superiority over baselines.

---

## Phase 1: Ablation Studies (Isolate Components)

### Experiment 1.1: Component Ablation

**Objective**: Quantify the contribution of each component

**Setup**:
- **Baseline (B)**: Standard supervised learning with cross-entropy loss on imbalanced data
- **B + Balanced Sampling (B+BS)**: Oversample minority class to balance, then train normally
- **B + Hard Mining (B+HM)**: Standard training + hard negative mining (no VGAE, just use model loss)
- **B + VGAE Mining (B+VM)**: Hard mining via VGAE reconstruction error
- **B + Curriculum (B+C)**: Balanced → Imbalanced progression without hard mining
- **B + Curriculum + VGAE (B+C+VM)**: Your full method

**Metrics**:
- Minority class F1 (weighted by class frequency)
- Majority class F1 (to ensure no degradation)
- Macro F1 (average of all classes)
- Area Under Precision-Recall Curve (AUPRC) for minority class
- Training time / convergence speed

**Expected Results**:
- B+VM should outperform B+HM (VGAE better difficulty signal)
- B+C should outperform B+BS (curriculum beats static balance)
- B+C+VM should be best overall
- Order of improvement: B < B+BS < B+C < B+C+VM

**Dataset**: Start with 2-3 medium-scale imbalanced datasets:
- CIFAR-100 with induced imbalance (delete 90% of rare classes)
- Your own automotive data (if available)
- Public imbalance benchmark (e.g., IMDB-Wiki)

---

### Experiment 1.2: Hyperparameter Sensitivity

**Objective**: Understand sensitivity to key hyperparameters

**Parameters to vary**:
1. **Momentum strength** ($\lambda$ in curriculum schedule)
   - Values: 0.1, 0.5, 1.0, 2.0, 5.0
   - Hypothesis: Very high $\lambda$ recovers imbalanced too fast; very low is too slow

2. **Hard mining buffer size** ($k$)
   - Values: 10%, 25%, 50%, 75%, 100% of minority class size
   - Hypothesis: 25-50% is optimal (enough diversity without too many outliers)

3. **Replay weight** ($\alpha$)
   - Values: 0.0 (no replay), 0.1, 0.2, 0.5, 1.0 (all replay)
   - Hypothesis: 0.1-0.3 balances stability and diversity

4. **VGAE architecture**
   - Encoder: 1-layer vs 2-layer vs 3-layer
   - Latent dimension: 8, 16, 32, 64
   - Hypothesis: Shallow encoder captures reconstruction error; too deep loses signal

**Setup**: Grid search over top 2-3 parameters (reduce computational cost)

**Metrics**: Minority class F1, training stability (variance across runs)

---

## Phase 2: Comparison Against SOTA Baselines

### Experiment 2.1: Class Imbalance Methods

**Objective**: Compare against well-established imbalance techniques

**Baselines**:
1. **Focal Loss** (Lin et al., 2017): Re-weights loss by inverse class frequency
2. **Class-Balanced Loss** (Cui et al., 2019): Effective number of samples
3. **SMOTE + Standard Training**: Synthetic minority oversampling
4. **Mixup on Imbalanced Data** (Verma et al., 2019): Interpolation-based augmentation
5. **Dynamic Curriculum Learning (DCL)** (Wang et al., 2019): Curriculum from imbalanced → balanced
6. **Cost-Sensitive Learning**: Assign higher loss weight to minority samples

**Protocol**:
- All methods use same backbone architecture (e.g., ResNet-18)
- Same train/val/test split
- Same hyperparameter tuning budget (e.g., 10 random seeds, 5 epochs of tuning)
- Report mean ± std over 5 runs

**Metrics**:
- Minority F1 (primary)
- Majority F1 (guard against collapse)
- Macro F1
- Computational cost (training time, memory)
- Convergence speed (epochs to 95% of final F1)

**Expected Results**:
- Your method > DCL (different curriculum direction + memory)
- Your method ≥ Focal Loss + SMOTE (more principled than additive approaches)
- Your method > Cost-Sensitive alone (hard mining adds specificity)

---

### Experiment 2.2: Curriculum Learning Methods

**Objective**: Compare your curriculum strategy against alternatives

**Baselines**:
1. **Standard Curriculum (Easy→Hard)**: Sort by loss, present in ascending order (Hacohen & Weinshall, 2019)
2. **Anti-Curriculum (Hard→Easy)**: Opposite ordering
3. **Self-Paced Learning**: Curriculum based on model's current difficulty
4. **Linear Imbalance Transition**: $p(t) = t/T$ (linear vs your exponential)
5. **Step-wise Imbalance**: Discrete phases (0-33% balanced, 33-66% mixed, 66-100% imbalanced)
6. **Your Momentum Curriculum**: $p(t) = 1 - e^{-\lambda t/T}$

**Setup**:
- All use same hard mining method (VGAE)
- Vary only curriculum schedule
- Same number of epochs

**Metrics**:
- Minority class F1
- Convergence speed
- Stability (variance in F1 over last 10 epochs)

**Expected Results**:
- Your exponential schedule outperforms linear (avoids abrupt shifts)
- Curriculum beats anti-curriculum (validates curriculum principle)
- Your approach > self-paced (VGAE signal more reliable than model-current loss)

---

### Experiment 2.3: Memory/Replay Strategies

**Objective**: Validate memory buffer design

**Baselines**:
1. **No Buffer**: Standard curriculum without replay (B+C+VM with $\alpha=0$)
2. **Random Buffer**: Replay random hard examples, not VGAE-selected
3. **Fixed Buffer**: Buffer contents fixed at start (no updating)
4. **Adaptive Buffer** (Your method): Update every $N$ steps based on current VGAE
5. **Uniform Memory**: Replay equal numbers from all classes (vs hard-only)
6. **EMA Buffer**: Exponential moving average of past examples (vs discrete updates)

**Setup**:
- Same curriculum, same hard mining
- Vary only buffer strategy

**Metrics**:
- Minority class F1
- Minority class recall (sensitivity)
- Majority class precision (specificity guard)
- Training stability

**Expected Results**:
- Adaptive > Fixed (updating captures distribution shift)
- VGAE Buffer > Random Buffer (VGAE signals meaningful difficulty)
- Your method > No Buffer (memory prevents majority class forgetting)

---

## Phase 3: Edge Case and Robustness Testing

### Experiment 3.1: Extreme Imbalance

**Objective**: Performance at severe imbalance ratios

**Setup**:
- Vary imbalance ratio: 1:10, 1:50, 1:100, 1:500
- Same method comparison as Exp 2.1
- Focus on minority class metrics

**Metrics**:
- Minority class F1
- Minority class AUPRC
- Minority class recall at 99% majority precision

**Expected Results**:
- Your method degradation < baselines at extreme imbalance
- Memory buffer critical at 1:500 (prevents mode collapse)

---

### Experiment 3.2: Multiple Imbalance Ratios Per Dataset

**Objective**: Robustness to unknown imbalance distribution

**Setup**:
- Train on one imbalance ratio, test on different ratio
- Example: Train on 1:50 imbalance, test on 1:100

**Metrics**:
- Transfer F1 (how well does training transfer?)

**Expected Results**:
- Your method transfers better (curriculum principle generalizes)

---

### Experiment 3.3: Ablate VGAE vs Random Hard Mining

**Objective**: Is VGAE necessary or is any hard mining sufficient?

**Setup**:
- **B+C+RM**: Curriculum + random hard mining (select random samples as "hard")
- **B+C+HM**: Curriculum + model-based hard mining (select by training loss)
- **B+C+VM**: Curriculum + VGAE hard mining (your method)

**Metrics**:
- Minority F1
- Convergence speed
- Correlation: Are VGAE-selected samples actually harder?

**Expected Results**:
- VGAE hard mining ~ Model-based hard mining (both signal difficulty)
- VGAE better at early training (model not yet confident)

---

## Phase 4: Dataset Diversity

### Experiment 4.1: Cross-Domain Validation

**Objective**: Generalization across domains

**Datasets**:
1. **Vision**: CIFAR-100 (induced imbalance), ImageNet-LT (natural long-tail)
2. **NLP**: IMDB sentiment (class imbalance via undersampling)
3. **Tabular**: Adult income dataset (gender/race imbalance)
4. **Your domain**: Automotive anomaly detection (if available)

**Protocol**:
- Run full baseline comparison (Exp 2.1) on each domain
- Report domain-specific best hyperparameters

**Expected Results**:
- Consistent improvement across domains validates generality
- Minor hyperparameter tuning required per domain (expected)

---

## Phase 5: Interpretation and Analysis

### Experiment 5.1: What is VGAE Learning?

**Objective**: Understand what reconstruction error signals

**Procedure**:
1. Train VGAE on full dataset (including test set, no labels)
2. Compute reconstruction error for:
   - Correctly classified samples
   - Misclassified samples
   - Boundary samples (predicted probability 0.4-0.6)
3. Visualize reconstruction error distribution

**Metrics**:
- Correlation: reconstruction error ↔ classification difficulty
- KL divergence of error distributions between classes

**Expected Results**:
- Misclassified samples have higher reconstruction error (validates VGAE signal)
- Boundary samples concentrated in high-error region

---

### Experiment 5.2: Curriculum Trajectory

**Objective**: Visualize what the model learns at each phase

**Procedure**:
1. Checkpoint model every epoch
2. Compute class-wise accuracy at each checkpoint
3. Plot: Minority F1 vs Majority F1 vs Epoch

**Expected Results**:
- Phase 1 (balanced): Both classes improve equally
- Phase 2 (curriculum): Minority F1 plateaus, majority F1 continues (deliberate trade-off)
- Phase 3 (imbalanced): Minority F1 improves again via hard mining

---

### Experiment 5.3: Attention/Saliency Analysis

**Objective**: Does model focus on different regions by phase?

**Procedure** (if vision domain):
1. Generate Grad-CAM for minority/majority samples
2. Compare Grad-CAM across phases
3. Do attention maps converge to similar regions or diverge?

**Expected Result**:
- Early phases: Broad attention (learning general features)
- Late phases: Focused attention (hard examples, decision boundaries)

---

## Experiment Execution Plan

### Timeline
- **Week 1**: Exp 1.1, 1.2 (component ablation, hyperparameter tuning)
- **Week 2**: Exp 2.1, 2.2, 2.3 (baseline comparisons)
- **Week 3**: Exp 3.1-3.3 (edge cases, robustness)
- **Week 4**: Exp 4.1 (cross-domain), 5.1-5.3 (analysis)

### Compute Requirements
- GPU: 1x V100 should suffice (most experiments parallelize)
- Time: ~1-2 weeks (wall-clock) with proper parallelization
- Storage: ~50GB for checkpoints if using ImageNet-LT

### Reporting
- Main paper: Exp 2.1 results + Exp 1.1 (ablation)
- Appendix: Exp 1.2, 2.2, 2.3, 3.1-3.3, 4.1
- Supplementary: Exp 5.1-5.3 (interpretation)

---

## Success Criteria

Your method should achieve:
1. **Minority class F1 > 85%** of best baseline on all datasets
2. **Majority class F1 > 95%** of best baseline (no collapse)
3. **Faster convergence**: Reach 90% of final F1 in <80% of baseline epochs
4. **Robustness**: Hyperparameters stable across domains (one set works for all)
5. **Interpretability**: Ablation study clearly shows contribution of each component

---

## Statistical Testing

For all comparisons:
- Report 95% confidence intervals (5 runs minimum)
- Use paired t-tests when comparing your method vs single baseline
- Use one-way ANOVA for multi-way comparisons
- Report p-values; * p<0.05, ** p<0.01

---

## Questions to Address in Paper

1. **Why exponential curriculum vs linear?** → Exp 1.2 + 2.2
2. **Why VGAE vs simpler hard mining?** → Exp 3.3
3. **What is the memory buffer doing?** → Exp 2.3 + 5.2
4. **Does this work outside class imbalance?** → Exp 4.1, test on balanced data
5. **How sensitive is performance to hyperparameters?** → Exp 1.2
6. **What types of samples does VGAE identify as hard?** → Exp 5.1

---

## References

- Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. In *CVPR* (pp. 9268-9277).
- Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. In *ICML* (pp. 173-182). PMLR.
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *ICCV* (pp. 2980-2988).
- Wang, Y., Gan, W., Yang, J., Wu, W., & Yan, J. (2019). Dynamic curriculum learning for imbalanced data classification. In *ICCV* (pp. 5017-5026).
- Verma, V., Lamb, A., Beckham, C., Najafi Abed-Esfahani, A., Mitliagkas, I., Lopez-Paz, D., ... & Bengio, Y. (2019). Manifold mixup: Better representations by interpolating hidden states. In *ICML* (pp. 6438-6447). PMLR.
