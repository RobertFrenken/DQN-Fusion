# Publication-Quality Visualizations Plan
# CAN-Graph: Knowledge Distillation with DQN Fusion for IDS

**Purpose**: Create publication-ready figures for research paper demonstrating:
1. Novel 15D DQN fusion with rich state representation
2. Knowledge distillation effectiveness for model compression
3. Superior performance on CAN intrusion detection
4. Interpretable decision-making through visualization

**Last Updated**: 2026-01-27
**Target**: 8-12 main figures + supplementary materials

---

## Prerequisites & Style Guide

### 1. Style Consistency
- **Font**: Use LaTeX-compatible fonts (Computer Modern or similar)
- **Font Size**: 8-10pt for labels, 10-12pt for titles (match paper template)
- **Color Palette**:
  - Primary: Colorblind-friendly palette (e.g., `seaborn-colorblind` or IBM Design palette)
  - Models: Distinct colors for VGAE (blue), GAT (orange), Fusion (purple)
  - Classes: Normal (green), Attack (red)
- **DPI**: 300+ for publication quality
- **Format**: PDF (vector) for scalability
- **Size**: Column width (3.5") or full width (7") based on IEEE/ACM templates

### 2. Style Guide Resources
- **Matplotlib**: Use `seaborn-paper` or `seaborn-whitegrid` styles
- **Template**: Create `paper_style.mplstyle` for consistency
- **Reference**: https://github.com/garrettj403/SciencePlots (publication-ready matplotlib styles)

### 3. Data Preprocessing
- Collect unique attack types per dataset: DoS, Fuzzing, RPM, Gear spoofing
- Ensure consistent label mapping: 0=Normal, 1=Attack (with attack type subcategories)
- Cache preprocessed data for visualization to avoid recomputation

---

## Figure 1: System Architecture & 15D State Space

**Purpose**: Introduce the complete fusion framework with 15D enhancement

**Components**:
1. **Top Panel**: High-level architecture
   - CAN message input → Graph construction
   - Parallel VGAE (unsupervised) and GAT (supervised) paths
   - DQN fusion agent with 15D state input
   - Final attack detection output

2. **Bottom Panel**: 15D State Space Breakdown
   - Visual decomposition showing:
     - VGAE Features (8 dims): 3 error bars + 4 statistics boxes + 1 confidence circle
     - GAT Features (7 dims): 2 logit bars + 4 statistics boxes + 1 confidence circle
   - Use color-coded blocks to show feature groups
   - Annotate with value ranges

**Implementation**:
- Tool: `matplotlib` with `patches` for boxes and arrows
- File: `visualizations/architecture_diagram.py`
- Output: `figures/fig1_architecture.pdf`

**Key Insight**: Shows why 15D > 2D (much richer information for fusion decisions)

---

## Figure 2: Embedding Space Visualization (UMAP/PyMDE)

**Purpose**: Demonstrate learned representations separate attack classes effectively

**Layout**: 2x3 grid comparing different embedding spaces

**Subplots**:
1. **Raw Features** (top-left): UMAP of original graph features
2. **VGAE Latent Space** (top-middle): UMAP of 48D latent z
3. **VGAE Decoder** (top-right): UMAP of reconstructed features
4. **GAT Layer 1** (bottom-left): UMAP of first GAT layer embeddings
5. **GAT Layer 2** (bottom-middle): UMAP of second GAT layer embeddings
6. **GAT Pre-Pooling** (bottom-right): UMAP of final pre-pooling embeddings (used in 15D state)

**Styling**:
- Scatter points colored by class (normal=green, attack=red)
- Optionally: attack subtypes with different shades
- Add contour density lines for each class
- Legend with class counts

**Implementation**:
- Primary: `umap-learn` (proven method)
- Alternative: `pymde` (potentially better preservation of structure)
- Dataset: Use hcrl_sa (medium size, good class balance)
- Sample size: ~5000 graphs for computational efficiency

**File**: `visualizations/embedding_umap.py`
**Output**: `figures/fig2_embeddings.pdf`

**Key Insight**: Shows progressive refinement of representations, validates that VGAE latent and GAT embeddings capture complementary information

---

## Figure 3: VGAE Reconstruction Analysis

**Purpose**: Validate VGAE learns meaningful anomaly scores through reconstruction

**Layout**: 2x2 grid

**Subplots**:
1. **Node Reconstruction Error** (top-left)
   - Histogram: Normal vs Attack distributions
   - Show overlap region
   - Mark optimal threshold

2. **Neighbor Reconstruction Error** (top-right)
   - Histogram: Normal vs Attack distributions
   - Different pattern from node error (shows complementary signal)

3. **CAN ID Reconstruction Error** (bottom-left)
   - Histogram: Normal vs Attack distributions
   - Often shows distinct separation for ID-based attacks

4. **Combined Weighted Error** (bottom-right)
   - Histogram using weights [0.4, 0.35, 0.25] (same as DQN uses)
   - Show superior separation vs individual components
   - Overlay KDE curves for smoothness

**Styling**:
- Semi-transparent histograms with overlapping distributions
- Vertical lines for means
- Shaded regions for std
- Add AUC score for each distribution

**Implementation**:
- Dataset: hcrl_sa validation set
- Bins: 50-100 for smooth distributions
- File: `visualizations/vgae_reconstruction.py`
- Output: `figures/fig3_vgae_analysis.pdf`

**Key Insight**: Validates that reconstruction error is a meaningful anomaly signal, justifies the weighted combination used in DQN state

---

## Figure 4: DQN Policy Analysis - Alpha Selection Strategy

**Purpose**: Understand how DQN selects fusion weights based on 15D state

**Layout**: 2x2 grid

**Subplots**:
1. **Alpha Heatmap** (top-left)
   - X-axis: VGAE confidence [0, 1]
   - Y-axis: GAT confidence [0, 1]
   - Color: Mean alpha selected by DQN
   - Show decision boundaries
   - Key pattern: High VGAE conf + Low GAT conf → α close to 0 (trust VGAE)
   - High GAT conf + Low VGAE conf → α close to 1 (trust GAT)

2. **Model Agreement vs Alpha** (top-right)
   - X-axis: |VGAE_prob - GAT_prob| (disagreement)
   - Y-axis: Alpha selected
   - Scatter plot with density coloring
   - Show trendline
   - Hypothesis: High disagreement → alpha more extreme (trust one model)

3. **Confidence-Based Trust** (bottom-left)
   - Box plots showing alpha distribution grouped by confidence scenarios:
     - Both high confidence
     - Both low confidence
     - VGAE high, GAT low
     - VGAE low, GAT high
   - Shows context-dependent trust

4. **Feature Importance for Alpha Selection** (bottom-right)
   - Bar chart of 15D feature importance (using gradient attribution or SHAP)
   - Which dimensions most influence alpha?
   - Separate bars for VGAE features (blue) vs GAT features (orange)

**Implementation**:
- Use validation set predictions from trained DQN
- Compute statistics over ~10k samples
- File: `visualizations/dqn_policy_analysis.py`
- Output: `figures/fig4_dqn_policy.pdf`

**Key Insight**: Shows DQN learned intelligent, interpretable fusion strategy (not just averaging)

---

## Figure 5: Performance Comparison - Main Results

**Purpose**: Primary results figure showing model comparison across datasets

**Layout**: 2x2 grid (or single wide panel)

**Subplots**:
1. **Accuracy by Model & Dataset** (bar chart)
   - X-axis: Datasets (hcrl_sa, hcrl_ch, set_01, set_02, set_03, set_04)
   - Y-axis: Accuracy (0.8 - 1.0 range)
   - Grouped bars: Teacher (dark), Student No-KD (medium), Student With-KD (light)
   - Error bars if multiple runs available
   - Horizontal line at 0.95 for reference

2. **F1-Score Comparison**
   - Same layout as accuracy
   - Highlights KD effectiveness

3. **Model Size vs Performance**
   - X-axis: Model parameters (log scale)
   - Y-axis: F1-Score
   - Scatter points: Teacher (large), Student (small)
   - Arrow showing KD improvement
   - Pareto frontier line

4. **ROC Curves** (if space, otherwise separate figure)
   - Single dataset (hcrl_sa) showing all models
   - Legend with AUC scores

**Styling**:
- Use distinct patterns (solid, hatched, dotted) for accessibility
- Add significance markers (*, **, ***) for statistical tests
- Include performance delta annotations (e.g., "+3.2%" for KD improvement)

**Implementation**:
- Load results from evaluation framework CSVs
- File: `visualizations/performance_comparison.py`
- Output: `figures/fig5_performance.pdf`

**Key Insight**: Demonstrates KD effectiveness and fusion superiority

---

## Figure 6: ROC and Precision-Recall Curves

**Purpose**: Detailed performance analysis across operating points

**Layout**: 1x2 panel

**Subplots**:
1. **ROC Curves** (left)
   - All models on hcrl_sa test set
   - Legend with AUC scores
   - Diagonal reference line
   - Zoom inset for TPR > 0.9 region

2. **Precision-Recall Curves** (right)
   - All models on hcrl_sa test set
   - Legend with AP scores
   - Horizontal line at class balance
   - Important for imbalanced datasets

**Styling**:
- Distinct line styles (solid, dashed, dotted)
- Thicker lines for fusion models
- Shaded confidence intervals if multiple runs

**Implementation**:
- Use evaluation framework outputs (y_scores)
- File: `visualizations/roc_pr_curves.py`
- Output: `figures/fig6_roc_pr.pdf`

**Key Insight**: Shows consistent superiority across all operating points

---

## Figure 7: Ablation Study - 15D State Space Impact

**Purpose**: Justify 15D enhancement over baseline 2D state

**Layout**: 2x2 grid

**Subplots**:
1. **State Dimension vs Accuracy** (top-left)
   - X-axis: State dimensions (2D baseline, 7D quick-win, 11D medium, 15D full)
   - Y-axis: Accuracy
   - Line plot showing improvement curve
   - Error bars from multiple runs
   - Diminishing returns analysis

2. **Feature Ablation** (top-right)
   - Bar chart showing accuracy drop when removing each feature group:
     - Remove VGAE errors (keep just latent stats)
     - Remove VGAE latent stats (keep just errors)
     - Remove GAT logits
     - Remove GAT embeddings
     - Remove confidence scores
   - Baseline: Full 15D performance
   - Negative bars showing performance drop

3. **Alpha Distribution Evolution** (bottom-left)
   - Violin plots of alpha distribution:
     - 2D state (limited expressiveness)
     - 15D state (richer decisions)
   - Show that 15D enables more nuanced fusion weights

4. **Reward Curves During Training** (bottom-right)
   - X-axis: Training episodes
   - Y-axis: Average reward
   - Compare 2D vs 15D training convergence
   - Show 15D learns faster and reaches higher reward

**Implementation**:
- Requires training multiple DQN variants (2D, 7D, 11D, 15D)
- File: `visualizations/ablation_state_space.py`
- Output: `figures/fig7_ablation_15d.pdf`

**Key Insight**: Quantifies the value of each enhancement step

---

## Figure 8: Knowledge Distillation Impact

**Purpose**: Show KD effectiveness for model compression

**Layout**: 2x2 grid

**Subplots**:
1. **Performance Delta Heatmap** (top-left)
   - Rows: Datasets
   - Columns: Metrics (Accuracy, F1, Precision, Recall, AUC)
   - Cell color: (Student+KD) - (Student No-KD)
   - Annotate cells with percentage values
   - Mostly positive (green) showing KD benefit

2. **Model Capacity vs Performance** (top-right)
   - Scatter plot:
     - X-axis: Model parameters (normalized)
     - Y-axis: F1-Score
   - Points: Teacher (100% params, 95% F1), Student No-KD (10% params, 88% F1), Student+KD (10% params, 93% F1)
   - Arrow showing KD uplift
   - Efficiency frontier curve

3. **Training Dynamics** (bottom-left)
   - X-axis: Epochs
   - Y-axis: Validation Loss
   - Lines: Student No-KD vs Student With-KD
   - Show KD converges faster and to better optimum
   - Shaded regions for variance

4. **Inference Time vs Accuracy** (bottom-right)
   - Scatter plot showing Pareto frontier
   - Teacher: High accuracy, slow
   - Student: Fast, lower accuracy
   - Student+KD: Fast, high accuracy (best of both)

**Implementation**:
- Use evaluation results + training logs
- File: `visualizations/kd_impact.py`
- Output: `figures/fig8_kd_impact.pdf`

**Key Insight**: Demonstrates KD recovers most teacher performance with 10x compression

---

## Figure 9: Per-Attack-Type Performance

**Purpose**: Detailed breakdown showing which attacks are detected well

**Layout**: Single heatmap

**Subplot**:
- **Attack Type Performance Matrix**
  - Rows: Attack types (DoS, Fuzzing, RPM spoofing, Gear spoofing, Normal)
  - Columns: Models (VGAE, GAT, Fusion 2D, Fusion 15D)
  - Cell color: F1-Score
  - Annotate with F1 values
  - Show which attacks benefit most from fusion

**Styling**:
- Diverging colormap (low=red, high=green)
- Bold border around best model per attack type
- Add marginal bar charts showing per-attack difficulty

**Implementation**:
- Requires per-class metrics from evaluation
- File: `visualizations/per_attack_analysis.py`
- Output: `figures/fig9_attack_breakdown.pdf`

**Key Insight**: Shows fusion helps most on hard-to-detect attacks

---

## Figure 10: Confusion Matrices (Supplementary)

**Purpose**: Detailed error analysis

**Layout**: 2x2 grid showing confusion matrices for key models

**Subplots**:
1. VGAE Teacher
2. GAT Teacher
3. Fusion Teacher (15D DQN)
4. Fusion Student+KD (15D DQN)

**Styling**:
- Heatmap with annotations
- Normalize by row (show percentages)
- Add marginal totals
- Highlight diagonal (correct predictions)

**Implementation**:
- Use evaluation framework outputs
- File: `visualizations/confusion_matrices.py`
- Output: `figures/fig10_confusion.pdf`

---

## Figure 11: Training Dynamics (Supplementary)

**Purpose**: Show convergence behavior and stability

**Layout**: 2x3 grid

**Subplots**:
1. **DQN Reward Curve**
2. **DQN Epsilon Decay**
3. **DQN Q-Value Evolution**
4. **VGAE Reconstruction Loss**
5. **GAT Classification Loss**
6. **Knowledge Distillation Loss**

**Implementation**:
- Parse training logs from MLflow or TensorBoard
- File: `visualizations/training_dynamics.py`
- Output: `figures/fig11_training.pdf`

---

## Figure 12: Computational Efficiency (Supplementary)

**Purpose**: Practical deployment considerations

**Layout**: 1x3 panel

**Subplots**:
1. **Model Size Comparison** (bar chart)
2. **Inference Time per Sample** (bar chart)
3. **GPU Memory Usage During Training** (bar chart)

**Styling**:
- Log scale if needed
- Group by model type
- Annotate with actual values

**Implementation**:
- Collect from GPU monitoring logs
- File: `visualizations/computational_cost.py`
- Output: `figures/fig12_efficiency.pdf`

---

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create `paper_style.mplstyle` with publication settings
- [ ] Set up `visualizations/` directory structure
- [ ] Implement data loading utilities
- [ ] Create base plotting functions with consistent styling

### Phase 2: Essential Figures (Week 1-2)
Priority order:
1. **Fig 5**: Performance comparison (main results)
2. **Fig 4**: DQN policy analysis (novel contribution)
3. **Fig 2**: Embedding visualization (validation)
4. **Fig 1**: Architecture diagram (introduction)

### Phase 3: Ablation & Analysis (Week 2-3)
5. **Fig 7**: 15D ablation study
6. **Fig 8**: KD impact
7. **Fig 3**: VGAE reconstruction
8. **Fig 6**: ROC/PR curves

### Phase 4: Supplementary Materials (Week 3-4)
9. **Fig 9**: Per-attack analysis
10. **Fig 10**: Confusion matrices
11. **Fig 11**: Training dynamics
12. **Fig 12**: Computational efficiency

### Phase 5: Refinement & Integration (Week 4)
- [ ] Generate all figures at final resolution
- [ ] Ensure consistent styling across all figures
- [ ] Write figure captions (save in `figure_captions.md`)
- [ ] Create supplementary materials document
- [ ] Get feedback and iterate

---

## Technical Tools & Resources

### Primary Libraries
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import umap  # or pymde
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import shap  # for feature importance
```

### Style Template (`paper_style.mplstyle`)
```
# IEEE/ACM paper style
figure.figsize: 7, 4
figure.dpi: 300
font.size: 10
font.family: serif
font.serif: Computer Modern Roman
text.usetex: True
axes.labelsize: 10
axes.titlesize: 11
xtick.labelsize: 9
ytick.labelsize: 9
legend.fontsize: 9
lines.linewidth: 1.5
grid.alpha: 0.3
```

### Interactive Development
- **Marimo**: https://marimo.io/for-researchers
  - Reactive notebook for rapid iteration
  - Better than Jupyter for reproducible figures
  - Directly export to PDF/PNG

### Directory Structure
```
visualizations/
├── __init__.py
├── utils.py              # Common utilities (loading, styling)
├── architecture_diagram.py
├── embedding_umap.py
├── vgae_reconstruction.py
├── dqn_policy_analysis.py
├── performance_comparison.py
├── roc_pr_curves.py
├── ablation_state_space.py
├── kd_impact.py
├── per_attack_analysis.py
├── confusion_matrices.py
├── training_dynamics.py
└── computational_cost.py

figures/              # Output directory
├── fig1_architecture.pdf
├── fig2_embeddings.pdf
├── ...
└── supplementary/
    ├── ...
```

---

## Data Requirements Checklist

- [ ] **Evaluation Results**: CSV exports from all models (Teacher, Student, Student+KD)
- [ ] **DQN Predictions**: Saved alpha values + 15D states from validation set
- [ ] **Embeddings**: Saved VGAE latent (z) and GAT pre-pooling embeddings
- [ ] **Training Logs**: MLflow/TensorBoard logs for loss curves
- [ ] **GPU Monitoring**: CSV files from all training runs
- [ ] **Attack Labels**: Dataset-specific attack type mappings
- [ ] **Ablation Results**: Separate evaluation runs for 2D, 7D, 11D, 15D DQN variants

---

## Success Criteria

1. **Visual Clarity**: Non-experts can understand the main contributions
2. **Publication Quality**: 300+ DPI, vector graphics, professional styling
3. **Consistency**: Uniform colors, fonts, sizes across all figures
4. **Completeness**: Every claim in paper backed by a figure
5. **Interpretability**: DQN decisions are explainable through visualizations
6. **Reproducibility**: All figures generated from saved evaluation outputs

---

## Next Steps

1. **Immediate**: Finish evaluation runs to collect all necessary data
2. **Setup**: Create visualization infrastructure (style guide, utils)
3. **Prototype**: Generate Fig 5 (performance comparison) as proof of concept
4. **Iterate**: Refine based on feedback, then expand to all figures
5. **Integrate**: Add to paper with detailed captions

**Target Completion**: 3-4 weeks from start of evaluation phase
**Owner**: Collaborative (user + Claude Code)
**Status**: Planning phase → Ready to implement after evaluation data collected
