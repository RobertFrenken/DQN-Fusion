## Dashboard Visualization Features

### Implemented (2026-02-16)

#### Component Architecture
- Config-driven ES module system with BaseChart base class, Registry, Theme
- PanelManager orchestrates panels from declarative `panelConfig.js`
- Adding a new panel = adding an entry to `panelConfig.js`
- 10 chart types: Table, Bar, Scatter, Line, Timeline, Bubble, ForceGraph, Histogram, Heatmap, Curve

#### Working Panels (5 original, data populated)
1. **Leaderboard** — Sortable table of all runs with metric columns (270 entries)
2. **Dataset Comparison** — Grouped bar chart comparing metrics across datasets
3. **KD Transfer** — Scatter plot showing student vs teacher metric transfer (108 pairs)
4. **Run Timeline** — Timeline scatter showing run history (70 timestamped runs)
5. **Training Curves** — Multi-series line chart (36 JSON files from 18,290 epoch_metrics)

#### Phase 4 Panels (6, panel code complete, some awaiting data)
6. **Force Graph** — D3 force simulation visualizing CAN bus graph structure (data: `graph_samples.json`, 18 samples ready)
7. **Bubble Chart** — Multi-metric model comparison with size encoding by param count (data: `model_sizes.json` + `leaderboard.json`, ready)
8. **VGAE Latent Space** — Scatter of UMAP/PyMDE-reduced VGAE embeddings with Voronoi overlay option
9. **GAT State Space** — Scatter of reduced GAT hidden representations with Voronoi overlay option
10. **DQN Policy** — Stacked histogram of DQN alpha values by normal/attack class
11. **Model Predictions** — Grouped bar chart breaking down metrics by test scenario

#### Phase 5 Panels — Advanced Visualizations (7 new panels)
12. **Confusion Matrix** — Heatmap of classification confusion matrix with counts and percentages per model/scenario
13. **ROC & PR Curves** — Multi-series evaluation curves (ROC with diagonal reference, PR) with AUC annotation
14. **VGAE Reconstruction Errors** — Overlaid histograms of recon errors by class with optimal threshold line
15. **GAT Attention Weights** — Force graph with edge thickness/color by attention weight magnitude per layer
16. **Attention Patterns** — Carpet heatmap showing attention weight evolution across GAT layers
17. **Training Curve Heatmap** — Carpet heatmap of metric convergence across all runs (rows=runs, cols=epochs)
18. **Knowledge Transfer (CKA)** — Layer x layer heatmap of CKA similarity between teacher and student GAT

#### QoL Features
19. **Summary Overview** — CSS Grid of metric cards as landing panel
20. **Pareto Frontier** — Scatter of F1 vs parameter count with Pareto-optimal step-line overlay
21. **URL Hash Routing** — Deep-linking to panels via `#panel-id`, browser back/forward support
22. **Table Search** — Filter rows in leaderboard table via text input
23. **Loading Skeleton** — Shimmer animation shown while panel data loads
24. **CSS Polish** — Panel box shadows, select focus glow, improved visual hierarchy

### Chart Type Extensions (Phase 5)
- **ScatterChart** — Voronoi cell overlay option (class-colored boundaries)
- **HistogramChart** — Threshold line overlay + dynamic x-domain for non-[0,1] data
- **ForceGraph** — Attention edge weight encoding (width + color by magnitude)

### Pipeline Enhancements (Phase 5)
- **Confusion matrix** in `metrics.json` → `core.confusion_matrix` field
- **ROC/PR curves** in `metrics.json` → `additional.roc_curve` / `additional.pr_curve` (downsampled to ~200 points)
- **DQN Q-values** in `dqn_policy.json` → `q_values` field
- **GAT attention weights** → `attention_weights.npz` (per-layer alpha, edge indices, node features, up to 50 samples)
- **CKA matrix** → `cka_matrix.json` for KD runs (Linear CKA between teacher/student GAT layers)

### Export Pipeline (Phase 5 additions)
- `export_roc_curves()` → `roc_curves/{run_id}_{model}.json`
- `export_attention()` → `attention/{run_id}.json`
- `export_recon_errors()` → `recon_errors/{run_id}.json`
- `export_cka()` → `cka/{run_id}.json`

### Data Status
| Export | File | Status |
|--------|------|--------|
| Leaderboard | `leaderboard.json` | Ready |
| Per-run metrics | `metrics/{run_id}.json` | Ready |
| Training curves | `training_curves/{run_id}.json` | Ready (36 files) |
| KD transfer | `kd_transfer.json` | Ready (108 pairs) |
| Datasets | `datasets.json` | Ready (6 datasets) |
| Runs | `runs.json` | Ready (70 runs, 100% timestamps) |
| Metric catalog | `metrics/metric_catalog.json` | Ready (20 metrics) |
| Graph samples | `graph_samples.json` | Ready (18 samples) |
| Model sizes | `model_sizes.json` | Ready (6 entries) |
| Embeddings | `embeddings/{run_id}_{model}_{method}.json` | Awaiting evaluation re-run |
| DQN policy | `dqn_policy/{run_id}.json` | Awaiting evaluation re-run |
| ROC/PR curves | `roc_curves/{run_id}_{model}.json` | Awaiting evaluation re-run |
| Attention weights | `attention/{run_id}.json` | Awaiting evaluation re-run |
| Recon errors | `recon_errors/{run_id}.json` | Awaiting evaluation re-run |
| CKA matrices | `cka/{run_id}.json` | Awaiting KD evaluation re-run |

### Next Steps to Complete Visualizations
1. Re-run evaluation stage (generates all new artifacts)
2. Re-export dashboard data (`python -m pipeline.export`)
3. Commit + push to GitHub Pages
4. Verify all panels render locally (`python -m http.server -d docs/dashboard`)
