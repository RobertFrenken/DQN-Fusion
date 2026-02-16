## Dashboard Visualization Features

### Implemented (2026-02-16)

#### Component Architecture
- Config-driven ES module system with BaseChart base class, Registry, Theme
- PanelManager orchestrates 11 panels from declarative `panelConfig.js`
- Adding a new panel = adding an entry to `panelConfig.js`

#### Working Panels (5 original, data populated)
1. **Leaderboard** — Sortable table of all runs with metric columns (270 entries)
2. **Dataset Comparison** — Grouped bar chart comparing metrics across datasets
3. **KD Transfer** — Scatter plot showing student vs teacher metric transfer (108 pairs)
4. **Run Timeline** — Timeline scatter showing run history (70 timestamped runs)
5. **Training Curves** — Multi-series line chart (36 JSON files from 18,290 epoch_metrics)

#### New Panels (6, panel code complete, some awaiting data)
6. **Force Graph** — D3 force simulation visualizing CAN bus graph structure (data: `graph_samples.json`, 18 samples ready)
7. **Bubble Chart** — Multi-metric model comparison with size encoding by param count (data: `model_sizes.json` + `leaderboard.json`, ready)
8. **VGAE Latent Space** — Scatter of UMAP/PyMDE-reduced VGAE embeddings colored by class (data: awaiting evaluation re-run)
9. **GAT State Space** — Scatter of reduced GAT hidden representations colored by class (data: awaiting evaluation re-run)
10. **DQN Policy** — Stacked histogram of DQN alpha values by normal/attack class (data: awaiting evaluation re-run)
11. **Model Predictions** — Grouped bar chart breaking down metrics by test scenario (data: needs per-scenario metric files)

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

### Next Steps to Complete Visualizations
1. Re-run evaluation stage (generates `embeddings.npz` + `dqn_policy.json` artifacts)
2. Re-export dashboard data (`python -m pipeline.export`)
3. Commit + push to GitHub Pages
4. Verify all 11 panels render locally (`python -m http.server -d docs/dashboard`)

### Future Ideas
- Curriculum learning analysis and visualization
- Contour density overlays on embedding scatter plots
- Raw point sampling + contour distribution for UMAP/embedding visuals
- Deeper DQN state space to prediction analysis
- VGAE reconstruction error distribution plots
