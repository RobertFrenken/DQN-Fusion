# MLOps Tooling Research: Comprehensive 218-Tool Evaluation

## Context

This document is a **research deliverable**, not an implementation plan. The implementation plan is `MLOPS_MIGRATION_PLAN_V2.md` (Metaflow + W&B + Observable Framework + Cloudflare R2 + pixi). This research systematically evaluates every tool on mlops-tools.com against the project's HPC constraints to confirm no better options were missed — and to surface new tools for the visualization paths the user wants.

**Prior decisions (from V1/V2 plans)**: Snakemake is the primary pain point and is being replaced by Metaflow. SQLite DB + custom dashboard are being replaced by W&B + Observable Framework. These decisions stand. This research validates them against the full landscape.

**New question from this session**: The user wants two visualization paths — (1) Jupyter for interactive exploration and (2) JS-based dashboard with Observable-like reactivity. This research evaluates tools for both.

---

## Part 1: Full Landscape Triage (218 tools -> 15 candidates)

### Eliminated: Dead/Archived Projects
| Tool | Status | Notes |
|------|--------|-------|
| **NNI (Microsoft)** | Archived Sep 2024 | Was good for NAS/model compression. Use `torch.nn.utils.prune` + `torch.ao.quantization` instead |
| **Ploomber** | Archived Jul 2025 | Had decent SLURM support via Soopervisor. Dead now |
| **Manifold (Uber)** | Dead since Jan 2020 | Tabular classification debugging. Wrong domain |
| **Keepsake** | Deprecated | Merged into Replicate's platform |
| **Guild AI** | ~900 stars, uncertain | No SLURM support |
| **Comet ML** | Pivoting to LLM observability | Classic ML tracking deprioritized |

### Eliminated: Requires Kubernetes (OSC runs SLURM, not K8s)
Kubeflow, Argo Workflows, Flyte, MLRun, KServe, Seldon Core, Polyaxon, Bodywork, TrueFoundry, Katonic, CNVRG, Nos, Sematic, Arrikto

### Eliminated: Cloud-Only Platforms
Azure ML, SageMaker, Vertex AI, DataRobot, Dataiku, Domino, Paperspace Gradient, Valohai, Lightning AI Platform (Studios), Banana, Beam

### Eliminated: Needs Docker Daemon on Server (OSC has none)
ClearML Server (ES + MongoDB + Redis), Neptune.ai self-hosted, Sacred (MongoDB), W&B Server self-hosted

### Eliminated: NFS Incompatible
**Aim** -- RocksDB on NFS causes `RocksIOError` failures ([aimhubio/aim#1865](https://github.com/aimhubio/aim/issues/1865)). Would need remote tracking server, adding more operational overhead than current SQLite

### Eliminated: Wrong Scale / Wrong Domain
Spark, Hadoop, Delta Lake, Databricks (TB+ scale, JVM ecosystem), all feature stores (Feast, Hopsworks, Feathr, Tecton -- no feature store needed), all labeling tools (Label Studio, Labelbox, CVAT, Snorkel -- data is pre-labeled), all vector DBs (Milvus, Pinecone, Qdrant), all AutoML (AutoGluon, AutoKeras, TPOT, Auto-sklearn -- custom GNN architecture), all model serving (TF Serving, TorchServe, Triton, BentoML -- not deploying yet), all data catalogs (Amundsen, DataHub, OpenMetadata -- 6 static datasets)

### Eliminated: No SLURM Support
ZenML, Kedro, Prefect (experimental only), Luigi, Nextflow (bioinformatics, marginal gain), Azkaban (Hadoop), DVC Pipelines (weaker than Snakemake + git locking issues under concurrent SLURM)

### Eliminated: Overengineered for Single Researcher on HPC
**Determined AI** -- Needs persistent master server + PostgreSQL + container-first model. SLURM integration exists (via HPC Launcher + Singularity) but requires admin setup. Lightning adapter has known limitations (no `test_step`, no `validation_step_end`). PyG support unverified. Code refactoring into `PyTorchTrial` class is non-trivial. **Verdict**: Team/enterprise tool, not single-researcher.

**Dagster** -- Software-defined assets paradigm is elegant but requires 3 persistent services (webserver + daemon + code server). Community `dagster-slurm` exists but is less mature than Snakemake's native SLURM profile. Would require full pipeline rewrite for marginal benefit.

**Ray (full platform)** -- Head-worker architecture maps awkwardly onto SLURM. Port conflicts on shared clusters. Doesn't do pipeline orchestration. **But**: Ray Tune for HPO is good (see survivors below).

---

## Part 2: Survivors -- Tools Worth Using

### Already Decided (V2 plan, confirmed by this research)

| Tool | Role | Why it won |
|------|------|-----------|
| **Metaflow + @slurm** | Orchestration (replaces Snakemake) | Only framework with first-party SLURM decorator. Python-native DAG. Immutable artifacts. No sentinel gymnastics. Risk accepted on v0.0.4 maturity. |
| **W&B** | Experiment tracking (replaces SQLite DB) | SaaS sidesteps all NFS/SQLite issues. Best-in-class UI. Offline mode for SLURM compute nodes. Free academic tier 200GB. Lightning `WandbLogger` built-in. |
| **Optuna** | HPO | SQLite storage works on NFS. Distributed via RDB. ASHA/TPE/CMA-ES. Native Lightning integration. Zero infrastructure. |
| **Observable Framework** | Dashboard (replaces custom D3.js) | Static site generator by D3 creator. Reactive JS cells. DuckDB-WASM for in-browser SQL. Data loaders (Python->Parquet at build time). Deploys to GitHub Pages/Cloudflare. ~80% D3 code reuse. |
| **DVC** | Data versioning (keep current) | Already in use. Moving remote to Cloudflare R2. Don't expand to pipelines. |
| **pixi** | Package management | Replaces unversioned conda env with lockfile. |

### New Findings from This Research

**1. Panel (HoloViz) -- Jupyter visualization path** NEW

The user wants a Jupyter-based interactive exploration path. Panel is the best fit:
- Works *inside* Jupyter notebooks with the same code -- no rewrite to go from notebook to served app
- `panel serve notebook.ipynb` via SSH tunnel (same pattern as Datasette)
- Works with any Python viz library (Matplotlib, Bokeh, Plotly, hvPlot)
- Reactive programming model (widgets -> chart updates)
- Datashader for large datasets
- 468k monthly PyPI downloads, actively maintained
- **Use for**: Ad-hoc exploration of embeddings, training curves, attention weights, model comparisons during research
- **Not for**: Public-facing dashboard (use Observable Framework for that)

**2. Evidence -- SQL-first dashboard layer** NEW

SQL-based BI tool that generates static sites from Markdown + SQL:
- Supports SQLite and DuckDB as data sources -- maps directly to `project.db`
- Built-in chart components (bar, line, scatter, table, heatmap)
- DuckDB-WASM runs SQL in browser
- Deploys as static site (GitHub Pages compatible)
- **Simpler than Observable Framework** for standard metrics views (leaderboards, training curves)
- **Less flexible** -- can't do force-directed graphs, custom D3 visualizations
- **Decision**: Evidence for standard metrics panels (leaderboard, training curves, dataset comparison, run timeline). Observable Framework for custom D3 visualizations (force graphs, attention heatmaps, embedding scatters, DQN policy, CKA matrices).

**3. Ray Tune -- Advanced HPO** (future watch list)

If Optuna's search space becomes insufficient:
- ASHA (async successive halving), PBT (population-based training), Bayesian optimization
- Clean Lightning integration via `TuneReportCallback`
- Works with conda on SLURM (no containers needed). NERSC has validated it.
- `pip install ray[tune]` -- no server needed
- **Not now**: For 18 fixed configurations, Optuna suffices. Ray Tune becomes valuable for large-scale unknown search spaces.

**4. Captum -- GNN Interpretability** (research tool)

PyTorch-native attribution methods relevant to GNN research:
- Integrated Gradients, GradCAM, feature ablation -- all work on PyTorch models
- More rigorous than raw attention weights for interpretability claims
- Could strengthen paper's explainability section
- **Not an infrastructure tool** -- a research library to add when writing interpretability results

**5. MLflow -- Tracking fallback** (if W&B blocked by data policy)

If OSC policies prevent syncing experiment data to W&B cloud:
- File-backed on NFS, zero infrastructure
- `mlflow server` on login node (same as Datasette)
- Same NFS/SQLite concurrency caveats you already have, but cleaner API
- **Dashboard concern**: MLflow UI doesn't save dashboard state between sessions. Each visit starts fresh.
- **Verdict**: Fallback only. W&B is strictly better if data policy allows it.

---

## Part 3: Visualization Strategy (Two Paths)

### Path 1: Jupyter + Panel (interactive research exploration)

**Purpose**: Day-to-day research workflow. Explore embeddings, compare training curves, debug attention patterns, tune hyperparameters.

- Same code in notebook cells and served dashboard
- `panel serve` via SSH tunnel for persistent access
- Libraries: Panel + hvPlot + Bokeh (interactive) + Matplotlib (publication figures)
- Data source: W&B API queries or local artifacts
- **Timing**: Adopt after Metaflow+W&B migration. Panel will query W&B API instead of SQLite.

### Path 2: Evidence + Observable Framework (public-facing dashboard)

**Purpose**: Shareable results dashboard for collaborators, reviewers, publications. Replaces current `docs/dashboard/`.

- **Evidence** for standard metrics panels: leaderboard, training curves, dataset comparison, run timeline. SQL->chart with zero custom JS.
- **Observable Framework** for domain-specific visualizations: force-directed graphs, attention heatmaps, embedding scatters, DQN policy, CKA matrices. Reactive JS cells + D3.
- Both deploy as static sites to Cloudflare Workers
- DuckDB-WASM queries Parquet on R2 in both tools
- Data loaders (Python scripts) replace `pipeline/export.py`

### What replaces what

| Current Component | Path 1 (Jupyter/Panel) | Path 2 (Evidence + Observable) |
|-------------------|----------------------|-------------------------------|
| `docs/dashboard/js/charts/` (8 chart types) | hvPlot/Bokeh equivalents for exploration | Evidence for simple charts, Observable + D3 for custom |
| `docs/dashboard/js/panels/` (11 panels) | Panel widgets + layouts | Evidence pages (SQL) + Observable pages (D3) |
| `pipeline/export.py` (15 export functions) | W&B API queries | Data loaders (Python->Parquet) |
| Embeddings explorer | Panel + UMAP interactive scatter | Observable + D3 scatter |
| Attention heatmaps | Panel + Bokeh heatmap | Observable + D3 heatmap |
| Training curves | Panel + hvPlot line chart | Evidence SQL->line chart |
| Leaderboard | Panel DataFrame widget | Evidence SQL->table |
| Run timeline | Panel + Bokeh timeline | Evidence SQL->timeline |
| DQN policy | Panel + Bokeh histogram | Observable + D3 histogram |
| CKA matrix | Panel + Bokeh heatmap | Observable + D3 heatmap |

---

## Part 4: Tools Explicitly Not Worth Adopting (with reasons)

This section exists so we don't re-evaluate these tools in future sessions.

| Tool | Category | Why not |
|------|----------|--------|
| Determined AI | Platform | Persistent master + PostgreSQL + container-first. Overengineered. Lightning adapter limited. |
| Ray (full) | Platform | Cluster-within-cluster on SLURM. Doesn't do orchestration. Only Tune component is useful. |
| Dagster | Orchestration | 3 persistent services. Community SLURM plugin. Full pipeline rewrite needed. |
| Aim | Tracking | RocksDB on NFS = broken. Needs remote server as workaround. |
| ClearML | Tracking | Server needs Docker (ES+Mongo+Redis). SLURM Glue = Enterprise-only. |
| ZenML | Orchestration | No SLURM. Positions K8s migration as the answer. |
| Airflow | Orchestration | Heavy server (webserver+scheduler+worker+DB). Designed for recurring ETL, not HPC research. |
| Luigi | Orchestration | No SLURM. Older design. |
| Prefect | Orchestration | SLURM = experimental via Dask. Needs persistent server. |
| Kedro | Orchestration | No SLURM. Local execution only. |
| Neptune.ai | Tracking | Self-hosted needs K8s. SaaS is viable but W&B is better. |
| Sacred | Tracking | Needs MongoDB. OSC has no managed DB. |
| Polyaxon | Platform | K8s only. |
| MLRun | Platform | K8s only. |
| Flyte | Platform | K8s only. |
| NNI | HPO/NAS | Archived Sep 2024. Dead. |
| Ploomber | Orchestration | Archived Jul 2025. Dead. |
| CML | CI/CD | Cloud-native GPU provisioning. No SLURM integration. |
| Nextflow | Orchestration | Bioinformatics-native like Snakemake. Marginal gain, high switching cost. |
| DVC Pipelines | Orchestration | Weaker than Snakemake. Git locking under concurrent SLURM. lakeFS acquisition uncertainty. |
| Manifold | Viz/Debug | Dead since 2020. Wrong domain. |
| LynxKite | Graph platform | Graph analytics (NetworkX/cuGraph), not GNN training. No PyG integration. |
| Streamlit | Viz | Re-runs entire script on interaction. RAM scales per user. Worse than Panel for Jupyter use. |
| Gradio | Viz | Model demo tool, not analytics dashboard. |
| Voila | Viz | Spawns kernel per user (heavy). Less flexible than Panel. |

---

## Part 5: Confirmed Stack (V2 plan + new findings)

```
Orchestration:  Metaflow + @slurm  (replaces Snakemake)
Tracking:       W&B SaaS           (replaces SQLite DB + analytics)
HPO:            Optuna             (new capability)
Viz Path 1:    Panel + hvPlot     (Jupyter exploration)          <- NEW
Viz Path 2a:   Evidence           (SQL-first metrics dashboard)  <- NEW
Viz Path 2b:   Observable Fwk     (custom D3 visualizations)
Data layer:     DuckDB-WASM + Parquet on Cloudflare R2
Data version:   DVC + Cloudflare R2 remote
Env:            pixi + Apptainer
Paper:          Quarto + OJS cells
CI/CD:          GitHub Actions + ruff + mypy + pytest
Research libs:  Captum (GNN interpretability)                    <- NEW
Future HPO:     Ray Tune (if search space grows)                 <- WATCH LIST
Fallback track: MLflow (if W&B data policy blocked)
```

**What this eliminates**: ~4,400 lines of custom infrastructure code (db.py, export.py, analytics.py, Snakefile, docs/dashboard/, sentinel system, SLURM profiles)

**What stays permanently**: config/ (Pydantic), src/ (models, training, preprocessing), pipeline/stages/ (wrapped by Metaflow steps), pipeline/memory.py (GNN-specific), pipeline/cli.py (simplified)

---

## Part 6: Decisions Made

1. **Dashboard**: **Evidence for standard metrics + Observable Framework for custom viz.** Evidence handles leaderboard, training curves, dataset comparison, run timeline via SQL->chart. Observable Framework handles force-directed graphs, attention heatmaps, embedding scatters, DQN policy, CKA matrices -- anything needing custom D3.
2. **Panel timing**: **After Metaflow+W&B migration.** Panel will query W&B API for data instead of SQLite. Add to pixi environment when that's set up.
3. **Coverage**: Satisfied. 218 tools evaluated, V2 plan confirmed, new tools (Panel, Evidence, Captum) identified.

## Remaining Open Question

- **OSC data policy**: Can experiment metrics sync to W&B cloud? If not, MLflow is the fallback tracker. This needs to be checked before W&B adoption proceeds.

## Relationship to Existing Plans

This document is a **research appendix** to `MLOPS_MIGRATION_PLAN_V2.md`. It does not replace V2 -- it validates its decisions against the full tool landscape and adds:
- **Panel (HoloViz)** as the Jupyter visualization path (Phase 7 addition, after migration)
- **Evidence** as the SQL-first dashboard layer alongside Observable Framework (Phase 7B refinement)
- **Captum** as a research library for GNN interpretability (independent of infrastructure migration)
- **Ray Tune** as a future HPO upgrade path beyond Optuna (watch list)

The V2 implementation order and phasing remain unchanged.
