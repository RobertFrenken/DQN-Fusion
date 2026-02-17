# MLOps Infrastructure Migration Plan

## Context

The KD-GAT project has outgrown its custom tooling. Current pain points: DB write-through disabled on compute nodes (tracking only works after full pipeline success), Snakemake sentinel file gymnastics for re-runs, DVC remote on 90-day-purge scratch, no environment lockfile, no CI/CD, no live monitoring, and a manual 15-function export pipeline for the dashboard.

This plan migrates to a top-tier research infrastructure stack that's scalable, reproducible, and publication-ready for TMLR Beyond PDF. The user prioritizes long-term scalability over short-term convenience, and wants to think bigger than this single repo.

**Decisions made**:
- Cloudflare R2 for storage (zero egress, ~$1/mo)
- Cloudflare Workers for dashboard hosting (tighter R2 integration, no CORS)
- Metaflow @slurm for orchestration (aggressive, replacing Snakemake)
- W&B for experiment tracking (academic tier, 200GB free)
- Optuna for HPO (offline on SLURM, W&B logging)
- pixi + Apptainer for environment reproducibility
- Observable Framework for companion dashboard (D3 creator's platform, 80% code reuse)
- Quarto for TMLR Beyond PDF paper (reactive OJS figures inline with narrative)
- DuckDB-WASM + Parquet on R2 as the data layer (SQL in browser, eliminates export pipeline)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                     KD-GAT MLOps Stack (Target)                      │
├──────────────┬───────────────┬───────────────┬───────────────────────┤
│  Env & Pkg   │ Orchestration │   Tracking    │  Data & Artifacts     │
│  pixi        │ Metaflow      │   W&B (SaaS)  │  DVC + Cloudflare R2  │
│  Apptainer   │ @slurm        │   Optuna      │  DuckDB-WASM + Parq.  │
├──────────────┴───────────────┴───────────────┴───────────────────────┤
│  CI/CD: GitHub Actions + ruff + mypy + pytest                        │
│  Paper: Quarto + Observable JS (TMLR Beyond PDF)                     │
│  Dashboard: Observable Framework + DuckDB-WASM on Cloudflare Workers │
│  Registry: W&B Artifacts (teacher → student → fused lineage)         │
│  Publish: HuggingFace Hub + model cards + ONNX export + Zenodo DOI   │
└──────────────────────────────────────────────────────────────────────┘
```

**Data flow** (replaces current 15-function export pipeline):
```
Training → W&B (live metrics) → SQLite DB (archival)
                                      ↓
                              Export as Parquet
                                      ↓
                              Upload to Cloudflare R2
                                      ↓
              ┌─────────────────────────────────────┐
              │  Browser (DuckDB-WASM)              │
              │  SQL queries on Parquet from R2      │
              │  → Observable Framework / D3.js      │
              │  → Quarto paper (OJS cells)          │
              └─────────────────────────────────────┘
```

---

## Phase 1: Foundation — Environment & Package Management

### 1A. pixi for package management (2-3 hours)

Replace the unversioned conda `gnn-experiments` with pixi. Single lockfile covers conda (PyTorch, CUDA, cuDNN) + PyPI (PyG, Lightning, Pydantic).

**Create**: `pixi.toml`, auto-generates `pixi.lock` (checked into git)

```toml
[workspace]
name = "kd-gat"
channels = ["https://prefix.dev/conda-forge"]
platforms = ["linux-64"]

[system-requirements]
cuda = "12.0"

[dependencies]
python = ">=3.11,<3.13"
pytorch-gpu = ">=2.5"
cuda-version = "12.6.*"

[pypi-dependencies]
torch_geometric = ">=2.6"
pytorch-lightning = ">=2.4"
pydantic = ">=2.0"
metaflow = ">=2.12"
wandb = ">=0.18"
optuna = ">=4.0"
dvc = {version = ">=3.0", extras = ["s3"]}
quarto-cli = "*"

[tasks]
test = "pytest tests/ -v"
train = "python -m pipeline.cli"
export-parquet = "python -m pipeline.export_parquet"
build-dashboard = "npm run build --prefix observable-dashboard"
build-paper = "quarto render paper/"

[feature.dev.dependencies]
pytest = ">=8.0"
ruff = ">=0.9"
mypy = ">=1.13"
pre-commit = ">=4.0"
```

**NFS workaround**: `PIXI_CACHE_DIR=/fs/scratch/PAS1266/pixi_cache`, `PIXI_HOME=/fs/scratch/PAS1266/pixi_home`

**Install**: `curl -fsSL https://pixi.sh/install.sh | bash`

### 1B. Apptainer container (4 hours, after 1A)

**Create**: `container/kd-gat.def`

- Base: `nvcr.io/nvidia/pytorch:25.02-py3` (PyTorch 2.6 + CUDA 12.8)
- Build on compute node: `apptainer build --fakeroot kd-gat.sif container/kd-gat.def`
- Store: `/fs/scratch/PAS1266/containers/kd-gat.sif`
- Push to GHCR: `apptainer push kd-gat.sif oras://ghcr.io/robertfrenken/kd-gat:v1.0`
- Run: `srun apptainer exec --nv --cleanenv --bind /fs/scratch/PAS1266:/scratch ...`

---

## Phase 2: Data — Cloudflare R2 (1 hour)

### 2A. DVC cloud remote

Replace the 90-day-purge scratch remote with Cloudflare R2. ~$1/mo for 75GB, zero egress.

```bash
dvc remote add -d r2 s3://kd-gat-data/
dvc remote modify r2 endpointurl https://ACCOUNT_ID.r2.cloudflarestorage.com
dvc remote modify r2 region auto
dvc remote modify --local r2 access_key_id 'KEY'
dvc remote modify --local r2 secret_access_key 'SECRET'
dvc push  # uploads all 12 existing .dvc-tracked items
```

**Files modified**: `.dvc/config`, `.dvc/config.local` (git-ignored)

### 2B. Parquet export for dashboard data layer

**Create**: `pipeline/export_parquet.py`

Export `data/project.db` tables to Parquet files for DuckDB-WASM consumption:
- `metrics.parquet` — all evaluation metrics
- `epoch_metrics.parquet` — training curves
- `runs.parquet` — run metadata + configs
- `datasets.parquet` — dataset catalog

Upload to R2 bucket (same bucket, `/dashboard/` prefix). This replaces most of the 15 functions in `pipeline/export.py` — instead of pre-aggregating into static JSON, we serve raw Parquet and let DuckDB-WASM query it in the browser.

Artifact files (`embeddings.npz`, `dqn_policy.json`, `attention_weights.npz`) stay as JSON/binary on R2 — fetched directly by D3 charts.

---

## Phase 3: Experiment Tracking — W&B (4-6 hours)

### 3A. Setup

1. Apply for academic tier at wandb.ai/site/research — free 200GB, all Pro features
2. `wandb login` on login node, API key stored in `~/.netrc`

### 3B. Instrument training stages

**Modify**:
- `pipeline/stages/autoencoder.py` — add `WandbLogger` to Lightning Trainer
- `pipeline/stages/curriculum.py` — add `WandbLogger`
- `pipeline/stages/fusion.py` — add `wandb.init()` + `wandb.log()`
- `pipeline/stages/evaluation.py` — log metrics + log artifacts (embeddings, policy)
- `pipeline/cli.py` — W&B run lifecycle, config logging

```python
from lightning.pytorch.loggers import WandbLogger
logger = WandbLogger(project="kd-gat", config=cfg.model_dump(),
    name=f"{cfg.model_type}_{cfg.scale}_{cfg.dataset}",
    tags=[cfg.model_type, cfg.scale, cfg.dataset])
trainer = Trainer(logger=logger, ...)
```

**Online/offline**: Test `WANDB_MODE=online` on OSC compute nodes first (outbound HTTPS works). Fall back to offline + `wandb sync` if needed.

### 3C. W&B Artifacts for KD lineage

Log checkpoints with explicit dependency declarations:
- Teacher: `run.log_artifact(wandb.Artifact("vgae-large-{dataset}", type="model"))`
- Student: `run.use_artifact("vgae-large-{dataset}:latest")` then `run.log_artifact(student_artifact)`
- DQN: `run.use_artifact("gat-large-{dataset}:latest")` + `run.use_artifact("gat-small-kd-{dataset}:latest")`

W&B renders teacher→student→fused as a browsable lineage DAG.

### 3D. W&B Alerts + SLURM notifications

- `wandb.alert()` for training anomalies (loss spikes, GPU util drops)
- `#SBATCH --mail-type=FAIL,END --mail-user=rf15@osc.edu` for job-level events
- Optional: Slack webhook `curl` in SLURM epilog for real-time Slack notifications

---

## Phase 4: Hyperparameter Optimization — Optuna (4 hours)

Optuna stores trial state in SQLite on NFS — works fully offline on SLURM. W&B logs each trial.

**Create**: `pipeline/hpo.py`

```python
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32])
    cfg = resolve("gat", "small", dataset="hcrl_sa",
                  overrides={"training.lr": lr, "gat.latent_dim": latent_dim})
    metrics = run_curriculum(cfg)
    return metrics["f1"]

study = optuna.create_study(study_name="gat-small-hpo",
    storage="sqlite:///data/optuna.db", direction="maximize", load_if_exists=True)
study.optimize(objective, n_trials=1)
```

SLURM array: `#SBATCH --array=1-32%8` — 32 trials, 8 concurrent, each runs 1 trial.

---

## Phase 5: Orchestration — Metaflow @slurm (2-4 weeks)

### 5A. Install and configure

```bash
pip install metaflow metaflow-slurm
```

Local metadata store (no server):
```json
// ~/.metaflowconfig/config.json
{
    "METAFLOW_DEFAULT_DATASTORE": "local",
    "METAFLOW_DATASTORE_SYSROOT_LOCAL": "/fs/scratch/PAS1266/metaflow"
}
```

### 5B. Pipeline flow

**Create**: `flows/kdgat_flow.py`

Python-native DAG with `@step` + `@slurm` decorators wrapping existing stage code. Key structure:

```
start → preprocess → train_vgae_large → [train_gat_large, train_gat_small_kd] → join → train_dqn → evaluate → end
```

Each `@slurm` step specifies `gpu=1, memory=128000, time_minutes=360`. Artifacts (checkpoints, embeddings) pass between steps automatically via Metaflow's content-addressed datastore.

### 5C. Migration strategy

1. Week 1: VGAE-only flow (2 steps), test with one dataset
2. Week 2: Verify SLURM submission, artifact storage, run history
3. Week 3: Full pipeline, foreach over datasets
4. Week 4: All 6 datasets, compare with Snakemake outputs
5. After validation: Remove Snakemake files

**Fallback**: Stage code is unchanged — only the orchestration wrapper changes. Snakemake stays until Metaflow is proven.

**Note**: metaflow-slurm is v0.0.4 (beta). Accept the risk per user decision. The `@checkpoint` decorator provides automatic preemption recovery for long V100 jobs.

---

## Phase 6: CI/CD (2 hours)

### 6A. GitHub Actions

**Create**: `.github/workflows/ci.yml`

```yaml
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/ruff-action@v3
      - run: pip install -e ".[dev]" && mypy src/ config/ pipeline/
      - run: pytest tests/ -m "not slurm and not gpu" -v
```

Free GitHub-hosted runners for CPU tests. GPU/SLURM tests stay manual.

### 6B. Pre-commit hooks

**Create**: `.pre-commit-config.yaml`

- `ruff` (lint + format, replaces Black/isort/Flake8)
- `check-yaml`, `check-added-large-files` (prevent accidental .pt commits), `detect-private-key`

---

## Phase 7: Dashboard & Paper — Observable Framework + Quarto + Cloudflare

This is the most significant architectural change. The current stack (15-function export → static JSON → hand-built D3.js ES modules → GitHub Pages) becomes:

**New stack**: Parquet on R2 → DuckDB-WASM in browser → Observable Framework dashboard + Quarto paper → Cloudflare Workers hosting

### 7A. Cloudflare Workers hosting (replaces GitHub Pages)

Migrate dashboard hosting from GitHub Pages to Cloudflare Workers:
- R2 binding is internal (no CORS config, no public bucket URLs needed)
- Free: 100K requests/day, unlimited bandwidth
- 300+ edge locations (better CDN than GitHub's Fastly)
- Static assets served natively by Workers

Deploy via `wrangler deploy` or GitHub Actions → Cloudflare Workers.

### 7B. Observable Framework dashboard (2-4 weeks)

**Create**: `observable-dashboard/` directory

Replace `docs/dashboard/` (hand-built ES modules) with Observable Framework:
- 8 existing D3 chart classes (`ForceGraph`, `ScatterChart`, `LineChart`, etc.) import directly as ES modules (~80% code reuse)
- `PanelManager` + `panelConfig` routing replaced by Framework's native page-per-panel routing
- Static charts (bar, scatter, line, histogram, heatmap) migrate to Observable Plot (70% less code)
- `ForceGraph` stays as raw D3 (Plot has no force layout)
- Data loaders (Python scripts) replace `pipeline/export.py` for build-time data
- `DuckDBClient` provides runtime SQL on Parquet from R2
- Built-in reactive cells for interactivity (dropdowns, sliders → chart updates)
- Theme: `near-midnight` (dark) matches existing GitHub-dark palette with ~50 lines CSS override

**Pages** (one `.md` file each):
1. Overview (summary stats from DuckDB query)
2. Leaderboard (sortable table, DuckDB GROUP BY)
3. Dataset Comparison (bar chart)
4. KD Transfer (scatter: teacher vs student F1)
5. Training Curves (line chart, per-epoch from epoch_metrics.parquet)
6. Run Timeline (timeline chart)
7. CAN Bus Graph (ForceGraph — raw D3, graph_samples from R2)
8. VGAE Latent Space (scatter — embeddings from R2)
9. GAT State Space (scatter — embeddings from R2)
10. DQN Policy (histogram — dqn_policy.json from R2)
11. Attention Heatmap (heatmap — attention_weights from R2)

**Data loaders** (`observable-dashboard/src/data/`):
- `metrics.parquet.py` — queries SQLite DB, outputs Parquet
- `epoch_metrics.parquet.py` — queries SQLite DB, outputs Parquet
- `runs.json.py` — queries SQLite DB, outputs JSON

These run at `npm run build` time. At runtime, DuckDB-WASM queries the baked Parquet + fetches artifact files from R2.

### 7C. Quarto paper for TMLR Beyond PDF (1-2 weeks)

**Create**: `paper/` directory

Write the TMLR submission as a Quarto document with embedded Observable JS interactive figures:

```
paper/
  index.qmd          # Main paper narrative
  _quarto.yml         # Project config
  references.bib      # Bibliography
  assets/             # Static images
  figures/            # Interactive OJS figure cells
```

**Interactive figures embedded in paper** (3-4 key "money shots"):
1. KD transfer scatter (teacher vs student F1, reactive dataset selector)
2. UMAP latent space (color by attack type, hover for details)
3. Training curves overlay (teacher vs student convergence, epoch slider)
4. GAT attention heatmap (layer selector)

Each figure uses `{ojs}` cells with DuckDB-WASM querying the same Parquet data on R2:

```qmd
```{ojs}
db = DuckDBClient.of({metrics: FileAttachment("data/leaderboard.parquet")})
result = db.sql`SELECT dataset, model_type, MAX(value) as f1
  FROM metrics WHERE metric_name='f1' GROUP BY dataset, model_type`
Plot.plot({ marks: [Plot.dot(result, {x: "dataset", y: "f1", fill: "model_type"})] })
```
```

**Deployment**: Both paper and dashboard deploy from the same repo:
```
docs/
  index.html          # Landing page (redirect or overview)
  paper/              # Quarto render output
  dashboard/          # Observable Framework build output
```

Hosted on Cloudflare Workers — single `wrangler.toml` serves both.

**For TMLR submission**: The interactive figures compile to self-contained HTML bundles placed in the submission's `assets/html/` folder. The companion dashboard is linked as "Explore full results at [URL]".

### 7D. Progressive disclosure

- **In the paper**: 3-4 interactive figures with the key results
- **Dashboard link in paper**: Full 11-panel exploration for reviewers wanting depth
- **W&B Report URL**: Reproducibility appendix with live training data
- **HuggingFace Spaces**: Model demo (upload CAN log → classification), if needed

---

## Phase 8: Model Registry & Publication

### 8A. W&B Model Registry

Promote best models: `Staging` → `Production` → `Archived`. Tag paper-cited models `paper-v1`.

### 8B. ONNX export

**Create**: `pipeline/stages/export_onnx.py`

Export student GAT + DQN policy to ONNX for edge deployment:
```python
torch.onnx.export(model, (x, edge_index), "gat_small.onnx", dynamo=True,
    dynamic_axes={"node_features": {0: "num_nodes"}, "edge_index": {1: "num_edges"}})
```

Target: ~5-20MB student models for automotive gateway devices via ONNX Runtime.

### 8C. Publication artifacts

- **HuggingFace Hub**: Teacher + student + fused model weights (18 models across 6 datasets) with model cards
- **Zenodo**: DOI for code + models (CERN-backed, 5+ year archival, free)
- **W&B Report**: Interactive reproducibility appendix
- Apply for **AWS Research Credits** ($5K) in parallel for future storage flexibility

---

## Implementation Order

| # | Phase | Effort | Dependency | What it unblocks |
|---|-------|--------|------------|------------------|
| 1 | **R2 DVC remote** (Phase 2A) | 1 hour | None | Data safety (90-day purge risk) |
| 2 | **pixi setup** (Phase 1A) | 2-3 hours | None | Reproducible env |
| 3 | **W&B setup + instrument 1 stage** (Phase 3A-B) | 4 hours | None | Live monitoring |
| 4 | **CI/CD + pre-commit** (Phase 6) | 2 hours | None | Code quality |
| 5 | **Parquet export to R2** (Phase 2B) | 3 hours | After #1 | Dashboard data layer |
| 6 | **W&B Artifacts + Alerts** (Phase 3C-D) | 3 hours | After #3 | KD lineage |
| 7 | **Optuna HPO** (Phase 4) | 4 hours | After #3 | Structured sweeps |
| 8 | **Observable Framework dashboard** (Phase 7B) | 2-4 weeks | After #5 | Modern dashboard |
| 9 | **Metaflow migration** (Phase 5) | 2-4 weeks | After #3 | Orchestration |
| 10 | **Cloudflare Workers hosting** (Phase 7A) | 4 hours | After #8 | Hosting migration |
| 11 | **Quarto paper** (Phase 7C) | 1-2 weeks | After #8 | TMLR submission |
| 12 | **Apptainer container** (Phase 1B) | 4 hours | After #2 | Cross-cluster portability |
| 13 | **ONNX export** (Phase 8B) | 4 hours | After models trained | Edge deployment |
| 14 | **HF Hub + Zenodo** (Phase 8C) | 1 day | Before submission | Publication |

Items 1-4 are independent — start in parallel. Items 8 and 9 can also run in parallel.

---

## What Gets Removed (after each phase is proven)

| Component | Lines | Replaced By | When |
|-----------|-------|-------------|------|
| `pipeline/Snakefile` + `pipeline/rules/*.smk` | ~400 | Metaflow flow | After Phase 5 |
| `pipeline/db.py` | ~825 | W&B tracking | After Phase 3 |
| `pipeline/export.py` (15 functions) | ~805 | Parquet export + DuckDB-WASM | After Phase 7B |
| `pipeline/analytics.py` | ~300 | W&B Workspaces + Optuna | After Phase 3+4 |
| `docs/dashboard/` (D3.js ES modules) | ~2000 JS | Observable Framework | After Phase 7B |
| `scripts/export_dashboard.sh` | ~50 | `wrangler deploy` | After Phase 7A |
| `profiles/slurm/` | ~50 | Metaflow @slurm | After Phase 5 |
| Sentinel `.done` files | concept | Metaflow step completion | After Phase 5 |

**Total removed**: ~4,400 lines of custom infrastructure code

**What stays permanently**:
- `config/` — Pydantic config system (no replacement needed — superior to Hydra)
- `src/` — domain models, training, preprocessing
- `pipeline/stages/` — stage implementations (wrapped by Metaflow steps)
- `pipeline/cli.py` — simplified entry point (remove DB calls, keep config + dispatch)
- `pipeline/memory.py`, `pipeline/tracking.py` — GPU utilities
- `data/project.db` — archived queryable reference

---

## Verification Plan

1. **pixi**: `pixi install && pixi run test` reproduces pytest from conda env
2. **DVC R2**: `dvc push && dvc pull` round-trip on one dataset
3. **W&B**: VGAE training on SLURM → metrics in W&B dashboard within minutes
4. **Optuna**: 4-trial SLURM array → all trials in W&B + Optuna DB
5. **Parquet**: `export_parquet.py` → files on R2 → DuckDB-WASM query in browser returns correct leaderboard
6. **Observable Framework**: `npm run build` → all 11 panels render with live DuckDB queries
7. **Quarto**: `quarto render paper/` → interactive figures work in HTML output
8. **Cloudflare Workers**: `wrangler deploy` → dashboard + paper accessible at production URL
9. **Metaflow**: `python flows/kdgat_flow.py run --dataset hcrl_sa` → all steps via SLURM
10. **Apptainer**: `apptainer exec --nv kd-gat.sif python -c "import torch; print(torch.cuda.is_available())"` → True
11. **CI**: Push to branch → GitHub Actions passes ruff + mypy + pytest
