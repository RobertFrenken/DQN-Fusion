# The Poor Man's AWS: A PhD Research Platform on HPC

## The Realization

You're building the same logical system as AWS, assembled from free tiers and OSC compute:

| AWS Service | Your Equivalent | Cost |
|-------------|----------------|------|
| S3 | Cloudflare R2 | Free (10GB) / ~$1/mo |
| Kinesis / Firehose | Cloudflare Pipelines | Free (beta) + $5/mo Workers |
| Step Functions | Prefect Cloud (free) | Free |
| SageMaker Training | OSC SLURM + PyTorch Lightning | Free (university allocation) |
| SageMaker Experiments | W&B (academic tier) | Free |
| Athena | DuckDB / DuckDB-WASM | Free |
| QuickSight | Observable Framework | Free |
| Glue Data Catalog | R2 Data Catalog (Iceberg) | Free (beta) |
| CodeBuild / CI | GitHub Actions | Free |
| SageMaker Notebooks | JupyterHub on OSC (OnDemand) | Free |
| CloudFormation | pixi + Apptainer | Free |
| **Total** | | **~$5-6/month** |

The constraint that makes this interesting: you're **compute-rich** (OSC GPUs) but **cash-poor** and **infrastructure-poor** (no Docker daemon, no managed DBs, NFS filesystem). Every architectural decision flows from this.

---

## The Five Stages of Your Research Pipeline

Here's every stage of your work, end-to-end, with the tool that handles each concern:

### Stage 1: Data Ingestion & Discovery

**The job**: Find new datasets across domains (CAN bus, network traffic, IoT, maybe more for ICML generalizability), download them, validate schemas, convert to graph representations, cache processed results.

| Concern | Tool | Notes |
|---------|------|-------|
| Dataset discovery | Manual + HuggingFace Hub | Search for IDS/anomaly detection datasets |
| Raw data versioning | DVC → R2 remote | Track which version of raw data produced which results |
| Schema validation | Pandera | DataFrame-level validation at Parquet boundary |
| Graph construction | Your existing `src/preprocessing/` | Sliding window → PyG Data objects |
| Cache storage | OSC scratch (`.pt` files) | Fast local access for training. 60-day purge mitigated by DVC |
| Cache metadata | POST to Cloudflare Pipeline | "dataset X processed at time Y, N graphs, schema version Z" |

**What's new vs. current**: Pandera validation at the ingestion boundary (catch corrupt data before graph construction), and metadata flowing to the lakehouse automatically.

**What stays the same**: Your preprocessing code, DVC tracking, `.pt` cache files on scratch.

### Stage 2: Experiment Execution

**The job**: Train models (VGAE → GAT → DQN pipeline), manage SLURM resources efficiently, handle failures gracefully, track everything.

| Concern | Tool | Notes |
|---------|------|-------|
| Orchestration + DAG | Prefect (ephemeral or Cloud) | `@flow` / `@task`, DaskTaskRunner → SLURMCluster |
| SLURM job submission | dask-jobqueue SLURMCluster | Automatic sbatch, resource requests, queue management |
| Experiment tracking | W&B (academic) | WandbLogger in Lightning, offline mode if needed |
| Config management | Existing Pydantic v2 `config/` | `resolve()` function, frozen models, schema validation |
| Artifact logging | W&B Artifacts | Checkpoints, embeddings, attention weights with lineage |
| Result archival | HTTP POST → Cloudflare Pipeline → R2 | Final metrics land in lakehouse automatically |
| Failure recovery | Prefect retries + W&B resume | `wandb.init(resume="allow")` picks up interrupted runs |

**Custom code eliminated**: `pipeline/db.py` (~825 lines), `pipeline/export.py` (~805 lines), `pipeline/analytics.py` (~300 lines), sentinel `.done` files, `pipeline/Snakefile` + rules (~400 lines). That's ~2,300 lines of infrastructure code replaced by `wandb.log()`, `httpx.post()`, and Prefect decorators.

**What stays**: `config/`, `src/models/`, `src/preprocessing/`, `pipeline/stages/` (the actual training code). Prefect wraps your stages; it doesn't rewrite them.

### Stage 3: Hyperparameter Optimization & Resource Efficiency

**The job**: Find good hyperparameters without wasting GPU hours. This is where sweeps, tuning, and resource efficiency live.

| Concern | Tool | Notes |
|---------|------|-------|
| HPO search | Optuna | TPE, CMA-ES, ASHA pruning. SQLite storage on NFS (single-writer per study = safe) |
| Parallelism | SLURM job arrays | `--array=1-32%8`: 32 trials, 8 concurrent. Each runs 1 Optuna trial. Zero infrastructure |
| Trial tracking | W&B + Optuna callback | `WeightsAndBiasesCallback` logs every trial to W&B automatically |
| Early stopping | Optuna MedianPruner / ASHA | Kill bad trials early, save GPU hours |
| Resource efficiency | Lightning's `precision="16-mixed"` | Already available. Halves memory, ~1.3x speedup |
| Gradient accumulation | Lightning `accumulate_grad_batches` | Simulate larger batch sizes without more memory |
| Future scale-up | Ray Tune | When search space grows beyond Optuna. SLURM-compatible via conda |

**Why Optuna over W&B Sweeps or Ray**: Optuna works fully offline on SLURM with SQLite storage and job arrays. No agent process needed. W&B Sweeps need a persistent agent; Ray Tune needs a head node. Optuna is the zero-infrastructure option, and at your scale (tens of trials, not thousands), it's sufficient.

**Why not Ray now**: Ray's value is distributed training across multiple GPUs/nodes and massive parallel search. You're training on single V100s. Ray becomes relevant when you move to multi-GPU training on A100s (Cardinal/Ascend clusters) or need PBT (population-based training).

### Stage 4: Sandboxing & Rapid Iteration

**The job**: You need a place to test ideas fast — "what if I change the attention mechanism?", "does this fusion approach work?", "let me plot this embedding space." This is your research workbench.

| Concern | Tool | Notes |
|---------|------|-------|
| Interactive exploration | JupyterHub via OSC OnDemand | Already available. GPU-enabled notebooks on compute nodes |
| Quick experiments | Prefect ephemeral mode | Run a flow locally, no server. Or just call `stage_fn.fn()` directly |
| Visualization (explore) | matplotlib + seaborn in notebooks | For quick looks during research |
| Visualization (interactive) | Panel / hvPlot (future) | When you need interactive widgets. Deferred to post-migration |
| Data access in notebooks | DuckDB + Parquet from R2 | `import duckdb; duckdb.sql("SELECT * FROM read_parquet('s3://...')")` |
| W&B data in notebooks | `wandb.Api()` | Pull any run's metrics/artifacts programmatically |
| Version control | Git + DVC | Code in Git, data/models in DVC |

**Key insight**: The sandbox doesn't need special infrastructure. JupyterHub on OSC already gives you GPU notebooks. The improvement is that your notebook can now *query* the lakehouse (R2 via DuckDB) and pull W&B runs, instead of manually loading `.pt` files and parsing `config.json` by hand.

### Stage 5: Analysis, Visualization & Publication

**The job**: Generate figures for TMLR paper, build interactive dashboard for reviewers, export publication-ready artifacts.

| Concern | Tool | Notes |
|---------|------|-------|
| Data source | R2 (Iceberg/Parquet via Pipelines) | The lakehouse is the single source of truth for all analysis |
| Public dashboard | Observable Framework | DuckDB-WASM queries Parquet on R2 in browser. D3.js charts. Static site |
| Paper figures | Quarto + OJS cells | Interactive figures in TMLR Beyond PDF format |
| Static figures | matplotlib (exported from notebooks) | For traditional PDF submission backup |
| Dashboard hosting | Cloudflare Workers / Pages | Free, global CDN, R2 binding for zero-CORS data access |
| Model cards | HuggingFace Hub | 18 models (6 datasets × 3 architectures) with documentation |
| Code/data DOI | Zenodo | Permanent archival for citation |
| Reproducibility | W&B Report | Interactive appendix linked from paper |

**The data flow for a figure**: R2 Parquet → DuckDB-WASM in Observable page → D3.js/Observable Plot → rendered in browser (dashboard) or embedded in Quarto (paper). Same data, same query, two output formats.

---

## The Unified Data Architecture

```
                         ┌──────────────────────────┐
                         │    Cloudflare R2          │
                         │    "The Lakehouse"        │
                         │                           │
                         │  ┌─────────────────────┐  │
                         │  │ Iceberg Tables       │  │
                         │  │ (via R2 Data Catalog)│  │
                         │  │  • experiment_runs   │  │
                         │  │  • epoch_metrics     │  │
                         │  │  • dataset_catalog   │  │
                         │  │  • model_registry    │  │
                         │  └────────▲────────────┘  │
                         │           │                │
                         │  ┌────────┴────────────┐  │
                         │  │ Raw Files            │  │
                         │  │  • DVC data objects  │  │
                         │  │  • model checkpoints │  │
                         │  │  • embeddings (.npz) │  │
                         │  └─────────────────────┘  │
                         └──────▲──────▲──────▲──────┘
                                │      │      │
              ┌─────────────────┘      │      └─────────────────┐
              │                        │                        │
    ┌─────────┴──────────┐   ┌────────┴─────────┐   ┌─────────┴──────────┐
    │ Cloudflare Pipeline│   │   DVC push        │   │ Observable / Quarto│
    │ (HTTP POST → R2)   │   │   (bulk upload)   │   │ (DuckDB-WASM read) │
    │ structured metrics │   │   raw data/models │   │ dashboards, paper   │
    └─────────▲──────────┘   └────────▲─────────┘   └────────────────────┘
              │                        │
              │                        │
┌─────────────┴────────────────────────┴─────────────────────────────────┐
│                        OSC Pitzer / Cardinal                           │
│                                                                        │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐      │
│  │ Login Node   │    │  Compute Nodes (SLURM)                   │      │
│  │              │    │                                          │      │
│  │  Prefect     │    │  @task train_vgae(cfg):                  │      │
│  │  flow runner │───▶│    wandb.init(project="kd-gat")          │      │
│  │  (ephemeral) │    │    trainer.fit(model, datamodule)        │      │
│  │              │    │    wandb.log(metrics)                ───────▶ W&B│
│  │  DaskTask    │    │    sync_to_lakehouse(final_results)  ───────▶ R2 │
│  │  Runner      │    │    return artifacts                      │      │
│  │  → SLURMClust│    │                                          │      │
│  └──────────────┘    └──────────────────────────────────────────┘      │
│                                                                        │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐      │
│  │ JupyterHub   │    │  Optuna HPO (SLURM arrays)              │      │
│  │ (OnDemand)   │    │  #SBATCH --array=1-32%8                  │      │
│  │ sandbox +    │    │  Each trial: optuna.trial → train →      │      │
│  │ rapid iter   │    │  wandb.log → sync_to_lakehouse           │      │
│  └──────────────┘    └──────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## What You Delete From DQN-Fusion

Based on the repo structure, here's what gets replaced:

| Current | Lines | Replaced By |
|---------|-------|-------------|
| `pipeline/Snakefile` + `pipeline/rules/*.smk` | ~400 | Prefect flows |
| `pipeline/db.py` (SQLite write-through, migrations) | ~825 | W&B + Cloudflare Pipeline |
| `pipeline/export.py` (15 export functions) | ~805 | `sync_to_lakehouse()` (~20 lines) |
| `pipeline/analytics.py` | ~300 | W&B Workspaces + DuckDB queries |
| `docs/dashboard/` (custom D3.js) | ~2000 | Observable Framework |
| `profiles/slurm/` | ~50 | Prefect DaskTaskRunner config |
| `experimentruns/` (manual tracking) | varies | W&B (automatic) |
| Sentinel `.done` files | concept | Prefect task state |
| **Total removed** | **~4,400+** | |

**What stays permanently** (the actual science):
- `config/` — Pydantic config system
- `src/models/` — VGAE, GAT, DQN architectures
- `src/preprocessing/` — graph construction
- `pipeline/stages/` — training logic (wrapped by Prefect tasks)
- `pipeline/memory.py`, `pipeline/tracking.py` — GPU utilities
- `tests/` — all existing tests

---

## Consolidated Tool Stack

```
COMPUTE (free, university allocation)
  └─ OSC Pitzer (V100) / Cardinal (A100/H100)

ORCHESTRATION ($0)
  └─ Prefect Cloud free tier (7-day retention, disposable)
  └─ prefect-dask + dask-jobqueue (SLURM submission)

EXPERIMENT TRACKING ($0)
  └─ W&B academic tier (200GB, permanent)

HPO ($0)
  └─ Optuna (SQLite local, SLURM arrays)
  └─ Ray Tune (future, when multi-GPU)

DATA LAKEHOUSE ($5/mo for Workers plan, R2 free tier)
  └─ Cloudflare R2 (storage, zero egress)
  └─ Cloudflare Pipelines (HTTP ingest → Iceberg/Parquet)
  └─ R2 Data Catalog (Iceberg table management)
  └─ DVC (raw data + model versioning → R2)

QUERY LAYER ($0)
  └─ DuckDB (local analytical queries, notebooks)
  └─ DuckDB-WASM (browser-side queries for dashboard)

VISUALIZATION & PUBLICATION ($0)
  └─ Observable Framework (interactive dashboard)
  └─ Quarto + OJS (TMLR Beyond PDF paper)
  └─ Cloudflare Pages (hosting)

DEVELOPMENT ($0)
  └─ JupyterHub (OSC OnDemand, GPU notebooks)
  └─ pixi (reproducible environments)
  └─ GitHub Actions (CI: ruff + mypy + pytest)

PUBLICATION ($0)
  └─ HuggingFace Hub (models + datasets)
  └─ Zenodo (DOI for citation)

TOTAL: ~$5/month
```

---

## The Missing Angles You Asked About

### 1. Data Ingestion Front-End (finding new datasets)

This was under-discussed. For ICML generalizability, you need datasets beyond CAN bus. The ingestion pattern should be:

```python
@task
def ingest_dataset(source: str, name: str, domain: str):
    """Generic dataset ingestion: download → validate → convert → cache → register."""
    raw_path = download_from_source(source)  # HuggingFace, Kaggle, direct URL
    
    # Schema validation (Pandera)
    df = pd.read_parquet(raw_path)
    schema.validate(df)  # catches corrupt/misformatted data
    
    # Convert to graph representation
    graphs = your_graph_construction(df, domain=domain)
    
    # Cache locally for training
    cache_path = save_to_scratch(graphs, name)
    
    # Register in lakehouse
    sync_to_lakehouse({
        "dataset": name, "domain": domain, 
        "n_graphs": len(graphs), "source": source,
        "schema_version": "v1", "cached_at": cache_path
    })
    
    # Version raw data
    dvc_add_and_push(raw_path)
    
    return cache_path
```

The key is making this **domain-agnostic**. Your current preprocessing is CAN-bus-specific. For ICML, you'll need a `domain` parameter that dispatches to different graph construction strategies (time-series → temporal graph, network traffic → flow graph, etc.).

### 2. Offloading Custom Code

The ~4,400 lines of custom infrastructure in DQN-Fusion gets replaced by configuration and API calls:

- **Tracking**: `pipeline/db.py` → `wandb.log()` + `wandb.log_artifact()`
- **Export**: `pipeline/export.py` → `httpx.post()` to Cloudflare Pipeline
- **Orchestration**: Snakefile → Prefect `@flow` / `@task`
- **Dashboard**: `docs/dashboard/` → Observable Framework pages
- **Analytics**: `pipeline/analytics.py` → DuckDB queries on R2 Parquet

Your research code (`src/`, `config/`, `pipeline/stages/`) doesn't change. You're replacing plumbing, not science.

### 3. Sweeps & Resource Efficiency

Covered in Stage 3 above. The progression is:

1. **Now**: Optuna + SLURM arrays (zero infrastructure, sufficient for your scale)
2. **Soon**: Mixed precision training + gradient accumulation (free speedup)
3. **When needed**: Ray Tune for PBT / distributed search across A100 nodes
4. **When needed**: Multi-GPU training with Lightning's `strategy="ddp"`

### 4. Notebook ↔ Lakehouse Connection

Your JupyterHub notebooks on OSC can query R2 directly:

```python
import duckdb

# Connect to your R2 lakehouse
con = duckdb.connect()
con.sql("""
    INSTALL httpfs; LOAD httpfs;
    SET s3_endpoint='ACCOUNT.r2.cloudflarestorage.com';
    SET s3_access_key_id='...';
    SET s3_secret_access_key='...';
    SET s3_region='auto';
""")

# Query experiment results
df = con.sql("""
    SELECT dataset, model_type, MAX(f1) as best_f1
    FROM read_parquet('s3://kd-gat-data/experiment_runs/*.parquet')
    GROUP BY dataset, model_type
    ORDER BY best_f1 DESC
""").df()
```

Same data, accessible from: notebooks (DuckDB), dashboard (DuckDB-WASM), paper figures (Quarto OJS), and W&B (via API). One lakehouse, four consumption patterns.

---

## Implementation Order (Revised)

| # | Action | Time | Unblocks |
|---|--------|------|----------|
| 1 | `pip install wandb && wandb login` + instrument 1 stage | 4 hrs | Live tracking |
| 2 | Cloudflare account + R2 bucket + DVC remote | 1 hr | Data safety |
| 3 | `pip install prefect prefect-dask dask-jobqueue` + POC flow | 4 hrs | Orchestration |
| 4 | Cloudflare Pipeline (HTTP endpoint → R2 Parquet) | 2 hrs | Lakehouse ingest |
| 5 | pixi.toml + lockfile | 2 hrs | Reproducibility |
| 6 | GitHub Actions CI | 2 hrs | Code quality |
| 7 | Optuna integration for 1 model | 4 hrs | HPO |
| 8 | Observable Framework dashboard | 2-4 wks | Public results |
| 9 | Quarto paper skeleton | 1-2 wks | TMLR submission |
| 10 | Delete old infrastructure code | 1 day | Clean repo |

Items 1-4 are the critical path. Once those work, you have: W&B tracking live experiments, Prefect orchestrating SLURM jobs, and results automatically landing in R2 as Parquet. Everything else builds on that foundation.
