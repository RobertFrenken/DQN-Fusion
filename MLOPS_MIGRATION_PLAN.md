# MLOps Tooling Migration Plan

## Context

The current KD-GAT pipeline uses three custom/repurposed systems that have become fragile:

1. **Snakemake** (orchestration) -- PRIMARY PAIN POINT: Bioinformatics-native, sentinel files + content-based triggers to fight NFS cascades, overwrites past runs but needs them for the DAG, manual `find ... -delete` incantations to re-run evaluation
2. **Custom SQLite DB** (tracking): Write-through disabled on SLURM compute nodes (relies on `onsuccess` backfill), concurrent NFS write corruption risk
3. **Custom JS Dashboard** (visualization) -- SECONDARY PRIORITY: Manual 15-function export pipeline (DB + filesystem -> JSON -> GitHub Pages), no live interactivity

**User priorities**: Fix orchestration FIRST, then dashboard.

---

## OSC Environment Constraints (Researched)

| Resource | Status | Impact |
|----------|--------|--------|
| **Docker** | Not available (no daemon) | ClearML Server is **out** |
| **Apptainer/Podman** | Available (rootless) | Can run single containers, but multi-container stacks (ES+Mongo+Redis) are impractical |
| **Managed databases** | None (no PG, MySQL, MongoDB) | Must use SQLite or run DB in a container/job |
| **Outbound HTTPS** | Yes (ports 80, 443, 22 from compute + login) | W&B SaaS, `pip install`, container pulls all work |
| **Persistent services** | Login nodes: 20-min CPU time limit | Light servers (MLflow UI, Datasette) work if mostly idle; heavy servers get killed |
| **Storage** | Home=NFS 500GB, Scratch=GPFS 100TB (60-day purge), Project=GPFS | NFS causes ghost files + timestamp issues |
| **GPUs** | V100 (Pitzer), A100/H100 (Cardinal/Ascend) | Current project uses V100 on Pitzer |

**Key constraint**: No Docker daemon means no ClearML Server, no self-hosted W&B Server, and no managed databases. Any server-side tool must be light enough to run on a login node (< 20 min cumulative CPU) or inside a SLURM job.

---

## 13 Tools Evaluated

### Eliminated (with reasons)

| Tool | Why it's out |
|------|-------------|
| **ClearML** | Server needs Docker (ES + MongoDB + Redis). No Docker at OSC. SaaS tier exists but loses the SLURM orchestration advantage. |
| **Kubeflow** | Needs Kubernetes. OSC runs SLURM. |
| **ZenML** | No SLURM orchestrator. Cloud/K8s-oriented. |
| **Kedro** | No SLURM support. Local execution only. |
| **Prefect** | SLURM support is experimental (via Dask). Needs persistent server. |
| **Neptune.ai** | Self-hosted needs K8s. SaaS-only is viable but W&B is better for that. |
| **Comet ML** | Pivoting to LLM observability (Opik). Classic ML tracking deprioritized. |
| **Guild AI** | 900 stars, uncertain future. No SLURM support. |
| **DVC pipelines** | Git locking under concurrent SLURM jobs is worse than current SQLite issues. |
| **Nextflow** | Bioinformatics-native like Snakemake. Marginal gain for ML, high switching cost. |
| **Sacred+SEML** | Needs MongoDB. OSC has no managed DB. Running MongoDB in userspace is fragile. |

### Viable Candidates

| Tool | Category | Why it works at OSC |
|------|----------|-------------------|
| **Metaflow + @slurm** | Orchestration | Python-native DAG, `@slurm` decorator submits jobs automatically, artifacts versioned and immutable. `pip install` only. |
| **W&B** | Tracking + Dashboard | SaaS (no server needed), best-in-class UI, offline mode on compute + sync from login, outbound HTTPS works. Free academic tier. |
| **MLflow** | Tracking + Dashboard (fallback) | Self-hosted, `mlflow ui` is light enough for login node, SQLite backend. But same NFS+SQLite concurrency issues you already have. |
| **Aim** | Tracking + Dashboard (fallback) | Self-hosted, best OSS UI, `pip install` only. But needs persistent server process. |

---

## Recommended Approach: Metaflow + W&B

This is a two-tool stack that replaces ALL three current systems:

### 1. Metaflow (replaces Snakemake) -- Priority 1

**What it does for you:**
- Define the VGAE -> GAT -> DQN pipeline as a Python class with `@step` decorators
- `@slurm` decorator on each step submits it as a SLURM job automatically
- Every run gets a unique ID. All past runs are preserved (immutable, content-addressed)
- Artifacts (model checkpoints, embeddings, configs) pass between steps automatically
- No sentinel files, no content-based triggers, no NFS timestamp games
- No predetermined paths -- Metaflow manages its own artifact store
- Branching/joining/foreach parallelism built-in

**What your pipeline would look like:**
```python
from metaflow import FlowSpec, step, slurm

class KDGATFlow(FlowSpec):
    @step
    def start(self):
        self.datasets = ["hcrl_sa", "hcrl_ch", "set_01", ...]
        self.next(self.train_vgae, foreach="datasets")

    @slurm(gpu=1, mem=16000, time_minutes=120)
    @step
    def train_vgae(self):
        # Your existing VGAE training code
        cfg = resolve("vgae", "large", dataset=self.input)
        model, metrics = run_autoencoder(cfg)
        self.vgae_checkpoint = model_path
        self.next(self.train_gat)

    @slurm(gpu=1, mem=16000, time_minutes=120)
    @step
    def train_gat(self):
        # Large GAT, then small GAT with KD
        cfg = resolve("gat", "large", dataset=...)
        ...
        self.next(self.train_dqn)

    @slurm(gpu=1, mem=8000, time_minutes=60)
    @step
    def train_dqn(self):
        # DQN fusion
        ...
        self.next(self.evaluate)

    @slurm(gpu=1, mem=8000, time_minutes=30)
    @step
    def evaluate(self):
        # Evaluation + artifact capture
        ...
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass
```

**Metadata service**: Metaflow needs a metadata service for run tracking. Options:
- **Local mode** (filesystem-based): Works immediately, no server needed. Stores metadata in `~/.metaflowconfig/`. Good enough for single-user.
- **Service mode**: Run `metadata service` in tmux on login node. Light process, mostly idle (should stay under 20-min CPU limit like Datasette does).

**Migration path:**
1. `pip install metaflow metaflow-slurm` in conda env
2. Write a single `kdgat_flow.py` that wraps existing stage code
3. Test with one dataset: `python kdgat_flow.py run --dataset hcrl_sa`
4. Verify SLURM jobs submitted correctly, artifacts stored
5. Expand to all datasets
6. Remove Snakemake files once Metaflow is proven

**What stays the same:** `config/` (resolver, schema, paths), `src/` (models, training), stage implementations in `pipeline/stages/`. Metaflow wraps your existing code, it doesn't replace it.

**Risk assessment:**
- `@slurm` decorator is from Outerbounds (not Netflix core), added ~2025. Newer but actively maintained (v2.19.19, Feb 2026).
- If `@slurm` has issues, fallback is keeping Snakemake. The stage code itself doesn't change.
- Metaflow's local metadata mode requires no server at all -- zero infrastructure.

### 2. W&B (replaces SQLite DB + Custom Dashboard) -- Priority 2

**What it does for you:**
- `WandbLogger` built into PyTorch Lightning -- zero-code metric logging
- Every run preserved with full history, never overwritten
- Best-in-class dashboard: real-time metrics, parallel coordinates, scatter plots, custom panels
- Artifact versioning for model checkpoints, embeddings, attention weights
- Handles concurrent writes from multiple SLURM jobs (SaaS backend, no SQLite)
- Free academic tier (100GB storage, unlimited public projects)

**How it works on OSC:**
- Compute nodes: `WANDB_MODE=offline` (logs to local `.wandb/` directory)
- Login node: `wandb sync <run-dir>` pushes to W&B cloud (outbound HTTPS works)
- Or use [wandb-offline-sync-hook](https://github.com/klieret/wandb-offline-sync-hook) for automatic sync from shared filesystem
- Dashboard accessible from anywhere via browser (no SSH tunnel needed)

**Integration points:**
```python
# In pipeline/cli.py or training code
import wandb
wandb.init(project="kd-gat", config=cfg.model_dump())

# In Lightning training (already supported)
from lightning.pytorch.loggers import WandbLogger
logger = WandbLogger(project="kd-gat", offline=True)
trainer = Trainer(logger=logger)

# Log artifacts
wandb.log_artifact(embeddings_path, type="embeddings")
wandb.log_artifact(checkpoint_path, type="model")
```

**Migration path:**
1. Create W&B account (free academic tier)
2. `pip install wandb && wandb login` on login node
3. Add `WandbLogger` to ONE training stage
4. Run via SLURM, sync from login, verify dashboard
5. Instrument remaining stages
6. Remove `pipeline/db.py`, `pipeline/export.py`, `docs/dashboard/`

**What about the existing dashboard on GitHub Pages?**
- W&B provides shareable dashboard links and report pages
- If you need a public-facing dashboard, W&B public projects are viewable by anyone
- The custom D3.js dashboard can be retired entirely

---

## Alternative: MLflow Instead of W&B

If you'd rather not send data to W&B cloud:

- **MLflow** with SQLite backend: `mlflow server --backend-store-uri sqlite:///mlflow.db`
- Run on login node in tmux, access via SSH tunnel
- Same NFS+SQLite concurrency caveats you already have, but MLflow's API is cleaner than your custom write-through
- UI is functional but not as good as W&B
- Fully self-hosted, no data leaves cluster

This is a fallback, not the recommendation. W&B's SaaS model is actually an advantage here because it sidesteps every NFS/SQLite/persistence problem on OSC.

---

## What Gets Eliminated

| Current Component | Replaced By | Lines of Code Removed |
|---|---|---|
| `pipeline/Snakefile` + `pipeline/rules/*.smk` | Metaflow `kdgat_flow.py` | ~400 lines of Snakemake rules |
| `pipeline/db.py` (write-through, backfill, migrations) | W&B tracking API | ~700 lines |
| `pipeline/export.py` (15 export functions) | W&B dashboard | ~800 lines |
| `pipeline/analytics.py` | W&B comparison UI | ~300 lines |
| `docs/dashboard/` (D3.js, panels, charts) | W&B dashboard | ~2000 lines JS |
| `scripts/export_dashboard.sh` | Not needed | ~50 lines |
| Sentinel files (`.done`) | Metaflow artifact store | Concept eliminated |
| `profiles/slurm/` | Metaflow `@slurm` config | ~20 lines |

**What stays:**
- `config/` -- resolver, schema, paths (100% kept)
- `src/` -- models, training, preprocessing (100% kept)
- `pipeline/stages/` -- stage implementations (kept, wrapped by Metaflow steps)
- `pipeline/cli.py` -- simplified (remove DB calls, keep config resolution + stage dispatch)
- `pipeline/memory.py`, `pipeline/tracking.py` -- GPU utilities (kept)

---

## Verification Plan

### Phase 1: Metaflow proof-of-concept
1. Install: `pip install metaflow metaflow-slurm`
2. Write minimal flow with 2 steps (start -> train_vgae -> end)
3. Run: `python flow.py run` (local first, then with `@slurm`)
4. Verify: SLURM job submitted, artifact stored, run ID preserved
5. Check: Can access artifacts from previous runs? (`Flow('KDGATFlow').latest_run`)

### Phase 2: W&B proof-of-concept
1. Install: `pip install wandb && wandb login`
2. Add `WandbLogger` to VGAE autoencoder training
3. Run one training job via SLURM with `WANDB_MODE=offline`
4. Sync: `wandb sync` from login node
5. Verify: metrics, config, and artifacts visible in W&B dashboard

### Phase 3: Full migration
1. Expand Metaflow flow to all stages (VGAE -> GAT -> DQN -> eval)
2. Add W&B logging to all stages
3. Run full pipeline for one dataset
4. Compare results with existing experiment outputs
5. Remove Snakemake, db.py, export.py, custom dashboard

---

## Sources

Key references from research:
- [Metaflow @slurm (Outerbounds)](https://outerbounds.com/blog/delightful-large-scale-compute-across-environments-with-slurm-and-kubernetes)
- [metaflow-slurm extension](https://github.com/outerbounds/metaflow-slurm)
- [Metaflow GitHub (9.8K stars)](https://github.com/Netflix/metaflow)
- [W&B + SLURM guide](https://wandb.ai/events/SLURM/reports/Improving-Your-Deep-Learning-Workflows-with-SLURM-and-Weights-Biases--VmlldzozOTU4NzU0)
- [wandb-offline-sync-hook](https://github.com/klieret/wandb-offline-sync-hook)
- [W&B on SLURM (Harvard Kempner)](https://handbook.eng.kempnerinstitute.harvard.edu/s5_ai_scaling_and_engineering/experiment_management/wandb_sweeps.html)
- [OSC Container Support](https://www.osc.edu/resources/getting_started/howto/howto_use_docker_and_singularity_containers_at_osc)
- [OSC Login Node Limits](https://www.osc.edu/supercomputing/login-environment-at-osc)
- [MLflow GitHub (23K stars)](https://github.com/mlflow/mlflow) -- fallback option
- [ClearML SLURM (eliminated)](https://clear.ml/blog/how-clearml-helps-teams-get-more-out-of-slurm) -- needs Docker
