# Pipeline Orchestration & Experiment Tracking: Research Findings

> **Status**: Architecture 1 (minimal fix) implemented 2026-02-16. This document preserved for future reference.

## Context

The Snakemake pipeline has repeated brittleness: SQLite WAL crashes under concurrent SLURM writes on NFS, group jobs cause all-or-nothing failures, sentinel file management is complex, and failures present as "silent" (jobs appear unknown in DB). This document captures comprehensive research into whether Snakemake should be replaced and with what.

---

## Root Cause Analysis

The pipeline failures on Feb 16, 2026 traced to a single root cause:

**SQLite WAL on NFS cannot handle concurrent writers from multiple SLURM compute nodes.**

- Batch 1 (18 eval jobs): 4 hit `sqlite3.OperationalError: disk I/O error` within 12 seconds of `record_run_start()`. Snakemake cancelled remaining 11 (group job behavior). 1 finished.
- Batch 2 (18 retry): 6 succeeded (race winners), 12 crashed with same error.
- Batch 3 (1 job solo): Succeeded — no contention.

Secondary issue: `group: "evaluation"` in Snakefile causes Snakemake to cancel all group members when one fails, turning partial failures into total failures.

---

## Tools Evaluated (8 total)

### Rejected — Wrong fit for HPC

| Tool | Why it fails |
|------|-------------|
| **ClearML** | SLURM Glue (`clearml-agent-slurm`) is **Enterprise-only** (paywall). Server needs Docker + MongoDB + Elasticsearch + Redis (8-16GB RAM). Cannot run on OSC login node (1GB memory / 20-min CPU limit). |
| **Metaflow** | SLURM extension (`metaflow-slurm`) is tightly coupled to **Outerbounds commercial platform**. Designed for remote SLURM from cloud VMs, not on-cluster execution. |
| **Prefect** | Requires **persistent server** (Prefect Cloud or self-hosted). No native SLURM executor. Would introduce its own SQLite-on-NFS state management — same problem or worse. |
| **Hydra + Submitit** | Designed for **hyperparameter sweeps**, not multi-stage pipelines. No job chaining. Would conflict with existing Pydantic config system. |
| **Parsl** | **Pilot job model** — reserves a worker pool upfront. Cannot give different resources (GPU, memory, walltime) to different stages within one executor. |
| **HyperQueue** | Overkill for ~54 tasks. Requires server process. Better suited for thousands of short tasks. |

### Viable alternatives

| Tool | Strengths | Weaknesses |
|------|-----------|------------|
| **Nextflow** | Native SLURM (mature, production). Content-hash caching (better than Snakemake). No shared DB. Each process is independent SLURM job (no group failures). `-resume` for incremental reruns. Proven on OSC-class clusters (NIH Biowulf = SLURM + NFS). | Groovy DSL (not Python). ~300 lines to rewrite Snakefile. Learning curve. |
| **Submitit** (Meta/FAIR) | Pure Python SLURM submission. Returns futures. Minimal, transparent. | No DAG, no retry-on-failure (only timeout/preemption), no skip-if-done. You rebuild half of Snakemake. |
| **Thin sbatch wrapper** | Raw `subprocess.run(["sbatch", ...])`. Maximum control and transparency. | Same as submitit but even more DIY. `--dependency=afterok` chaining breaks when upstream is skipped (no job ID). Need no-op job trick or conditional dependency. |
| **Snakemake (fix in place)** | Already working. Content-based triggers, between-workflow caching, NFS hardening all proven. ~50 lines to fix root causes. | Bioinformatics-flavored DSL. Sentinel complexity. But the alternatives don't eliminate sentinels either. |

---

## The Container Angle

OSC supports **Apptainer** (formerly Singularity):
- `apptainer build --fakeroot myimage.sif mydef.def`
- `buildah build -f Dockerfile --format docker -t tag .`
- GPU via `--nv` flag
- Home, CWD, `/fs/ess`, `/tmp` auto-mounted
- Single-node jobs only (MPI planned, not available)
- Docker Hub images convert directly: `apptainer pull docker://image:tag`

### The Sidecar Service Pattern

**Documented and used in practice** (University of Nebraska HCC, Grand Valley State):

1. Submit a **service job** (CPU partition, cheap) running MLflow/PostgreSQL in Apptainer
2. Service writes hostname:port to a shared file before blocking
3. Training jobs use `--dependency=after:SERVICE_JOB_ID` + health-check retry loop
4. All DB writes go through the service (single writer) — **eliminates NFS concurrent write problem**

```bash
# Service job
#SBATCH --dependency=singleton     # only one instance
#SBATCH --signal=B:SIGINT@60      # graceful shutdown before walltime
apptainer run --bind $PGDATA:/var/lib/postgresql/data postgres.sif

# Training job
source $SCRATCH/connection.env
until pg_isready -h $HOST -p $PORT; do sleep 5; done
python -m pipeline.cli autoencoder ...
```

Key detail: **OSC compute nodes have outbound internet via NAT** (`nat.osc.edu`). This means cloud tracking services (W&B, ClearML hosted) work directly from GPU jobs without offline mode hacks.

---

## Experiment Tracking Options

### Option 1: Weights & Biases (W&B)

- **Free academic Pro tier** — unlimited runs, 200GB storage, all Pro features
- **Zero infrastructure** — cloud-hosted, no server to run
- **Works on OSC** — compute nodes have outbound NAT, `WANDB_API_KEY` + `wandb.init()` should just work
- **Lightning integration**: `WandbLogger` is first-class, ~5 lines
- **Offline fallback**: `WANDB_MODE=offline` + `wandb sync` from login node. `wandb-osh` package automates sync via filesystem triggers
- **Best UI** of all options for experiment comparison, sweeps, artifact tracking
- **Tradeoff**: Data lives on wandb.ai cloud. Vendor dependency. But free for academics.

### Option 2: MLflow (self-hosted)

- **Serverless mode**: `sqlite:////fs/scratch/PAS1266/mlruns.db` — no server needed
  - Same NFS concurrent write risk as current project.db
  - Fine for serial execution (one job at a time)
  - **Filesystem backend deprecated in MLflow 3.7 (Dec 2025)** — use SQLite, not file-based
- **Sidecar mode**: MLflow server in Apptainer as SLURM job
  - Single-writer serialization eliminates NFS issues
  - `mlflow server --backend-store-uri sqlite:///mlruns.db --host 0.0.0.0 --port 5000`
  - Training jobs set `MLFLOW_TRACKING_URI=http://$(hostname):5000`
  - More infrastructure to manage but fully self-hosted
- **Lightning integration**: `MLFlowLogger` is first-class, ~8 lines
- **UI**: Functional but significantly worse than W&B

### Option 3: Keep existing SQLite DB

- Already works for serial execution
- `pipeline.analytics` + `pipeline.export` + D3.js dashboard already built
- Root cause fix: remove concurrent writes from compute nodes (defer to `onsuccess`)
- No new infrastructure, no new dependencies

---

## Orchestration Options

### Option A: Fix Snakemake (~50 lines changed)

1. **Remove DB writes from compute nodes**: Delete `record_run_start()`/`record_run_end()` calls in `cli.py` when running under SLURM (detect via `$SLURM_JOB_ID`). Let `onsuccess -> db populate` handle all DB writes as a single-writer post-pipeline step.
2. **Remove `group: "evaluation"`**: Each eval becomes an independent SLURM job. One failure doesn't cancel the rest.
3. **Keep everything else**: Content-based triggers, between-workflow caching, sentinel files, NFS hardening — these work and took real effort to get right.

**Pros**: Lowest risk. Addresses root causes directly. No learning curve.
**Cons**: Still Snakemake. Still bioinformatics-flavored DSL. Sentinel complexity remains.

### Option B: Thin Python orchestrator (~200 lines)

Replace `Snakefile` + rules with a Python script that:
1. Reads config to determine what stages need to run
2. Checks `data/project.db` (or W&B/MLflow) for completed runs
3. Submits `sbatch` jobs with `--dependency=afterok:ID` chaining
4. Handles skip-if-done via DB query (not sentinel files)

```python
# Pseudocode
for ds in datasets:
    vgae_id = submit_if_needed("autoencoder", "vgae", "large", ds)
    gat_id = submit_if_needed("curriculum", "gat", "large", ds, dep=vgae_id)
    dqn_id = submit_if_needed("fusion", "dqn", "large", ds, dep=gat_id)
    eval_id = submit_if_needed("evaluation", "vgae", "large", ds, dep=dqn_id)
```

**Pros**: Pure Python. Fully transparent. No DSL. No sentinel files. Uses existing DB for state.
**Cons**: Lose content-based caching (must rebuild or accept sentinel-based skip). Lose NFS hardening (Snakemake's `rerun-triggers`). Lose automatic retry with escalating memory.

### Option C: Nextflow (~300 lines Groovy DSL)

Full replacement with the other HPC-native workflow engine:
- Native SLURM support (one `sbatch` per process, like Snakemake)
- Content-hash caching via `work/` directory (more robust than Snakemake)
- `-resume` for incremental reruns
- No shared database
- Each process is independent (no group job problem)
- `process.scratch = true` for local-SSD execution (avoids NFS during computation)

**Pros**: Solves all pain points. Proven on HPC. Better caching than Snakemake.
**Cons**: Groovy DSL learning curve. Migration effort. Less common in ML (more in genomics).

---

## Recommended Architectures (3 combinations)

### Architecture 1: Minimal fix (lowest effort, unblocks now)
- **Orchestration**: Fix Snakemake (Option A)
- **Tracking**: Keep existing SQLite + add W&B as supplementary tracker
- **Effort**: ~50 lines Snakemake fix + ~10 lines W&B integration

### Architecture 2: Clean break, cloud tracking (medium effort)
- **Orchestration**: Thin Python orchestrator (Option B)
- **Tracking**: W&B cloud (free academic tier)
- **Effort**: ~200 lines new orchestrator + ~10 lines W&B + deprecate Snakefile

### Architecture 3: Full migration, self-hosted (highest effort)
- **Orchestration**: Nextflow (Option C) or thin Python orchestrator
- **Tracking**: MLflow sidecar in Apptainer container on CPU partition
- **Effort**: ~300 lines Nextflow or ~200 lines Python + Apptainer setup

---

## OSC-Specific Facts

| Fact | Implication |
|------|-------------|
| Login node: 1GB memory, 20-min CPU limit | Cannot run MLflow/ClearML server on login node |
| Compute nodes: outbound internet via NAT | W&B cloud works directly, no offline mode needed |
| Apptainer available (`apptainer build --fakeroot`) | Can run Docker images as SLURM jobs |
| GPFS scratch (`/fs/scratch/PAS1266/`) | Better locking than NFS home, but still not safe for heavy concurrent SQLite |
| GPU partition: V100, account PAS3209 | Service jobs should use CPU partition to avoid wasting GPU hours |
| `--dependency=after:ID` | For sidecar pattern: "start after job starts" (not "after it completes") |
| Inter-node networking works (required for MPI/DDP) | Sidecar services reachable from any compute node |

---

## Key Technical Details

### Snakemake: what it actually provides that's hard to replace
1. Content-based rerun triggers (`rerun-triggers: [mtime, input, code, params]`)
2. Between-workflow caching (`cache: True` with `SNAKEMAKE_OUTPUT_CACHE`)
3. Per-attempt escalating memory (`mem_mb=lambda wc, attempt: 128000 * attempt`)
4. Diamond dependency graph (vgae_large feeds both gat_large AND vgae_small_kd)
5. `onsuccess`/`onerror` lifecycle hooks

### What `pipeline.cli` already handles independently
- Full config system (YAML composition + Pydantic validation + freezing)
- All path derivation
- DB write-through (`record_run_start`, `record_run_end`)
- Config validation before dispatch
- Archiving of completed runs
- Stage dispatch + all ML training logic
- Dashboard export, analytics, state sync

### The db.py retry fix (applied 2026-02-16)
```python
_RETRYABLE_ERRORS = ("locked", "disk i/o error", "database is locked")
# Extended _retry_on_locked to catch NFS I/O errors
# Increased busy_timeout from 15s to 30s
```

---

## Decision Matrix

| Criterion | Fix Snakemake | Thin Python | Nextflow |
|-----------|:---:|:---:|:---:|
| Effort to implement | Low | Medium | Medium-High |
| Solves SQLite concurrent writes | Yes (defer writes) | Yes (single orchestrator) | Yes (no shared DB) |
| Solves group job failures | Yes (remove `group:`) | N/A (no groups) | N/A (independent processes) |
| Content-based caching | Existing | Must rebuild | Native (better) |
| NFS timestamp protection | Existing | Must rebuild | Native |
| Retry with escalating memory | Existing | Must rebuild | Config-based |
| Python-native (no DSL) | No (Snakemake DSL) | Yes | No (Groovy DSL) |
| Debugging transparency | Medium | High | Medium |
| Long-term maintainability | Medium | High | High |

---

## References

- [OSC Containers HOWTO](https://www.osc.edu/resources/getting_started/howto/howto_use_docker_and_singularity_containers_at_osc)
- [OSC Firewall/NAT](https://www.osc.edu/documentation/knowledge_base/firewall_and_proxy_settings)
- [PostgreSQL on SLURM - HCC Nebraska](https://hcc.unl.edu/docs/applications/app_specific/running_postgres/)
- [PostgreSQL via Apptainer - GVSU](https://services.gvsu.edu/TDClient/60/Portal/KB/ArticleDet?ID=24265)
- [MLflow on HPC shared filesystem - GitHub #5908](https://github.com/mlflow/mlflow/discussions/5908)
- [MLflow filesystem deprecation - GitHub #18534](https://github.com/mlflow/mlflow/issues/18534)
- [W&B Academic Program](https://wandb.ai/site/research/)
- [wandb-osh (offline sync)](https://pypi.org/project/wandb-osh/)
- [ClearML SLURM Glue (Enterprise)](https://clear.ml/docs/latest/docs/clearml_agent/clearml_agent_deployment_slurm/)
- [Nextflow SLURM executor](https://www.nextflow.io/docs/latest/executor.html)
- [Nextflow on Biowulf (NIH SLURM+NFS)](https://hpc.nih.gov/apps/nextflow.html)
- [submitit - Meta/FAIR](https://github.com/facebookincubator/submitit)
- [Parsl SlurmProvider](https://parsl.readthedocs.io/en/stable/stubs/parsl.providers.SlurmProvider.html)
- [Metaflow SLURM extension](https://pypi.org/project/metaflow-slurm/)
- [Prefect + SLURM gap - GitHub #10136](https://github.com/PrefectHQ/prefect/issues/10136)
