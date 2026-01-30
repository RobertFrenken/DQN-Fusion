# ML Pipeline Infrastructure Plan

**Status**: Planning
**Last Updated**: 2026-01-30
**Scope**: KD-GAT project and future projects using this workflow as a template

---

## Table of Contents

- [Current State](#current-state)
- [Tool Analysis](#tool-analysis)
  - [Snakemake (Workflow Orchestration)](#snakemake-workflow-orchestration)
  - [DVC (Data Version Control)](#dvc-data-version-control)
  - [MLflow (Experiment Tracking + Model Registry)](#mlflow-experiment-tracking--model-registry)
  - [Weights & Biases](#weights--biases-wb)
  - [ClearML](#clearml)
  - [Neptune.ai](#neptuneai)
  - [Supporting Tools: DVCLive, Aim](#supporting-tools-dvclive-aim)
- [Cross-Tool Capability Matrix](#cross-tool-capability-matrix)
- [DVC vs Snakemake Deep Comparison](#dvc-vs-snakemake-deep-comparison)
- [Recommended Architecture](#recommended-architecture)
- [Template Structure for Future Projects](#template-structure-for-future-projects)
- [Action Plan](#action-plan)

---

## Current State

### What Exists

| Component | Status | Details |
|-----------|--------|---------|
| **Pipeline code** | Production-ready | 4-stage pipeline (VGAE, GAT, DQN, Eval) in `pipeline/` and `Snakefile` |
| **Snakemake** | Configured | SLURM profile, V100 GPUs, 20 parallel jobs, conda per-rule |
| **DVC** | Freshly initialized | `.dvc/config` is empty, no remote, no tracked files |
| **Frozen configs** | ~50+ JSON files | Saved alongside every checkpoint, but not versioned by DVC |
| **MLflow** | Referenced in configs | Present in `requirements.txt`, `Justfile`, and frozen configs, but not actively integrated into training loops |
| **Experiment tracking** | Manual | CSV logs from Lightning, no centralized tracking UI |

### Known Pain Points

The error log at `results/hcrl_sa/teacher/slurm_eval_43989348.err` illustrates a
concrete infrastructure failure: a teacher-sized checkpoint (e.g., `id_embedding:
[2049, 64]`) was loaded into a student-sized model (`[2032, 32]`). This is a
**config-artifact coupling failure** -- the evaluation code did not know which
architecture hyperparameters produced that checkpoint. Every tool in this plan
addresses some aspect of preventing this class of error.

---

## Tool Analysis

### Snakemake (Workflow Orchestration)

**Role**: Execution engine -- decides what to run, where, with what resources, and in what order.

**Currently used features**:
- SLURM job submission via `sbatch` with per-rule resource specs (V100, 64GB, 16 CPUs, 360min)
- Wildcard-based pipeline expansion across 6 datasets x 2 model sizes
- DAG dependency resolution (students wait for teachers)
- `--jobs 20` parallel SLURM submission
- Custom `slurm-status.py` for job state monitoring
- `conda:` directives for per-rule environment isolation
- Failure recovery: `--rerun-incomplete`, `restart-times: 1`, `--keep-going`
- NFS latency handling: `latency-wait: 60`

**Strengths for this project**:
- Native SLURM integration with no external wrappers
- Per-rule resource specification (GPU type, memory, CPUs, walltime, partition)
- Parallel job submission across independent datasets
- Conda/Singularity environment per rule
- Execution profiles (`--profile slurm` vs local)

**Limitations**:
- No data versioning
- No experiment tracking or metrics comparison
- No model registry
- Timestamp-based change detection (less reliable than content hashes)

---

### DVC (Data Version Control)

**Role**: Version datasets, model checkpoints, and pipeline artifacts with content-addressable hashing.

**Current state**: Initialized (`.dvc/` exists) but not configured (no remote, no tracked files).

**Core capabilities**:
- `dvc add` -- track large files (datasets, model weights) via content hashes stored in small `.dvc` metafiles committed to git
- `dvc remote` -- push/pull artifacts to shared storage (S3, GCS, SSH, local filesystem)
- `dvc.yaml` / `dvc repro` -- define and reproduce pipelines with deps/outs/params tracking
- `dvc exp` -- run, compare, and manage experiment variations
- `dvc metrics` / `dvc plots` -- track and compare evaluation metrics
- `params.yaml` -- automatic detection of hyperparameter changes for cache invalidation
- Content-hash-based caching -- only re-runs stages when file contents actually change (more reliable than timestamps)

**Strengths for this project**:
- Version the 6 CAN datasets in `data/automotive/`
- Version model checkpoints in `experimentruns/`
- Lock exact data-code-config-output relationships via `dvc.lock`
- Share artifacts across OSC cluster via shared filesystem remote
- `dvc exp show` for tabular experiment comparison without a server

**Limitations (important)**:
- **No SLURM integration** -- cannot submit or manage cluster jobs
- **No parallel stage execution** -- global `rw.lock` enforces sequential `dvc repro`
- **No per-stage resource specs** -- no concept of GPUs, memory, or walltime
- **No conda/container management** -- user must handle environments externally
- **No failure recovery** -- no retries, no `--keep-going` equivalent
- **`foreach` is per-stage only** -- cannot template an entire multi-stage pipeline across datasets
- **No execution profiles** -- same `dvc repro` behavior everywhere

**Key DVC features for pipelines**:

```yaml
# dvc.yaml example
stages:
  train_vgae:
    foreach:
      - hcrl_ch
      - hcrl_sa
      - set_01
      - set_02
      - set_03
      - set_04
    do:
      cmd: python -m pipeline.cli autoencoder --preset vgae,teacher --dataset ${item}
      deps:
        - data/automotive/${item}
        - pipeline/stages.py
        - src/models/vgae.py
      params:
        - params.yaml:
      outs:
        - experimentruns/automotive/${item}/.../best_model.pt
```

---

### MLflow (Experiment Tracking + Model Registry)

**Role**: Centralized experiment logging, model versioning, and lifecycle management.

**Current state**: Referenced in `requirements.txt` and `Justfile` (`just mlflow`), but not integrated into training loops.

**Core components**:

1. **MLflow Tracking** -- log parameters, metrics, and artifacts per training run. Provides a web UI for searching and comparing runs.
   - Runs are grouped into Experiments
   - Each run logs: params (hyperparameters), metrics (loss, accuracy over time), artifacts (model files, plots), tags, and source code version
   - Tracking backend: local filesystem (default) or a centralized tracking server (SQLite/PostgreSQL + artifact store)

2. **MLflow Model Registry** -- central catalog of trained models with versioning, stage transitions, and aliases.
   - Register models from runs: `mlflow.register_model()`
   - Version lifecycle: `None` -> `Staging` -> `Production` -> `Archived`
   - Aliases: name a version `champion` or `challenger`
   - Links every model version back to the run (and its full config) that produced it

3. **MLflow Projects** -- convention for packaging code as reproducible units (MLproject file + conda.yaml). Can target local, Docker, or Kubernetes.

4. **MLflow Models** -- standardized model serialization for deployment (REST API, batch inference, cloud platforms).

**PyTorch Lightning integration**:
```python
from lightning.pytorch.loggers import MLFlowLogger

logger = MLFlowLogger(
    experiment_name="vgae_teacher",
    tracking_uri="http://mlflow-server:5000",
    log_model=True,  # auto-log model checkpoints as artifacts
)
trainer = pl.Trainer(logger=logger)
# All self.log() calls in LightningModule are captured automatically
```

**SLURM/HPC considerations**:
- **No native SLURM integration.** MLflow does not submit or schedule jobs.
- A centralized tracking server on a login node or lab VM can receive logs from any SLURM job over HTTP.
- Artifact store can be a shared filesystem path (ideal for OSC).
- No network dependency if using local file-based tracking (each job writes to shared FS).

**How it prevents the checkpoint-config mismatch**:
- Every model artifact is linked to the run that produced it
- The run contains all logged hyperparameters (architecture dims, hidden sizes, etc.)
- Model Registry version points to exactly one run
- Evaluation code loads from Registry, which guarantees matching config

**Limitations**:
- No pipeline DAG or workflow orchestration
- No data versioning (only stores artifacts)
- Tracking server requires setup and maintenance
- No resource specification or scheduler integration
- Hyperparameter sweeps are manual (script loops), no built-in Bayesian optimization

---

### Weights & Biases (W&B)

**Role**: Managed experiment tracking platform with sweeps, artifacts, reports, and (partial) job orchestration.

**Core components**:

1. **Experiment Tracking** -- auto-captures hyperparams, metrics (live streaming), system metrics (GPU utilization, memory), code diffs. Rich web dashboard with interactive charts.

2. **Sweeps** -- built-in hyperparameter optimization (grid, random, Bayesian). Sweep agents poll the W&B server for the next configuration. Can run multiple agents as separate SLURM jobs for parallel sweeps.

3. **Artifacts** -- versioned datasets and models with lineage tracking. Knows which run produced which artifact and which artifact was consumed by which run.

4. **Reports** -- collaborative documents embedding live charts, markdown, LaTeX. Useful for publication workflows.

5. **Model Registry** -- link artifacts to named model entries with version control.

6. **Launch** -- job queueing that can target SLURM, Kubernetes, or cloud compute. A launch agent on the login node submits jobs. Less capable than Snakemake (no DAG dependencies, no per-job resource inference from pipeline structure).

**PyTorch Lightning integration**:
```python
from lightning.pytorch.loggers import WandbLogger

logger = WandbLogger(project="KD-GAT", name="vgae_teacher_hcrl_sa")
trainer = pl.Trainer(logger=logger)
```

**SLURM/HPC considerations**:
- **Offline mode** (`WANDB_MODE=offline`) works fully disconnected, critical for compute nodes without internet. Sync later with `wandb sync`.
- **W&B Launch** can submit jobs to SLURM queues via an agent on the login node, but this is less mature than Snakemake's SLURM integration.
- Sweeps can be parallelized by running multiple `wandb agent` processes as separate SLURM jobs.

**Strengths over MLflow**:
- Significantly better UI for experiment comparison
- Built-in Bayesian hyperparameter sweeps
- Live metric streaming during training
- Reports feature for publication-ready figures
- System resource monitoring (GPU utilization) auto-captured

**Limitations**:
- **SaaS dependency** -- vendor lock-in risk. Self-hosted W&B Server is enterprise-priced.
- Free tier: 100GB storage, personal use only.
- Network dependency (mitigated by offline mode, but adds a manual sync step).
- No data versioning at DVC's depth (artifacts are immutable snapshots, not content-addressable).
- No pipeline DAG orchestration beyond Launch.

---

### ClearML

**Role**: Open-source (Apache 2.0) end-to-end ML platform. The most feature-rich tool analyzed.

**Core components**:

1. **Experiment Manager** -- near-zero-code auto-logging. `Task.init()` captures everything (params, metrics, artifacts, git diff, installed packages). PyTorch Lightning is auto-detected.

2. **ClearML Agent** -- remote execution agent that pulls tasks from queues and runs them. This is ClearML's key differentiator.
   - **Glue mode**: runs on a login node, monitors a ClearML queue, and submits tasks as SLURM jobs via `sbatch`. Generates SLURM scripts, submits them, monitors their state.
   - Can specify resource requirements per task (GPUs, memory) translated into SLURM directives.
   - Supports dynamic autoscaling: spin up/down SLURM jobs based on queue depth.

3. **ClearML Pipelines** -- DAG-based pipeline orchestration where each step is a ClearML Task. Pipelines can be triggered manually, on schedule, or by data/model triggers.

4. **ClearML Data** -- dataset versioning with delta storage (only stores changed files).

5. **ClearML Orchestrator** -- autoscaler for cloud and HPC resources.

**PyTorch Lightning integration**:
```python
from clearml import Task
task = Task.init(project_name="KD-GAT", task_name="vgae_teacher")
# Lightning auto-logs from here -- no logger swap needed
```

**SLURM/HPC**: The strongest native SLURM integration among all tracking tools (Agent Glue mode). However, this overlaps with and is less mature than Snakemake's SLURM integration for DAG-based pipelines.

**Limitations**:
- Server setup required (docker-compose for self-hosted, or SaaS)
- More complex to operate than MLflow or W&B (Agent + Queue + Server architecture)
- Pipeline DSL is Python-only (no declarative YAML)
- Less widely adopted in academia than MLflow or W&B
- Redundant with Snakemake for SLURM orchestration in this project

---

### Neptune.ai

**Role**: Managed metadata store focused purely on experiment tracking and comparison.

**Core components**:
- **Runs** -- structured metadata store with nested namespaces, file series, image series
- **Model Registry** -- version models with lifecycle stages
- **Dashboards** -- customizable comparison views

**PyTorch Lightning integration**: Native `NeptuneLogger`.

**SLURM/HPC**: No integration. Pure tracking tool. Requires network access (no robust offline mode).

**Limitations**:
- SaaS-only for most tiers (no self-hosted option)
- No orchestration, no data versioning, no pipeline management
- Narrowest scope of all tools analyzed
- Free tier limited to 50GB

**Assessment**: Does not offer unique capabilities over MLflow or W&B for this project. Not recommended.

---

### Supporting Tools: DVCLive, Aim

**DVCLive** -- lightweight library from Iterative (DVC's maintainers) that bridges DVC experiment tracking with live metric logging during training.

```python
from dvclive.lightning import DVCLiveLogger
logger = DVCLiveLogger()
trainer = pl.Trainer(logger=logger)
```

Writes metrics to `dvclive/` in DVC-tracked format, enabling `dvc exp show` without any tracking server. No server, no SaaS, no network -- everything is local files tracked by git+DVC. Useful as a lightweight alternative if MLflow/W&B are not adopted.

**Aim** -- open-source (Apache 2.0) experiment tracking with a high-performance local UI. Stores data in a local `.aim` repository. Key differentiator is its query language for filtering thousands of runs. Lightning integration via `AimLogger`. No orchestration, no data versioning. Useful as a local-first alternative to MLflow.

---

## Cross-Tool Capability Matrix

| Capability | Snakemake | DVC | MLflow | W&B | ClearML | Neptune |
|---|---|---|---|---|---|---|
| **SLURM Job Submission** | Native | None | None | Launch (partial) | Agent Glue | None |
| **Per-Job Resources** (GPU/mem/time) | Full | None | None | Launch (limited) | Agent (moderate) | None |
| **Parallel Execution** | `--jobs N` | None (global lock) | None | Sweep agents | Agent workers | None |
| **Pipeline DAG** | Native wildcards | `deps`/`outs` | None | None | Pipeline DSL | None |
| **Data Versioning** | None | Core feature | None | Artifacts (limited) | ClearML Data | None |
| **Experiment Tracking** | None | `dvc exp` (basic) | Full (runs, UI) | Full (live, rich UI) | Full (auto-capture) | Full |
| **Hyperparam Comparison** | None | `dvc exp show` | UI + API | Rich interactive UI | Full UI | Best-in-class UI |
| **Model Registry** | None | None | Full (stages/aliases) | Full (lineage) | Full (auto-publish) | Full |
| **Metrics/Plots** | None | `dvc metrics/plots` | UI charts | Live dashboards | Auto-plots | Dashboards |
| **Lightning Logger** | N/A | DVCLiveLogger | MLFlowLogger | WandbLogger | Auto-detected | NeptuneLogger |
| **Offline HPC** | Native | Native | Local tracking OK | `WANDB_MODE=offline` | Self-hosted server | Needs network |
| **Reproducibility** | DAG + conda | Content hashes + lock | Projects (limited) | None | Task replay | None |
| **Failure Recovery** | Retries, keep-going | None | None | None | Agent re-queuing | None |
| **Conda per Stage** | `conda:` directive | None | Docker (Projects) | Launch (Docker) | Agent (Docker, venv) | None |
| **Cost** | Free (OSS) | Free (OSS) | Free (OSS) | Free tier / paid | Free (OSS) / paid | Free tier / paid |
| **Setup Complexity** | Moderate | Low | Moderate | Low (SaaS) | High | Low (SaaS) |

---

## DVC vs Snakemake Deep Comparison

These two tools appear to overlap (both define pipeline DAGs) but solve fundamentally different problems.

### Can DVC replace Snakemake?

**No.** DVC cannot:
- Submit SLURM jobs or communicate with any HPC scheduler
- Allocate different resources (GPU, memory, time) to different pipeline stages
- Run independent stages in parallel (global `rw.lock` prevents it)
- Retry failed stages or continue past failures
- Manage conda environments per stage
- Switch execution profiles between local and cluster

### Can Snakemake replace DVC?

**No.** Snakemake cannot:
- Version datasets or model checkpoints with content-addressable hashing
- Push/pull artifacts to/from remote storage
- Track which hyperparameters produced which outputs
- Compare experiments across runs
- Provide reproducibility guarantees beyond "same DAG + same timestamps"

### Where they overlap

Both define a DAG of stages with inputs and outputs, and both skip stages whose outputs are already up to date. The difference is in _how_ they detect staleness:

| Aspect | Snakemake | DVC |
|--------|-----------|-----|
| Change detection | Timestamp-based (file mtime) | Content-hash-based (MD5) |
| False positives | `touch` triggers re-run | Only actual content changes trigger re-run |
| False negatives | Possible if mtime is wrong | None (content is authoritative) |

### Recommendation

Use both. Snakemake for execution, DVC for versioning. They do not conflict.

---

## Recommended Architecture

### Layer Diagram

```
+------------------------------------------------------------------+
|  Layer 4: EXPERIMENT TRACKING + MODEL REGISTRY                   |
|                                                                  |
|  MLflow Tracking Server (self-hosted on login/lab node)          |
|    - MLFlowLogger in Lightning logs all runs                     |
|    - Model Registry for promoting best teacher/student models    |
|    - Artifact store on shared OSC filesystem                     |
|                                                                  |
+-------------------------------+----------------------------------+
                                | Logs from training jobs
+-------------------------------v----------------------------------+
|  Layer 3: WORKFLOW ORCHESTRATION                                 |
|                                                                  |
|  Snakemake (unchanged from current setup)                        |
|    - --profile slurm: V100 GPUs, 64GB, 16 CPUs                  |
|    - --jobs 20: parallel across datasets                         |
|    - Wildcards: {dataset}, {model_size}, {distillation}          |
|    - conda: per-rule environments                                |
|    - Retry/recovery on failure                                   |
|                                                                  |
+-------------------------------+----------------------------------+
                                | Reads/writes DVC-tracked files
+-------------------------------v----------------------------------+
|  Layer 2: DATA + ARTIFACT VERSIONING                             |
|                                                                  |
|  DVC                                                             |
|    - dvc add data/automotive/*       (version 6 datasets)        |
|    - dvc add experimentruns/.../*.pt (version checkpoints)       |
|    - dvc remote (shared OSC filesystem)                          |
|    - params.yaml (tracked hyperparameters)                       |
|    - dvc push / dvc pull (artifact distribution)                 |
|                                                                  |
+-------------------------------+----------------------------------+
                                |
+-------------------------------v----------------------------------+
|  Layer 1: SOURCE CONTROL                                         |
|                                                                  |
|  Git                                                             |
|    - Code, configs, .dvc metafiles, dvc.lock, Snakefile          |
|    - Frozen configs committed alongside code changes             |
|                                                                  |
+------------------------------------------------------------------+
```

### Integration Flow

```
1. Snakemake rule fires
       |
       v
2. SLURM job starts on compute node
       |
       v
3. Training script runs with Lightning Trainer(logger=MLFlowLogger)
       |
       +---> MLflow logs params, metrics, model artifact per run
       |
       v
4. On success, Snakemake output file written
       |
       +---> DVC tracks output checkpoint (content hash in dvc.lock)
       |
       v
5. After all stages complete:
       +---> Best model promoted in MLflow Model Registry
       +---> dvc push sends artifacts to shared remote
       +---> git commit captures dvc.lock + code state
```

### Why This Stack

| Decision | Rationale |
|----------|-----------|
| **Keep Snakemake** | No other tool handles SLURM resource allocation, parallel job submission, per-rule conda, and failure recovery. ClearML Agent Glue is the closest alternative but adds operational complexity and is less mature for DAG-based HPC pipelines. |
| **Keep DVC** | Content-addressed artifact caching and remote storage are not replicated by any tracking tool. `dvc.lock` provides the strongest reproducibility guarantees. |
| **Add MLflow** | Already referenced in the codebase (`requirements.txt`, `Justfile`). Fully OSS and self-hosted (no SaaS dependency, no cost). Model Registry directly solves the checkpoint-config mismatch issue. Any lab can replicate the setup. |
| **Skip ClearML** | Its SLURM Agent Glue is redundant with Snakemake. Adding ClearML's server + agent architecture would increase operational complexity for marginal gain. |
| **Skip Neptune** | Pure tracking tool with no unique capability over MLflow for this use case. |
| **Skip W&B (for now)** | Better UI and Sweeps, but introduces SaaS dependency, cost risk, and network requirements on an HPC cluster. Can be added later via `WandbLogger` alongside `MLFlowLogger` if needed. |

### MLflow vs W&B Decision Summary

| Factor | MLflow | W&B |
|--------|--------|-----|
| OSC network restrictions | Self-hosted, no external network | Needs offline mode + sync |
| Cost | Free (OSS) | Free tier, then paid |
| Setup | `mlflow server` on lab node | SaaS account or enterprise self-hosted |
| UI quality | Functional | Significantly better |
| Hyperparameter sweeps | Manual loops | Built-in Bayesian optimization |
| Publication reports | Export via API | W&B Reports (built-in) |
| Model Registry | Full, OSS | Partly behind paid tier |
| Template portability | Any lab can self-host | Requires W&B account |
| Already in codebase | Yes | No |

**Verdict**: MLflow is the better fit for an academic HPC project that needs to be
self-contained, zero-cost, and reproducible by other researchers without accounts.

---

## Template Structure for Future Projects

This infrastructure can be cloned as a starting template for any HPC-based ML
research pipeline:

```
project-template/
|-- Snakefile                      # Pipeline DAG with SLURM profile
|-- profiles/
|   `-- slurm/
|       |-- config.yaml            # Cluster-specific resources
|       `-- slurm-status.py        # Job state checker
|-- dvc.yaml                       # Data + artifact dependency DAG
|-- params.yaml                    # All tunable hyperparameters (DVC-tracked)
|-- .dvc/
|   `-- config                     # DVC remote (shared FS or S3)
|-- MLproject                      # MLflow project definition
|-- src/
|   |-- training/                  # Lightning modules + MLFlowLogger
|   |-- models/                    # Architecture definitions
|   `-- evaluation/                # Eval scripts (load from Model Registry)
|-- config/
|   |-- envs/                      # Conda environments per stage
|   `-- frozen_configs/            # Auto-generated, DVC-tracked
|-- data/                          # DVC-tracked datasets (.dvc metafiles in git)
`-- experimentruns/                # DVC-tracked model outputs
```

### What each tool owns

| Tool | Owns | Does NOT own |
|------|------|-------------|
| **Git** | Code, configs, `.dvc` metafiles, `dvc.lock`, `Snakefile`, `params.yaml` | Large files, binary artifacts |
| **DVC** | Dataset versions, checkpoint versions, artifact remote, content hashes | Execution, scheduling, tracking UI |
| **Snakemake** | Job submission, resource allocation, parallelism, DAG execution, conda envs, retries | Versioning, experiment comparison |
| **MLflow** | Run logs (params/metrics/artifacts), Model Registry, tracking UI, model lifecycle | Execution, data versioning, scheduling |

---

## Action Plan

### Phase 1: DVC Foundation

- [ ] **Configure DVC remote** on OSC shared filesystem
  ```bash
  dvc remote add -d osc_store /fs/ess/PAS2022/dvc-cache
  # Or scratch: /fs/ess/scratch/PAS2022/dvc-cache
  ```
- [ ] **Track datasets** with DVC
  ```bash
  dvc add data/automotive/hcrl_ch
  dvc add data/automotive/hcrl_sa
  dvc add data/automotive/set_01
  dvc add data/automotive/set_02
  dvc add data/automotive/set_03
  dvc add data/automotive/set_04
  git add data/automotive/*.dvc data/automotive/.gitignore
  ```
- [ ] **Create `params.yaml`** extracting key hyperparameters from frozen configs (lr, epochs, hidden dims, kd_temperature, kd_alpha, safety_factor)
- [ ] **Add `.dvcignore` patterns** for SLURM logs, `__pycache__`, `.snakemake/`
- [ ] **Push initial data** with `dvc push`
- [ ] **Commit** `.dvc/config`, `.dvc` metafiles, `params.yaml` to git

### Phase 2: MLflow Tracking Server

- [ ] **Set up MLflow server** on OSC login node or lab VM
  ```bash
  mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /fs/ess/PAS2022/mlflow-artifacts \
    --host 0.0.0.0 --port 5000
  ```
- [ ] **Add `MLFlowLogger`** to the Lightning training path in `src/training/trainer.py`
  ```python
  from lightning.pytorch.loggers import MLFlowLogger
  mlflow_logger = MLFlowLogger(
      experiment_name=f"{model_type}_{dataset}",
      tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"),
      log_model=True,
  )
  ```
- [ ] **Log frozen config** as MLflow artifact per run
- [ ] **Log model architecture summary** (parameter counts, layer dims) as run params
- [ ] **Verify** tracking works from a SLURM compute node (set `MLFLOW_TRACKING_URI` in Snakemake rule)

### Phase 3: Model Registry

- [ ] **Register best models** after evaluation stage
  ```python
  import mlflow
  mlflow.register_model(
      model_uri=f"runs:/{run_id}/model",
      name=f"{model_type}_{model_size}_{dataset}"
  )
  ```
- [ ] **Add stage transitions** -- promote from `None` to `Staging` to `Production` based on evaluation metrics
- [ ] **Update evaluation code** to load models from Registry instead of raw file paths (prevents the checkpoint-config mismatch)
- [ ] **Add alias support** (`champion` / `challenger`) for A/B comparison of teacher vs student

### Phase 4: DVC Pipeline Definition (Optional)

- [ ] **Create `dvc.yaml`** mirroring the Snakemake DAG (for reproducibility tracking, not execution)
  - Snakemake remains the execution engine
  - DVC tracks which inputs/params/code produced which outputs
- [ ] **Add `foreach` stages** for the 6 datasets
- [ ] **Mark metrics** in `dvc.yaml` (`metrics:` field for evaluation JSON outputs)
- [ ] **Run `dvc repro --dry`** to verify DAG matches Snakemake

### Phase 5: Versioning Existing Artifacts

- [ ] **Track existing model checkpoints** with `dvc add`
  ```bash
  # Track the best model for each completed experiment
  find experimentruns -name "best_model.pt" -exec dvc add {} \;
  ```
- [ ] **Push all artifacts** to DVC remote
- [ ] **Tag git** with `v1.0-infrastructure` marking the infrastructure baseline

### Phase 6: Template Extraction

- [ ] **Document the setup** in this file (done -- update as you proceed)
- [ ] **Create a `cookiecutter` or manual template** extracting the 4-layer architecture (Git + DVC + Snakemake + MLflow)
- [ ] **Write a `CONTRIBUTING.md`** describing how new projects should use the template
- [ ] **Validate portability** by cloning into a new project directory and running smoke tests

### Phase 7: Future Enhancements (Backlog)

- [ ] Add `DVCLiveLogger` alongside `MLFlowLogger` for `dvc exp show` support without server access
- [ ] Evaluate W&B Sweeps for KD hyperparameter tuning (kd_temperature, kd_alpha) if Bayesian optimization is needed
- [ ] Add `dvc plots` definitions for training loss curves
- [ ] Set up CI/CD pipeline that runs `dvc repro --dry` + `snakemake -n` on pull requests to validate pipeline consistency
- [ ] Investigate ClearML Agent Glue if Snakemake maintenance becomes burdensome
