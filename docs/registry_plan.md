# Experiment Registry: Design & Implementation Plan

**Date**: 2026-02-02 (updated)
**Status**: Implemented (Snakemake + MLflow)

## Problem Statement

The original experiment output system encoded every configuration dimension as a directory level:

```
experimentruns/{modality}/{dataset}/{size}/{learning_type}/{model}/{distill}/{stage}/
```

This 8-level hierarchy had two issues:
1. **Redundancy**: 4 of 8 levels are deterministic (modality=always "automotive", learning_type/model_arch derived from stage, distill string from use_kd bool)
2. **Scaling**: Adding any new configuration dimension requires restructuring the directory tree. A filesystem tree forces one fixed query ordering, but experiments are queried ad-hoc along different dimensions.

**Solution**: Separate the index from the storage. Use a shallow 2-level directory structure for files, and MLflow as the queryable metadata layer.

---

## Decision: Snakemake + MLflow (Revised 2026-01-31)

After evaluating 9 orchestration tools and 7 unified tracking+orchestration platforms, the conclusion is that no single tool does both orchestration and tracking well on HPC/SLURM:

- **Snakemake** stays for DAG orchestration + SLURM job management
- **MLflow** replaces the originally planned custom SQLite registry for experiment tracking

### Why MLflow over Custom SQLite

1. Already installed (3.8.1), zero setup
2. Auto-logging for Lightning stages via `mlflow.pytorch.autolog()`
3. Built-in web UI via OSC OnDemand (no custom Jupyter notebook needed)
4. Model registry with versioning and aliases
5. Industry standard, transferable skills

**What we lose**: Native SQL JOINs for teacher-student comparisons. Mitigation: `mlflow.search_runs()` into pandas DataFrame, do JOINs in pandas (~3-5 extra lines vs raw SQL).

### What Changed from Original Plan

| Original plan | Implemented |
|--------------|-------------|
| `pipeline/registry.py` (~130 lines custom SQLite) | **Dropped** — `pipeline/tracking.py` uses MLflow |
| Manual `register_start/complete/failed` calls | MLflow `start_run/end_run` + `autolog()` |
| Custom schema with `parent_id` FK | MLflow tags: `teacher_run_id` |
| `json_extract()` queries | `mlflow.search_runs()` + pandas |
| Jupyter query notebook against SQLite | `pipeline/query.py` CLI + MLflow UI via OnDemand |
| No model versioning | MLflow model registry with aliases |

### What Stayed the Same

- 2-level path simplification (8 -> 2 levels) — orthogonal to tracking choice
- Deterministic paths for Snakemake DAG
- `PipelineConfig` frozen dataclass (logged to MLflow as params)
- `pipeline/cli.py` as entry point (MLflow hooks added here)
- DVC for data versioning

---

## Current Architecture

### Filesystem (NFS home, permanent) -- owned by Snakemake

```
experimentruns/{dataset}/{model_size}_{stage}[_kd]/
    best_model.pt    # Snakemake output, DAG trigger for downstream rules
    config.json      # Frozen config (also logged to MLflow as params)
    metrics.json     # Evaluation stage only
```

### MLflow (GPFS scratch) -- supplementary metadata layer

```
/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db   # SQLite tracking DB
```

### Key Principles

- **Filesystem is for Snakemake, MLflow is for humans.** Models saved to NFS (permanent, Snakemake DAG trigger), logged to MLflow on scratch (supplementary).
- **If scratch purges**: All checkpoints and configs survive on NFS. Only MLflow tracking history is lost. Acceptable — MLflow is a convenience layer, not the source of truth.
- **Deterministic run IDs**: `{dataset}/{model_size}_{stage}[_kd]` — same convention used for filesystem paths, Snakemake DAG targets, and MLflow run names.

### Run ID Convention

```python
def run_id(cfg, stage):
    kd = "_kd" if cfg.use_kd else ""
    return f"{cfg.dataset}/{cfg.model_size}_{stage}{kd}"

# Examples:
# "hcrl_sa/teacher_autoencoder"
# "hcrl_sa/student_curriculum_kd"
# "set_01/teacher_fusion"
```

### MLflow Configuration

- **Tracking URI**: `sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db` (GPFS, safe for concurrent writes)
- **Experiment naming**: Single experiment `kd-gat-pipeline`; run names match filesystem paths
- **autolog**: Lightning stages (VGAE, GAT) use `mlflow.pytorch.autolog(log_models=False)` for epoch-level metrics; DQN fusion uses manual `mlflow.log_metrics()` calls
- **Teacher-student lineage**: `mlflow.set_tag("teacher_run_id", "<teacher_variant>")` on student runs; delta queries via `mlflow.search_runs()` into pandas

### Files Implemented

| File | Role |
|------|------|
| `pipeline/tracking.py` | MLflow integration: `setup_tracking()`, `start_run()`, `end_run()`, `log_failure()` |
| `pipeline/query.py` | CLI for querying experiments: `--all`, `--dataset`, `--leaderboard`, `--compare` |
| `pipeline/migrate.py` | Migration from 8-level to 2-level paths + MLflow backfill |
| `pipeline/paths.py` | 2-level `stage_dir()` + `run_id()` |
| `pipeline/cli.py` | MLflow run lifecycle hooks around stage dispatch |
| `pipeline/stages.py` | `mlflow.pytorch.autolog()` for Lightning; manual `log_metrics()` for DQN + eval |
| `pipeline/Snakefile` | 19 rules using 2-level `_p()` helper |

### Query Examples

```bash
python -m pipeline.query --all
python -m pipeline.query --dataset hcrl_sa --stage curriculum
python -m pipeline.query --leaderboard --top 10
python -m pipeline.query --compare teacher student_kd
python -m pipeline.query --running
```

---

## Database Research

Six embedded database options were evaluated for this HPC environment (OSC, NFS home, GPFS scratch, SLURM, Python/conda).

### Environment Facts

| Path | Filesystem | Safe for DBs? |
|------|-----------|---------------|
| `/users/PAS2022/rf15/` (home) | NFS v4 | Risky for concurrent writes |
| `/fs/scratch/PAS1266/` | GPFS (IBM Spectrum Scale) | Yes - true POSIX locking |
| `$TMPDIR` (in SLURM jobs) | local ext4 | Yes but ephemeral |

### Comparison

| Criterion | SQLite | DuckDB | TinyDB | LMDB | Polars/Parquet | MLflow |
|-----------|--------|--------|--------|------|----------------|--------|
| GPFS concurrent writes | Yes (busy_timeout) | No (single writer) | No | Risky | Yes (separate files) | SQLite backend: same as SQLite |
| SQL queries + JOINs | Full SQL + json_extract | Full SQL | No | No | Polars expressions | MLflow search API |
| Install needed | No (stdlib) | pip install | pip install | pip install | pip install | Already installed (3.8.1) |
| Currently available | Yes (3.51.1) | No | No | No | No | Yes |
| Complexity | Low | Low | Very low | Medium | Low | High |

### MLflow Deep-Dive

MLflow 3.8.1 is already installed and OSC provides an OnDemand MLflow app ([bc_osc_mlflow](https://github.com/OSC/bc_osc_mlflow)).

**MLflow advantages**:
- Already installed, ~15 lines of integration vs ~130 for custom
- PyTorch Lightning autolog captures params + metrics for 3/5 stages automatically (autoencoder, curriculum, normal -- but NOT DQN fusion or evaluation, which use custom loops)
- Built-in web UI via OSC OnDemand (launches `mlflow server` on a compute node via SLURM)
- Model registry with versioning and aliases
- Industry standard, well-documented

**MLflow limitations on OSC**:
- **Concurrent writes**: MLflow's SQLAlchemy layer doesn't expose SQLite pragmas. Mitigated by using GPFS scratch (reliable POSIX locking) and having at most 6 concurrent writers.
- **Snakemake integration**: MLflow run IDs are UUIDs, not deterministic. Filesystem paths remain the source of truth for Snakemake DAG construction.
- **Query power**: MLflow search API supports filtering but NOT SQL JOINs. Teacher-student comparisons require client-side merging in pandas.

### Concurrent SLURM Write Safety (Reference)

SQLite is a single-file database. Writes require an EXCLUSIVE lock on the entire file:
1. Job A (node-101) acquires lock -> writes (~5ms) -> releases
2. Job B (node-203) tries to lock -> blocked -> retries (busy_timeout) -> acquires -> writes
3. This works IF the filesystem implements `fcntl()` locking correctly

| Filesystem | Lock reliability | Safe for SQLite? |
|-----------|-----------------|------------------|
| NFS v4 (home dir) | Delegated to NFS server, often buggy | No |
| GPFS (scratch) | Distributed lock manager, POSIX-compliant | Yes |
| Local ext4 ($TMPDIR) | Kernel-mediated, reliable | Yes (but ephemeral) |

---

## Architecture Research (2026-01-31)

### Can We Collapse to One System?

A comprehensive evaluation of 9 orchestration tools and 7 unified tracking+orchestration platforms was conducted to determine if a single tool could replace both Snakemake and the experiment registry.

**Orchestration alternatives evaluated:**

| Tool | SLURM+DAG | Why rejected |
|------|-----------|-------------|
| Nextflow | Excellent SLURM, excellent DAG | Groovy language (not Python), bioinformatics-only community |
| Prefect | Poor SLURM (Dask bridge) | Cloud-native, compute nodes need connectivity to Prefect server |
| DVC pipelines | No native SLURM | `rw.lock` blocks concurrent jobs; Issue #7419 still open |
| Luigi | Via abandoned SciLuigi | Declining ecosystem, SLURM integration unmaintained |
| Metaflow | SLURM v0.0.4 (Jan 2026) | Promising but pre-alpha; SSH-based access doesn't match OSC architecture |
| Ray | Runs ON SLURM (single allocation) | Cannot orchestrate separate SLURM jobs; wrong model for multi-stage DAGs |
| Submitit | Excellent SLURM submission | No DAG support; job submission only, not orchestration |

**Unified platforms evaluated:**

| Platform | Tracking | Orchestration | SLURM | KD lineage | Verdict |
|----------|----------|---------------|-------|-----------|---------|
| ClearML | Excellent | PipelineController | First-class (May 2024) | InputModel/OutputModel (no SQL JOINs) | **Only real contender** -- but SLURM support is 8 months old, requires Docker/Singularity server, risky for PhD timeline |
| W&B | Excellent | Launch: no SLURM backend | Sweeps only | Via tags (manual) | Tracking-only; academic plan is free |
| ZenML | Good | No SLURM orchestrator | None | N/A | Cloud-native (Kubernetes) |
| Kedro | Deprecated | No scheduler | None | N/A | Corporate data science tool |
| DVC+CML | Good | Blocked by rw.lock | Broken for concurrent | Data lineage only | Concurrency issues are a dealbreaker |
| MLflow | Good | No DAG support | Single-job via NCSA plugin | Parent-child runs (limited queries) | **Best tracking option** -- already installed |
| Guild AI | N/A | N/A | None | N/A | Abandoned (acquired by Posit, team left) |

### Snakemake Path Requirements

Snakemake fundamentally requires deterministic file paths at DAG construction time:
- The entire DAG is built before any job executes (3 phases: init -> DAG -> scheduling)
- `checkpoint` rules allow DAG re-evaluation after execution, but are complex and have bugs (Snakemake 7.7-7.15 regression, Issue #1818)
- UUID-based paths (MLflow) require workarounds (manifests, symlinks) that negate Snakemake's incremental computation advantage
- The current `_p()` helper with deterministic paths is the correct pattern

### Key Insight: Orchestration and Tracking are Different Concerns

No single tool does both well on HPC/SLURM:
- Snakemake excels at DAG orchestration + SLURM job management
- MLflow excels at experiment tracking + model registry + visualization
- They complement each other without conflicting

---

## Historical: Original SQLite Design (Superseded)

> **Note**: This section documents the original custom SQLite registry design that was
> considered before the decision to use MLflow instead. It is preserved for reference
> only. The implementation uses MLflow -- see `pipeline/tracking.py` and `pipeline/query.py`.

### Database Schema (not implemented)

```sql
CREATE TABLE IF NOT EXISTS runs (
    -- Identity
    run_id       TEXT PRIMARY KEY,   -- "{dataset}/{model_size}_{stage}[_kd]"
    dataset      TEXT NOT NULL,
    model_size   TEXT NOT NULL,      -- teacher | student
    stage        TEXT NOT NULL,      -- autoencoder | curriculum | normal | fusion | evaluation
    model_arch   TEXT NOT NULL,      -- vgae | gat | dqn | eval
    use_kd       BOOLEAN NOT NULL DEFAULT 0,
    seed         INTEGER NOT NULL DEFAULT 42,

    -- Lifecycle
    status       TEXT NOT NULL DEFAULT 'running',
    created_at   TEXT NOT NULL,
    completed_at TEXT,
    wall_seconds REAL,
    error_msg    TEXT,

    -- File locations
    path         TEXT NOT NULL,
    checkpoint   TEXT,

    -- Lineage
    parent_id    TEXT REFERENCES runs(run_id),

    -- Denormalized metrics
    val_loss     REAL,
    accuracy     REAL,
    f1           REAL,
    precision_   REAL,
    recall       REAL,
    auc          REAL,
    mcc          REAL,
    best_epoch   INTEGER,

    -- Full snapshots
    config_json  TEXT NOT NULL,
    metrics_json TEXT
);
```

### Why it was superseded

1. MLflow provides auto-logging for Lightning stages (zero code vs ~130 lines)
2. MLflow has a built-in web UI via OSC OnDemand
3. MLflow model registry provides versioning and aliases for free
4. Teacher-student comparisons can be done in pandas (~3-5 extra lines)
5. Industry standard, transferable beyond this project

### Example SQLite queries (for reference)

```python
import sqlite3, pandas as pd
db = sqlite3.connect("/fs/scratch/PAS1266/kd_gat_registry.db")

# KD vs baseline comparison (native SQL JOIN -- not available in MLflow)
pd.read_sql("""
    SELECT s.dataset, s.stage,
           s.f1 AS student_f1, t.f1 AS teacher_f1,
           s.f1 - t.f1 AS delta_f1
    FROM runs s JOIN runs t ON s.parent_id = t.run_id
    WHERE s.use_kd = 1
    ORDER BY s.dataset, s.stage
""", db)

# Query any hyperparameter via json_extract
pd.read_sql("""
    SELECT run_id,
           json_extract(config_json, '$.lr') AS lr,
           json_extract(config_json, '$.gat_hidden') AS hidden,
           f1
    FROM runs WHERE model_arch='gat'
    ORDER BY f1 DESC
""", db)
```

### MLflow equivalents of the above

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db")
exp = mlflow.get_experiment_by_name("kd-gat-pipeline")

# KD vs baseline comparison (pandas merge)
students = mlflow.search_runs(experiment_ids=[exp.experiment_id],
    filter_string="tags.use_kd = 'True' AND tags.status = 'complete'")
teachers = mlflow.search_runs(experiment_ids=[exp.experiment_id],
    filter_string="tags.model_size = 'teacher' AND tags.status = 'complete'")
comparison = pd.merge(students, teachers, on="tags.dataset", suffixes=("_student", "_teacher"))
comparison["f1_delta"] = comparison["metrics.f1_student"] - comparison["metrics.f1_teacher"]

# Query by parameter
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
    filter_string="params.gat_hidden = '128'",
    order_by=["metrics.f1 DESC"])
```
