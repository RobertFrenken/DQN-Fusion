# Experiment Registry: Design & Implementation Plan

**Date**: 2026-01-31 (revised)

## Problem Statement

The current experiment output system encodes every configuration dimension as a directory level:

```
experimentruns/{modality}/{dataset}/{size}/{learning_type}/{model}/{distill}/{stage}/
```

This 8-level hierarchy has two issues:
1. **Redundancy**: 4 of 8 levels are deterministic (modality=always "automotive", learning_type/model_arch derived from stage, distill string from use_kd bool)
2. **Scaling**: Adding any new configuration dimension (preprocessing variant, loss function, augmentation) requires restructuring the directory tree. A filesystem tree forces one fixed query ordering, but experiments are queried ad-hoc along different dimensions.

**Solution**: Separate the index from the storage. Use a shallow directory structure for files and a SQLite database as the queryable index.

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

### Key Findings

- **SQLite 3.51.1** is in the stdlib with full JSON support (`json_extract()` confirmed working)
- **MLflow 3.8.1** is already installed in the conda env, but adds significant complexity for limited benefit at this scale
- **DuckDB** would be excellent for analytics but doesn't support concurrent writes from multiple SLURM jobs
- **TinyDB** and **LMDB** are not suitable (no concurrent access, no SQL)
- **Parquet files** are safe by design (one file per job) but require building all registry logic manually

### MLflow Deep-Dive (2026-01-31)

MLflow 3.8.1 is already installed and OSC provides an OnDemand MLflow app ([bc_osc_mlflow](https://github.com/OSC/bc_osc_mlflow)). A thorough comparison was done before making the final decision.

**MLflow advantages**:
- Already installed, ~15 lines of integration vs ~130 for custom
- PyTorch Lightning autolog captures params + metrics for 3/5 stages automatically (autoencoder, curriculum, normal — but NOT DQN fusion or evaluation, which use custom loops)
- Built-in web UI via OSC OnDemand (launches `mlflow server` on a compute node via SLURM)
- Model registry with versioning and aliases
- Industry standard, well-documented

**MLflow limitations on OSC**:
- **Concurrent writes**: MLflow's SQLAlchemy layer doesn't expose SQLite pragmas (`busy_timeout`, `journal_mode`). Without a persistent tracking server, concurrent SLURM jobs writing to the same SQLite file risk `SQLITE_BUSY` errors. Running a tracking server on a login node is fragile (OSC may kill long-running processes; [no deployed servers at OSC](https://www.osc.edu/resources/getting_started/howto/howto_using_mlflow_to_track_ml_training_and_models)).
- **File store workaround**: MLflow's `mlruns/` file store IS concurrency-safe (each run writes to its own directory). However, the file store backend is [deprecated as of MLflow 3.7](https://github.com/mlflow/mlflow/issues/18534) (Dec 2025) in favor of SQLite.
- **Snakemake integration**: MLflow run IDs are UUIDs, not deterministic. Snakemake's DAG requires predictable output paths. Using MLflow means maintaining two parallel systems: filesystem paths for Snakemake + MLflow DB for querying.
- **Query power**: MLflow search API supports filtering (`params.model_arch = 'gat' AND metrics.f1 > 0.9`) but NOT SQL JOINs. Teacher-student delta comparisons (core to the KD thesis) require client-side merging. No `parent_id` foreign key concept.
- **UI as SLURM job**: The OnDemand MLflow app works well for viewing — it launches on a compute node, user sets "Tracking URI directory" to data location, explores, kills session. But the app doesn't pass `--backend-store-uri` so it defaults to `mlruns/` (deprecated) or `mlflow.db` (auto-created).

**Viable MLflow workflow** (if chosen):
1. Training jobs set `MLFLOW_TRACKING_URI` to a directory on GPFS scratch
2. Each job writes independently to `mlruns/` (file store, concurrency-safe)
3. User launches OnDemand MLflow app pointed at that directory when exploring
4. Deterministic file paths kept for Snakemake; MLflow is a secondary metadata layer

**Why custom SQLite was chosen instead**:
1. **Single system**: Custom registry IS the filesystem index. No dual tracking.
2. **Pragma control**: Full control over `busy_timeout`, `journal_mode = DELETE` on GPFS.
3. **KD lineage**: `parent_id` FK enables `SELECT student.f1 - teacher.f1 FROM runs s JOIN runs t ON s.parent_id = t.run_id` — a core query for the thesis.
4. **json_extract()**: Any config parameter queryable without schema changes when adding new models/modalities.
5. **Deterministic run IDs**: Same convention as filesystem paths; Snakemake integration is native.

### Decision: Custom SQLite on GPFS scratch

SQLite wins because:
- Zero dependencies (Python stdlib)
- Full SQL with `json_extract()` for querying any config parameter
- JOIN support for teacher-student comparisons
- GPFS provides reliable POSIX file locking (unlike NFS)
- `PRAGMA busy_timeout` handles concurrent SLURM job writes
- Matches existing pattern (frozen JSON configs)
- Simple backup: `cp registry.db ~/backup/`

**Critical**: The database file MUST live on `/fs/scratch/PAS1266/` (GPFS), NOT on `/users/` (NFS). WAL mode must NOT be enabled on shared filesystems.

### Concurrent SLURM Write Safety (Reference)

SQLite is a single-file database. Writes require an EXCLUSIVE lock on the entire file:
1. Job A (node-101) acquires lock → writes (~5ms) → releases
2. Job B (node-203) tries to lock → blocked → retries (busy_timeout) → acquires → writes
3. This works IF the filesystem implements `fcntl()` locking correctly

| Filesystem | Lock reliability | Safe for SQLite? |
|-----------|-----------------|------------------|
| NFS v4 (home dir) | Delegated to NFS server, often buggy | No |
| GPFS (scratch) | Distributed lock manager, POSIX-compliant | Yes |
| Local ext4 ($TMPDIR) | Kernel-mediated, reliable | Yes (but ephemeral) |

`busy_timeout = 5000` makes writers retry for 5 seconds. At ~46 experiments, write contention is minimal.
`journal_mode = DELETE` avoids WAL mode, which uses shared memory files (`-shm`) that don't work across network nodes.

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
| ClearML | Excellent | PipelineController | First-class (May 2024) | InputModel/OutputModel (no SQL JOINs) | **Only real contender** — but SLURM support is 8 months old, requires Docker/Singularity server, risky for PhD timeline |
| W&B | Excellent | Launch: no SLURM backend | Sweeps only | Via tags (manual) | Tracking-only; academic plan is free |
| ZenML | Good | No SLURM orchestrator | None | N/A | Cloud-native (Kubernetes) |
| Kedro | Deprecated | No scheduler | None | N/A | Corporate data science tool |
| DVC+CML | Good | Blocked by rw.lock | Broken for concurrent | Data lineage only | Concurrency issues are a dealbreaker |
| MLflow | Good | No DAG support | Single-job via NCSA plugin | Parent-child runs (limited queries) | **Best tracking option** — already installed |
| Guild AI | N/A | N/A | None | N/A | Abandoned (acquired by Posit, team left) |

### Snakemake Path Requirements

Snakemake fundamentally requires deterministic file paths at DAG construction time:
- The entire DAG is built before any job executes (3 phases: init → DAG → scheduling)
- `checkpoint` rules allow DAG re-evaluation after execution, but are complex and have bugs (Snakemake 7.7-7.15 regression, Issue #1818)
- UUID-based paths (MLflow) require workarounds (manifests, symlinks) that negate Snakemake's incremental computation advantage
- The current `_p()` helper with deterministic paths is the correct pattern

### Key Insight: Orchestration and Tracking are Different Concerns

No single tool does both well on HPC/SLURM:
- Snakemake excels at DAG orchestration + SLURM job management
- MLflow excels at experiment tracking + model registry + visualization
- They complement each other without conflicting

### Revised Decision: Snakemake + MLflow

**Snakemake** stays for orchestration. **MLflow** replaces the custom SQLite registry for tracking.

**Why MLflow over custom SQLite:**
1. Already installed (3.8.1), zero setup
2. Auto-logging for Lightning stages (3/5 stages)
3. Built-in web UI via OSC OnDemand (no Jupyter notebook needed)
4. Model registry with versioning and aliases
5. Industry standard, transferable skills for career

**What we lose:** Native SQL JOINs for teacher-student comparisons. Mitigation: use `mlflow.search_runs()` into pandas DataFrame, do JOINs in pandas (~3-5 extra lines vs raw SQL).

**MLflow on OSC concurrency:** SQLite backend on GPFS scratch (`sqlite:////fs/scratch/PAS1266/mlflow/mlflow.db`). GPFS has reliable POSIX locking. With at most 6 concurrent writers per stage and ~5ms write duration, contention is negligible. File store (`mlruns/`) is an alternative but deprecated as of MLflow 3.7.

### What Changes from Original Plan

| Original plan | Revised plan |
|--------------|-------------|
| `pipeline/registry.py` (~130 lines custom SQLite) | **Dropped** — MLflow replaces it |
| Manual `register_start/complete/failed` calls | MLflow `start_run/end_run` + `autolog()` |
| Custom schema with `parent_id` FK | MLflow tags: `teacher_run_id` |
| `json_extract()` queries | `mlflow.search_runs()` + pandas |
| Jupyter query notebook against SQLite | MLflow UI via OnDemand + lightweight pandas notebook |
| No model versioning | MLflow model registry with aliases |

### What Stays the Same

- 2-level path simplification (8→2 levels) — orthogonal to tracking choice
- Deterministic paths for Snakemake DAG
- `PipelineConfig` frozen dataclass (logged to MLflow as params)
- `pipeline/cli.py` as entry point (MLflow hooks added here)
- DVC for data versioning

---

## Design

### Directory Structure (Before vs After)

**Before** (8 levels):
```
experimentruns/automotive/hcrl_sa/teacher/unsupervised/vgae/no_distillation/autoencoder/
    config.json, best_model.pt, logs/
```

**After** (2 levels + registry):
```
experimentruns/
    registry.db -> /fs/scratch/PAS1266/kd_gat_registry.db  (symlink)
    hcrl_sa/
        teacher_autoencoder/
            config.json, best_model.pt, logs/, metrics.json
        teacher_curriculum/
        teacher_fusion/
        student_autoencoder_kd/
        student_curriculum_kd/
        student_fusion_kd/
        student_autoencoder/
        student_curriculum/
        student_fusion/
        eval_teacher/
        eval_student_kd/
        eval_student/
    set_01/
        ...
```

Run names are deterministic: `{model_size}_{stage}` or `{model_size}_{stage}_kd`

### Database Schema

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
    status       TEXT NOT NULL DEFAULT 'running',  -- running | complete | failed
    created_at   TEXT NOT NULL,
    completed_at TEXT,
    wall_seconds REAL,
    error_msg    TEXT,               -- populated on failure

    -- File locations (relative to project root)
    path         TEXT NOT NULL,      -- experiment directory
    checkpoint   TEXT,               -- path to best_model.pt

    -- Lineage
    parent_id    TEXT REFERENCES runs(run_id),  -- teacher for KD runs

    -- Denormalized metrics (NULL until evaluation)
    val_loss     REAL,
    accuracy     REAL,
    f1           REAL,
    precision_   REAL,
    recall       REAL,
    auc          REAL,
    mcc          REAL,
    best_epoch   INTEGER,

    -- Full snapshots (queryable via json_extract)
    config_json  TEXT NOT NULL,
    metrics_json TEXT               -- full metrics dict on completion
);

CREATE INDEX IF NOT EXISTS idx_dataset    ON runs(dataset);
CREATE INDEX IF NOT EXISTS idx_stage      ON runs(stage);
CREATE INDEX IF NOT EXISTS idx_status     ON runs(status);
CREATE INDEX IF NOT EXISTS idx_model_size ON runs(model_size);
CREATE INDEX IF NOT EXISTS idx_use_kd     ON runs(use_kd);
```

### Run ID Convention

Deterministic from config, so Snakemake can compute paths without querying the DB:

```python
def run_id(cfg, stage):
    kd = "_kd" if cfg.use_kd else ""
    return f"{cfg.dataset}/{cfg.model_size}_{stage}{kd}"

# Examples:
# "hcrl_sa/teacher_autoencoder"
# "hcrl_sa/student_curriculum_kd"
# "set_01/teacher_fusion"
```

### Connection Configuration

```python
DB_PATH = Path("/fs/scratch/PAS1266/kd_gat_registry.db")

def _connect():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA busy_timeout = 5000")    # retry on lock contention
    conn.execute("PRAGMA journal_mode = DELETE")   # NOT WAL (shared filesystem)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn
```

---

## Implementation Plan

### Phase 1: Registry Module (`pipeline/registry.py`)

New file ~120 lines. Functions:

| Function | Purpose |
|----------|---------|
| `_connect()` | Get DB connection with GPFS-safe pragmas |
| `_ensure_schema()` | Create tables + indexes if not exist |
| `run_id(cfg, stage)` | Deterministic ID from config |
| `register_start(cfg, stage)` | INSERT row with status='running' |
| `register_complete(cfg, stage, metrics)` | UPDATE with status='complete', metrics, wall_seconds |
| `register_failed(cfg, stage, error)` | UPDATE with status='failed', error_msg |
| `find_runs(**filters)` | Query helper returning list of Row objects |
| `get_run(run_id)` | Fetch single run by ID |
| `get_lineage(run_id)` | Fetch run + parent chain |

### Phase 2: Path Simplification (`pipeline/paths.py`)

Replace `stage_dir()`:

```python
# BEFORE: 8 levels
def stage_dir(cfg, stage):
    learning_type, model_arch, mode = STAGES[stage]
    distillation = "distilled" if cfg.use_kd else "no_distillation"
    return (
        Path(cfg.experiment_root) / cfg.modality / cfg.dataset
        / cfg.model_size / learning_type / model_arch
        / distillation / mode
    )

# AFTER: 2 levels
def stage_dir(cfg, stage):
    kd = "_kd" if cfg.use_kd else ""
    return Path(cfg.experiment_root) / cfg.dataset / f"{cfg.model_size}_{stage}{kd}"
```

`STAGES` dict is still needed for model_arch lookup (used by registry and stages.py), but no longer drives the path.

### Phase 3: CLI Integration (`pipeline/cli.py`)

Add registry hooks at the two dispatch points:

```python
from .registry import register_start, register_complete, register_failed

# Before dispatch (~line 109)
register_start(cfg, args.stage)

# After dispatch (~line 116)
try:
    result = STAGE_FNS[args.stage](cfg)
    register_complete(cfg, args.stage, result)
except Exception as e:
    register_failed(cfg, args.stage, str(e))
    raise
```

### Phase 4: Snakefile Update (`pipeline/Snakefile`)

Replace `_p()` helper:

```python
# BEFORE
def _p(ds, size, learn, arch, distill, mode):
    return f"{EXP}/{MOD}/{ds}/{size}/{learn}/{arch}/{distill}/{mode}/best_model.pt"

# AFTER
def _p(ds, size, stage, kd=False):
    suffix = "_kd" if kd else ""
    return f"{EXP}/{ds}/{size}_{stage}{suffix}/best_model.pt"
```

All 18 rules update their input/output paths. Rule logic is unchanged.

### Phase 5: Query Notebook (`notebooks/query_registry.ipynb`)

Jupyter notebook for OSC OnDemand with:
- `pandas.read_sql()` queries against the registry
- Dropdown filters (dataset, stage, model_size) via `ipywidgets` (needs `pip install ipywidgets`)
- Comparison tables (teacher vs student, KD vs no-KD)
- Leaderboard sorted by F1/accuracy

Fallback for no-Jupyter environments: `pipeline/registry.py` includes a `main()` function for CLI queries:

```bash
python -m pipeline.registry --dataset hcrl_sa --stage curriculum --sort-by f1
python -m pipeline.registry --status running
python -m pipeline.registry --top 3 --stage evaluation
```

### Phase 6: Migration

Move existing `experimentruns/` checkpoints from old 8-level paths to new 2-level paths. Backfill the registry by scanning existing `config.json` + `metrics.json` files.

```python
def backfill_registry():
    """Scan existing experiment dirs and populate registry from config.json files."""
    for cfg_file in Path("experimentruns").rglob("config.json"):
        cfg = PipelineConfig.load(cfg_file)
        stage = infer_stage_from_path(cfg_file)
        register_backfill(cfg, stage, cfg_file.parent)
```

---

## Example Queries

```python
import sqlite3, pandas as pd
db = sqlite3.connect("/fs/scratch/PAS1266/kd_gat_registry.db")

# Best model per dataset
pd.read_sql("""
    SELECT dataset, model_size, f1, accuracy, path
    FROM runs WHERE stage='evaluation' AND status='complete'
    ORDER BY dataset, f1 DESC
""", db)

# Top 3 GAT classifiers by F1
pd.read_sql("""
    SELECT run_id, dataset, f1, accuracy, wall_seconds/60 AS minutes
    FROM runs WHERE model_arch='gat' AND f1 IS NOT NULL
    ORDER BY f1 DESC LIMIT 3
""", db)

# KD vs baseline comparison
pd.read_sql("""
    SELECT s.dataset, s.stage,
           s.f1 AS student_f1, t.f1 AS teacher_f1,
           s.f1 - t.f1 AS delta_f1
    FROM runs s JOIN runs t ON s.parent_id = t.run_id
    WHERE s.use_kd = 1
    ORDER BY s.dataset, s.stage
""", db)

# What's currently running?
pd.read_sql("""
    SELECT run_id, created_at, stage FROM runs WHERE status='running'
""", db)

# Query any hyperparameter without schema changes
pd.read_sql("""
    SELECT run_id,
           json_extract(config_json, '$.lr') AS lr,
           json_extract(config_json, '$.gat_hidden') AS hidden,
           f1
    FROM runs WHERE model_arch='gat'
    ORDER BY f1 DESC
""", db)

# Average wall time by stage
pd.read_sql("""
    SELECT stage, model_size,
           ROUND(AVG(wall_seconds)/60, 1) AS avg_min,
           COUNT(*) AS n
    FROM runs WHERE status='complete'
    GROUP BY stage, model_size
""", db)
```

---

## Files Modified

| File | Change |
|------|--------|
| `pipeline/registry.py` | **New** ~120 lines. Registry API. |
| `pipeline/paths.py` | Simplify `stage_dir()` from 8 levels to 2. Keep `STAGES` dict. |
| `pipeline/cli.py` | Add `register_start/complete/failed` hooks around dispatch. |
| `pipeline/Snakefile` | Replace `_p()` helper, update all 18 rule input/output paths. |
| `pipeline/__init__.py` | No change (registry is imported by cli.py only). |
| `.gitignore` | Add `*.db` to experiment output patterns (registry is on scratch, symlinked). |
| `notebooks/query_registry.ipynb` | **New**. Query UI for Jupyter on OnDemand. |

## Files NOT Modified

| File | Why |
|------|-----|
| `pipeline/stages.py` | Stages call `paths.stage_dir()` — they don't know about the path structure. |
| `pipeline/config.py` | PipelineConfig is unchanged. `config_json` in DB is `asdict(cfg)`. |
| `src/**` | No changes to models, preprocessing, or datamodules. |

## Dependencies

| Package | Status | Needed For |
|---------|--------|-----------|
| `sqlite3` | stdlib (3.51.1) | Registry |
| `pandas` | installed (2.3.3) | Query convenience |
| `ipywidgets` | NOT installed | Jupyter dropdown UI (optional) |

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Scratch 90-day purge deletes registry.db | Periodic backup: `cp /fs/scratch/.../registry.db ~/backup/`. Also, registry can be rebuilt from `config.json` + `metrics.json` files via `backfill_registry()`. |
| Concurrent SLURM writes cause lock contention | `PRAGMA busy_timeout = 5000` retries for 5 seconds. At 46 total experiments, contention is minimal. |
| Old experiment paths break | Migration script moves checkpoints to new paths. Backfill populates registry from existing configs. |
| Snakemake DAG changes | `_p()` helper update is mechanical — same rules, same dependencies, shorter paths. Dry run (`-n`) validates before real run. |

## Implementation Order

1. `pipeline/registry.py` — can be built and tested independently
2. `pipeline/paths.py` — path simplification (small, focused change)
3. `pipeline/cli.py` — registry hooks (3 lines added)
4. `pipeline/Snakefile` — path updates (mechanical, validate with dry run)
5. Migration script — move existing checkpoints, backfill registry
6. Query notebook — build after registry has data
