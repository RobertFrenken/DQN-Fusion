# KD-GAT Infrastructure Overhaul — Synthesized Plan

**Created**: 2026-02-22
**Last audited**: 2026-02-22
**Status**: Phases 0-3 complete. Phase 4 partial. Phase 5 not started.

## Context

The KD-GAT project (CAN bus intrusion detection via VGAE → GAT → DQN knowledge distillation) has accumulated five separate improvement plans spanning orchestration, data management, preprocessing, model enhancements, and GPU optimization. The project currently uses Prefect + dask-jobqueue for orchestration, but the decision is to migrate to Ray. uv is already the package manager. The codebase works but is fragile — data is scattered across NFS/scratch/S3/W&B, preprocessing is monolithic, and GATConv ignores the 11-D edge features.

This plan synthesizes all five user plans into a single prioritized execution order. Verification between phases is not a blocker — we work through bugs as they arise rather than gating on before/after comparisons.

---

## Phase 0: Foundation & Data Stabilization

**Why first**: Everything else depends on stable data paths and clean dependencies. Zero risk, immediate payoff.

### 0.1 — Data staging infrastructure

- Add `KD_GAT_DATA_ROOT` env var to a top-level config (e.g., `.env` or `config/paths.py`)
  - Now: defaults to home dir `/users/PAS2022/rf15/kd-gat-data/` (500GB capacity, sufficient for dev)
  - Later: swap to project storage `/fs/ess/PAS1266/kd-gat/` once 1TB is approved — only a path change
- Add `KD_GAT_CACHE_ROOT` env var override to `config/paths.py:cache_dir()`
- Create `scripts/stage_data.sh` (data root → scratch → `$TMPDIR` staging for jobs)
- One-time: copy caches to the data root and scratch
- Source plans: data_flow_and_storage, RAPIDS P0

### 0.2 — SLURM resource tuning

- Update SLURM scripts: `--cpus-per-task=8`, `--mem=85G`, `--gres=gpu:v100:1`
- Create `scripts/job_epilog.sh` (post-job GPU utilization report)
- Source plan: RAPIDS P1, P5

### 0.3 — Dependency cleanup

- Remove `mlflow>=2.9` from `pyproject.toml` (unused — no imports found)
- `rm uv.lock && uv sync --extra dev`

### Files

- `config/paths.py` — add env var overrides for data root and cache root
- `.env` — add `KD_GAT_DATA_ROOT` with home dir default
- `pyproject.toml` — remove mlflow
- New: `scripts/stage_data.sh`, `scripts/job_epilog.sh`
- Existing SLURM scripts — update directives

---

## Phase 1: Prefect → Ray Migration

**Why second**: Orchestration is the backbone. Ray unblocks HPO, parallel preprocessing, and multi-node training. Everything downstream benefits.

### 1.1 — Add Ray dependencies (keep Prefect temporarily)

- Add to `pyproject.toml`: `"ray[default]>=2.49"`, `"ray[tune]>=2.49"`, `optuna`
- `uv sync --extra dev`

### 1.2 — Ray SLURM bootstrap

- New: `scripts/ray_slurm.sh` — sbatch template using `ray symmetric-run` (Ray 2.49+)
- Uses `module load python/3.12` + uv venv activation (NOT conda)
- Sources `scripts/stage_data.sh`
- Parameterized: nodes, GPUs, entrypoint

### 1.3 — Ray orchestration layer

- New: `pipeline/orchestration/ray_pipeline.py`
  - Each stage as `@ray.remote(num_gpus=1)` wrapping existing `_run_stage()` subprocess dispatch
  - Preserve same DAG: preprocess → large → small_kd (depends on teacher) → small_nokd
  - Per-dataset fan-out via Ray ObjectRefs (replaces Prefect `wait(futures)`)
- New: `pipeline/orchestration/ray_slurm.py` — SLURM config helpers (replaces `slurm_config.py`)

### 1.4 — Update CLI dispatch

- Modify `pipeline/cli.py` `_run_flow()` to dispatch to Ray
- Add `--orchestrator ray|prefect` flag for migration period
- Keep `--local` flag working (Ray local mode = `ray.init()`)

### 1.5 — Remove Prefect

- Delete: `pipeline/flows/` directory (`train_flow.py`, `eval_flow.py`, `slurm_config.py`, `__init__.py`)
- Remove from `pyproject.toml`: `prefect>=3.0`, `prefect-dask>=0.3`, `dask-jobqueue>=0.9`
- Update CLAUDE.md: all Prefect/Dask references → Ray
- Update skills, session modes, architecture decisions sections
- `rm uv.lock && uv sync --extra dev`

### 1.6 — Ray Tune HPO scaffold

- New: `pipeline/orchestration/tune_config.py`
  - Search spaces per stage (VGAE, GAT, DQN)
  - OptunaSearch + ASHAScheduler
  - WandbLoggerCallback integration
  - Replaces `scripts/generate_sweep.py` parallel-command approach

### Key risk

The Prefect DAG logic in `train_flow.py` must be faithfully translated to Ray ObjectRef chains. Read it carefully before deleting.

### Files

- `pipeline/cli.py` — update `_run_flow()`
- `pyproject.toml` — swap deps
- New: `pipeline/orchestration/ray_pipeline.py`, `pipeline/orchestration/ray_slurm.py`, `pipeline/orchestration/tune_config.py`, `scripts/ray_slurm.sh`
- Delete: `pipeline/flows/` (4 files)

---

## Phase 2: Quick-Win PyG Enhancements + Profiler

**Why third**: Low-risk model improvements that produce immediate research results. Profiler guides future optimization decisions. Can begin as soon as Phase 1.3 is done.

### 2.1 — TransformerConv swap (highest research impact)

- Add `conv_type: Literal["gat", "gatv2", "transformer"] = "gat"` to `GATArchitecture` in `config/schema.py`
- Add `edge_dim: int = 11` field
- In `src/models/gat.py`, select conv class based on config
- TransformerConv natively uses `edge_attr` — the 11-D edge features (frequency, temporal intervals, bidirectionality, degree products) that GATConv currently ignores
- Same pattern in `src/models/vgae.py` encoder layers

### 2.2 — MultiAggregation pooling

- Add `pool_aggrs: tuple[str, ...] = ("mean",)` to `GATArchitecture`
- Replace `global_mean_pool` with `MultiAggregation` when multiple aggrs specified
- Update FC input dimension calculation accordingly
- Update `src/models/fusion_features.py` to use model's own pooling

### 2.3 — PyTorch Profiler integration

- Add `profile: bool = False`, `profile_steps: int = 5` to `TrainingConfig` in schema
- New: `ProfilerCallback` in `pipeline/stages/utils.py` (alongside existing `MemoryMonitorCallback`)
- Outputs Chrome trace to `experimentruns/{run_id}/profiler_traces/`
- Decision gate: profiler results determine whether cuGraph is worth pursuing in Phase 5

### 2.4 — PyGOD baselines (purely additive)

- Add `pygod` as optional dependency
- New: `scripts/run_pygod_baselines.py` — DOMINANT, OCGNN comparison
- Does not touch main pipeline

### Expected outcome

TransformerConv + MultiAggregation should improve F1 by leveraging the 11-D edge features that GATConv ignores. Profiler traces guide Phase 5 cuGraph decision.

### Files

- `config/schema.py` — add `conv_type`, `edge_dim`, `pool_aggrs`, `profile` fields
- `src/models/gat.py` — conditional conv class, MultiAggregation
- `src/models/vgae.py` — conditional conv class
- `src/models/fusion_features.py` — use model's pooling
- `pipeline/stages/utils.py` — ProfilerCallback
- New: `scripts/run_pygod_baselines.py`

---

## Phase 3: Preprocessing Refactor

**Why fourth**: Largest body of work, but mostly independent once orchestration is stable. Required for ICML cross-domain generalizability story. Ray (Phase 1) enables parallel file processing.

### 3.1 — Extract interfaces (no behavior change)

- New: `src/preprocessing/schema.py` — Pandera `StandardizedSchema` for the intermediate representation
- New: `src/preprocessing/vocabulary.py` — `EntityVocabulary` (replaces both `build_*_id_mapping` functions)
- New: `src/preprocessing/engine.py` — `GraphEngine` class (copy of current `create_graph_from_window` + feature functions)
- Test: old `preprocessing.py` and new modules produce identical `.pt` outputs

### 3.2 — CAN bus domain adapter

- New: `src/preprocessing/adapters/base.py` — `DomainAdapter` ABC with `discover_files`, `read_raw`, `build_vocabulary`, `to_standard_form`
- New: `src/preprocessing/adapters/can_bus.py` — wraps current CAN-specific logic
- Route: `CANBusAdapter.to_standard_form()` → `GraphEngine.create_graphs()`
- Test: output tensors bitwise identical to Phase 3.1

### 3.3 — Vectorize graph construction

- Replace `compute_edge_features` O(E×W) Python loop with `np.unique(..., return_inverse=True)` + `pd.groupby`
- Replace `compute_node_features` O(N×W) Python loop with `np.add.at` scatter operations
- Test: outputs within `atol=1e-6`. Benchmark speedup on largest dataset.

### 3.4 — Parallel file processing + caching

- Use Ray tasks (`@ray.remote`) for parallel CSV processing (each file independent)
- ShardedCache: write `.pt` shards to scratch, keyed by config hash
- Add Pandera validation at the standardized form boundary
- Delete old `preprocessing.py` once validated

### 3.5 — Network flow adapter (for ICML)

- New: `src/preprocessing/adapters/network_flow.py` — for UNSW-NB15, CICIDS
- Explicit source/target (IP addresses), no temporal adjacency shift needed
- Feature alignment: `nn.Linear(domain_features, hidden_dim)` projection layer in model

### Key risk

The monolith has implicit assumptions (column indexing, `shift(-1)` temporal adjacency) that must be preserved in the CAN bus adapter. Bugs here surface as silent wrong-tensor issues, not crashes.

### Files

- `src/preprocessing/preprocessing.py` — reference, then delete
- `src/training/datamodules.py` — update to use new engine
- New: `src/preprocessing/schema.py`, `src/preprocessing/vocabulary.py`, `src/preprocessing/engine.py`, `src/preprocessing/adapters/` (base, can_bus, network_flow)

---

## Phase 4: Data Lake & Storage Architecture

**Why fifth**: Can start in parallel with Phase 3. Uses home dir (`KD_GAT_DATA_ROOT`) immediately — the path configured in Phase 0. When project storage is approved, change one env var.

### 4.1 — Canonical directory structure

```
$KD_GAT_DATA_ROOT/          # Now: ~/kd-gat-data/  Later: /fs/ess/PAS1266/kd-gat/
├── raw/          # Immutable source CSVs (DVC-tracked)
├── cache/        # Preprocessed graph caches
├── runs/         # experimentruns mirror (weights, metrics)
└── lakehouse/    # Structured JSON metrics
```

All code references `KD_GAT_DATA_ROOT` — the migration to project storage is a single path change in `.env`.

### 4.2 — Dual-write lakehouse

- Modify `pipeline/lakehouse.py`: write locally (project storage) + S3 (fire-and-forget)
- Local write is primary (always succeeds on OSC); S3 is secondary
- DuckDB can query both: `read_json('/fs/ess/PAS1266/kd-gat/lakehouse/*.json')`

### 4.3 — W&B sync integration

- Keep streaming metrics to W&B cloud (real-time dashboards)
- Write identical metrics to local lakehouse (durability)
- Add post-job `wandb sync` to `scripts/ray_slurm.sh`

### 4.4 — Failed run handling

- Extend lakehouse records with `success: false` + failure reason + partial metrics
- Add `scripts/cleanup_orphans.sh` for incomplete runs

### Files

- `pipeline/lakehouse.py` — dual-write
- `scripts/ray_slurm.sh` — post-job W&B sync
- New: `scripts/cleanup_orphans.sh`

---

## Phase 5: Advanced Enhancements (Ongoing)

**Why last**: These depend on stable infrastructure from Phases 0-4 and profiling data from Phase 2.

### 5.1 — GNNExplainer integration

- Post-hoc interpretability using `torch_geometric.explain`
- New: `src/explain/explainer.py`
- Outputs per-node/per-edge importance to `experimentruns/{run_id}/explanations/`
- Dashboard force-graph visualization

### 5.2 — PyG Temporal (A3TGCN)

- New model variant: `src/models/temporal_gat.py`
- Requires temporal sequence grouping in preprocessing (consecutive windows → sequences)
- Strongest for slow-onset attacks

### 5.3 — cuGraph integration (decision gate)

- Only proceed if Phase 2.3 profiler shows >30% time in message passing AND TransformerConv `edge_attr` doesn't already capture the benefit
- `CuGraphGATConv` does NOT support `edge_attr` — may conflict with TransformerConv gains
- Requires separate RAPIDS conda env on OSC

### 5.4 — Trial-based batch size auto-tuning

- Binary search for max batch in `pipeline/memory.py`
- Replaces heuristic estimation
- Cache result per `(model_type, scale, GPU_type)` tuple

---

## Stale References to Clean Up

Address as part of each phase:

| Reference | Location | Fix | Phase |
|-----------|----------|-----|-------|
| `prefect>=3.0`, `prefect-dask`, `dask-jobqueue` | `pyproject.toml` | Replace with Ray | 1 |
| `mlflow>=2.9` | `pyproject.toml` | Remove (unused) | 0 |
| `pipeline/flows/` directory | 4 Python files | Delete after Ray works | 1 |
| "flow" stage in CLI | `pipeline/cli.py` | Update dispatch | 1 |
| "Prefect flows" in CLAUDE.md | Multiple sections | Rewrite for Ray | 1 |
| `module load miniconda3` / `conda activate` | `gnn-research-infrastructure.md` | Should be `module load python/3.12` + uv | 1 |
| SLURM `--cpus-per-task=4` / `--mem=32G` | Various SLURM scripts | Bump to 8/85G | 0 |

---

## Parallelization Opportunities

```
Phase 0 ──────► Phase 1 ──────► Phase 2 ──► Phase 5
                    │               │
                    └───► Phase 3 ──┘
                    │
                    └───► Phase 4 (when storage approved)
```

- Phase 2 and Phase 3 can run in parallel after Phase 1
- Phase 4 can start as soon as project storage is approved (independent of code changes)
- Phase 5 items are independent of each other

---

## Progress Tracker

_Updated as phases are completed. Check boxes when done._

- [x] **Phase 0**: Foundation & Data Stabilization ✅ (completed 2026-02)
  - [x] 0.1 Data staging infrastructure
  - [x] 0.2 SLURM resource tuning
  - [x] 0.3 Dependency cleanup
- [x] **Phase 1**: Prefect → Ray Migration ✅ (completed 2026-02)
  - [x] 1.1 Add Ray dependencies
  - [x] 1.2 Ray SLURM bootstrap
  - [x] 1.3 Ray orchestration layer
  - [x] 1.4 Update CLI dispatch
  - [x] 1.5 Remove Prefect
  - [x] 1.6 Ray Tune HPO scaffold
- [x] **Phase 2**: Quick-Win PyG Enhancements + Profiler ✅ (completed 2026-02)
  - [x] 2.1 TransformerConv swap (GAT + VGAE — _make_conv factory, edge_attr threading)
  - [x] 2.2 MultiAggregation pooling
  - [x] 2.3 PyTorch Profiler integration (ProfilerCallback + schema fields + make_trainer wiring)
  - [x] 2.4 PyGOD baselines (baselines dep group + scripts/run_pygod_baselines.py)
- [x] **Phase 3**: Preprocessing Refactor ✅ (completed 2026-02)
  - [x] 3.1 Extract interfaces (schema.py, vocabulary.py, engine.py, dataset.py)
  - [x] 3.2 CAN bus domain adapter (adapters/base.py, adapters/can_bus.py)
  - [x] 3.3 Vectorize graph construction (np.unique + scatter ops, 1.7x speedup)
  - [x] 3.4 Parallel file processing + caching (parallel.py + Ray @ray.remote, datamodules.py rewired, preprocessing.py deleted)
  - [x] 3.5 Network flow adapter (adapters/network_flow.py — UNSW-NB15 + CICIDS, explicit IP edges, schema-derived node features)
- [ ] **Phase 4**: Data Lake & Storage Architecture (PARTIAL)
  - [x] 4.1 Canonical directory structure (done via Phase 0 KD_GAT_DATA_ROOT)
  - [ ] 4.2 Dual-write lakehouse (S3 only — no local write)
  - [x] 4.3 W&B sync integration (in ray_slurm.sh)
  - [ ] 4.4 Failed run handling (no cleanup_orphans.sh)
- [ ] **Phase 5**: Advanced Enhancements (NOT STARTED)
  - [ ] 5.1 GNNExplainer integration
  - [ ] 5.2 PyG Temporal (A3TGCN)
  - [ ] 5.3 cuGraph integration (decision gate — needs Phase 2.3 profiler first)
  - [ ] 5.4 Batch size auto-tuning
