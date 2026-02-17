# Data Engineering Tooling Exploration for KD-GAT

## Context

The KD-GAT project has outgrown its initial tooling. Two migration plans (V1 and V2) plus a pragmatic fix (planpipeline.md) have been written. The pragmatic fixes (SQLite retry, WAL deferral, Snakemake sentinel hardening) are implemented. This document explores what a comprehensive data engineering foundation looks like — paradigms, patterns, and tools — as the project's demands grow.

This is a **reference document**, not an implementation plan. It captures research findings to inform future migration decisions alongside the existing `MLOPS_MIGRATION_PLAN.md` and `MLOPS_MIGRATION_PLAN_V2.md`.

---

## Current State: What Already Works Well

The project already implements many data engineering best practices under different names:

| Pattern | Current Implementation |
|---------|----------------------|
| Medallion architecture | CSV → Parquet → .pt cache → model outputs |
| Idempotency | Sentinel files + content-based rerun triggers |
| Config schema validation | Pydantic v2 frozen models |
| Pipeline pattern | Snakemake DAG + CLI dispatch |
| Metadata tracking | SQLite write-through + config.json per run |
| Hybrid FP/OOP | Pure-function stages + OOP models/DB |
| Data versioning | DVC for raw data |
| Layered architecture | 3-layer import hierarchy with boundary tests |

**Current data scale:** 10.3 GB raw CSV across 6 datasets (333 files), producing 14.3 GB of cached .pt graph files (797,891 graphs total). All caches built 2026-02-16.

---

## Part 1: Practices & Patterns

### 1.1 StartDataEngineering.com

**Central recommendation:** Data transformations as pure functions; system interactions (DB, GPU, filesystem) as OOP classes. KD-GAT already follows this pattern.

**Key practices:**
- **Medallion / 3-hop model** (Bronze→Silver→Gold) — already implemented as CSV→Parquet→.pt
- **Idempotent pipelines** — already implemented via sentinels + content triggers
- **Data quality gates** — **gap**: no DataFrame-level validation between layers
- **Testing pyramid** — unit + integration + e2e, partially implemented
- **Coding patterns** — Factory (dispatch), Strategy (swappable transforms), Decorators (cross-cutting concerns)
- **Metadata/logging** — partially implemented (DB write-through), gap in lineage closure

Sources: [Best Practices](https://www.startdataengineering.com/post/de_best_practices/), [Design Patterns](https://www.startdataengineering.com/post/design-patterns/), [Code Patterns](https://www.startdataengineering.com/post/code-patterns/), [FP vs OOP](https://www.startdataengineering.com/post/python-fp-v-oop/)

### 1.2 Pipeline Paradigms: Function-Based vs Object-Oriented

| Aspect | Function-Based | Object-Oriented |
|--------|---------------|-----------------|
| Testing | Easy (pure functions, no state) | Harder (mock state, setUp/tearDown) |
| Parallelism | Natural (no shared mutable state) | Requires careful synchronization |
| State management | Awkward (pass through args) | Natural (encapsulated in instances) |
| External connections | Needs wrappers | Natural (context managers) |
| PyTorch fit | Transforms, preprocessing | nn.Module, DataLoader |

**Industry consensus:** Hybrid. KD-GAT already does this correctly — `config/schema.py` (Pydantic OOP), `pipeline/stages/` (functional), `src/models/` (nn.Module OOP), `pipeline/db.py` (context manager OOP).

### 1.3 Data Engineering Patterns

**ETL vs ELT:** KD-GAT correctly uses ETL — transforms (sliding-window graph construction) are compute-heavy Python/PyG operations that can't be expressed as SQL.

**Data Contracts:** Config layer covered by Pydantic v2. **Gap:** no DataFrame-level validation (Pandera recommended).

**Lineage:** Partial — Snakemake DAG (implicit), config.json per run, project.db, DVC. **Gap:** no single query connecting metrics.json to code commit + config + data version.

**Idempotency:** Strong via sentinels + content triggers. **Gap:** seed pinning audit needed.

---

## Part 2: BDD Testing (pytest-bdd)

### 2.1 How pytest-bdd Works

pytest-bdd is a pytest plugin implementing Gherkin BDD on top of pytest. Unlike standalone BDD frameworks (Behave, Cucumber), tests are collected from `test_*.py` files and run by pytest's standard runner. All existing fixtures, markers, and parametrize work unchanged.

**Three-part structure:**
1. **Feature files** (`.feature`) — Gherkin syntax, human-readable scenarios
2. **Step definitions** — Python functions decorated with `@given`/`@when`/`@then`
3. **Test binding files** (`test_*.py`) — Connect features to pytest via `scenarios()`

**Project structure if adopted:**
```
tests/
  conftest.py                    # Existing (unchanged)
  test_layer_boundaries.py       # Existing (unchanged)
  features/                      # NEW — Gherkin feature files
    ingestion.feature
    graph_construction.feature
    config_resolution.feature
    training_pipeline.feature
    kd_transfer.feature
  step_defs/                     # NEW — Step implementations
    conftest.py                  # BDD-specific shared fixtures (BDDContext dataclass)
    test_ingestion.py
    test_graph_construction.py
    test_config_resolution.py
    test_training_pipeline.py
    test_kd_transfer.py
```

### 2.2 Example Feature Files

**Data Ingestion:**
```gherkin
Feature: CAN Bus Data Ingestion
  Scenario: Valid CAN CSV converts to Parquet with correct schema
    Given a CAN CSV file with 500 messages and columns "timestamp,id,data,label"
    When the CSV is ingested to Parquet
    Then the Parquet file exists
    And the Parquet schema has columns "timestamp,id,data_field,label,source_file"
    And the Parquet column "id" has type "uint32"
    And the Parquet row count is 500

  Scenario: Hex CAN IDs are parsed to integers
    Given a CAN CSV file with hex IDs "1A,FF,0,7DF"
    When the CSV is ingested to Parquet
    Then the Parquet column "id" contains values "26,255,0,2015"
```

**Graph Construction with Scenario Outline (parametric):**
```gherkin
Feature: Sliding Window Graph Construction
  Scenario: OOV CAN IDs are mapped correctly
    Given a CAN dataset with known IDs "1A,2B,3C"
    And a test message with unseen ID "FF"
    When graphs are constructed with the training ID mapping
    Then the unseen ID maps to the OOV index
    And the ID mapping size does not increase

  Scenario Outline: Window size controls graph count
    Given a CAN dataset with <total> messages
    And a window size of <window> and stride of <stride>
    When graphs are constructed
    Then <expected> graphs are created

    Examples:
      | total | window | stride | expected |
      | 100   | 100    | 100    | 1        |
      | 200   | 100    | 100    | 2        |
      | 150   | 100    | 50     | 2        |
      | 50    | 100    | 100    | 0        |
```

**Config Resolution:**
```gherkin
Feature: YAML Config Resolution
  Scenario: KD auxiliary adds knowledge distillation config
    Given config for "gat" scale "small" with auxiliary "kd_standard"
    Then has_kd is true
    And kd temperature is greater than 0
    And kd alpha is between 0 and 1

  Scenario Outline: All model/scale combinations resolve
    Given config for "<model>" scale "<scale>"
    Then the config is a frozen PipelineConfig

    Examples:
      | model | scale |
      | vgae  | large |
      | vgae  | small |
      | gat   | large |
      | gat   | small |
      | dqn   | large |
      | dqn   | small |
```

**Training Pipeline Smoke Test:**
```gherkin
@slurm
Feature: Training Pipeline Smoke Test
  Background:
    Given synthetic CAN graphs with 50 samples and 11 features
    And smoke test config overrides

  Scenario: VGAE trains and produces finite loss
    Given config for "vgae" scale "large"
    And a VGAEModule initialized from the config
    When the model trains for 2 epochs on CPU
    Then training loss is finite
    And training loss is not None

  Scenario: DQN fusion agent learns from replay buffer
    Given a DQN agent with buffer_size 500 and hidden 64
    And 100 random experiences in the replay buffer
    When the agent trains for 10 steps
    Then at least 1 training loss is produced
    And all losses are finite
```

**KD Transfer Validation:**
```gherkin
@slurm
Feature: Knowledge Distillation Transfer
  Background:
    Given smoke test config overrides

  Scenario: VGAE student trains with teacher guidance
    Given a trained VGAE teacher model at "large" scale
    And a VGAE student config at "small" scale with KD
    And a projection layer matching teacher to student dimensions
    When the student trains for 2 epochs with the teacher
    Then training loss is finite
    And the student config has_kd is true
```

### 2.3 Step Definition Pattern

Steps communicate via a shared `BDDContext` dataclass to avoid fixture proliferation:

```python
# tests/step_defs/conftest.py
@dataclass
class BDDContext:
    config: Any = None
    dataset: list[Data] = field(default_factory=list)
    graphs: list[Data] = field(default_factory=list)
    model: Any = None
    teacher: Any = None
    losses: list[float] = field(default_factory=list)
    error: Exception | None = None
    parquet_path: Path | None = None
    csv_path: Path | None = None
    tmp_dir: Path | None = None
    overrides: dict = field(default_factory=dict)
    id_mapping: dict | None = None

@pytest.fixture
def ctx():
    return BDDContext()
```

Step definitions use `parsers.parse()` for parametric matching:
```python
@given(parsers.parse('config for "{model}" scale "{scale}" with auxiliary "{aux}"'))
def _resolve_with_aux(ctx, model, scale, aux):
    from config import resolve
    ctx.config = resolve(model, scale, auxiliaries=aux)

@then(parsers.parse('the Parquet column "{col}" has type "{dtype}"'))
def _check_dtype(ctx, col, dtype):
    import pyarrow as pa, pyarrow.parquet as pq
    type_map = {"uint32": pa.uint32(), "uint8": pa.uint8(), "float64": pa.float64()}
    table = pq.read_table(ctx.parquet_path)
    actual = table.schema.field(col).type
    assert actual == type_map[dtype]
```

### 2.4 Integration with Existing pytest

- BDD tests run alongside regular tests: `python -m pytest tests/ -v`
- Gherkin `@slurm` tags map to `pytest.mark.slurm` — existing skip logic works unchanged
- Existing `conftest.py` fixtures (`_make_graph`, `SMOKE_OVERRIDES`) available to BDD steps automatically
- Run BDD-only: `python -m pytest tests/step_defs/ -v`
- Run non-BDD: `python -m pytest tests/ -v --ignore=tests/step_defs/`
- Config: `bdd_features_base_dir = "tests/features/"` in `pyproject.toml`

### 2.5 Assessment: When pytest-bdd Adds Value

| Use Case | Value | Rationale |
|----------|-------|-----------|
| Config resolution specs | **Good** | Complex YAML composition; Gherkin makes contract explicit |
| Data ingestion contracts | **Good** | Clear input/output schema; documents valid ingestion |
| Graph construction behaviors | **Moderate** | OOV mapping and windowing worth documenting |
| Training smoke tests | **Marginal** | Existing pytest already clear |
| KD transfer | **Marginal** | Tightly coupled to PyTorch internals |
| Layer boundary tests | **None** | AST-based structural tests, not behavioral |

**Maintenance cost:** ~15 min per scenario (vs ~10 min for equivalent pytest). Two-file problem (feature + steps). Step reuse payoff after ~15 scenarios. Estimated: 5 feature files, 25 scenarios, 80 step definitions, 4-6 hours initial setup.

**Recommendation:** Use selectively for integration/contract tests (config, ingestion), not universally. Do not retroactively convert existing test files. BDD shines when collaborators or reviewers need to read specs without knowing Python — if that audience exists, adopt; if not, pytest is sufficient.

### 2.6 Comparison: Behave vs pytest-bdd vs Cucumber

| Tool | Fit for KD-GAT | Reason |
|------|----------------|--------|
| **Gherkin** (Given/When/Then DSL) | Low | Adds verbosity without an audience |
| **Behave** (standalone Python BDD) | Low | Separate runner, no pytest integration, no fixture reuse |
| **Cucumber** (JVM-native BDD) | None | Wrong language ecosystem entirely |
| **pytest-bdd** (Gherkin inside pytest) | Low-Medium | Full pytest integration if BDD is adopted |

---

## Part 3: Stage Decorator Pattern

### 3.1 Current State

`cli.py` already calls `record_run_start()` before dispatch and `record_run_end()` after, with try/except for failures. Memory utilities exist in `pipeline/tracking.py` and `pipeline/memory.py`. The `_ON_COMPUTE_NODE` guard skips DB writes on SLURM compute nodes.

### 3.2 Core `@track_stage` Decorator

Lives in `pipeline/decorators.py`. Wraps stage functions with timing, memory, and error tracking **without changing behavior**:

```python
@dataclass
class StageMetrics:
    stage_name: str
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0      # CUDA event timing (0 if no GPU)
    peak_rss_mb: float = 0.0      # resource.getrusage
    peak_gpu_mb: float = 0.0      # torch.cuda.max_memory_allocated
    gpu_allocated_start_mb: float = 0.0
    gpu_allocated_end_mb: float = 0.0
    success: bool = False
    error_type: str = ""
    error_message: str = ""
    error_traceback: str = ""
    started_at: str = ""
    completed_at: str = ""

def track_stage(
    stage_name: str | None = None,
    *,
    record_to_db: bool = False,
    reset_gpu_stats: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wraps a stage function with timing, memory, and error tracking.

    Args:
        stage_name: Human-readable name for logging. Defaults to function name.
        record_to_db: Whether to write metrics to DB. Default False because
                      cli.py already handles DB recording — enabling this would
                      cause double-recording. Set True only if calling stages
                      outside cli.py (e.g., from tests or notebooks).
        reset_gpu_stats: Reset CUDA peak memory stats before the stage runs,
                         so peak_gpu_mb reflects only this stage's usage.
    """
```

Key design decisions:
- **Never swallows exceptions** — re-raises in `finally` block after recording metrics
- **CUDA events created at call time**, not decoration time — safe with `spawn` multiprocessing
- **`record_to_db=False` by default** — avoids double-recording with existing `cli.py`
- **Respects `_ON_COMPUTE_NODE` guard** — no DB writes from SLURM workers

Implementation sketch:
```python
def decorator(fn: Callable[P, R]) -> Callable[P, R]:
    name = stage_name or fn.__name__

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        metrics = StageMetrics(stage_name=name)
        metrics.started_at = datetime.now(timezone.utc).isoformat()

        if reset_gpu_stats and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # CUDA event timing (created at call time, NOT decoration time)
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        wall_start = time.monotonic()
        try:
            result = fn(*args, **kwargs)
            metrics.success = True
            return result
        except Exception as exc:
            metrics.success = False
            metrics.error_type = type(exc).__name__
            metrics.error_traceback = traceback.format_exc()
            raise  # NEVER swallow
        finally:
            metrics.wall_seconds = time.monotonic() - wall_start
            metrics.peak_rss_mb = _get_peak_rss_mb()
            metrics.peak_gpu_mb = _get_gpu_peak_mb()
            log.info("[%s] %s in %.1fs | RSS=%.0fMB GPU_peak=%.0fMB",
                     name, "OK" if metrics.success else "FAILED",
                     metrics.wall_seconds, metrics.peak_rss_mb, metrics.peak_gpu_mb)
            if record_to_db and not bool(os.environ.get("SLURM_JOB_ID")):
                _record_stage_metrics(metrics)

    wrapper._stage_name = name
    wrapper._tracked = True
    return wrapper
```

### 3.3 Composable Decorator Stack

```python
@retry_on_failure(max_attempts=2, retryable=(OSError, IOError))  # outermost
@track_stage("autoencoder")                                       # timing per attempt
@validate_inputs                                                  # type-check config
@validate_outputs(["best_model.pt", "config.json"])               # check artifacts
def train_autoencoder(cfg: PipelineConfig) -> Path:
    ...
```

**Stacking order matters:**
1. `retry_on_failure` outermost — retries entire tracked execution on NFS errors
2. `track_stage` next — each attempt gets fresh timing
3. `validate_inputs` before outputs — no point checking outputs if inputs wrong
4. `validate_outputs` innermost — needs the return value

### 3.4 Individual Decorators

**`@retry_on_failure(max_attempts, retryable, backoff_base)`**

Handles within-process transient errors (NFS stale handles). Deliberately narrow: retries only `(OSError, IOError, TimeoutError)`, not `RuntimeError` (CUDA OOM) or `ValueError` (config errors). Complementary to Snakemake's job-level retry.

```python
def retry_on_failure(
    max_attempts: int = 2,
    retryable: tuple[type[Exception], ...] = (OSError, IOError, TimeoutError),
    backoff_base: float = 5.0,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable as exc:
                    if attempt == max_attempts:
                        raise
                    time.sleep(backoff_base * attempt)
        return wrapper
    return decorator
```

**`@validate_inputs`**

Lightweight guard that the first argument is a `PipelineConfig`. Catches cases where someone passes a dict from a notebook.

```python
def validate_inputs(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if args and not isinstance(args[0], PipelineConfig):
            raise TypeError(f"{fn.__name__} expects PipelineConfig, got {type(args[0]).__name__}")
        return fn(*args, **kwargs)
    return wrapper
```

**`@validate_outputs(expected)`**

Post-stage check that expected artifacts exist. **Warns only, never blocks** — missing artifacts may be non-fatal (optional embeddings.npz on early stopping).

```python
def validate_outputs(expected: Callable | list[str]):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            # Resolve paths and check existence
            missing = [p for p in resolved_paths if not p.exists()]
            if missing:
                log.warning("[validate_outputs] %s: missing: %s", fn.__name__, missing)
            return result
        return wrapper
    return decorator
```

**`gpu_context` — Context Manager (NOT a decorator)**

```python
@contextmanager
def gpu_context(device: str = "cuda", empty_cache: bool = True):
    """Context manager for GPU lifecycle.

    Why context manager, not decorator:
      - Device string comes from PipelineConfig at runtime, not known at decoration time
      - Cleanup (empty_cache) should happen even on failure
      - Composing with @track_stage as a decorator creates ordering ambiguity
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    try:
        yield device_obj
    finally:
        if device_obj.type == "cuda" and empty_cache:
            torch.cuda.empty_cache()
```

### 3.5 Integration with cli.py

**Option A (recommended, minimal):** Apply decorators to stage functions in `pipeline/stages/`. Leave `cli.py` dispatch unchanged. The `try/except` in `cli.py` continues to own DB recording. Decorators add logging/timing transparently. Zero risk of double-recording.

```python
# cli.py — NO CHANGES NEEDED
try:
    result = STAGE_FNS[args.stage](cfg)  # decorators fire transparently
    record_run_end(run_name, success=True, metrics=...)
except Exception as e:
    record_run_end(run_name, success=False)
    raise
```

**Option B (future):** Move DB recording into `@track_stage(record_to_db=True)`. Requires `stage_metrics` DB table. Bigger refactor, deferred.

**Snakemake interaction:** Snakemake retries at job level (SLURM preemption, OOM kills). `@retry_on_failure` handles process-level NFS flakiness. Complementary, not conflicting.

### 3.6 Anti-Patterns to Avoid

- **Never apply to inner functions** — stage-level only (4 functions). ~1us overhead is negligible for hour-long stages but deadly in per-batch loops or graph construction.
- **Never swallow exceptions** — every decorator re-raises. `cli.py` must see failures.
- **No CUDA state in closures** — events created at call time, not decoration time, to be safe with `spawn`.
- **Avoid double-recording** — `record_to_db=False` by default since cli.py owns DB writes.
- **Avoid hiding control flow** — decorators only add observability, never change behavior.

### 3.7 Testing Decorators

```python
# tests/test_decorators.py
class TestTrackStage:
    def test_passthrough_return_value(self, sample_cfg):
        @track_stage("test")
        def my_stage(cfg): return "result_42"
        assert my_stage(sample_cfg) == "result_42"

    def test_exception_reraise(self, sample_cfg):
        @track_stage("test")
        def failing(cfg): raise ValueError("deliberate")
        with pytest.raises(ValueError, match="deliberate"):
            failing(sample_cfg)

    @patch.dict("os.environ", {"SLURM_JOB_ID": "12345"})
    def test_no_db_write_on_compute_node(self, sample_cfg):
        @track_stage("test", record_to_db=True)
        def my_stage(cfg): return "ok"
        with patch("pipeline.decorators._record_stage_metrics") as mock:
            my_stage(sample_cfg)
            mock.assert_not_called()

class TestRetryOnFailure:
    def test_retries_on_oserror(self):
        call_count = 0
        @retry_on_failure(max_attempts=3, backoff_base=0.01)
        def flaky():
            nonlocal call_count; call_count += 1
            if call_count < 3: raise OSError("stale NFS")
            return "recovered"
        assert flaky() == "recovered"
        assert call_count == 3

    def test_no_retry_on_valueerror(self):
        call_count = 0
        @retry_on_failure(max_attempts=3)
        def bad():
            nonlocal call_count; call_count += 1
            raise ValueError("config error")
        with pytest.raises(ValueError): bad()
        assert call_count == 1  # No retry
```

---

## Part 4: Dask on SLURM

### 4.1 Architecture

`dask-jobqueue` `SLURMCluster` creates a scheduler on the current process and submits `sbatch` jobs for workers. Workers connect back via InfiniBand. Task distribution via `dask.delayed`.

**OSC-specific configuration:**
```python
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

cluster = SLURMCluster(
    account="PAS3209",
    queue="serial",                    # CPU-only for preprocessing
    cores=4, memory="16GB",
    walltime="01:00:00",
    processes=1,
    job_script_prologue=[
        "source ~/.bashrc",
        "conda activate gnn-experiments",
    ],
    local_directory="$TMPDIR",         # Local SSD, NOT NFS — critical
    interface="ib0",                   # InfiniBand inter-node comm
    worker_extra_args=["--lifetime", "55m", "--lifetime-stagger", "2m"],
    log_directory="slurm_logs/dask",
)

# Fixed scaling
cluster.scale(jobs=6)
# Or adaptive: cluster.adapt(minimum=0, maximum=10)

client = Client(cluster)
client.wait_for_workers(1, timeout=300)
```

**Critical:** `local_directory="$TMPDIR"` directs Dask spill files to local SSD (1-4 TB per Pitzer node). NFS cannot handle Dask's many small temp files.

### 4.2 Concrete Example: Parallel Graph Construction

```python
"""scripts/dask_preprocess.py — Parallel CAN bus graph construction for OSC."""
import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

@dask.delayed
def process_single_csv(csv_file: str, id_mapping: dict) -> list:
    """Process one CSV → list of PyG Data objects. Runs on Dask worker."""
    from src.preprocessing.preprocessing import (
        dataset_creation_streaming, create_graphs_numpy,
    )
    from config.constants import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE
    df = dataset_creation_streaming(csv_file, id_mapping=id_mapping, chunk_size=5000)
    if df.empty: return []
    return create_graphs_numpy(df, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE)

def parallel_graph_creation(dataset_name, dataset_path, cache_dir, n_workers=6):
    csv_files = find_csv_files(str(dataset_path), "train_")
    id_mapping = build_lightweight_id_mapping(csv_files)  # Fast, single-threaded

    client = create_osc_cluster(n_workers=min(n_workers, len(csv_files)))
    try:
        # Broadcast id_mapping once (avoids per-task serialization)
        id_mapping_future = client.scatter(id_mapping, broadcast=True)
        delayed_results = [process_single_csv(f, id_mapping_future) for f in csv_files]
        results = dask.compute(*delayed_results)
        all_graphs = [g for batch in results for g in batch]

        # Atomic NFS-safe cache write
        torch.save(all_graphs, tmp_cache, pickle_protocol=4)
        os.fsync(...)
        tmp_cache.rename(cache_file)
    finally:
        client.close()
```

### 4.3 Dask + PyG Compatibility

PyG `Data` objects are fully pickle-able (via `__getstate__`/`__setstate__`). PyTorch tensors have optimized serialization in Dask. `dask.delayed` is the right abstraction — maps directly to "process CSV, return list of Data objects."

Memory per worker: ~400-500 MB (CSV read + DataFrame + graph construction). Well within 16 GB. Aggregation bottleneck: all graphs must fit in scheduler memory for `torch.save()` — max 3.7 GB for largest dataset (set_02), manageable.

### 4.4 Integration with Snakemake

**Option A (hybrid, recommended if using Dask):** Dask inside a Snakemake rule. The Snakemake job becomes the Dask scheduler, submits workers, waits, writes cache. Caveat: nested SLURM submission counts against `--jobs` limit.

**Option B (simplest):** Standalone script before Snakemake. Preprocess all datasets, then Snakemake sees caches and skips preprocessing entirely.

### 4.5 Honest Assessment: Is Dask Worth It?

| Metric | Value |
|--------|-------|
| Sequential preprocessing time (all 6 datasets) | ~30-60 min |
| Dask overhead (cluster startup + serialization) | 2-6 min |
| Dask parallel time (6 workers) | ~16-21 min |
| Net speedup | 2-3x |
| Times this speedup matters | **Once** (cache exists, never re-runs unless params change) |
| New dependencies | `dask`, `dask-jobqueue`, `distributed` |
| New failure modes | scheduler crashes, worker timeouts, SLURM queue contention |

**Dask becomes worthwhile at:** 50+ datasets, frequent preprocessing parameter sweeps, datasets >50 GB each, or streaming/online preprocessing.

**Dask is overkill when:** 6 datasets with caching (current), preprocessing is a one-time paid cost, bottleneck is training not preprocessing.

### 4.6 Better Alternatives for Current Scale

| Approach | Setup | Dependencies | Best For |
|----------|-------|-------------|----------|
| **Sequential + cache (current)** | None | None | **This project today** |
| **SLURM job arrays** | 10 lines bash | None | 6-20 datasets |
| **GNU Parallel** | 5 lines bash | Pre-installed on OSC | Quick one-offs |
| **multiprocessing.Pool** | 20 lines Python | None | Single large dataset |
| **Dask SLURMCluster** | 100+ lines | 3 packages | 50+ datasets, dynamic scaling |
| **Ray on SLURM** | 100+ lines | `ray` | Mixed CPU/GPU, distributed training |

**SLURM job arrays** (zero dependencies, already used in `scripts/build_test_cache.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=PAS3209
#SBATCH --partition=serial
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

DATASETS=(hcrl_sa hcrl_ch set_01 set_02 set_03 set_04)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
python scripts/preprocess_single.py --dataset $DATASET
```

**multiprocessing.Pool** (zero dependencies, within existing code):
```python
def graph_creation_parallel(root_folder, n_workers=4, **kwargs):
    csv_files = find_csv_files(root_folder, 'train_')
    id_mapping = build_lightweight_id_mapping(csv_files)
    ctx = mp.get_context('spawn')  # CUDA-safe per project constraints
    with ctx.Pool(min(n_workers, len(csv_files))) as pool:
        results = pool.starmap(
            _process_one_csv,
            [(f, id_mapping, DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE) for f in csv_files],
        )
    return GraphDataset([g for batch in results for g in batch]), id_mapping
```

### 4.7 Filesystem Considerations on OSC

| Concern | NFS Home (`/users/`) | GPFS Scratch (`/fs/scratch/`) | Local SSD (`$TMPDIR`) |
|---------|---------------------|-------------------------------|----------------------|
| Dask temp files | **Never** | OK but crosses network | **Best** |
| Input CSVs | OK (read-only) | Better parallel throughput | Copy first if needed |
| Output cache (.pt) | OK (write once) | Better for large files | **No** — ephemeral |

---

## Part 5: Frameworks Assessment

| Framework | Data Scale | HPC/SLURM | PyTorch/PyG | Cost | Verdict |
|-----------|-----------|-----------|-------------|------|---------|
| **Spark** | TB+ | Awkward (expects own resource manager) | Poor (JVM) | Free | **Not recommended** — wrong scale, wrong ecosystem |
| **Databricks** | TB+ | None (cloud-only) | Poor | $$$+ | **Not applicable** — cannot run on OSC |
| **Dask** | GB-TB | Native (`dask-jobqueue`) | Preprocessing only | Free | **Conditional** — only if scale grows 10x |

**Why not Spark:** Data is 10 GB (not TB). Pipeline is GPU-bound, not data-bound. JVM overhead makes Spark slower than native Python at this scale. No PyG integration. SLURM expects its own resource manager.

**Why not Databricks:** Cloud-only. Cannot run on OSC. GPU support targets Spark ML, not PyTorch. Significant cost ($0.20-0.55/DBU-hour plus cloud infra). MLflow is available standalone.

---

## Part 6: Gaps & Recommendations

### Immediate Value (Tier 1)

1. **Pandera for DataFrame validation** — Schemas at the Parquet boundary. Catches silent data corruption before graph construction. Pydantic handles config; Pandera handles data.
   - Files: `pipeline/ingest.py`, `src/preprocessing/preprocessing.py`

2. **Lineage closure** — Add `code_version` (git hash) and `data_version` (DVC hash) to `runs` table. Single query: "what code + data + config produced this result?"
   - Files: `pipeline/db.py`, `pipeline/cli.py`

3. **Seed pinning audit** — Verify `torch.manual_seed()`, `np.random.seed()`, `random.seed()` set consistently. Check `torch.use_deterministic_algorithms(True)` feasibility.

### Selective Adoption (Tier 2)

4. **`@track_stage` decorator** — Apply to 4 stage functions for structured timing/memory logging. Option A integration (decorators add logging, cli.py owns DB). File: `pipeline/decorators.py`.

5. **pytest-bdd for contract tests** — Config resolution + data ingestion feature files only. Don't convert existing tests. Files: `tests/features/`, `tests/step_defs/`.

6. **Parallel preprocessing** — SLURM job arrays first (zero dependencies). `multiprocessing.Pool` second. Dask only if scale demands it.

### Not Recommended (Tier 3)

7. Spark / Databricks — wrong scale, wrong ecosystem
8. Full Repository/Unit-of-Work patterns — current db.py is right abstraction
9. Great Expectations — Pandera covers same needs with 10x less setup
10. Standalone BDD (Behave/Cucumber) — overhead without audience
