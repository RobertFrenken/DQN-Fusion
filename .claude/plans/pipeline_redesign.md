# Fresh-Eyes Architecture Review: Ideal vs Existing

**Date**: 2026-02-11
---

## Part 1: Ideal "From Scratch" Design

### The Problem
CAN bus intrusion detection via knowledge distillation. Teacher models (VGAE, GAT, DQN) are trained on full-size architectures, then compressed into lightweight students for edge deployment. Runs on HPC with SLURM. Must handle 6+ datasets, track experiments, and be reproducible.

### Design Philosophy
If starting from zero, these principles would guide every decision:
1. **Single responsibility** — each module does one thing well
2. **Config-driven** — all behavior parameterized, no magic constants
3. **Fail fast** — validate early, error messages point to root cause
4. **Reproducible** — any run can be exactly recreated from its config + data version
5. **Composable** — stages, models, losses are plug-and-play
6. **Minimal layers** — every abstraction must justify its existence

---

### A. Project Structure (Ideal)

```
kd_gat/                          # Single installable package (not split src/ + pipeline/)
  __init__.py
  models/                        # Model definitions only (no training logic)
    registry.py                  # Model registry: @register("vgae") decorator
    base.py                      # Abstract base: encode(), decode(), forward()
    vgae.py                      # GraphAutoencoderNeighborhood
    gat.py                       # GATWithJK
    dqn.py                       # EnhancedDQNFusionAgent
  data/                          # Data loading and preprocessing
    catalog.py                   # Dataset catalog (reads datasets.yaml)
    preprocessing.py             # CSV → graph construction
    caching.py                   # Graph cache management (build, validate, load)
    loaders.py                   # DataLoader factory (spawn-safe, batch size optimization)
  training/                      # Training logic
    stages.py                    # Stage definitions: autoencoder, curriculum, fusion, eval
    losses.py                    # All loss functions: recon, KD latent, KD soft-label, RL reward
    callbacks.py                 # Memory monitor, curriculum scheduler, early stopping
    distillation.py              # KD framework: TeacherWrapper, ProjectionHead, DistillationLoss
    modules.py                   # Lightning modules: VGAEModule, GATModule (thin wrappers)
  pipeline/                      # Orchestration (CLI, Snakemake, tracking)
    cli.py                       # Entry point
    config.py                    # Config dataclass + presets
    paths.py                     # Path layout
    tracking.py                  # MLflow integration
  db/                            # Data warehouse
    schema.py                    # Table definitions
    populate.py                  # Scan filesystem → insert
    query.py                     # Analytics queries
config/                          # External config files (not Python)
  presets/                       # YAML preset files per model/size
    vgae_teacher.yaml
    vgae_student.yaml
    gat_teacher.yaml
    ...
  datasets.yaml                  # Dataset catalog
data/                            # Data artifacts (DVC-tracked)
  automotive/                    # Raw CSVs
  cache/                         # Preprocessed graphs
  parquet/                       # Columnar format
  project.db                     # Results DB
experiments/                     # Run outputs (Snakemake-managed)
orchestration/                   # Snakemake + SLURM
  Snakefile
  profiles/slurm/
tests/                           # Test pyramid
  unit/                          # Model, data, loss tests
  integration/                   # Stage tests
  e2e/                           # Full pipeline tests
  conftest.py
  fixtures/                      # Shared test data generators
```

**Key difference from current**: Single `kd_gat/` package instead of split `src/` + `pipeline/`. This eliminates:
- Conditional imports (`from src.models...` inside functions)
- Unclear ownership (is preprocessing data or pipeline?)
- Separate `sys.path` hacks

### B. Configuration (Ideal)

**Choice: Frozen dataclass + JSON** (same as current — this is correct)

Why NOT Hydra/Pydantic:
- Hydra: Too complex for a fixed pipeline. Good for rapid experimentation with composition, but this project has a fixed 3-stage structure. The overhead isn't justified.
- Pydantic v2: Better validation than dataclasses, but adds a dependency for marginal benefit. The frozen dataclass + JSON approach is simpler and sufficient when you have a preset system.

**Improvement: Separate preset files**

Instead of a giant `PRESETS` dict in config.py, use YAML preset files:
```yaml
# config/presets/gat_teacher.yaml
model: gat
size: teacher
gat_hidden: 48
gat_layers: 3
gat_heads: 8
lr: 0.003
max_epochs: 300
patience: 100
```

Benefits:
- Presets are diffable, reviewable, versionable
- Adding a new model variant = adding a YAML file (no Python changes)
- Can be overridden via CLI: `--preset gat_teacher --lr 0.001`

**Improvement: Typed sub-configs**

Instead of one flat 90-field dataclass, nest by concern:
```python
@dataclass(frozen=True)
class VGAEConfig:
    hidden_dims: tuple = (480, 240, 48)
    latent_dim: int = 48
    heads: int = 4
    dropout: float = 0.15

@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 0.003
    max_epochs: int = 300
    patience: int = 100
    batch_size: int = 4096
    precision: str = "16-mixed"
    gradient_checkpointing: bool = True

@dataclass(frozen=True)
class KDConfig:
    enabled: bool = False
    teacher_path: str = ""
    temperature: float = 4.0
    alpha: float = 0.7

@dataclass(frozen=True)
class PipelineConfig:
    dataset: str = ""
    model_size: str = "teacher"
    seed: int = 42
    vgae: VGAEConfig = field(default_factory=VGAEConfig)
    gat: GATConfig = field(default_factory=GATConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    kd: KDConfig = field(default_factory=KDConfig)
```

Benefits:
- IDE autocomplete: `cfg.vgae.latent_dim` instead of `cfg.vgae_latent_dim`
- Clear ownership: VGAE params live in VGAEConfig
- Easier to validate per-section
- Still frozen, still JSON-serializable

### C. Data Pipeline (Ideal)

The current flow is solid: CSV → preprocessing → graph cache → DataLoader. Key improvements:

**1. Formal data validation with pandera or manual schema checks**
```python
# At ingestion time, validate schema before processing
expected_schema = {"timestamp": float, "arbitration_id": str, "data_field": str, "attack": int}
validate_csv_schema(csv_path, expected_schema)  # Fail fast with clear error
```

**2. Content-addressed caching**
```python
# Cache key = hash(preprocessing_version + data_files_hash + config_hash)
# If any input changes, cache is automatically invalidated
cache_key = hashlib.sha256(f"{PREPROCESS_VERSION}:{data_hash}:{window_size}:{stride}").hexdigest()[:12]
cache_path = cache_dir / f"graphs_{cache_key}.pt"
```

Current approach: `cache_metadata.json` with manual version tracking. Content-addressed is more robust.

**3. Streaming graph construction**
Current approach loads all CSVs into memory, then creates all graphs. For large datasets:
```python
def create_graphs_streaming(csv_files, window_size, stride):
    """Yield graphs one at a time — never holds all in memory."""
    for csv_file in csv_files:
        for chunk in pd.read_csv(csv_file, chunksize=100_000):
            for window in sliding_window(chunk, window_size, stride):
                yield create_graph(window)
```

### D. Model Architecture (Ideal)

**Registry pattern for extensibility**:
```python
MODEL_REGISTRY = {}

def register(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

@register("vgae")
class GraphAutoencoderNeighborhood(nn.Module): ...

@register("gat")
class GATWithJK(nn.Module): ...

# Usage
model = MODEL_REGISTRY[cfg.model_arch].from_config(cfg)
```

This is clean but may be over-engineering for 3 fixed models. The current approach (import directly by name) is pragmatic and fine for a fixed architecture.

**Teacher-Student abstraction**:
```python
class TeacherStudentPair:
    """Manages teacher loading, freezing, projection, and KD loss computation."""
    def __init__(self, student, teacher_path, model_type, config):
        self.student = student
        self.teacher = self._load_and_freeze(teacher_path, model_type)
        self.projection = self._make_projection()

    def kd_loss(self, student_output, batch):
        with torch.no_grad():
            teacher_output = self.teacher(batch)
        return self._compute_kd_loss(student_output, teacher_output)
```

### E. Knowledge Distillation (Ideal)

**Composable loss system**:
```python
class DistillationLoss:
    """Combines task loss and KD loss with configurable weighting."""
    def __init__(self, task_loss_fn, kd_loss_fn, alpha=0.7):
        self.task_loss = task_loss_fn
        self.kd_loss = kd_loss_fn
        self.alpha = alpha

    def __call__(self, student_out, teacher_out, targets):
        task = self.task_loss(student_out, targets)
        kd = self.kd_loss(student_out, teacher_out)
        return self.alpha * kd + (1 - self.alpha) * task

# VGAE KD
vgae_kd = DistillationLoss(
    task_loss_fn=VGAEReconstructionLoss(),
    kd_loss_fn=CompositeLoss([
        WeightedLoss(LatentMSE(projection), weight=0.5),
        WeightedLoss(ReconstructionMSE(), weight=0.5),
    ]),
    alpha=0.7,
)

# GAT KD
gat_kd = DistillationLoss(
    task_loss_fn=nn.CrossEntropyLoss(),
    kd_loss_fn=SoftLabelKL(temperature=4.0),
    alpha=0.7,
)
```

This separates "what losses exist" from "how they're combined" — cleaner than embedding KD logic in Lightning modules.

### F. Training Orchestration (Ideal)

**Stage as a protocol**:
```python
class Stage(Protocol):
    def run(self, cfg: PipelineConfig) -> Path | dict:
        """Execute stage, return checkpoint path or metrics."""
        ...

    def validate(self, cfg: PipelineConfig) -> None:
        """Validate prerequisites before running."""
        ...

STAGES = {
    "autoencoder": AutoencoderStage(),
    "curriculum": CurriculumStage(),
    "fusion": FusionStage(),
    "evaluation": EvaluationStage(),
}
```

Current approach: function dispatch via `STAGE_FNS` dict. Functionally equivalent, but the Protocol approach enables shared pre/post hooks (logging, checkpointing) without code duplication.

### G. Experiment Tracking (Ideal)

**The current 3-layer approach is excellent**:
1. Filesystem (Snakemake DAG triggers) — deterministic paths
2. MLflow (live tracking) — params, metrics, artifacts
3. SQLite/DuckDB (structured queries) — post-hoc analysis

This is genuinely well-designed. The separation of concerns is correct:
- Snakemake needs deterministic file paths at DAG construction time
- MLflow generates UUID-based paths (incompatible with DAG)
- SQLite provides queryable structure after runs complete

**One improvement: Unified tracking abstraction**
```python
class ExperimentTracker:
    """Single interface for all tracking backends."""
    def log_params(self, params: dict): ...
    def log_metrics(self, metrics: dict, step: int): ...
    def log_artifact(self, path: Path): ...
    def save_checkpoint(self, state_dict, path: Path): ...

class CompositeTracker(ExperimentTracker):
    """Delegates to multiple backends (MLflow + filesystem + DB)."""
    def __init__(self, backends: list[ExperimentTracker]):
        self.backends = backends
```

### H. HPC Orchestration (Ideal)

**Snakemake is the right choice** for this use case:
- File-based DAG: natural for checkpoint dependencies
- SLURM integration: first-class via profiles
- Reproducibility: rules are declarative
- Partial re-runs: only recompute what changed

Alternatives considered:
- **Nextflow**: Better container support, but Snakemake's Python integration is superior for this project
- **Prefect/Airflow**: Overkill for batch HPC, designed for long-running services
- **Plain Makefile**: Too primitive for 20 rules with wildcards
- **Metaflow**: Good for ML but weak SLURM support

**Improvement: Separate Snakefile from pipeline package**

Currently `pipeline/Snakefile` lives inside the Python package. Better:
```
orchestration/
  Snakefile           # Rules only
  rules/              # Split into focused rule files
    teachers.smk
    students.smk
    evaluation.smk
  profiles/slurm/     # SLURM config
```

This makes it clear that Snakemake orchestrates the pipeline but isn't part of it.

### I. Testing (Ideal)

**Test pyramid**:
```
tests/
  unit/
    test_vgae.py          # Model forward pass, shapes, gradient flow
    test_gat.py
    test_dqn.py
    test_preprocessing.py  # Graph construction correctness
    test_losses.py         # KD loss computation
    test_config.py         # Preset loading, serialization
  integration/
    test_autoencoder.py    # Full autoencoder stage with synthetic data
    test_curriculum.py     # Curriculum stage with VGAE dependency
    test_fusion.py         # DQN fusion with frozen VGAE + GAT
  e2e/
    test_full_pipeline.py  # All 3 stages end-to-end (slow)
  fixtures/
    synthetic.py           # Shared graph/dataset generators
    conftest.py            # Fixtures, markers
```

**Property-based testing** for preprocessing:
```python
@given(st.lists(st.integers(0, 2047), min_size=10, max_size=100))
def test_id_mapping_covers_all_ids(ids):
    mapping = build_id_mapping(ids)
    for id in ids:
        assert id in mapping or 'OOV' in mapping
```

### J. Data Warehouse (Ideal)

The current SQLite + DuckDB approach is pragmatic and correct for a serverless HPC environment. No changes needed to the strategy.

**One refinement: Schema-as-code**
```python
# db/schema.py
SCHEMA_VERSION = 3

MIGRATIONS = {
    1: "CREATE TABLE datasets ...",
    2: "CREATE TABLE runs ...; CREATE TABLE metrics ...",
    3: "ALTER TABLE runs ADD COLUMN config_json TEXT",
}

def migrate(conn):
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version in range(current + 1, SCHEMA_VERSION + 1):
        conn.executescript(MIGRATIONS[version])
        conn.execute(f"PRAGMA user_version = {version}")
```

### K. Reproducibility (Ideal)

1. **Frozen config per run** (current: good)
2. **DVC for data versioning** (current: good)
3. **Git SHA in run metadata** (improvement: tag each run with the exact code version)
4. **Conda lock file** (improvement: `conda-lock.yml` for exact environment reproduction)
5. **Container** (nice-to-have: Singularity/Apptainer image for full reproducibility on any HPC)

---

## Summary: Ideal Architecture Scorecard

| Aspect | Current Approach | Ideal Approach | Delta |
|--------|-----------------|----------------|-------|
| Package structure | Split `src/` + `pipeline/` | Single `kd_gat/` package | Medium improvement |
| Config | Flat 90-field frozen dataclass | Nested sub-configs + YAML presets | Medium improvement |
| Data pipeline | CSV→Graph with cache_metadata | Content-addressed caching | Small improvement |
| Models | Direct imports by name | Same (registry is over-engineering for 3 models) | No change needed |
| KD framework | KD logic embedded in Lightning modules | Composable DistillationLoss | Medium improvement |
| Training stages | Function dispatch dict | Same (Protocol adds little value for 4 stages) | No change needed |
| Experiment tracking | 3-layer (filesystem + MLflow + SQLite) | Same — this is genuinely well-designed | No change needed |
| HPC orchestration | Snakemake + SLURM profiles | Same + separate orchestration/ dir | Small improvement |
| Testing | Flat tests/ | Structured test pyramid | Small improvement |
| Data warehouse | SQLite + DuckDB | Same + formal migrations | Small improvement |
| Reproducibility | Frozen config + DVC | Same + conda-lock + git SHA tagging | Small improvement |

**The current system is ~80% aligned with the ideal.** The biggest gaps are:
1. Split package structure (src/ + pipeline/)
2. Flat config with 90+ fields
3. KD logic tightly coupled to Lightning modules

---

## Part 2: Comparison with architecture-consolidation.md

### High-Level Alignment: ~85%

The existing consolidation plan and the "ideal from scratch" design converge on the same core issues and reach remarkably similar conclusions. This is a good sign — it means the consolidation plan was already well-reasoned. Here's the detailed comparison:

---

### Where Both Plans Fully Agree

| Topic | Consensus |
|-------|-----------|
| **Frozen dataclasses, not Hydra/Pydantic** | Both explicitly reject Hydra/Pydantic/OmegaConf. Frozen dataclasses + JSON is the right approach. |
| **Nested per-model configs** | Nearly identical designs: VGAEConfig, GATConfig, DQNConfig, TrainingConfig. Both propose Optional model configs so saved configs only contain relevant fields. |
| **Snakemake for HPC** | Both agree it's the right orchestration tool. No alternatives needed. |
| **Defer preprocessing refactor** | Both say: keep the 740-line monolith until a second domain is added. Don't prematurely abstract. |
| **SQLite + DuckDB data warehouse** | Both agree the serverless query layer is correct. Fix the population mechanism, not the architecture. |
| **Don't over-engineer for 3 models** | Both acknowledge that a full registry pattern is borderline over-engineering for the current 3-model setup, but may be justified by future goals. |

---

### Where the Plans Differ

#### 1. Package Structure: `src/` + `pipeline/` Merge

| | Ideal Plan | Consolidation Plan |
|--|-----------|-------------------|
| **Proposes** | Merge into single `kd_gat/` package | Does not address src/pipeline split |
| **Rationale** | Eliminates conditional imports, unclear ownership, sys.path issues | Focuses on config/registry/fusion as the primary scaling problems |

**Analysis**: The consolidation plan is pragmatically correct to skip this. Merging `src/` + `pipeline/` is a large mechanical refactor with moderate benefit. The conditional imports work. The bigger wins come from config restructuring and the model registry. However, if you're doing a large refactor anyway (per-model configs touch every file), doing the merge at the same time reduces total churn.

**Verdict**: The ideal plan is technically cleaner, but the consolidation plan's omission is a reasonable scope decision. If Phase 2 is a big refactor, bundle the merge. If it's incremental, skip it.

#### 2. KD Framework: Composable Losses vs Embedded Logic

| | Ideal Plan | Consolidation Plan |
|--|-----------|-------------------|
| **Proposes** | Composable `DistillationLoss` class — separates task loss, KD loss, and weighting | Does not address KD specifically |
| **Focus** | KD loss composition | Fusion feature extraction (2B) |

**Analysis**: The consolidation plan focuses on the DQN fusion state vector as the biggest scalability wall (adding model N changes the state vector). The ideal plan focuses on the loss function side (embedding KD logic in Lightning modules makes them harder to extend). Both are real problems, but they're orthogonal — you need both.

**Verdict**: The consolidation plan's 2B (FusionFeatureExtractor protocol) is higher priority than composable losses. The KD loss refactor is a nice-to-have that pays off when adding a 4th model type with a novel distillation strategy.

#### 3. Model Registry Scope

| | Ideal Plan | Consolidation Plan |
|--|-----------|-------------------|
| **Proposes** | Simple decorator-based registry (`@register("vgae")`) with caveat it may be over-engineering | Rich registry with `ModelEntry(config_cls, model_cls, stage, fusion_extractor)` + concrete list of 9 files that need touching today |
| **Depth** | Surface-level pattern | Deep analysis of what "add a model" really costs |

**Analysis**: The consolidation plan is significantly more thorough here. It maps out the exact pain — adding a model today means touching 9 files — and proposes a registry that eliminates most of those touchpoints. The ideal plan's decorator registry doesn't solve the real problem (which is the scattered magic strings and implicit model mappings, not the model class lookup itself).

**Verdict**: Consolidation plan wins. Its `ModelEntry` approach centralizes stage mapping, config type, and fusion extraction — not just the class lookup.

#### 4. Tracking Architecture: 3 → 2 Systems

| | Ideal Plan | Consolidation Plan |
|--|-----------|-------------------|
| **Assessment** | "The 3-layer approach is genuinely well-designed" | "Three overlapping systems... populate() reverse-engineers metadata from directory names. This is fragile." |
| **Proposes** | Optional CompositeTracker abstraction | Three concrete options (write-through, drop DB, drop MLflow), recommends write-through |

**Analysis**: The consolidation plan is more honest here. The 3-layer architecture is conceptually sound, but the _population mechanism_ (scanning directory names and parsing `_kd` suffixes) is genuinely fragile. The ideal plan glosses over this because it evaluates the architecture in theory, not the implementation. The consolidation plan's Option A (write-through from cli.py) is a clean fix that keeps all 3 layers but eliminates the fragile scraping.

**Verdict**: Consolidation plan wins. Write-through is the right fix. The CompositeTracker abstraction from the ideal plan is unnecessary indirection for a system that already works.

#### 5. Preset Files: YAML vs Python Dicts

| | Ideal Plan | Consolidation Plan |
|--|-----------|-------------------|
| **Proposes** | External YAML preset files (`config/presets/gat_teacher.yaml`) | Per-model preset dicts in Python (`VGAE_PRESETS = {"teacher": VGAEConfig(...)}`) |

**Analysis**: Both achieve the same goal (presets per model/size). YAML files are more reviewable and diffable. Python dicts are simpler (no file I/O, no parsing, no validation of YAML structure). For a research project with a small team, Python dicts are fine. YAML files add value when non-developers need to create presets or when you have dozens of variants.

**Verdict**: Slight preference for the consolidation plan's Python dicts. YAML presets add file I/O complexity for marginal benefit at current scale. Revisit if preset count grows beyond ~10.

#### 6. Dynamic Fusion (N-Model DQN)

| | Ideal Plan | Consolidation Plan |
|--|-----------|-------------------|
| **Coverage** | Not addressed | Detailed `FusionFeatureExtractor` protocol with concrete state vector analysis |

**Analysis**: The consolidation plan identifies this as the biggest scalability wall and proposes a clean solution. The ideal plan omits it entirely — a genuine blind spot. The current 15-D hardcoded state vector is the single biggest obstacle to adding new model types.

**Verdict**: Consolidation plan fills a critical gap that the ideal plan missed.

---

### Synthesis: What the Combined "Best of Both" Plan Looks Like

**Priority order** (combining both plans):

1. **Per-model configs** (Both agree, nearly identical design)
   - Nested frozen dataclasses: VGAEConfig, GATConfig, DQNConfig, TrainingConfig, KDConfig
   - RunConfig wrapper with `Optional[ModelConfig]` per model
   - Python-dict presets (not YAML files)
   - Migration support for existing flat config.json files

2. **Dynamic fusion + Model registry** (Consolidation plan, enhanced)
   - `FusionFeatureExtractor` protocol on each model
   - `ModelEntry` registry centralizing config_cls, model_cls, stage, fusion_extractor
   - DQN adapts to dynamic `state_dim = sum(fusion_dims)`
   - Single registration point for new models

3. **Write-through DB** (Consolidation plan)
   - `cli.py` writes directly to project DB after each run
   - Eliminates fragile `populate()` directory scanning
   - Keep MLflow for live tracking, filesystem for Snakemake DAG

4. **KD loss composition** (Ideal plan, lower priority)
   - Extract KD loss computation from Lightning modules into composable `DistillationLoss`
   - Only do this when adding a 4th model type with a novel KD strategy
   - Current embedded approach works fine for VGAE + GAT

5. **Package structure merge** (Ideal plan, opportunistic)
   - If doing a large refactor (items 1-2), consider merging `src/` + `pipeline/` into `kd_gat/`
   - If incremental, skip — the conditional imports work

6. **Reproducibility enhancements** (Ideal plan, low effort)
   - Git SHA in run metadata (tag during `start_run()`)
   - Conda lock file for environment pinning
   - Content-addressed cache keys (when touching caching code)

---

### What Both Plans Correctly Avoid

- Hydra / OmegaConf / Pydantic
- Dynamic Snakefile generation
- Premature preprocessing abstraction
- Merging db.py and analytics.py
- Over-engineering for hypothetical future requirements

### Final Assessment

The consolidation plan is **remarkably well-targeted**. It identified the exact pain points (flat config, hardcoded fusion state, fragile DB population, scattered model mappings) and proposed focused fixes. The ideal plan adds value in two areas: composable KD losses and the package structure merge. But the consolidation plan's unique contributions — dynamic fusion, concrete model registry, and write-through DB — are higher priority and more impactful.

**Alignment score: ~85%.** The 15% gap is mostly the ideal plan's additions (KD composition, package merge, reproducibility polish) rather than disagreements. There are no fundamental conflicts between the two plans.

---

## Part 3: Actionable Implementation Plan

### Key Design Decision: Incremental, Not Big-Bang

Rather than replacing `PipelineConfig`, we **extend it** with computed properties that return typed sub-configs. This means:
- All 77 flat fields stay (no breaking change to any consumer)
- New `cfg.vgae`, `cfg.gat`, etc. properties return typed sub-config objects
- Existing JSON serialization unchanged (flat dict)
- CLI unchanged, Snakefile unchanged
- Consumers migrate incrementally from `cfg.vgae_latent_dim` → `cfg.vgae.latent_dim`

Package merge (`src/` + `pipeline/` → `kd_gat/`) is **deferred** — the conditional imports work, and the merge would touch 30+ import sites for cosmetic benefit.

---

### Phase 1: Per-Model Sub-Configs (LOW RISK) — COMPLETED 2026-02-12

**Goal**: Add typed sub-config views without breaking anything.

**Files modified**:
| File | Change |
|---|---|
| `pipeline/config.py` | Add 7 frozen dataclasses (VGAEConfig, GATConfig, DQNConfig, KDConfig, CurriculumConfig, FusionConfig, TrainingConfig) + 7 `@property` methods on PipelineConfig |
| `pipeline/stages/modules.py` | Migrate VGAEModule/GATModule to use `cfg.vgae`, `cfg.gat`, `cfg.kd` |
| `pipeline/stages/utils.py` | Migrate `load_teacher()`, `load_vgae()`, `load_gat()` to use sub-configs |
| `pipeline/stages/training.py` | Migrate model construction to use `cfg.vgae`, `cfg.gat` |
| `pipeline/stages/fusion.py` | Use `cfg.dqn`, `cfg.fusion` |
| `tests/test_pipeline_integration.py` | Add tests: sub-config properties match flat fields, presets produce correct sub-configs |

**No changes to**: cli.py, Snakefile, validate.py, conftest.py, existing test expectations.

**Migration for existing config.json**: None needed — flat JSON still loads into flat fields, properties compute sub-configs on the fly.

**Gate**: All 24 existing tests pass + `snakemake -n` succeeds.

---

### Phase 3: Model Registry + Dynamic Fusion (MEDIUM-HIGH RISK)

**Goal**: Centralize model registration. Make fusion state vector adapt to N models.

**New files**:
| File | Purpose |
|---|---|
| `src/models/fusion_features.py` | `FusionFeatureExtractor` protocol + `VGAEFusionExtractor` (8-D) + `GATFusionExtractor` (7-D) |
| `src/models/registry.py` | `ModelEntry` dataclass + registration functions + default registrations for vgae/gat/dqn |

**Modified files**:
| File | Change |
|---|---|
| `src/models/vgae.py` | Add `from_config(cls, cfg: VGAEConfig)` classmethod |
| `src/models/gat.py` | Add `from_config(cls, cfg: GATConfig)` classmethod |
| `src/models/dqn.py` | Add `from_config(cls, cfg: DQNConfig)` on QNetwork; accept dynamic `state_dim` |
| `pipeline/stages/utils.py` | Rewrite `cache_predictions()` to use extractors. Replace `load_teacher()`/`load_vgae()`/`load_gat()` with generic `load_model()` |
| `pipeline/stages/fusion.py` | Use registry to discover extractors, compute dynamic `state_dim` |
| `pipeline/stages/evaluation.py` | Use registry+extractors for fusion evaluation |
| `pipeline/stages/training.py` | Migrate `load_teacher()` calls to `load_model()` |

**Backward compatibility**: Existing DQN checkpoints (`state_dim=15`) stay valid because extractors produce the same 8+7=15 features in the same order.

**Safety**: Keep legacy `cache_predictions` as `_cache_predictions_legacy` during transition. Run both in tests, compare outputs for numerical equivalence.

**Gate**: All tests pass + run `hcrl_sa` teachers+students through full pipeline on GPU.

---

### Phase 4: Write-Through DB (LOW RISK) — COMPLETED 2026-02-12

**Goal**: `cli.py` writes to project DB directly instead of post-hoc `populate()` scanning.

**Modified files**:
| File | Change |
|---|---|
| `pipeline/db.py` | Add `record_run_start()`, `record_run_end()`, `_insert_metrics()` (~60 lines) |
| `pipeline/cli.py` | Add DB write calls before/after stage dispatch (~15 lines) |

**Keep `populate()` as backfill** for pre-existing runs and DB recovery. Just no longer the primary data path.

**Gate**: Tests pass + verify DB has correct entries after a stage run.

---

### Execution Order

```
Phase 1: Per-Model Configs ──→ Phase 3: Registry + Dynamic Fusion
                              ↗
Phase 4: Write-Through DB ────  (independent, can run in parallel with Phase 3)
```

### Total Scope

| Phase | New Lines | Modified Lines | New Files | Risk |
|---|---|---|---|---|
| Phase 1 | ~210 | ~100 | 0 | LOW |
| Phase 3 | ~300 | ~200 | 2 | MEDIUM-HIGH |
| Phase 4 | ~75 | ~20 | 0 | LOW |
| **Total** | **~585** | **~320** | **2** | |

### What's Explicitly Deferred

- **Package merge** (src/ + pipeline/) — cosmetic, high churn, low payoff
- **KD loss composition** — do when adding a 4th model with novel distillation
- **Preprocessing refactor** — do when adding a non-automotive domain
- **YAML preset files** — Python dicts are sufficient at current scale
- **Conda lock / containers** — nice-to-have reproducibility polish

### Verification Plan

After each phase:
1. `python -m pytest tests/ -v` — all tests pass
2. `snakemake -s pipeline/Snakefile -n` — dry run succeeds
3. Single dataset GPU run: `snakemake --config 'datasets=["hcrl_sa"]' --profile profiles/slurm`
4. DB integrity: `python -m pipeline.db populate && python -m pipeline.db summary`
