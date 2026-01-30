# Codebase Refinement Analysis

**Date**: 2026-01-30
**Goal**: Identify essential, edge-case, and obsolete files before extensive training runs.

---

## Executive Summary

Two parallel systems exist in this repo:

| System | Entry Point | Config | Status |
|--------|-------------|--------|--------|
| **`pipeline/`** | `python -m pipeline.cli` | `PipelineConfig` (frozen dataclass) | **Active** -- used by Snakemake |
| **`src/` + legacy** | `train_with_hydra_zen.py`, `can-train` | `CANGraphConfig` (Hydra-Zen, 25+ dataclasses) | **Obsolete** |

`pipeline/stages.py` imports only 5 modules from `src/`:

```
src.training.datamodules    -> load_dataset(), CANGraphDataModule
src.models.vgae             -> GraphAutoencoderNeighborhood
src.models.models           -> GATWithJK
src.models.dqn              -> QNetwork, EnhancedDQNFusionAgent
src.preprocessing.preprocessing -> graph_creation()
```

Everything else in `src/` is dead code relative to the active pipeline.

**Impact**: ~90 files -> ~30 files (60 obsolete files identified).

---

## Part 1: File Categorization

### ESSENTIAL (required for pipeline execution)

| File | Role |
|------|------|
| `pipeline/__init__.py` | Package init |
| `pipeline/config.py` | `PipelineConfig` frozen dataclass + `PRESETS` dict |
| `pipeline/paths.py` | Path derivation: `stage_dir`, `checkpoint_path`, `config_path`, `log_dir`, `data_dir`, `cache_dir` |
| `pipeline/validate.py` | Pre-flight config + stage validation |
| `pipeline/stages.py` | All training logic: VGAE, GAT curriculum, GAT normal, DQN fusion, evaluation |
| `pipeline/cli.py` | CLI entry point, parses args, dispatches to `STAGE_FNS` |
| `pipeline/Snakefile` | SLURM workflow DAG (6 datasets x 3 tiers x 3 stages + eval) |
| `src/models/models.py` | `GATWithJK` -- Graph Attention Network with Jumping Knowledge |
| `src/models/vgae.py` | `GraphAutoencoderNeighborhood` -- VGAE with progressive compression |
| `src/models/dqn.py` | `QNetwork`, `EnhancedDQNFusionAgent` -- DQN fusion agent |
| `src/preprocessing/preprocessing.py` | `graph_creation()`, `load_dataset()`, `GraphDataset`, CAN ID mapping |
| `src/training/datamodules.py` | `load_dataset()` wrapper (called by `_load_data()`), `CANGraphDataModule` (used by `_auto_tune_batch_size()`) |
| `src/paths.py` | `PathResolver` -- transitive dependency of `load_dataset()` (~1200 lines, only ~100 actually needed) |
| `src/__init__.py` | Package init (exports model classes) |
| `src/models/__init__.py` | Package init |
| `src/preprocessing/__init__.py` | Package init |
| `src/training/__init__.py` | Package init |
| `config/config.yaml` | SLURM profile for Snakemake (account, partition, resources) |
| `config/snakemake_config.yaml` | Pipeline-level Snakemake config (datasets, modalities, sizes) |
| `config/slurm-status.py` | Snakemake SLURM job status checker |
| `pyproject.toml` | Project metadata + dependencies |
| `requirements.txt` | Dependency manifest |

### EDGE CASE / FUTURE INTEGRATION (not needed for training, useful for analysis/publication)

| File | Role | Notes |
|------|------|-------|
| `src/visualizations/model_figures.py` | Model architecture & inference viz | Recently patched with clone fix; useful for paper |
| `src/visualizations/embedding_umap.py` | UMAP embedding plots | Paper figures |
| `src/visualizations/metrics_figures.py` | Metrics visualization | Paper figures |
| `src/visualizations/performance_comparison.py` | Model comparison plots | Paper figures |
| `src/visualizations/utils.py` | Viz helpers | Supports above |
| `src/visualizations/data_loader.py` | Data loading for viz | Supports above |
| `src/visualizations/model_loader.py` | Model loading for viz | Supports above |
| `src/visualizations/__init__.py` | Package init | |
| `src/evaluation/evaluation.py` | Comprehensive eval pipeline | Parallel to `pipeline/stages.py:evaluate()` -- richer output formats |
| `src/evaluation/metrics.py` | `compute_all_metrics`, threshold detection | Could replace inline sklearn calls in `stages.py:_compute_metrics()` |
| `src/evaluation/ablation.py` | Ablation study analysis | Post-training analysis |
| `src/evaluation/__init__.py` | Package init | |
| `src/config/plotting_config.py` | Publication matplotlib settings | Paper figures |
| `src/utils/plotting_utils.py` | Publication-quality plots | Paper figures |
| `src/utils/seeding.py` | `set_global_seeds()` | Small utility; pipeline seeds via Lightning |
| `config/paper_style.mplstyle` | Matplotlib style file | Paper figures |
| `config/plotting_config.py` | Root copy of plotting config | Duplicate of `src/config/plotting_config.py` |
| `params.csv` | Model parameter reference table | Documentation/paper |
| `docs/` | All documentation files | Reference only |

### REDUNDANT / OBSOLETE (fully replaced by pipeline/ system)

| File | Replaced By | Notes |
|------|-------------|-------|
| `src/training/trainer.py` | `pipeline/stages.py` | `HydraZenTrainer` orchestrator -- entire file obsolete |
| `src/training/lightning_modules.py` | `pipeline/stages.py` | `VAELightningModule`, `GATLightningModule`, etc. -- replaced by inline `VGAEModule`, `GATModule` |
| `src/training/modes/curriculum.py` | `pipeline/stages.py:train_curriculum()` | `CurriculumTrainer` class -- fully replaced |
| `src/training/modes/fusion.py` | `pipeline/stages.py:train_fusion()` | `FusionTrainer` class -- fully replaced |
| `src/training/modes/__init__.py` | N/A | Only exports obsolete classes |
| `src/training/prediction_cache.py` | `pipeline/stages.py:_cache_predictions()` | `FusionDataExtractor`, `PredictionCacheBuilder` -- replaced by inline function |
| `src/training/batch_optimizer.py` | `pipeline/stages.py:_auto_tune_batch_size()` | `BatchSizeOptimizer` -- replaced |
| `src/training/knowledge_distillation.py` | `pipeline/stages.py` VGAEModule/GATModule | `KDHelper` class -- KD logic now inline in Lightning modules |
| `src/training/adaptive_batch_size.py` | `pipeline/config.py:safety_factor` | `SafetyFactorDatabase` -- not used by pipeline |
| `src/training/memory_monitor_callback.py` | Lightning built-in monitoring | Not used by pipeline |
| `src/training/memory_preserving_curriculum.py` | `pipeline/stages.py:CurriculumDataModule` | Not used by pipeline |
| `src/training/momentum_curriculum.py` | `pipeline/stages.py:_curriculum_sample()` | Not used by pipeline |
| `src/training/model_manager.py` | `pipeline/stages.py` (direct model creation) | Not used by pipeline |
| `src/config/hydra_zen_configs.py` | `pipeline/config.py` | Entire Hydra-Zen config schema (25+ dataclasses) -- replaced by single `PipelineConfig` |
| `src/config/frozen_config.py` | `PipelineConfig.save()/load()` | JSON serialization -- replaced by dataclass methods |
| `src/config/__init__.py` | N/A | Package init for old config |
| `src/paths.py` | `pipeline/paths.py` | `PathResolver` (~1200 lines) -- must refactor `load_dataset()` before deleting (see Action Plan Phase 2b) |
| `src/cli/main.py` | `pipeline/cli.py` | Old CLI entry point with subcommands |
| `src/cli/executor.py` | `pipeline/cli.py` + Snakemake | `ExecutionRouter` -- SLURM routing replaced by Snakemake |
| `src/cli/validator.py` | `pipeline/validate.py` | Old validation framework |
| `src/cli/pydantic_validators.py` | `pipeline/validate.py` | Pydantic model for CLI validation |
| `src/cli/environment.py` | Snakemake SLURM profile | Environment detection -- Snakemake handles this |
| `src/cli/__init__.py` | N/A | Package init for old CLI |
| `train_with_hydra_zen.py` | `pipeline/cli.py` | Legacy entry point (two duplicate `main()` defs, broken non-frozen path) |
| `can-train` | `python -m pipeline.cli` | Shell wrapper for old CLI |
| `Snakefile` (root) | `pipeline/Snakefile` | Root Snakefile calls old `train_with_hydra_zen.py` |
| `config/__main__.py` | N/A | Config management CLI using old config system |
| `config/frozen_config.py` (root) | `PipelineConfig.save()/load()` | Root-level copy of `src/config/frozen_config.py` |
| `config/hydra_zen_configs.py` (root) | `pipeline/config.py` | Root-level copy of `src/config/hydra_zen_configs.py` |
| `config/envs/` | `requirements.txt` + conda env on HPC | Environment specs |
| `Justfile` | Snakemake commands directly | Task runner referencing old scripts |
| `src/scripts/` (entire directory) | Various / unused | Analysis scripts from old workflow |
| `src/utils/analyze_gpu_monitor.py` | Not used | GPU monitoring analysis |
| `src/utils/get_teacher_jobs.py` | Snakemake | SLURM job utilities |
| `src/utils/parse_squeue.py` | Snakemake | SLURM queue parsing |
| `src/utils/cache_manager.py` | `pipeline/stages.py` caching inline | Cache utilities |
| `src/utils/lightning_gpu_utils.py` | Lightning built-ins | GPU memory utilities |
| `src/utils/dependency_manifest.py` | `pipeline/validate.py` | Dependency tracking |
| `src/visualizations/demo_visualization.py` | N/A | Demo pipeline -- not needed |
| `pytest.ini` | Tests not in active use | Test config (tests/ dir deleted per git status) |

---

## Part 2: Function-Level Analysis

### Essential Files

#### `pipeline/stages.py` -- All functions essential
Every function is actively called by the training pipeline dispatch table (`STAGE_FNS`). No dead code.

**Key functions**: `_load_data()`, `_effective_batch_size()`, `_auto_tune_batch_size()`, `_load_teacher()`, `_make_projection()`, `_make_trainer()`, `_score_difficulty()`, `_curriculum_sample()`, `_cache_predictions()`, `_compute_metrics()`, `_run_gat_inference()`, `_run_vgae_inference()`, `_run_fusion_inference()`, `_vgae_threshold()`

**Key classes**: `VGAEModule`, `GATModule`, `CurriculumDataModule`, `_NS`

**Training functions**: `train_autoencoder()`, `train_curriculum()`, `train_normal()`, `train_fusion()`, `evaluate()`

#### `src/models/models.py` -- GATWithJK

| Element | Status | Notes |
|---------|--------|-------|
| `GATWithJK.__init__()` | Essential | Full constructor used by pipeline |
| `GATWithJK.forward(data, return_intermediate)` | Essential | `return_intermediate=True` used by fusion stage for embedding extraction |
| Gradient checkpointing support | Marginal | `use_checkpointing` param exists but defaults to `False` |

#### `src/models/vgae.py` -- GraphAutoencoderNeighborhood

| Element | Status | Notes |
|---------|--------|-------|
| `__init__()`, `forward()` | Essential | Used by autoencoder and curriculum stages |
| `sample_z()`, `decode()` | Essential | VAE sampling used in forward pass |
| `create_neighborhood_targets()` | Essential | Edge prediction targets |
| Gradient checkpointing support | Marginal | Defaults to off |

#### `src/models/dqn.py`

| Element | Status | Notes |
|---------|--------|-------|
| `QNetwork` | Essential | Q-network for fusion agent |
| `EnhancedDQNFusionAgent.__init__()` | Essential | Agent construction |
| `select_action()` | Essential | Epsilon-greedy policy |
| `normalize_state()` | Essential | 15D state normalization |
| `compute_fusion_reward()` | Essential | Reward function |
| `update()`, `store_transition()` | Essential | Training loop |

#### `src/preprocessing/preprocessing.py`

| Element | Status | Notes |
|---------|--------|-------|
| `graph_creation()` | Essential | Core graph construction from CAN data |
| `load_dataset()` | Essential | Full preprocessing pipeline with caching |
| `GraphDataset` | Essential | PyG Dataset subclass |
| `safe_hex_to_int()` | Essential | Hex parsing for CAN data |
| `normalize_payload_bytes()` | Essential | Feature normalization |
| `build_complete_id_mapping_streaming()` | Essential | CAN ID enumeration |
| `build_id_mapping_from_normal()` | Marginal | Alternative ID mapping -- may not be called by pipeline |

#### `src/training/datamodules.py`

| Element | Status | Notes |
|---------|--------|-------|
| `load_dataset()` | Essential | Called by `pipeline/stages.py:_load_data()`. Depends on `PathResolver` from `src/paths.py` |
| `DEFAULT_MP_CONTEXT` | Essential | Module constant for spawn safety |
| `CANGraphDataModule` | Essential | Used by `_auto_tune_batch_size()` for Lightning Tuner |
| `EnhancedCANGraphDataModule` | **Obsolete** | Not used by pipeline |
| `AdaptiveGraphDataset` | **Obsolete** | Not used by pipeline |
| `CurriculumCallback` | **Obsolete** | Not used by pipeline |
| `create_dataloaders()` | **Obsolete** | Not used by pipeline |

#### `src/paths.py` (transitive dependency)

| Element | Status | Notes |
|---------|--------|-------|
| `PathResolver.__init__()` | Essential (temporarily) | Called by `load_dataset()` with legacy `_NS` config object |
| `PathResolver.resolve_dataset_path()` | Essential (temporarily) | Finds dataset on disk |
| `PathResolver.get_cache_paths()` | Essential (temporarily) | Returns cache file paths |
| All other methods (~30+) | **Obsolete** | `discover_model()`, `get_experiment_dir()`, `get_checkpoint_dir()`, etc. -- replaced by `pipeline/paths.py` |

### Edge Case Files

#### `src/evaluation/evaluation.py`

| Element | Status | Notes |
|---------|--------|-------|
| `EvaluationConfig` | Edge case | Richer than pipeline's inline evaluation |
| Eval functions | Edge case | More output formats (CSV, LaTeX) than pipeline |

#### `src/evaluation/metrics.py`

| Element | Status | Notes |
|---------|--------|-------|
| `compute_all_metrics()` | Edge case | Could replace inline sklearn calls in `stages.py:_compute_metrics()` |
| `detect_optimal_threshold()` | Edge case | Similar to `stages.py:_vgae_threshold()` |

#### `src/visualizations/model_figures.py`

| Element | Status | Notes |
|---------|--------|-------|
| `_infer_gat()`, `_infer_vgae()`, etc. | Edge case | Already patched with `.clone().to(device)` fix; useful for paper figures |

---

## Part 3: Dependency Map

```
pipeline/cli.py (entry point)
  |-- pipeline/config.py    (PipelineConfig, PRESETS)
  |-- pipeline/paths.py     (STAGES, config_path)
  |-- pipeline/validate.py  (validate)
  '-- pipeline/stages.py    (STAGE_FNS)  <-- deferred import
        |-- src/training/datamodules.py  (load_dataset, CANGraphDataModule)
        |     |-- src/preprocessing/preprocessing.py  (GraphDataset)
        |     '-- src/paths.py  (PathResolver)  <-- 1200-line transitive dep
        |-- src/models/vgae.py  (GraphAutoencoderNeighborhood)
        |-- src/models/models.py  (GATWithJK)
        |-- src/models/dqn.py  (QNetwork, EnhancedDQNFusionAgent)
        '-- src/preprocessing/preprocessing.py  (graph_creation)

pipeline/Snakefile (workflow DAG)
  '-- python -m pipeline.cli <stage> [args]
```

**Key finding**: `load_dataset()` in `src/training/datamodules.py` depends on `PathResolver` from `src/paths.py`. But `pipeline/stages.py:_load_data()` already computes paths via `pipeline/paths.py`, then wraps them in a legacy `_NS` adapter just so `PathResolver` can extract them again. This circular redundancy is the main obstacle to deleting `src/paths.py`.

---

## Part 4: Suggested Action Plan

### Phase 1: Delete clearly obsolete files (safe -- no pipeline dependency)

```
# Old training orchestration (replaced by pipeline/stages.py)
src/training/trainer.py
src/training/lightning_modules.py
src/training/modes/                 # entire directory
src/training/prediction_cache.py
src/training/batch_optimizer.py
src/training/knowledge_distillation.py
src/training/adaptive_batch_size.py
src/training/memory_monitor_callback.py
src/training/memory_preserving_curriculum.py
src/training/momentum_curriculum.py
src/training/model_manager.py

# Old CLI (replaced by pipeline/cli.py)
src/cli/                            # entire directory

# Old config system (replaced by pipeline/config.py)
src/config/hydra_zen_configs.py
src/config/frozen_config.py
src/config/__init__.py

# Old entry points
train_with_hydra_zen.py
can-train
Snakefile                           # root-level only (NOT pipeline/Snakefile)

# Root config duplicates
config/__main__.py
config/frozen_config.py
config/hydra_zen_configs.py
config/envs/

# Old task runner
Justfile

# Analysis scripts (not used by pipeline)
src/scripts/                        # entire directory

# SLURM utilities (replaced by Snakemake)
src/utils/analyze_gpu_monitor.py
src/utils/get_teacher_jobs.py
src/utils/parse_squeue.py
src/utils/cache_manager.py
src/utils/lightning_gpu_utils.py
src/utils/dependency_manifest.py

# Dead test infrastructure
pytest.ini
```

**Verify after**: `python -c "from pipeline.stages import STAGE_FNS; print(list(STAGE_FNS.keys()))"`

### Phase 2: Trim `src/training/datamodules.py`

Remove unused classes/functions (400+ lines -> ~150 lines):
- **Remove**: `EnhancedCANGraphDataModule`, `AdaptiveGraphDataset`, `CurriculumCallback`, `create_dataloaders()`
- **Keep**: `load_dataset()` + helpers, `CANGraphDataModule`, `DEFAULT_MP_CONTEXT`

### Phase 2b: Refactor `load_dataset()` to eliminate `src/paths.py` dependency

Current flow (circular redundancy):
```
pipeline/stages.py:_load_data()
  -> computes data_dir/cache_dir via pipeline/paths.py
  -> wraps in legacy _NS object
  -> calls datamodules.load_dataset()
    -> creates PathResolver from _NS  (src/paths.py, 1200 lines)
    -> PathResolver extracts the same paths back out
```

**Fix**: Change `load_dataset()` to accept `dataset_path` and `cache_dir` as direct arguments. Update `_load_data()` to pass paths from `pipeline/paths.py`. Delete `_NS` class and `src/paths.py`.

### Phase 3: Quarantine visualization & evaluation (defer to post-training)

Leave as-is but do not delete:
- `src/visualizations/` -- paper figures
- `src/evaluation/` -- richer evaluation
- `src/config/plotting_config.py` -- supports visualizations
- `src/utils/plotting_utils.py` -- supports visualizations
- `src/utils/seeding.py` -- small utility

### Phase 4: Clean root `config/` directory

- **Keep**: `config.yaml`, `snakemake_config.yaml`, `slurm-status.py`, `paper_style.mplstyle`, `plotting_config.py`
- **Remove**: everything else

### Phase 5: Update `pyproject.toml`

Remove the obsolete `can-train` entry point:
```toml
# Remove:
can-train = "src.cli.main:main"
```

---

## Final State After Cleanup

```
KD-GAT/
├── pipeline/                    # Active training system (untouched)
│   ├── __init__.py, config.py, paths.py, validate.py
│   ├── stages.py, cli.py, Snakefile
├── src/
│   ├── __init__.py
│   ├── models/                  # Model architectures (essential)
│   │   ├── models.py, vgae.py, dqn.py
│   ├── preprocessing/           # Data pipeline (essential)
│   │   └── preprocessing.py
│   ├── training/                # Trimmed
│   │   └── datamodules.py      # load_dataset() + CANGraphDataModule only
│   ├── evaluation/              # Quarantined (paper)
│   ├── visualizations/          # Quarantined (paper)
│   ├── config/
│   │   └── plotting_config.py
│   └── utils/
│       ├── seeding.py, plotting_utils.py
├── config/                      # Snakemake infrastructure
│   ├── config.yaml, snakemake_config.yaml, slurm-status.py
│   └── paper_style.mplstyle, plotting_config.py
├── docs/                        # Historical reference
├── pyproject.toml, requirements.txt, params.csv, notes.md
```

**~90 files -> ~30 files** (60 obsolete files removed)

---

## Risk Notes

- Do NOT delete `src/paths.py` before Phase 2b (refactoring `load_dataset()` away from `PathResolver`)
- `pyproject.toml` references `src.cli.main:main` -- update after deleting `src/cli/`
- `src/__init__.py` exports model classes -- verify imports resolve after removing subpackages
- `docs/` references the old system -- leave as historical reference

## Verification

After each phase:
```bash
python -c "from pipeline.stages import STAGE_FNS; print('OK:', list(STAGE_FNS.keys()))"
python -m pipeline.cli --help
snakemake -n -s pipeline/Snakefile --configfile config/snakemake_config.yaml
```
