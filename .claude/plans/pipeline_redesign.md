# Pipeline Architecture: Current State and Remaining Work

**Date**: 2026-02-13

---

## Completed

### Phase 1: Config System (2026-02-12)

Replaced flat 90-field frozen dataclass with Pydantic v2 frozen models + YAML composition:
- `config/schema.py` — `PipelineConfig`, `VGAEArchitecture`, `GATArchitecture`, `DQNArchitecture`, `AuxiliaryConfig`, `TrainingConfig`, `FusionConfig`, `PreprocessingConfig`
- `config/resolver.py` — `resolve(model_type, scale, auxiliaries, **overrides)` → frozen config
- `config/paths.py` — Path layout: `{dataset}/{model_type}_{scale}_{stage}[_{aux}]`
- YAML files: `defaults.yaml`, `models/{vgae,gat,dqn}/{large,small}.yaml`, `auxiliaries/{none,kd_standard}.yaml`

### Phase 4: Write-Through DB (2026-02-12)

`cli.py` writes directly to `data/project.db` via `record_run_start()`/`record_run_end()`. `populate()` is kept as backfill/recovery only.

### Snakemake Features (2026-02-13)

- Retries with resource scaling (`mem_mb=lambda wc, attempt: 128000 * attempt`, `retries: 2`)
- Between-workflow caching (`preprocess` rule with `cache: True`)
- Group jobs for evaluation rules
- Benchmarks on all training + eval rules

### GAT Size Ratio Fix (2026-02-13)

Large GAT: `fc_layers: 1` (343k params). Small GAT: `fc_layers: 2` (65k params). Teacher/student ratio: 5.3x (was 16.3x).

---

## Remaining: Phase 3 — Model Registry + Dynamic Fusion

**Goal**: Centralize model registration. Make fusion state vector adapt to N models.

### New files

| File | Purpose |
|---|---|
| `src/models/fusion_features.py` | `FusionFeatureExtractor` protocol + `VGAEFusionExtractor` (8-D) + `GATFusionExtractor` (7-D) |
| `src/models/registry.py` | `ModelEntry` dataclass + registration functions + default registrations for vgae/gat/dqn |

### Modified files

| File | Change |
|---|---|
| `src/models/vgae.py` | Add `from_config(cls, cfg)` classmethod |
| `src/models/gat.py` | Add `from_config(cls, cfg)` classmethod |
| `src/models/dqn.py` | Add `from_config(cls, cfg)` on QNetwork; accept dynamic `state_dim` |
| `pipeline/stages/utils.py` | Rewrite `cache_predictions()` to use extractors. Replace `load_teacher()`/`load_vgae()`/`load_gat()` with generic `load_model()` |
| `pipeline/stages/fusion.py` | Use registry to discover extractors, compute dynamic `state_dim` |
| `pipeline/stages/evaluation.py` | Use registry+extractors for fusion evaluation |
| `pipeline/stages/training.py` | Migrate `load_teacher()` calls to `load_model()` |

### Backward compatibility

Existing DQN checkpoints (`state_dim=15`) stay valid because extractors produce the same 8+7=15 features in the same order.

### Gate

All tests pass + run `hcrl_sa` large+small through full pipeline on GPU.

---

## Deferred

- **Package merge** (src/ + pipeline/) — cosmetic, high churn, low payoff
- **KD loss composition** — do when adding a 4th model with novel distillation
- **Preprocessing refactor** — do when adding a non-automotive domain
- **Conda lock / containers** — nice-to-have reproducibility polish
