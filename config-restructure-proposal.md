# Config Restructure Proposal: Group Folders + Thin Presets

## Status Quo

The current preset system lives in `pipeline_config.py` (~338 lines) and uses a 3-layer Python dict merge:

```
ARCHITECTURES[(model, size)]   →  weight shapes (must match checkpoint)
       +
TRAINING[model]                →  training hyperparams (tunable)
       +
TRAINING_SIZE[(model, size)]   →  size-specific overrides (e.g., student gets use_kd=True)
       =
PRESETS[(model, size)]         →  complete config for one (model, size) combination
```

This produces 6 presets (3 models × 2 sizes). The Snakefile consumes them as `--preset vgae,teacher`. The merge logic is clean (~8 lines), explicit, and tested.

**What's not broken:** the composition logic, the Snakefile interface, the preset concept itself.

**What is painful:**

- A flat 60+ field dataclass — hard to scan, hard to tell which fields interact
- Manual validation in a separate `validate.py` instead of declarative constraints on fields
- Sub-configs are property views (~60 lines of boilerplate) instead of real nested classes
- Adding a variant means editing Python dicts inside a 338-line file
- New lab members can't discover available configs without reading source

---

## Proposed Structure

```
config/
├── schema.py              # Pydantic models — source of truth for types + validation
├── resolver.py            # ~60 line loader that composes group files via presets
├── defaults.yaml          # Global defaults for every field
│
├── model/                 # Axis 1: architecture
│   ├── vgae.yaml
│   ├── gat.yaml
│   └── gcn.yaml
│
├── training/              # Axis 2: training regime
│   ├── base.yaml
│   ├── distillation.yaml  # use_kd: true, kd_alpha, kd_temperature
│   └── lightweight.yaml   # (optional) smaller runs for debugging / students
│
├── size/                  # Axis 3: model scale
│   ├── teacher.yaml
│   └── student.yaml
│
└── presets/               # Thin recipes — name which group file from each axis
    ├── vgae_teacher.yaml
    ├── vgae_student.yaml
    ├── gat_teacher.yaml
    ├── gat_student.yaml
    ├── gcn_teacher.yaml
    └── gcn_student.yaml
```

### Preset files are recipes, not blobs

A preset names its ingredients and optionally overrides scalars:

```yaml
# presets/vgae_student.yaml
inherit:
  model: vgae
  training: distillation
  size: student

overrides:
  training:
    lr: 0.001  # this specific combo needs a lower LR
```

No duplication. Change `patience` in `training/base.yaml` and it propagates to every preset that inherits `base`.

### Resolver (~60 lines of Python)

```python
# config/resolver.py
from pathlib import Path
import yaml
from config.schema import PipelineConfig

CONFIG_DIR = Path(__file__).parent

def load_group(group: str, name: str) -> dict:
    path = CONFIG_DIR / group / f"{name}.yaml"
    return yaml.safe_load(path.read_text())

def resolve_preset(preset_name: str, cli_overrides: dict = None) -> PipelineConfig:
    # 1. Load the thin preset recipe
    preset = yaml.safe_load(
        (CONFIG_DIR / "presets" / f"{preset_name}.yaml").read_text()
    )

    # 2. Start from defaults
    merged = yaml.safe_load((CONFIG_DIR / "defaults.yaml").read_text())

    # 3. Layer each group file (same merge logic as current system)
    for group, name in preset["inherit"].items():
        deep_merge(merged, load_group(group, name))

    # 4. Preset-specific overrides
    if preset.get("overrides"):
        deep_merge(merged, preset["overrides"])

    # 5. CLI overrides (dot-path: training.lr=0.001)
    if cli_overrides:
        deep_merge(merged, cli_overrides)

    # 6. Validate + return typed config
    return PipelineConfig(**merged)

def deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base
```

This is the same 3-layer merge, reading from files instead of dicts. Composition logic stays in Python, stays explicit.

### Schema (Pydantic replaces flat dataclass + validate.py)

```python
# config/schema.py
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    vgae_latent_dim: int = Field(48, ge=1, le=512)
    encoder_layers: int = Field(2, ge=1, le=8)
    # ... architecture fields only

class TrainingConfig(BaseModel):
    lr: float = Field(0.002, gt=0, le=1.0)
    patience: int = Field(100, ge=1)
    epochs: int = Field(500, ge=1)
    use_kd: bool = False
    kd_alpha: float = Field(0.5, ge=0, le=1.0)
    kd_temperature: float = Field(4.0, gt=0)
    # ... training fields only

class SizeConfig(BaseModel):
    hidden_dim: int = Field(128, ge=1)
    # ... size-dependent fields only

class PipelineConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    size: SizeConfig = SizeConfig()
    # access: config.training.lr, config.model.vgae_latent_dim
```

**What this replaces:**

- The flat 60+ field dataclass → nested, scannable sub-configs
- The manual `validate.py` → declarative `Field(ge=0, le=1.0)` constraints
- The ~60 lines of property-view boilerplate → native nested access

---

## Potential Coupling Points Between Axes

These are places where "model," "training," and "size" are not fully orthogonal — where a value on one axis constrains or changes the meaning of a value on another. Based on the current system description:

### 1. `use_kd` lives on the training axis but is determined by size

Knowledge distillation is enabled for students and disabled for teachers. Currently this is handled by `TRAINING_SIZE` — the size-specific override sets `use_kd=True` for students. In the new system, this coupling moves to the preset's `overrides` block or into `size/student.yaml` directly.

**Risk:** If someone creates a preset with `size: student` but `training: base` (which has `use_kd: false`), is that valid? It might be for a standalone student baseline. Or it might silently produce a broken run. You need to decide: is `use_kd` a training concern or a size concern?

**Recommendation:** Keep `use_kd` and related KD fields (`kd_alpha`, `kd_temperature`) in `TrainingConfig`, but put the `distillation.yaml` training group as the expected pairing for students. A Pydantic `model_validator` can warn (not error) if `size=student` and `use_kd=False`.

### 2. Architecture weight shapes must match checkpoints

`ARCHITECTURES[(model, size)]` sets weight shapes that must align with saved checkpoints. This means model and size are *not* independently combinable for inference — a `vgae` architecture with `teacher` sizing produces specific tensor dimensions that a `student` checkpoint can't load.

**Risk:** This is already handled by having 6 explicit (model, size) combinations, but the group-folder structure could mislead someone into thinking they can freely mix `model/vgae.yaml` with any `size/*.yaml`. In practice, `vgae_latent_dim: 48` for a teacher vs `vgae_latent_dim: 16` for a student means the model and size axes are coupled on dimension fields.

**Recommendation:** Architecture fields that vary by size (like `vgae_latent_dim`) probably belong in `size/` rather than `model/`, or you need a cross-validation rule. Alternatively, accept that model × size is a coupled pair and put the architecture params directly in the preset (since presets already name both axes).

### 3. Training hyperparams that are model-specific

The current `TRAINING[model]` dict suggests that different models have different training defaults (e.g., GAT might need a different LR than VGAE). This means the training and model axes aren't fully orthogonal.

**Recommendation:** If the model-specific training differences are just a few scalar overrides (LR, weight decay), handle them in the preset's `overrides` block. If they're substantial (different schedulers, different loss functions), you may need `training/base_vgae.yaml` etc., which partially collapses the two axes. That's fine — not every system is perfectly orthogonal, and the preset layer exists precisely to handle these couplings.

---

## Migration Path

This doesn't need to happen all at once. A phased approach:

1. **Phase 1: Pydantic schema.** Replace the flat dataclass and `validate.py` with nested Pydantic models. Keep presets in Python dicts. This is the highest-value change with the lowest risk — it fixes the 60-field readability problem and gives you declarative validation immediately.

2. **Phase 2: Extract YAML group files.** Move the dict values into `model/`, `training/`, `size/` YAML files. Write the resolver. Presets can still be Python dicts that reference file names, or you can make them YAML too.

3. **Phase 3: Thin preset YAML files.** Once the groups exist as files, presets become trivial YAML recipes. Add `--show-config` to dump the resolved config.

Each phase is independently useful and testable. You can stop after Phase 1 if the YAML move doesn't feel worth it at 6 presets.

---

## Quick Reference: What Goes Where

| Concern | Current location | Proposed location |
|---|---|---|
| Field types + ranges | `validate.py` (manual) | `schema.py` (Pydantic `Field`) |
| Architecture params | `ARCHITECTURES` dict | `model/*.yaml` |
| Training defaults | `TRAINING` dict | `training/*.yaml` |
| Size-specific overrides | `TRAINING_SIZE` dict | `size/*.yaml` + preset overrides |
| Preset definitions | `PRESETS` dict (pipeline_config.py) | `presets/*.yaml` (thin recipes) |
| Composition logic | 8 lines in pipeline_config.py | `resolver.py` (~60 lines) |
| Sub-config access | Property views (~60 lines) | Native Pydantic nesting |
| CLI interface | `--preset vgae,teacher` | `--preset vgae_teacher` (unchanged UX) |

---

## Open Questions for Discussion

1. **Where does `vgae_latent_dim` live?** If it varies by size (48 for teacher, 16 for student), it's a size concern, not a model concern. But the field name suggests it's model-specific. This is the biggest coupling to resolve.

2. **Should `use_kd=False` + `size=student` be an error or a warning?** Determines how strict the cross-axis validation is.

3. **Is Phase 1 alone sufficient?** If the flat dataclass is the real pain point, Pydantic nesting might solve 80% of the problem without touching the preset storage format at all.
