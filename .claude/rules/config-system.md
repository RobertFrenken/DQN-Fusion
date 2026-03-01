# KD-GAT Config System

Config is defined by four orthogonal concerns: **model_type** (architecture), **scale** (capacity), **auxiliaries** (loss modifiers like KD), and **dataset**. Adding a new value along any axis = adding a YAML file.

**Resolution order**: `defaults.yaml` → `models/{type}/{scale}.yaml` → `auxiliaries/{aux}.yaml` → CLI overrides → Pydantic validation → frozen.

```python
from graphids.config import resolve, PipelineConfig
cfg = resolve("vgae", "large", dataset="hcrl_sa")          # No KD
cfg = resolve("gat", "small", auxiliaries="kd_standard")    # With KD
cfg.vgae.latent_dim    # Nested sub-config access
cfg.training.lr        # Training hyperparameters
cfg.has_kd             # Property: any KD auxiliary?
cfg.kd.temperature     # KD auxiliary config (via property)
cfg.active_arch        # Architecture config for active model_type
```

**Path layout**: `experimentruns/{dataset}/{model_type}_{scale}_{stage}[_{aux}]`

**Legacy config loading**: Old flat JSON config files still load via `PipelineConfig.load()` with automatic migration.
