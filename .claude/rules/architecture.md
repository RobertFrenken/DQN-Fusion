# KD-GAT Architecture Decisions

## 3-Layer Import Hierarchy

Enforced by `tests/test_layer_boundaries.py`:
- `config/` → never imports from `pipeline/` or `src/`. Pure data definitions + path helpers.
- `pipeline/` → imports `config/` at top level; imports `src/` only inside functions (lazy).
- `src/` → imports `config.constants` for shared constants; never imports from `pipeline/`.

## Config Architecture

- Pydantic v2 frozen BaseModels + YAML composition + JSON serialization.
- Sub-configs: `cfg.vgae`, `cfg.gat`, `cfg.dqn`, `cfg.training`, `cfg.fusion`, `cfg.temporal` — nested Pydantic models. Always use nested access, never flat.
- Auxiliaries: `cfg.auxiliaries` is a list of `AuxiliaryConfig`. KD is a composable loss modifier, not a model identity. Use `cfg.has_kd` / `cfg.kd` properties.
- Constants: domain/infrastructure constants live in `config/constants.py` (not in PipelineConfig). Hyperparameters live in PipelineConfig.

## Experiment Tracking

- W&B (online/offline) for live metrics + S3 lakehouse for structured JSON.
- `cli.py` owns `wandb.init()`/`wandb.finish()` lifecycle; Lightning's `WandbLogger` attaches to the active run.
- Compute nodes auto-set `WANDB_MODE=offline`.

## Orchestration

- Ray (`pipeline/orchestration/`) with `@ray.remote` tasks. `train_pipeline()` fans out per-dataset work concurrently.
- `--local` flag uses Ray local mode. HPO via Ray Tune with OptunaSearch + ASHAScheduler.
- Archive restore: `cli.py` archives previous runs before re-running, restores on failure.

## Inference Serving

`pipeline/serve.py` — FastAPI endpoints (`/predict`, `/health`) loading VGAE+GAT+DQN from `experimentruns/`.

## Dashboard

Config-driven ES module architecture. Adding a visualization = adding an entry to `panelConfig.js`. `BaseChart` provides SVG/tooltip/responsive infrastructure; 8 chart types. `PanelManager` reads config → builds nav + panels + controls. All chart types registered in `Registry`.

Dashboard data: `export.py` scans `experimentruns/` filesystem. `--skip-heavy` runs light exports (~2s, login node safe). Dashboard JS fetches from `s3://kd-gat/dashboard/` with `data/` fallback.

## General Principles

- Delete unused code completely. No compatibility shims or `# removed` comments.
- Dataset catalog: `config/datasets.yaml` — single place to register new datasets.
