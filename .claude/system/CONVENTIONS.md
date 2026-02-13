# Conventions

## Architecture

- Config: Pydantic v2 frozen BaseModels + YAML composition. No Hydra, no OmegaConf.
- All config resolved via `resolve(model_type, scale, auxiliaries="none", **overrides)` — returns frozen `PipelineConfig`.
- **Nested access only**: Use `cfg.vgae.latent_dim`, `cfg.training.lr`, `cfg.kd.temperature`. Never flat access.
- **Auxiliaries**: KD is a composable loss modifier, not a model identity. Use `cfg.has_kd` / `cfg.kd` properties.
- **Adding new config**: New model type → add `config/models/{name}/` dir with scale YAMLs + Architecture class in schema.py. New auxiliary → add `config/auxiliaries/{name}.yaml`.
- **Write-through DB**: `cli.py` records runs directly to project DB. Don't rely on `populate()` as the primary data path.

## Import Rules (3-layer hierarchy)

Enforced by `tests/test_layer_boundaries.py`:

1. **`config/`** (top): Never imports from `pipeline/` or `src/`. Pure data definitions + path helpers.
2. **`pipeline/`** (middle): Imports `config/` freely at top level. Imports `src/` only inside functions (lazy/conditional). This prevents heavy ML dependencies from loading at import time.
3. **`src/`** (bottom): Imports `config.constants` for shared constants. Never imports from `pipeline/`.

When adding new code:
- Constants (window sizes, feature counts, DB paths) → `config/constants.py`
- Hyperparameters → Pydantic models in `config/schema.py`
- Architecture defaults → YAML files in `config/models/` or `config/auxiliaries/`
- Path helpers → `config/paths.py`
- `from config import PipelineConfig, resolve, checkpoint_path` — use the package re-exports

## Code Style

- Don't over-engineer. Minimal changes, no speculative abstractions.
- Don't add docstrings, comments, or type annotations to code that wasn't changed.
- If something is unused, delete it completely. No compatibility shims or `# removed` comments.

## Iteration Hygiene

Before implementing a new feature or fix:
1. **Audit touchpoints** — Identify files that will be modified
2. **Cut stale code** — Remove dead code, hack workarounds, or overly complex solutions in those files
3. **Simplify** — Replace complex patterns with simpler ones if the workaround reason no longer applies
4. **Delete, don't comment** — Unused code gets deleted, not commented out

This keeps the codebase lean. Every PR should leave the code cleaner than it was found.

## Git

- Commit messages: short summary line, body explains why not what.
- Don't use GitLens Commit Composer on HPC — it fails with NFS lock files. Use terminal or Source Control panel.
- Push via SSH (`git@github.com:`), not HTTPS.

## HPC / SLURM

- Always use `spawn` multiprocessing, never `fork` with CUDA.
- Test on small datasets (`hcrl_ch`) before large ones (`set_02`+).
- SLURM logs go to `slurm_logs/`, experiment outputs to `experimentruns/`.

## Session Management

- Say "update state" at end of session to rewrite `STATE.md` with current status.
- Add new crash-prevention rules to `PROJECT_OVERVIEW.md` → Critical Constraints.
- `STATE.md` is rewritten each session, not appended to.
