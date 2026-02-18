# Conventions

## Architecture

- Config: Pydantic v2 frozen BaseModels + YAML composition. No Hydra, no OmegaConf.
- All config resolved via `resolve(model_type, scale, auxiliaries="none", **overrides)` — returns frozen `PipelineConfig`.
- **Nested access only**: Use `cfg.vgae.latent_dim`, `cfg.training.lr`, `cfg.kd.temperature`. Never flat access.
- **Auxiliaries**: KD is a composable loss modifier, not a model identity. Use `cfg.has_kd` / `cfg.kd` properties.
- **Adding new config**: New model type → add `config/models/{name}/` dir with scale YAMLs + Architecture class in schema.py. New auxiliary → add `config/auxiliaries/{name}.yaml`.
- **W&B tracking**: `cli.py` owns `wandb.init()`/`wandb.finish()` lifecycle. Lightning's `WandbLogger` attaches to the active run. S3 lakehouse sync is fire-and-forget.

## Import Rules (3-layer hierarchy)

Enforced by `tests/test_layer_boundaries.py`:

1. **`config/`** (top): Never imports from `pipeline/` or `src/`. Pure data definitions + path helpers.
2. **`pipeline/`** (middle): Imports `config/` freely at top level. Imports `src/` only inside functions (lazy/conditional). This prevents heavy ML dependencies from loading at import time.
3. **`src/`** (bottom): Imports `config.constants` for shared constants. Never imports from `pipeline/`.

When adding new code:
- Constants (window sizes, feature counts, SLURM defaults) → `config/constants.py`
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

## Repeated Failure Protocol

**If the same command fails twice for the same category of reason, STOP retrying and diagnose.**

The instinct to tweak-and-retry is wrong. Two failures with the same shape (missing env, wrong path, permission error, config conflict) means the problem is structural, not transient. Retrying with small variations wastes time and context window.

Instead:
1. **Name the pattern.** What category of failure is this? (env setup, path resolution, config incompatibility, NFS issue, etc.)
2. **Read the actual error.** Don't skim — parse the full traceback. The root cause is usually in the first or last frame, not the middle.
3. **Check if it's a known issue.** Search CONVENTIONS.md, CLAUDE.md Critical Constraints, and recent git log for prior fixes.
4. **Fix the root cause, not the symptom.** If the env isn't on PATH, don't add it to one command — add it to the documented pattern. If two config values conflict, don't patch one — fix the underlying inconsistency.
5. **Document it.** If this will come up again, add it to the appropriate file (CONVENTIONS.md for workflow, CLAUDE.md Critical Constraints for crash-prevention rules).

Examples of structural vs transient:
- `command not found` → **structural** (env not loaded). Fix the invocation pattern, don't retry.
- `OSError: [Errno 116] Stale file handle` → **transient** (NFS contention). Retry is reasonable once.
- Same Prefect/SLURM error across 3 different flag combinations → **structural**. Stop, read the docs or error message, fix the flow definition.

## Git

- Commit messages: short summary line, body explains why not what.
- Don't use GitLens Commit Composer on HPC — it fails with NFS lock files. Use terminal or Source Control panel.
- Push via SSH (`git@github.com:`), not HTTPS.

## Shell Environment

**This is critical — Claude does not inherit conda.** The login shell has no `conda` on PATH. Every command that needs Python packages (torch, pytest, prefect, etc.) must set up the environment explicitly:

```bash
# Required prefix for ALL Python commands:
export PATH="$HOME/.conda/envs/gnn-experiments/bin:$PATH"
export PYTHONPATH=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT:$PYTHONPATH

# Then run the actual command:
python -m pipeline.cli ...
python -m pipeline.cli flow --dataset hcrl_sa
python -m pytest tests/ -v
```

**Common failures without this:**
- `ModuleNotFoundError: No module named 'config'` → missing PYTHONPATH
- `ModuleNotFoundError: No module named 'torch'` → missing PATH (system Python has no ML packages)

For Prefect-submitted SLURM jobs, the dask-jobqueue SLURMCluster handles environment setup via `job_script_prologue` in `pipeline/flows/slurm_config.py`.

## HPC / SLURM

- Always use `spawn` multiprocessing, never `fork` with CUDA.
- Test on small datasets (`hcrl_ch`) before large ones (`set_02`+).
- SLURM logs go to `slurm_logs/`, experiment outputs to `experimentruns/`.
- Heavy tests use `@pytest.mark.slurm` — auto-skipped on login nodes, run via `scripts/run_tests_slurm.sh`.
- **Always run tests via SLURM** (`cpu` partition, 8 CPUs, 16GB) — login nodes are too slow (tests take 2-5 min on SLURM vs 10+ min on login). Submit with: `sbatch --account=PAS3209 --partition=cpu --time=15 --mem=16G --cpus-per-task=8 --job-name=pytest --output=slurm_logs/%j-pytest.out --wrap='module load miniconda3/24.1.2-py310 && source activate gnn-experiments && cd /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT && python -m pytest tests/ -v'`
- Note: `serial` partition no longer exists on OSC Pitzer — all scripts now use `cpu` partition.

## Session Management

- Say "update state" at end of session to rewrite `STATE.md` with current status.
- Add new crash-prevention rules to `PROJECT_OVERVIEW.md` → Critical Constraints.
- `STATE.md` is rewritten each session, not appended to.
