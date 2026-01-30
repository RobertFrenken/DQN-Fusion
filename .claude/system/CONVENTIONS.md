# Conventions

## Architecture

- Pipeline system uses frozen dataclasses + JSON. No Hydra, no Pydantic, no OmegaConf in new code.
- All config is `PipelineConfig` — one object, one file. Presets for model/size combos.
- Imports from `src/` are conditional (inside functions) to avoid top-level coupling.

## Code Style

- Don't over-engineer. Minimal changes, no speculative abstractions.
- Don't add docstrings, comments, or type annotations to code that wasn't changed.
- If something is unused, delete it completely. No compatibility shims or `# removed` comments.

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
