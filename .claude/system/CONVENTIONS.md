# Conventions

## Architecture

- Pipeline system uses frozen dataclasses + JSON. No Hydra, no Pydantic, no OmegaConf in new code.
- All config is `PipelineConfig` — one object, one file. Presets for model/size combos.
- **Sub-config views**: Use `cfg.vgae.latent_dim` (not `cfg.vgae_latent_dim`) in new/modified code. Flat fields remain for serialization; sub-config properties are computed views.
- **Write-through DB**: `cli.py` records runs directly to project DB. Don't rely on `populate()` as the primary data path.
- Imports from `src/` are conditional (inside functions) to avoid top-level coupling.

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
