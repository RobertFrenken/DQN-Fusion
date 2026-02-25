# KD-GAT Code Style

## Import Rules (3-layer hierarchy)

Enforced by `tests/test_layer_boundaries.py`:

1. **`config/`** (top): Never imports from `pipeline/` or `src/`. Pure data definitions + path helpers.
2. **`pipeline/`** (middle): Imports `config/` freely at top level. Imports `src/` only inside functions (lazy).
3. **`src/`** (bottom): Imports `config.constants` for shared constants. Never imports from `pipeline/`.

When adding new code:
- Constants → `config/constants.py`
- Hyperparameters → Pydantic models in `config/schema.py`
- Architecture defaults → YAML files in `config/models/` or `config/auxiliaries/`
- Path helpers → `config/paths.py`
- `from config import PipelineConfig, resolve, checkpoint_path` — use the package re-exports

## General Style

- Don't over-engineer. Minimal changes, no speculative abstractions.
- Don't add docstrings, comments, or type annotations to code that wasn't changed.
- If something is unused, delete it completely. No compatibility shims.

## Iteration Hygiene

Before implementing a new feature or fix:
1. **Audit touchpoints** — identify files that will be modified
2. **Cut stale code** — remove dead code in those files
3. **Simplify** — replace complex patterns with simpler ones
4. **Delete, don't comment** — unused code gets deleted

## Repeated Failure Protocol

If the same command fails twice for the same category of reason, STOP retrying and diagnose:
1. Name the pattern (env setup, path resolution, config incompatibility, NFS issue)
2. Read the full traceback — root cause is usually in the first or last frame
3. Check known issues in `critical-constraints.md` and `knowledge-bank.md`
4. Fix the root cause, not the symptom
5. Document if it will recur

## Git

- Short summary line, body explains why not what.
- Push via SSH (`git@github.com:`), not HTTPS.
