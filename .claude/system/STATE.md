# Current State

**Date**: 2026-01-30

## What's Working

- Pipeline system (`pipeline/`) committed and pushed to `main`
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- SSH key auth configured for OSC → GitHub
- `.bashrc` fixed (was broken by template placeholders and unmatched quotes)
- `.gitignore` handles `.nfs*` artifacts
- Codebase consolidated: 149 files changed, obsolete files removed

## What's Not Working / Incomplete

- **No training runs validated yet** post-consolidation. The pipeline code has all fixes applied but hasn't been end-to-end tested since the CUDA crash fix + consolidation.
- **`src/` still has dead code**: `src/cli/`, `src/config/`, `src/evaluation/`, `src/utils/` are not imported by pipeline but still on disk. See `docs/CODEBASE_AUDIT.md` Phase 1-5 for cleanup plan.
- **`load_dataset()` still depends on `PathResolver`**: The `_NS` adapter in `stages.py` bridges old and new config formats. Refactoring this (audit Phase 2b) would let us delete `src/paths.py` (1200 lines).

## Next Steps (priority order)

1. Run a training job to validate pipeline post-consolidation
2. Execute cleanup phases from `docs/CODEBASE_AUDIT.md` (trim dead code from `src/`)
3. Refactor `load_dataset()` to remove `PathResolver` dependency

## Recent Decisions

- Accepted deletion of 130 obsolete files (tests/, shell scripts, old docs, visualizations/)
- Old files recoverable from `git stash@{1}` if needed
- Stash list still has 5 entries; stash@{1} has the pre-consolidation untracked files
