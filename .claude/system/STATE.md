# Current State

**Date**: 2026-01-30

## What's Working

- Pipeline system (`pipeline/`) committed and pushed to `main`
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- SSH key auth configured for OSC -> GitHub (ed25519)
- `.bashrc` fixed, `.gitignore` handles `.nfs*` artifacts
- Codebase audit phases 1-5 complete: 36 files deleted, 12,605 lines removed
- `load_dataset()` refactored to accept direct paths (PathResolver eliminated)
- `datamodules.py` trimmed from 917 to 305 lines
- `_NS` legacy adapter removed from `stages.py`
- `src/paths.py` (1200 lines) deleted
- `pyproject.toml` entry point updated to `pipeline.cli:main`
- All 7 verification tests pass (stages, datamodules, models, preprocessing, CLI, src exports, no PathResolver)

## What's Not Working / Incomplete

- **No training runs validated yet** post-cleanup. All code changes are structural (deletions + refactoring), but an end-to-end run hasn't been done since the CUDA crash fix + audit cleanup.

## Remaining `src/` Files

Essential (imported by pipeline):
- `src/models/` — GATWithJK, VGAE, DQN (untouched)
- `src/preprocessing/preprocessing.py` — graph construction (untouched)
- `src/training/datamodules.py` — load_dataset(), CANGraphDataModule (trimmed)

Quarantined (for paper/future):
- `src/evaluation/` — richer eval pipeline (4 files)
- `src/config/plotting_config.py`
- `src/utils/plotting_utils.py`, `src/utils/seeding.py`

## Next Steps

1. Run a training job to validate pipeline end-to-end post-cleanup
2. Begin extensive training runs across datasets

## Recent Decisions

- Accepted deletion of 130+ obsolete files across two commits
- Old files recoverable from `git stash@{1}` if needed (5 stashes remain)
- `src/visualizations/` Python files already gone (removed in earlier consolidation commit)
