# Current State

**Date**: 2026-02-02

## What's Working

- Pipeline system (`pipeline/`) committed and pushed to `main`
- CUDA multiprocessing fixes in place (clone-before-to, spawn context)
- SSH key auth configured for OSC -> GitHub (ed25519)
- `.bashrc` fixed, `.gitignore` handles `.nfs*` artifacts
- Codebase audit phases 1-5 complete + additional cleanup pass
- `load_dataset()` refactored to accept direct paths (PathResolver eliminated)
- `datamodules.py` trimmed, metadata-based cache validation (replaces hardcoded expected_sizes)
- Graph caches consolidated to `data/cache/{dataset}/` with `cache_metadata.json`
- DVC configured: 6 raw datasets + 6 caches tracked, remote on `/fs/scratch/PAS1266/can-graph-dvc`
- Dead code removed: `src/evaluation/`, `src/utils/seeding.py`, duplicate `config/plotting_config.py`
- Files relocated: `snakemake_config.yaml` -> `pipeline/`, `notes.md` -> `docs/`, `params.csv` -> `docs/`
- **Path simplification complete**: `paths.py` uses 2-level `stage_dir()` (`experimentruns/{dataset}/{size}_{stage}[_kd]`)
- **Migration executed**: 13 old 8-level experiments moved to 2-level structure; `experimentruns/automotive/` deleted
- **MLflow tracking fully wired**:
  - `tracking.py`: `setup_tracking()`, `start_run()`, `end_run()`, `log_failure()`
  - `cli.py`: MLflow run lifecycle wraps stage dispatch (start/end/failure hooks)
  - `stages.py`: `mlflow.pytorch.autolog(log_models=False)` in `_make_trainer()` for Lightning stages; manual `mlflow.log_metrics()` in DQN fusion loop and evaluation
  - `query.py`: CLI for querying experiments (`--all`, `--dataset`, `--leaderboard`, `--compare`, `--running`)
  - `migrate.py`: Migration tool with `--dry-run`, `--execute`, `--backfill-only` modes
- **Snakefile rewritten**: All 19 rules use 2-level `_p()` helper; DATASETS configurable via `config.get()`; `onstart:` hook for MLflow DB init
- **SLURM profile**: `profiles/slurm/config.yaml` + `profiles/slurm/status.sh` for Snakemake cluster submission
- **`docs/registry_plan.md` cleaned up**: MLflow decision surfaced as primary; original SQLite design marked as historical appendix
- **`docs/snakemake_guide.md`**: Reference guide for Snakemake commands, profiles, skip logic
- **`docs/terminal_upgrades.md`**: Terminal workflow guide (tmux, shells, terminal emulators, CLI tools)

## Active `src/` Files

Essential (imported by pipeline):
- `src/models/` -- GATWithJK, VGAE, DQN (untouched)
- `src/preprocessing/preprocessing.py` -- graph construction (untouched)
- `src/training/datamodules.py` -- load_dataset(), CANGraphDataModule (updated: metadata-based cache validation)

Quarantined (for paper/future):
- `src/config/plotting_config.py`
- `src/utils/plotting_utils.py`

## What's Working (Pipeline Execution)

- **hcrl_sa single-dataset pipeline partially complete** (2026-02-02):
  - VGAE teacher + student (no-KD) + student (KD): All 3 completed, `best_model.pt` saved
  - GAT teacher + student (no-KD): Both completed, `best_model.pt` saved
  - DQN teacher + student (no-KD): Initially failed with `TypeError: list indices must be integers or slices, not list` in `_cache_predictions()`. **Fixed** in `stages.py:702` -- replaced `data[:max_samples]` slice (incompatible with PyTorch `Subset`) with explicit `range(min(len(data), max_samples))` iteration. Re-submitted and running.
  - GAT student (KD), DQN student (KD): Pending (depend on teacher completion)
- **MLflow concurrent initialization race condition fixed**: `onstart:` hook pre-initializes DB; `tracking.py` has retry logic with backoff
- **Snakemake SLURM profile created**: `profiles/slurm/config.yaml` + `status.sh`
- **Snakemake single-dataset support**: `--config 'datasets=["hcrl_sa"]'`

## What's Not Working / Incomplete

- **Old experiment checkpoints have no `config.json`**: The 13 migrated VGAE runs predate the frozen config system. MLflow backfill was skipped for these. Future runs will produce config.json automatically.
- **Data cache race condition** (cosmetic): Two simultaneous DQN jobs on the same dataset both try to write `data/cache/hcrl_sa/processed_graphs.pt`. One succeeds, the other logs `Failed to save cache: No such file or directory` (atomic rename conflict). The job still runs -- it just rebuilds from scratch. Not a blocker but could be optimized.

## Next Steps (Implementation Order)

1. **Verify hcrl_sa pipeline completion** -- check that all 9 jobs finished (DQN re-run + KD student jobs)
   - Check: `squeue -u rf15` for remaining jobs
   - Verify: all `experimentruns/hcrl_sa/*/best_model.pt` files exist
   - Verify: MLflow runs visible via `python -m pipeline.query --all`
2. **Run remaining 5 datasets** via Snakemake (remove `--config` to run all)
3. **Evaluate and collect results** for thesis
4. **Publication plots** using quarantined `src/config/plotting_config.py` + `src/utils/plotting_utils.py`

## Architecture: Snakemake/MLflow Coexistence

**Filesystem** (NFS home, permanent) -- owned by Snakemake:
```
experimentruns/{ds}/{variant}/
  best_model.pt    # Snakemake output, DAG trigger for downstream rules
  config.json      # Frozen config (also logged to MLflow as params)
  metrics.json     # Evaluation stage only
```

**MLflow** (GPFS scratch, 90-day purge) -- supplementary metadata layer:
```
/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db   # SQLite tracking DB
```

**Key principle**: Snakemake needs `best_model.pt` at deterministic paths to compute its DAG. MLflow artifact paths contain run UUIDs (not deterministic at DAG time). So models are saved to the filesystem first (for Snakemake), then logged to MLflow (for tracking). No tmp directory needed.

**If scratch purges**: All checkpoints and configs survive on NFS. Only MLflow tracking history is lost. Acceptable -- MLflow is a convenience layer, not the source of truth.

**MLflow experiment naming**: One experiment (`kd-gat-pipeline`). Run name = `{dataset}/{model_size}_{stage}[_kd]`:
```
Run: "hcrl_sa/teacher_autoencoder"    -> experimentruns/hcrl_sa/teacher_autoencoder/
Run: "hcrl_sa/student_curriculum_kd"  -> experimentruns/hcrl_sa/student_curriculum_kd/
```

**autolog vs manual**: VGAE + GAT use Lightning Trainer -> `mlflow.pytorch.autolog()`. DQN is a custom loop -> manual `mlflow.log_metric()` calls. Evaluation logs flattened core metrics (e.g., `gat_f1`, `vgae_accuracy`, `fusion_mcc`).

**Teacher-student lineage**: `mlflow.set_tag("teacher_run_id", "<teacher_variant>")` on student runs. Delta queries via `mlflow.search_runs()` into pandas.

## Recent Decisions

- **tmux for session persistence** -- Snakemake must stay alive on the login node to submit downstream SLURM jobs. Running it inside tmux prevents SSH disconnects from killing the orchestrator. See `docs/terminal_upgrades.md`.
- **Snakemake + MLflow architecture** -- After evaluating 9 orchestration tools and 7 unified platforms, concluded that no single tool replaces both orchestration and tracking on HPC/SLURM. Snakemake is irreplaceable for DAG + SLURM. MLflow is the best tracking layer. Full analysis in `docs/registry_plan.md`.
- **MLflow replaces custom SQLite registry** -- Earlier plan for `pipeline/registry.py` (~130 lines custom SQLite) is superseded. MLflow provides logging, UI, and model registry out of the box.
- **MLflow backend on GPFS scratch** -- SQLite backend at `sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db`. GPFS has reliable POSIX locking.
- **2-level experiment paths** -- `experimentruns/{dataset}/{model_size}_{stage}[_kd]/` replaces 8-level hierarchy. Migration complete, old `automotive/` directory removed.
- **Filesystem is for Snakemake, MLflow is for humans** -- Models saved to NFS (permanent, Snakemake DAG trigger), logged to MLflow on scratch (supplementary). `config.json` kept on disk as backup.
- Deleted old-format experiment runs outright (~2GB freed)
- Deleted orphaned `data/cache/` (~1.6GB) -- pipeline now uses `data/cache/` via updated `paths.py`
- DVC for data versioning only (not pipeline orchestration -- Snakemake handles that)
- Kept `pipeline/` + `src/` separate (stability over restructuring during active experimentation)

## OSC Environment Details

- **Home** (`/users/PAS2022/rf15/`): NFS v4, 1.7TB mount -- permanent, safe for checkpoints
- **Scratch** (`/fs/scratch/PAS1266/`): GPFS (IBM Spectrum Scale) -- 90-day purge, safe for concurrent DB writes
- **SQLite**: 3.51.1 (stdlib, json_extract confirmed working)
- **MLflow**: 3.8.1 (installed, chosen for experiment tracking -- see `docs/registry_plan.md`)
- **Pandas**: 2.3.3 (for `mlflow.search_runs()` -> DataFrame queries)
- **Jupyter**: Available via OSC OnDemand portal
- **MLflow UI**: Available via OSC OnDemand app (`bc_osc_mlflow`)
