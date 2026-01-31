# Current State

**Date**: 2026-01-31

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
- Files relocated: `snakemake_config.yaml` → `pipeline/`, `notes.md` → `docs/`, `params.csv` → `docs/`

## Active `src/` Files

Essential (imported by pipeline):
- `src/models/` — GATWithJK, VGAE, DQN (untouched)
- `src/preprocessing/preprocessing.py` — graph construction (untouched)
- `src/training/datamodules.py` — load_dataset(), CANGraphDataModule (updated: metadata-based cache validation)

Quarantined (for paper/future):
- `src/config/plotting_config.py`
- `src/utils/plotting_utils.py`

## What's Not Working / Incomplete

- **No training runs validated yet** post-cleanup. All code changes are structural (deletions + refactoring), but an end-to-end run hasn't been done since the CUDA crash fix + audit cleanup.
- **Experiment tracking not yet integrated** — MLflow integration planned per revised `docs/registry_plan.md`.
- **Path hierarchy still 8 levels** — will be collapsed to 2 levels alongside MLflow integration.
- **`docs/registry_plan.md` has stale SQLite design** — Full custom SQLite schema/implementation below the Architecture Research section is now superseded by MLflow decision. Should be trimmed or marked historical.

## Next Steps (Implementation Order)

1. **Simplify paths from 8→2 levels** (`paths.py`, Snakefile, `config.py`)
   - `paths.py`: Rewrite `stage_dir()` to `experimentruns/{dataset}/{model_size}_{stage}[_kd]/`
   - `Snakefile`: Rewrite `_p()` helper + all 18 rules to use new paths
   - `config.py`: Drop `modality` field (always "automotive") — or keep for forward compat
   - No code changes in `stages.py` (it calls paths.py functions, not path strings)
2. **Add MLflow integration** to `pipeline/cli.py` and `pipeline/stages.py` (~30-50 lines)
   - `cli.py`: Set tracking URI, start/end MLflow run around dispatch, log params + tags
   - `stages.py`: `mlflow.pytorch.autolog()` for Lightning stages (VGAE, GAT); manual `mlflow.log_metric()` in DQN loop (~5 lines)
   - Drop `logs/` directory creation (MLflow handles epoch metrics)
   - Keep `config.json` + `metrics.json` on filesystem (belt-and-suspenders)
3. **Clean up `docs/registry_plan.md`** — Trim/mark stale SQLite design as historical
4. **Validate pipeline end-to-end** with one dataset (e.g., `hcrl_sa`)
5. **Begin full training runs** across all 6 datasets via Snakemake

## Architecture: Snakemake/MLflow Coexistence

**Filesystem** (NFS home, permanent) — owned by Snakemake:
```
experimentruns/{ds}/{variant}/
  best_model.pt    # Snakemake output, DAG trigger for downstream rules
  config.json      # Frozen config (also logged to MLflow as params)
  metrics.json     # Evaluation stage only
```

**MLflow** (GPFS scratch, 90-day purge) — supplementary metadata layer:
```
/fs/scratch/PAS1266/mlflow/mlflow.db       # SQLite tracking DB
/fs/scratch/PAS1266/mlflow/mlruns/         # Artifact store (model copies, plots)
```

**Key principle**: Snakemake needs `best_model.pt` at deterministic paths to compute its DAG. MLflow artifact paths contain run UUIDs (not deterministic at DAG time). So models are saved to the filesystem first (for Snakemake), then logged to MLflow (for tracking). No tmp directory needed.

**If scratch purges**: All checkpoints and configs survive on NFS. Only MLflow tracking history is lost. Acceptable — MLflow is a convenience layer, not the source of truth.

**MLflow experiment naming**: One experiment per dataset. Run name = variant:
```
Experiment: "hcrl_sa"
  Run: "teacher_vgae"      → experimentruns/hcrl_sa/teacher_vgae/
  Run: "student_gat_kd"    → experimentruns/hcrl_sa/student_gat_kd/
```

**autolog vs manual**: VGAE + GAT use Lightning Trainer → `mlflow.pytorch.autolog()`. DQN is a custom loop → manual `mlflow.log_metric()` calls.

**Teacher-student lineage**: `mlflow.set_tag("teacher_run_id", "<teacher_variant>")` on student runs. Delta queries via `mlflow.search_runs()` into pandas.

## Recent Decisions

- **Snakemake + MLflow architecture** — After evaluating 9 orchestration tools (Nextflow, Prefect, DVC, Luigi, Metaflow, Ray, Submitit, ClearML, SLURM native) and 7 unified platforms, concluded that no single tool replaces both orchestration and tracking on HPC/SLURM. Snakemake is irreplaceable for DAG + SLURM. MLflow (3.8.1, already installed) is the best tracking layer: auto-logging for Lightning, OSC OnDemand UI, model registry. ClearML was the only genuine unified contender but its SLURM support (May 2024) is too new. Full analysis in `docs/registry_plan.md`.
- **MLflow replaces custom SQLite registry** — Earlier plan for `pipeline/registry.py` (~130 lines custom SQLite) is superseded. MLflow provides logging, UI, and model registry out of the box. Teacher-student lineage tracked via tags + pandas, not SQL JOINs. Trade-off: loses native SQL JOIN power, gains UI + auto-logging + model versioning + industry standard.
- **MLflow backend on GPFS scratch** — SQLite backend at `sqlite:////fs/scratch/PAS1266/mlflow/mlflow.db`. GPFS has reliable POSIX locking. At most 6 concurrent writers per stage; contention is negligible.
- **2-level experiment paths** — `experimentruns/{dataset}/{model_size}_{stage}[_kd]/` replaces 8-level hierarchy. Deterministic paths remain for Snakemake DAG; MLflow is a supplementary metadata layer.
- **Filesystem is for Snakemake, MLflow is for humans** — No tmp directory. Models saved to NFS (permanent, Snakemake DAG trigger), logged to MLflow on scratch (supplementary). `logs/` directory dropped; MLflow handles epoch metrics. `config.json` + `metrics.json` kept on disk as backup.
- Deleted old-format experiment runs outright (~2GB freed)
- Deleted orphaned `data/cache/` (~1.6GB) — pipeline now uses `data/cache/` via updated `paths.py`
- DVC for data versioning only (not pipeline orchestration — Snakemake handles that)
- DVC remote on scratch (90-day purge is acceptable — `dvc push` rebuilds it from working tree)
- Kept `pipeline/` + `src/` separate (stability over restructuring during active experimentation)

## OSC Environment Details

- **Home** (`/users/PAS2022/rf15/`): NFS v4, 1.7TB mount — permanent, safe for checkpoints
- **Scratch** (`/fs/scratch/PAS1266/`): GPFS (IBM Spectrum Scale) — 90-day purge, safe for concurrent DB writes
- **SQLite**: 3.51.1 (stdlib, json_extract confirmed working)
- **MLflow**: 3.8.1 (installed, chosen for experiment tracking — see `docs/registry_plan.md`)
- **Pandas**: 2.3.3 (for `mlflow.search_runs()` → DataFrame queries)
- **Jupyter**: Available via OSC OnDemand portal
- **MLflow UI**: Available via OSC OnDemand app (`bc_osc_mlflow`)
