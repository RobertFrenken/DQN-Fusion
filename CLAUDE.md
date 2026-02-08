# KD-GAT: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection using a 3-stage teacher-student knowledge distillation pipeline:
VGAE (unsupervised reconstruction) → GAT (supervised classification) → DQN (RL fusion).
Teachers are compressed into lightweight students for edge deployment.

## Key Commands

```bash
# Run a single stage
python -m pipeline.cli <stage> --preset <model>,<size> --dataset <name>

# Stages: autoencoder, curriculum, fusion, evaluation
# Models: vgae, gat, dqn
# Sizes: teacher, student

# Full pipeline via Snakemake + SLURM
snakemake -s pipeline/Snakefile --profile profiles/slurm

# Single dataset
snakemake -s pipeline/Snakefile --profile profiles/slurm --config 'datasets=["hcrl_sa"]'

# Dry run (always do this first)
snakemake -s pipeline/Snakefile -n

# Data management
python -m pipeline.ingest <dataset>     # CSV → Parquet conversion
python -m pipeline.ingest --all         # Convert all datasets
python -m pipeline.ingest --list        # List catalog entries
python -m pipeline.db populate          # Populate project DB from existing outputs
python -m pipeline.db summary           # Show dataset/run/metric counts
python -m pipeline.db query "SQL"       # Run arbitrary SQL on project DB

# Run tests
python -m pytest tests/test_pipeline_integration.py -v

# Check SLURM jobs
squeue -u $USER
```

## Project Structure

```
pipeline/           # Main orchestration (frozen dataclasses, no Hydra)
  cli.py            # Entry point: python -m pipeline.cli <stage>
  config.py         # PipelineConfig frozen dataclass + PRESETS
  paths.py          # Canonical 2-level path layout
  stages/           # Stage implementations (training, fusion, evaluation)
  ingest.py         # CSV → Parquet conversion + dataset registration
  db.py             # SQLite project DB (datasets, runs, metrics tables)
  Snakefile         # 19 rules, all stages + evaluation + onsuccess hooks
src/                # Supporting modules
  models/           # VGAE, GATWithJK, DQN architectures
  training/         # CANGraphDataModule, data loading
  preprocessing/    # Graph construction from CAN CSVs
data/
  datasets.yaml     # Dataset catalog (add entries here for new datasets)
  project.db        # SQLite DB: queryable datasets, runs, metrics
  automotive/       # 6 datasets (DVC-tracked): hcrl_ch, hcrl_sa, set_01-04
  parquet/          # Columnar format (from ingest), queryable via DuckDB
  cache/            # Preprocessed graph cache (.pt, .pkl, metadata)
experimentruns/     # Outputs: best_model.pt, config.json, metrics.json
profiles/slurm/     # SLURM submission profile for Snakemake
```

## Critical Constraints

These fix real crashes -- do not violate:

- **PyG `Data.to()` is in-place.** Always `.clone().to(device)`, never `.to(device)` on shared data.
- **Use spawn multiprocessing.** Never `fork` with CUDA. Set `mp_start_method='spawn'` and `multiprocessing_context='spawn'` on all DataLoaders.
- **NFS filesystem.** `.nfs*` ghost files appear on delete. Already in `.gitignore`.
- **No GUI on HPC.** Git auth via SSH key, not HTTPS.

## Architecture Decisions

- Config: frozen dataclasses + JSON. No Hydra, no Pydantic, no OmegaConf.
- Imports from `src/` are conditional (inside functions) to avoid top-level coupling.
- Dual storage: Snakemake owns filesystem paths (DAG triggers), MLflow owns metadata (tracking/UI).
- Data layer: Parquet (columnar storage) + SQLite (project DB) + DuckDB (query engine). All serverless.
- Dataset catalog: `data/datasets.yaml` — single place to register new datasets.
- Delete unused code completely. No compatibility shims or `# removed` comments.

## Environment

- **Cluster**: OSC (Ohio Supercomputer Center), RHEL 9, SLURM
- **GPU**: V100 (account PAS3209, gpu partition)
- **Python**: conda `gnn-experiments` (PyTorch, PyG, Lightning, MLflow, DuckDB)
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **MLflow DB**: `/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db` (auto-backed up to `~/backups/` on pipeline success)
- **Project DB**: `data/project.db` (SQLite — datasets, runs, metrics)

## Detailed Documentation

- `.claude/system/PROJECT_OVERVIEW.md` -- Full architecture, models, memory optimization
- `.claude/system/CONVENTIONS.md` -- Code style, iteration hygiene, git rules
- `.claude/system/STATE.md` -- Current session state (updated each session)
- `docs/user_guides/` -- Snakemake guide, MLflow usage, terminal setup
