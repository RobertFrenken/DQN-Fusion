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

# Experiment analytics (queries project DB)
python -m pipeline.analytics sweep --param lr --metric f1
python -m pipeline.analytics leaderboard --metric f1 --top 10
python -m pipeline.analytics compare <run_a> <run_b>
python -m pipeline.analytics diff <run_a> <run_b>
python -m pipeline.analytics dataset <name>
python -m pipeline.analytics query "SELECT json_extract(...) FROM ..."

# MLflow UI (inside tmux on login node)
mlflow ui --backend-store-uri sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db --host 0.0.0.0 --port 5000
# Local: ssh -L 5000:localhost:5000 rf15@pitzer.osc.edu → http://localhost:5000

# Datasette (interactive DB browsing, inside tmux on login node)
datasette data/project.db --port 8001
# Local: ssh -L 8001:localhost:8001 rf15@pitzer.osc.edu → http://localhost:8001

# Snakemake report (after eval runs)
snakemake -s pipeline/Snakefile --report report.html

# Run tests
python -m pytest tests/ -v

# Check SLURM jobs
squeue -u $USER
```

## Project Structure

```
pipeline/           # Main orchestration (frozen dataclasses, no Hydra)
  cli.py            # Entry point + write-through DB recording
  config.py         # PipelineConfig + typed sub-configs (VGAEConfig, GATConfig, etc.)
  paths.py          # Canonical 2-level path layout
  stages/           # Stage implementations (training, fusion, evaluation)
  ingest.py         # CSV → Parquet conversion + dataset registration
  db.py             # SQLite project DB + write-through record_run_start/end
  analytics.py      # Post-run analysis: sweeps, leaderboards, comparisons
  Snakefile         # 20 rules, all stages + evaluation + onsuccess hooks
src/                # Supporting modules
  models/           # vgae.py, gat.py, dqn.py
  training/         # load_dataset(), graph caching
  preprocessing/    # Graph construction from CAN CSVs
data/
  datasets.yaml     # Dataset catalog (add entries here for new datasets)
  project.db        # SQLite DB: queryable datasets, runs, metrics
  automotive/       # 6 datasets (DVC-tracked): hcrl_ch, hcrl_sa, set_01-04
  parquet/          # Columnar format (from ingest), queryable via Datasette or SQL
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
- Sub-configs: `cfg.vgae`, `cfg.gat`, `cfg.dqn`, `cfg.kd`, `cfg.fusion` — typed views over flat fields. Use sub-config access (`cfg.vgae.latent_dim`) in new code, not flat access (`cfg.vgae_latent_dim`).
- Write-through DB: `cli.py` records run start/end directly to project DB. `populate()` is a backfill/recovery tool only.
- Imports from `src/` are conditional (inside functions) to avoid top-level coupling.
- Triple storage: Snakemake owns filesystem paths (DAG triggers), MLflow owns metadata (tracking/UI), project DB owns structured results (write-through from cli.py).
- Data layer: Parquet (columnar storage) + SQLite (project DB) + Datasette (interactive browsing). All serverless.
- Dataset catalog: `data/datasets.yaml` — single place to register new datasets.
- Delete unused code completely. No compatibility shims or `# removed` comments.

## Environment

- **Cluster**: OSC (Ohio Supercomputer Center), RHEL 9, SLURM
- **GPU**: V100 (account PAS3209, gpu partition)
- **Python**: conda `gnn-experiments` (PyTorch, PyG, Lightning, MLflow, Datasette)
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **MLflow DB**: `/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db` (auto-backed up to `~/backups/` on pipeline success)
- **Project DB**: `data/project.db` (SQLite — datasets, runs, metrics)

## Detailed Documentation

- `.claude/system/PROJECT_OVERVIEW.md` -- Full architecture, models, memory optimization
- `.claude/system/CONVENTIONS.md` -- Code style, iteration hygiene, git rules
- `.claude/system/STATE.md` -- Current session state (updated each session)
- `docs/user_guides/` -- Snakemake guide, MLflow usage, Datasette usage, terminal setup
