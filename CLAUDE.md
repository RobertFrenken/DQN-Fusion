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
  Snakefile         # 19 rules, all stages + evaluation
src/                # Supporting modules
  models/           # VGAE, GATWithJK, DQN architectures
  training/         # CANGraphDataModule, data loading
  preprocessing/    # Graph construction from CAN CSVs
data/automotive/    # 6 datasets (DVC-tracked): hcrl_ch, hcrl_sa, set_01-04
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
- Delete unused code completely. No compatibility shims or `# removed` comments.

## Environment

- **Cluster**: OSC (Ohio Supercomputer Center), RHEL 9, SLURM
- **GPU**: V100 (account PAS3209, gpu partition)
- **Python**: conda `gnn-experiments` (PyTorch, PyG, Lightning, MLflow)
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **MLflow DB**: `/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db`

## Detailed Documentation

- `.claude/system/PROJECT_OVERVIEW.md` -- Full architecture, models, memory optimization
- `.claude/system/CONVENTIONS.md` -- Code style, iteration hygiene, git rules
- `.claude/system/STATE.md` -- Current session state (updated each session)
- `docs/user_guides/` -- Snakemake guide, MLflow usage, terminal setup
