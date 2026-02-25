# KD-GAT: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection using a 3-stage knowledge distillation pipeline:
VGAE (unsupervised reconstruction) → GAT (supervised classification) → DQN (RL fusion).
Large models are compressed into small models via KD auxiliaries for edge deployment.

## Key Commands

```bash
# Run a single stage
python -m pipeline.cli <stage> --model <type> --scale <size> --dataset <name>
# Stages: autoencoder, curriculum, normal, fusion, evaluation, temporal
# Models: vgae, gat, dqn | Scales: large, small | Auxiliaries: none, kd_standard

# Examples
python -m pipeline.cli autoencoder --model vgae --scale large --dataset hcrl_sa
python -m pipeline.cli curriculum --model gat --scale small --auxiliaries kd_standard --teacher-path <path> --dataset hcrl_sa
python -m pipeline.cli fusion --model dqn --scale large --dataset hcrl_ch
python -m pipeline.cli temporal --model gat --scale large --dataset hcrl_sa -O temporal.enabled true
python -m pipeline.cli autoencoder --model vgae --scale large -O training.lr 0.001 -O vgae.latent_dim 16

# Full pipeline via Ray + SLURM
python -m pipeline.cli flow --dataset hcrl_sa
sbatch scripts/ray_slurm.sh flow --dataset hcrl_sa
python -m pipeline.cli flow --dataset hcrl_sa --local  # No SLURM

# Export + analytics
python -m pipeline.export --skip-heavy         # Light exports (~2s, login node OK)
python -m pipeline.build_analytics             # DuckDB rebuild (sub-second, views over Parquet)
python -m pipeline.migrate_datalake            # One-time: filesystem → Parquet datalake
bash scripts/export_dashboard.sh               # Export + commit + push

# Tests — ALWAYS submit to SLURM
bash scripts/run_tests_slurm.sh
bash scripts/run_tests_slurm.sh -k "test_full_pipeline"

# Docs site (Astro) — requires: module load node-js/22.12.0
cd docs-site && npm run dev
cd docs-site && npm run build
```

## Skills

| Skill | Usage | Description |
|-------|-------|-------------|
| `/run-pipeline` | `/run-pipeline hcrl_sa large` | Submit Ray pipeline to SLURM |
| `/check-status` | `/check-status hcrl_sa` | Check SLURM queue, checkpoints, W&B |
| `/run-tests` | `/run-tests` or `/run-tests test_config` | Run pytest suite |
| `/sync-state` | `/sync-state` | Update STATE.md from current outputs |

## Rules (auto-loaded from `.claude/rules/`)

All project conventions, architecture decisions, and constraints are in modular rule files:
- `project-structure.md` — directory tree, layer hierarchy
- `config-system.md` — Pydantic + YAML resolution
- `critical-constraints.md` — crash-prevention rules (DO NOT VIOLATE)
- `architecture.md` — 3-layer hierarchy, orchestration, dashboard, docs-site
- `code-style.md` — imports, iteration hygiene, git
- `shell-environment.md` — uv + `.venv/` setup (CANONICAL — not conda)
- `slurm-hpc.md` — SLURM conventions, login node safety
- `experiment-tracking.md` — W&B, lakehouse, DuckDB analytics
- `docs-site.md` — Astro + Svelte (path-scoped: `docs-site/**`)
- `pytorch-compat.md` — uv + PyTorch + PyG version pinning

> Cross-repo propagation: See `~/.claude/rules/cross-repo-propagation.md`
> Environment variables: See `~/.claude/rules/secrets-and-env-vars.md`

## Detailed Documentation

- `.claude/system/PROJECT_OVERVIEW.md` — full architecture, models, memory optimization
- `.claude/system/STATE.md` — current session state (updated each session)
