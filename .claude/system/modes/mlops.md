# Mode: MLOps

**Active mode**: MLOps — Pipeline execution, infrastructure, and operational tasks.

## Focus Areas

- Snakemake pipeline execution and debugging
- SLURM job management (submission, monitoring, resource allocation)
- Config system maintenance (YAML composition, Pydantic validation)
- MLflow/WandB tracking and experiment management
- Training failure triage (OOM, CUDA errors, NaN losses)
- Data pipeline health (ingest, preprocessing, caching)
- Project DB queries and analytics

## Context Files (prioritize reading these)

- `CLAUDE.md` — Full command reference and project structure
- `.claude/system/STATE.md` — Current pipeline state and experiment status
- `pipeline/Snakefile` — DAG definitions, SLURM resources, retry logic
- `config/` — Schema, resolver, constants, YAML files
- `pipeline/stages/` — Stage implementations
- `pipeline/tracking.py` — MLflow integration
- `pipeline/db.py` — Project DB write-through

## Suppressed Topics

Do NOT initiate discussion about:
- Research hypotheses (OOD generalization, JumpReLU, cascading KD)
- Paper writing or literature review
- Documentation site updates
- Unless the user explicitly asks

## Available Commands

| Command | Description |
|---------|-------------|
| `/run-pipeline <dataset> [target]` | Submit Snakemake jobs to SLURM |
| `/check-status [dataset]` | Check SLURM queue, checkpoints, DB status |
| `/run-tests [pattern]` | Run pytest suite |
| `/sync-state` | Update STATE.md from current outputs |
| `python -m pipeline.cli <stage> ...` | Run single stage directly |
| `python -m pipeline.db summary` | DB summary counts |
| `python -m pipeline.analytics leaderboard` | Top metrics |

## Quick Diagnostics

When troubleshooting, check in this order:
1. `squeue -u $USER` — Are jobs running/pending?
2. `slurm_logs/<jobid>-<rule>.err` — SLURM-level errors
3. `experimentruns/{ds}/{run}/log.err` — Application-level errors
4. `python -m pipeline.db query "SELECT * FROM runs WHERE status='failed' ORDER BY started_at DESC LIMIT 5"` — Recent failures
