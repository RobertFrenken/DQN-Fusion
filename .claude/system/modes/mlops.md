# Mode: MLOps

**Active mode**: MLOps — Pipeline execution, infrastructure, and operational tasks.

## Focus Areas

- Prefect flow execution and debugging
- SLURM job management (submission, monitoring, resource allocation)
- Config system maintenance (YAML composition, Pydantic validation)
- W&B tracking and experiment management
- Training failure triage (OOM, CUDA errors, NaN losses)
- Data pipeline health (preprocessing, caching)
- R2 lakehouse sync and DuckDB queries

## Context Files (prioritize reading these)

- `CLAUDE.md` — Full command reference and project structure
- `.claude/system/STATE.md` — Current pipeline state and experiment status
- `pipeline/flows/` — Prefect flow definitions (train_flow, eval_flow, slurm_config)
- `config/` — Schema, resolver, constants, YAML files
- `pipeline/stages/` — Stage implementations
- `pipeline/lakehouse.py` — R2 lakehouse sync
- `pipeline/cli.py` — CLI entry point + W&B lifecycle

## Suppressed Topics

Do NOT initiate discussion about:
- Research hypotheses (OOD generalization, JumpReLU, cascading KD)
- Paper writing or literature review
- Documentation site updates
- Unless the user explicitly asks

## Available Commands

| Command | Description |
|---------|-------------|
| `/run-pipeline <dataset> [scale]` | Submit Prefect flow to SLURM |
| `/check-status [dataset]` | Check SLURM queue, checkpoints, W&B |
| `/run-tests [pattern]` | Run pytest suite |
| `python -m pipeline.cli <stage> ...` | Run single stage directly |
| `python -m pipeline.cli flow --dataset <ds>` | Run full Prefect pipeline |
| `python -m pipeline.cli flow --eval-only` | Re-run evaluation only |
| `wandb sync wandb/run-*` | Sync offline W&B runs |

## Quick Diagnostics

When troubleshooting, check in this order:
1. `squeue -u $USER` — Are jobs running/pending?
2. `slurm_logs/<jobid>.{out,err}` — SLURM-level errors
3. `experimentruns/{ds}/{run}/log.{out,err}` — Application-level errors
4. W&B dashboard (project `kd-gat`) — Run metrics and status
