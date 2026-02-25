# KD-GAT SLURM / HPC Conventions

## Environment

- **Cluster**: OSC Pitzer (Ohio Supercomputer Center), RHEL 9, SLURM
- **GPU**: 2x V100 per node, ~362 GB RAM (account PAS3209, gpu partition)
- **Python**: 3.12 via `module load python/3.12`, uv venv `.venv/`
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Node.js**: 22.12.0 via `module load node-js/22.12.0` (docs-site only)

## Rules

- Spawn/fork CUDA rule: See critical-constraints.md.
- Test on small datasets (`hcrl_ch`) before large ones (`set_02`+).
- SLURM logs go to `slurm_logs/`, experiment outputs to `experimentruns/`.
- Heavy tests use `@pytest.mark.slurm` — auto-skipped on login nodes.
- **Always run tests via SLURM** (`cpu` partition, 8 CPUs, 16GB). Submit with `bash scripts/run_tests_slurm.sh`.

## Login Node Safety

**Safe on login node:**
- Import checks: `python -c "from module import func; print('OK')"`
- Exports: `python -m pipeline.export`
- DuckDB rebuild: `python -m pipeline.build_analytics`
- Node.js: `npm install`, `npm run build`
- Git, DVC, W&B sync, ruff

**Must go through SLURM:**
- `python -m pipeline.cli <any stage>` — all training/evaluation
- `python -m pytest` — test suite
- Any script that imports and runs models
