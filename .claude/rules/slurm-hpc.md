# KD-GAT SLURM / HPC Conventions

## Environment

- **Cluster**: OSC (Ohio Supercomputer Center), RHEL 9, SLURM
- **GPU**: V100 (account PAS3209, gpu partition)
- **Python**: 3.12 via `module load python/3.12`, uv venv `.venv/`
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Node.js**: 22.12.0 via `module load node-js/22.12.0` (docs-site only)

## Rules

- Always use `spawn` multiprocessing, never `fork` with CUDA.
- Test on small datasets (`hcrl_ch`) before large ones (`set_02`+).
- SLURM logs go to `slurm_logs/`, experiment outputs to `experimentruns/`.
- Heavy tests use `@pytest.mark.slurm` — auto-skipped on login nodes.
- **Always run tests via SLURM** (`cpu` partition, 8 CPUs, 16GB). Submit with `bash scripts/run_tests_slurm.sh`.

## Login Node Safety

**Safe on login node:**
- Import checks: `python -c "from module import func; print('OK')"`
- Light exports: `python -m pipeline.export --skip-heavy`
- DuckDB rebuild: `python -m pipeline.build_analytics`
- Node.js: `npm install`, `npm run build`
- Git, DVC, W&B sync, ruff

**Must go through SLURM:**
- `python -m pipeline.cli <any stage>` — all training/evaluation
- `python -m pytest` — test suite
- `python -m pipeline.export --only-heavy` — UMAP, attention, graph samples
- Any script that imports and runs models
