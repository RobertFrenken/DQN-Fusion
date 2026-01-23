# Jobs Workflow: Spec-driven submission & chaining

Overview
--------
We follow a config-driven, programmatic job submission approach. Jobs are defined as JSON files in `jobs/` and executed through `scripts/drive_jobs.py` which uses `OSCJobManager` to generate SBATCH scripts and submit them to SLURM.

Files
-----
- `jobs/*.json` — Example job specs (vgae, gat, fusion, pipeline).
- `scripts/drive_jobs.py` — Driver that reads a job spec, generates scripts, optionally submits and chains jobs.
- `slurm_jobs/` — Generated scripts.
- `slurm_logs/` — Logs produced by job runs.
- `experiment_runs/` — Canonical artifact directory (default). Override by exporting `CAN_EXPERIMENT_ROOT` in your environment or in the SBATCH header.

Job spec format
---------------
A minimal job spec looks like:

```json
{
  "name": "my_job",
  "datasets": "all",
  "training_types": ["autoencoder"],
  "extra_args": {"epochs": 5, "batch_size": 16},
  "dependency_manifests": {"default": "/project/manifests/{dataset}_manifest.json"}
}
```

- `datasets`: list of dataset names or the string `"all"` (expands to automotive datasets).
- `training_types`: list of training stages (`autoencoder`, `curriculum`, `fusion`, etc.).
- `extra_args`: CLI overrides passed to `train_with_hydra_zen.py` (e.g., `epochs`, `batch_size`, `teacher_path`, `dependency_manifest`).
- `dependency_manifests` (optional): per-dataset manifest path template. Use `{dataset}` placeholder.

Commands
--------
- Generate scripts only (no submission):

  ```bash
  python scripts/drive_jobs.py --job jobs/example_vgae.json --generate-only
  ```

- Submit jobs (no chaining):

  ```bash
  python scripts/drive_jobs.py --job jobs/example_vgae.json --submit
  ```

- Submit and chain pipeline per dataset (autoencoder -> curriculum -> fusion):

  ```bash
  python scripts/drive_jobs.py --job jobs/example_pipeline.json --submit --chain
  ```

Notes
-----
- Fusion jobs require pre-trained artifacts; use `dependency_manifests` to point to manifests that validate and pin artifacts.
- You can custom-edit generated scripts in `slurm_jobs/` before submitting if you use `--generate-only` first.
- Monitor with `squeue` or use `OSCJobManager.monitor_jobs(job_ids)` programmatically.

If you'd like, I can:
- Add example manifest files under `docs/examples/manifests/` for `hcrl_sa` and `hcrl_ch`, or
- Add an automated chaining helper that submits autoencoder jobs and waits for completion (polling) before submitting curriculum and fusion (more robust than SBATCH dependency if you want cross-queue control).

Which next? (manifests / polling-chaining)