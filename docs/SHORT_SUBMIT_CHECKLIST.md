# Short OSC Submission Checklist

Use this checklist to run experiments on OSC safely and reproducibly.

Pre-flight (one-time)
- [ ] Create / verify conda env: `conda create -n gnn-gpu python=3.10` + `conda activate gnn-gpu` and `pip install -r requirements.txt`
- [ ] Decide experiment root (e.g., `/scratch/$USER/experiment_runs`) and ensure it is writable
- [ ] If running fusion later, generate manifests and stage them on shared FS: `python scripts/generate_manifests.py --experiment-root /scratch/$USER/experiment_runs --datasets hcrl_sa hcrl_ch`

Per-job (before submitting)
- [ ] Generate SBATCH scripts for the job spec (no submit):
  - `python scripts/drive_jobs.py --job jobs/example_pipeline.json --generate-only`
- [ ] Inspect the generated script in `slurm_jobs/` (look for `CAN_EXPERIMENT_ROOT` export and the `python train_with_hydra_zen.py` command)
- [ ] Ensure manifests and teacher artifacts exist (or use `--dependency-manifest` pointing to generated manifest in `docs/examples/manifests/generated/`)
- [ ] Run validation via the driver (driver runs validation automatically when submitting). If validation fails, fix errors before submission.

Submit & monitor
- [ ] Submit and chain (per-dataset pipeline):
  - `python scripts/drive_jobs.py --job jobs/example_pipeline.json --submit --chain`
- [ ] Check job status: `squeue -u $USER | grep can_`
- [ ] Tail logs: `tail -f slurm_logs/<script_prefix>_<jobid>.out`

Post-run
- [ ] Check artifacts under `experiment_runs/` (or your `CAN_EXPERIMENT_ROOT`)
- [ ] View MLflow: `mlflow ui --backend-store-uri experiment_runs/.mlruns`
- [ ] If a job failed, inspect the job metadata sidecar: `slurm_jobs/<script>.meta.json` for submission error or job id

Troubleshooting hints
- If `conda activate` fails inside sbatch, replace with `source /path/to/miniconda3/bin/activate gnn-gpu` in the generated script.
- If a fusion job fails due to missing artifacts, either run the teacher jobs first or provide a dependency manifest that points to valid artifacts.

