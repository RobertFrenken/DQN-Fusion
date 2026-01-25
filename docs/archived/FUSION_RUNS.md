🔧 Fusion smoke run checklist

1) Required artifacts (fusion training expects pure state_dicts):
   - VGAE autoencoder state dict at:
     `experimentruns/<modality>/<dataset>/unsupervised/vgae/teacher/<distillation>/autoencoder/vgae_autoencoder.pth`
   - GAT classifier state dict at:
     `experimentruns/<modality>/<dataset>/supervised/gat/teacher/<distillation>/normal/gat_<dataset>_normal.pth`
   - The fusion agent will be saved as a dict in
     `experimentruns/.../rl_fusion/<...>/fusion/models/fusion_agent_<dataset>.pth`

2) If you need to re-create teacher artifacts for smoke runs:
   - Use `scripts/local_smoke_experiment.py --model vgae --training autoencoder --use-synthetic-data --run --write-summary`
   - Use `scripts/local_smoke_experiment.py --model gat --training normal --use-synthetic-data --run --write-summary`

3) Reproducibility:
   - Exported conda env: `envs/gnn-experiments.yml` (created from `conda env export -n gnn-experiments --no-builds`).
   - Important: Ensure `torch` and `torch-geometric` versions in this YAML are compatible with your cluster.

4) Fast validation tools added:
   - `scripts/inspect_artifacts.py` — checks artifact existence and whether they look like state_dicts
   - `scripts/validate_fusion_agent.py` — loads fusion caches and agent and runs a small validation; run via `conda run -n gnn-experiments python scripts/validate_fusion_agent.py`

5) Small sweep example:
   - `jobs/smoke_fusion_sweep.json` and `scripts/run_fusion_sweep.sh` (example sweep using `oscjobmanager.py` preview/submit)

Notes:
- The fusion agent checkpoint may contain numpy arrays that require `torch.load(..., weights_only=False)` to fully inspect. Use caution: setting `weights_only=False` may execute arbitrary code on untrusted checkpoints.
