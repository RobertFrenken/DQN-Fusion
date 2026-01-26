Modality in training script - Need to verify train_with_hydra_zen.py accepts --modality argument
Dependency paths - Stage 2 (curriculum) needs VGAE checkpoint path, Stage 3 (fusion) needs classifier path
Per-stage SLURM overrides - Currently all stages use same walltime/memory