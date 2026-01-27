Modality in training script - Need to verify train_with_hydra_zen.py accepts --modality argument
Dependency paths - Stage 2 (curriculum) needs VGAE checkpoint path, Stage 3 (fusion) needs classifier path
Per-stage SLURM overrides - Currently all stages use same walltime/memory

Claude keeps forgetting the CLI arguments and their dependencies. Need to provide it some sort of lookup


./can-train pipeline --model vgae,gat,dqn --learning-type unsupervised,supervised,rl_fusion --mode autoencoder,curriculum,fusion --dataset hcrl_sa --distillation with-kd --submit 

I keep finding myself needing to rework the file pathing, I am making an experiments.csv to help me figure out all the configurations I want to run.

Main Table Results:
2 pipelines [teacher, student], 3 models per pipeline [gat, vgae, dqn], and there are 6 datasets [hcrl_sa, hcrl_ch, set_01, set_02, set_03, set_04] for a totla 36 models that need to be run for the main table.

Ablations: 
- Knowledge distillation: [GAT, VGAE] x [with-KD, no-KD] x  2 datasets = 8 unique models, 4 pulled from the main experiments 4 will need to be run independent of experiments
- GAT training: [curriculum, random undersampling fixed ratio, VGAE undersampling fixed ratio] x 2 datasets = 6 unique models. 2 from main experiments 4 will need to be run independent of main experiments.
- DQN Fusion: [Fusion, Fixed alpha, GAT only, VGAE only] = no new training but will need an evaluation file.

Total needed experiments: 36 main + 8 ablations + unique DQN evaluation = 46 experiment runs.


| Dataset      | Safety Factor | Initial BS | Tuner BS | Final BS | KD Status | Status     |
| ------------ | ------------- | ---------- | -------- | -------- | --------- | ---------- |
| hcrl_ch      | 0.6           | 64         | 7782     | 4669     | DISABLED  | ✅ Complete |
| hcrl_sa      | 0.55          | 64         | 7782     | 4280     | DISABLED  | ✅ Complete |
| set_01       | 0.55          | 64         | 7782     | 4280     | DISABLED  | ✅ Complete |
| set_02/03/04 | 0.35          | 64         | 7782     | 2723     | DISABLED  | ✅ Complete |

| Aspect            | Your Current State       | Recommendation                              |
|-------------------|--------------------------|---------------------------------------------|
| ConfigStore       | CANGraphConfigStore ✓    | Keep – excellent pattern                    |
| Validation        | Multiple places          | Consolidate to single Pydantic v2 layer     |
| Path Resolution   | PathResolver ✓           | Extend with manifest pattern                |
| SLURM Integration | Argument passing         | Refactor to frozen config file              |