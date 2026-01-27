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



I think with the frozen configs, I should be able to cycle through all possible permutations and make sure that the pathing is correct. I think the training logic seems okay right now. But the paths need to be good, the logging needs to be in its correct spot, and then I need to drive down the combinations to make sure all the validation handlers are not screwing up.

1.26.26: If this new CLI interface can work, then I should be able to run the teacher and student-no-kd pipelines, only leaving the student-with-kd for tomorrow to get at least 1 full run complete.

modality, dataset, model_size, learning_type, model_architecture,  use_distillation, training_strategy
automotive, hcrl_sa, teacher, unsupervvised, vgae, no-kd, autoencoder
automotive, hcrl_sa, teacher, supervvised, gat, no-kd, cirriculum
automotive, hcrl_sa, teacher, rl_fusion, dqn, no-kd, fusion
automotive, hcrl_sa, student, unsupervvised, vgae, no-kd, autoencoder
automotive, hcrl_sa, student, supervvised, gat, no-kd, cirriculum
automotive, hcrl_sa, student, rl_fusion, dqn, no-kd, fusion
automotive, hcrl_sa, teacher, unsupervvised, vgae, with-kd, autoencoder
automotive, hcrl_sa, teacher, supervvised, gat, with-kd, cirriculum
automotive, hcrl_sa, teacher, rl_fusion, dqn, no-kd, fusion

Look into using nvidia-smi to better diagnose the GPU usage

---

## 2026-01-27: Run Counter & Batch Size Config Implementation

**Completed:**
- ✅ Added `get_run_counter()` to PathResolver - increments model filename versioning
- ✅ Added `BatchSizeConfig` dataclass to frozen configs with 6 simple fields
- ✅ Updated trainer.py to log batch size decisions (tuner output, safety factor, final size)
- ✅ Model filenames now include run counter: `dqn_student_fusion_run_001.pth`, `run_002.pth`, etc.
- ✅ Created test frozen config for hcrl_sa with 10 epochs (faster testing)

**Design Decisions:**
- Run counter stored in `experiment_dir/run_counter.txt` - works across SLURM systems
- `tuned_batch_size` parameter captures successful batch sizes for feedback loop
- `safety_factor` pre-baked into each frozen config (no conditional overrides)
- Each run type (curriculum/kd/fusion) gets independent config with appropriate safety_factor
- Wall time strategy: 20-25 min for quick test jobs (10 epochs), 60+ min for full training

**Batch Size Feedback Loop:**
```
Run 1: tuner finds 192, apply 0.55 → use 105 ✅
  → Save: tuned_batch_size=105 to frozen config
Run 2: Load tuned_batch_size=105, skip tuner (faster) ✅
  → Can re-tune or use directly based on optimize_batch_size flag
Run 3+: Same pattern with different random seed
```

**Logging Output:**
- `🔢 Run number: 001` - identifies which run
- `🔧 Running batch size optimization...` - shows tuner starting
- `📊 Tuner found max safe batch size: 192` - reports tuning result
- `🎯 Applied safety_factor 0.55: 192 × 0.55 = 105` - shows calculation
- `✅ Training batch size: 105` - confirms final value
- `🔄 Updated batch_size_config: tuned_batch_size = 105` - saves feedback

**Status:** Ready for testing; SLURM job 43977157 submitted (pending)

**GPU Monitoring Added (2026-01-27 15:15):**
- ✅ SLURM script now logs GPU metrics every 2 seconds to CSV
- ✅ Created `analyze_gpu_monitor.py` to parse and visualize GPU data
- ✅ Reports: memory peak, growth rate (leak detection), utilization patterns
- ✅ Classifies bottleneck: compute-bound vs memory-bound vs data-starved

**Why GPU monitoring matters for your batch size tuning:**
1. Validates safety_factor is appropriate (peak memory should be 60-75% of GPU)
2. Detects memory leaks during validation/checkpointing
3. Confirms GPU utilization is healthy (70%+ indicates good compute utilization)
4. Tracks if tuned_batch_size produces consistent memory across runs

**After job 43977157 completes, run:**
```bash
python analyze_gpu_monitor.py gpu_monitor_43977157.csv
```

This will generate:
- `gpu_monitor_43977157_analysis.png` - 3-panel memory/utilization/growth plots
- Console output with memory leak detection and bottleneck classification
- Recommendations for next tuning iteration