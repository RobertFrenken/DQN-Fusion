# Snakemake Migration TODO List

Comprehensive checklist for migrating from the custom `can-train` CLI to Snakemake.

## Progress Tracker

- [ ] **Phase 1: Environment Setup** (1-2 hours)
- [ ] **Phase 2: Code Modifications** (4-8 hours) ⚠️ CRITICAL
- [ ] **Phase 3: Testing** (2-4 hours)
- [ ] **Phase 4: Validation** (1-2 days)
- [ ] **Phase 5: Full Migration** (ongoing)

---

## Phase 1: Environment Setup

### 1.1 Install Snakemake
- [ ] Install Snakemake in conda environment
  ```bash
  conda activate base
  conda install -c conda-forge snakemake
  # Or create dedicated environment
  conda create -n snakemake -c conda-forge snakemake
  ```
- [ ] Verify installation: `snakemake --version`
- [ ] Test basic Snakemake functionality
  ```bash
  snakemake --help
  ```

### 1.2 Configure SLURM Profile
- [ ] Make status script executable
  ```bash
  chmod +x profiles/slurm/slurm-status.py
  ```
- [ ] Test status script
  ```bash
  # Submit a test job first
  sbatch --wrap "sleep 60" --account=PAS3209 --partition=gpu
  # Get job ID, then test
  python profiles/slurm/slurm-status.py <job_id>
  ```
- [ ] Update email in `config/snakemake_config.yaml`
- [ ] Verify SLURM account and partition access
  ```bash
  sacctmgr show assoc where user=$USER
  sinfo -o "%20P %5a %10l %6D"
  ```

### 1.3 Create Backup
- [ ] Create git branch for backup
  ```bash
  git checkout -b pre-snakemake-backup
  git add -A
  git commit -m "Backup before Snakemake migration"
  git checkout main
  ```
- [ ] Backup key files
  ```bash
  cp train_with_hydra_zen.py train_with_hydra_zen.py.backup
  cp -r src/cli src/cli.backup
  ```

---

## Phase 2: Code Modifications ⚠️ CRITICAL

### 2.1 Modify Training Entry Point

**File:** `train_with_hydra_zen.py`

- [ ] **Add simple CLI parser for Snakemake**
  ```python
  def create_snakemake_parser():
      """Simple parser for Snakemake invocation."""
      parser = argparse.ArgumentParser()
      parser.add_argument('--model', required=True, choices=['vgae', 'gat', 'dqn'])
      parser.add_argument('--model-size', required=True, choices=['teacher', 'student'])
      parser.add_argument('--dataset', required=True)
      parser.add_argument('--modality', required=True)
      parser.add_argument('--training', required=True)
      parser.add_argument('--learning-type', required=True)
      parser.add_argument('--teacher-path', default=None)
      parser.add_argument('--vgae-path', default=None)
      parser.add_argument('--gat-path', default=None)
      parser.add_argument('--output-dir', required=True)
      return parser
  ```

- [ ] **Add dual-mode detection in main()**
  ```python
  def main():
      if '--frozen-config' in sys.argv:
          # Legacy: frozen config from can-train
          run_with_frozen_config()
      else:
          # New: simple args from Snakemake
          parser = create_snakemake_parser()
          args = parser.parse_args()
          config = build_config_from_simple_args(args)
          train(config)
  ```

- [ ] **Implement `build_config_from_simple_args()`**
  - Map simple arguments to full config structure
  - Handle teacher paths for distillation
  - Handle VGAE/GAT paths for fusion
  - Set output directory correctly

- [ ] **Test CLI works standalone**
  ```bash
  python train_with_hydra_zen.py \
      --model vgae \
      --model-size teacher \
      --dataset hcrl_sa \
      --modality automotive \
      --training autoencoder \
      --learning-type unsupervised \
      --output-dir test_output
  ```

### 2.2 Standardize Model Output Paths

**Files:** Training code (likely in `src/training/` or model-specific files)

- [ ] **Find where models are saved**
  ```bash
  grep -r "torch.save" src/
  grep -r "save_checkpoint" src/
  ```

- [ ] **Ensure models saved with predictable names**
  - Format: `{model}_{model_size}_{distillation}_best.pth`
  - Example: `vgae_teacher_no_distillation_best.pth`
  - Location: `{output_dir}/models/`

- [ ] **Also save `last.ckpt` for resume**
  - Location: `{output_dir}/checkpoints/last.ckpt`

- [ ] **Verify paths match Snakefile expectations**
  - Check `get_model_path()` function in Snakefile
  - Ensure output paths align

### 2.3 Verify Evaluation Script

**File:** `src/evaluation/evaluation.py`

- [ ] **Check evaluation accepts required args**
  - `--dataset`
  - `--model-path`
  - `--training-mode`
  - `--csv-output`
  - `--json-output`
  - For DQN: `--vgae-path`, `--gat-path`

- [ ] **Test evaluation standalone**
  ```bash
  # Test with existing model if available
  python -m src.evaluation.evaluation \
      --dataset hcrl_sa \
      --model-path experimentruns/.../vgae_best.pth \
      --training-mode autoencoder \
      --csv-output test_results.csv \
      --json-output test_results.json
  ```

- [ ] **Verify outputs are created at specified paths**

### 2.4 Update Imports and Dependencies

- [ ] **Check all imports still work**
  ```bash
  python -c "import src.evaluation.evaluation"
  python -c "from src.config.frozen_config import save_frozen_config"
  ```

- [ ] **Ensure no circular dependencies**

- [ ] **Verify conda environment has all packages**
  ```bash
  conda activate gnn-experiments
  python -c "import torch; import pytorch_lightning; import hydra"
  ```

---

## Phase 3: Testing

### 3.1 Dry Run Tests

- [ ] **Test basic dry run**
  ```bash
  snakemake --profile profiles/slurm -n
  ```
  - Should show all rules that will be executed
  - No errors about missing rules or files

- [ ] **Test DAG generation**
  ```bash
  snakemake --dag | dot -Tpdf > pipeline_dag.pdf
  ```
  - Open PDF and verify pipeline structure
  - VGAE → GAT → DQN dependencies visible
  - Student models depend on teachers

- [ ] **Test specific target dry run**
  ```bash
  snakemake --profile profiles/slurm -n \
      results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json
  ```
  - Should show all upstream dependencies

### 3.2 Single Job Tests

- [ ] **Test VGAE training (simplest rule)**
  ```bash
  snakemake --profile profiles/slurm \
      experimentruns/automotive/hcrl_sa/vgae/teacher/no_distillation/autoencoder/models/vgae_teacher_no_distillation_best.pth \
      --jobs 1
  ```
  - Check job submitted: `squeue -u $USER`
  - Monitor logs: `tail -f experimentruns/.../logs/training.log`
  - Verify model created after completion
  - Check model size is reasonable (not 0 bytes)

- [ ] **Test GAT training (tests dependency)**
  ```bash
  # Delete GAT model if exists
  rm -f experimentruns/automotive/hcrl_sa/gat/teacher/no_distillation/curriculum/models/gat_teacher_no_distillation_best.pth

  # Run GAT
  snakemake --profile profiles/slurm \
      experimentruns/automotive/hcrl_sa/gat/teacher/no_distillation/curriculum/models/gat_teacher_no_distillation_best.pth \
      --jobs 1
  ```
  - Should ensure VGAE exists first
  - Verify dependency handling

- [ ] **Test DQN fusion (tests multiple inputs)**
  ```bash
  snakemake --profile profiles/slurm \
      experimentruns/automotive/hcrl_sa/dqn/teacher/no_distillation/fusion/models/dqn_teacher_no_distillation_best.pth \
      --jobs 1
  ```
  - Should wait for both VGAE and GAT
  - Check both model paths passed correctly

- [ ] **Test evaluation**
  ```bash
  snakemake --profile profiles/slurm \
      results/automotive/hcrl_sa/teacher/no_distillation/evaluation/vgae_eval.json \
      --jobs 1
  ```
  - Verify JSON and CSV created
  - Check metrics look reasonable

### 3.3 Student Model Tests

- [ ] **Test student VGAE with KD**
  ```bash
  snakemake --profile profiles/slurm \
      experimentruns/automotive/hcrl_sa/vgae/student/with_kd/autoencoder/models/vgae_student_with_kd_best.pth \
      --jobs 1
  ```
  - Should ensure teacher exists first
  - Verify teacher path passed to training

- [ ] **Verify student waits for teacher**
  ```bash
  # Check DAG shows dependency
  snakemake --dag experimentruns/automotive/hcrl_sa/vgae/student/with_kd/autoencoder/models/vgae_student_with_kd_best.pth \
      | dot -Tpdf > student_dag.pdf
  ```

### 3.4 Multi-Job Tests

- [ ] **Test parallel execution**
  ```bash
  # Run teacher pipeline for 2 datasets in parallel
  snakemake --profile profiles/slurm \
      results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json \
      results/automotive/hcrl_ch/teacher/no_distillation/evaluation/dqn_eval.json \
      --jobs 6
  ```
  - Check multiple jobs running: `squeue -u $USER`
  - Verify no dependency conflicts

- [ ] **Test failure recovery**
  ```bash
  # Cancel a running job
  scancel <job_id>

  # Resume
  snakemake --profile profiles/slurm --jobs 20 --rerun-incomplete
  ```
  - Should detect incomplete job
  - Should rerun only failed job

---

## Phase 4: Validation

### 4.1 Compare Results with Old Pipeline

- [ ] **Run same experiment with old pipeline**
  ```bash
  ./can-train pipeline \
      --modality automotive \
      --model vgae,gat,dqn \
      --learning-type unsupervised,supervised,rl_fusion \
      --training-strategy autoencoder,curriculum,fusion \
      --dataset hcrl_sa \
      --model-size teacher \
      --distillation no-kd \
      --submit
  ```

- [ ] **Run same experiment with Snakemake**
  ```bash
  snakemake --profile profiles/slurm \
      results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json \
      --jobs 3
  ```

- [ ] **Compare outputs**
  - Model file sizes similar
  - Evaluation metrics similar (within tolerance)
  - Training time similar

- [ ] **Document any differences**
  - Note in validation log
  - Investigate if significant

### 4.2 Full Pipeline Test

- [ ] **Run all teachers across all datasets**
  ```bash
  snakemake --profile profiles/slurm all_teachers --jobs 20
  ```
  - Monitor progress: `watch -n 10 'squeue -u $USER'`
  - Check for failures
  - Verify all models created

- [ ] **Run all students (depends on teachers)**
  ```bash
  snakemake --profile profiles/slurm all_students --jobs 20
  ```
  - Verify teachers used for distillation
  - Check student models smaller than teachers

- [ ] **Run complete pipeline**
  ```bash
  snakemake --profile profiles/slurm --jobs 20
  ```
  - Should run teachers then students
  - All evaluations should complete

### 4.3 Edge Case Testing

- [ ] **Test with missing intermediate files**
  ```bash
  # Delete intermediate GAT model
  rm -f experimentruns/automotive/hcrl_sa/gat/teacher/no_distillation/curriculum/models/gat_teacher_no_distillation_best.pth

  # Run DQN (should retrain GAT)
  snakemake --profile profiles/slurm \
      results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json \
      --jobs 3
  ```

- [ ] **Test with corrupted model file**
  ```bash
  # Corrupt file
  echo "corrupted" > experimentruns/.../vgae_teacher_no_distillation_best.pth

  # Try to run downstream
  snakemake --profile profiles/slurm \
      experimentruns/.../gat_teacher_no_distillation_best.pth \
      --jobs 1
  ```
  - Should use existing (corrupted) file
  - Use `--forcerun` to regenerate

- [ ] **Test directory cleanup and rerun**
  ```bash
  # Clean everything
  snakemake clean_all

  # Rerun from scratch
  snakemake --profile profiles/slurm --jobs 20
  ```

---

## Phase 5: Full Migration

### 5.1 Documentation

- [ ] **Update README.md**
  - Add Snakemake usage section
  - Link to SNAKEMAKE_QUICKSTART.md
  - Add quick examples

- [ ] **Create troubleshooting guide**
  - Common errors and solutions
  - SLURM-specific issues
  - Debugging tips

- [ ] **Document configuration options**
  - How to add new datasets
  - How to modify SLURM resources
  - How to customize pipeline

- [ ] **Add example workflows**
  - Training single dataset
  - Running sweeps
  - Evaluation only

### 5.2 Deprecation

- [ ] **Mark old scripts as deprecated**
  ```bash
  mkdir -p legacy
  git mv student_kd.sh legacy/
  git mv run_pipeline.sh legacy/
  ```

- [ ] **Add deprecation notice to can-train**
  ```python
  # In src/cli/main.py
  print("WARNING: can-train CLI is deprecated. Use Snakemake instead.")
  print("See SNAKEMAKE_QUICKSTART.md for details.")
  ```

- [ ] **Keep can-train for backward compatibility (optional)**
  - Useful for quick one-off experiments
  - Leave frozen config system functional

### 5.3 Cleanup

- [ ] **Remove unused code**
  - [ ] Archive `src/cli/job_manager.py` (no longer needed)
  - [ ] Consider removing frozen config if not used by anything else
  - [ ] Remove old SLURM script templates

- [ ] **Clean up experiment outputs**
  ```bash
  # Remove old checkpoint files
  snakemake clean_checkpoints

  # Remove old logs
  snakemake clean_logs
  ```

- [ ] **Update .gitignore**
  ```gitignore
  # Snakemake
  .snakemake/
  slurm_logs/
  *.pyc
  __pycache__/
  ```

### 5.4 Team Communication

- [ ] **Notify team of migration**
  - Send migration guide
  - Schedule training session if needed
  - Set migration timeline

- [ ] **Update lab wiki/docs**
  - Add Snakemake as standard workflow
  - Link to resources
  - Add examples

- [ ] **Archive old job IDs**
  - Document old pipeline run IDs for reference
  - Save old results for comparison

---

## Phase 6: Ongoing Maintenance

### 6.1 Monitoring

- [ ] **Set up automated monitoring** (optional)
  - Email notifications on completion
  - Slack integration for failures
  - Resource usage tracking

- [ ] **Create dashboard** (optional)
  - Use Snakemake's built-in reporting
  - Track experiment status

### 6.2 Optimization

- [ ] **Profile resource usage**
  ```bash
  snakemake --profile profiles/slurm --benchmark-extended --jobs 20
  ```

- [ ] **Optimize SLURM resources per rule**
  - Adjust time limits based on actual usage
  - Tune memory requirements
  - Right-size GPU types

- [ ] **Add caching** (if beneficial)
  - Cache expensive operations
  - Reuse common computations

### 6.3 Extensions

- [ ] **Add new rules as needed**
  - Hyperparameter sweeps
  - Ablation studies
  - Cross-validation

- [ ] **Integrate with MLflow** (if not already)
  - Log experiments from Snakemake
  - Track metrics automatically

- [ ] **Add visualization rules**
  - Generate plots automatically
  - Create comparison figures
  - Summary dashboards

---

## Success Criteria

Migration is considered successful when:

- [ ] All datasets can be trained via Snakemake
- [ ] Results match old pipeline (within tolerance)
- [ ] Dependency management works automatically
- [ ] Failures recover gracefully
- [ ] Documentation is complete
- [ ] Team is trained and comfortable with new system
- [ ] Old pipeline can be deprecated

---

## Emergency Rollback

If critical issues arise:

- [ ] **Switch back to can-train**
  ```bash
  git checkout pre-snakemake-backup
  ./run_pipeline.sh  # Old method
  ```

- [ ] **Document issue**
  - What failed
  - Error messages
  - Attempted solutions

- [ ] **Report to team**
  - Timeline for fix
  - Impact assessment
  - Alternative plans

---

## Notes

- Estimated total migration time: **1-2 weeks**
- Critical path: Modifying `train_with_hydra_zen.py` (Phase 2.1)
- Highest risk: Model path mismatches causing dependency failures
- Mitigation: Extensive testing in Phase 3

**Last Updated:** 2026-01-28
