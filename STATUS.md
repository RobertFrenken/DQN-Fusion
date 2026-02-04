# KD-GAT Pipeline Status

Last updated: 2026-02-04

## Bugs Fixed

### 1. Silent teacher corruption via `strict=False` (HIGH)
**File:** `pipeline/stages/utils.py` `load_teacher()`
**Symptom:** KD training runs to completion but produces garbage results.
**Root cause:** `load_state_dict(strict=False)` silently ignores weight dimension
mismatches. If the teacher config failed to parse, the code fell back to the student's
config, built a teacher with wrong dimensions, and loaded whatever weights happened to
match. Unmatched layers stayed randomly initialized.
**Fix:** Removed all `strict=False` from teacher loading. Made missing teacher config a
hard `FileNotFoundError` instead of a silent fallback.

### 2. Validation didn't check teacher files exist (MEDIUM)
**File:** `pipeline/validate.py`
**Symptom:** Jobs wait in SLURM queue, start, process data, then crash on
`torch.load()` when the teacher checkpoint doesn't exist.
**Root cause:** Validation only checked that `teacher_path` was non-empty, not that the
file was actually on disk.
**Fix:** Added existence checks for both the teacher checkpoint and its `config.json`.

### 3. Validation didn't check frozen configs for prerequisites (MEDIUM)
**File:** `pipeline/validate.py`
**Symptom:** Curriculum or fusion stage crashes with `FileNotFoundError` from
`load_frozen_cfg()` after passing validation.
**Root cause:** Validation checked that `best_model.pt` existed for prerequisites but
not the accompanying `config.json`.
**Fix:** Added `config.json` existence checks alongside checkpoint checks for all
prerequisite stages.

### 4. DQN teacher used student's `alpha_steps` (LOW — hidden by identical defaults)
**File:** `pipeline/stages/utils.py` `load_teacher()` DQN branch
**Root cause:** Used `cfg.alpha_steps` (student config) instead of `tcfg.alpha_steps`
(teacher's frozen config). Currently both default to 21, so this never crashed.
**Fix:** Changed to `tcfg.alpha_steps`.

### 5. `share_memory_()` exceeded mmap limit (CRITICAL — previous session)
**Files:** `pipeline/stages/utils.py`, `pipeline/stages/modules.py`,
`src/training/datamodules.py`
**Symptom:** `RuntimeError: unable to mmap N bytes: Cannot allocate memory (12)`
**Root cause:** `share_memory_()` creates one mmap entry per tensor. With 150K+ graphs
x 3 tensors, it exceeds `vm.max_map_count` (65530) during the sharing itself.
**Fix:** Replaced with `num_workers=0` fallback when tensor count > 60000.

### 6. Double `_kd` suffix in path construction (CRITICAL — previous session)
**Files:** `pipeline/stages/fusion.py`, `pipeline/stages/evaluation.py`
**Root cause:** Stage names like `"autoencoder_kd"` were passed to `run_id()` which
also appends `_kd` when `cfg.use_kd=True`, producing `student_autoencoder_kd_kd`.
**Fix:** Changed to bare stage names (`"autoencoder"`, `"curriculum"`).

### 7. Silent config fallback for missing frozen configs (previous session)
**File:** `pipeline/stages/utils.py` `load_frozen_cfg()`
**Root cause:** Missing `config.json` silently fell back to current `cfg`, which could
have wrong architecture dimensions (e.g., teacher VGAE dims for a student model).
**Fix:** Changed to raise `FileNotFoundError`.

### 8. MLflow `end_run()` crash on nested dicts
**File:** `pipeline/tracking.py`
**Root cause:** Evaluation returns nested dicts (`{"gat": {"core": {...}}}`).
`mlflow.log_metrics()` requires flat `{str: float}`.
**Fix:** Filter to only log flat numeric values.

### 9. `dqn_hidden` and `dqn_layers` were dead config params (HIGH)
**File:** `src/models/dqn.py`
**Root cause:** `QNetwork` hardcoded `hidden_dim=128` with 4 layers. Config values
`dqn_hidden=576/160` and `dqn_layers=3/2` were never passed to the constructor.
Teacher and student DQN were architecturally identical.
**Fix:** `QNetwork` now accepts `hidden_dim` and `num_layers`. `EnhancedDQNFusionAgent`
passes these through. `fusion.py` and `load_teacher` wire config values.
**Breaking:** Deleted old fusion checkpoints (hcrl_sa teacher_fusion, student_fusion_kd)
trained with wrong architecture. These will retrain with correct dims.

### 10. DQN agent silently overrode batch_size and buffer_size on GPU
**File:** `src/models/dqn.py`
**Root cause:** `batch_size = max(batch_size, 8192)` and `buffer_size = max(buffer_size, 200000)`
on CUDA, ignoring config values entirely.
**Fix:** Removed overrides. Config values are used as-is.

### 11. Redundant CSV file discovery using different algorithm
**File:** `src/training/datamodules.py`
**Root cause:** `_process_dataset_from_scratch()` used `glob.glob()` to find CSVs,
then called `graph_creation()` which used `os.walk()` + `find_csv_files()` internally.
The two algorithms could find different files (glob didn't filter EXCLUDED_ATTACK_TYPES).
**Fix:** Replaced glob-based discovery with call to `find_csv_files()` — same function
that `graph_creation` uses internally.

### 12. `_MMAP_LIMIT` constant duplicated in 3 files
**Files:** `pipeline/stages/utils.py`, `src/training/datamodules.py`
**Root cause:** Same constant (60000) defined independently in multiple files.
**Fix:** Single `MMAP_TENSOR_LIMIT` constant in `src/training/datamodules.py`, imported
by `pipeline/stages/utils.py`.

### 13. Evaluation DQN agent missing `hidden_dim`, `num_layers`, `alpha_steps` (HIGH)
**File:** `pipeline/stages/evaluation.py`
**Root cause:** `EnhancedDQNFusionAgent()` in evaluation was not updated when Issue #9
fixed the same constructor call in `fusion.py`. The evaluation instantiation used
default `hidden_dim=128, num_layers=3`, causing a `RuntimeError: size mismatch` when
loading trained checkpoints (teacher: 576/3, student: 160/2).
**Fix:** Added `alpha_steps=fusion_cfg.alpha_steps`, `hidden_dim=fusion_cfg.dqn_hidden`,
`num_layers=fusion_cfg.dqn_layers` to match the working call in `fusion.py`.

### 14. Validation rejects `use_kd=True` for evaluation stage (HIGH)
**File:** `pipeline/validate.py`
**Symptom:** All `student_evaluation_kd` jobs fail immediately with
`ValueError: use_kd=True but teacher_path is empty`.
**Root cause:** The `teacher_path` check at line 40 fired unconditionally for all stages.
Evaluation doesn't use the teacher — `use_kd` only affects path resolution to find
`student_*_kd/` directories. The Snakefile passes `--use-kd true` without
`--teacher-path`, so validation rejected the config before evaluation could start.
**Fix:** Added `stage != "evaluation"` guard to the `teacher_path` check.

## Known Limitations (not bugs, but worth knowing)

### Frozen configs contain irrelevant architecture fields
A GAT-stage config saved via `--preset gat,student` has correct GAT dims but
teacher-sized VGAE dims (from defaults). The VGAE fields are unused by the GAT
stage, so this doesn't cause incorrect behavior, but the configs are misleading
as audit trails.

### Old-format teacher_autoencoder configs (hcrl_ch, set_01-04)
These are from older code and are missing ~30 fields. `PipelineConfig.load()`
fills missing fields with defaults. The VGAE architecture dims are correct. They
also have smaller `fusion_max_samples=10000` vs the standard `150000`.

### Dataset cache race condition
Multiple concurrent SLURM jobs processing the same dataset can race on temp file
writes during cache creation. Non-fatal: data stays in memory and training
proceeds, but causes redundant re-processing.

### Curriculum epoch counter can drift
`CurriculumDataModule._current_epoch` increments each time
`train_dataloader()` is called, which Lightning may call during sanity checks.
The counter is not synced with the trainer's actual epoch.

## Checkpoint Status

### Fusion checkpoint audit (2026-02-04)

All 18 fusion checkpoints were inspected to verify their weights match the config's
`dqn_hidden` and `dqn_layers`. 17 of 18 are clean. One is stale from before Issue #9:

| Dataset  | teacher_fusion | student_fusion_kd | student_fusion |
|----------|:-:|:-:|:-:|
| hcrl_sa  | ok (576/3) | ok (160/2) | **STALE** (128/4 → needs 160/2) |
| hcrl_ch  | ok (576/3) | ok (160/2) | ok (160/2) |
| set_01   | ok (576/3) | ok (160/2) | ok (160/2) |
| set_02   | ok (576/3) | ok (160/2) | ok (160/2) |
| set_03   | ok (576/3) | ok (160/2) | ok (160/2) |
| set_04   | ok (576/3) | ok (160/2) | ok (160/2) |

Format: (hidden_dim/num_layers). "STALE" = checkpoint weights don't match config;
must be deleted and retrained.

**Action required:** Delete `experimentruns/hcrl_sa/student_fusion/best_model.pt`
and re-run the fusion training so Snakemake retrains it with correct architecture.
Evaluation for `hcrl_sa/student_evaluation` will then succeed.

## Integration Tests

Run: `python -m pytest tests/test_pipeline_integration.py -v`

24 tests covering:
- Config serialization round-trips (all 6 presets)
- Model construction matches config (VGAE, GAT, DQN teacher/student differ)
- Checkpoint save/load with strict=True (catches dimension mismatches)
- Teacher loading uses frozen config, not student config
- Missing teacher config raises FileNotFoundError (not silent fallback)
- Path construction matches Snakefile _p() helper exactly
- Validation catches missing teacher checkpoints, configs, and prerequisites
- Frozen config propagation (curriculum gets student VGAE dims, not teacher)
- MMAP constant is single source of truth
