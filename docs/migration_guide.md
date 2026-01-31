# Pipeline Migration Guide

This guide documents the migration from the old 8-level directory structure to the new 2-level structure with MLflow tracking.

## Summary of Changes

### Before (8 levels)
```
experimentruns/automotive/hcrl_sa/teacher/unsupervised/vgae/no_distillation/autoencoder/
    ├── best_model.pt
    ├── lightning_logs/
    └── slurm.out/err
```

### After (2 levels)
```
experimentruns/hcrl_sa/teacher_autoencoder/
    ├── best_model.pt
    ├── config.json         # NEW: frozen configuration
    ├── logs/
    └── slurm.out/err
```

**Benefits:**
- 75% reduction in path depth (8 → 2 levels)
- Shorter, more readable paths
- Deterministic run IDs for Snakemake
- Easier navigation and file management
- MLflow tracking for all runs

## What Changed

### 1. Path Structure ([pipeline/paths.py](../pipeline/paths.py))

**Old `stage_dir()` function:**
```python
def stage_dir(cfg, stage):
    learning_type, model_arch, mode = STAGES[stage]
    distillation = "distilled" if cfg.use_kd else "no_distillation"
    return (
        Path(cfg.experiment_root) / cfg.modality / cfg.dataset
        / cfg.model_size / learning_type / model_arch
        / distillation / mode
    )
```

**New `stage_dir()` function:**
```python
def stage_dir(cfg, stage):
    return Path(cfg.experiment_root) / run_id(cfg, stage)

def run_id(cfg, stage):
    kd_suffix = "_kd" if cfg.use_kd else ""
    return f"{cfg.dataset}/{cfg.model_size}_{stage}{kd_suffix}"
```

### 2. MLflow Tracking ([pipeline/tracking.py](../pipeline/tracking.py))

**New module** that replaces custom SQLite registry:
- `start_run()`: Initialize MLflow run with deterministic names
- `end_run()`: Log metrics and completion status
- `log_failure()`: Handle errors gracefully
- Automatic parameter logging from `PipelineConfig`
- Teacher-student lineage tracking via tags

**Tracking database:** `/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db`

### 3. CLI Integration ([pipeline/cli.py](../pipeline/cli.py))

Added MLflow hooks around stage dispatch:

```python
# Start MLflow tracking
run_name = run_id(cfg, args.stage)
start_run(cfg, args.stage, run_name)

# Dispatch
try:
    result = STAGE_FNS[args.stage](cfg)
    end_run(result, success=True)
except Exception as e:
    log_failure(str(e))
    raise
```

### 4. Snakefile ([pipeline/Snakefile](../pipeline/Snakefile))

**Old `_p()` helper:**
```python
def _p(ds, size, learn, arch, distill, mode):
    return f"{EXP}/{MOD}/{ds}/{size}/{learn}/{arch}/{distill}/{mode}/best_model.pt"
```

**New `_p()` helper:**
```python
def _p(ds, size, stage, kd=False):
    suffix = "_kd" if kd else ""
    return f"{EXP}/{ds}/{size}_{stage}{suffix}/best_model.pt"
```

All 18 Snakemake rules updated to use new paths.

### 5. Query Tools

**New files:**
- [notebooks/query_mlflow.ipynb](../notebooks/query_mlflow.ipynb): Jupyter notebook for pandas-based queries
- [pipeline/query.py](../pipeline/query.py): CLI tool for querying MLflow data
- [docs/mlflow_usage.md](mlflow_usage.md): Complete MLflow usage guide

**Example queries:**
```bash
# Show all runs
python -m pipeline.query --all

# Filter by dataset and stage
python -m pipeline.query --dataset hcrl_sa --stage curriculum

# Show leaderboard
python -m pipeline.query --leaderboard --top 10

# Compare teacher vs student with KD
python -m pipeline.query --compare teacher student_kd
```

## Migration Instructions

### Step 1: Review Existing Experiments

Check what experiments exist:
```bash
python -m pipeline.migrate --dry-run
```

This shows:
- Number of old experiments found
- Old → new path mappings
- Any potential conflicts

### Step 2: Execute Migration

If dry run looks good:
```bash
python -m pipeline.migrate --execute
```

This will:
1. Move experiments from old 8-level to new 2-level paths
2. Backfill MLflow tracking (if config.json exists)
3. Preserve all checkpoint and log files

**Note:** Old experiments without `config.json` will be migrated (files moved) but won't have MLflow tracking data. This is expected for runs before the new system.

### Step 3: Verify Migration

Check new structure:
```bash
ls -la experimentruns/
# Should show dataset directories (hcrl_sa, set_01, etc.)

ls -la experimentruns/hcrl_sa/
# Should show run directories (teacher_autoencoder, student_curriculum_kd, etc.)
```

Check MLflow data:
```bash
python -m pipeline.query --all
```

### Step 4: Clean Up (Optional)

Once satisfied, remove old automotive directory:
```bash
rm -rf experimentruns/automotive
```

## Backwards Compatibility

**Breaking changes:**
- Old Snakefile rules won't work with new paths
- Custom scripts using old path format need updates
- MLflow tracking database is required for new runs

**Migration safety:**
- Existing checkpoints preserved
- No data loss (moves files, doesn't delete)
- Dry run available for preview
- Can keep old directory until satisfied

## Path Examples

| Old Path | New Path |
|----------|----------|
| `experimentruns/automotive/hcrl_sa/teacher/unsupervised/vgae/no_distillation/autoencoder` | `experimentruns/hcrl_sa/teacher_autoencoder` |
| `experimentruns/automotive/hcrl_sa/student/supervised/gat/distilled/curriculum` | `experimentruns/hcrl_sa/student_curriculum_kd` |
| `experimentruns/automotive/set_01/teacher/rl_fusion/dqn/no_distillation/fusion` | `experimentruns/set_01/teacher_fusion` |
| `experimentruns/automotive/hcrl_sa/student/evaluation/eval/distilled/evaluation` | `experimentruns/hcrl_sa/student_evaluation_kd` |

## Run ID Format

Deterministic run IDs enable Snakemake compatibility and MLflow tracking:

```
{dataset}/{model_size}_{stage}[_kd]
```

Examples:
- `hcrl_sa/teacher_autoencoder`
- `hcrl_sa/student_curriculum_kd`
- `set_01/teacher_fusion`
- `set_02/student_evaluation`

## Testing the New System

### Test 1: Run a Single Stage
```bash
python -m pipeline.cli autoencoder --preset vgae,teacher --dataset hcrl_ch
```

Check:
- ✓ Files created in `experimentruns/hcrl_ch/teacher_autoencoder/`
- ✓ config.json saved
- ✓ MLflow run logged

### Test 2: Run Snakemake Dry Run
```bash
snakemake -s pipeline/Snakefile -n
```

Check:
- ✓ DAG builds successfully
- ✓ No path errors
- ✓ Correct number of jobs

### Test 3: Query MLflow
```bash
python -m pipeline.query --all
```

Check:
- ✓ Runs appear in tracking database
- ✓ Parameters logged correctly
- ✓ Status tags set

## Troubleshooting

### Migration Issues

**"Target already exists"**
- New path collision with migrated/existing experiment
- Resolution: Manually inspect and remove/rename conflicting directory

**"Could not compute new path"**
- Experiment doesn't match expected 8-level structure
- Resolution: Check experiment path, may need manual migration

**"No config.json - skipping MLflow backfill"**
- Old experiment doesn't have frozen config
- Resolution: Normal for pre-migration runs, files still migrated

### Runtime Issues

**"Experiment not found" in MLflow**
- MLflow database not initialized
- Resolution: Run any stage once to create experiment

**"Database locked"**
- Concurrent write contention (rare)
- Resolution: Increase `busy_timeout` in `pipeline/tracking.py`

**Module import errors**
- Pipeline package not in Python path
- Resolution: Run from project root with `python -m pipeline.cli`

## Rollback (If Needed)

If migration causes issues:

1. **Stop all running jobs**
2. **Restore old directory:**
   ```bash
   # If you kept a backup
   mv experimentruns_backup/automotive experimentruns/
   ```
3. **Revert code changes:**
   ```bash
   git checkout HEAD~1 -- pipeline/
   ```

## Next Steps After Migration

1. **Update any custom scripts** to use new path helpers
2. **Test Snakemake workflows** with dry run
3. **Explore MLflow UI** via OSC OnDemand
4. **Run new experiments** to verify end-to-end flow
5. **Clean up old directories** once satisfied

## Reference

- [MLflow Usage Guide](mlflow_usage.md)
- [Pipeline Architecture](registry_plan.md)
- Migration script: `python -m pipeline.migrate --help`
- Query tool: `python -m pipeline.query --help`
