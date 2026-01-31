# Pipeline Framework Changelog

## 2026-01-31 - Major Refactor: MLflow + Path Simplification

### Overview

Complete overhaul of the experiment tracking and path management system:
- Simplified directory structure from 8 levels to 2 levels
- Replaced custom SQLite registry with MLflow tracking
- Added query tools and migration utilities
- Updated Snakemake orchestration

### New Features

#### 1. Simplified Path Structure
- **Before:** `experimentruns/automotive/hcrl_sa/teacher/unsupervised/vgae/no_distillation/autoencoder/`
- **After:** `experimentruns/hcrl_sa/teacher_autoencoder/`
- 75% reduction in path depth
- Deterministic run IDs: `{dataset}/{model_size}_{stage}[_kd]`

#### 2. MLflow Experiment Tracking
- Industry-standard tracking with MLflow 3.8.1
- Automatic logging of all hyperparameters
- Metric tracking (F1, accuracy, precision, recall, etc.)
- Teacher-student lineage via tags
- Web UI via OSC OnDemand

#### 3. Query Tools
- **CLI tool:** `python -m pipeline.query` for command-line queries
- **Jupyter notebook:** `notebooks/query_mlflow.ipynb` for pandas analysis
- **MLflow UI:** Visual experiment comparison via OSC OnDemand

#### 4. Migration Utilities
- `python -m pipeline.migrate --dry-run`: Preview migration
- `python -m pipeline.migrate --execute`: Migrate old experiments
- Automatic backfill of MLflow data (when config.json exists)

### Files Added

```
pipeline/
  tracking.py          # MLflow integration (140 lines)
  query.py             # CLI query tool (190 lines)
  migrate.py           # Migration script (280 lines)

notebooks/
  query_mlflow.ipynb   # Interactive query notebook

docs/
  mlflow_usage.md      # Complete MLflow guide
  migration_guide.md   # Migration instructions
  CHANGELOG.md         # This file
```

### Files Modified

```
pipeline/
  paths.py             # Simplified stage_dir(), added run_id()
  cli.py               # Added MLflow tracking hooks
  Snakefile            # Updated all 18 rules to 2-level paths

.gitignore             # Added DVC-tracked directories
```

### Breaking Changes

⚠️ **Path structure changed** - old Snakefile rules incompatible with new paths
⚠️ **MLflow required** - new runs need MLflow tracking database
⚠️ **Snakemake rules updated** - all input/output paths changed

### Migration Required

For existing experiments:
```bash
# Preview migration
python -m pipeline.migrate --dry-run

# Execute migration
python -m pipeline.migrate --execute

# Clean up old structure (after verification)
rm -rf experimentruns/automotive
```

### Benefits

1. **Usability**
   - Shorter, more readable paths
   - Easier navigation in file browsers
   - Less typing for manual operations

2. **Compatibility**
   - Deterministic paths work with Snakemake DAG
   - Standard MLflow format for data science tools
   - Industry-standard tracking (transferable skills)

3. **Functionality**
   - Rich queries with MLflow search API
   - Visual comparison in MLflow UI
   - Automatic lineage tracking
   - Metric history and visualization

4. **Performance**
   - Reduced filesystem depth
   - Efficient SQLite queries
   - Concurrent-safe writes on GPFS

### Testing

Run Snakemake dry run to verify:
```bash
snakemake -s pipeline/Snakefile -n
```

Expected output: 55 jobs (18 training rules × 6 datasets - overlap)

### Documentation

- [MLflow Usage Guide](mlflow_usage.md)
- [Migration Guide](migration_guide.md)
- [Registry Design Plan](registry_plan.md)

### Compatibility

- **Python:** 3.9+
- **MLflow:** 3.8.1 (already installed)
- **Snakemake:** 7.x+
- **Filesystem:** GPFS scratch required for concurrent writes

### Known Limitations

1. Old experiments without `config.json` can be migrated (files moved) but won't have MLflow tracking
2. MLflow UI requires OSC OnDemand session (not persistent)
3. SQLite backend requires GPFS (not NFS) for concurrent writes
4. Run IDs are deterministic (good for Snakemake, but no UUIDs)

### Next Steps

1. Run migration on existing experiments
2. Update any custom analysis scripts to use new paths
3. Test end-to-end workflow with Snakemake
4. Explore MLflow UI for experiment analysis
5. Create first experiments with new system

---

## Previous Changes

### 2026-01-30 - DVC Integration
- Added DVC tracking for all datasets (hcrl_ch, hcrl_sa, set_01-04)
- Updated .gitignore for DVC-tracked directories
- Committed .dvc files for version control

### Earlier
- Initial pipeline implementation
- PyTorch Lightning integration
- Knowledge distillation framework
- Multi-stage training (autoencoder → curriculum → fusion → evaluation)
