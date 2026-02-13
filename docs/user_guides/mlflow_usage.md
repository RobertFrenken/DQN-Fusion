# MLflow Experiment Tracking Guide

This project uses MLflow for experiment tracking, replacing the custom SQLite registry.

## Tracking Database Location

The MLflow tracking database is stored on GPFS scratch for concurrent write safety:

```
/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db
```

This location is configured via:
- Default: `pipeline/tracking.py` (TRACKING_URI)
- Environment variable: `MLFLOW_TRACKING_URI`

## Viewing Experiments

### Option 1: SSH Tunnel (Fastest)

Start the MLflow UI on a login node and tunnel to your local machine:

```bash
# On login node (inside tmux):
mlflow ui --backend-store-uri sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db --host 0.0.0.0 --port 5000

# Local machine:
ssh -L 5000:localhost:5000 rf15@pitzer.osc.edu
# Open http://localhost:5000
```

### Option 2: OSC OnDemand MLflow UI

1. Go to https://ondemand.osc.edu
2. Navigate to **Interactive Apps** → **MLflow**
3. Configure the session:
   - **Tracking URI directory**: `/fs/scratch/PAS1266/kd_gat_mlflow`
   - Select compute resources (usually defaults are fine)
4. Launch the session
5. Once running, click "Connect to MLflow"
6. Browse experiments in the web UI

The MLflow UI provides:
- Visual experiment comparison
- Metric plots and charts
- Parameter search and filtering
- Model artifact browsing

### Option 3: Jupyter Notebook

Use the provided analytics notebook for pandas-based analysis:

```bash
# On OSC OnDemand
# 1. Launch a Jupyter session
# 2. Navigate to notebooks/03_analytics.ipynb
# 3. Run cells to query and analyze experiments
```

The notebook includes:
- Filtered queries by dataset/stage/model
- Teacher vs student comparisons
- Leaderboard generation
- Export to CSV

### Option 4: CLI Analytics Tool

Query from the command line:

```bash
# Show leaderboard (top 10 by F1)
python -m pipeline.analytics leaderboard --metric f1 --top 10

# Compare two runs
python -m pipeline.analytics compare <run_a> <run_b>

# Config diff between runs
python -m pipeline.analytics diff <run_a> <run_b>

# Sweep a hyperparameter
python -m pipeline.analytics sweep --param lr --metric f1

# Dataset summary
python -m pipeline.analytics dataset hcrl_sa

# Custom SQL query
python -m pipeline.analytics query "SELECT * FROM runs WHERE dataset = 'hcrl_sa'"
```

## When to Use What

| Task | Tool |
|------|------|
| Visual metric comparison, artifact browsing | MLflow UI |
| Hyperparameter sweep analysis | `pipeline.analytics sweep` |
| Arbitrary SQL on results | `pipeline.analytics query` or Datasette |
| Per-dataset summary | `pipeline.analytics dataset` |
| Interactive DB exploration | Datasette (`datasette data/project.db`) |

## What Gets Tracked

For each training run, MLflow automatically logs:

### Parameters (from PipelineConfig)
- All hyperparameters (lr, batch_size, hidden dims, etc.)
- Dataset and model configuration
- Knowledge distillation settings

### Tags
- `dataset`: Dataset name (hcrl_sa, set_01, etc.)
- `stage`: Training stage (autoencoder, curriculum, fusion, evaluation)
- `model_size`: teacher or student
- `use_kd`: Whether KD was used
- `model_arch`: Model architecture (vgae, gat, dqn, eval)
- `teacher_run_id`: Parent teacher run (for KD runs)
- `status`: complete, failed, or running
- `start_time`, `end_time`: Timestamps

### Metrics (from training results)
- `val_loss`: Validation loss
- `accuracy`: Accuracy score
- `f1`: F1 score
- `precision_`: Precision
- `recall`: Recall
- `auc`: AUC score
- `mcc`: Matthews correlation coefficient
- Additional stage-specific metrics

## Common Queries

### Find best model for a dataset
```python
import mlflow
mlflow.set_tracking_uri("sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db")
runs = mlflow.search_runs(
    experiment_ids=["<experiment_id>"],
    filter_string="tags.dataset = 'hcrl_sa' AND tags.stage = 'evaluation'",
    order_by=["metrics.f1 DESC"],
    max_results=1
)
```

### Compare KD vs non-KD students
```python
kd_runs = mlflow.search_runs(
    filter_string="tags.use_kd = 'True' AND tags.status = 'complete'"
)
no_kd_runs = mlflow.search_runs(
    filter_string="tags.use_kd = 'False' AND tags.status = 'complete'"
)
```

### Find failed runs
```python
failed = mlflow.search_runs(
    filter_string="tags.status = 'failed'"
)
```

## Querying Teacher-Student Lineage

Student runs with KD have a `teacher_run_id` tag pointing to their teacher:

```python
# Get a student run
student = mlflow.search_runs(
    filter_string="tags.dataset = 'hcrl_sa' AND tags.stage = 'curriculum' AND tags.use_kd = 'True'"
).iloc[0]

teacher_id = student["tags.teacher_run_id"]

# Get the teacher run
teacher = mlflow.search_runs(
    filter_string=f"tags.run_id = '{teacher_id}'"
).iloc[0]

# Compare
print(f"Student F1: {student['metrics.f1']}")
print(f"Teacher F1: {teacher['metrics.f1']}")
print(f"Delta: {student['metrics.f1'] - teacher['metrics.f1']}")
```

## Backup and Maintenance

### Backup the database
```bash
# Periodic backup to home directory
cp /fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db ~/backups/mlflow_$(date +%Y%m%d).db
```

### Clean up old runs
```bash
# Use MLflow CLI to delete old experiment runs
mlflow gc --backend-store-uri sqlite:////fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db
```

## Integration with Snakemake

MLflow tracking is automatic when using the pipeline CLI:

```bash
# This automatically logs to MLflow
python -m pipeline.cli autoencoder --preset vgae,teacher --dataset hcrl_sa
```

The run ID in MLflow matches the filesystem path for easy correlation:
- MLflow run name: `hcrl_sa/teacher_autoencoder`
- Filesystem path: `experimentruns/hcrl_sa/teacher_autoencoder/`

## Concurrent Writes on SLURM

The tracking database is on GPFS scratch, which provides:
- POSIX-compliant file locking
- Safe concurrent writes from multiple SLURM jobs
- 5-second busy timeout for lock retries

At typical experimental scale (~6 concurrent jobs), write contention is minimal.

## Troubleshooting

### "Experiment not found"
Ensure the tracking URI is correct and the database exists:
```bash
ls -lh /fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db
```

### Database locked errors
Increase the busy timeout in `pipeline/tracking.py` if needed (default: 5s).

### OnDemand MLflow app doesn't show experiments
1. Verify the "Tracking URI directory" path is correct
2. Check that the database file exists at that location
3. Ensure you have read permissions

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [OSC MLflow App](https://github.com/OSC/bc_osc_mlflow)
