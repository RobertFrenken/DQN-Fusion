# Reference Guide: Concurrent MLflow on Slurm via Shared NFS

This guide outlines how to run up to 12 independent, parallel Slurm jobs that write to a single, global MLflow database on a shared NFS filesystem without database locks or write corruption.

Because MLflow requires a database backend to use the **Model Registry** (`registered_model_name`), this solution uses an SQLite database file stored on the NFS, protected by a robust cross-node file-locking mechanism.

---

## 🏗️ Architecture Overview

- **Metadata & Parameters Backend:** An SQLite file residing on the shared NFS (`sqlite:////mnt/nfs/my_project/mlflow.db`).
- **Artifact Store (Models/Weights):** A dedicated directory on the NFS (`/mnt/nfs/my_project/artifacts`).
- **Concurrency Protection:** The Python `filelock` library. It creates a physical `.lock` file on the NFS that all independent cluster nodes respect, safely queueing database writes.

---

## 🛠️ Implementation

### 1. Dependencies

Ensure you have the required locking package installed in your cluster environment:

```bash
pip install filelock mlflow
```

### 2. Python Training & Registration Script (`train_and_register.py`)

This script executes heavy model training completely unlocked, but wraps the initialization, parameter logging, and model registration phases inside a cross-node lock.

```python
import os
import time
from filelock import FileLock
import mlflow
import mlflow.sklearn  # Switch to mlflow.pytorch, mlflow.xgboost, etc., as needed

# ---------------------------------------------------------
# 1. PATH CONFIGURATION ON SHARED NFS
# ---------------------------------------------------------
NFS_BASE_DIR = "/mnt/nfs/my_project"
DB_PATH = os.path.join(NFS_BASE_DIR, "mlflow.db")
ARTIFACTS_DIR = os.path.join(NFS_BASE_DIR, "artifacts")

# Ensure base tracking directories exist on the NFS
os.makedirs(NFS_BASE_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Point MLflow to the global SQLite database on the NFS
# Note: sqlite:/// requires 3 slashes followed by the absolute path (total 4 slashes)
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

# ---------------------------------------------------------
# 2. CONCURRENCY & LOCK CONFIGURATION
# ---------------------------------------------------------
LOCK_FILE = os.path.join(NFS_BASE_DIR, "mlflow_db.lock")

# Timeout is set to 180 seconds to allow up to 12 jobs to queue up comfortably
db_lock = FileLock(LOCK_FILE, timeout=180)

def locked_mlflow_call(func, *args, **kwargs):
    """Executes any MLflow interaction safely behind the NFS file lock."""
    with db_lock:
        # Critical: Adds a small buffer for NFS metadata caching lag across nodes
        time.sleep(0.15)
        return func(*args, **kwargs)

# ---------------------------------------------------------
# STEP 1: Safe Experiment Initialization
# ---------------------------------------------------------
def init_experiment():
    mlflow.set_experiment(experiment_name="Global_Slurm_Training")

locked_mlflow_call(init_experiment)

# ---------------------------------------------------------
# STEP 2: Heavy Model Training (Completely Unlocked)
# ---------------------------------------------------------
# Extract the unique Slurm Job ID to distinguish this specific run
slurm_job_id = os.environ.get("SLURM_JOB_ID", f"local_{int(time.time())}")
print(f"Starting heavy model training for Job ID: {slurm_job_id}...")

# [YOUR ACTUAL MACHINE LEARNING TRAINING CODE HERE]
# Dummy placeholders for demonstration purposes:
mock_model = ...
feature_params = {"learning_rate": 0.05, "batch_size": 64, "features_count": 142}
metrics = {"val_loss": 0.21, "accuracy": 0.94}

# ---------------------------------------------------------
# STEP 3: Log Parameters, Model, and Register (Strictly Locked)
# ---------------------------------------------------------
print(f"Job {slurm_job_id} training complete. Entering lock queue to register assets...")

def save_and_register_run():
    # Explicitly bound the artifact location to the shared NFS folder
    with mlflow.start_run(run_name=f"run_job_{slurm_job_id}", artifact_location=ARTIFACTS_DIR) as run:
        # 1. Log feature and training parameters
        mlflow.log_params(feature_params)
        mlflow.log_metrics(metrics)

        # 2. Log the model file and register it to the Model Registry
        # The registered_model_name parameter triggers the Model Registry database schema
        model_info = mlflow.sklearn.log_model(
            sk_model=mock_model,
            artifact_path="model",
            registered_model_name="Production_Global_Model"
        )
        print(f"Successfully registered Model Version to Registry for Job {slurm_job_id}")

# Run the final logging and registration sequence inside a single lock block
locked_mlflow_call(save_and_register_run)
```

### 3. Slurm Job Submission Script (`submit_job.sh`)

You can submit this script 1 to 12 times concurrently using `sbatch`. Every execution will run concurrently and queue safely at the end.

```bash
#!/bin/bash
#SBATCH --job-name=independent_ml_run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/mlflow_out_%j.log

# Ensure your logs directory exists
mkdir -p logs

# Activate your Python cluster environment
source /path/to/your/env/bin/activate

# Run the locked tracking script
python train_and_register.py
```

---

## 💡 Crucial Best Practices for NFS Tracking

1. **Why `filelock` is required over standard SQLite:** Standard SQLite databases rely on OS-level file locking (`fcntl`). Many high-performance cluster NFS configurations disable or poorly support network-level `fcntl`, resulting in database corruption or crashing errors. The `filelock` library circumvents this by creating physical tracking files on the storage system that all nodes can read and write to manually.
2. **Minimize Lock Time:** Keep the actual training code completely separate and _outside_ the locked sections. The lock should only be acquired for brief moments to create the experiment, and at the very end of the job to save variables and register models.
3. **The Metadata Buffer (`time.sleep(0.15)`):** Distributed filesystems (NFS/Lustre) have inherent file attribute caching delays. When Job A releases a lock file, Job B might acquire it so fast that it tries to read the SQLite database before the file system has fully synced Job A's changes. The brief pause right after gaining the lock prevents this edge case entirely.
