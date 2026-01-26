# Justfile for common dev tasks
# Usage: `just <task>`

# Default task
default: help

help:
	@echo "Available tasks:"
	@echo "  just check-env           # Check that required Python packages and tools are available"
	@echo "  just install-uv          # Install dependencies using uv (if you prefer uv)"
	@echo "  just install-conda       # Create & populate conda env (for cluster/OSC)"
	@echo "  just smoke               # Run short smoke experiment (requires dataset or --use-synthetic-data)"
	@echo "  just smoke-synthetic     # Run smoke using synthetic dataset"
	@echo "  just mlflow              # Start mlflow UI for experimentruns"
	@echo "  just preview             # Preview a sweep with oscjobmanager"
	@echo "  just collect-summaries   # Collect summary.json files across experimentruns"
	@echo "  just submit-dryrun       # Create slurm script for a config (dry-run)"

# Check environment (non-fatal - prints diagnostics)
check-env:
	python scripts/check_environment.py

# Install via uv (if you use uv locally). This assumes `uv` is the correct CLI.
# Adjust commands as needed for your 'uv' usage (e.g., 'uv env create' if different).
install-uv:
	@command -v uv >/dev/null 2>&1 || { echo "uv not found - please install uv first (see https://uv.dev)"; exit 1; }
	@echo "Installing dependencies with uv..."
	uv install || echo "uv install failed - check your uv workflow"

# Install via conda (for HPC/OSC). Edit ENV_NAME, YML if needed.
install-conda:
	conda env create -f environment.yml -n gnn-gpu || conda env update -f environment.yml -n gnn-gpu || echo "conda create/update failed"
	@echo "Activate with: conda activate gnn-gpu"

# Small smoke run (assumes dataset is present in config dataset path)
smoke:
	python scripts/local_smoke_experiment.py --run

# Synthetic smoke run (safe for dev machines)
smoke-synthetic:
	python scripts/local_smoke_experiment.py --use-synthetic-data --run --write-summary

# Integration smoke test (runs a canned integration test that uses a fake trainer)
smoke-integration:
	pytest -q tests/test_integration_smoke.py::test_integration_smoke_end_to_end

# Pre-submit one-command check
pre-submit:
	python scripts/pre_submit_check.py --dataset hcrl_ch --run-load --smoke --smoke-synthetic --preview-json

# Check datasets via config store or explicit path (useful before running experiments)
check-data:
	python scripts/check_datasets.py --dataset hcrl_ch

# Check a custom dataset path and attempt to load it
check-data-load:
	python scripts/check_datasets.py --path datasets/can-train-and-test-v1.5/hcrl-ch --run-load --force-rebuild

# Start MLflow UI pointing at canonical experiment root
mlflow:
	@echo "Starting MLflow UI pointing at experimentruns/.mlruns"
	mlflow ui --backend-store-uri experimentruns/.mlruns

# Preview a sweep via oscjobmanager
preview:
	python oscjobmanager.py preview --dataset hcrl_ch --model-sizes student,teacher --distillations no --training-modes all_samples --json

# Collect summary.json files
collect-summaries:
	python scripts/collect_summaries.py --experiment-root experimentruns

# Generate a dry-run slurm script (edit config_name)
submit-dryrun:
	python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_student_no_all_samples --dry-run
