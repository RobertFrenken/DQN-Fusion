# Refactor Helper Agent — Runbook & Task List ✅
I appreciate the suggestion, but I'm following a different approach. I'm using:
Hydra-Zen for configuration management
PyTorch Lightning for training
MLflow for tracking
Slurm (via oscjobmanager) for cluster execution
This approach handles everything I need without CI/CD complexity.
I want to focus on research, not infrastructure.
Purpose
-------
This document defines a lightweight "agent" (runbook + checklist + automation commands) to help coordinate the refactor of the project into a strict, canonical experiment layout and save/load behaviour.

Design Principles
-----------------
- Strictness: No silent fallbacks. If a required artifact/path is missing, fail with a clear, actionable error.
- Single source of truth: `CANGraphConfig` exposes canonical paths via `canonical_experiment_dir()`.
- Reproducibility: All saved models are plain `state_dict` dictionaries and every run logs the canonical artifact path to MLflow (optionally copying artifact into MLflow if requested).
- Incremental & test-driven: Make small PRs, add tests, run CI; avoid large sweeping changes in one step.

Agent Responsibilities (what it does for you)
---------------------------------------------
- Provide a checklist for each refactor step (config → trainer → training code → job manager → docs/tests)
- Run static checks and unit tests locally and in CI
- Provide exact commands to run/verify changes (lint, unit tests, flake/pylint/ruff, python compile)


Local verification commands
---------------------------
- Activate venv: `source .venv/bin/activate`
- Quick syntax check: `python -m py_compile train_with_hydra_zen.py src/config/hydra_zen_configs.py src/training/*.py`
- Linting: `ruff check .` (or `flake8`, `pylint` as project uses)
- Run tests: `pytest tests -q`
- Run focused tests: `pytest tests/test_config.py::test_canonical_dir -q`
- Type checks: `mypy src` (optional)

Core checklists per step
-----------------------
1) Config refactor — `src/config/hydra_zen_configs.py` (PR #1)
   - Add canonical experiment fields on `CANGraphConfig` (done)
   - Remove redundant `experiment_root` from dataset-level config (done or to be done)
   - Ensure `canonical_experiment_dir()` follows exact grammar:
     `experiment_runs/{modality}/{dataset}/{learning_type}/{model_arch}/{model_size}/{distillation}/{training_mode}/`
   - Implement `required_artifacts()` that returns canonical paths (no discovery)
   - Add strict `validate_config()` that raises `FileNotFoundError` or `ValueError` with remediation instructions
   - Add unit tests for canonical paths and validate_config missing artifact errors

2) Trainer & saving behaviour — `train_with_hydra_zen.py` (PR #2)
   - `HydraZenTrainer.get_hierarchical_paths()` should call `config.canonical_experiment_dir()` and create directories.
   - `HydraZenTrainer._save_state_dict()` should always save plain `state_dict` dicts and print/log the saved full path.
   - Ensure MLflow logger uses the same `mlruns` dir inside the canonical path and logs the `model_path` as an MLflow param.
   - Unit tests: simulate saving state_dict and ensure load via `torch.load` returns a dict.

3) Training modules & fusion/eval — `src/training/*` (PR #3)
   - All modules that load models must accept only explicit config paths or use `CANGraphConfig.required_artifacts()` and fail if missing.
   - Replace any legacy glob/auto-discovery with explicit errors.
   - Add integration tests for fusion: fail early if pre-trained GAT/VGAE missing.

4) `osc_job_manager.py` rewrite (PR #4)
   - Use strict `resource_profiles` + explicit training types.
   - Output directories MUST be under `experiment_runs`; SBATCH scripts should create those dirs and not `osc_jobs`.
   - Do NOT fallback to legacy paths; provide a `--auto-chain` opt-in flag if desired.
   - Add script dry-run tests that verify generated SBATCH scripts reference canonical paths.

5) Presets & docs (PR #5)
   - Update `src/config/training_presets.py` to point to canonical paths (or leave placeholders and document needing to be populated)
   - Update `JOB_TEMPLATES.md` and `README.md` with the canonical path grammar and examples
   - Add migration guide: shell script to move or symlink useful legacy artifacts into `experiment_runs`


Notes on MLflow integration
--------------------------
- The agent recommends the trainer **write artifacts to canonical paths** and then **log the artifact path** to MLflow using `mlflow.log_param("model_path", str(model_path))`. Uploading model artifacts to MLflow is optional and can be controlled by a training config flag (e.g., `logging.log_model_artifact: bool`).

Security & safety notes
-----------------------
- Avoid unpickling arbitrary legacy checkpoints. If a conversion is required, do it in a controlled helper script and log the source.
- Ensure saved checkpoints are `torch.save(state_dict)` only.
