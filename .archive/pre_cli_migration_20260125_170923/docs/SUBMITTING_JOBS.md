# Submitting Jobs & Inspecting Hyperparameters 🚀

This document explains how to submit training jobs (single or sweep), how to preview them, and how to inspect and override hyperparameters. It also shows which settings you can change directly on the CLI versus which ones you must change in the Python configuration (Hydra-Zen) files.

---

## High-level quick commands ✅

- Submit a single job (via OSC job manager):

  ```bash
  python oscjobmanager.py submit <CONFIG_NAME> [--dry-run] [--walltime 04:00:00] [--memory 64G] [--pre-submit]
  ```

  Example:
  ```bash
  # Prefer canonical preset names; pass a preset directly to train script
  python oscjobmanager.py submit autoencoder_hcrl_ch --dry-run
  # Legacy names are still accepted by the job manager (case-insensitive), but prefer canonical names for clarity:
  python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_teacher_no_autoencoder --dry-run
  ```

- Submit directly (run locally / inside container or on a node):

  ```bash
  python train_with_hydra_zen.py --model vgae --dataset hcrl_ch --training autoencoder --epochs 200 --batch_size 64
  ```

- Preview a sweep without creating scripts (shows run directories and expected artifacts):

  ```bash
  python oscjobmanager.py preview --dataset hcrl_ch --model-architecture VGAE --model-sizes teacher --training-modes all_samples
  ```

- Submit a sweep (single dataset per invocation):

  ```bash
  python oscjobmanager.py sweep --dataset hcrl_ch --model-sizes teacher --model-architecture VGAE --training-modes all_samples
  ```

  Note: `sweep` currently accepts a single dataset per invocation. Use a shell loop to submit multiple datasets in one go:

  ```bash
  for d in hcrl_ch hcrl_sa set_01 set_02; do
    python oscjobmanager.py sweep --dataset "$d" --model-sizes teacher --model-architecture VGAE --training-modes all_samples
  done
  ```

---

## Which component generates the Slurm file and submits? 📝➤🖥️

- `OSCJobManager.submit_job()` will generate a Slurm script under `experimentruns/.../slurm_runs/` and call `sbatch` to submit it (unless `--dry-run` is used).
- Use `--dry-run` to only create and inspect the generated script without submitting.
- Use `--pre-submit` to run `scripts/pre_submit_check.py` (dataset validation & smoke previews) prior to submission.

Yes — the job manager both writes the Slurm script and submits it with `sbatch` by default.

---

## How to list / inspect presets and hyperparameters 🔎

### Presets (high-level curated configs)

- List built-in training presets:
  ```bash
  python train_with_hydra_zen.py --list-presets
  ```
- Presets are defined in `src/config/training_presets.py` (e.g. `teacher_vgae_autoencoder`, `distillation_aggressive`, `fusion_standard`).
- Presets can be used directly with `--preset <PRESET_NAME>` when calling `train_with_hydra_zen.py`.

### Inspect the full assembled config programmatically

- To dump a full runtime config (model + dataset + training), use this Python snippet:

  ```bash
  python - <<'PY'
  import json
  from dataclasses import asdict
  from src.config.hydra_zen_configs import CANGraphConfigStore

  cfg = CANGraphConfigStore().create_config('vgae', 'hcrl_ch', 'autoencoder')
  print(json.dumps(asdict(cfg), indent=2))
  PY
  ```

  This prints every field and the defaults from the hydra-zen dataclasses.

### Quick CLI overrides (see `train_with_hydra_zen.py` args)

You can override common training params directly on the command line when invoking `train_with_hydra_zen.py`:

- General overrides:
  - `--epochs INT` → overrides `training.max_epochs`
  - `--batch_size INT` → overrides `training.batch_size`
  - `--learning_rate FLOAT` → overrides `training.learning_rate`
  - `--early-stopping-patience INT`
  - `--tensorboard` → enables tensorboard logging
  - `--force-rebuild-cache`

- Knowledge distillation overrides:
  - `--teacher_path PATH` (required for KD)
  - `--student_scale FLOAT` (maps to `training.student_model_scale`)
  - `--distillation_alpha FLOAT`
  - `--temperature FLOAT`

- Curriculum / Fusion overrides:
  - `--vgae_path PATH` (curriculum)
  - `--autoencoder_path PATH` (fusion)
  - `--classifier_path PATH` (fusion)

- Preset usage: `--preset <PRESET_NAME>` picks a curated config; you can still pass the CLI overrides above to tweak it.

- Note: `oscjobmanager` calls `train_with_hydra_zen.py` with `config_store=<CONFIG_NAME>` when submitting; that `config_store` entry corresponds to a stored composite config assembled by `CANGraphConfigStore`.

---

## Which hyperparameters must be changed in code/config files? 🛠️

Major model and training hyperparameters live in the Hydra-Zen dataclasses and must be edited in `src/config/hydra_zen_configs.py` or changed via code that constructs a config (e.g., `get_preset_with_overrides`). These include:

- Model-level (change in `src/config/hydra_zen_configs.py`):
  - VGAE (Teacher): `hidden_dims`, `latent_dim`, `attention_heads`, `embedding_dim`, `dropout`, `batch_norm`, `target_parameters` (`VGAEConfig`)
  - Student VGAE: `encoder_dims`, `decoder_dims`, `latent_dim`, `attention_heads` (`StudentVGAEConfig`)
  - GAT: `hidden_channels`, `num_layers`, `heads`, `dropout`, `num_fc_layers`, `embedding_dim` (`GATConfig` / `StudentGATConfig`)
  - DQN: `num_layers`, `hidden_units`, `hidden_channels`, `buffer_size`, etc. (`DQNConfig`)

- Training-level (change defaults in `hydra_zen_configs.py` or pass a preset):
  - `BaseTrainingConfig` fields: `max_epochs`, `batch_size`, `learning_rate`, `optimizer`, `scheduler`, `precision`, `early_stopping_patience`, `accumulate_grad_batches`, and so on.
  - Curriculum-specific parameters are in `CurriculumTrainingConfig` (e.g., `difficulty_percentile`, `use_vgae_mining`).
  - Fusion training hooks and agent params in `FusionTrainingConfig` and `FusionAgentConfig`.

- Dataset-level: the dataset entries and their `data_path` are configured in `CANGraphConfigStore._register_presets()` (see `get_dataset_config()`), and per-dataset `CANDatasetConfig` fields (e.g., `time_window`, `overlap`) are declared in `src/config/hydra_zen_configs.py`.

If you want a consistent, repeatable change for multiple runs, prefer editing these dataclasses or creating a new preset wrapper in `src/config/training_presets.py` which sets `overrides`.

---

## How to enumerate every hyperparameter quickly 🧭

- Programmatic field listing (shows each dataclass and default values):

  ```bash
  python - <<'PY'
  import inspect
  from src.config import hydra_zen_configs as cfg
  print('=== VGAEConfig ===')
  print(inspect.getsource(cfg.VGAEConfig))
  print('\n=== Training Config defaults (BaseTrainingConfig) ===')
  print(inspect.getsource(cfg.BaseTrainingConfig))
  PY
  ```

- Or use `asdict()` on a created config (recommended, prints runtime values after composition):
  ```bash
  python - <<'PY'
  import json
  from dataclasses import asdict
  from src.config.hydra_zen_configs import CANGraphConfigStore
  print(json.dumps(asdict(CANGraphConfigStore().create_config('vgae','hcrl_ch','autoencoder')), indent=2))
  PY
  ```

---

## How `--preset <PRESET_NAME>` maps to final behavior

- `oscjobmanager` generates Slurm scripts that call `train_with_hydra_zen.py --preset <PRESET_NAME>`; the CLI is strict and expects canonical preset names (no legacy fuzzy matching).
- Canonical preset naming convention:
  - `gat_normal_<dataset>` (e.g., `gat_normal_hcrl_ch`)
  - `autoencoder_<dataset>` (VGAE autoencoder, e.g., `autoencoder_set_04`)
  - `distillation_<dataset>_scale_<scale>` (e.g., `distillation_hcrl_ch_scale_0.5`)
  - `fusion_<dataset>` (e.g., `fusion_hcrl_sa`)
- `oscjobmanager` accepts training shorthand (e.g., `--training gat_normal`) when creating jobs and will emit Slurm scripts invoking the corresponding canonical presets.
- Use `python train_with_hydra_zen.py --list-presets` to see all valid preset names before submission.
---

## Pre-submit checklist ✅

- Use `python oscjobmanager.py preview ...` to confirm run dir and expected artifacts.
- Use `--dry-run` to inspect the generated Slurm script content before submission.
- If using curriculum/fusion, ensure required artifacts exist (e.g., trained VGAE teacher or classifier weights) or pass correct `--vgae_path` / `--autoencoder_path` etc.
- Run `scripts/pre_submit_check.py --run-load --smoke` with `--pre-submit` via oscjobmanager to smoke-test the config.

---

## Want me to add: `--datasets a,b,c` support to `sweep`? 🧩

I can add a comma-separated `--datasets` option so a single `sweep` invocation will submit jobs for multiple datasets — I can implement it and add a small unit test and docs if you'd like.

---

If you'd like, I can also add a small helper script `tools/inspect_config.py` that prints the full flattened config for a named model/dataset/training-mode and emits JSON (useful for automated reviews & job manifests). Would you like that? 
