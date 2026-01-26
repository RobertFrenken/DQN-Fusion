# Dependency Manifest (for Fusion / Composite Jobs) 🧾

Purpose
-------
A dependency manifest documents exact pre-trained artifacts a composite job requires (for example, a DQN fusion job depending on a VGAE autoencoder and a GAT classifier). It makes dependency resolution explicit and reproducible.

Schema
------
JSON object where top-level keys are dependency names (e.g., `autoencoder`, `classifier`) and values are objects with required fields:
- path: absolute path to the artifact file
- model_size: `teacher` or `student`
- training_mode: e.g., `autoencoder`, `normal`, `curriculum`
- distillation: `no_distillation` or `distilled`

Example
-------
{
  "autoencoder": {
    "path": "/path/to/experiment_runs/automotive/hcrl_sa/unsupervised/vgae/teacher/no_distillation/autoencoder/vgae_autoencoder.pth",
    "model_size": "teacher",
    "training_mode": "autoencoder",
    "distillation": "no_distillation"
  },
  "classifier": {
    "path": "/path/to/experiment_runs/automotive/hcrl_sa/supervised/gat/teacher/no_distillation/normal/gat_hcrl_sa_normal.pth",
    "model_size": "teacher",
    "training_mode": "normal",
    "distillation": "no_distillation"
  }
}

Validation
----------
- Use `src.utils.dependency_manifest.validate_manifest_for_config(manifest_dict, config)` to validate a loaded manifest for a fusion config.
- Validation checks presence, metadata fields, and that the referenced paths exist (fail-fast behavior).

Recommended workflow
--------------------
1. After completing teacher trainings, store canonical paths to artifacts in a manifest (or record them in the run metadata).
2. For fusion/external jobs, provide the manifest file and validate it prior to running the heavy pipeline.
3. Use the manifest to pin exact artifact versions (optionally add checksums in a future iteration for stronger provenance).

CLI usage
---------
- `train_with_hydra_zen.py` accepts `--dependency-manifest /path/to/manifest.json` and will validate then apply the manifest **for fusion jobs** (automatically setting `training.autoencoder_path` and `training.classifier_path`).
- `src/training/fusion_training.py` accepts `--dependency-manifest /path/to/manifest.json` and will validate and use the artifact paths before starting the pipeline.

Notes
-----
- Validation is strict and will fail early if paths are missing or metadata is incomplete. This avoids expensive failures later in heavy GPU jobs.
Why prefer canonical locations
-----------------------------
- Single source-of-truth avoids ambiguous copies or sibling-relative logic.
- The manifest allows explicit overrides for reproducible runs while keeping canonical paths as default names.

