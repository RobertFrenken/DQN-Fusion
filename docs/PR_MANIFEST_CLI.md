Title: feat(manifest): CLI manifest validation & application for fusion/training

What changed
-----------
- Added `--dependency-manifest` CLI flag to `train_with_hydra_zen.py` and `src/training/fusion_training.py`.
- Implemented `apply_manifest_to_config()` in `train_with_hydra_zen.py` and `_extract_manifest_paths()` in `fusion_training.py` to load, validate and apply manifest paths (strict, fail-fast).
- Added `src/utils/dependency_manifest.py` with `load_manifest()` and `validate_manifest_for_config()`.
- Added tests: `tests/test_cli_manifest_integration.py`, `tests/test_fusion_manifest.py`.
- Updated `docs/DEPENDENCY_MANIFEST.md` with CLI usage and examples.

Why
---
- Makes dependency resolution for fusion jobs explicit and reproducible.
- Fails early when artifacts or metadata are missing, avoiding long failed GPU jobs.

How to test locally
-------------------
- Run unit tests: `pytest -q` (all tests pass locally: 21 passed).
- Lint: `ruff check .`
- Smoke (optional): create a small manifest pointing at existing canonical artifacts and run:
  - `python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training fusion --dependency-manifest /path/to/manifest.json`
  - OR `python src/training/fusion_training.py --dataset hcrl_sa --dependency-manifest /path/to/manifest.json`

Checklist
--------
- [x] Added/updated unit tests
- [x] Linting passed locally (`ruff check .`)
- [x] Tests pass locally (`pytest -q`)
- [x] Updated docs (`docs/DEPENDENCY_MANIFEST.md`)

Notes
-----
- The manifest validation is intentionally strict and will raise `ManifestValidationError` for missing keys/paths or incomplete metadata.
- The code contains fallbacks for importing manifest utilities by file path to support running tests in environments where `src` isn't an installed package.

Next steps (optional)
---------------------
- Add an integration test to run the manifest flow on a GPU runner or in a smoke job (requires GPU access).
- Add an example manifest file under `docs/examples/manifest_hcrl_sa.json`.