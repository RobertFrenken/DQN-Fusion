"""Utilities to create and validate dependency manifests for fusion/complex jobs.

Manifest schema (JSON):
{
  "autoencoder": {"path": "/abs/path/to/vgae_autoencoder.pth", "model_size": "teacher", "training_mode": "autoencoder", "distillation": "no_distillation"},
  "classifier": {"path": "/abs/path/to/gat_hcrl_sa_normal.pth", "model_size": "teacher", "training_mode": "normal", "distillation": "no_distillation"}
}

Provide helpers to load the manifest and validate it against a CANGraphConfig's required artifacts.
"""
from pathlib import Path
import json
from typing import Dict, Any, Tuple


class ManifestValidationError(Exception):
    pass


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Dependency manifest not found: {manifest_path}")
    with open(p, 'r') as f:
        data = json.load(f)
    return data


def validate_manifest_for_config(manifest: Dict[str, Any], config) -> Tuple[bool, str]:
    """Validate a loaded manifest against a CANGraphConfig for fusion-like jobs.

    Checks performed:
    - Required keys present (autoencoder, classifier) for fusion configs
    - Each entry contains a path and minimal metadata
    - The `path` exists on disk (raises if missing)
    - Metadata fields (`model_size`, `training_mode`, `distillation`) are present and non-empty

    Returns (True, 'ok') if validation passes, otherwise raises ManifestValidationError with details.
    """
    if not isinstance(manifest, dict):
        raise ManifestValidationError("Manifest must be a JSON object/dict")

    # Only support validation for fusion-type configs for now
    if getattr(config.training, 'mode', None) != 'fusion':
        return True, "manifest validation skipped for non-fusion config"

    required = ['autoencoder', 'classifier']
    missing = [k for k in required if k not in manifest]
    if missing:
        raise ManifestValidationError(f"Manifest missing required entries: {missing}")

    errors = []
    for key in required:
        entry = manifest.get(key)
        if not isinstance(entry, dict):
            errors.append(f"Entry '{key}' must be an object with keys: path, model_size, training_mode, distillation")
            continue
        path = entry.get('path')
        if not path:
            errors.append(f"Entry '{key}' missing 'path' field")
            continue
        p = Path(path)
        if not p.exists():
            errors.append(f"Path for '{key}' does not exist: {path}")
        # Check metadata
        for meta in ['model_size', 'training_mode', 'distillation']:
            if not entry.get(meta):
                errors.append(f"Entry '{key}' missing metadata '{meta}'")

    if errors:
        raise ManifestValidationError("; ".join(errors))

    return True, "ok"
