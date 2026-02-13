"""Compose config from YAML layers: defaults -> model_def -> auxiliaries -> CLI overrides."""
from __future__ import annotations

import yaml
from pathlib import Path
from .schema import PipelineConfig

CONFIG_DIR = Path(__file__).parent


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def resolve(
    model_type: str,
    scale: str,
    auxiliaries: str = "none",
    **cli_overrides,
) -> PipelineConfig:
    # 1. Global defaults
    defaults_path = CONFIG_DIR / "defaults.yaml"
    merged = yaml.safe_load(defaults_path.read_text()) if defaults_path.exists() else {}

    # 2. Model definition (architecture + scale-specific overrides)
    model_path = CONFIG_DIR / "models" / model_type / f"{scale}.yaml"
    if model_path.exists():
        _deep_merge(merged, yaml.safe_load(model_path.read_text()))

    # 3. Auxiliaries
    if auxiliaries != "none":
        aux_path = CONFIG_DIR / "auxiliaries" / f"{auxiliaries}.yaml"
        if aux_path.exists():
            _deep_merge(merged, yaml.safe_load(aux_path.read_text()))

    # 4. CLI overrides (nested dict from caller)
    if cli_overrides:
        _deep_merge(merged, cli_overrides)

    # 5. Set identity fields
    merged["model_type"] = model_type
    merged["scale"] = scale

    # 6. Pydantic validates + freezes
    return PipelineConfig.model_validate(merged)


def list_models() -> dict[str, list[str]]:
    """Discover available model types and scales from filesystem."""
    models = {}
    models_dir = CONFIG_DIR / "models"
    if models_dir.exists():
        for model_dir in sorted(models_dir.iterdir()):
            if model_dir.is_dir():
                scales = [f.stem for f in sorted(model_dir.glob("*.yaml"))]
                if scales:
                    models[model_dir.name] = scales
    return models


def list_auxiliaries() -> list[str]:
    """Discover available auxiliary configs from filesystem."""
    aux_dir = CONFIG_DIR / "auxiliaries"
    if aux_dir.exists():
        return [f.stem for f in sorted(aux_dir.glob("*.yaml"))]
    return ["none"]
