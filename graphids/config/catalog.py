"""Pydantic-validated dataset catalog loader.

Validates entries in config/datasets.yaml against a strict schema so that
malformed entries are caught at load time, not deep in preprocessing.
"""
from __future__ import annotations

import yaml
from pydantic import BaseModel

from .constants import CATALOG_PATH


class DatasetEntry(BaseModel, frozen=True):
    """Schema for one dataset catalog entry."""
    domain: str
    protocol: str
    source: str = ""
    description: str = ""
    csv_dir: str
    csv_columns: dict[str, str]
    train_subdir: str
    train_attack_subdir: str = ""
    test_subdirs: list[str]
    added_by: str = ""


def load_catalog() -> dict[str, DatasetEntry]:
    """Load and validate all dataset entries from config/datasets.yaml.

    Raises ValidationError on malformed entries.
    """
    with open(CATALOG_PATH) as f:
        raw = yaml.safe_load(f)

    catalog = {}
    for name, entry_data in raw.items():
        catalog[name] = DatasetEntry.model_validate(entry_data)
    return catalog
