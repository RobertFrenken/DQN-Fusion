"""Standardized Intermediate Representation (IR) schema for preprocessing.

All domain adapters produce DataFrames conforming to this schema before
handing off to GraphEngine. The IR decouples domain-specific parsing
(CAN bus hex, network flow CSVs, etc.) from domain-agnostic graph
construction.

Column layout (order matters for numpy conversion):
    entity_id      – dense integer ID for the node (post-vocabulary encoding)
    feature_0..N   – continuous features (e.g. normalized payload bytes)
    source_id      – dense integer ID of the edge source node
    target_id      – dense integer ID of the edge target node
    label          – binary label (0=normal, 1=attack)
    attack_type    – integer-coded attack type (0=normal, 1+=attack subtypes)
                     Optional: only present when include_attack_type=True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Column name constants used throughout the preprocessing pipeline
COL_ENTITY_ID = "entity_id"
COL_SOURCE_ID = "source_id"
COL_TARGET_ID = "target_id"
COL_LABEL = "label"
COL_ATTACK_TYPE = "attack_type"


def feature_columns(n: int) -> list[str]:
    """Return feature column names for *n* features."""
    return [f"feature_{i}" for i in range(n)]


@dataclass(frozen=True)
class IRSchema:
    """Describes and validates the Intermediate Representation DataFrame.

    Parameters
    ----------
    num_features : int
        Number of continuous feature columns (``feature_0`` … ``feature_{n-1}``).
    include_attack_type : bool
        If True, include an ``attack_type`` column after ``label``.
    """

    num_features: int
    include_attack_type: bool = False
    _required_prefix: list[str] = field(init=False, repr=False)
    _required_suffix: list[str] = field(init=False, repr=False)

    def __post_init__(self):
        prefix = [COL_ENTITY_ID] + feature_columns(self.num_features)
        suffix = [COL_SOURCE_ID, COL_TARGET_ID, COL_LABEL]
        if self.include_attack_type:
            suffix.append(COL_ATTACK_TYPE)
        object.__setattr__(self, "_required_prefix", prefix)
        object.__setattr__(self, "_required_suffix", suffix)

    @property
    def columns(self) -> list[str]:
        """Full ordered column list."""
        return self._required_prefix + self._required_suffix

    @property
    def feature_slice(self) -> slice:
        """Numpy column slice for the feature block."""
        start = 1  # after entity_id
        return slice(start, start + self.num_features)

    @property
    def col_source(self) -> int:
        """Numpy column index for source_id."""
        return len(self._required_prefix)

    @property
    def col_target(self) -> int:
        """Numpy column index for target_id."""
        return len(self._required_prefix) + 1

    @property
    def col_label(self) -> int:
        """Numpy column index for label."""
        return len(self._required_prefix) + 2

    @property
    def col_attack_type(self) -> int | None:
        """Numpy column index for attack_type, or None if not included."""
        if not self.include_attack_type:
            return None
        return len(self._required_prefix) + 3

    def validate(self, df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
        """Validate that *df* conforms to the IR schema.

        Parameters
        ----------
        df : pd.DataFrame
            Intermediate representation DataFrame.
        strict : bool
            If True, raise on any violation. If False, log warnings and return.

        Returns
        -------
        pd.DataFrame
            The validated (unchanged) DataFrame.

        Raises
        ------
        ValueError
            If strict=True and validation fails.
        """
        errors: list[str] = []
        expected = self.columns

        # Column presence and order
        if list(df.columns) != expected:
            missing = set(expected) - set(df.columns)
            extra = set(df.columns) - set(expected)
            if missing:
                errors.append(f"Missing columns: {sorted(missing)}")
            if extra:
                errors.append(f"Unexpected columns: {sorted(extra)}")
            if not missing and not extra:
                errors.append(f"Column order mismatch: expected {expected}, got {list(df.columns)}")

        if errors:
            msg = "IR schema validation failed:\n  " + "\n  ".join(errors)
            if strict:
                raise ValueError(msg)
            log.warning(msg)
            return df

        # Null check
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols) > 0:
            msg = f"IR contains nulls: {dict(null_cols)}"
            if strict:
                raise ValueError(msg)
            log.warning(msg)

        # Numeric dtype check
        for col in expected:
            if not np.issubdtype(df[col].dtype, np.number):
                msg = f"Column '{col}' has non-numeric dtype: {df[col].dtype}"
                if strict:
                    raise ValueError(msg)
                log.warning(msg)

        # Label values
        unique_labels = df[COL_LABEL].unique()
        if not set(unique_labels).issubset({0, 1, 0.0, 1.0}):
            msg = f"Label column has values outside {{0, 1}}: {sorted(unique_labels)}"
            if strict:
                raise ValueError(msg)
            log.warning(msg)

        return df


# Default schema for the CAN bus domain (8 data bytes → 8 features)
CAN_BUS_SCHEMA = IRSchema(num_features=8)

# CAN bus schema with attack type metadata
CAN_BUS_SCHEMA_WITH_ATTACK_TYPE = IRSchema(num_features=8, include_attack_type=True)

# Default schema for network flow domain (35 flow features)
NETWORK_FLOW_SCHEMA = IRSchema(num_features=35)
