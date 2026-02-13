"""Pandera schemas for data validation at ingest time.

Validates Parquet output after CSV → Parquet conversion. Keeps pandera as an
optional dependency — import errors are caught by the caller in ingest.py.
"""
from __future__ import annotations

import pandera.pandas as pa

CAN_PARQUET_SCHEMA = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(float, pa.Check.ge(0), nullable=False),
        "id": pa.Column(int, pa.Check.ge(0), nullable=False),
        "data_field": pa.Column(str, nullable=False),
        "label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "source_file": pa.Column(str, nullable=False),
    },
    strict=False,
    coerce=True,
)
