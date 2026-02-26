"""CAN bus domain adapter.

Encapsulates all CAN-bus-specific knowledge:
- 4-column CSV format: [Timestamp, arbitration_id, data_field, attack]
- Hex-encoded CAN IDs and payload bytes
- Temporal adjacency: edge topology via ``CAN_ID.shift(-1)``
- 8-byte payload extraction from hex data_field
- Attack type exclusion (suppress, masquerade)

The ``shift(-1)`` temporal adjacency is the most critical implicit
assumption: message N connects to message N+1. This is CAN-specific
and does NOT belong in the GraphEngine.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

from config.constants import EXCLUDED_ATTACK_TYPES, MAX_DATA_BYTES
from ..schema import IRSchema, CAN_BUS_SCHEMA, feature_columns
from ..vocabulary import EntityVocabulary
from .base import DomainAdapter

log = logging.getLogger(__name__)

# Sentinel for hex character validation
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def _safe_hex_to_int(value) -> int | None:
    """Safely convert hex string or numeric value to integer."""
    if pd.isna(value):
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if all(c in _HEX_CHARS for c in value):
                return int(value, 16)
            return int(value)
        return int(value)
    except (ValueError, TypeError):
        return None


class CANBusAdapter(DomainAdapter):
    """Adapter for CAN bus CSV datasets.

    Parameters
    ----------
    chunk_size : int
        Number of CSV rows to process per chunk (for memory efficiency).
    excluded_attacks : sequence of str
        Attack type substrings to exclude from file discovery.
    """

    def __init__(
        self,
        chunk_size: int = 5000,
        excluded_attacks: Sequence[str] = EXCLUDED_ATTACK_TYPES,
    ):
        self._chunk_size = chunk_size
        self._excluded_attacks = list(excluded_attacks)

    @property
    def schema(self) -> IRSchema:
        return CAN_BUS_SCHEMA

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def discover_files(
        self,
        root: str | Path,
        split: str = "train_",
    ) -> list[Path]:
        """Find CAN bus CSV files for a given split.

        Walks the directory tree, matching folders whose basename
        contains *split* (case-insensitive). Excludes files matching
        any substring in ``excluded_attacks``.
        """
        csv_files: list[Path] = []
        root_str = str(root)

        for dirpath, _dirnames, filenames in os.walk(root_str):
            dir_basename = os.path.basename(dirpath).lower()
            if split.lower().rstrip("_") not in dir_basename:
                continue
            for fname in filenames:
                if not fname.endswith(".csv"):
                    continue
                if any(at in fname.lower() for at in self._excluded_attacks):
                    continue
                csv_files.append(Path(dirpath) / fname)

        return sorted(csv_files)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def build_vocabulary(
        self,
        files: Sequence[str | Path],
    ) -> EntityVocabulary:
        """Build vocabulary by scanning the arbitration_id column."""
        return EntityVocabulary.build_from_files(
            [str(f) for f in files],
            id_column=1,
            hex_convert=True,
        )

    # ------------------------------------------------------------------
    # Raw → IR conversion
    # ------------------------------------------------------------------

    def read_and_convert(
        self,
        file_path: str | Path,
        vocab: EntityVocabulary,
    ) -> pd.DataFrame:
        """Read a CAN bus CSV and convert to IR format.

        Processing steps:
        1. Read CSV in chunks (streaming)
        2. Parse hex data_field into byte columns
        3. Build temporal adjacency (shift(-1))
        4. Convert hex to int, apply vocabulary encoding
        5. Normalize payload bytes to [0, 1]
        6. Rename to IR column layout
        """
        chunks = self._read_chunks(str(file_path))
        if not chunks:
            return pd.DataFrame(columns=self.schema.columns)

        ir_chunks = []
        for chunk in chunks:
            ir = self._chunk_to_ir(chunk, vocab)
            if not ir.empty:
                ir_chunks.append(ir)

        if not ir_chunks:
            return pd.DataFrame(columns=self.schema.columns)

        return pd.concat(ir_chunks, ignore_index=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_chunks(self, csv_path: str) -> list[pd.DataFrame]:
        """Read a CSV file in chunks. Returns list of raw chunks."""
        chunks = []
        try:
            reader = pd.read_csv(
                csv_path, chunksize=self._chunk_size, engine="python", dtype=str,
            )
            for chunk in reader:
                chunk.columns = ["Timestamp", "arbitration_id", "data_field", "attack"]
                chunk.rename(columns={"arbitration_id": "CAN ID"}, inplace=True)
                chunks.append(chunk)
        except Exception as e:
            log.debug("Chunked reading failed for %s: %s. Trying full read.", csv_path, e)
            try:
                full = pd.read_csv(csv_path, engine="python", dtype=str)
                full.columns = ["Timestamp", "arbitration_id", "data_field", "attack"]
                full.rename(columns={"arbitration_id": "CAN ID"}, inplace=True)
                chunks.append(full)
            except Exception as e2:
                log.warning("Failed to read %s: %s", csv_path, e2)
        return chunks

    def _chunk_to_ir(self, chunk: pd.DataFrame, vocab: EntityVocabulary) -> pd.DataFrame:
        """Convert a raw CAN bus chunk to IR format."""
        # Parse hex data_field into byte columns
        chunk["data_field"] = chunk["data_field"].astype(str).fillna("").str.strip()
        chunk["DLC"] = chunk["data_field"].str.len() // 2

        data_field = chunk["data_field"].values
        byte_cols = []
        for i in range(MAX_DATA_BYTES):
            start = i * 2
            end = start + 2
            col_name = f"Data{i + 1}"
            chunk[col_name] = [s[start:end] if len(s) >= end else "00" for s in data_field]
            byte_cols.append(col_name)

        # Pad short payloads
        mask = chunk["DLC"] < MAX_DATA_BYTES
        for i in range(MAX_DATA_BYTES):
            chunk.loc[mask & (chunk["DLC"] <= i), byte_cols[i]] = "00"
        chunk.fillna("00", inplace=True)

        # --- CRITICAL: Temporal adjacency (CAN-bus specific) ---
        # Message N's target is message N+1's CAN ID
        chunk["Source"] = chunk["CAN ID"]
        chunk["Target"] = chunk["CAN ID"].shift(-1)

        # Convert hex to int
        hex_columns = ["CAN ID", "Source", "Target"] + byte_cols
        for col in hex_columns:
            chunk[col] = chunk[col].apply(_safe_hex_to_int)

        # Apply vocabulary encoding to ID columns
        oov = vocab.oov_index
        for col in ["CAN ID", "Source", "Target"]:
            chunk[col] = chunk[col].map(vocab.to_dict()).fillna(oov).astype(int)

        # Drop last row (no valid target after shift)
        chunk = chunk.iloc[:-1].copy()
        chunk["label"] = chunk["attack"].astype(int)

        # Normalize payload bytes to [0, 1]
        for col in byte_cols:
            chunk[col] = chunk[col] / 255.0

        # Rename to IR column layout
        feat_names = feature_columns(MAX_DATA_BYTES)
        rename_map = {byte_cols[i]: feat_names[i] for i in range(MAX_DATA_BYTES)}
        rename_map["CAN ID"] = "entity_id"
        rename_map["Source"] = "source_id"
        rename_map["Target"] = "target_id"
        chunk.rename(columns=rename_map, inplace=True)

        return chunk[self.schema.columns]
