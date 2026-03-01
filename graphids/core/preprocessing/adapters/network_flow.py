"""Network flow domain adapter for UNSW-NB15 and CICIDS datasets.

Unlike CAN bus data (which uses temporal adjacency via ``shift(-1)``),
network flow data has **explicit** source and target identifiers from
IP addresses.  Edges represent actual network connections, not temporal
sequencing.

Supported formats:
    UNSW-NB15  — CSV with columns including srcip, sport, dstip, dsport,
                 plus 40+ statistical flow features and a binary 'label' column.
    CICIDS2017 — CSV with 'Source IP', 'Destination IP', 'Source Port',
                 'Destination Port', plus 78+ flow features and ' Label' column.

Feature alignment:
    CAN bus data has 8 continuous features (payload bytes).  Network flow
    data has many more (40–78).  To allow cross-domain model reuse, the
    adapter selects a configurable subset of ``num_features`` most
    informative flow features and normalizes them to [0, 1].  The model
    should use a projection layer (``nn.Linear(num_features, hidden_dim)``)
    when feature counts differ between domains.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import IRSchema, feature_columns
from ..vocabulary import EntityVocabulary
from .base import DomainAdapter

log = logging.getLogger(__name__)

# Default feature subsets (most discriminative flow features)
# These are common across UNSW-NB15 and CICIDS — adapt per dataset if needed.
UNSW_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit",
    "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "ct_src_ltm", "ct_srv_dst",
]

CICIDS_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
]

# Number of features to keep (matching schema)
DEFAULT_FLOW_FEATURES = 35


def _ip_to_int(ip_str) -> int | None:
    """Convert an IP address string to a 32-bit integer."""
    if pd.isna(ip_str):
        return None
    try:
        ip_str = str(ip_str).strip()
        parts = ip_str.split(".")
        if len(parts) == 4:
            return sum(int(p) << (8 * (3 - i)) for i, p in enumerate(parts))
        return hash(ip_str) & 0xFFFFFFFF  # IPv6 or hostname fallback
    except (ValueError, TypeError):
        return None


class NetworkFlowAdapter(DomainAdapter):
    """Adapter for network flow datasets (UNSW-NB15, CICIDS2017).

    Parameters
    ----------
    dataset_format : str
        One of ``"unsw"`` or ``"cicids"``.
    num_features : int
        Number of flow features to select (padded/truncated to this count).
    """

    def __init__(
        self,
        dataset_format: str = "unsw",
        num_features: int = DEFAULT_FLOW_FEATURES,
    ):
        if dataset_format not in ("unsw", "cicids"):
            raise ValueError(f"Unknown format: {dataset_format}. Use 'unsw' or 'cicids'.")
        self._format = dataset_format
        self._num_features = num_features
        self._schema = IRSchema(num_features=num_features)

    @property
    def schema(self) -> IRSchema:
        return self._schema

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def discover_files(
        self,
        root: str | Path,
        split: str = "train_",
    ) -> list[Path]:
        """Find network flow CSV files for a given split."""
        csv_files: list[Path] = []
        root_str = str(root)

        for dirpath, _dirnames, filenames in os.walk(root_str):
            dir_basename = os.path.basename(dirpath).lower()
            if split.lower().rstrip("_") not in dir_basename:
                continue
            for fname in filenames:
                if fname.endswith(".csv"):
                    csv_files.append(Path(dirpath) / fname)

        return sorted(csv_files)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def build_vocabulary(
        self,
        files: Sequence[str | Path],
    ) -> EntityVocabulary:
        """Build vocabulary from IP addresses in source/destination columns."""
        unique_ids: set[int] = set()

        src_col, dst_col = self._get_ip_columns()

        for i, f in enumerate(files):
            if i % 10 == 0:
                log.info("Scanning file %d/%d for entity IDs...", i + 1, len(files))
            try:
                df = pd.read_csv(f, usecols=[src_col, dst_col], dtype=str)
                for col in df.columns:
                    for val in df[col].dropna().unique():
                        converted = _ip_to_int(val)
                        if converted is not None:
                            unique_ids.add(converted)
                del df
            except Exception as e:
                log.warning("Could not scan %s: %s", f, e)

        log.info("Built vocabulary with %d entities (+ OOV)", len(unique_ids))
        return EntityVocabulary.build_from_ids(unique_ids)

    # ------------------------------------------------------------------
    # Raw → IR conversion
    # ------------------------------------------------------------------

    def read_and_convert(
        self,
        file_path: str | Path,
        vocab: EntityVocabulary,
    ) -> pd.DataFrame:
        """Read a network flow CSV and convert to IR format.

        Key difference from CAN bus: edges are **explicit** (src IP → dst IP),
        not derived from temporal adjacency (no ``shift(-1)``).
        """
        try:
            df = pd.read_csv(str(file_path), dtype=str, engine="python")
        except Exception as e:
            log.warning("Failed to read %s: %s", file_path, e)
            return pd.DataFrame(columns=self.schema.columns)

        # Normalize column names (CICIDS has leading/trailing spaces)
        df.columns = df.columns.str.strip()

        src_col, dst_col = self._get_ip_columns()
        label_col = self._get_label_column()
        feature_cols = self._get_feature_columns(df)

        # Validate required columns exist
        for col in [src_col, dst_col, label_col]:
            if col not in df.columns:
                log.warning("Missing column '%s' in %s", col, file_path)
                return pd.DataFrame(columns=self.schema.columns)

        # Convert IPs to integers
        df["_src_int"] = df[src_col].apply(_ip_to_int)
        df["_dst_int"] = df[dst_col].apply(_ip_to_int)

        # Drop rows with invalid IPs
        df = df.dropna(subset=["_src_int", "_dst_int"]).copy()

        # Apply vocabulary encoding
        oov = vocab.oov_index
        df["_src_enc"] = df["_src_int"].map(vocab.to_dict()).fillna(oov).astype(int)
        df["_dst_enc"] = df["_dst_int"].map(vocab.to_dict()).fillna(oov).astype(int)

        # Entity ID = source IP (same convention as CAN bus CAN_ID = source)
        df["entity_id"] = df["_src_enc"]

        # Extract and normalize features
        ir_features = self._extract_features(df, feature_cols)

        # Build label column
        labels = self._parse_labels(df[label_col])

        # Assemble IR DataFrame
        ir_data = {"entity_id": df["entity_id"].values}
        feat_names = feature_columns(self._num_features)
        for i, fname in enumerate(feat_names):
            ir_data[fname] = ir_features[:, i] if i < ir_features.shape[1] else 0.0

        ir_data["source_id"] = df["_src_enc"].values
        ir_data["target_id"] = df["_dst_enc"].values
        ir_data["label"] = labels

        return pd.DataFrame(ir_data)

    # ------------------------------------------------------------------
    # Internal: format-specific helpers
    # ------------------------------------------------------------------

    def _get_ip_columns(self) -> tuple[str, str]:
        if self._format == "unsw":
            return "srcip", "dstip"
        return "Source IP", "Destination IP"

    def _get_label_column(self) -> str:
        if self._format == "unsw":
            return "label"
        return "Label"

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Select feature columns that exist in the dataframe."""
        candidates = UNSW_FEATURES if self._format == "unsw" else CICIDS_FEATURES
        available = [c for c in candidates if c in df.columns]

        # Pad with any remaining numeric columns if we need more
        if len(available) < self._num_features:
            ip_cols = set(self._get_ip_columns())
            label_col = self._get_label_column()
            skip = ip_cols | {label_col, "attack_cat", "Attack", "id"}
            for col in df.columns:
                if col in skip or col in available:
                    continue
                if len(available) >= self._num_features:
                    break
                available.append(col)

        return available[:self._num_features]

    def _extract_features(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """Extract and normalize feature columns to [0, 1]."""
        n = len(df)
        features = np.zeros((n, self._num_features), dtype=np.float32)

        for i, col in enumerate(feature_cols):
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values.astype(np.float64)

            # Replace inf values
            vals = np.where(np.isinf(vals), 0.0, vals)

            # Min-max normalization to [0, 1]
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                vals = (vals - vmin) / (vmax - vmin)
            else:
                vals = np.zeros_like(vals)

            features[:, i] = vals.astype(np.float32)

        return features

    def _parse_labels(self, label_series: pd.Series) -> np.ndarray:
        """Parse labels to binary 0/1."""
        if self._format == "unsw":
            return pd.to_numeric(label_series, errors="coerce").fillna(0).astype(int).values
        # CICIDS: "BENIGN" → 0, anything else → 1
        return (label_series.str.strip().str.upper() != "BENIGN").astype(int).values
