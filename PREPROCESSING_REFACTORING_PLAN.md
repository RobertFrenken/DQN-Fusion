# Preprocessing Refactoring Plan: From CAN-Bus Script to Domain-Agnostic Data Engine

## Part 1: Diagnosis — What's Actually Wrong

I'm going to be surgical here. Your preprocessing.py is 734 lines and I count **7 distinct architectural problems**, not just "it's messy." Understanding each one precisely determines the refactoring strategy.

### Problem 1: CAN-bus assumptions are load-bearing walls, not decorations

These aren't surface-level references you can find-and-replace. The CAN protocol *shapes the data structures*:

```
Line 227:  chunk.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
Line 266:  chunk['DLC'] = chunk['data_field'].str.len() // 2
Line 270:  chunk[f'Data{i+1}'] = [s[start:end] if len(s) >= end else '00' for s in data_field]
Line 277:  chunk['Source'] = chunk['CAN ID']
Line 278:  chunk['Target'] = chunk['CAN ID'].shift(-1)
```

The `shift(-1)` on line 278 is the most consequential: it defines your graph topology as "message N talks to message N+1" — a temporal adjacency assumption specific to CAN bus sequential broadcasts. Network flow data (UNSW-NB15, CICIDS) already *has* source/target IPs. IoT data (BoT-IoT) has device-to-device communication pairs. Each domain needs a fundamentally different edge construction strategy.

### Problem 2: The "magic number" column indexing is a time bomb

```python
COL_CAN_ID = 0
COL_DATA_START = 1
COL_DATA_END = 9    # exclusive — assumes exactly 8 data bytes
COL_SOURCE = -3
COL_TARGET = -2
COL_LABEL = -1
```

Then later:
```python
node_features[i, :COL_DATA_END] = node_data[:, :COL_DATA_END].mean(axis=0)  # line 469
```

This uses negative indexing into a numpy array whose shape is implicitly defined by the CAN protocol (1 ID + 8 bytes + source + target + label = 12 columns). If you add a column, remove a column, or process a dataset with different feature counts, every downstream function silently produces wrong results. No error, just wrong tensors fed to your GNN.

### Problem 3: Two redundant ID mapping codepaths

```
build_id_mapping_from_normal()    — reads FULL CSVs, filters by attack==0, builds mapping
build_lightweight_id_mapping()    — reads only column 1, builds mapping from all IDs
```

`build_id_mapping_from_normal` is called nowhere in the current main path (`graph_creation` uses `build_lightweight_id_mapping`). It's dead code that still shapes how you think about the system. More importantly, the concept of "CAN ID mapping" is domain-specific — network flow data uses IP addresses, IoT data uses device IDs. The mapping concept needs to become generic "entity vocabulary."

### Problem 4: Feature computation is O(N×E) per window with Python loops

```python
# Line 409 — Python loop over edges
for i, (src, tgt) in enumerate(unique_edges):
    edge_mask = (source == src) & (target == tgt)     # O(W) per edge
    edge_positions = positions[edge_mask]
    ...

# Line 462 — Python loop over nodes
for i, node in enumerate(nodes):
    node_mask = source == node                          # O(W) per node
    node_data = window_data[node_mask]
    ...
```

For a window of 100 messages with 30 unique nodes and 50 unique edges, that's 50×100 + 30×100 = 8,000 comparisons *per window*, all in Python. With 100K+ windows per dataset, this is your bottleneck. The operations inside the loops are vectorizable.

### Problem 5: No validation at data boundaries

The only validation is `GraphDataset._validate_data_consistency()` which checks tensor shapes *after* graph construction. But the real failure modes are:

- Corrupt hex values silently become `None` → `fillna(0)` → wrong features
- Files with different column counts get `chunk.columns = [...]` forced on them → misaligned data
- `NaN` values from `safe_hex_to_int` propagate through `shift(-1)` into Source/Target → phantom edges
- No schema validation between raw CSV → processed DataFrame → numpy array transitions

### Problem 6: Memory architecture is "hold everything in RAM"

```python
all_graphs = []
for csv_file in csv_files:
    df = dataset_creation_streaming(csv_file, ...)  # Streams chunks but concat's full file
    graphs = create_graphs_numpy(df, ...)
    all_graphs.extend(graphs)   # Accumulates ALL graphs in memory
```

The streaming within a file helps, but the outer loop accumulates every PyG `Data` object. For 6 datasets × 100K+ graphs each, you're holding millions of graph objects. This is why you need scratch caching — but the caching should be *inside* the pipeline, not a manual afterthought.

### Problem 7: No separation between "what to compute" and "how to compute it"

Every function directly does I/O, transformation, and feature engineering in one pass. There's no place to:
- Swap graph construction strategy without rewriting `_process_dataframe_chunk`
- Test edge features independently of file loading
- Profile which stage is slow
- Cache intermediate results (processed DataFrames before windowing)

---

## Part 2: The Target Datasets

For ICML generalizability, these are the domains your system needs to handle:

| Domain | Datasets | Raw Format | Nodes Are | Edges Are | Label Column |
|--------|----------|-----------|-----------|-----------|-------------|
| **CAN Bus** (current) | OTIDS, Car-Hacking, SynCAN | CSV: timestamp, arb_id, data_field, attack | CAN IDs (hex) | Temporal adjacency (shift) | Binary attack flag |
| **Network Flow** | UNSW-NB15, CICIDS-2017/2018, CSE-CIC | CSV: 49-83 features incl. src_ip, dst_ip, src_port, dst_port, protocol | IP addresses | (src_ip, dst_ip) pairs — **explicit** | Attack category string |
| **IoT Traffic** | BoT-IoT, ToN-IoT, ACI-IoT-2023 | CSV: flow features, device IDs | Device/IP endpoints | Communication pairs | Multi-class label |
| **Industrial** | OPCUA, SWaT, WADI | CSV/PCAP: sensor readings, control commands | Sensor/actuator IDs | Process flow topology | Attack flag |
| **NetFlow** | NF-UQ-NIDS-v2 (unified) | CSV: 45 NetFlow features, standardized | IP:port endpoints | Bidirectional flows | Unified labels |

**Key insight**: CAN bus is the *hardest* case because edges are implicit (temporal adjacency). Every other domain has explicit source/destination fields. Your current code solves the hard case; the refactoring needs to make the easy cases easy too.

---

## Part 3: The Architecture

### Core Principle: Separate the Four Concerns

```
┌─────────────────────────────────────────────────────────────────┐
│                    Current: preprocessing.py                     │
│  find_csv → read_csv → hex_convert → shift(-1) → window → PyG  │
│                    734 lines, one file                           │
└─────────────────────────────────────────────────────────────────┘

                              ▼ refactor into ▼

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Ingest     │  │   Transform  │  │   Construct  │  │   Materialize│
│              │  │              │  │              │  │              │
│ Read raw     │→ │ Normalize to │→ │ Build PyG    │→ │ Cache, shard │
│ domain data  │  │ standard     │  │ Data objects │  │ write .pt    │
│              │  │ tabular form │  │ from windows │  │              │
│ Domain-      │  │ Domain-      │  │ Domain-      │  │ Domain-      │
│ SPECIFIC     │  │ SPECIFIC     │  │ AGNOSTIC     │  │ AGNOSTIC     │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
   Per-domain       Per-domain        Shared code       Shared code
   adapter          adapter           (graph engine)    (I/O layer)
```

The left two boxes are domain-specific adapters. The right two boxes are the shared graph engine. The boundary between them is a **standardized intermediate representation** — a DataFrame/dict with a defined schema.

### The Intermediate Representation (IR)

Every domain adapter must produce this standard form before graph construction:

```python
@dataclass
class ProcessedRecord:
    """Standard intermediate representation between domain adapters and graph engine."""
    source_id: int          # Categorical index of source entity
    target_id: int          # Categorical index of target entity  
    features: np.ndarray    # (num_feature_cols,) — numeric, normalized to [0,1]
    label: int              # 0=normal, 1=anomaly (binary) or multi-class int
    timestamp: float        # Normalized position in sequence [0,1]
```

Or equivalently as a DataFrame schema (validated by Pandera):

```python
import pandera as pa
from pandera.typing import Series

class StandardizedSchema(pa.DataFrameModel):
    """Schema that all domain adapters must produce."""
    source_id: Series[int] = pa.Field(ge=0)
    target_id: Series[int] = pa.Field(ge=0)
    label: Series[int] = pa.Field(ge=0)
    timestamp: Series[float] = pa.Field(ge=0.0, le=1.0)
    
    # Dynamic feature columns: feat_0, feat_1, ..., feat_N
    # Validated separately since count varies by domain
    
    class Config:
        strict = True  # No extra columns allowed
        coerce = True  # Auto-cast types
```

### The Domain Adapter Protocol

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class DomainConfig(BaseModel):
    """Base config for all domain adapters. Extend per domain."""
    window_size: int = 100
    stride: int = 50
    label_column: str = "label"
    excluded_labels: list[str] = []
    chunk_size: int = 10_000
    
    class Config:
        frozen = True  # Immutable after creation

class DomainAdapter(ABC):
    """Protocol that every domain must implement."""
    
    @abstractmethod
    def discover_files(self, root: Path, split: str = "train") -> list[Path]:
        """Find raw data files for this domain."""
        ...
    
    @abstractmethod  
    def read_raw(self, path: Path, chunk_size: int = 10_000) -> Iterator[pd.DataFrame]:
        """Stream raw data in chunks. Handles format quirks."""
        ...
    
    @abstractmethod
    def build_vocabulary(self, paths: list[Path]) -> EntityVocabulary:
        """Scan files to build entity-to-index mapping (CAN IDs, IPs, etc.)."""
        ...
    
    @abstractmethod
    def to_standard_form(self, chunk: pd.DataFrame, vocab: EntityVocabulary) -> pd.DataFrame:
        """Transform raw chunk → StandardizedSchema DataFrame."""
        ...
    
    @property
    @abstractmethod
    def num_features(self) -> int:
        """Number of feature columns this domain produces."""
        ...
    
    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Names of feature columns for interpretability."""
        ...
```

### CAN Bus Adapter (your current code, cleaned)

```python
class CANBusConfig(DomainConfig):
    """CAN-bus specific configuration."""
    max_data_bytes: int = 8
    hex_columns: list[str] = ["CAN ID", "Source", "Target"]
    # Replaces: MAX_DATA_BYTES, EXCLUDED_ATTACK_TYPES from config/constants.py
    excluded_attack_types: list[str] = ["fuzzy"]  
    edge_strategy: str = "temporal_adjacency"  # CAN-specific: shift(-1)

class CANBusAdapter(DomainAdapter):
    def __init__(self, config: CANBusConfig):
        self.config = config
    
    def discover_files(self, root: Path, split: str = "train") -> list[Path]:
        """Your current find_csv_files, cleaned up."""
        pattern = f"*{split}*/**/*.csv"
        return [
            p for p in root.rglob(pattern)
            if not any(exc in p.name.lower() for exc in self.config.excluded_attack_types)
        ]
    
    def build_vocabulary(self, paths: list[Path]) -> EntityVocabulary:
        """Your current build_lightweight_id_mapping, generalized."""
        unique_ids = set()
        for path in paths:
            df = pd.read_csv(path, usecols=[1], dtype=str)
            for val in df.iloc[:, 0].dropna().unique():
                converted = safe_hex_to_int(val)
                if converted is not None:
                    unique_ids.add(converted)
        return EntityVocabulary.from_values(sorted(unique_ids))
    
    def to_standard_form(self, chunk: pd.DataFrame, vocab: EntityVocabulary) -> pd.DataFrame:
        """Your current _process_dataframe_chunk, producing StandardizedSchema."""
        # Parse CAN-specific format
        chunk.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
        
        # Extract payload bytes (CAN-specific: hex pairs from data_field)
        data_field = chunk['data_field'].astype(str).fillna('').str.strip()
        features = np.zeros((len(chunk), self.config.max_data_bytes), dtype=np.float32)
        for i in range(self.config.max_data_bytes):
            start, end = i * 2, i * 2 + 2
            byte_vals = [safe_hex_to_int(s[start:end]) if len(s) >= end else 0 
                         for s in data_field.values]
            features[:, i] = np.array(byte_vals, dtype=np.float32) / 255.0
        
        # CAN-specific edge strategy: temporal adjacency
        can_ids = chunk['arbitration_id'].apply(safe_hex_to_int)
        source_ids = vocab.encode(can_ids.values)
        target_ids = np.roll(source_ids, -1)  # shift(-1) equivalent
        
        # Build standard form
        result = pd.DataFrame({
            'source_id': source_ids,
            'target_id': target_ids,
            'label': chunk['attack'].astype(int),
            'timestamp': np.linspace(0, 1, len(chunk)),
        })
        
        # Add feature columns
        for i in range(self.config.max_data_bytes):
            result[f'feat_{i}'] = features[:, i]
        
        # Drop last row (no valid target from shift)
        return result.iloc[:-1].reset_index(drop=True)
    
    @property
    def num_features(self) -> int:
        return self.config.max_data_bytes  # 8
    
    @property
    def feature_names(self) -> list[str]:
        return [f"payload_byte_{i}" for i in range(self.config.max_data_bytes)]
```

### Network Flow Adapter (NEW — for UNSW-NB15, CICIDS)

```python
class NetworkFlowConfig(DomainConfig):
    """Network flow dataset configuration."""
    source_col: str = "srcip"
    dest_col: str = "dstip"
    label_col: str = "label"            # or "attack_cat" for multi-class
    feature_cols: list[str] | None = None  # None = auto-detect numeric columns
    timestamp_col: str | None = None       # "Stime" for UNSW-NB15
    
    # Which dataset variant
    variant: str = "unsw-nb15"  # or "cicids-2017", "bot-iot", etc.

class NetworkFlowAdapter(DomainAdapter):
    def __init__(self, config: NetworkFlowConfig):
        self.config = config
        self._feature_cols: list[str] = []
    
    def discover_files(self, root: Path, split: str = "train") -> list[Path]:
        return sorted(root.glob(f"*{split}*.csv"))
    
    def build_vocabulary(self, paths: list[Path]) -> EntityVocabulary:
        """Build IP-to-index mapping from source/dest columns."""
        unique_entities = set()
        for path in paths:
            df = pd.read_csv(path, usecols=[self.config.source_col, self.config.dest_col],
                             dtype=str, nrows=None)
            unique_entities.update(df[self.config.source_col].dropna().unique())
            unique_entities.update(df[self.config.dest_col].dropna().unique())
        return EntityVocabulary.from_values(sorted(unique_entities))
    
    def to_standard_form(self, chunk: pd.DataFrame, vocab: EntityVocabulary) -> pd.DataFrame:
        """Network flow → standard form. Edges are EXPLICIT (src_ip → dst_ip)."""
        # Encode endpoints
        source_ids = vocab.encode(chunk[self.config.source_col].values)
        target_ids = vocab.encode(chunk[self.config.dest_col].values)
        
        # Auto-detect or use specified feature columns
        if self.config.feature_cols is None:
            self._feature_cols = [c for c in chunk.select_dtypes(include=[np.number]).columns
                                   if c not in {self.config.label_col, self.config.source_col, 
                                               self.config.dest_col, self.config.timestamp_col}]
        else:
            self._feature_cols = self.config.feature_cols
        
        # Extract and normalize features
        features = chunk[self._feature_cols].fillna(0).values.astype(np.float32)
        # Per-column min-max normalization
        col_min = features.min(axis=0)
        col_max = features.max(axis=0)
        range_ = col_max - col_min
        range_[range_ == 0] = 1.0
        features = (features - col_min) / range_
        
        # Labels
        label_col = chunk[self.config.label_col]
        if label_col.dtype == object:
            # Multi-class string labels → binary for now
            labels = (label_col.str.lower() != 'normal').astype(int)
        else:
            labels = label_col.astype(int)
        
        # Timestamps
        if self.config.timestamp_col and self.config.timestamp_col in chunk.columns:
            ts = chunk[self.config.timestamp_col].astype(float)
            ts = (ts - ts.min()) / max(ts.max() - ts.min(), 1e-8)
        else:
            ts = np.linspace(0, 1, len(chunk))
        
        result = pd.DataFrame({
            'source_id': source_ids,
            'target_id': target_ids,
            'label': labels,
            'timestamp': ts,
        })
        for i, col_name in enumerate(self._feature_cols):
            result[f'feat_{i}'] = features[:, i]
        
        return result
```

### The Shared Graph Engine (domain-agnostic)

This is the code that currently lives in `create_graph_from_window`, `compute_edge_features`, and `compute_node_features` — but refactored to work on StandardizedSchema DataFrames:

```python
class GraphEngine:
    """Domain-agnostic graph construction from standardized data."""
    
    def __init__(self, window_size: int, stride: int, num_features: int):
        self.window_size = window_size
        self.stride = stride
        self.num_features = num_features
        # Feature column names in standard form
        self._feat_cols = [f'feat_{i}' for i in range(num_features)]
    
    def create_graphs(self, df: pd.DataFrame) -> list[Data]:
        """Sliding window → PyG Data objects. Input must be StandardizedSchema."""
        arr = df[['source_id', 'target_id', 'label', 'timestamp'] + self._feat_cols].to_numpy()
        
        n_windows = max(1, (len(arr) - self.window_size) // self.stride + 1)
        
        # This is the parallelization point
        return [self._window_to_graph(arr[i*self.stride : i*self.stride + self.window_size])
                for i in range(n_windows)]
    
    def _window_to_graph(self, window: np.ndarray) -> Data:
        """Single window → PyG Data. FULLY VECTORIZED — no Python loops."""
        src_col, tgt_col, lbl_col, ts_col = 0, 1, 2, 3
        feat_start = 4
        
        source = window[:, src_col].astype(np.int64)
        target = window[:, tgt_col].astype(np.int64)
        labels = window[:, lbl_col]
        timestamps = window[:, ts_col]
        features = window[:, feat_start:]
        
        # Unique edges and nodes
        edges = np.column_stack((source, target))
        unique_edges, inverse, edge_counts = np.unique(
            edges, axis=0, return_inverse=True, return_counts=True
        )
        nodes = np.unique(np.concatenate((source, target)))
        node_to_idx = {int(n): i for i, n in enumerate(nodes)}
        
        # Edge index (vectorized mapping)
        edge_src = np.array([node_to_idx[int(e[0])] for e in unique_edges])
        edge_tgt = np.array([node_to_idx[int(e[1])] for e in unique_edges])
        edge_index = torch.tensor(np.stack([edge_src, edge_tgt]), dtype=torch.long)
        
        # Edge features (VECTORIZED — replaces the O(E×W) loop)
        edge_attr = self._compute_edge_features_vectorized(
            source, target, timestamps, unique_edges, edge_counts, inverse
        )
        
        # Node features (VECTORIZED — replaces the O(N×W) loop)
        x = self._compute_node_features_vectorized(
            source, features, timestamps, nodes
        )
        
        # Graph label
        y = torch.tensor(1 if np.any(labels == 1) else 0, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    def _compute_edge_features_vectorized(
        self, source, target, timestamps, unique_edges, edge_counts, inverse
    ) -> torch.Tensor:
        """
        Vectorized edge feature computation.
        
        Replaces the Python loop in compute_edge_features (lines 409-438).
        Key technique: use `inverse` from np.unique to group-aggregate.
        """
        W = len(source)
        E = len(unique_edges)
        
        # --- Features that are already vectorized ---
        feats = np.zeros((E, 11), dtype=np.float32)
        feats[:, 0] = edge_counts                    # raw count
        feats[:, 1] = edge_counts / W                # frequency
        
        # --- Temporal features via scatter/group-by ---
        # For each message position, we know which unique edge it belongs to (inverse)
        positions = np.arange(W, dtype=np.float32)
        
        # First and last occurrence per edge (vectorized with np.ufunc.reduceat or pandas)
        # Using a groupby approach for clarity:
        edge_groups = pd.DataFrame({
            'edge_idx': inverse, 
            'position': positions,
            'timestamp': timestamps
        })
        grouped = edge_groups.groupby('edge_idx')['position']
        
        first_pos = grouped.first().values
        last_pos = grouped.last().values
        mean_interval = grouped.apply(lambda x: np.diff(x.values).mean() if len(x) > 1 else 0).values
        std_interval = grouped.apply(lambda x: np.diff(x.values).std() if len(x) > 1 else 0).values
        
        feats[:, 2] = mean_interval
        feats[:, 3] = std_interval
        feats[:, 4] = 1.0 / (1.0 + std_interval)
        feats[:, 5] = first_pos / W
        feats[:, 6] = last_pos / W
        feats[:, 7] = (last_pos - first_pos) / W
        
        # --- Structural features (already fast) ---
        edge_set = set(map(tuple, unique_edges))
        feats[:, 8] = np.array([float((e[1], e[0]) in edge_set) for e in unique_edges])
        
        # Degree computation (vectorized)
        all_nodes = np.concatenate([source, target])
        _, deg_counts = np.unique(all_nodes, return_counts=True)
        node_deg_map = dict(zip(*np.unique(all_nodes, return_counts=True)))
        
        src_deg = np.array([node_deg_map.get(e[0], 0) for e in unique_edges], dtype=np.float32)
        tgt_deg = np.array([node_deg_map.get(e[1], 0) for e in unique_edges], dtype=np.float32)
        feats[:, 9] = src_deg * tgt_deg
        feats[:, 10] = src_deg / np.maximum(tgt_deg, 1e-8)
        
        return torch.tensor(feats, dtype=torch.float)
    
    def _compute_node_features_vectorized(
        self, source, features, timestamps, nodes
    ) -> torch.Tensor:
        """
        Vectorized node feature computation.
        
        Replaces the Python loop in compute_node_features (lines 462-487).
        """
        W = len(source)
        N = len(nodes)
        F = features.shape[1]
        
        # Map each message to its node index
        node_idx_map = {int(n): i for i, n in enumerate(nodes)}
        msg_node_idx = np.array([node_idx_map[int(s)] for s in source])
        
        # Scatter-add features per node (vectorized)
        node_feat_sum = np.zeros((N, F), dtype=np.float64)
        node_counts = np.zeros(N, dtype=np.float64)
        
        np.add.at(node_feat_sum, msg_node_idx, features)
        np.add.at(node_counts, msg_node_idx, 1)
        
        # Mean features
        safe_counts = np.maximum(node_counts, 1)
        node_feat_mean = (node_feat_sum / safe_counts[:, None]).astype(np.float32)
        
        # Last timestamp per node
        last_ts = np.zeros(N, dtype=np.float32)
        np.maximum.at(last_ts, msg_node_idx, timestamps.astype(np.float32))
        
        # Normalized occurrence count
        count_min, count_max = node_counts.min(), node_counts.max()
        if count_max > count_min:
            norm_counts = ((node_counts - count_min) / (count_max - count_min)).astype(np.float32)
        else:
            norm_counts = node_counts.astype(np.float32)
        
        # Combine: [features..., norm_count, last_timestamp]
        x = np.column_stack([node_feat_mean, norm_counts, last_ts])
        
        return torch.tensor(x, dtype=torch.float)
```

### EntityVocabulary (replaces both id_mapping functions)

```python
class EntityVocabulary:
    """Generic entity-to-index mapping with OOV handling.
    
    Replaces: build_id_mapping_from_normal, build_lightweight_id_mapping,
    apply_dynamic_id_mapping. Works for CAN IDs, IP addresses, device IDs, etc.
    """
    
    def __init__(self, entity_to_idx: dict, oov_idx: int):
        self._map = entity_to_idx
        self._oov = oov_idx
        self._reverse = {v: k for k, v in entity_to_idx.items()}
    
    @classmethod
    def from_values(cls, sorted_values: list) -> "EntityVocabulary":
        mapping = {val: idx for idx, val in enumerate(sorted_values)}
        oov_idx = len(mapping)
        mapping['__OOV__'] = oov_idx
        return cls(mapping, oov_idx)
    
    def encode(self, values: np.ndarray) -> np.ndarray:
        """Vectorized encoding with OOV handling."""
        return np.array([self._map.get(v, self._oov) for v in values], dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self._map)
    
    def save(self, path: Path):
        """Persist for reproducibility."""
        torch.save({'map': self._map, 'oov': self._oov}, path)
    
    @classmethod
    def load(cls, path: Path) -> "EntityVocabulary":
        data = torch.load(path)
        return cls(data['map'], data['oov'])
```

---

## Part 4: The File Layout

```
src/preprocessing/
├── __init__.py              # Public API: preprocess_dataset()
├── schema.py                # StandardizedSchema (Pandera), ProcessedRecord
├── vocabulary.py            # EntityVocabulary (replaces all id_mapping code)
├── engine.py                # GraphEngine (domain-agnostic windowing + feature computation)
├── dataset.py               # GraphDataset (PyG wrapper, mostly unchanged)
├── cache.py                 # ShardedCache: write/read .pt shards to scratch
├── adapters/
│   ├── __init__.py
│   ├── base.py              # DomainAdapter ABC, DomainConfig base
│   ├── can_bus.py           # CANBusAdapter (your current preprocessing, cleaned)
│   ├── network_flow.py      # NetworkFlowAdapter (UNSW-NB15, CICIDS, BoT-IoT)
│   └── netflow.py           # NetFlowAdapter (NF-UQ-NIDS-v2, standardized 45-feature)
└── parallel.py              # SLURM-aware parallel file processing (multiprocessing or Dask)
```

### Registration & Config Integration

```python
# In config/preprocessing.py (extends your existing Pydantic config system)

from pydantic import BaseModel, Field
from typing import Literal

class PreprocessingConfig(BaseModel):
    """Top-level preprocessing config. Integrates with your existing config/ system."""
    domain: Literal["can_bus", "network_flow", "netflow", "iot"] = "can_bus"
    window_size: int = 100
    stride: int = 50
    chunk_size: int = 10_000
    cache_dir: str = "/fs/scratch/PAS1266/graph_cache"
    num_workers: int = 4  # Parallel file processing
    
    # Domain-specific config (discriminated union)
    can_bus: CANBusConfig | None = None
    network_flow: NetworkFlowConfig | None = None
    
    class Config:
        frozen = True

# Usage from your existing pipeline stages:
from config import load_config
from src.preprocessing import preprocess_dataset

cfg = load_config("experiments/unsw_nb15.yaml")
dataset, vocab = preprocess_dataset(cfg.preprocessing)
```

### The Public API (what your pipeline stages call)

```python
# src/preprocessing/__init__.py

def preprocess_dataset(
    config: PreprocessingConfig,
    root: Path,
    split: str = "train",
) -> tuple[GraphDataset, EntityVocabulary]:
    """
    One-call entry point. Handles caching, parallelism, domain dispatch.
    
    This replaces graph_creation() from the current preprocessing.py.
    """
    # 1. Check cache
    cache = ShardedCache(config.cache_dir)
    cache_key = cache.compute_key(config, root, split)
    if cache.exists(cache_key):
        logger.info("Loading cached graphs from %s", cache_key)
        return cache.load(cache_key)
    
    # 2. Resolve domain adapter
    adapter = get_adapter(config)
    
    # 3. Discover files
    files = adapter.discover_files(root, split)
    logger.info("Found %d files for %s/%s", len(files), config.domain, split)
    
    # 4. Build vocabulary (scan pass)
    vocab = adapter.build_vocabulary(files)
    
    # 5. Process files → standard form → graphs  (parallel over files)
    engine = GraphEngine(config.window_size, config.stride, adapter.num_features)
    
    all_graphs = process_files_parallel(
        files=files,
        adapter=adapter,
        vocab=vocab,
        engine=engine,
        chunk_size=config.chunk_size,
        num_workers=config.num_workers,
    )
    
    # 6. Validate & wrap
    dataset = GraphDataset(all_graphs)
    dataset.print_stats()
    
    # 7. Cache for next time
    cache.save(cache_key, dataset, vocab)
    
    return dataset, vocab
```

---

## Part 5: Parallelization Strategy

### Current bottleneck profile (estimated):

```
File I/O (pd.read_csv, engine='python'):    ~30% of time
Hex conversion (safe_hex_to_int per cell):  ~25% of time  ← Python loop over every cell
Graph feature computation:                   ~35% of time  ← Python loops in compute_*
PyG Data object creation:                    ~10% of time
```

### Three-level parallelism:

**Level 1: File-level** (multiprocessing.Pool or Dask)
```python
def process_files_parallel(files, adapter, vocab, engine, chunk_size, num_workers):
    """Process multiple CSV files in parallel."""
    fn = partial(_process_single_file, adapter=adapter, vocab=vocab, 
                 engine=engine, chunk_size=chunk_size)
    
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(fn, files)
    
    return [graph for file_graphs in results for graph in file_graphs]
```

This is the easiest win. Your current code processes files sequentially. On a 48-core OSC node, even `num_workers=8` gives ~5-7x speedup with zero algorithmic changes.

**Level 2: Vectorized operations** (numpy, already shown above)
- `np.add.at` for scatter-sum node features (replaces Python loop)
- `np.unique(..., return_inverse=True)` for edge grouping
- `pd.groupby` for temporal edge features
- Estimated 3-5x speedup over current Python loops

**Level 3: Window-level** (SLURM arrays for massive datasets)
```bash
# For a dataset with 500 CSV files, split across SLURM array jobs
#SBATCH --array=0-49%10   # 50 jobs, 10 concurrent, each processes 10 files
```

This integrates with Prefect: each array job is a Prefect task that processes a file shard.

---

## Part 6: Migration Path (incremental, not rewrite)

### Phase 1: Extract without changing behavior (Week 1)
- Create `src/preprocessing/` directory structure
- Move `GraphDataset` → `dataset.py` (no changes)
- Move `EntityVocabulary` (extracted from the two `build_*_mapping` functions)
- Move `GraphEngine` (copy of current `create_graph_from_window` + feature functions)
- **Test**: Old `preprocessing.py` and new package produce identical `.pt` outputs

### Phase 2: Introduce standard form for CAN bus (Week 2)
- Write `CANBusAdapter` wrapping current `_process_dataframe_chunk` logic
- Write `StandardizedSchema` Pandera model
- Route: `CANBusAdapter.to_standard_form()` → `GraphEngine.create_graphs()`
- **Test**: Output tensors are bitwise identical to Phase 1

### Phase 3: Vectorize the inner loops (Week 2-3)
- Replace `compute_edge_features` loop with vectorized version
- Replace `compute_node_features` loop with vectorized version
- **Test**: Output tensors within `atol=1e-6` (floating point differences from operation order)
- **Benchmark**: Measure speedup on your largest CAN bus dataset

### Phase 4: Add NetworkFlowAdapter (Week 3-4)
- Write `NetworkFlowAdapter` for UNSW-NB15
- Download UNSW-NB15 to OSC scratch
- Run full pipeline: ingest → standard form → graphs → train → W&B
- This is your first multi-domain experiment for ICML

### Phase 5: Caching + parallelism (Week 4)
- Add `ShardedCache` (write `.pt` shards to scratch, keyed by config hash)
- Add `process_files_parallel` with multiprocessing
- Add Pandera validation at the standard form boundary
- Delete old `preprocessing.py`

### Phase 6: Prefect integration (Week 5)
- Wrap `preprocess_dataset()` as a Prefect `@task`
- Add lakehouse sync for dataset metadata
- Full pipeline: Prefect flow → preprocess → train → W&B + R2

---

## Part 7: What This Unlocks for ICML

With domain-agnostic preprocessing, your paper's experimental section becomes:

> "We evaluate our approach on **6 datasets across 3 domains**: CAN bus (OTIDS, Car-Hacking, SynCAN), network flow (UNSW-NB15, CICIDS-2017), and IoT (BoT-IoT). **All datasets are processed through a unified pipeline** that converts heterogeneous network traffic into temporal graph representations, using domain-specific adapters for raw data parsing and a shared graph construction engine for windowing and feature extraction."

That's a much stronger generalizability claim than "we tested on 3 CAN bus datasets." The adapter pattern is itself a contribution — you can cite it as a reusable framework.

### Feature dimension alignment

Different domains produce different numbers of features:
- CAN bus: 8 features (payload bytes)
- UNSW-NB15: 42 numeric features
- NF-UQ-NIDS: 45 NetFlow features

Your GNN input layer needs to handle this. Two options:

1. **Per-domain models**: Separate input projection layers, shared GNN backbone. This is the "expert per domain" approach you're already doing with DQN fusion.

2. **Feature alignment layer**: A learnable linear projection `nn.Linear(domain_features, hidden_dim)` that maps any feature count to a fixed embedding dimension before the GNN. Same architecture, different first layer weights per domain.

Option 2 is cleaner for ICML and directly supports your fusion narrative: domain-specific encoders → shared GNN → DQN fusion of expert predictions.
