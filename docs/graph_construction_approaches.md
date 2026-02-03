# Alternative Graph Construction Approaches for CAN Bus Data

This document outlines various methods for converting CAN bus time series data into graph representations suitable for GNN-based anomaly detection.

## Current Implementation: Sequential Transition Graph

**Method:** Sliding window over message stream. Nodes = unique CAN IDs within window. Edges = temporal transitions (A→B if message from ID B immediately follows message from ID A).

```
Window: [msg1(ID=0x100), msg2(ID=0x200), msg3(ID=0x100), msg4(ID=0x300)]

Nodes: {0x100, 0x200, 0x300}
Edges: {0x100→0x200, 0x200→0x100, 0x100→0x300}

Node Features [11]: CAN_ID, mean(payload bytes 1-8), occurrence_count, temporal_position
Edge Features [11]: frequency, relative_freq, avg_interval, std_interval, regularity,
                    first_occurrence, last_occurrence, temporal_spread,
                    reverse_edge_exists, degree_product, degree_ratio
```

### Strengths
- Compact representation (10-50 nodes vs 100 raw messages)
- Captures ID-to-ID transition patterns (normal traffic has predictable transitions)
- Rich edge features capture timing anomalies
- Compatible with standard GNN architectures

### Weaknesses
- **Payload aggregation loses information** - different payloads from same ID are averaged
- **Window boundaries are arbitrary** - attacks spanning boundaries may be missed
- **No distinction between message instances** - 10 messages from ID 0x100 become 1 node

---

## Alternative Approaches

### 1. Co-occurrence Graph (Undirected)

**Method:** Nodes = CAN IDs. Edges connect IDs that appear within the same time window (no direction).

```
Window: [0x100, 0x200, 0x100, 0x300]

Nodes: {0x100, 0x200, 0x300}
Edges: {0x100--0x200, 0x100--0x300, 0x200--0x300}  (undirected)
```

| Pros | Cons |
|------|------|
| Simpler structure | Loses temporal ordering |
| Captures "which IDs appear together" | No edge direction |
| Faster to compute | Less expressive for sequence attacks |

**Best for:** Detecting anomalous ID combinations (e.g., IDs that never co-occur suddenly appearing together).

---

### 2. Message-as-Node Graph (Fine-Grained)

**Method:** Each message is a node. Edges connect sequential messages.

```
Window: [msg1(0x100), msg2(0x200), msg3(0x100), msg4(0x300)]

Nodes: {msg1, msg2, msg3, msg4}
Edges: {msg1→msg2, msg2→msg3, msg3→msg4}

Node Features: CAN_ID (embedded), payload bytes, timestamp delta
```

| Pros | Cons |
|------|------|
| Full temporal resolution | Large graphs (window_size nodes) |
| Message-level features preserved | Computationally expensive |
| Can detect single-message anomalies | Harder to train (variable graph size) |

**Best for:** Fine-grained intrusion detection where individual message payloads matter.

**Optimization:** Use k-hop connections (connect msg_i to msg_{i+1}, msg_{i+2}, ..., msg_{i+k}) to capture longer-range dependencies without full connectivity.

---

### 3. Heterogeneous Graph

**Method:** Multiple node types (CAN ID, ECU, Message Type) with typed edges.

```
Node Types:
  - ECU nodes (if ECU mapping is known)
  - CAN ID nodes
  - Optionally: Message nodes

Edge Types:
  - ECU --owns--> CAN_ID
  - CAN_ID --sends--> CAN_ID (transition)
  - CAN_ID --responds_to--> CAN_ID (request/response pairs)
```

| Pros | Cons |
|------|------|
| Rich semantic structure | Requires ECU topology knowledge |
| Models full bus architecture | Complex implementation |
| Can leverage domain knowledge | Needs heterogeneous GNN (HAN, HGT) |

**Best for:** When ECU topology is known and you want to model legitimate communication patterns.

---

### 4. Temporal Graph Network (TGN)

**Method:** Dynamic graph where edges have timestamps. Uses memory modules to track node states over time.

```
Events: [(t1, 0x100→0x200), (t2, 0x200→0x100), (t3, 0x100→0x300), ...]

Each event updates node memories via:
  - Message function (encodes event)
  - Memory updater (GRU/LSTM)
  - Embedding function (combines memory + features)
```

| Pros | Cons |
|------|------|
| State-of-art for temporal data | Complex training loop |
| Captures long-term patterns | Requires specialized architecture |
| No fixed window size | Higher computational cost |

**Best for:** Long-term behavioral anomalies, learning normal communication rhythms.

**References:**
- Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs" (2020)
- PyG: `torch_geometric.nn.models.TGN`

---

### 5. Static Topology Graph + Feature Injection

**Method:** Build a fixed graph from known CAN bus topology (or learned from training data). At inference, inject aggregated window statistics as node/edge features.

```
Fixed Structure (learned once):
  Nodes: All CAN IDs seen in training
  Edges: All transitions seen in training (with frequency threshold)

Per-Window Features:
  Node: [count_in_window, mean_payload, payload_variance, ...]
  Edge: [transition_count, avg_interval, ...]
```

| Pros | Cons |
|------|------|
| Very fast inference (fixed structure) | Requires topology to be stable |
| Stable graph structure | Can't detect new ID anomalies |
| Easy batching (same structure) | Loses dynamic structure info |

**Best for:** Production deployment where inference speed matters and bus topology is known.

---

### 6. Dual-View Fusion

**Method:** Maintain two parallel representations:
1. Aggregated ID graph (current method) - coarse structural view
2. Raw message sequence (Transformer/LSTM) - fine temporal view

Fuse representations for final classification.

```
                    ┌──────────────────┐
Window ────────────►│ ID Graph (GAT)   │────► graph_embedding
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Fusion Layer     │────► classification
                    └──────────────────┘
                              ▲
                              │
                    ┌──────────────────┐
Window ────────────►│ Sequence (LSTM)  │────► sequence_embedding
                    └──────────────────┘
```

| Pros | Cons |
|------|------|
| Best of both worlds | More complex architecture |
| Graph captures structure, sequence captures dynamics | Higher training cost |
| Proven effective in multimodal learning | Need to balance loss terms |

**Best for:** When you need both structural anomaly detection and sequence-level anomaly detection.

---

### 7. Bipartite Message Graph

**Method:** Two node types - CAN IDs and time bins. Edges connect IDs to the time bins in which they sent messages.

```
Time bins: [t0-t1], [t1-t2], [t2-t3]
CAN IDs: 0x100, 0x200, 0x300

Edges:
  0x100 -- [t0-t1]  (sent 3 messages)
  0x200 -- [t0-t1]  (sent 1 message)
  0x100 -- [t1-t2]  (sent 2 messages)
  ...
```

| Pros | Cons |
|------|------|
| Separates temporal and ID dimensions | Less intuitive |
| Can use bipartite GNN methods | Harder to interpret |
| Naturally handles variable message rates | Time bin size is a hyperparameter |

**Best for:** Academic research, explicit temporal-structural factorization.

---

## Recommendation

For CAN bus intrusion detection, the **current Sequential Transition Graph** is a reasonable choice. However, consider:

1. **If payload content matters:** Switch to Message-as-Node or Dual-View Fusion
2. **If inference speed is critical:** Use Static Topology Graph
3. **If you have ECU mapping:** Use Heterogeneous Graph
4. **If attacks are long-duration:** Consider TGN or larger windows with overlap

### Quick Wins for Current Implementation

1. **Overlapping windows** (stride < window_size) to avoid missing boundary attacks
2. **Payload variance as node feature** instead of just mean
3. **Multi-scale windows** - create graphs at multiple window sizes and ensemble

---

## Implementation Complexity Ranking

| Approach | Complexity | Code Changes Required |
|----------|------------|----------------------|
| Current (Sequential Transition) | ★☆☆☆☆ | Already implemented |
| Co-occurrence Graph | ★☆☆☆☆ | Minor edge logic change |
| Static Topology + Features | ★★☆☆☆ | Pre-compute topology, change features |
| Message-as-Node | ★★★☆☆ | Significant restructure |
| Dual-View Fusion | ★★★☆☆ | Add sequence branch, fusion layer |
| Heterogeneous Graph | ★★★★☆ | New architecture, need ECU data |
| Temporal Graph Network | ★★★★★ | Complete rewrite, TGN architecture |
