# PyG Enhancement Plan for KD-GAT

## Current PyG Usage (6 components)

| Component | Location | Purpose |
|---|---|---|
| `Data` | `src/preprocessing/preprocessing.py` | Graph objects (11-D node features, 11-D edge features) |
| `GATConv` | `src/models/gat.py`, `src/models/vgae.py` | All graph convolution layers |
| `JumpingKnowledge` | `src/models/gat.py` | Multi-layer aggregation |
| `global_mean_pool` | `src/models/gat.py`, `src/models/fusion_features.py` | Graph-level readout |
| `DataLoader` | `pipeline/stages/utils.py` | Mini-batch loading |
| `DynamicBatchSampler` | `pipeline/stages/utils.py` | Variable-size graph batching |

---

## Top 5 Enhancements — Detailed Implementation Plans

---

### 1. Replace `GATConv` with `TransformerConv` (Edge-Aware Attention)

**Problem**: `GATConv` computes attention weights using only node features. Your 11-dimensional edge features (`edge_attr`) — frequency, temporal intervals, bidirectionality, degree products — are completely ignored during message passing. This is wasted information.

**Solution**: `TransformerConv` uses edge features directly in the attention computation via key/query/value projections that incorporate `edge_attr`. It is a near drop-in replacement.

**Files to modify**:

#### `src/models/gat.py`
```python
# Change import
from torch_geometric.nn import TransformerConv, JumpingKnowledge, global_mean_pool

# In GATWithJK.__init__, replace GATConv with TransformerConv
for i in range(num_layers):
    in_dim = embedding_dim + (in_channels - 1) if i == 0 else hidden_channels * heads
    self.convs.append(TransformerConv(
        in_dim, hidden_channels, heads=heads,
        edge_dim=11,  # your 11-D edge features
        concat=True,
        dropout=dropout,
    ))

# In forward(), pass edge_attr
x = conv(x, edge_index, edge_attr=data.edge_attr)
```

#### `src/models/vgae.py`
```python
# Same pattern for encoder and decoder layers
self.encoder_layers.append(TransformerConv(
    in_dim, out_per_head, heads=heads,
    edge_dim=11,
))
```

#### `config/schema.py`
```python
class GATArchitecture(BaseModel, frozen=True):
    conv_type: str = "transformer"  # or "gatv2" as alternative
    edge_dim: int = 11  # edge feature dimension
    # ... existing fields ...
```

**Risk**: Low — the layer interface is compatible. The main difference is `edge_dim` parameter and passing `edge_attr` in forward. Training time may increase ~15-20% due to the edge feature projections.

**Validation**: Compare attention weight distributions with and without edge features. If edge features are informative, you should see attention patterns that correlate with temporal/frequency edge properties. Track val_loss and F1 on existing datasets.

**Alternative**: `GATv2Conv` is an even simpler swap (same API as `GATConv`, no edge features) that fixes the static attention limitation. Consider testing both:
- `GATv2Conv`: fixes expressiveness, no edge features → quick A/B test
- `TransformerConv`: fixes expressiveness + uses edge features → full improvement

---

### 2. Add GNNExplainer / AttentionExplainer for Interpretability

**Problem**: Your model can classify attacks but cannot explain *why* a graph was flagged. For a security-critical IDS, interpretability is essential — operators need to know which ECU connections and message features drove the anomaly prediction.

**Solution**: PyG's `torch_geometric.explain` module provides a unified framework for post-hoc GNN explanations with quantitative evaluation metrics.

**Files to create/modify**:

#### `src/explain/__init__.py` (new module)
```python
"""GNN explainability for CAN bus anomaly detection."""
```

#### `src/explain/explainer.py` (new file)
```python
from torch_geometric.explain import Explainer, GNNExplainer, AttentionExplainer
from torch_geometric.explain import ModelConfig, ThresholdConfig

def build_explainer(model, algorithm="gnnexplainer"):
    """Build an Explainer for a trained GAT or VGAE model."""
    if algorithm == "attention":
        algo = AttentionExplainer()
    else:
        algo = GNNExplainer(epochs=200, lr=0.01)

    return Explainer(
        model=model,
        algorithm=algo,
        explanation_type="model",
        model_config=ModelConfig(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        ),
        node_mask_type="attributes",
        edge_mask_type="object",
        threshold_config=ThresholdConfig(
            threshold_type="topk",
            value=10,  # top-10 most important edges
        ),
    )

def explain_graph(explainer, data):
    """Generate explanation for a single graph."""
    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        batch=data.batch,
        edge_attr=data.edge_attr,
    )
    return {
        "node_mask": explanation.node_mask,    # which node features matter
        "edge_mask": explanation.edge_mask,    # which edges matter
        "prediction": explanation.prediction,
    }
```

#### `pipeline/stages/evaluation.py` — add explanation generation
```python
# After model evaluation, generate explanations for misclassified or
# high-confidence anomaly graphs
from src.explain.explainer import build_explainer, explain_graph

explainer = build_explainer(gat_model, algorithm="attention")
explanations = []
for graph in flagged_anomalies[:50]:  # top 50 anomalies
    exp = explain_graph(explainer, graph)
    explanations.append(exp)
# Save as artifact alongside embeddings.npz
```

#### `pipeline/export.py` — export explanation data for dashboard
```python
# Add explanation panel data: which CAN IDs and edges are most important
# per attack type
```

**Evaluation metrics** (built into PyG):
- `fidelity`: Does removing the explanation subgraph change the prediction?
- `unfaithfulness`: Does the explanation faithfully represent model behavior?
- `characterization_score`: Overall explanation quality

**Dashboard integration**: Add a force-directed graph panel showing the explanation subgraph with edge thickness = importance, node color = feature importance. This would be a powerful visualization for the paper.

**Risk**: Low — this is purely additive (post-training analysis). No changes to training.

---

### 3. Integrate PyTorch Geometric Temporal (MTGNN or A3TGCN)

**Problem**: Your sliding-window approach creates independent static graph snapshots. Temporal dependencies *between* windows are lost — an attack that develops over multiple windows (e.g., slowly increasing message frequency) cannot be detected by any single window.

**Solution**: PyTorch Geometric Temporal provides models that maintain temporal state across graph snapshots. This is a new model variant alongside your existing VGAE/GAT, not a replacement.

**Scope**: This is the largest enhancement. It introduces a new model type into the pipeline.

**Files to create/modify**:

#### `src/models/temporal.py` (new file)
```python
"""Temporal GNN models for CAN bus anomaly detection."""
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import A3TGCN2
# Alternative: from torch_geometric_temporal.nn.attention import MTGNN

class TemporalGATClassifier(nn.Module):
    """A3TGCN-based temporal graph classifier for CAN bus windows.

    Processes a sequence of T graph snapshots and produces a graph-level
    classification (normal vs. attack) that incorporates temporal context.
    """
    def __init__(self, node_features, hidden_dim, num_classes=2, periods=12):
        super().__init__()
        # A3TGCN2 expects: (num_nodes, node_features, periods) input
        self.temporal_conv = A3TGCN2(
            in_channels=node_features,
            out_channels=hidden_dim,
            periods=periods,  # number of time steps to look back
            batch_size=1,     # per-sequence processing
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x_seq, edge_index_seq):
        """
        Args:
            x_seq: list of T node feature tensors [num_nodes, features]
            edge_index_seq: list of T edge index tensors [2, num_edges]
        Returns:
            logits: [num_classes] classification output
        """
        h = None
        for x_t, edge_index_t in zip(x_seq, edge_index_seq):
            h = self.temporal_conv(x_t, edge_index_t, h)
        # Pool over nodes → graph-level
        graph_repr = h.mean(dim=0)  # [hidden_dim]
        return self.classifier(graph_repr.unsqueeze(0))
```

#### `src/preprocessing/temporal.py` (new file)
```python
"""Convert sliding-window graphs into temporal sequences."""

def create_temporal_sequences(graphs, seq_length=12, stride=1):
    """Group consecutive graph windows into temporal sequences.

    Args:
        graphs: list of PyG Data objects (already windowed)
        seq_length: number of consecutive windows per sequence
        stride: step between sequence starts

    Returns:
        list of (graph_sequence, label) tuples
    """
    sequences = []
    for i in range(0, len(graphs) - seq_length + 1, stride):
        seq = graphs[i:i + seq_length]
        # Label: attack if any window in sequence contains attack
        label = max(g.y.item() for g in seq)
        sequences.append((seq, label))
    return sequences
```

#### `config/schema.py` — add temporal architecture config
```python
class TemporalArchitecture(BaseModel, frozen=True):
    hidden: int = Field(64, ge=1)
    periods: int = Field(12, ge=1)    # lookback window count
    seq_stride: int = Field(1, ge=1)  # stride between sequences
    model_variant: str = "a3tgcn"     # or "mtgnn"
```

#### `config/models/temporal/large.yaml`, `config/models/temporal/small.yaml` (new)

#### Pipeline integration
- Register `temporal` as a new `model_type` in `src/models/registry.py`
- Add temporal training stage in `pipeline/stages/training.py`
- Add temporal fusion extractor in `src/models/fusion_features.py`

**Prerequisite**: Install `torch_geometric_temporal`:
```bash
pip install torch-geometric-temporal
```

**Risk**: Medium — this is a new model type with different data loading requirements. The temporal sequence grouping adds complexity. Start with A3TGCN (simpler) before attempting MTGNN (learns adjacency).

**Validation**: Compare detection latency and F1 against static GAT on multi-window attack sequences. The temporal model should excel on slow-onset attacks (gradual frequency shifts, progressive payload changes).

---

### 4. Replace `global_mean_pool` with `MultiAggregation`

**Problem**: `global_mean_pool` averages all node features, losing information about feature variance, extremes, and distribution shape. An anomalous ECU with extreme values gets smoothed away in the mean.

**Solution**: `MultiAggregation` combines multiple aggregation functions (mean, max, std, etc.) into a single pooling operation, producing a richer graph-level representation.

**Files to modify**:

#### `src/models/gat.py`
```python
from torch_geometric.nn import TransformerConv, JumpingKnowledge
from torch_geometric.nn import MultiAggregation

class GATWithJK(nn.Module):
    def __init__(self, ..., pool_aggrs=("mean", "max", "std")):
        super().__init__()
        # ...existing code...

        # Replace global_mean_pool with MultiAggregation
        self.pool = MultiAggregation(
            aggrs=list(pool_aggrs),
            mode="cat",  # concatenate outputs
        )

        # FC input dim now multiplied by number of aggregators
        fc_input_dim = hidden_channels * heads * num_layers * len(pool_aggrs)
        # ...build fc_layers with new fc_input_dim...

    def forward(self, data, ...):
        # ...existing conv + JK code...
        x = self.jk(xs)
        x = self.pool(x, batch)  # [batch_size, jk_dim * num_aggrs]
        # ...FC layers...
```

#### `src/models/fusion_features.py`
```python
# Update GATFusionExtractor to use the model's pool instead of global_mean_pool
class GATFusionExtractor:
    def extract(self, model, graph, batch_idx, device):
        xs = model(graph, return_intermediate=True)
        jk_out = model.jk(xs)
        pooled = model.pool(jk_out, batch_idx)  # use model's own pooling
        # ...rest of extraction...
```

#### `config/schema.py`
```python
class GATArchitecture(BaseModel, frozen=True):
    pool_aggrs: tuple[str, ...] = ("mean", "max", "std")
    # ...existing fields...
```

**Risk**: Low — this is a straightforward replacement. The only impact is that FC layer input dimensions change (multiplied by number of aggregators), so existing checkpoints are incompatible. New training run required.

**Validation**: Compare graph-level classification F1 and AUC. The multi-aggregation should improve detection of anomalies that manifest as unusual variance or extreme values rather than shifted means.

**Note on DQN fusion features**: The `GATFusionExtractor` currently computes `emb_mean, emb_std, emb_max, emb_min` manually from the pooled output. With `MultiAggregation`, some of these statistics are already captured in the pooled representation, so the fusion features may become redundant or need adjustment.

---

### 5. Benchmark Against PyGOD (DOMINANT, OCGNN)

**Problem**: Your VGAE is a custom graph autoencoder for anomaly detection, but you're not comparing against established graph anomaly detection baselines. PyGOD provides standardized implementations of 15+ graph outlier detection algorithms, all built on PyG.

**Solution**: Add PyGOD baselines to the evaluation pipeline. This strengthens the paper's related work comparison and validates that your KD-GAT approach outperforms general-purpose graph anomaly detectors.

**Scope**: This is an evaluation/comparison task, not a model replacement.

**Files to create/modify**:

#### `src/baselines/__init__.py` (new module)

#### `src/baselines/pygod_baselines.py` (new file)
```python
"""PyGOD baseline anomaly detectors for comparison."""
from pygod.detector import DOMINANT, OCGNN, CoLA, CONAD

def train_dominant(train_data, contamination=0.1):
    """Train DOMINANT (GNN autoencoder) baseline."""
    detector = DOMINANT(
        hid_dim=64,
        num_layers=3,
        epoch=100,
        contamination=contamination,
        gpu=0,
    )
    detector.fit(train_data)
    return detector

def train_ocgnn(train_data, contamination=0.1):
    """Train OCGNN (one-class GNN) baseline.
    Learns only from normal data — no attack labels needed.
    """
    detector = OCGNN(
        hid_dim=64,
        num_layers=3,
        epoch=100,
        contamination=contamination,
        gpu=0,
    )
    detector.fit(train_data)
    return detector

def evaluate_baseline(detector, test_data):
    """Evaluate a PyGOD detector on test data."""
    labels = detector.predict(test_data)       # binary predictions
    scores = detector.decision_function(test_data)  # anomaly scores
    return labels, scores
```

#### `pipeline/stages/evaluation.py` — add baseline comparison
```python
# After KD-GAT evaluation, run PyGOD baselines on the same test scenarios
from src.baselines.pygod_baselines import train_dominant, train_ocgnn

# Note: PyGOD operates at node level. For graph-level comparison,
# aggregate node anomaly scores per graph (max, mean, or proportion
# of anomalous nodes).
```

#### Comparison metrics
- Node-level: AUROC, AUPRC, F1 at optimal threshold
- Graph-level: aggregate node scores → graph score → compare with GAT predictions
- Latency: inference time per graph (critical for real-time CAN IDS)
- Model size: parameter count (relevant for edge deployment via KD)

**Prerequisite**: Install PyGOD:
```bash
pip install pygod
```

**Risk**: Low — purely additive comparison. No changes to existing models or pipeline.

**Important caveat**: PyGOD detectors operate at the **node level** (flagging individual ECUs as anomalous), while your KD-GAT operates at the **graph level** (flagging entire windows). The comparison requires an aggregation strategy to make results comparable. Options:
1. Max-score: graph is anomalous if any node score exceeds threshold
2. Mean-score: average node anomaly scores
3. Proportion: fraction of nodes flagged as anomalous

---

## Remaining Enhancements — Prioritized by Category

### Category A: Model Architecture Improvements

| Priority | Feature | Description | Effort | Impact |
|---|---|---|---|---|
| A1 | **`GINEConv`** | Maximally expressive (1-WL) conv that uses edge features. Alternative to TransformerConv with theoretical guarantees. | Low | Medium |
| A2 | **`PNAConv`** | Multiple aggregators (sum, mean, max, std) with degree scalers inside the conv layer itself. More expressive than single-aggregator GAT. | Medium | Medium |
| A3 | **`GPSConv`** | Combines local MPNN + global self-attention. Captures both local ECU neighborhoods and global bus-wide patterns. Best for large graphs. | Medium | High |
| A4 | **`GCN2Conv`** | Initial residual connections that fight oversmoothing. Enables deeper GNNs without performance degradation. | Low | Low |
| A5 | **`Set2Set`** | Iterative attention-based readout. More expressive alternative to MultiAggregation for graph-level pooling. | Low | Medium |
| A6 | **`SAGPooling` / `TopKPooling`** | Hierarchical pooling that learns to discard irrelevant nodes. Could learn which ECUs matter for classification. | Medium | Medium |
| A7 | **`DiffPool`** (`dense_diff_pool`) | Differentiable hierarchical pooling that discovers ECU clusters. Expensive but powerful for graph classification. | High | High |
| A8 | **`DeepGraphInfomax`** | Self-supervised pre-training via mutual information maximization. Pre-train on unlabeled CAN data, fine-tune for detection. | Medium | Medium |
| A9 | **`TGNMemory`** | Temporal Graph Network memory module. Maintains per-ECU state over time. Alternative to full PyG Temporal integration. | High | High |
| A10 | **`SortAggregation`** | Sort node features by channel values for CNN-style readout. Different inductive bias than attention-based pooling. | Low | Low |

### Category B: Data & Preprocessing

| Priority | Feature | Description | Effort | Impact |
|---|---|---|---|---|
| B1 | **`HeteroData`** | Model CAN bus with typed nodes (sensor ECUs, actuator ECUs, controllers) and typed edges (diagnostic, control, status messages). Unlocks type-aware attention. | High | High |
| B2 | **`VirtualNode` transform** | Add a global node connected to all ECUs, representing the shared CAN bus medium. Enables global information flow in a single message-passing step. | Low | Medium |
| B3 | **`AddRandomWalkPE`** | Random walk positional encodings. Captures graph topology information not present in node features. Especially useful with TransformerConv/GPSConv. | Low | Medium |
| B4 | **`AddLaplacianEigenvectorPE`** | Spectral positional encodings. Provides structural position information for each node. | Low | Medium |
| B5 | **`LineGraph` transform** | Convert to message-centric view where CAN messages become nodes and edges connect sequential messages. Alternative graph construction. | Medium | Medium |
| B6 | **`TemporalData`** | Native PyG structure for timestamped event streams. More natural representation for CAN messages than Data. | Medium | Medium |
| B7 | **`ImbalancedSampler`** | Weighted random sampling for class imbalance. Your anomaly classes are minority — this ensures balanced mini-batches. | Low | Medium |
| B8 | **`Pad` transform** | Enforce consistent tensor shapes across graphs. Simplifies batching for fixed-size architectures. | Low | Low |
| B9 | **`KNNGraph` transform** | Build graphs based on feature similarity instead of message sequence. Alternative or supplementary graph construction. | Medium | Medium |
| B10 | **`StochasticBlockModelDataset`** | Synthetic graph generator for modeling ECU cluster structure. Useful for ablation studies. | Low | Low |

### Category C: Training & Robustness

| Priority | Feature | Description | Effort | Impact |
|---|---|---|---|---|
| C1 | **`dropout_edge` / `dropout_node`** | Stochastic graph augmentation during training. Makes the model robust to missing messages or ECUs going offline. | Low | Medium |
| C2 | **`negative_sampling`** | Generate negative edges for contrastive learning. Enables contrastive pre-training or auxiliary losses. | Low | Medium |
| C3 | **`PRBCDAttack` / `GRBCDAttack`** | Adversarial robustness testing. Can an attacker fool your IDS by subtly modifying CAN graph structure? Essential for security research. | Medium | High |
| C4 | **`mask_feature` / `add_random_edge`** | Additional augmentation strategies. Feature masking trains robustness to partial observations. | Low | Low |
| C5 | **`CorrectAndSmooth`** | Post-processing label propagation. Can refine predictions after GAT inference at zero training cost. | Low | Low |

### Category D: Normalization

| Priority | Feature | Description | Effort | Impact |
|---|---|---|---|---|
| D1 | **`GraphNorm`** | Per-graph normalization with learnable shift. Better than BatchNorm for variable-size graphs (your DynamicBatchSampler produces variable batches). | Low | Medium |
| D2 | **`GraphSizeNorm`** | Normalize by graph size. Your CAN graphs vary in node count (different windows have different active ECUs). | Low | Medium |
| D3 | **`PairNorm`** | Prevents oversmoothing. Useful if you deepen your GAT beyond 3 layers. | Low | Low |
| D4 | **`MessageNorm`** | Normalizes aggregated messages. Stabilizes training for deep GNNs. | Low | Low |

### Category E: Utilities & Analysis

| Priority | Feature | Description | Effort | Impact |
|---|---|---|---|---|
| E1 | **`homophily`** | Measure whether anomalous CAN nodes cluster together or are dispersed. Informs model design (homophilic vs. heterophilic GNN). | Low | Medium |
| E2 | **`k_hop_subgraph`** | Extract local neighborhoods around suspicious ECUs. Useful for focused analysis and explanation. | Low | Medium |
| E3 | **`assortativity`** | Degree correlation analysis. Attack patterns may change degree assortativity. | Low | Low |
| E4 | **`to_networkx`** | Convert PyG graphs to NetworkX for graph-theoretic analysis (centrality, community detection, etc.). | Low | Low |

### Category F: Profiling & Infrastructure

| Priority | Feature | Description | Effort | Impact |
|---|---|---|---|---|
| F1 | **`profileit` / `benchmark`** | Profile training runtime and memory. Essential for meeting real-time CAN IDS latency constraints (~1ms per message). | Low | Medium |
| F2 | **`count_parameters` / `get_model_size`** | Quantify KD compression ratios precisely. Useful for paper tables. | Low | Medium |
| F3 | **`CachedLoader`** | Cache mini-batch outputs for repeated access. Speeds up DQN fusion training where the same graphs are sampled repeatedly. | Low | Low |
| F4 | **`PrefetchLoader`** | Async GPU prefetching. Overlaps data transfer with computation. | Low | Low |
| F5 | **GraphGym registration system** | Register custom CAN-specific encoders, metrics, and datasets for reproducible experiment configs. | Medium | Low |

### Category G: Advanced / Exploratory

| Priority | Feature | Description | Effort | Impact |
|---|---|---|---|---|
| G1 | **PyGOD `CoLA`** | Contrastive self-supervised anomaly detection. No labels needed — learns by contrasting normal subgraph structures. | Medium | High |
| G2 | **PyGOD `CONAD`** | Contrastive + augmentation-based anomaly detection. State-of-the-art unsupervised graph anomaly detection. | Medium | High |
| G3 | **`ARGA` / `ARGVA`** | Adversarially regularized graph autoencoders. Could improve your VGAE's latent space quality. | Medium | Medium |
| G4 | **`MetaLayer`** | Flexible Graph Network block with separate node/edge/global update functions. Enables full edge feature updates during message passing. | Medium | Medium |
| G5 | **`LSTMAggregation` / `GRUAggregation`** | Recurrent neighbor aggregation. Orders neighbors and processes sequentially — captures temporal ordering of CAN messages within a window. | Medium | Medium |
| G6 | **`NeighborLoader` with `temporal_strategy`** | Time-aware neighbor sampling. For large-graph variants where full-graph training is infeasible. | Medium | Low |
| G7 | **`HGTConv` / `HANConv`** | Heterogeneous graph attention. Required if you adopt `HeteroData` (B1). Type-aware attention across ECU/message types. | High | High |
| G8 | **`MTGNN`** (from PyG Temporal) | Learns adaptive adjacency matrices from multivariate time series. Can discover CAN communication patterns that aren't in the hand-crafted graph. | High | High |

---

## Implementation Order (Suggested)

**Phase 1 — Quick wins (1-2 weeks)**
1. Enhancement 1: `TransformerConv` swap (or `GATv2Conv` as quick test)
2. Enhancement 4: `MultiAggregation` pooling
3. Items D1, B7, C1, F1, F2 (all low-effort)

**Phase 2 — Interpretability & baselines (2-3 weeks)**
4. Enhancement 2: GNNExplainer integration
5. Enhancement 5: PyGOD baselines
6. Items E1, E2, C3 (analysis and adversarial testing)

**Phase 3 — Temporal modeling (3-4 weeks)**
7. Enhancement 3: PyTorch Geometric Temporal integration
8. Items B2, B3 (transforms that feed into temporal model)

**Phase 4 — Advanced exploration (ongoing)**
9. Items from Category G based on Phase 1-3 results
10. HeteroData modeling (B1 + G7) if type-aware attention shows promise
