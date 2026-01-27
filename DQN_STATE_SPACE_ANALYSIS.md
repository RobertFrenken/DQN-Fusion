# DQN Fusion Training: State Space Analysis & Enhancement Options

**Date**: 2026-01-27
**Status**: Research Complete
**Purpose**: Clarify DQN inputs and identify quick wins for richer state representation

---

## Current Implementation

### What DQN Receives Now

**State Dimension**: 2 (very limited!)

```python
state = [anomaly_score, gat_prob]  # Shape: [batch_size, 2]
```

### 1. Anomaly Score from VGAE (Single Scalar)

**Source**: `src/training/prediction_cache.py` lines 61-110

**Computation**:
```python
# VGAE forward pass → 3 types of reconstruction errors
node_errors = MSE(reconstructed_features, original_features)  # Per node
neighbor_errors = BCE(neighbor_logits, neighbor_targets)      # Per node
canid_errors = (predicted_id != true_id)                       # Per node

# Aggregate per graph (take max error across nodes)
graph_node_error = max(node_errors)
graph_neighbor_error = max(neighbor_errors)
graph_canid_error = max(canid_errors)

# Weighted composite (fixed weights)
composite = 0.4 × graph_node_error +
            0.35 × graph_neighbor_error +
            0.25 × graph_canid_error

# Final anomaly score (normalized to [0, 1])
anomaly_score = sigmoid(composite × 3 - 1.5)
```

**What's Lost**:
- Individual error components (node, neighbor, canid) are collapsed
- Node-level detail is aggregated to graph-level max
- VGAE latent space (z) is completely discarded
- Reconstruction quality variance across graph nodes

### 2. GAT Probability (Single Scalar)

**Source**: `src/training/prediction_cache.py` lines 112-127

**Computation**:
```python
logits = GAT.forward(graph)  # Shape: [batch_size, 2]
probabilities = softmax(logits, dim=1)  # Normalize to [0, 1]
gat_prob = probabilities[:, 1]  # Probability of class 1 (attack)
```

**What's Available But Not Used**:
- Full logits [batch_size, 2] - not just class 1 probability
- Softmax entropy (confidence indicator)
- GAT intermediate layer activations
- Node-level embeddings before pooling
- Attention weights from GAT layers

---

## Why This is Limited

The DQN agent has to make fusion decisions with only **2 scalar values**:
- One summarizing VGAE's anomaly detection (collapsed from 3+ error types)
- One summarizing GAT's classification (collapsed from 2 logits)

**Key Information Lost**:
1. **Uncertainty/Confidence**: No measure of how certain each model is
2. **Error Breakdown**: VGAE's 3 error components are combined too early
3. **Feature Richness**: Latent representations (VGAE's z, GAT embeddings) unused
4. **Model Agreement**: Can't distinguish "both confident" vs "both uncertain"

---

## Enhancement Options: Quick Wins Table

| Option | State Dim | Effort | Impact | Complexity | Implementation Notes |
|--------|-----------|--------|--------|------------|----------------------|
| **1. Separate VGAE Error Components** | +2 | 🟢 LOW | 🟡 MEDIUM | Simple | Already computed, just don't collapse them |
| **2. GAT Full Logits** | +1 | 🟢 LOW | 🟢 HIGH | Simple | Use both logits instead of just class 1 prob |
| **3. Model Confidence Indicators** | +2 | 🟢 LOW | 🟢 HIGH | Simple | Softmax entropy + VGAE variance |
| **4. VGAE Latent Space Summary** | +4-16 | 🟡 MEDIUM | 🟢 HIGH | Moderate | Pool z vector (mean/max/std) |
| **5. GAT Pre-Pooling Embeddings** | +4-16 | 🟡 MEDIUM | 🟢 HIGH | Moderate | Extract node embeddings before global pool |
| **6. Attention Weights Statistics** | +2-4 | 🟠 HIGH | 🟡 MEDIUM | Complex | Aggregate attention patterns |
| **7. Per-Component Reconstruction** | +3 | 🟢 LOW | 🟡 MEDIUM | Simple | Node/neighbor/canid errors separately |

---

## Detailed Enhancement Descriptions

### ✅ OPTION 1: Separate VGAE Error Components (RECOMMENDED QUICK WIN)

**Current State**: `[anomaly_score, gat_prob]` → **2 dims**
**Enhanced State**: `[node_error, neighbor_error, canid_error, gat_prob]` → **4 dims**

**Why This Helps**:
- Different attack types cause different error patterns
- Node-level attacks → high node_error
- Topology attacks → high neighbor_error
- ID spoofing → high canid_error
- DQN can learn which error type is most informative per scenario

**Implementation** (5 minutes):
```python
# In FusionDataExtractor.compute_anomaly_scores()
# BEFORE (current):
composite_scores = (graph_errors * fusion_weights).sum(dim=1)
return torch.sigmoid(composite_scores * 3 - 1.5)

# AFTER (enhanced):
return graph_errors  # Return all 3 components separately!
# Shape: [batch_size, 3] instead of [batch_size]
```

Then update state stacking in FusionLightningModule:
```python
# BEFORE:
states = torch.stack([anomaly_scores, gat_probs], dim=1)  # [batch, 2]

# AFTER:
states = torch.cat([
    anomaly_scores,  # [batch, 3] - now 3 components
    gat_probs.unsqueeze(1)  # [batch, 1]
], dim=1)  # [batch, 4]
```

---

### ✅ OPTION 2: GAT Full Logits (RECOMMENDED QUICK WIN)

**Current State**: `[anomaly_score, gat_prob]` → **2 dims**
**Enhanced State**: `[anomaly_score, gat_logit_0, gat_logit_1]` → **3 dims**

**Why This Helps**:
- Preserves pre-normalization information
- Logit magnitude indicates confidence (large |logit| = confident)
- Avoids information loss from softmax squashing
- DQN can learn from raw model outputs

**Implementation** (3 minutes):
```python
# In FusionDataExtractor.compute_gat_probabilities()
# BEFORE:
probabilities = torch.softmax(logits, dim=-1)[:, 1]
return probabilities  # Single scalar

# AFTER:
return logits  # Return both logits!
# Shape: [batch_size, 2]
```

---

### ✅ OPTION 3: Model Confidence Indicators (RECOMMENDED QUICK WIN)

**Current State**: `[anomaly_score, gat_prob]` → **2 dims**
**Enhanced State**: `[anomaly_score, vgae_confidence, gat_prob, gat_confidence]` → **4 dims**

**Why This Helps**:
- DQN learns to trust confident predictions more
- Can distinguish "both models agree and confident" vs "both agree but uncertain"
- Critical for adversarial robustness

**Confidence Metrics**:

1. **VGAE Confidence**: Inverse of reconstruction variance
```python
# In compute_anomaly_scores(), also return:
error_variance = graph_errors.std(dim=1)  # Variance across 3 error types
vgae_confidence = 1.0 / (1.0 + error_variance)  # [0, 1]
```

2. **GAT Confidence**: Softmax entropy
```python
# In compute_gat_probabilities(), also return:
probs = torch.softmax(logits, dim=1)
entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
gat_confidence = 1.0 - entropy / math.log(2)  # Normalize by max entropy
```

**Implementation** (10 minutes)

---

### ⚠️ OPTION 4: VGAE Latent Space Summary (MEDIUM EFFORT)

**Current State**: 2 dims → **Enhanced State**: 6-18 dims

**Why This Helps**:
- VGAE's latent z captures high-level graph structure
- Different attack types have different latent signatures
- Rich semantic representation beyond reconstruction error

**Implementation** (20-30 minutes):
```python
# Modify VGAE forward to also return z
def compute_anomaly_scores_with_latent(self, batch):
    cont_out, canid_logits, neighbor_logits, z, _ = self.autoencoder(...)

    # z has shape [num_nodes, latent_dim] (e.g., [1000, 16])
    # Aggregate per graph:
    num_graphs = batch.batch.max().item() + 1
    latent_summary = torch.zeros(num_graphs, 4, device=self.device)

    for graph_idx in range(num_graphs):
        node_mask = (batch.batch == graph_idx)
        graph_latent = z[node_mask]  # [num_nodes_in_graph, latent_dim]

        # Summary statistics
        latent_summary[graph_idx, 0] = graph_latent.mean()  # Mean activation
        latent_summary[graph_idx, 1] = graph_latent.std()   # Variability
        latent_summary[graph_idx, 2] = graph_latent.max()   # Peak activation
        latent_summary[graph_idx, 3] = graph_latent.min()   # Min activation

    return anomaly_score, latent_summary  # [batch, 1], [batch, 4]
```

**State**: `[anomaly_score, latent_mean, latent_std, latent_max, latent_min, gat_prob]` → **6 dims**

---

### ⚠️ OPTION 5: GAT Pre-Pooling Embeddings (MEDIUM EFFORT)

**Current State**: 2 dims → **Enhanced State**: 6-18 dims

**Why This Helps**:
- GAT creates rich node embeddings before pooling
- Global pooling (mean) loses per-node detail
- Statistics of embeddings reveal graph structure

**Implementation** (25-35 minutes):
```python
# Modify GAT forward to return pre-pooling embeddings
def compute_gat_with_embeddings(self, batch):
    # Forward through GAT layers
    x = batch.x
    edge_index = batch.edge_index

    for layer in self.gat_layers:
        x = layer(x, edge_index)  # [num_nodes, hidden_dim]

    # x is now node embeddings before pooling
    # Aggregate per graph:
    num_graphs = batch.batch.max().item() + 1
    embedding_summary = torch.zeros(num_graphs, 4, device=self.device)

    for graph_idx in range(num_graphs):
        node_mask = (batch.batch == graph_idx)
        graph_embeddings = x[node_mask]  # [num_nodes_in_graph, hidden_dim]

        embedding_summary[graph_idx, 0] = graph_embeddings.mean()
        embedding_summary[graph_idx, 1] = graph_embeddings.std()
        embedding_summary[graph_idx, 2] = graph_embeddings.max()
        embedding_summary[graph_idx, 3] = graph_embeddings.min()

    # Final classification
    logits = self.classifier_head(x, edge_index, batch.batch)

    return logits, embedding_summary
```

---

### ❌ OPTION 6: Attention Weights Statistics (HIGH EFFORT, SKIP)

**Complexity**: Requires modifying GAT internals to expose attention coefficients
**Effort**: 45-60 minutes
**Recommendation**: Skip for now, diminishing returns

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (30 minutes total) ✅

Implement Options 1, 2, and 3 together:

**New State Vector**:
```python
state = [
    node_error,        # VGAE component 1
    neighbor_error,    # VGAE component 2
    canid_error,       # VGAE component 3
    vgae_confidence,   # VGAE certainty
    gat_logit_0,       # GAT logit for class 0
    gat_logit_1,       # GAT logit for class 1
    gat_confidence     # GAT certainty (entropy-based)
]
# Total: 7 dimensions (up from 2!)
```

**Expected Improvement**:
- 🎯 Better fusion decisions due to richer context
- 🎯 Distinguishes attack types (different error patterns)
- 🎯 Learns to trust confident predictions
- 🎯 Reduced false positives/negatives

---

### Phase 2: Medium Wins (60 minutes total) 🔄

If Phase 1 shows promise, add Option 4 (VGAE latent):

**Enhanced State Vector**:
```python
state = [
    # Phase 1 (7 dims)
    node_error, neighbor_error, canid_error, vgae_confidence,
    gat_logit_0, gat_logit_1, gat_confidence,
    # Phase 2 additions (4 dims)
    latent_mean, latent_std, latent_max, latent_min
]
# Total: 11 dimensions
```

---

## Expected Performance Impact

### Current (2D state):
- ⚠️ DQN has limited context for fusion decisions
- ⚠️ Can't distinguish attack types
- ⚠️ No confidence awareness
- ⚠️ Sub-optimal fusion weights

### With Phase 1 (7D state):
- ✅ 3.5× richer state representation
- ✅ Attack-type-aware fusion
- ✅ Confidence-weighted decisions
- ✅ Expected accuracy improvement: +2-5%

### With Phase 2 (11D state):
- ✅ 5.5× richer state representation
- ✅ Semantic latent features
- ✅ Deep structural understanding
- ✅ Expected accuracy improvement: +4-8%

---

## Implementation Checklist

### Phase 1: Quick Wins (Implement First) ✅

- [ ] **File**: `src/training/prediction_cache.py`
  - [ ] Modify `compute_anomaly_scores()` to return 3 error components
  - [ ] Modify `compute_gat_probabilities()` to return both logits
  - [ ] Add `compute_vgae_confidence()` method
  - [ ] Add `compute_gat_confidence()` method

- [ ] **File**: `src/training/lightning_modules.py` (FusionLightningModule)
  - [ ] Update state stacking to handle 7D state
  - [ ] Update DQN agent instantiation: `state_dim=7`

- [ ] **File**: `src/models/dqn.py` (EnhancedDQNFusionAgent)
  - [ ] Update Q-network input layer: `nn.Linear(7, hidden_dim)`
  - [ ] Verify action space unchanged (alpha selection)

- [ ] **Testing**:
  - [ ] Run quick test to verify shapes: state=[batch, 7]
  - [ ] Check DQN forward pass works
  - [ ] Verify training loop runs

---

### Phase 2: Medium Wins (If Phase 1 succeeds) 🔄

- [ ] **File**: `src/models/vgae.py`
  - [ ] Ensure `forward()` returns z (latent vector)
  - [ ] Already implemented - verify accessibility

- [ ] **File**: `src/training/prediction_cache.py`
  - [ ] Add `compute_latent_summary()` method
  - [ ] Aggregate z per graph (mean/std/max/min)

- [ ] **File**: `src/training/lightning_modules.py`
  - [ ] Update state stacking to handle 11D state
  - [ ] Update DQN agent: `state_dim=11`

---

## Comparison Table: State Space Options

| State Configuration | Dims | Effort | Expected Accuracy | Training Time | Recommended? |
|---------------------|------|--------|-------------------|---------------|--------------|
| **Current (Baseline)** | 2 | - | Baseline | 1× | - |
| **Quick Wins (Phase 1)** | 7 | 30 min | +2-5% | 1.1× | ✅ YES |
| **+ VGAE Latent (Phase 2)** | 11 | +30 min | +4-8% | 1.15× | ✅ YES (after Phase 1) |
| **+ GAT Embeddings** | 15 | +35 min | +5-10% | 1.2× | ⚠️ MAYBE |
| **Full (All Options)** | 19+ | +90 min | +7-12% | 1.3× | ❌ NO (diminishing returns) |

---

## Answer to User's Questions

### Q1: Does DQN only get binary scores from GAT [0,1]?

**A**: Almost - it gets a single probability value from GAT's softmax (probability of class 1, attack). Not binary (0 or 1), but a continuous value in [0, 1]. However, it's a **single scalar** which collapses information.

**Full GAT Output Available**:
- Logits: [batch_size, 2] - raw scores before softmax
- Softmax probs: [batch_size, 2] - normalized probabilities
- **Currently used**: Only `probs[:, 1]` (probability of attack)

### Q2: Is there something more informative?

**A**: YES! Many options available (see table above). The DQN could receive:
- Full logits (both classes) instead of just class 1 probability
- Model confidence indicators (softmax entropy)
- VGAE's 3 error components separately
- Latent space summaries from VGAE
- Pre-pooling embeddings from GAT

### Q3: Quick wins for richer state space?

**A**: **Top 3 Quick Wins** (30 min total implementation):

1. **Separate VGAE error components** (5 min) - 2 → 4 dims
2. **GAT full logits** (3 min) - +1 dim
3. **Confidence indicators** (10 min) - +2 dims

**Result**: 2D → 7D state space in 30 minutes with minimal code changes.

---

## Next Steps

1. ✅ Review this analysis with user
2. ⏸️ Get approval for Phase 1 implementation
3. ⏸️ Implement Phase 1 quick wins (30 min)
4. ⏸️ Test on small dataset to verify improvements
5. ⏸️ If successful, proceed to Phase 2

---

**Status**: Analysis complete, awaiting user decision on implementation priority

