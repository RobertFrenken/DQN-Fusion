# DQN 15D State Space: Detailed Embedding Explanation

**Date**: 2026-01-27
**Status**: Implementation Complete
**Purpose**: Comprehensive documentation of the 15-dimensional state representation for DQN fusion

---

## Executive Summary

The DQN fusion agent now operates on a **15-dimensional state space**, up from the previous 2-dimensional representation. Each state vector represents a **single graph sample** and contains rich features extracted from both VGAE and GAT models.

**Critical Distinction**: All embedding statistics are **per-graph aggregations**. For each individual graph sample, we compute statistics by aggregating node-level features within that specific graph. These are **NOT** fixed values computed from all training data.

---

## Overview: What is a "Graph Sample"?

In our CAN intrusion detection system:
- **Input**: Time-windowed sequences of CAN bus messages (e.g., 100 messages)
- **Graph Representation**: Each window becomes a graph where:
  - **Nodes** = CAN messages
  - **Edges** = Temporal or ID-based connections between messages
  - **Node Features** = CAN ID embedding + 7 numerical features (DLC, data bytes, etc.)
- **Graph-Level Label**: Binary classification (0 = normal, 1 = attack/anomaly)

**Key Point**: Each graph has multiple nodes (typically 100 nodes for window_size=100), and we need a single state vector per graph for the DQN.

---

## 15D State Vector Structure

For a single graph sample, the 15D state vector is composed of:

```
[vgae_err_node, vgae_err_neighbor, vgae_err_canid,    # VGAE errors (3 dims)
 vgae_lat_mean, vgae_lat_std, vgae_lat_max, vgae_lat_min,  # VGAE latent stats (4 dims)
 vgae_confidence,                                       # VGAE confidence (1 dim)
 gat_logit_0, gat_logit_1,                             # GAT logits (2 dims)
 gat_emb_mean, gat_emb_std, gat_emb_max, gat_emb_min,  # GAT embedding stats (4 dims)
 gat_confidence]                                        # GAT confidence (1 dim)
```

**Total: 8 VGAE features + 7 GAT features = 15 dimensions**

---

## Part 1: VGAE Features (8 Dimensions)

### What is VGAE?

**VGAE (Variational Graph Autoencoder)** is an unsupervised model that learns to reconstruct graph structures. It consists of:
1. **Encoder**: Maps nodes to a latent space (mean and variance vectors)
2. **Latent Sampling**: Samples latent embeddings `z` from N(mean, variance)
3. **Decoder**: Reconstructs node features and graph structure from `z`

### VGAE Architecture in Our Implementation

- **Encoder**: 4 layers of GATConv with batch normalization
  - Input: CAN ID embedding (8D) + 7 continuous features
  - Hidden dimensions: [256, 128, 96, 48]
  - Outputs: `z_mean` and `z_logvar` for each node
  - **Latent dimension**: 48 (each node gets a 48-dimensional latent vector)

- **Decoder**: 2 pathways
  - **Node Decoder**: Reconstructs 7 continuous features
  - **CAN ID Classifier**: Predicts CAN ID from latent `z`
  - **Neighborhood Decoder**: Predicts which CAN IDs are neighbors

### VGAE Feature Extraction (Per Graph)

For each graph with N nodes, we compute:

#### 1. Error Components (3 dimensions)

These measure how well the VGAE reconstructs different aspects of the graph:

**a) Node Reconstruction Error** (`vgae_err_node`):
- **Definition**: Mean squared error between reconstructed and original continuous features
- **Computation**:
  ```python
  node_errors = MSE(reconstructed_features, original_features)  # [N_nodes]
  vgae_err_node = mean(node_errors[nodes_in_this_graph])  # Scalar per graph
  ```
- **Interpretation**: Higher error = anomalous patterns that deviate from learned normal structure
- **Range**: [0, ∞), typically normalized to [0, 1]

**b) CAN ID Classification Error** (`vgae_err_canid`):
- **Definition**: Cross-entropy loss for CAN ID prediction
- **Computation**:
  ```python
  canid_errors = CrossEntropy(predicted_canid, true_canid)  # [N_nodes]
  vgae_err_canid = mean(canid_errors[nodes_in_this_graph])  # Scalar per graph
  ```
- **Interpretation**: Higher error = CAN ID usage patterns differ from normal
- **Range**: [0, ∞), typically [0, 5]

**c) Neighborhood Structure Error** (`vgae_err_neighbor`):
- **Definition**: Binary cross-entropy for neighbor prediction
- **Computation**:
  ```python
  neighbor_errors = BCE(predicted_neighbors, true_neighbors)  # [N_nodes]
  vgae_err_neighbor = mean(neighbor_errors[nodes_in_this_graph])  # Scalar per graph
  ```
- **Interpretation**: Higher error = graph connectivity differs from normal patterns
- **Range**: [0, ∞), typically [0, 1]

**Key Point**: Each error component is computed **per node**, then **averaged across nodes within a single graph**. Different graphs will have different error values based on their content.

#### 2. Latent Space Statistics (4 dimensions)

These capture the distribution of latent representations within a single graph:

**What is Latent Space?**
- After encoding, each node has a latent vector `z` ∈ ℝ^48 (48-dimensional)
- The latent space is a compressed representation capturing the "essence" of each node
- Normal traffic tends to cluster in certain regions of latent space
- Anomalies often appear as outliers or in unexplored regions

**Per-Graph Statistics**:

**a) Latent Mean** (`vgae_lat_mean`):
- **Computation**:
  ```python
  z_vectors = [z_1, z_2, ..., z_N]  # Each z_i is 48-dimensional
  vgae_lat_mean = mean(flatten(z_vectors))  # Mean of all 48×N values in this graph
  ```
- **Interpretation**: Average activation level in latent space for this graph
- **Range**: Unbounded, typically [-5, 5]

**b) Latent Std** (`vgae_lat_std`):
- **Computation**:
  ```python
  vgae_lat_std = std(flatten(z_vectors))  # Std dev of all values in this graph
  ```
- **Interpretation**: Spread/diversity of node representations within this graph
  - High std = nodes have diverse representations
  - Low std = nodes are similar to each other
- **Range**: [0, ∞), typically [0, 10]

**c) Latent Max** (`vgae_lat_max`):
- **Computation**:
  ```python
  vgae_lat_max = max(flatten(z_vectors))  # Maximum value across all dimensions
  ```
- **Interpretation**: Peak activation in latent space
  - Unusually high values may indicate anomalies
- **Range**: Unbounded, typically [-10, 10]

**d) Latent Min** (`vgae_lat_min`):
- **Computation**:
  ```python
  vgae_lat_min = min(flatten(z_vectors))  # Minimum value across all dimensions
  ```
- **Interpretation**: Minimum activation in latent space
- **Range**: Unbounded, typically [-10, 10]

**Critical Insight**: These statistics are computed **per graph** by aggregating the latent vectors of all nodes in that specific graph. Each graph sample gets its own unique set of statistics based on its content. The DQN learns to associate certain latent space distributions with normal vs. anomalous traffic.

#### 3. VGAE Confidence (1 dimension)

**Definition**: Inverse of error variance across the three error components

**Computation**:
```python
error_vector = [vgae_err_node, vgae_err_canid, vgae_err_neighbor]
vgae_confidence = 1.0 / (1.0 + variance(error_vector))
```

**Interpretation**:
- **High confidence** (close to 1): All three error components are consistent (similar magnitude)
  - Example: [0.1, 0.12, 0.11] → low variance → high confidence
- **Low confidence** (close to 0): Error components disagree (one is much higher/lower)
  - Example: [0.1, 0.5, 0.09] → high variance → low confidence

**Range**: [0, 1]

**Why This Matters**: When the VGAE's three error types align, it's more confident in its assessment. Disagreement suggests uncertain or borderline cases.

---

## Part 2: GAT Features (7 Dimensions)

### What is GAT?

**GAT (Graph Attention Network)** is a supervised classification model that uses attention mechanisms to weigh neighbor importance. It's trained to classify graphs as normal (0) or attack (1).

### GAT Architecture in Our Implementation

- **3 GATConv Layers** with Jumping Knowledge (JK) connections
  - Input: CAN ID embedding (8D) + 7 continuous features
  - Hidden channels: Typically 128 or 256
  - Attention heads: 4 (multi-head attention)
  - Dropout: 0.2

- **Jumping Knowledge Aggregation**
  - Mode: Concatenation
  - Combines representations from all layers

- **Global Mean Pooling**
  - Aggregates node embeddings into a single graph-level embedding

- **Fully Connected Layers**
  - 3 FC layers with ReLU and dropout
  - Output: 2 logits (one for each class)

### GAT Feature Extraction (Per Graph)

#### 1. Logits (2 dimensions)

**Definition**: Raw output scores from the final FC layer **before** softmax activation

**Computation**:
```python
# Forward pass through GAT
logits = GAT_model(graph)  # [num_nodes, 2] initially

# Aggregate per graph (mean across nodes before pooling)
graph_logits = logits[nodes_in_this_graph].mean(dim=0)  # [2]
gat_logit_0 = graph_logits[0]  # Logit for class 0 (normal)
gat_logit_1 = graph_logits[1]  # Logit for class 1 (attack)
```

**Interpretation**:
- **Logit for class 0**: How much the model "believes" this is normal traffic
- **Logit for class 1**: How much the model "believes" this is an attack
- **Prediction**: argmax(logits) = 1 if logit_1 > logit_0, else 0
- **Probability**: softmax([logit_0, logit_1]) converts to probabilities summing to 1

**Range**: Unbounded, typically [-10, 10]

**Why Both Logits?**
- Provides more information than just a single probability
- The **difference** (logit_1 - logit_0) indicates confidence magnitude
- The **absolute values** indicate how strongly the model leans toward each class
- Example:
  - [0.1, 0.2]: Weak preference for attack (low confidence)
  - [−5.0, 5.0]: Strong confidence in attack prediction
  - [−0.1, 0.1]: Very uncertain (near decision boundary)

#### 2. Pre-Pooling Embedding Statistics (4 dimensions)

**What are Pre-Pooling Embeddings?**

The GAT model processes graphs in several stages:
1. **Input**: Node features (CAN ID + 7 values)
2. **3 GATConv Layers**: Each node gets updated embeddings based on neighbors
3. **Jumping Knowledge**: Concatenate embeddings from all 3 layers
4. **Result**: Each node has a high-dimensional embedding (e.g., 384D if hidden=128, 3 layers, heads=4)
5. **Global Pooling**: Aggregate all node embeddings into a single graph-level embedding
6. **FC Layers**: Final classification

**Pre-pooling embeddings** are the node-level representations **after** JK aggregation but **before** global pooling (step 4 → 5). These capture fine-grained node-level patterns.

**Per-Graph Statistics**:

For each graph with N nodes, we have N embedding vectors (each ~384D). We compute:

**a) Embedding Mean** (`gat_emb_mean`):
- **Computation**:
  ```python
  embeddings = [emb_1, emb_2, ..., emb_N]  # Each emb_i is 384-dimensional
  gat_emb_mean = mean(flatten(embeddings))  # Mean across all 384×N values
  ```
- **Interpretation**: Average node activation in this graph
- **Range**: Unbounded, typically [-5, 5]

**b) Embedding Std** (`gat_emb_std`):
- **Computation**:
  ```python
  gat_emb_std = std(flatten(embeddings))  # Std dev across all values
  ```
- **Interpretation**: Diversity of node representations
  - High std = nodes have different patterns
  - Low std = nodes are homogeneous
- **Range**: [0, ∞), typically [0, 10]

**c) Embedding Max** (`gat_emb_max`):
- **Computation**:
  ```python
  gat_emb_max = max(flatten(embeddings))  # Maximum activation
  ```
- **Interpretation**: Peak response in any dimension
  - Extreme values may indicate unusual patterns
- **Range**: Unbounded, typically [-10, 10]

**d) Embedding Min** (`gat_emb_min`):
- **Computation**:
  ```python
  gat_emb_min = min(flatten(embeddings))  # Minimum activation
  ```
- **Interpretation**: Minimum response
- **Range**: Unbounded, typically [-10, 10]

**Critical Insight**: Like VGAE, these are **per-graph statistics**. Each graph sample gets its own embedding statistics computed from its nodes. The DQN can learn to detect attacks based on patterns like:
- "Normal traffic has std=2.5, but this graph has std=5.0" (high diversity = suspicious)
- "Normal max activation is ~3.0, but this graph has max=8.0" (extreme activation = anomaly)

#### 3. GAT Confidence (1 dimension)

**Definition**: 1 − normalized entropy of softmax probabilities

**Computation**:
```python
# Convert logits to probabilities
probs = softmax([gat_logit_0, gat_logit_1])  # [p_0, p_1], sum to 1

# Compute entropy
entropy = -(p_0 * log(p_0) + p_1 * log(p_1))

# Normalize by max possible entropy (log(2) for binary classification)
max_entropy = log(2)
normalized_entropy = entropy / max_entropy

# Confidence is inverse of entropy
gat_confidence = 1.0 - normalized_entropy
```

**Interpretation**:
- **High confidence** (close to 1): Peaked distribution
  - Example: probs=[0.95, 0.05] → low entropy → high confidence
- **Low confidence** (close to 0): Uniform distribution
  - Example: probs=[0.51, 0.49] → high entropy → low confidence (near decision boundary)

**Range**: [0, 1]

**Why This Matters**: High confidence means the GAT is certain about its prediction. Low confidence indicates borderline cases where additional information (from VGAE) would be valuable.

---

## Answering the User's Question

**User Asked**: "Are the statistics an aggregation of each graph sample (aggregating nodes) or are they a fixed value derived from all the graph data?"

**Answer**: **Per-Graph Aggregations** (NOT fixed values)

### Detailed Explanation:

1. **VGAE Latent Statistics** (4D):
   - Each graph sample has N nodes
   - Each node has a 48-dimensional latent vector `z`
   - We compute mean/std/max/min **across all N×48 values within that single graph**
   - Result: 4 scalar statistics unique to that graph
   - **Example**:
     - Graph A (100 nodes): latent_mean=1.2, latent_std=2.5
     - Graph B (100 nodes): latent_mean=−0.3, latent_std=4.1
     - Different values because different graph content

2. **GAT Embedding Statistics** (4D):
   - Each graph sample has N nodes
   - Each node has a 384-dimensional embedding vector (after JK)
   - We compute mean/std/max/min **across all N×384 values within that single graph**
   - Result: 4 scalar statistics unique to that graph
   - **Example**:
     - Graph A: emb_mean=0.8, emb_std=1.9
     - Graph B: emb_mean=−1.2, emb_std=3.5
     - Different values because different node patterns

### Why NOT Fixed Values?

If we used fixed statistics computed from all training data, we would lose all sample-specific information:
- Every sample would have the same latent_mean (e.g., global average of all training samples)
- Every sample would have the same emb_std (e.g., global std from all training)
- The DQN would have no way to distinguish between samples

Instead, by computing per-graph statistics:
- The DQN can learn: "Normal traffic has latent_std ≈ 2.0, but attacks have latent_std > 4.0"
- The DQN can detect: "This specific graph has unusually high embedding variance compared to typical normal traffic"

---

## How the DQN Uses These 15 Features

### Learning Process

During training, the DQN observes thousands of 15D state vectors paired with labels (normal/attack). It learns patterns like:

1. **VGAE Error Patterns**:
   - "If vgae_err_node > 0.5 AND vgae_err_neighbor > 0.3, likely attack"
   - "Low error but high error variance → uncertain, rely more on GAT"

2. **Latent Space Patterns**:
   - "Normal traffic: latent_mean ∈ [−1, 1], latent_std ∈ [1, 3]"
   - "Attacks: latent_mean > 3 OR latent_std > 5 (outliers in latent space)"

3. **GAT Logit Patterns**:
   - "If gat_logit_1 > gat_logit_0 + 2, strong attack signal"
   - "If |gat_logit_1 − gat_logit_0| < 0.5, GAT is uncertain → check VGAE errors"

4. **Embedding Statistics Patterns**:
   - "Normal: emb_std ≈ 2.0, attacks: emb_std > 4.0 (nodes have diverse patterns)"
   - "Extreme emb_max or emb_min → potential anomaly"

5. **Confidence-Based Weighting**:
   - "If vgae_confidence > 0.8 AND gat_confidence < 0.3, trust VGAE more"
   - "If both confidence values > 0.9 and they agree, high certainty"
   - "If confidences disagree, combine features to resolve ambiguity"

### Fusion Decision

The DQN outputs an action (alpha ∈ [0, 1]) that determines the fusion weight:
```python
fused_score = (1 - alpha) * vgae_anomaly_score + alpha * gat_probability
prediction = 1 if fused_score > 0.5 else 0
```

The 15D state allows the DQN to learn **adaptive fusion**:
- When VGAE is confident and GAT is uncertain → increase (1−alpha) (trust VGAE)
- When GAT is confident and VGAE has low errors → increase alpha (trust GAT)
- When both models disagree but have unique patterns in their embeddings → learn nuanced fusion

---

## Summary Table

| Feature | Dimensions | Type | Computation | Range | Aggregation Type |
|---------|-----------|------|-------------|-------|-----------------|
| VGAE Errors | 3 | Scalar per graph | MSE/CE/BCE averaged over nodes | [0, ∞) | **Per-graph mean** |
| VGAE Latent Stats | 4 | Scalar per graph | mean/std/max/min of all node latents | Unbounded | **Per-graph statistics** |
| VGAE Confidence | 1 | Scalar per graph | 1/(1+var(errors)) | [0, 1] | **Per-graph variance** |
| GAT Logits | 2 | Scalar per graph | Final FC layer outputs (before softmax) | Unbounded | **Per-graph mean** |
| GAT Embedding Stats | 4 | Scalar per graph | mean/std/max/min of all node embeddings | Unbounded | **Per-graph statistics** |
| GAT Confidence | 1 | Scalar per graph | 1 − normalized entropy | [0, 1] | **Per-graph entropy** |
| **Total** | **15** | **15 scalars** | **All per-graph** | **Various** | **Per-graph aggregations** |

---

## Key Takeaways

1. **Every feature is a per-graph aggregation**: We compute statistics from the nodes within each individual graph sample.

2. **No global statistics**: We do NOT use fixed values computed from all training data.

3. **Rich representation**: 15D captures reconstruction errors, latent space structure, classification logits, intermediate embeddings, and confidence metrics.

4. **Enables adaptive fusion**: The DQN can learn complex policies for combining VGAE and GAT based on each model's internal representations and confidence.

5. **Expected improvement**: +6-10% accuracy compared to the previous 2D state space (simple [anomaly_score, gat_prob]).

---

## Files Modified for 15D Implementation

- `src/training/prediction_cache.py`: Feature extraction from VGAE and GAT
- `src/training/lightning_modules.py`: FusionPredictionCache dataclass and training loop
- `src/models/dqn.py`: Q-network input layer, state normalization, reward computation
- `src/evaluation/evaluation.py`: Fusion inference with 15D states
- `src/training/modes/fusion.py`: Updated to use new 15D cache format

---

**Status**: Implementation complete and syntax-checked across entire pipeline.
**Next Steps**: Test fusion training with 15D states and evaluate performance improvement.
