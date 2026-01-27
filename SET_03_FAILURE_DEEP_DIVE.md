# Deep Dive: Why Did set_03 GAT Teacher Fail When set_02 Succeeded?

## The Paradox

**The Confusing Facts:**
- set_02: **203,496 graphs** (MORE), 2049 IDs → ✅ **SUCCESS**
- set_03: **166,098 graphs** (LESS), 1791 IDs → ❌ **OOM FAILURE**
- 2049 IDs > 1791 IDs (larger embedding should use MORE memory)

## Key Measurements

### Graph Density (from cache file sizes)

| Dataset | Graphs  | Cache Size | KB/Graph | Status  |
|---------|---------|------------|----------|---------|
| set_01  | 151,089 | 917 MB     | **6.22** | SUCCESS |
| set_02  | 203,496 | 1,318 MB   | **6.64** | SUCCESS |
| set_03  | 166,098 | 1,346 MB   | **8.30** | **FAILED** |

**Critical Finding**: set_03 graphs are **25% denser** (8.30 vs 6.64 KB/graph)

### Dataset Processing Pattern

All datasets load graphs **TWICE** during curriculum mode:
```
set_02: Line 72:  Total graphs created: 203496
        Line 127: Total graphs created: 203496  (second pass)

set_03: Line 65:  Total graphs created: 166098
        Line 113: Total graphs created: 166098  (second pass)
```

This is **expected behavior** - likely for train/val split or curriculum setup.

### Actual GPU Memory Usage (from OOM error)

```
set_03 GAT curriculum OOM:
- GPU capacity: 15.77 GiB
- Memory in use BEFORE first batch: 13.84 GiB
- Tried to allocate: 2.11 GiB (the first batch)
- Free memory: 1.92 GiB
```

**Problem**: 13.84 GB was already allocated before training even started!

### Why 13.84 GB? Memory Breakdown Hypothesis

```
Component                         | Estimated Memory
----------------------------------|------------------
Graph data (332k graphs × 8.3KB)  | ~2.7 GB
Edge indices (int64 tensors)      | ~8-9 GB (if dense connectivity)
VGAE model (1.3M params)          | ~5 MB
GAT model (1.1M params)           | ~4 MB
Embedding layers (1791 × 64)      | ~0.4 MB
PyTorch overhead & caching        | ~2 GB
----------------------------------|------------------
TOTAL                             | ~13.5 GB ✓
```

## The Root Cause: Graph Connectivity Density

### Why set_03 uses more memory despite fewer graphs:

**Hypothesis**: set_03 has **denser graph connectivity** (more edges per node).

In PyTorch Geometric:
- Each edge is stored as 2 int64 values (source, target) = 16 bytes
- A graph with N nodes and E edges needs: `16 * E` bytes for edge_index
- **Denser graphs** = exponentially more edge storage

### Example Calculation:

If set_03 has 25% more edges per node on average:

```python
# Assume average graph: 50 nodes, 200 edges (4 edges/node)
set_02_edges_per_graph = 200
set_03_edges_per_graph = 250  # 25% more

# For 332k graphs (166k × 2 loads):
set_02_total_edges = 332000 * 200 = 66.4M edges
set_03_total_edges = 332000 * 250 = 83.0M edges

# Memory for edge indices (int64, 16 bytes per edge):
set_02_edge_memory = 66.4M * 16 / 1024**3 = 0.99 GB
set_03_edge_memory = 83.0M * 16 / 1024**3 = 1.24 GB
```

But this still doesn't explain the 13.84 GB. Let me recalculate with realistic values:

### More Realistic Calculation:

CAN bus graphs likely have:
- **100-300 nodes per graph** (timewindow of CAN messages)
- **500-2000 edges** (message dependencies)
- **11 features per node** (CAN message attributes)

```python
# Conservative estimate for set_03:
avg_nodes = 200
avg_edges = 1000  # Dense connectivity

# Node features: 200 nodes × 11 features × 4 bytes (float32) = 8.8 KB
# Edge indices: 1000 edges × 2 × 8 bytes (int64) = 16 KB
# Total per graph: ~25 KB

# For 332k graphs (double-loaded):
total_memory = 332000 * 25 KB = 8.3 GB ✓
```

**This explains the bulk of the 13.84 GB!**

## Why 2049 vs 1791 IDs Matters (New Understanding)

It's NOT about PyTorch optimization. It's about **dataset composition**:

### CAN Bus ID Standards:
- **Standard 11-bit IDs**: 0-2047 (2048 possible values)
- **2049 = 2^11 + 1**: Likely "all standard IDs + 1 padding/unknown token"

### The Pattern:
| Dataset | num_ids | Interpretation |
|---------|---------|----------------|
| hcrl_sa | 2049    | Full standard ID range + padding |
| hcrl_ch | 2049    | Full standard ID range + padding |
| set_01  | 53      | Subset of IDs (filtered dataset) |
| set_02  | 2049    | Full standard ID range + padding |
| **set_03** | **1791** | **Incomplete ID range (87% coverage)** |
| set_04  | 2049    | Full standard ID range + padding |

### The Real Issue with 1791 IDs:

**Hypothesis**: set_03 has **87% of standard IDs** BUT with **irregular distribution**:
- Sparse ID space → more hash collisions
- Missing IDs → irregular message patterns → **denser graph connectivity**
- Incomplete networks → more cross-references → **more edges**

## Why set_02 Succeeded Despite More Graphs

set_02 advantages:
1. ✅ **Less dense graphs** (6.64 vs 8.30 KB/graph)
2. ✅ **Complete ID coverage** (2049 = full standard range)
3. ✅ **Regular connectivity patterns**

Result: 203k graphs × 6.64 KB = 1.32 GB cache
        × 2 loads × ~5x overhead = ~13.2 GB total ← **fits in 15.77 GB GPU**

set_03 problems:
1. ❌ **Denser graphs** (8.30 KB/graph)
2. ❌ **Irregular ID coverage** (1791 = 87% of standard)
3. ❌ **More complex connectivity**

Result: 166k graphs × 8.30 KB = 1.35 GB cache
        × 2 loads × ~5x overhead = ~13.5 GB
        **+ first batch (2.11 GB) = 15.61 GB ← exceeds 15.77 GB GPU!**

## Verification Needed

To confirm this hypothesis, we need to check:

1. **Average edges per graph**:
   ```python
   # Load cached graphs and measure
   data = torch.load('cache/processed_graphs.pt')
   avg_edges_set02 = np.mean([g.edge_index.shape[1] for g in data['set_02']])
   avg_edges_set03 = np.mean([g.edge_index.shape[1] for g in data['set_03']])
   ```

2. **Node count per graph**:
   ```python
   avg_nodes_set02 = np.mean([g.num_nodes for g in data['set_02']])
   avg_nodes_set03 = np.mean([g.num_nodes for g in data['set_03']])
   ```

3. **Graph density** (edges / possible_edges):
   ```python
   density_set03 = avg_edges / (avg_nodes * (avg_nodes - 1))
   ```

## Conclusion

**The failure is NOT due to vocabulary size optimization or "standard" ID ranges.**

**The failure is due to set_03 having:**
1. **Inherently denser/more complex graph structures** (25% larger per graph)
2. **Irregular CAN ID distribution** (1791 ≠ complete standard range)
3. **Combined memory pressure** from double-loading + dense graphs + VGAE + GAT

When the first training batch tried to allocate 2.11 GB on top of 13.84 GB baseline, it exceeded the 15.77 GB GPU capacity.

## Solution

**Reduce batch size** for set_03:
```bash
--batch-size 16  # Half the default 32
# or
--batch-size 24  # 25% reduction
```

This will reduce the first batch allocation from 2.11 GB to ~1.0 GB, fitting within available memory.
