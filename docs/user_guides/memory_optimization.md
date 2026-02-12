# Memory Optimization Guide

This guide documents the memory optimization features in the KD-GAT pipeline, designed to prevent OOM (Out of Memory) errors on both CPU and GPU when processing large datasets.

## Overview

The pipeline processes CAN bus data as graphs, which can consume significant memory:
- **hcrl_sa**: ~9K graphs (works on most systems)
- **set_01-04, hcrl_ch**: ~50-100K+ graphs (requires optimization)

## Memory Optimizations Implemented

### 1. Memory-Mapped Graph Loading

**File**: `src/training/datamodules.py`

**Problem**: Loading cached graphs via `torch.load()` deserializes the entire cache into RAM, causing CPU OOM on large datasets.

**Solution**: Use PyTorch's `mmap=True` parameter to memory-map the cache file. Graphs are loaded on-demand from disk rather than all at once.

```python
# Before (loads everything into RAM)
graphs = torch.load(cache_file, map_location='cpu')

# After (memory-mapped, loads on-demand)
graphs = torch.load(cache_file, map_location='cpu', mmap=True)
```

**Requirements**: PyTorch 2.1+

**Expected Impact**:
- 50-80% reduction in peak RAM usage during data loading
- Slight increase in data access latency (disk I/O)

---

### 2. Teacher CPU Offloading

**File**: `pipeline/stages/modules.py` (VGAEModule, GATModule)

**Problem**: During Knowledge Distillation, both teacher and student models occupy GPU memory simultaneously, causing GPU OOM.

**Solution**: Offload teacher model to CPU after each forward pass, bringing it back only when needed.

**Configuration**:
```bash
# Enable via CLI
python -m pipeline.cli autoencoder --preset vgae,student \
    --use-kd true \
    --teacher-path path/to/teacher.pt \
    --offload-teacher-to-cpu true
```

Or in config:
```json
{
  "offload_teacher_to_cpu": true
}
```

**Expected Impact**:
- Teacher model (~50-200MB) freed from VRAM between batches
- ~10-20% training slowdown due to CPU<->GPU transfers
- Enables training larger students or using larger batch sizes

**When to Use**:
- GPU memory < 16GB
- Teacher model > 100MB
- Getting "CUDA out of memory" during KD training

---

### 3. Chunked Difficulty Scoring

**File**: `pipeline/stages/training.py` (`_score_difficulty`)

**Problem**: Curriculum learning scores every training graph's difficulty, iterating through all graphs sequentially. Memory accumulates from tensor allocations.

**Solution**: Process graphs in chunks and clear GPU cache between chunks.

```python
# Process 500 graphs at a time
for chunk in chunks(graphs, size=500):
    scores.extend(score_chunk(chunk))
    torch.cuda.empty_cache()  # Free accumulated tensors
```

**Configuration**: Chunk size is hardcoded at 500 graphs (optimal for most GPU memory sizes).

**Expected Impact**:
- Prevents GPU memory accumulation during curriculum stage
- Adds ~5% overhead from cache clearing
- Enables curriculum learning on datasets with 100K+ graphs

---

### 4. Context-Aware Batch Size Estimation

**File**: `pipeline/memory.py`

**Problem**: Static `safety_factor` approach missed many memory consumers:
- CUDA context overhead (~500MB fixed)
- Embedding layers (`nn.Embedding` can be huge for large num_ids)
- Activation memory (scales with nodes x hidden x layers)
- Message passing tensors (scales with edges x hidden)
- GAT attention weights (scales with edges x heads)
- Allocator fragmentation (10-30% overhead)

**Solution**: Two-mode estimation system:

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| `static` | Fast | Conservative | Quick estimates |
| `measured` | Medium | Accurate | Default for training |

```python
from pipeline.memory import compute_batch_size

# Fast heuristic
budget = compute_batch_size(model, sample_graph, device, mode="static")

# Forward hooks measure actual activations (default)
budget = compute_batch_size(model, sample_graph, device, mode="measured")
```

**How it works**:
1. **Static**: Count parameters, embeddings, optimizer states, CUDA context
2. **Measured**: Run forward hooks to measure actual activation memory

**Memory Budget Breakdown**:
```
MemoryBudget(total=16000MB static=545.0MB
  [params=10.0 embed=5.0 opt=20.0 grad=10.0 teacher=0.0]
  activation=50.0MB per_graph=0.003MB
  available=10000.0MB batch_size=2048 mode=measured)
```

**Configuration**:
```json
{
  "optimize_batch_size": true,
  "memory_estimation": "measured"
}
```

**Expected Impact**:
- Explicit embedding memory tracking (catches large num_ids)
- CUDA context overhead (500MB) properly accounted
- Fragmentation buffer (10%) prevents edge-case OOMs
- Activation memory measured, not guessed

---

## Configuration Reference

Memory-related settings in `PipelineConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gradient_checkpointing` | `True` | Trade compute for memory in model layers |
| `offload_teacher_to_cpu` | `False` | Move teacher to CPU between batches during KD |
| `use_teacher_cache` | `True` | Cache teacher outputs (increases memory) |
| `clear_cache_every_n` | `100` | Clear CUDA cache every N steps |
| `optimize_batch_size` | `True` | Use context-aware batch sizing |
| `memory_estimation` | `measured` | Estimation mode: `static` or `measured` |
| `safety_factor` | `0.5-0.6` | Target GPU utilization (maps to ~65-75%) |
| `batch_size` | `4096` | Maximum batch size (upper bound for auto-sizing) |

---

## Troubleshooting OOM Errors

### CPU OOM (system memory exhausted)

1. **Check dataset size**: `ls -lh data/cache/*/processed_graphs.pt`
2. **Reduce num_workers**: `--num-workers 4` (fewer parallel data loaders)
3. **Increase SLURM memory**: `--mem=256000` in profile
4. **Memory mapping**: Ensure PyTorch 2.1+ for `mmap=True` support

### GPU OOM (CUDA out of memory)

1. **Enable gradient checkpointing**: `--gradient-checkpointing true`
2. **Reduce batch size**: Lower `safety_factor` (0.4-0.5)
3. **Offload teacher**: `--offload-teacher-to-cpu true`
4. **Use mixed precision**: `--precision 16-mixed` (already default)

### Diagnosing Memory Issues

Monitor memory during training:
```bash
# Watch GPU memory
watch -n 1 nvidia-smi

# Check CPU memory in job
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

MLflow logs memory metrics at each epoch - check the tracking UI for trends.

---

## Architecture: Memory Module

The memory management system is in `pipeline/memory.py`:

```
pipeline/memory.py
├── MemoryBudget (dataclass)        # Detailed breakdown
├── compute_batch_size()            # Main entry point
├── log_memory_state()              # Debug logging
│
├── Static Estimation (internal)
│   ├── _get_gpu_memory_mb()
│   ├── _count_embedding_memory_mb()
│   ├── _estimate_model_memory_mb()
│   ├── _estimate_graph_memory_mb()
│   └── _estimate_activation_heuristic()
│
└── Measured Estimation (internal)
    └── _measure_activation_memory_mb()
```

**Key Parameters** for `compute_batch_size()`:
- `mode`: `"static"` or `"measured"` (string)
- `target_utilization`: How much GPU memory to use (0.0-1.0)
- `min_batch_size` / `max_batch_size`: Bounds for output

**Constants**:
- `CUDA_CONTEXT_MB = 500`: Fixed CUDA context overhead
- `FRAGMENTATION_BUFFER = 0.10`: 10% buffer for allocator fragmentation
