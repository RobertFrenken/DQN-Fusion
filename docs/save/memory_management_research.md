# Memory Management for KD-GAT Pipeline: Research Report

**Date:** 2026-02-03
**Context:** Addressing CPU/GPU OOM errors in PyTorch Geometric GNN training on SLURM HPC

---

## Executive Summary

Your KD-GAT pipeline processes CAN bus graph data through multiple training stages. Given the difficulty in pre-sizing graph data and recurring OOM issues, this report evaluates memory management libraries with SLURM integration.

**Key Finding:** A layered approach works best:
- **Preprocessing:** Dask or cuDF for CSV → graph conversion
- **Training:** PyTorch native tools (checkpointing, AMP) + PyG sampling
- **Scaling:** Ray or DeepSpeed for multi-node distribution

---

## 1. Dask + dask-jobqueue

### How It Handles Memory
- **Lazy evaluation**: Task graphs built without execution until `.compute()`
- **Chunked processing**: Data split into memory-fitting partitions
- **Automatic spilling**: Workers spill to disk when exceeding limits
- **Memory-aware scheduling**: Tasks scheduled based on available memory

### SLURM Integration (Excellent)
```python
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

cluster = SLURMCluster(
    queue='gpu',
    account='PAS2022',
    cores=8,
    memory='64GB',
    processes=1,
    walltime='04:00:00',
    worker_extra_args=['--memory-limit', '60GB'],  # Leave headroom
)
cluster.scale(jobs=4)  # Launch 4 SLURM jobs as workers
client = Client(cluster)
```

### Applicability to Your Pipeline

| Aspect | Rating | Notes |
|--------|--------|-------|
| CSV preprocessing | ✅ Excellent | Parallelizes `graph_creation_optimized()` across files |
| ID mapping build | ✅ Excellent | Perfect for `build_complete_id_mapping_streaming()` |
| GNN training | ❌ Poor | Dask workers don't share GPU contexts well |
| Setup complexity | Medium | Requires cluster configuration |

### Recommended Use
```python
import dask.bag as db

# Parallelize preprocessing across CSV files
csv_files = find_csv_files(root_folder, 'train_')
graphs = (
    db.from_sequence(csv_files, npartitions=len(csv_files))
    .map(lambda f: dataset_creation_streaming(f, id_mapping))
    .map(lambda df: create_graphs_numpy(df, window_size, stride))
    .compute()
)
```

**Verdict:** Use for preprocessing only, not training.

---

## 2. Ray

### How It Handles Memory
- **Object Store (Plasma)**: Shared memory for zero-copy object sharing
- **Memory-aware scheduling**: Tracks object sizes for scheduling decisions
- **Automatic spilling**: Objects spill to disk when store is full
- **Reference counting**: Automatic distributed garbage collection

### SLURM Integration (Good)
```yaml
# cluster.yaml for ray-slurm
cluster_name: can-graph
provider:
  type: slurm
  account: PAS2022
  partition: gpu
```

Or manual startup:
```bash
# In SLURM script
ray start --head --port=6379 --object-store-memory=50000000000
```

### Applicability to Your Pipeline

| Aspect | Rating | Notes |
|--------|--------|-------|
| CSV preprocessing | ✅ Good | Ray Data supports streaming |
| GNN training | ✅ Good | Ray Train wraps PyTorch cleanly |
| Hyperparameter tuning | ✅ Excellent | Ray Tune could optimize batch sizes |
| Setup complexity | High | More complex than Dask |

### Recommended Use
```python
from ray import train
from ray.train.torch import TorchTrainer

def train_func():
    # Your GATModule training logic
    ...

trainer = TorchTrainer(
    train_func,
    scaling_config=train.ScalingConfig(
        num_workers=4,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 4}
    ),
)
```

**Verdict:** Good option for scaling to multi-node training later.

---

## 3. Vaex

### How It Handles Memory
- **Memory mapping**: Files mapped to virtual memory, OS handles paging
- **Zero-copy operations**: Operations don't duplicate data
- **Lazy evaluation**: Expressions computed on-demand
- **Virtual columns**: Computed columns use no additional memory

### SLURM Integration
None native. Works within single SLURM job.

### Applicability to Your Pipeline

| Aspect | Rating | Notes |
|--------|--------|-------|
| CSV preprocessing | ✅ Excellent | True out-of-core processing |
| ID mapping | ✅ Excellent | Memory-efficient unique operations |
| PyTorch integration | ❌ Poor | Requires conversion step |
| GPU support | ❌ None | CPU only |

### Recommended Use
```python
import vaex

def build_id_mapping_vaex(csv_files):
    """Memory-efficient ID mapping."""
    df = vaex.open(csv_files)  # Lazy load all files
    unique_ids = df['arbitration_id'].unique()  # Out-of-core
    return {id: idx for idx, id in enumerate(sorted(unique_ids))}
```

**Verdict:** Useful for memory-constrained preprocessing on CPU.

---

## 4. RAPIDS cuDF

### How It Handles Memory
- **GPU memory pools**: RMM (RAPIDS Memory Manager) for efficient allocation
- **Host spilling**: Spills to CPU when GPU is full (since RAPIDS 23.04)
- **Managed memory**: CUDA Unified Memory for auto CPU-GPU migration
- **Per-operation limits**: Can set memory thresholds

### Configuration
```python
import rmm
import cudf

# Initialize memory pool with spilling
rmm.reinitialize(
    pool_allocator=True,
    initial_pool_size=8 * 1024**3,  # 8GB
    maximum_pool_size=16 * 1024**3,  # 16GB max
)

# Enable spilling to host memory
cudf.set_option("spill", True)
cudf.set_option("spill_device_limit", "14GB")
```

### PyTorch Integration (Zero-Copy)
```python
import cudf
import torch

# cuDF -> PyTorch via DLPack (zero-copy)
gdf = cudf.read_csv('data.csv')
tensor = torch.as_tensor(gdf['column'].values, device='cuda')
```

### Applicability to Your Pipeline

| Aspect | Rating | Notes |
|--------|--------|-------|
| CSV preprocessing | ✅ Excellent | 10-100x faster than pandas |
| PyTorch transfer | ✅ Excellent | Zero-copy via DLPack |
| Hex conversion | ✅ Excellent | GPU-accelerated string ops |
| Availability | ⚠️ Check | Requires RAPIDS on your HPC |

### Recommended Use
```python
import cudf
import torch

def dataset_creation_cudf(csv_path, id_mapping):
    """GPU-accelerated preprocessing."""
    gdf = cudf.read_csv(csv_path)

    # GPU-accelerated hex conversion
    gdf['CAN_ID_int'] = gdf['arbitration_id'].str.htoi()

    # Normalize on GPU
    for i in range(8):
        gdf[f'Data{i+1}'] = gdf[f'Data{i+1}'].str.htoi() / 255.0

    # Zero-copy transfer to PyTorch
    x_tensor = torch.as_tensor(
        gdf[['CAN_ID_int'] + [f'Data{i+1}' for i in range(8)]].values,
        device='cuda'
    )
    return x_tensor
```

**Verdict:** Best option if RAPIDS is available on your cluster.

---

## 5. PyTorch Native Tools

### Already Available in Your Pipeline

#### Gradient Checkpointing
Your `GATWithJK` model already supports this:
```python
# src/models/models.py line 86
if self.use_checkpointing and x.requires_grad:
    x = checkpoint(lambda x_in, c=conv, ei=edge_index: c(x_in, ei).relu(),
                   x, use_reentrant=False)
```

**Enable by default** - saves ~30-50% activation memory with ~20% compute overhead.

#### Mixed Precision
Already configured via Lightning:
```python
precision=cfg.precision,  # "16-mixed", "bf16-mixed", "32"
```

**Memory savings:** ~50% reduction in model/activation memory.

### Additional Recommendations

#### 1. Memory-Efficient Attention (PyTorch 2.0+)
```python
# For attention-heavy models
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

#### 2. Memory Profiling
```python
from pytorch_memlab import MemReporter

reporter = MemReporter(model)
reporter.report()  # See memory by layer
```

#### 3. OOM Recovery in Auto-Tuning
```python
try:
    trainer.fit(...)
except RuntimeError as e:
    if "out of memory" in str(e):
        torch.cuda.empty_cache()
        cfg.batch_size //= 2
        # Retry with smaller batch
```

---

## 6. PyTorch Geometric Specific Solutions

### Mini-Batch Graph Sampling
For graphs too large to fit in memory:
```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # Sample 25 1-hop, 10 2-hop neighbors
    batch_size=128,
    input_nodes=train_mask,
)
```

### ClusterGCN for Large Graphs
```python
from torch_geometric.loader import ClusterData, ClusterLoader

cluster_data = ClusterData(data, num_parts=100)
loader = ClusterLoader(cluster_data, batch_size=10, shuffle=True)
```

---

## 7. DeepSpeed (For Future Scaling)

Microsoft's optimization library for large-scale training:

```python
import deepspeed

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config={
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},  # Offload to CPU
        },
        "fp16": {"enabled": True},
    }
)
```

**SLURM Integration:** Native multi-node support via `deepspeed --hostfile`.

---

## Summary Comparison

| Library | Memory Handling | SLURM Support | PyTorch/PyG Fit | Best For |
|---------|-----------------|---------------|-----------------|----------|
| **Dask** | Spilling, chunking | ✅ Excellent | ❌ Poor (training) | Preprocessing |
| **Ray** | Object store, spilling | ✅ Good | ✅ Good | Distributed training |
| **Vaex** | Memory mapping | ❌ None | ⚠️ Conversion needed | CPU preprocessing |
| **cuDF** | GPU pools, spilling | ⚠️ In-job only | ✅ Excellent | GPU preprocessing |
| **PyTorch** | Checkpointing, AMP | Via Lightning | ✅ Native | Training |
| **DeepSpeed** | ZeRO, offloading | ✅ Excellent | ✅ Excellent | Large-scale training |

---

## Recommended Action Plan

### Immediate (This Week)
1. **Enable gradient checkpointing** in config.json:
   ```json
   {"gradient_checkpointing": true}
   ```

2. **Ensure mixed precision** is on:
   ```json
   {"precision": "16-mixed"}
   ```

3. **Add memory monitoring** to MLflow:
   ```python
   import psutil, torch

   def log_memory():
       mlflow.log_metrics({
           "cpu_mem_%": psutil.virtual_memory().percent,
           "gpu_mem_gb": torch.cuda.memory_allocated() / 1024**3
       })
   ```

### Medium-Term (Next Sprint)
1. Check if **RAPIDS/cuDF** is available on your cluster:
   ```bash
   module avail rapids
   ```
   If yes, replace pandas in preprocessing.

2. Implement **NeighborLoader** for larger datasets:
   ```python
   from torch_geometric.loader import NeighborLoader
   ```

3. Use **Dask** for parallel preprocessing across CSV files.

### Long-Term (Scaling Goal)
1. **Ray Train** for multi-node distributed training
2. **DeepSpeed ZeRO** for optimizer memory efficiency
3. **Memory-mapped graph storage** for truly large datasets:
   ```python
   # PyTorch 2.1+ memory mapping
   graphs = torch.load('graphs.pt', mmap=True)
   ```

---

## Conclusion

For your specific use case:
- **No single library solves everything** - use a layered approach
- **cuDF is the most impactful** if available (massive preprocessing speedup + zero-copy to PyTorch)
- **Dask is the safest bet** for SLURM integration in preprocessing
- **PyTorch native tools** (checkpointing + AMP) give immediate relief with zero integration cost
- **Ray/DeepSpeed** are the path forward for true multi-node scaling

The graph data sizing problem is best addressed by:
1. Streaming preprocessing (Dask/Ray Data)
2. Graph sampling during training (PyG NeighborLoader)
3. Automatic batch size tuning with OOM recovery (already partially implemented)
