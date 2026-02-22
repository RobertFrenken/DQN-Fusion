# GPU Training Throughput Optimization — RAPIDS Phase 2 + OSC Best Practices

## Context

The KD-GAT pipeline currently trains on V100 GPUs at OSC but has several throughput bottlenecks:
1. **Data I/O**: Graph caches (.pt files, up to 1.4GB) load from NFS home directory — the slowest filesystem at OSC
2. **GATConv**: Standard PyG GATConv doesn't fuse GPU kernels and ignores the 11-D edge features entirely
3. **No profiling**: No operator-level visibility into where training time is spent
4. **Batch sizing**: Heuristic-based estimation; no trial-based validation against actual GPU memory
5. **SLURM config**: Under-provisioned CPU/memory for DataLoader workers

RAPIDS Phase 1 (cudf.pandas + cuML) is already complete. This plan extends GPU acceleration to the training loop itself and applies OSC-specific I/O and scheduling best practices.

All 6 optimizations compose together — none are alternatives.

---

## P0: Data Staging to Scratch + $TMPDIR

**Goal**: Move cache reads from NFS (~slow) → scratch GPFS (~fast) → $TMPDIR local disk (~fastest).

### Files to modify

| File | Change |
|------|--------|
| `config/paths.py` | Add `KD_GAT_CACHE_ROOT` env var override in `cache_dir()` |
| `scripts/stage_data.sh` | **New**: reusable data staging script |
| `scripts/preprocess_gpu_slurm.sh` | Source `stage_data.sh` |

### Implementation

**`config/paths.py:114-116`** — Add env var override to `cache_dir()`:
```python
def cache_dir(cfg: PipelineConfig) -> Path:
    """Processed-graph cache directory."""
    override = os.environ.get("KD_GAT_CACHE_ROOT")
    if override:
        return Path(override) / cfg.dataset
    return Path("data") / "cache" / cfg.dataset
```
Add `import os` at top of file.

**`scripts/stage_data.sh`** (new file):
```bash
#!/usr/bin/env bash
# Source this from SLURM scripts to stage cache files to $TMPDIR.
# Usage: source scripts/stage_data.sh <dataset> [<dataset2> ...]
SCRATCH_CACHE="/fs/scratch/PAS1266/kd-gat/cache"
LOCAL_CACHE="$TMPDIR/kd-gat-cache"

for ds in "$@"; do
    mkdir -p "$LOCAL_CACHE/$ds"
    cp "$SCRATCH_CACHE/$ds/"*.pt "$LOCAL_CACHE/$ds/" 2>/dev/null || true
    cp "$SCRATCH_CACHE/$ds/"*.pkl "$LOCAL_CACHE/$ds/" 2>/dev/null || true
    cp "$SCRATCH_CACHE/$ds/"*.json "$LOCAL_CACHE/$ds/" 2>/dev/null || true
    echo "Staged $ds to $LOCAL_CACHE/$ds ($(du -sh "$LOCAL_CACHE/$ds" | cut -f1))"
done
export KD_GAT_CACHE_ROOT="$LOCAL_CACHE"
```

**One-time setup**: Copy cache from NFS to scratch:
```bash
mkdir -p /fs/scratch/PAS1266/kd-gat/cache
cp -r data/cache/* /fs/scratch/PAS1266/kd-gat/cache/
```

---

## P1: SLURM Optimization

**Goal**: Better resource allocation for GPU training jobs.

### Files to modify

| File | Change |
|------|--------|
| `scripts/preprocess_gpu_slurm.sh` | Update SBATCH directives |
| `scripts/job_epilog.sh` | **New**: reusable post-job diagnostics (shared with P5) |

### Changes to SBATCH directives

```bash
#SBATCH --cpus-per-task=8       # was 4; match cfg.num_workers=8
#SBATCH --mem=85G               # was 32G; room for DataLoader prefetch
#SBATCH --gres=gpu:v100:1       # was gpu:1; explicit GPU type
```

### New training SLURM template

Create `scripts/train_gpu.sh` as a general-purpose GPU training template that:
- Sources `stage_data.sh` for data staging
- Loads `cuda/12.4.1` module
- Activates `gnn-rapids` env
- Runs any pipeline CLI command passed as args
- Sources `job_epilog.sh` at end

---

## P2: PyTorch Profiler Integration

**Goal**: Operator-level profiling to identify bottlenecks before and after optimizations.

### Files to modify

| File | Change |
|------|--------|
| `pipeline/stages/utils.py` | Add `ProfilerCallback` class |
| `pipeline/stages/utils.py` | Wire into `make_trainer()` when `cfg.training.profile` is True |
| `pipeline/memory.py` | Add `memory_snapshot()` context manager |
| `config/schema.py` | Add `profile: bool = False` and `profile_steps: int = 5` to `TrainingConfig` |

### ProfilerCallback (in `pipeline/stages/utils.py`)

New Lightning Callback alongside existing `MemoryMonitorCallback`:

```python
class ProfilerCallback(pl.Callback):
    """Record PyTorch profiler traces for first N steps."""

    def __init__(self, output_dir: Path, wait=1, warmup=1, active=3):
        self.output_dir = output_dir
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self._profiler = None

    def on_train_start(self, trainer, pl_module):
        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=self.wait, warmup=self.warmup,
                active=self.active, repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.output_dir / "profiler_traces")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self._profiler.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._profiler:
            self._profiler.step()

    def on_train_end(self, trainer, pl_module):
        if self._profiler:
            self._profiler.__exit__(None, None, None)
```

### `make_trainer()` modification

Add `ProfilerCallback` to callbacks list when `cfg.training.profile` is True:
```python
if t.profile:
    callbacks.append(ProfilerCallback(
        output_dir=out,
        active=t.profile_steps,
    ))
```

### Memory snapshot (`pipeline/memory.py`)

```python
from contextlib import contextmanager

@contextmanager
def memory_snapshot(output_path: Path):
    """Record GPU memory allocation history for visualization."""
    torch.cuda.memory._record_memory_history(max_entries=100000)
    try:
        yield
    finally:
        torch.cuda.memory._dump_snapshot(str(output_path))
        torch.cuda.memory._record_memory_history(enabled=None)
```

---

## P3: Trial-Based Batch Size Auto-Tuning

**Goal**: Replace heuristic batch sizing with actual forward+backward trial runs.

### Files to modify

| File | Change |
|------|--------|
| `pipeline/memory.py` | Add `_trial_based_batch_size()`, integrate into `compute_batch_size()` |
| `config/schema.py` | Document `"trial"` as valid `memory_estimation` value |

### Implementation in `pipeline/memory.py`

Add new function and wire into existing `compute_batch_size()`:

```python
def _trial_based_batch_size(
    model: nn.Module,
    sample_graph: "Data",
    device: torch.device,
    target_utilization: float = 0.85,
    min_batch_size: int = 8,
    max_batch_size: int = 8192,
) -> tuple[int, List[str]]:
    """Binary search for max batch size via actual forward+backward passes."""
    from torch_geometric.loader import DataLoader

    warnings = []
    dummy_data = [sample_graph.clone() for _ in range(max_batch_size)]

    lo, hi = min_batch_size, max_batch_size
    best = min_batch_size

    while lo <= hi:
        mid = (lo + hi) // 2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            loader = DataLoader(dummy_data[:mid], batch_size=mid)
            batch = next(iter(loader)).to(device)
            model.train()
            out = model(batch)
            loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
            loss.backward()
            model.zero_grad()
            best = mid
            lo = mid + 1
            del batch, out, loss
        except RuntimeError as e:
            if "out of memory" in str(e):
                hi = mid - 1
                torch.cuda.empty_cache()
            else:
                raise

    # Apply safety margin
    safe = max(min_batch_size, int(best * target_utilization))
    log.info("Trial batch size: max_fit=%d, safe=%d", best, safe)
    return safe, warnings
```

In `compute_batch_size()`, add an `elif mode == "trial"` branch that calls `_trial_based_batch_size()` and populates the `MemoryBudget` with measured values.

---

## P4: CuGraphGATConv Integration (with edge_attr)

**Goal**: Replace `GATConv` with fused GPU kernel `CuGraphGATConv`, and wire up the 11-D edge features that are currently ignored.

### Files to modify

| File | Change |
|------|--------|
| `config/constants.py` | Add `CUGRAPH_AVAILABLE` flag |
| `config/schema.py` | Add `use_cugraph: bool = False` to `GATArchitecture` and `VGAEArchitecture` |
| `src/models/gat.py` | Conditional CuGraphGATConv + edge_attr wiring |
| `src/models/vgae.py` | Same pattern for encoder/decoder |
| `scripts/setup_rapids_env.sh` | Add `cugraph` to conda install |

### `config/constants.py` — Add flag

```python
CUGRAPH_AVAILABLE = False
try:
    from torch_geometric.nn.conv import CuGraphGATConv  # noqa: F401
    CUGRAPH_AVAILABLE = True
except ImportError:
    pass
```

### `config/schema.py` — Add config field

```python
class GATArchitecture(BaseModel, frozen=True):
    # ... existing fields ...
    use_cugraph: bool = False  # Use CuGraphGATConv for fused GPU kernels
```

Same for `VGAEArchitecture`.

### `src/models/gat.py` — Core changes

**Import block** (top of file):
```python
from torch_geometric.nn import GATConv, JumpingKnowledge, global_mean_pool
from config.constants import CUGRAPH_AVAILABLE

if CUGRAPH_AVAILABLE:
    from torch_geometric.nn.conv import CuGraphGATConv
```

**`__init__`** — Add `use_cugraph` param:
```python
def __init__(self, ..., use_cugraph=False):
    self.use_cugraph = use_cugraph and CUGRAPH_AVAILABLE
    ConvClass = CuGraphGATConv if self.use_cugraph else GATConv

    for i in range(num_layers):
        in_dim = embedding_dim + (in_channels - 1) if i == 0 else hidden_channels * heads
        self.convs.append(ConvClass(in_dim, hidden_channels, heads=heads, concat=True))
```

**`from_config`** — Pass through:
```python
@classmethod
def from_config(cls, cfg, num_ids: int, in_ch: int) -> "GATWithJK":
    return cls(
        ...,
        use_cugraph=cfg.gat.use_cugraph,
    )
```

**`forward`** — Pass edge_attr when using CuGraphGATConv:
```python
for conv in self.convs:
    if return_attention_weights and not self.use_cugraph:
        x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
        x = x.relu()
        attention_weights.append(alpha.detach().cpu())
    elif self.use_cugraph:
        x = conv(x, edge_index, data.edge_attr).relu()
    elif self.use_checkpointing and x.requires_grad:
        x = checkpoint(lambda x_in, c=conv, ei=edge_index: c(x_in, ei).relu(), x, use_reentrant=False)
    else:
        x = conv(x, edge_index).relu()
```

**Note**: `return_attention_weights` is not supported by CuGraphGATConv. When `use_cugraph=True`, attention weight extraction (used only for dashboard visualization in `export.py`) is silently skipped. This is logged as a warning.

### `src/models/vgae.py` — Same pattern

Apply identical conditional swap in `GraphAutoencoderNeighborhood`'s encoder and decoder GATConv layers.

### `scripts/setup_rapids_env.sh`

Add `cugraph` and `pylibcugraphops` to the conda install line.

---

## P5: OSC Profiling Tools in SLURM Scripts

**Goal**: Post-job GPU utilization analysis for every training run.

### Files to modify

| File | Change |
|------|--------|
| `scripts/job_epilog.sh` | **New**: reusable post-job diagnostics |
| All SLURM scripts | Source `job_epilog.sh` at end |

### `scripts/job_epilog.sh`

```bash
#!/usr/bin/env bash
# Source at end of any GPU SLURM job for post-job diagnostics.
echo "=== Job ${SLURM_JOB_ID} complete ==="
echo "Peak GPU memory (PyTorch):"
python -c "import torch; print(f'  {torch.cuda.max_memory_allocated()/1e9:.2f} GB')" 2>/dev/null || echo "  N/A"
echo "GPU utilization snapshot:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || true
echo "Post-job analysis:"
echo "  get_gpu_usage -M pitzer ${SLURM_JOB_ID}"
```

---

## How These Optimizations Compose

All six are complementary:

```
P0 (Data staging)     -> Faster data load for ALL training
P1 (SLURM tuning)     -> Better resource utilization for ALL jobs
P2 (PyTorch Profiler)  -> Identifies bottlenecks, informs P3 and P4
P3 (Trial batch size)  -> Maximizes GPU memory utilization
P4 (CuGraphGATConv)    -> Faster per-step computation + edge_attr integration
P5 (OSC profiling)     -> Post-hoc validation that P0-P4 improved utilization
```

Recommended workflow:
1. Implement P0 + P1 + P5 first (all low effort, immediate benefit)
2. Implement P2, run a profiling job on `hcrl_sa` to establish baseline
3. Look at profiler output: if GATConv dominates, implement P4; if memory is underutilized, implement P3
4. Re-profile to measure improvement

---

## Verification

### P0 — Data staging
```bash
# One-time: copy cache to scratch
cp -r data/cache/* /fs/scratch/PAS1266/kd-gat/cache/
# Test: stage_data.sh sets KD_GAT_CACHE_ROOT correctly
source scripts/stage_data.sh hcrl_sa
echo $KD_GAT_CACHE_ROOT  # should point to $TMPDIR/kd-gat-cache
ls $KD_GAT_CACHE_ROOT/hcrl_sa/  # should have .pt files
# Test: Python resolves correct path
KD_GAT_CACHE_ROOT=$TMPDIR/kd-gat-cache python -c "
from config import resolve
cfg = resolve('vgae', 'large', dataset='hcrl_sa')
from config.paths import cache_dir
print(cache_dir(cfg))  # should use TMPDIR path
"
```

### P1 — SLURM optimization
- Submit a short training job with new directives, verify `squeue` shows correct resource allocation
- Check job runs with 8 CPU workers active (no fallback to num_workers=0)

### P2 — Profiler
```bash
python -m pipeline.cli autoencoder --model vgae --scale large --dataset hcrl_sa \
    -O training.profile true -O training.max_epochs 2
# Check for profiler_traces/ directory in the run output
ls experimentruns/hcrl_sa/vgae_large_autoencoder/profiler_traces/
```

### P3 — Trial batch size
```bash
python -m pipeline.cli autoencoder --model vgae --scale large --dataset hcrl_sa \
    -O training.memory_estimation trial -O training.max_epochs 1
# Check logs for "Trial batch size: max_fit=X, safe=Y"
```

### P4 — CuGraphGATConv
```bash
# In gnn-rapids env with cugraph-ops:
python -m pipeline.cli curriculum --model gat --scale large --dataset hcrl_sa \
    -O gat.use_cugraph true -O training.max_epochs 5
# Compare val_loss / F1 against baseline (use_cugraph=false) run
# Verify edge_attr is being consumed (check logs for tensor shapes)
```

### P5 — Job epilog
- Run any GPU SLURM job, check end-of-log for GPU utilization summary and `get_gpu_usage` command

### Full integration test
```bash
bash scripts/run_tests_slurm.sh  # All 108 existing tests must pass
```

---

## Sources

- [OSC File Systems](https://www.osc.edu/supercomputing/storage-environment-at-osc/available-file-systems) — $TMPDIR is fastest, scratch is high-performance
- [OSC GPU Memory Profiling](https://www.osc.edu/resources/getting_started/howto/howto_estimating_and_profiling_gpu_memory_usage_for_generative_ai) — get_gpu_usage, Grafana, PyTorch profiler
- [CuGraphGATConv PyG docs](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cugraph/gat_conv.html) — fused GPU kernel GAT
- [cuGraph-PyG GitHub](https://github.com/rapidsai/cugraph-gnn/blob/main/readme_pages/cugraph_pyg.md) — GraphStore/FeatureStore/Sampler
- [RAPIDS cuGraph-PyG intro](https://medium.com/rapids-ai/intro-to-graph-neural-networks-with-cugraph-pyg-6fe32c93a2d0) — integration overview
- [PyTorch batch size optimization](https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1/) — trial-based approach
- [OSC SLURM Directives](https://www.osc.edu/supercomputing/batch-processing-at-osc/slurm_directives_summary)
