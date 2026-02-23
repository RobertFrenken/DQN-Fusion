# GNN Research Infrastructure: Ray Migration & Batch Size Investigation

## Context

PhD student at Ohio State, CAR Mobility Systems Lab. Building graph neural networks (GAT, VGAE, DQN fusion ensemble) for CAN bus intrusion detection, targeting ICML 2026. Current infra: Prefect for orchestration, W&B for tracking, SLURM on Ohio Supercomputer Center (OSC). PyTorch Geometric for GNN implementation.

This document covers two workstreams:

1. Migrating from Prefect to Ray as the unified orchestration/compute layer
2. Investigating dynamic graph batching to replace the current 95th-percentile static batch size estimation

---

## Part 1: Ray Migration

### Why Ray Over Prefect

Prefect is a workflow DAG orchestrator (scheduling, retries, observability). Ray is a distributed compute framework that also orchestrates. For ML training workloads on SLURM, Ray is the correct primitive because:

- Native SLURM integration via `ray symmetric-run` (Ray 2.49+)
- GPU resource management per task/actor
- Ray Tune for distributed HPO with early stopping
- Fault tolerance for GPU training failures
- Zero code change to go from single-GPU to multi-node

Prefect adds a second system with no benefit for experiment execution. The `prefect-ray` integration is a thin wrapper that submits Prefect tasks to Ray — it's not deep integration.

### Target Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Cluster bootstrap | SLURM sbatch + `ray symmetric-run` | Start Ray cluster on OSC |
| Orchestration | Ray Core + Ray Train | Distributed training, fault tolerance |
| HPO | Ray Tune + OptunaSearch + ASHAScheduler | Smart search + parallelism + early stopping |
| GNN Training | PyG + cuGraph-PyG | Drop-in GPU acceleration for GAT/VGAE |
| Experiment Tracking | W&B | Ray Tune has native W&B callback |
| Data Preprocessing | cuDF (optional) | GPU-accelerated CAN bus CSV processing |

### Task 1: SLURM Bootstrap Script for OSC

Create a SLURM sbatch script that bootstraps a Ray cluster on OSC. This should work for both single-node (1 GPU) and multi-node configurations.

```bash
#!/bin/bash
#SBATCH --job-name=gnn-can-ids
#SBATCH --nodes=1              # Start with 1, scale later
#SBATCH --gpus-per-node=1      # Adjust per OSC partition
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=04:00:00
#SBATCH --account=<PROJECT_ACCOUNT>

# --- Module setup (adjust to OSC's module system) ---
module load cuda/12.x
module load miniconda3

conda activate gnn-research

# --- Ray cluster bootstrap ---
set -x

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=6379
ip_head=$head_node:$port
export ip_head

echo "Head node: $head_node"
echo "IP Head: $ip_head"

# For single node, ray init handles everything
# For multi-node, use symmetric-run:
if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
    srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
        ray symmetric-run \
        --address "$ip_head" \
        --min-nodes "$SLURM_JOB_NUM_NODES" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" \
        --num-gpus="${SLURM_GPUS_PER_TASK:-1}" \
        -- \
        python -u train_ensemble.py
else
    python -u train_ensemble.py
fi
```

**Implementation notes for Claude Code:**

- Check OSC's available modules with `module avail cuda` and `module avail conda`
- OSC may use `module load python` instead of miniconda — verify
- The script should be parameterized so `--nodes`, `--gpus-per-node`, and the python entrypoint can be easily swapped
- Ensure Ray version >= 2.49 for `symmetric-run` support

### Task 2: Ray Tune HPO Configuration

Replace any existing Prefect-based experiment sweeps with Ray Tune. Use Optuna as the search backend and ASHA for early stopping.

```python
"""
HPO configuration for CAN bus GNN ensemble.
Uses Ray Tune + Optuna search + ASHA scheduler.
Logs to W&B.
"""
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.air.config import RunConfig, ScalingConfig

# --- Search space for the ensemble ---
search_space = {
    # GAT hyperparameters
    "gat_heads": tune.choice([2, 4, 8]),
    "gat_hidden_dim": tune.choice([64, 128, 256]),
    "gat_layers": tune.choice([2, 3, 4]),
    "gat_dropout": tune.uniform(0.1, 0.5),

    # VGAE hyperparameters
    "vgae_latent_dim": tune.choice([16, 32, 64]),
    "vgae_hidden_dim": tune.choice([64, 128, 256]),

    # DQN fusion hyperparameters
    "dqn_lr": tune.loguniform(1e-5, 1e-2),
    "dqn_gamma": tune.uniform(0.9, 0.999),
    "dqn_hidden_dim": tune.choice([64, 128, 256]),
    "dqn_buffer_size": tune.choice([10000, 50000, 100000]),

    # Shared
    "lr": tune.loguniform(1e-5, 1e-2),
    "weight_decay": tune.loguniform(1e-6, 1e-3),
    "node_budget": tune.choice([2000, 4000, 8000, 16000]),  # dynamic batching
}

# --- Scheduler: ASHA for early stopping ---
scheduler = tune.schedulers.ASHAScheduler(
    max_t=100,          # max epochs
    grace_period=10,    # minimum epochs before pruning
    reduction_factor=3,
)

# --- Search algorithm: Optuna TPE ---
search_alg = OptunaSearch(
    metric="val_f1",
    mode="max",
)

# --- Tune config ---
tune_config = tune.TuneConfig(
    search_alg=search_alg,
    scheduler=scheduler,
    num_samples=200,       # total trials
    max_concurrent_trials=4,  # adjust to available GPUs
)

# --- Run config with W&B logging ---
run_config = RunConfig(
    name="can-ids-ensemble-hpo",
    callbacks=[
        WandbLoggerCallback(
            project="can-ids-gnn",
            log_config=True,
        )
    ],
    storage_path="/fs/scratch/<PROJECT>/ray_results",  # OSC scratch
)

# --- Training function signature ---
def train_ensemble(config):
    """
    This function should:
    1. Build GAT, VGAE, and DQN models from config
    2. Use NodeBudgetBatchSampler with config["node_budget"]
    3. Train for one epoch
    4. Report metrics via ray.train.report({"val_f1": ..., "val_loss": ...})
    5. Loop until stopped by scheduler

    Ray Tune calls this function once per trial.
    For single-GPU: each trial gets 1 GPU automatically.
    """
    import ray.train
    # ... model construction and training loop ...
    # ray.train.report({"val_f1": f1, "val_loss": loss})
    pass

# --- Launch ---
tuner = tune.Tuner(
    tune.with_resources(train_ensemble, {"gpu": 1, "cpu": 4}),
    tune_config=tune_config,
    run_config=run_config,
    param_space=search_space,
)

results = tuner.fit()
best = results.get_best_result("val_f1", "max")
print(f"Best config: {best.config}")
print(f"Best F1: {best.metrics['val_f1']}")
```

**Implementation notes for Claude Code:**

- `storage_path` must point to a shared filesystem on OSC (scratch or project space), not local `/tmp`
- Install: `pip install "ray[tune]" optuna wandb`
- For single-GPU development, this works as-is — Ray just runs trials sequentially
- `max_concurrent_trials` should match available GPUs; on a single GPU set to 1
- The `train_ensemble` function needs to be implemented to match the existing model code
- Trials that OOM (e.g., from large `node_budget`) fail fast and Tune moves on — this is a feature

### Task 3: cuGraph-PyG Drop-in Replacements

Where the existing code uses PyG's built-in convolution layers, swap to cuGraph-accelerated versions. These are API-compatible drop-in replacements.

```python
# --- Before ---
from torch_geometric.nn import GATConv

# --- After ---
from cugraph_pyg.nn import CuGraphGATConv as GATConv
```

Also swap the data loader for GPU-accelerated sampling:

```python
# --- Before ---
from torch_geometric.loader import NeighborLoader

# --- After ---
from cugraph_pyg.loader import CuGraphNeighborLoader as NeighborLoader
```

**Implementation notes for Claude Code:**

- Install: `pip install cugraph-cu12 cugraph-pyg-cu12` (match CUDA version on OSC)
- cuGraph requires NVIDIA GPU with compute capability >= 7.0 (V100+)
- If OSC's CUDA version doesn't match available cugraph wheels, use NVIDIA's NGC PyG container instead
- For VGAE: there's no `CuGraphVGAE` — the encoder uses standard GATConv/GCNConv which CAN be swapped, but the reparameterization trick and decoder stay as-is
- Test: verify output shapes and loss values match between PyG native and cuGraph versions on a small sample before full training runs

### Task 4: Remove Prefect Dependencies

Once Ray is working:

1. Remove `prefect`, `prefect-ray` from requirements
2. Delete any `@flow` / `@task` decorated functions
3. Replace Prefect's block-based configuration with Ray's `RunConfig` or plain YAML/dataclass configs
4. If there are scheduled data ingestion flows (cron-style), either convert to simple cron + Python script or keep Prefect solely for that narrow use case

### Prefect Removal Checklist

Once Ray migration is validated (Tasks 1–3 working), execute this cleanup:

**Files to delete:**

- `pipeline/flows/train_flow.py`
- `pipeline/flows/eval_flow.py`
- `pipeline/flows/slurm_config.py`
- `pipeline/flows/` directory (if empty after above)

**Dependencies to remove from `pyproject.toml`:**

- `prefect`
- `prefect-dask`
- `dask-jobqueue`

Then `uv sync` to clean the lockfile.

**CLAUDE.md references to update:**

- Key Commands section: remove `python -m pipeline.cli flow` examples, replace with Ray-based launch
- Architecture Decisions → Orchestration bullet: replace Prefect/Dask description with Ray Core + `ray symmetric-run`
- Skills table: update `/run-pipeline` skill to use Ray instead of Prefect
- Session Modes: remove "Prefect" from `mlops` mode description

**What replaces each Prefect capability:**

| Prefect Feature | Ray Replacement |
|----------------|-----------------|
| `@flow` / `@task` decorators | `@ray.remote` tasks + DAG via `ObjectRef` dependencies |
| Retry / failure handling | Ray fault tolerance (task-level `max_retries`, actor restart) |
| SLURM dispatch via `dask-jobqueue` | `ray symmetric-run` + SLURM `sbatch` bootstrap |
| Flow composition (`_dataset_pipeline` sub-flows) | Ray DAG — chain `ObjectRef` outputs as inputs |
| `--local` flag (local Dask cluster) | `ray.init()` with no args (local mode) |
| Prefect UI / run history | W&B dashboard (already in use) + Ray Dashboard (`ray dashboard`) |

---

## Part 2: Graph Batch Size Investigation

### Problem Statement

CAN bus graph samples vary in size (number of nodes and edges) depending on:

- Time window length
- Attack type and injection pattern
- ECU communication topology during that window

Current approach: estimate batch size based on the 95th percentile graph sample size. This is conservative — wastes GPU memory on most batches and risks OOM on the tail.

### Task 5: Profile the Dataset's Graph Size Distribution

Before implementing dynamic batching, we need to understand the actual distribution. Create a profiling script.

```python
"""
Profile CAN bus graph dataset to understand size distribution.
Outputs statistics and histogram data for batch size decisions.
"""
import numpy as np
import torch
from torch_geometric.data import Dataset
from collections import defaultdict

def profile_graph_dataset(dataset: Dataset, output_path: str = "graph_profile.json"):
    """
    Compute per-graph statistics across the entire dataset.

    Args:
        dataset: PyG dataset of CAN bus graphs
        output_path: where to save the profile JSON

    Collects:
        - num_nodes per graph
        - num_edges per graph
        - feature tensor bytes per graph
        - total memory footprint estimate per graph
    """
    stats = defaultdict(list)

    for i, data in enumerate(dataset):
        n_nodes = data.num_nodes
        n_edges = data.num_edges

        # Estimate memory footprint in bytes
        node_feat_bytes = data.x.nelement() * data.x.element_size() if data.x is not None else 0
        edge_index_bytes = data.edge_index.nelement() * data.edge_index.element_size()
        edge_feat_bytes = data.edge_attr.nelement() * data.edge_attr.element_size() if data.edge_attr is not None else 0

        total_bytes = node_feat_bytes + edge_index_bytes + edge_feat_bytes

        stats["num_nodes"].append(n_nodes)
        stats["num_edges"].append(n_edges)
        stats["node_feat_bytes"].append(node_feat_bytes)
        stats["edge_index_bytes"].append(edge_index_bytes)
        stats["edge_feat_bytes"].append(edge_feat_bytes)
        stats["total_bytes"].append(total_bytes)

    # Compute summary statistics
    summary = {}
    for key, values in stats.items():
        arr = np.array(values)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "count": len(arr),
        }

    # Key ratios
    nodes = np.array(stats["num_nodes"])
    edges = np.array(stats["num_edges"])
    summary["edge_to_node_ratio"] = {
        "mean": float((edges / np.maximum(nodes, 1)).mean()),
        "std": float((edges / np.maximum(nodes, 1)).std()),
    }
    summary["coefficient_of_variation_nodes"] = float(nodes.std() / max(nodes.mean(), 1e-8))
    summary["coefficient_of_variation_edges"] = float(edges.std() / max(edges.mean(), 1e-8))

    import json
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print key findings
    print(f"Dataset: {len(dataset)} graphs")
    print(f"Nodes — mean: {summary['num_nodes']['mean']:.0f}, "
          f"std: {summary['num_nodes']['std']:.0f}, "
          f"p95: {summary['num_nodes']['p95']:.0f}, "
          f"max: {summary['num_nodes']['max']:.0f}")
    print(f"Edges — mean: {summary['num_edges']['mean']:.0f}, "
          f"std: {summary['num_edges']['std']:.0f}, "
          f"p95: {summary['num_edges']['p95']:.0f}, "
          f"max: {summary['num_edges']['max']:.0f}")
    print(f"CV(nodes): {summary['coefficient_of_variation_nodes']:.3f}")
    print(f"CV(edges): {summary['coefficient_of_variation_edges']:.3f}")
    print(f"Edge/Node ratio: {summary['edge_to_node_ratio']['mean']:.2f} ± {summary['edge_to_node_ratio']['std']:.2f}")

    # Decision guidance
    cv_nodes = summary["coefficient_of_variation_nodes"]
    cv_edges = summary["coefficient_of_variation_edges"]
    if cv_nodes < 0.3 and cv_edges < 0.3:
        print("\n→ Low variance: fixed batch size is probably fine. "
              "Use p95-based estimation.")
    elif cv_edges > cv_nodes:
        print("\n→ Edge count varies more than node count. "
              "Use EDGE budget for dynamic batching.")
    else:
        print("\n→ Node count has high variance. "
              "Use NODE budget for dynamic batching.")

    return summary
```

**Implementation notes for Claude Code:**

- Run this first before implementing dynamic batching — the results determine which strategy to use
- If CV (coefficient of variation) is low (< 0.3), the current 95th percentile approach may be adequate and dynamic batching is unnecessary complexity
- If CV is high, dynamic batching is worth implementing
- Log the profile JSON to W&B as an artifact for reproducibility
- The profile should be run on ALL dataset splits (train, val, test) separately since attack distributions may differ

### Task 6: Dynamic Batch Sampler (Node Budget)

If profiling shows high variance in node counts, implement a node-budget-based batch sampler.

```python
"""
Dynamic batch sampler that packs graphs up to a node (or edge) budget.
Keeps GPU memory utilization consistent regardless of graph size distribution.
"""
import random
from typing import Iterator, List, Optional
from torch.utils.data import Sampler
from torch_geometric.data import Dataset


class BudgetBatchSampler(Sampler[List[int]]):
    """
    Packs graphs into batches such that the total node count
    (or edge count) per batch does not exceed a budget.

    Args:
        dataset: PyG dataset
        budget: maximum total nodes (or edges) per batch
        budget_key: "nodes" or "edges" — which dimension to budget on
        shuffle: whether to shuffle indices each epoch
        drop_last: drop the final incomplete batch
        min_batch_size: minimum graphs per batch (prevents degenerate batches)
    """

    def __init__(
        self,
        dataset: Dataset,
        budget: int,
        budget_key: str = "nodes",
        shuffle: bool = True,
        drop_last: bool = False,
        min_batch_size: int = 1,
    ):
        self.dataset = dataset
        self.budget = budget
        self.budget_key = budget_key
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.min_batch_size = min_batch_size

        # Pre-compute sizes to avoid repeated dataset access
        self.sizes = []
        for data in dataset:
            if budget_key == "nodes":
                self.sizes.append(data.num_nodes)
            elif budget_key == "edges":
                self.sizes.append(data.num_edges)
            else:
                raise ValueError(f"Unknown budget_key: {budget_key}")

        # Warn about graphs that exceed budget individually
        oversized = sum(1 for s in self.sizes if s > budget)
        if oversized > 0:
            print(f"WARNING: {oversized} graphs exceed budget of {budget} {budget_key}. "
                  f"These will be placed in singleton batches.")

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.sizes)))
        if self.shuffle:
            random.shuffle(indices)

        batch: List[int] = []
        current_cost = 0

        for idx in indices:
            size = self.sizes[idx]

            # If adding this graph exceeds budget AND we already have a batch, yield it
            if current_cost + size > self.budget and len(batch) >= self.min_batch_size:
                yield batch
                batch = []
                current_cost = 0

            batch.append(idx)
            current_cost += size

        # Handle last batch
        if batch and (not self.drop_last or len(batch) >= self.min_batch_size):
            yield batch

    def __len__(self) -> int:
        # Approximate — actual count depends on shuffle order
        total = sum(self.sizes)
        return max(1, total // self.budget)


# --- Usage with PyG DataLoader ---
def create_dynamic_dataloader(dataset, budget, budget_key="nodes", **loader_kwargs):
    """
    Create a PyG DataLoader with dynamic batching.

    Args:
        dataset: PyG dataset
        budget: node or edge budget per batch
        budget_key: "nodes" or "edges"
        **loader_kwargs: additional DataLoader args (num_workers, pin_memory, etc.)
    """
    from torch_geometric.loader import DataLoader

    sampler = BudgetBatchSampler(
        dataset=dataset,
        budget=budget,
        budget_key=budget_key,
        shuffle=loader_kwargs.pop("shuffle", True),
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        **loader_kwargs,
    )
```

**Implementation notes for Claude Code:**

- The `budget_key` choice (nodes vs edges) should be informed by Task 5's profiling output
- For CAN bus graphs: if node count is near-constant (same ECUs) but edge count varies (message volume), use `budget_key="edges"`
- The `budget` value can be found empirically via Task 7, or included as a hyperparameter in Ray Tune's search space (`node_budget` in the HPO config above)
- `min_batch_size=1` allows oversized graphs through as singletons — this is correct behavior, don't silently skip them
- This replaces the current 95th-percentile batch size function entirely

### Task 7: GPU Memory Budget Finder

Binary search for the maximum budget that fits in GPU memory including forward + backward pass.

```python
"""
Find the maximum node/edge budget that fits in GPU memory for a given model.
Uses binary search with actual forward+backward passes.
"""
import torch
import gc
from torch_geometric.data import Data, Batch


def find_max_budget(
    model: torch.nn.Module,
    sample_graph_fn,
    device: torch.device,
    budget_key: str = "nodes",
    low: int = 100,
    high: int = 100000,
    safety_factor: float = 0.85,
    num_warmup: int = 3,
) -> int:
    """
    Binary search for the maximum node/edge budget that fits in GPU memory.

    Args:
        model: the GNN model (already on device)
        sample_graph_fn: callable(budget) -> Batch
            Given a target budget, returns a PyG Batch with approximately
            that many total nodes (or edges). Should generate realistic
            feature dimensions and edge structure.
        device: CUDA device
        budget_key: "nodes" or "edges" — what the budget measures
        low: minimum budget to test
        high: maximum budget to test
        safety_factor: multiply final result by this (0.85 = 15% headroom)
        num_warmup: warmup forward passes before profiling

    Returns:
        Maximum safe budget (int)
    """
    model.train()

    # Warmup to stabilize CUDA memory allocator
    for _ in range(num_warmup):
        try:
            small_batch = sample_graph_fn(low).to(device)
            out = model(small_batch)
            if hasattr(out, "backward"):
                out.sum().backward() if out.dim() > 0 else out.backward()
            model.zero_grad(set_to_none=True)
            del small_batch, out
            torch.cuda.empty_cache()
        except Exception:
            pass

    best = low

    while low <= high:
        mid = (low + high) // 2
        torch.cuda.empty_cache()
        gc.collect()

        try:
            batch = sample_graph_fn(mid).to(device)
            out = model(batch)

            # Backward pass is where peak memory typically occurs
            loss = out.sum() if out.dim() > 0 else out
            loss.backward()

            # If we got here, it fits
            best = mid
            low = mid + 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = mid - 1
            else:
                raise  # Re-raise non-OOM errors

        finally:
            model.zero_grad(set_to_none=True)
            # Aggressively free memory
            for var_name in ["batch", "out", "loss"]:
                if var_name in locals():
                    del locals()[var_name]
            torch.cuda.empty_cache()
            gc.collect()

    safe_budget = int(best * safety_factor)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    total_mb = torch.cuda.get_device_properties(device).total_mem / 1024**2

    print(f"Max budget ({budget_key}): {best}")
    print(f"Safe budget (×{safety_factor}): {safe_budget}")
    print(f"Peak GPU memory: {peak_mb:.0f} MB / {total_mb:.0f} MB")

    return safe_budget


# --- Example sample_graph_fn for CAN bus data ---
def make_can_bus_batch(node_budget: int) -> Batch:
    """
    Generate a synthetic batch of CAN bus graphs with ~node_budget total nodes.
    Feature dimensions and edge density should match your actual data.

    IMPORTANT: Replace these values with your actual dataset characteristics
    from the profiling step (Task 5).
    """
    NODE_FEATURE_DIM = 16      # adjust to match your data
    EDGE_FEATURE_DIM = 8       # adjust to match your data (None if no edge features)
    AVG_NODES_PER_GRAPH = 50   # from profiling
    EDGE_TO_NODE_RATIO = 4.0   # from profiling

    num_graphs = max(1, node_budget // AVG_NODES_PER_GRAPH)
    graphs = []

    remaining_budget = node_budget
    for i in range(num_graphs):
        n = min(AVG_NODES_PER_GRAPH, remaining_budget)
        if n <= 0:
            break
        remaining_budget -= n

        num_edges = int(n * EDGE_TO_NODE_RATIO)
        edge_index = torch.randint(0, n, (2, num_edges))
        x = torch.randn(n, NODE_FEATURE_DIM)

        data = Data(x=x, edge_index=edge_index)
        if EDGE_FEATURE_DIM:
            data.edge_attr = torch.randn(num_edges, EDGE_FEATURE_DIM)

        graphs.append(data)

    return Batch.from_data_list(graphs)
```

**Implementation notes for Claude Code:**

- Run this ONCE at the start of a training campaign, not every run
- The `sample_graph_fn` should produce batches that are representative of your actual data — use profiling stats from Task 5 to set `AVG_NODES_PER_GRAPH`, `EDGE_TO_NODE_RATIO`, and feature dims
- The safety factor of 0.85 accounts for: optimizer state, activation caching, W&B overhead, CUDA fragmentation
- For the ensemble (GAT + VGAE + DQN), run this for the LARGEST model since they may share GPU during fusion training
- On OSC, GPU memory varies by partition (V100 16/32GB, A100 40/80GB) — run budget finder on the target partition
- Log the result to W&B config so it's reproducible

### Task 8: Integration — Putting It Together

The end-to-end workflow for a training run:

```python
"""
Example end-to-end training script using all components.
This is the entrypoint called from the SLURM script.
"""
import ray
from ray import tune

def train_ensemble(config):
    import torch
    import ray.train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset
    dataset = load_can_bus_dataset(config)  # your existing data loading

    # 2. Create dynamic dataloader
    train_loader = create_dynamic_dataloader(
        dataset["train"],
        budget=config["node_budget"],
        budget_key=config.get("budget_key", "nodes"),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = create_dynamic_dataloader(
        dataset["val"],
        budget=config["node_budget"],
        budget_key=config.get("budget_key", "nodes"),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Build models from config
    gat_model = build_gat(config)
    vgae_model = build_vgae(config)
    dqn_model = build_dqn_fusion(config)

    # 4. Training loop
    for epoch in range(config.get("max_epochs", 100)):
        train_loss = train_one_epoch(gat_model, vgae_model, dqn_model, train_loader, device)
        val_metrics = evaluate(gat_model, vgae_model, dqn_model, val_loader, device)

        # 5. Report to Ray Tune (triggers ASHA scheduling decisions)
        ray.train.report({
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "epoch": epoch,
        })


if __name__ == "__main__":
    # For single-GPU development, just run directly:
    ray.init()  # local mode

    # Quick test with fixed config
    test_config = {
        "gat_heads": 4,
        "gat_hidden_dim": 128,
        "gat_layers": 3,
        "gat_dropout": 0.3,
        "vgae_latent_dim": 32,
        "vgae_hidden_dim": 128,
        "dqn_lr": 1e-3,
        "dqn_gamma": 0.99,
        "dqn_hidden_dim": 128,
        "dqn_buffer_size": 50000,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "node_budget": 4000,
        "max_epochs": 5,
    }

    # Direct call for debugging — bypasses Tune
    train_ensemble(test_config)

    # When ready for HPO, switch to the Tune config from Task 2
```

---

## Execution Order

1. **Task 5** — Profile the dataset. This determines whether dynamic batching is needed and which budget key to use.
2. **Task 6** — Implement BudgetBatchSampler (only if profiling shows high CV).
3. **Task 7** — Run GPU memory budget finder on OSC target partition.
4. **Task 8** — Wire everything together in the training script.
5. **Task 1** — Set up SLURM bootstrap script for OSC.
6. **Task 2** — Configure Ray Tune HPO.
7. **Task 3** — Swap to cuGraph-PyG (benchmark before/after).
8. **Task 4** — Remove Prefect.

Tasks 1-4 (Ray migration) and Tasks 5-7 (batch size) can be done in parallel.

---

## Dependencies to Install

```bash
# Core
pip install "ray[tune]" optuna wandb torch torch_geometric

# GPU acceleration (match CUDA version)
pip install cugraph-cu12 cugraph-pyg-cu12

# Or if using OSC's module system, these may need conda:
# conda install -c rapidsai -c conda-forge -c nvidia cugraph cugraph-pyg cuda-version=12.x
```

## Key Decision Points

- **If profiling shows CV < 0.3**: Skip dynamic batching, keep 95th percentile approach, focus on Ray migration.
- **If edge CV >> node CV**: CAN bus topology is stable but message volume varies. Budget on edges.
- **If node CV >> edge CV**: Graph construction is creating variable-size subgraphs. Budget on nodes.
- **If cuGraph wheels don't match OSC's CUDA**: Use NVIDIA's NGC PyG container via Singularity/Apptainer instead of pip install.
