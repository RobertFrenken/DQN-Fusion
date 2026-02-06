# Plan: Batched Inference + Evaluation Fix

**Created**: 2026-02-05
**Revised**: 2026-02-05 — Dropped `data.py` after codebase audit (see Revision Notes)
**Status**: Revised, pending approval

## Problem

Evaluation runs take 30-60 minutes (timing out on larger datasets) because:
1. Single-graph Python loops for all inference (no DataLoader batching)
2. Test data rebuilt from CSVs every run (no caching)
3. Models loaded and run twice in evaluation (once for metrics, once for fusion)
4. 60-minute SLURM time limit too short
5. `eval_student_kd` rule fails: passes `--use-kd true` without `--teacher-path` (already fixed in validate.py, just needs re-run)

## Revision Notes

Original plan proposed two new files (`data.py` + `inference.py`). After auditing the codebase:

- `load_data()` is already a single 10-line function in `utils.py:67-77`, imported by all 3 stages. Moving it to `data.py` adds a file without reducing any duplication.
- The actual duplication is in inference logic: the same VGAE forward-pass pattern is written 3 times across `evaluation.py`, `training.py`, and `utils.py`.

**One new file (`inference.py`) instead of two. `load_data()` stays in `utils.py`.**

## Solution: One New File + Targeted Modifications

### NEW: `pipeline/stages/inference.py` (~180 lines)

Batched versions of the 4 duplicated inference patterns. All use existing `make_dataloader()` from utils.

```python
run_vgae_inference(vgae, data, device, cfg, batch_size=512)
    # -> (errors, labels) as numpy
    # Replaces: evaluation._run_vgae_inference (lines 236-248)

run_gat_inference(gat, data, device, cfg, batch_size=512)
    # -> (preds, labels, scores) as numpy
    # Replaces: evaluation._run_gat_inference (lines 222-233)

build_fusion_cache(vgae, gat, data, device, cfg, max_samples, batch_size=256)
    # -> {states: Tensor, labels: Tensor}
    # Replaces: utils.cache_predictions (lines 527-585)

score_difficulty(vgae, graphs, device, cfg, batch_size=512)
    # -> list[float]
    # Replaces: training._score_difficulty (lines 150-182)
```

Per-graph aggregation via `torch_geometric.utils.scatter`:
```python
per_node_err = (cont - batch.x[:, 1:]).pow(2).mean(dim=1)
per_graph_err = scatter(per_node_err, batch.batch, dim=0,
                        dim_size=n_graphs, reduce="mean")
```

### MODIFY: `pipeline/stages/evaluation.py`

- Import `run_gat_inference`, `run_vgae_inference`, `build_fusion_cache` from `inference.py`
- Add `.pt` caching to existing `_load_test_data()` — save/load from `data/cache/{dataset}/{scenario}.pt`
- Restructure `evaluate()`: load VGAE + GAT **once**, reuse for metrics + fusion
- Delete: `_run_gat_inference` (lines 222-233), `_run_vgae_inference` (lines 236-248)
- Keep: `_load_test_data()` (with caching added), `_run_fusion_inference()` (CPU-only, fast), `_vgae_threshold()`, `_compute_metrics()`, `_graph_label()`

### MODIFY: `pipeline/stages/training.py`

- Import `score_difficulty` from `inference.py`
- Delete `_score_difficulty()` (lines 150-182)

### MODIFY: `pipeline/stages/fusion.py`

- Import `build_fusion_cache` from `inference.py` (replaces `cache_predictions` from utils)

### MODIFY: `pipeline/stages/utils.py`

- Delete `cache_predictions()` (lines 527-585)
- Remove `global_mean_pool` import (line 20, moved to inference.py)

### MODIFY: `pipeline/Snakefile`

- Bump `_EVAL_RES` `time_min` from 60 to 120 (line 73)

## Implementation Order

1. Create `inference.py` (additive — no callers changed yet)
2. Switch `training.py`: import `score_difficulty`, delete `_score_difficulty`
3. Switch `fusion.py`: import `build_fusion_cache`, drop `cache_predictions`
4. Rewrite `evaluation.py`: use inference imports, add test data caching, load models once
5. Clean `utils.py`: delete `cache_predictions` + unused `global_mean_pool` import
6. Bump Snakefile eval time
7. Run tests

## Detailed Function Designs

### inference.py: run_vgae_inference (batched)

```python
def run_vgae_inference(vgae, data, device, cfg, batch_size=512):
    loader = make_dataloader(data, cfg, batch_size, shuffle=False)
    all_errors, all_labels = [], []
    vgae.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            cont, _, _, _, _ = vgae(batch.x, batch.edge_index, batch.batch)
            per_node_err = (cont - batch.x[:, 1:]).pow(2).mean(dim=1)
            n_graphs = int(batch.batch.max().item()) + 1
            per_graph_err = scatter(per_node_err, batch.batch, dim=0,
                                    dim_size=n_graphs, reduce="mean")
            all_errors.append(per_graph_err.cpu())
            all_labels.append(batch.y.cpu())
    return torch.cat(all_errors).numpy(), torch.cat(all_labels).numpy()
```

### inference.py: build_fusion_cache (batched, 15-D state vectors)

Per-graph features from both models, computed with scatter:
- `[0:3]`  VGAE errors: recon (scatter mean MSE), nbr (scatter mean BCE), canid (scatter mean CE)
- `[3:7]`  VGAE latent stats: z per-node reduce then scatter (mean/mean, std/mean, max/max, min/min)
- `[7]`    VGAE confidence: 1/(1+recon_err)
- `[8:10]` GAT logits: softmax on fc_layers output after global_mean_pool (already per-graph)
- `[10:14]` GAT embedding stats: stats on pooled vector (already per-graph)
- `[14]`   GAT confidence: 1 - entropy/log(2)

### evaluation.py: restructured evaluate()

```python
def evaluate(cfg):
    train_data, val_data, num_ids, in_ch = load_data(cfg)     # from utils.py (unchanged)
    test_scenarios = _load_test_data(cfg)                       # CACHED via .pt files

    # Load models ONCE
    vgae = load_vgae(...) if vgae_ckpt.exists() else None
    gat = load_gat(...)   if gat_ckpt.exists()  else None

    # GAT metrics (batched via inference.py)
    if gat: run_gat_inference(gat, val_data, device, cfg)

    # VGAE metrics (batched via inference.py)
    if vgae: run_vgae_inference(vgae, val_data, device, cfg)

    # Fusion metrics (models already loaded, no redundant reload)
    if fusion_ckpt.exists() and vgae and gat:
        build_fusion_cache(vgae, gat, val_data, device, cfg, ...)

    del vgae, gat; cleanup()
```

## Numerical Equivalence

- VGAE MSE: `mean(per_node_means)` = `scatter_mean` over `batch.batch`
- GAT: PyG `Batch` handles graph assignment natively
- Latent stats: `.nan_to_num(0.0)` for single-node graphs in `z.std(dim=1)`
- `build_fusion_cache` state vector: same 15-D layout as old `cache_predictions`

## Edge Cases

- `[data[i] for i in range(n)]` instead of `data[:n]` (GraphDataset doesn't support slicing)
- `batch.to(device)`: safe (creates new Batch, doesn't mutate originals)
- Empty test scenarios: skip gracefully (same as current)
- mmap worker limits: handled by existing `_safe_num_workers` in utils.py

## Import Chain (no circular deps)

```
inference.py  -> stages.utils (make_dataloader only)
evaluation.py -> stages.inference, stages.utils (model loaders, load_data)
training.py   -> stages.inference, stages.utils (load_data, model loaders)
fusion.py     -> stages.inference, stages.utils (load_data, model loaders)
modules.py    -> stages.utils (unchanged)
```

## Expected Performance

| Operation | Old (set_02) | New | Source |
|-----------|-------------|-----|--------|
| Load test data | ~50 min | ~5 sec | Cache .pt files |
| VGAE inference (200K) | ~40 min | ~3 min | Batched DataLoader |
| GAT inference (200K) | ~30 min | ~2 min | Batched DataLoader |
| Fusion cache (150K) | ~25 min | ~2 min | Batched DataLoader |
| Difficulty scoring | ~10 min | ~30 sec | Batched DataLoader |

## Verification

1. `python -m pytest tests/test_pipeline_integration.py -v`
2. Run evaluation on hcrl_sa, compare metrics.json to existing output
3. Spot-check: old single-graph vs new batched on 100 graphs, `np.allclose(atol=1e-5)`

## Files Summary

| File | Action | Change |
|------|--------|--------|
| `pipeline/stages/inference.py` | CREATE | ~180 lines |
| `pipeline/stages/evaluation.py` | MODIFY | use inference imports, add test caching, load models once |
| `pipeline/stages/training.py` | MODIFY | ~3 lines changed, 33 deleted |
| `pipeline/stages/fusion.py` | MODIFY | ~2 lines changed |
| `pipeline/stages/utils.py` | MODIFY | ~60 lines deleted |
| `pipeline/Snakefile` | MODIFY | 1 line changed |
