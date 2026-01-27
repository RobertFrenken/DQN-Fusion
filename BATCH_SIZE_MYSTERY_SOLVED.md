# Batch Size Mystery - PARTIAL SOLUTION

## The Evidence

### Different Datasets Show Different Batch Counts

**hcrl_sa** (9,364 graphs):
- Batches per epoch: **1/1**
- Log: `Epoch 0: 100%|██████████| 1/1`
- Effective batch size: ~9,364 graphs/batch

**set_02** (203,496 graphs):
- Batches per epoch: **4/4**
- Log: `Epoch 0: 100%|██████████| 4/4`
- Effective batch size: ~50,874 graphs/batch

## Key Finding

The batch size tuner infrastructure EXISTS but has a critical bug preventing it from being applied (see [BATCH_SIZE_TUNER_INVESTIGATION.md](BATCH_SIZE_TUNER_INVESTIGATION.md)).

However, even WITHOUT the tuner fix, the actual batch counts vary by dataset:
- Small datasets (hcrl_sa): 1 batch
- Large datasets (set_02): 4 batches

This suggests PyG DataLoader may have internal batching logic based on dataset size, or there's dynamic batch sizing happening somewhere.

## Hypothesis

PyTorch Geometric DataLoader has internal logic that creates multiple batches for very large datasets, but this is NOT the same as proper batch size optimization. The batches are still enormous (50k+ graphs).

## Next Investigation Steps

1. Check set_01 (small dataset with 53 IDs) - should have 1 batch?
2. Check if PyG DataLoader has max_batch_size or similar parameter
3. Verify if the "4 batches" are from curriculum dataset growth over epochs
4. Check if num_workers=8 is creating 4 subprocesses (suspicious correlation)

## The Core Issue Remains

Even with 4 batches for set_02, each batch contains ~50k graphs. With proper tuning:
- Expected batch size: 512-2048 graphs per batch
- Expected batch count: 100-400 batches per epoch

The tuner fix is still CRITICAL.
