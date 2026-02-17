# Mode: Data Engineering

**Active mode**: Data Engineering — Dataset management, preprocessing, validation, and cache management.

## Focus Areas

- Dataset catalog management (`config/datasets.yaml`)
- Graph construction validation (temporal windows, node/edge features)
- Preprocessing cache health (`data/cache/`)
- Data quality checks (missing values, label distribution, attack type coverage)
- DVC data versioning and S3 remote sync

## Context Files (prioritize reading these)

- `config/datasets.yaml` — Dataset catalog (single source of truth for new datasets)
- `src/preprocessing/preprocessing.py` — Graph construction from CAN CSVs
- `config/constants.py` — Window sizes, feature counts, excluded attack types
- `src/training/datamodules.py` — Data loading + caching

## Available Commands

| Command | Description |
|---------|-------------|
| `dvc push -r s3` | Push data to S3 remote |
| `dvc pull` | Pull data from remote |
| `python -m pipeline.export` | Export dashboard data from filesystem |

## Dataset Catalog Schema

To add a new dataset, create an entry in `config/datasets.yaml`:
```yaml
new_dataset:
  path: data/automotive/new_dataset
  domain: automotive
  attack_types: [flooding, fuzzing, spoofing]
  num_classes: 4  # benign + attack types
  description: "Brief description"
```

## Validation Checklist (for new datasets)

1. Raw CSV structure: timestamp, CAN_ID, data bytes, label columns
2. Graph construction: `python -c "from src.training.datamodules import load_dataset; load_dataset('new_dataset', 'train')"` works
3. Label distribution: check class balance, verify attack type coverage
4. Cache generation: `data/cache/{dataset}/` populated after first training run

## Suppressed Topics

Do NOT initiate discussion about:
- Research hypotheses or literature review
- Paper writing or documentation
- Pipeline execution beyond data preprocessing
- Unless the user explicitly asks
