# Mode: Data Engineering

**Active mode**: Data Engineering — Dataset ingestion, preprocessing, validation, and cache management.

## Focus Areas

- New dataset ingestion (CSV → Parquet conversion)
- Dataset catalog management (`config/datasets.yaml`)
- Graph construction validation (temporal windows, node/edge features)
- Preprocessing cache health (`data/cache/`)
- Data quality checks (missing values, label distribution, attack type coverage)
- Parquet/SQLite querying for exploratory analysis

## Context Files (prioritize reading these)

- `config/datasets.yaml` — Dataset catalog (single source of truth for new datasets)
- `pipeline/ingest.py` — CSV → Parquet conversion pipeline
- `src/preprocessing/preprocessing.py` — Graph construction from CAN CSVs
- `config/constants.py` — Window sizes, feature counts, excluded attack types
- `data/project.db` — Project DB with dataset metadata

## Available Commands

| Command | Description |
|---------|-------------|
| `python -m pipeline.ingest <dataset>` | Convert single dataset CSV → Parquet |
| `python -m pipeline.ingest --all` | Convert all catalog datasets |
| `python -m pipeline.ingest --list` | List dataset catalog entries |
| `python -m pipeline.db summary` | Dataset/run/metric counts |
| `python -m pipeline.db query "SQL"` | Arbitrary SQL on project DB |
| `python -m pipeline.analytics dataset <name>` | Dataset summary stats |

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

Then run:
```bash
python -m pipeline.ingest new_dataset
```

## Validation Checklist (for new datasets)

1. Raw CSV structure: timestamp, CAN_ID, data bytes, label columns
2. Parquet conversion: `data/parquet/automotive/{dataset}/` exists with correct schema
3. Graph construction: `python -c "from src.training.datamodules import load_dataset; load_dataset('new_dataset', 'train')"` works
4. Label distribution: check class balance, verify attack type coverage
5. Cache generation: `data/cache/{dataset}/` populated after first training run

## Suppressed Topics

Do NOT initiate discussion about:
- Research hypotheses or literature review
- Paper writing or documentation
- Pipeline execution beyond data preprocessing
- Unless the user explicitly asks
