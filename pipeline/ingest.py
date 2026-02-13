"""CSV → Parquet ingestion and dataset registration.

Reads the dataset catalog (data/datasets.yaml), validates CSV structure,
converts to Parquet with proper types, computes statistics, and registers
the dataset in the project SQLite database.

Usage:
    python -m pipeline.ingest <dataset>
    python -m pipeline.ingest --all
    python -m pipeline.ingest --list
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import yaml

from .paths import CATALOG_PATH

log = logging.getLogger(__name__)

PARQUET_ROOT = Path("data/parquet")
ROW_GROUP_SIZE = 500_000


def load_catalog() -> dict:
    """Load the dataset catalog from data/datasets.yaml."""
    with open(CATALOG_PATH) as f:
        return yaml.safe_load(f)


def _csv_files(directory: Path) -> list[Path]:
    """Collect CSV files from a directory, sorted for determinism."""
    return sorted(directory.glob("*.csv"))


def _convert_csv_to_parquet(
    csv_path: Path,
    out_path: Path,
    source_label: str,
) -> int:
    """Convert a single CAN CSV file to Parquet.

    Reads the CSV with PyArrow (streaming, low memory), parses hex CAN IDs
    to uint32, stores the data_field as-is (string), and adds a source_file
    column for provenance.

    Returns the number of rows written.
    """
    read_opts = pcsv.ReadOptions(block_size=64 * 1024 * 1024)
    table = pcsv.read_csv(csv_path, read_options=read_opts)

    col_names = table.column_names
    if len(col_names) < 4:
        raise ValueError(
            f"{csv_path}: expected >=4 columns (timestamp, id, data, label), "
            f"got {col_names}"
        )

    ts_col = table.column(0).cast(pa.float64())
    id_col = table.column(1)
    data_col = table.column(2).cast(pa.string())
    label_col = table.column(3).cast(pa.uint8())

    # Parse hex CAN IDs to uint32
    id_strings = id_col.to_pylist()
    id_ints = pa.array([int(str(v), 16) for v in id_strings], type=pa.uint32())

    source_col = pa.array(
        [source_label] * len(table), type=pa.string()
    )

    out_table = pa.table({
        "timestamp": ts_col,
        "id": id_ints,
        "data_field": data_col,
        "label": label_col,
        "source_file": source_col,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, out_path, row_group_size=ROW_GROUP_SIZE)

    # Validate schema on a sample (pandera is optional)
    try:
        from .schemas import CAN_PARQUET_SCHEMA
        sample = pq.read_table(out_path).to_pandas().head(1000)
        CAN_PARQUET_SCHEMA.validate(sample)
    except ImportError:
        pass  # pandera not installed
    except Exception as e:
        log.error("Schema validation failed for %s: %s", out_path, e)
        raise

    return len(out_table)


def ingest_dataset(name: str, catalog: dict | None = None) -> dict:
    """Ingest a single dataset: validate, convert to Parquet, compute stats.

    Returns a dict of computed statistics for database registration.
    """
    if catalog is None:
        catalog = load_catalog()

    if name not in catalog:
        raise KeyError(f"Dataset '{name}' not found in {CATALOG_PATH}")

    entry = catalog[name]
    csv_dir = Path(entry["csv_dir"])
    domain = entry["domain"]
    parquet_dir = PARQUET_ROOT / domain / name

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    log.info("Ingesting %s from %s → %s", name, csv_dir, parquet_dir)

    total_rows = 0
    total_files = 0
    unique_ids: set[int] = set()

    # Process all subdirs that contain CSVs
    subdirs = [d for d in sorted(csv_dir.iterdir()) if d.is_dir()]
    for subdir in subdirs:
        csv_files = _csv_files(subdir)
        for csv_file in csv_files:
            out_name = f"{subdir.name}__{csv_file.stem}.parquet"
            out_path = parquet_dir / out_name
            source_label = f"{subdir.name}/{csv_file.name}"

            log.info("  Converting %s", source_label)
            n_rows = _convert_csv_to_parquet(csv_file, out_path, source_label)
            total_rows += n_rows
            total_files += 1

            # Compute stats from the written Parquet
            pf = pq.read_table(out_path, columns=["id"])
            unique_ids.update(pf.column("id").to_pylist())

    # Determine attack types from test subdirs
    test_subdirs = entry.get("test_subdirs", [])
    attack_types = [
        s.replace("test_", "").lstrip("0123456789_")
        for s in test_subdirs
    ]

    stats = {
        "name": name,
        "domain": domain,
        "protocol": entry.get("protocol", ""),
        "source": entry.get("source", ""),
        "description": entry.get("description", ""),
        "num_files": total_files,
        "num_samples": total_rows,
        "num_unique_ids": len(unique_ids),
        "attack_types": attack_types,
        "added_by": entry.get("added_by", ""),
        "parquet_path": str(parquet_dir),
    }

    log.info(
        "Done: %s — %d files, %d rows, %d unique IDs",
        name, total_files, total_rows, len(unique_ids),
    )
    return stats


def ingest_all(catalog: dict | None = None) -> list[dict]:
    """Ingest all datasets in the catalog."""
    if catalog is None:
        catalog = load_catalog()
    results = []
    for name in catalog:
        stats = ingest_dataset(name, catalog)
        results.append(stats)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline.ingest",
        description="Ingest CAN CSV datasets into Parquet format",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("dataset", nargs="?", help="Dataset name to ingest")
    group.add_argument("--all", action="store_true", help="Ingest all datasets")
    group.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )

    catalog = load_catalog()

    if args.list:
        for name, entry in catalog.items():
            print(f"  {name:12s}  {entry['domain']:12s}  {entry.get('description', '')[:60]}")
        return

    if args.all:
        results = ingest_all(catalog)
        # Register all in project DB
        from .db import register_datasets
        register_datasets(results)
        log.info("Registered %d datasets in project DB", len(results))
        return

    stats = ingest_dataset(args.dataset, catalog)
    from .db import register_datasets
    register_datasets([stats])
    log.info("Registered %s in project DB", args.dataset)


if __name__ == "__main__":
    main()
