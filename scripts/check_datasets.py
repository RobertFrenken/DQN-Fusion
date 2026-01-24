"""
Check that required datasets exist and optionally attempt to load them via project loader.

Usage:
  # Check dataset path registered in config store
  python scripts/check_datasets.py --dataset hcrl_ch

  # Check a custom path
  python scripts/check_datasets.py --path /path/to/my/dataset --run-load

  # Force rebuild cache when attempting to load
  python scripts/check_datasets.py --path /path/to/my/dataset --run-load --force-rebuild
"""
import argparse
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_path(p: Path) -> bool:
    if p.exists():
        logger.info(f"✅ Dataset exists: {p}")
        return True
    else:
        logger.warning(f"❌ Dataset NOT found: {p}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check dataset availability and optionally try to load it")
    parser.add_argument("--dataset", help="Dataset key from config store (e.g., hcrl_ch)")
    parser.add_argument("--path", help="Explicit dataset path to check")
    parser.add_argument("--run-load", action="store_true", help="Attempt to call the project's loader to validate CSV discovery and cache")
    parser.add_argument("--force-rebuild", action="store_true", help="If running load, force rebuilding caches")
    args = parser.parse_args()

    # Prefer explicit path
    if args.path:
        p = Path(args.path)
        ok = check_path(p)
    elif args.dataset:
        # Try to use hydra store if available
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from src.config.hydra_zen_configs import CANGraphConfigStore
            store = CANGraphConfigStore()
            ds = store.get_dataset_config(args.dataset)
            p = Path(ds.data_path)
            ok = check_path(p)
        except Exception as e:
            logger.warning(f"Could not resolve dataset via config store: {e}")
            print("Provide --path if you want to check a custom dataset location")
            return
    else:
        parser.print_help()
        return

    if args.run_load:
        # Try to call load_dataset (may require heavy deps)
        try:
            logger.info("Attempting to run dataset loader (load_dataset)")
            from src.training.datamodules import load_dataset
            # Create a minimal dummy config object only for this call
            class _Cfg:
                class dataset:
                    name = args.dataset if args.dataset else p.name
                    data_path = str(p)
            cfg = _Cfg()
            train, val, num_ids = load_dataset(cfg.dataset.name, cfg, force_rebuild_cache=args.force_rebuild)
            logger.info(f"Loaded dataset: train={len(train)}, val={len(val)}, ids={num_ids}")
        except Exception as e:
            logger.error(f"Dataset load failed: {e}")
            logger.error("If this is a local dev machine, consider using --force-rebuild or --use-synthetic-data in smoke script")


if __name__ == '__main__':
    main()
