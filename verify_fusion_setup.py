#!/usr/bin/env python3
"""
Verification Checklist for Fusion Training Integration

Run this script to verify that all fusion training components are properly set up.
This ensures the Lightning-based fusion pipeline is ready to use.

Usage:
    python verify_fusion_setup.py
    python verify_fusion_setup.py --verbose
"""

import os
import sys
from pathlib import Path
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FusionSetupChecker:
    """Verify fusion training setup."""
    
    def __init__(self, verbose: bool = False):
        self.project_root = Path(__file__).parent
        self.verbose = verbose
        self.checks = []
        self.failed = []
    
    def check_file_exists(self, path: str, description: str) -> bool:
        """Check if a file exists."""
        full_path = self.project_root / path
        exists = full_path.exists()
        
        status = "✅" if exists else "❌"
        self.checks.append((status, description, exists))
        
        if self.verbose:
            logger.info(f"  {status} {description} ({path})")
        
        if not exists:
            self.failed.append(f"{description}: {path}")
        
        return exists
    
    def check_directory_exists(self, path: str, description: str) -> bool:
        """Check if a directory exists."""
        full_path = self.project_root / path
        exists = full_path.is_dir()
        
        status = "✅" if exists else "⚠️"
        self.checks.append((status, description, exists))
        
        if self.verbose:
            logger.info(f"  {status} {description} ({path})")
        
        return exists
    
    def check_python_import(self, module_path: str, description: str) -> bool:
        """Check if a Python module can be imported."""
        try:
            # Add project root to path
            sys.path.insert(0, str(self.project_root))
            
            parts = module_path.replace('/', '.').replace('.py', '').split('.')
            __import__('.'.join(parts))
            
            status = "✅"
            exists = True
        except ImportError as e:
            status = "❌"
            exists = False
            if self.verbose:
                logger.info(f"    Import error: {e}")
        
        self.checks.append((status, description, exists))
        
        if self.verbose:
            logger.info(f"  {status} {description}")
        
        if not exists:
            self.failed.append(f"{description}: {module_path}")
        
        return exists
    
    def run_all_checks(self) -> Tuple[int, int]:
        """Run all verification checks."""
        
        logger.info("\n" + "="*70)
        logger.info("🔍 FUSION TRAINING SETUP VERIFICATION")
        logger.info("="*70)
        
        # === Core Files ===
        logger.info("\n📂 Core Fusion Training Files:")
        self.check_file_exists("src/training/fusion_lightning.py", "FusionLightningModule")
        self.check_file_exists("src/training/prediction_cache.py", "PredictionCacheBuilder")
        self.check_file_exists("src/config/fusion_config.py", "Fusion Hydra-Zen Config")
        
        # === Training Scripts ===
        logger.info("\n📄 Training Scripts:")
        self.check_file_exists("train_with_hydra_zen.py", "Main training script (with fusion support)")
        self.check_file_exists("train_fusion_lightning.py", "Standalone fusion training script")
        self.check_file_exists("fusion_training_demo.py", "Fusion training demo")
        
        # === Documentation ===
        logger.info("\n📖 Documentation:")
        self.check_file_exists("FUSION_TRAINING_GUIDE.md", "Complete fusion training guide")
        self.check_file_exists("FUSION_QUICK_START.sh", "Quick reference guide")
        self.check_file_exists("FUSION_INTEGRATION_SUMMARY.md", "Integration summary")
        
        # === Dependencies ===
        logger.info("\n📦 Python Imports:")
        self.check_python_import("src/training/fusion_lightning.py", "FusionLightningModule import")
        self.check_python_import("src/training/prediction_cache.py", "PredictionCacheBuilder import")
        self.check_python_import("src/config/fusion_config.py", "Fusion config import")
        self.check_python_import("train_models.py", "CANGraphLightningModule import")
        
        # === Directories ===
        logger.info("\n📁 Output Directories:")
        self.check_directory_exists("src/training", "Training module directory")
        self.check_directory_exists("src/config", "Config module directory")
        self.check_directory_exists("saved_models", "Model checkpoints directory")
        self.check_directory_exists("logs", "Training logs directory")
        
        # === Optional Caches ===
        logger.info("\n💾 Optional Caches (will be created on first run):")
        self.check_directory_exists("cache/fusion", "Fusion prediction cache")
        
        # === Summary ===
        passed = sum(1 for status, _, _ in self.checks if status == "✅")
        warned = sum(1 for status, _, _ in self.checks if status == "⚠️")
        failed = sum(1 for status, _, _ in self.checks if status == "❌")
        
        logger.info("\n" + "="*70)
        logger.info("📊 VERIFICATION SUMMARY")
        logger.info("="*70)
        logger.info(f"✅ Passed:  {passed}")
        logger.info(f"⚠️  Warned: {warned}")
        logger.info(f"❌ Failed: {failed}")
        
        if self.failed:
            logger.info("\n⚠️  Missing components:")
            for item in self.failed:
                logger.info(f"  - {item}")
        
        return passed, failed
    
    def print_next_steps(self):
        """Print recommended next steps."""
        
        logger.info("\n" + "="*70)
        logger.info("🚀 NEXT STEPS")
        logger.info("="*70)
        
        logger.info("\n1️⃣  Quick test (demo with existing models):")
        logger.info("   python fusion_training_demo.py --dataset hcrl_sa --quick")
        
        logger.info("\n2️⃣  Full fusion training (main script):")
        logger.info("   python train_with_hydra_zen.py --preset fusion_hcrl_sa")
        
        logger.info("\n3️⃣  Full fusion training (standalone script):")
        logger.info("   python train_fusion_lightning.py --dataset hcrl_sa")
        
        logger.info("\n4️⃣  View available presets:")
        logger.info("   python train_with_hydra_zen.py --list-presets | grep fusion")
        
        logger.info("\n5️⃣  Read complete guide:")
        logger.info("   cat FUSION_TRAINING_GUIDE.md")
        
        logger.info("\n" + "="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify fusion training setup')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    checker = FusionSetupChecker(verbose=args.verbose)
    passed, failed = checker.run_all_checks()
    
    if failed == 0:
        logger.info("\n✅ All checks passed! Fusion training is ready to use.")
    else:
        logger.info(f"\n⚠️  {failed} checks failed. See above for details.")
    
    checker.print_next_steps()
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
