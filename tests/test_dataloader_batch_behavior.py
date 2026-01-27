#!/usr/bin/env python3
"""
Test script to investigate why batch_size=32 creates 1 batch instead of 293 batches.

This script tests PyG DataLoader behavior in isolation to understand if the issue is:
1. PyG DataLoader itself
2. Our dataset implementation
3. Lightning's trainer
4. Something else in the pipeline
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

def test_simple_pyg_dataloader():
    """Test PyG DataLoader with a simple synthetic dataset."""
    print("="*80)
    print("TEST 1: Simple PyG DataLoader with synthetic graphs")
    print("="*80)

    # Create 100 simple graphs
    graphs = []
    for i in range(100):
        x = torch.randn(10, 5)  # 10 nodes, 5 features
        edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
        y = torch.tensor([i % 2])  # Binary label
        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    print(f"Created {len(graphs)} synthetic graphs")

    # Test with batch_size=32
    batch_size = 32
    dataloader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    print(f"\nDataLoader Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  Expected batches: {len(graphs) // batch_size + (1 if len(graphs) % batch_size else 0)}")
    print(f"  len(dataloader): {len(dataloader)}")

    # Iterate and check
    actual_batches = 0
    batch_sizes = []
    for batch in dataloader:
        actual_batches += 1
        batch_sizes.append(batch.num_graphs)
        if actual_batches <= 5:
            print(f"  Batch {actual_batches}: {batch.num_graphs} graphs")

    print(f"\nResults:")
    print(f"  Total batches: {actual_batches}")
    print(f"  Batch sizes: {batch_sizes[:5]}... (showing first 5)")

    expected = 4  # 100/32 = 3 full + 1 partial = 4 batches
    if actual_batches == expected:
        print(f"  ✅ PASS: Got expected {expected} batches")
        return True
    else:
        print(f"  ❌ FAIL: Expected {expected} batches, got {actual_batches}")
        return False


def test_with_real_dataset():
    """Test with actual CAN graph dataset."""
    print("\n" + "="*80)
    print("TEST 2: Real CAN Graph Dataset (hcrl_sa)")
    print("="*80)

    try:
        from src.training.datamodules import load_dataset
        from src.config.hydra_zen_configs import (
            CANGraphConfig, GATConfig, CANDatasetConfig,
            CurriculumTrainingConfig, OptimizerConfig, SchedulerConfig,
            MemoryOptimizationConfig, TrainerConfig
        )

        # Create minimal config
        config = CANGraphConfig(
            model=GATConfig(
                type='gat',
                input_dim=11,
                hidden_channels=64,
                output_dim=2,
                num_layers=3,
                heads=4,
                dropout=0.2,
                num_fc_layers=1,
                embedding_dim=32
            ),
            dataset=CANDatasetConfig(
                name='hcrl_sa',
                modality='automotive',
                data_path='data/automotive/hcrl_sa',
                cache_dir='experimentruns/automotive/hcrl_sa/cache',
                experiment_root='experimentruns'
            ),
            training=CurriculumTrainingConfig(
                mode='curriculum',
                max_epochs=10,
                batch_size=32,
                learning_rate=0.001,
                weight_decay=0.0001,
                optimizer=OptimizerConfig(name='adam', lr=0.001, weight_decay=0.0001),
                scheduler=SchedulerConfig(use_scheduler=False),
                memory_optimization=MemoryOptimizationConfig()
            ),
            trainer=TrainerConfig(
                accelerator='cpu',
                devices=1
            )
        )

        print("Loading dataset...")
        train_dataset, val_dataset, num_ids = load_dataset('hcrl_sa', config)

        print(f"\nDataset Info:")
        print(f"  Train dataset length: {len(train_dataset)}")
        print(f"  Validation dataset length: {len(val_dataset)}")
        print(f"  Number of unique IDs: {num_ids}")

        # Test with batch_size=32
        batch_size = 32
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        expected_batches = len(train_dataset) // batch_size + (1 if len(train_dataset) % batch_size else 0)

        print(f"\nDataLoader Configuration:")
        print(f"  batch_size: {batch_size}")
        print(f"  Expected batches: {expected_batches}")
        print(f"  len(dataloader): {len(dataloader)}")

        # Iterate and check
        actual_batches = 0
        batch_sizes = []
        for batch in dataloader:
            actual_batches += 1
            batch_sizes.append(batch.num_graphs)
            if actual_batches <= 5:
                print(f"  Batch {actual_batches}: {batch.num_graphs} graphs")
            if actual_batches >= 10:  # Don't iterate through thousands of batches
                print(f"  ... (stopping after 10 batches for speed)")
                break

        print(f"\nResults:")
        print(f"  Iterated batches (first 10): {actual_batches}")
        print(f"  Batch sizes: {batch_sizes}")

        if actual_batches >= 10 and all(bs == batch_size for bs in batch_sizes[:9]):
            print(f"  ✅ PASS: DataLoader creating proper batches of size {batch_size}")
            return True
        elif actual_batches == 1:
            print(f"  ❌ FAIL: Only 1 batch! This reproduces the bug.")
            print(f"  First batch contained {batch_sizes[0]} graphs (should be {batch_size})")
            return False
        else:
            print(f"  ⚠️  UNEXPECTED: Got {actual_batches} batches")
            return False

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_curriculum_datamodule():
    """Test EnhancedCANGraphDataModule specifically."""
    print("\n" + "="*80)
    print("TEST 3: EnhancedCANGraphDataModule (Curriculum)")
    print("="*80)

    try:
        from src.training.datamodules import load_dataset, EnhancedCANGraphDataModule
        from src.config.hydra_zen_configs import (
            CANGraphConfig, GATConfig, CANDatasetConfig,
            CurriculumTrainingConfig, OptimizerConfig, SchedulerConfig,
            MemoryOptimizationConfig, TrainerConfig
        )

        # Create config
        config = CANGraphConfig(
            model=GATConfig(
                type='gat', input_dim=11, hidden_channels=64, output_dim=2,
                num_layers=3, heads=4, dropout=0.2, num_fc_layers=1, embedding_dim=32
            ),
            dataset=CANDatasetConfig(
                name='hcrl_sa', modality='automotive',
                data_path='data/automotive/hcrl_sa',
                cache_dir='experimentruns/automotive/hcrl_sa/cache',
                experiment_root='experimentruns'
            ),
            training=CurriculumTrainingConfig(
                mode='curriculum', max_epochs=10, batch_size=32,
                learning_rate=0.001, weight_decay=0.0001,
                optimizer=OptimizerConfig(name='adam', lr=0.001, weight_decay=0.0001),
                scheduler=SchedulerConfig(use_scheduler=False),
                memory_optimization=MemoryOptimizationConfig()
            ),
            trainer=TrainerConfig(accelerator='cpu', devices=1)
        )

        print("Loading dataset...")
        full_dataset, val_dataset, num_ids = load_dataset('hcrl_sa', config)

        # Separate normal and attack
        train_normal = [g for g in full_dataset if g.y.item() == 0]
        train_attack = [g for g in full_dataset if g.y.item() == 1]
        val_normal = [g for g in val_dataset if g.y.item() == 0]
        val_attack = [g for g in val_dataset if g.y.item() == 1]

        print(f"\nDataset Separation:")
        print(f"  Train normal: {len(train_normal)}")
        print(f"  Train attack: {len(train_attack)}")
        print(f"  Val normal: {len(val_normal)}")
        print(f"  Val attack: {len(val_attack)}")

        # Create datamodule WITHOUT VGAE (test basic functionality)
        batch_size = 32
        datamodule = EnhancedCANGraphDataModule(
            train_normal=train_normal,
            train_attack=train_attack,
            val_normal=val_normal,
            val_attack=val_attack,
            vgae_model=None,  # Skip VGAE for this test
            batch_size=batch_size,
            num_workers=0,  # Single-threaded for testing
            total_epochs=10
        )

        print(f"\nDataModule Info:")
        print(f"  batch_size: {datamodule.batch_size}")
        print(f"  train_dataset length: {len(datamodule.train_dataset)}")

        # Get dataloader
        train_loader = datamodule.train_dataloader()

        expected_batches = len(datamodule.train_dataset) // batch_size + (1 if len(datamodule.train_dataset) % batch_size else 0)

        print(f"\nDataLoader Info:")
        print(f"  Expected batches: {expected_batches}")
        print(f"  len(train_loader): {len(train_loader)}")

        # Iterate
        actual_batches = 0
        batch_sizes = []
        for batch in train_loader:
            actual_batches += 1
            batch_sizes.append(batch.num_graphs)
            if actual_batches <= 5:
                print(f"  Batch {actual_batches}: {batch.num_graphs} graphs")
            if actual_batches >= 10:
                print(f"  ... (stopping after 10 batches)")
                break

        print(f"\nResults:")
        print(f"  Iterated batches: {actual_batches}")
        print(f"  Batch sizes: {batch_sizes}")

        if actual_batches >= 10 and all(bs == batch_size for bs in batch_sizes[:9]):
            print(f"  ✅ PASS: Curriculum datamodule creating proper batches")
            return True
        elif actual_batches == 1:
            print(f"  ❌ FAIL: Only 1 batch! Bug reproduced in curriculum datamodule.")
            return False
        else:
            print(f"  ⚠️  Got {actual_batches} batches (expected {expected_batches})")
            return False

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("PyG DataLoader Batch Behavior Investigation")
    print("=" * 80)
    print()

    results = {}

    # Test 1: Simple synthetic data
    results['synthetic'] = test_simple_pyg_dataloader()

    # Test 2: Real dataset
    results['real_dataset'] = test_with_real_dataset()

    # Test 3: Curriculum datamodule
    results['curriculum'] = test_curriculum_datamodule()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All tests passed! DataLoader behavior is normal.")
        print("The issue must be in Lightning's trainer or elsewhere in the pipeline.")
    else:
        print("\n❌ Some tests failed! DataLoader is not behaving as expected.")
        print("This confirms there's an issue with batch creation.")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
