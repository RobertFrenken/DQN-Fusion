import os
import sys
from pathlib import Path
import pytest

# Ensure project root is on PYTHONPATH so `src` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Use real imports since we're running in gnn-experiments environment
from src.training.datamodules import load_dataset
from src.config.hydra_zen_configs import CANDatasetConfig, GATConfig, NormalTrainingConfig


def test_load_dataset_raises_error_when_dataset_not_found(tmp_path):
    """Test that load_dataset raises FileNotFoundError when dataset cannot be found."""
    import types
    
    model = GATConfig()
    dataset = CANDatasetConfig(name='nonexistent_dataset_xyz')
    # Use a path that doesn't exist and won't match any fallback
    dataset.data_path = str(tmp_path / "nonexistent")

    training = NormalTrainingConfig()
    cfg = types.SimpleNamespace()
    cfg.dataset = dataset

    # Should raise FileNotFoundError when dataset path doesn't exist
    with pytest.raises(FileNotFoundError) as exc:
        load_dataset('nonexistent_dataset_xyz', cfg)
    assert 'nonexistent_dataset_xyz' in str(exc.value) and 'not found' in str(exc.value)


def test_adaptive_graph_dataset_requires_vgae():
    """Test that AdaptiveGraphDataset raises error when VGAE model is required but not provided."""
    from src.training.datamodules import AdaptiveGraphDataset
    import types
    
    # Create dummy graph objects
    normal = [types.SimpleNamespace(x=None, edge_index=None) for _ in range(10)]
    attack = [types.SimpleNamespace(x=None, edge_index=None) for _ in range(2)]
    
    # Create instance without VGAE model - the __init__ will fail when trying to generate epoch samples
    # This test verifies that initialization properly detects missing VGAE and raises RuntimeError
    with pytest.raises(RuntimeError) as exc:
        ag = AdaptiveGraphDataset(
            normal_graphs=normal,
            attack_graphs=attack,
            vgae_model=None,
            current_epoch=0,
            total_epochs=200
        )
    assert 'VGAE model required' in str(exc.value)
