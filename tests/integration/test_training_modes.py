"""
Integration tests for training mode initialization.

Tests that all training modes can initialize their trainers correctly.
"""

import pytest
from src.config.hydra_zen_configs import (
    create_gat_normal_config,
    create_autoencoder_config,
    create_fusion_config,
)
from src.training.trainer import HydraZenTrainer


class TestTrainerInitialization:
    """Test that trainers can be initialized for all modes."""
    
    @pytest.mark.parametrize("dataset", ["hcrl_sa", "hcrl_ch"])
    def test_normal_mode_trainer(self, dataset):
        """Test normal mode trainer initialization."""
        config = create_gat_normal_config(dataset)
        trainer = HydraZenTrainer(config)
        
        assert trainer is not None
        assert trainer.config == config
        assert hasattr(trainer, 'train')
    
    @pytest.mark.parametrize("dataset", ["hcrl_sa", "hcrl_ch"])
    def test_autoencoder_mode_trainer(self, dataset):
        """Test autoencoder mode trainer initialization."""
        config = create_autoencoder_config(dataset)
        trainer = HydraZenTrainer(config)
        
        assert trainer is not None
        assert trainer.config == config
        assert config.training.mode == "autoencoder"
    
    @pytest.mark.parametrize("dataset", ["hcrl_sa", "hcrl_ch"])
    def test_fusion_mode_trainer(self, dataset):
        """Test fusion mode trainer initialization.
        
        Note: Fusion requires pre-trained artifacts, so this test will fail
        if the required models haven't been trained yet. This is expected behavior.
        """
        config = create_fusion_config(dataset)
        
        # Try to create trainer - will fail if artifacts missing
        try:
            trainer = HydraZenTrainer(config)
            assert trainer is not None
            assert trainer.config == config
            assert config.training.mode == "fusion"
        except FileNotFoundError as e:
            # Expected if artifacts don't exist yet
            pytest.skip(f"Fusion requires pre-trained artifacts: {e}")


class TestTrainingModeCompatibility:
    """Test that models are compatible with their training modes."""
    
    def test_gat_with_normal_mode(self):
        """Test GAT model with normal training mode."""
        config = create_gat_normal_config("hcrl_sa")
        assert config.model.type == "gat"
        assert config.training.mode == "normal"
    
    def test_vgae_with_autoencoder_mode(self):
        """Test VGAE model with autoencoder mode."""
        config = create_autoencoder_config("hcrl_sa")
        assert config.model.type == "vgae"
        assert config.training.mode == "autoencoder"
    
    def test_dqn_with_fusion_mode(self):
        """Test DQN model with fusion mode."""
        config = create_fusion_config("hcrl_sa")
        assert config.model.type == "dqn"
        assert config.training.mode == "fusion"


class TestDatasetCompatibility:
    """Test that all datasets work with all model types."""
    
    @pytest.mark.parametrize("dataset", ["hcrl_sa", "hcrl_ch", "set_01", "set_02"])
    def test_gat_with_all_datasets(self, dataset):
        """Test GAT with all datasets."""
        config = create_gat_normal_config(dataset)
        trainer = HydraZenTrainer(config)
        
        assert trainer is not None
        assert config.dataset.name == dataset
    
    @pytest.mark.parametrize("dataset", ["hcrl_sa", "hcrl_ch", "set_01", "set_02"])
    def test_vgae_with_all_datasets(self, dataset):
        """Test VGAE with all datasets."""
        config = create_autoencoder_config(dataset)
        trainer = HydraZenTrainer(config)
        
        assert trainer is not None
        assert config.dataset.name == dataset


class TestExperimentOutput:
    """Test experiment output directory structure."""
    
    def test_output_directory_creation(self):
        """Test that output directories are created correctly."""
        config = create_gat_normal_config("hcrl_sa")
        canonical_dir = config.canonical_experiment_dir()
        
        # Should contain experiment_runs or experimentruns
        canonical_str = str(canonical_dir)
        assert any(x in canonical_str for x in ["experiment_runs", "experimentruns"])
        
        # Should contain dataset folder
        assert "automotive" in canonical_str
        
        # Should contain key components (dataset, model type, training mode)
        assert "hcrl_sa" in canonical_str
        assert "gat" in canonical_str
        assert "normal" in canonical_str
    
    def test_different_experiments_different_dirs(self):
        """Test that different experiments get different directories."""
        config1 = create_gat_normal_config("hcrl_sa")
        config2 = create_autoencoder_config("hcrl_sa")
        config3 = create_fusion_config("hcrl_sa")
        
        dir1 = config1.canonical_experiment_dir()
        dir2 = config2.canonical_experiment_dir()
        dir3 = config3.canonical_experiment_dir()
        
        # All should be different
        assert dir1 != dir2
        assert dir2 != dir3
        assert dir1 != dir3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
