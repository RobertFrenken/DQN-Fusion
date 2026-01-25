"""
Integration tests for configuration creation and validation.

Tests that all configuration presets create valid, trainable configs.
"""

import pytest
from src.config.hydra_zen_configs import (
    CANGraphConfigStore,
    validate_config,
    create_gat_normal_config,
    create_autoencoder_config,
    create_distillation_config,
    create_fusion_config,
)


class TestConfigPresets:
    """Test all configuration preset factories."""
    
    @pytest.mark.parametrize("dataset", ["hcrl_sa", "hcrl_ch"])
    def test_gat_normal_config(self, dataset):
        """Test GAT normal training config creation."""
        config = create_gat_normal_config(dataset)
        
        assert config.model.type == "gat"
        assert config.dataset.name == dataset
        assert config.training.mode == "normal"
        assert validate_config(config)
    
    @pytest.mark.parametrize("dataset", ["hcrl_sa", "hcrl_ch"])
    def test_autoencoder_config(self, dataset):
        """Test VGAE autoencoder config creation."""
        config = create_autoencoder_config(dataset)
        
        assert config.model.type == "vgae"
        assert config.dataset.name == dataset
        assert config.training.mode == "autoencoder"
        assert validate_config(config)
    
    @pytest.mark.parametrize("dataset", ["hcrl_sa", "hcrl_ch"])
    @pytest.mark.parametrize("student_model", ["gat_student", "vgae_student"])
    def test_distillation_config(self, dataset, student_model):
        """Test knowledge distillation config creation."""
        config = create_distillation_config(dataset, student_model)
        
        assert config.model.type == student_model
        assert config.dataset.name == dataset
        assert config.training.mode == "knowledge_distillation"
        assert config.training.distillation_temperature > 0
        assert 0 <= config.training.distillation_alpha <= 1
        # Note: validate_config() will fail without teacher model, so skip validation here
    
    @pytest.mark.parametrize("dataset", ["hcrl_sa", "hcrl_ch"])
    def test_fusion_config(self, dataset):
        """Test fusion agent config creation."""
        config = create_fusion_config(dataset)
        
        assert config.model.type == "dqn"
        assert config.dataset.name == dataset
        assert config.training.mode == "fusion"
        # Note: validate_config() will fail without artifacts, so skip validation here


class TestConfigStore:
    """Test the configuration store interface."""
    
    def test_store_initialization(self):
        """Test that config store initializes correctly."""
        store = CANGraphConfigStore()
        assert store is not None
    
    @pytest.mark.parametrize("model_type", [
        "gat", "gat_student", "vgae", "vgae_student", "dqn", "dqn_student"
    ])
    @pytest.mark.parametrize("dataset_name", ["hcrl_sa", "hcrl_ch"])
    @pytest.mark.parametrize("training_mode", [
        "normal", "autoencoder", "curriculum", "knowledge_distillation", "fusion"
    ])
    def test_create_all_combinations(self, model_type, dataset_name, training_mode):
        """Test creating configs for all valid combinations."""
        store = CANGraphConfigStore()
        
        # Some combinations are invalid (e.g., GAT with autoencoder mode)
        # Only test valid combinations
        valid_combinations = {
            ("gat", "normal"),
            ("gat", "curriculum"),
            ("gat_student", "knowledge_distillation"),
            ("gat_student", "normal"),  # student baseline
            ("vgae", "autoencoder"),
            ("vgae", "normal"),
            ("vgae_student", "knowledge_distillation"),
            ("vgae_student", "normal"),  # student baseline
            ("dqn", "fusion"),
            ("dqn_student", "knowledge_distillation"),
        }
        
        if (model_type, training_mode) not in valid_combinations:
            # Skip invalid combinations
            pytest.skip(f"Invalid combination: {model_type} + {training_mode}")
        
        config = store.create_config(
            model_type=model_type,
            dataset_name=dataset_name,
            training_mode=training_mode
        )
        
        assert config.model.type == model_type
        assert config.dataset.name == dataset_name
        assert config.training.mode == training_mode
        # Skip validation for configs that need artifacts (fusion, curriculum, distillation)
        if training_mode not in ["fusion", "curriculum", "knowledge_distillation"]:
            assert validate_config(config)


class TestConfigValidation:
    """Test configuration validation logic."""
    
    def test_valid_gat_config(self):
        """Test that valid GAT config passes validation."""
        config = create_gat_normal_config("hcrl_sa")
        assert validate_config(config)
    
    def test_valid_vgae_config(self):
        """Test that valid VGAE config passes validation."""
        config = create_autoencoder_config("hcrl_sa")
        assert validate_config(config)
    
    def test_valid_fusion_config(self):
        """Test that valid fusion config structure is correct."""
        config = create_fusion_config("hcrl_sa")
        assert config.model.type == "dqn"
        assert config.training.mode == "fusion"
        # Note: validation requires artifacts, skip here
    
    def test_experiment_name_generation(self):
        """Test that experiment names are generated correctly."""
        config = create_gat_normal_config("hcrl_sa")
        assert "gat" in config.experiment_name.lower()
        assert "hcrl_sa" in config.experiment_name.lower()
    
    def test_canonical_dir_generation(self):
        """Test that canonical experiment dirs are generated correctly."""
        config = create_gat_normal_config("hcrl_sa")
        canonical_dir = str(config.canonical_experiment_dir())
        
        assert "experiment_runs" in canonical_dir or "experimentruns" in canonical_dir
        assert "automotive" in canonical_dir
        # Experiment name components should be in the path (like 'gat', 'hcrl_sa', 'normal')
        assert "gat" in canonical_dir
        assert "hcrl_sa" in canonical_dir


class TestConfigParameters:
    """Test that configs have reasonable parameter values."""
    
    def test_gat_parameters(self):
        """Test GAT model parameters."""
        config = create_gat_normal_config("hcrl_sa")
        
        assert config.model.hidden_channels > 0
        assert config.model.num_layers > 0
        assert 0 <= config.model.dropout < 1
        assert config.model.heads > 0
    
    def test_vgae_parameters(self):
        """Test VGAE model parameters."""
        config = create_autoencoder_config("hcrl_sa")
        
        assert config.model.node_embedding_dim > 0
        assert config.model.latent_dim > 0
        assert 0 <= config.model.dropout < 1
    
    def test_training_parameters(self):
        """Test training parameters."""
        config = create_gat_normal_config("hcrl_sa")
        
        assert config.training.max_epochs > 0
        assert config.training.batch_size > 0
        assert config.training.learning_rate > 0
        assert config.training.weight_decay >= 0
    
    def test_distillation_parameters(self):
        """Test distillation-specific parameters."""
        config = create_distillation_config("hcrl_sa", "gat_student")
        
        assert config.training.distillation_temperature > 0
        assert 0 <= config.training.distillation_alpha <= 1
        # teacher_model_path is optional and can be auto-detected
    
    def test_fusion_parameters(self):
        """Test fusion-specific parameters."""
        config = create_fusion_config("hcrl_sa")
        
        assert config.training.fusion_episodes > 0
        assert config.training.max_train_samples > 0
        # Artifact paths are optional and auto-detected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
