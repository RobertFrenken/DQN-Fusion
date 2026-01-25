"""
Integration tests for model loading and checkpointing.

Tests that models can save/load checkpoints correctly.
"""

import pytest
import torch
from src.config.hydra_zen_configs import (
    create_gat_normal_config,
    create_autoencoder_config,
)
# Import lightning modules which wrap the actual models
from src.training.lightning_modules import GATLightningModule, VAELightningModule


class TestModelCheckpointing:
    """Test model save/load functionality."""
    
    def test_gat_state_dict_structure(self):
        """Test GAT model state dict structure."""
        config = create_gat_normal_config("hcrl_sa")
        # Use Lightning module which wraps the model
        lightning_module = GATLightningModule(cfg=config, num_ids=1000)
        
        state_dict = lightning_module.state_dict()
        assert len(state_dict) > 0
        # Check for model components in state dict
        assert any(k for k in state_dict.keys() if "model" in k or "conv" in k or "fc" in k)
    
    def test_vgae_state_dict_structure(self):
        """Test VGAE model state dict structure."""
        config = create_autoencoder_config("hcrl_sa")
        # Use Lightning module which wraps the model
        lightning_module = VAELightningModule(cfg=config, num_ids=1000)
        
        state_dict = lightning_module.state_dict()
        assert len(state_dict) > 0
        # Check for encoder components in state dict
        assert any(k for k in state_dict.keys() if "encoder" in k or "vgae" in k or "model" in k)
    
    def test_gat_save_and_load(self, tmp_path):
        """Test saving and loading GAT model."""
        config = create_gat_normal_config("hcrl_sa")
        
        # Create lightning module
        module = GATLightningModule(cfg=config, num_ids=1000)
        
        # Save
        checkpoint_path = tmp_path / "gat_checkpoint.pt"
        torch.save(module.state_dict(), checkpoint_path)
        
        # Load
        loaded_module = GATLightningModule(cfg=config, num_ids=1000)
        loaded_module.load_state_dict(torch.load(checkpoint_path))
        
        # Compare parameters
        for p1, p2 in zip(module.parameters(), loaded_module.parameters()):
            assert torch.equal(p1, p2)
    
    def test_vgae_save_and_load(self, tmp_path):
        """Test saving and loading VGAE model."""
        config = create_autoencoder_config("hcrl_sa")
        
        # Create lightning module
        module = VAELightningModule(cfg=config, num_ids=1000)
        
        # Save
        checkpoint_path = tmp_path / "vgae_checkpoint.pt"
        torch.save(module.state_dict(), checkpoint_path)
        
        # Load
        loaded_module = VAELightningModule(cfg=config, num_ids=1000)
        loaded_module.load_state_dict(torch.load(checkpoint_path))
        
        # Compare parameters
        for p1, p2 in zip(module.parameters(), loaded_module.parameters()):
            assert torch.equal(p1, p2)


class TestModelCompatibility:
    """Test model compatibility across different scenarios."""
    
    def test_teacher_student_compatibility(self):
        """Test that teacher and student models have compatible architectures."""
        # GAT teacher
        teacher_config = create_gat_normal_config("hcrl_sa")
        teacher_module = GATLightningModule(cfg=teacher_config, num_ids=1000)
        
        # GAT student - same architecture type, different size
        student_config = create_gat_normal_config("hcrl_sa")
        student_config.model.hidden_channels = 32  # Smaller
        student_config.model.num_layers = 2  # Fewer layers
        student_config.model.heads = 2  # Fewer heads
        student_module = GATLightningModule(cfg=student_config, num_ids=1000)
        
        # Both should have the same number of output classes
        # Verify both modules were created successfully
        assert teacher_module.batch_size > 0
        assert student_module.batch_size > 0
        assert teacher_config.model.output_dim == student_config.model.output_dim
    
    def test_model_parameter_counts(self):
        """Test that models have reasonable parameter counts."""
        # GAT
        gat_config = create_gat_normal_config("hcrl_sa")
        gat_module = GATLightningModule(cfg=gat_config, num_ids=1000)
        
        gat_params = sum(p.numel() for p in gat_module.parameters())
        assert 100_000 < gat_params < 5_000_000  # Reasonable range
        
        # VGAE
        vgae_config = create_autoencoder_config("hcrl_sa")
        vgae_module = VAELightningModule(cfg=vgae_config, num_ids=1000)
        
        vgae_params = sum(p.numel() for p in vgae_module.parameters())
        assert 100_000 < vgae_params < 5_000_000  # Reasonable range


class TestCheckpointPaths:
    """Test checkpoint path resolution."""
    
    def test_teacher_checkpoint_path_resolution(self):
        """Test that teacher checkpoint paths are resolved correctly."""
        from src.config.hydra_zen_configs import create_distillation_config
        
        config = create_distillation_config("hcrl_sa", "gat_student")
        
        # Check that the attribute exists (it may be None until validation runs)
        assert hasattr(config.training, 'teacher_model_path')
        
        # If set, should contain experiment_runs or experimentruns
        if config.training.teacher_model_path:
            assert "experiment_runs" in str(config.training.teacher_model_path) or "experimentruns" in str(config.training.teacher_model_path)
    
    def test_fusion_artifact_paths(self):
        """Test that fusion artifact paths are specified."""
        from src.config.hydra_zen_configs import create_fusion_config
        config = create_fusion_config("hcrl_sa")
        
        # Check that artifacts are required
        artifacts = config.required_artifacts()
        
        assert len(artifacts) > 0
        # Should have autoencoder and classifier artifacts
        assert any("autoencoder" in str(k) or "vgae" in str(v) for k, v in artifacts.items())
        assert any("classifier" in str(k) or "gat" in str(v) for k, v in artifacts.items())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
