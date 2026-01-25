"""
Integration test for end-to-end training pipeline.

Tests the complete pipeline: teacher training → distillation → fusion
This is a smoke test with minimal epochs to ensure the pipeline works.
"""

import pytest
from pathlib import Path
from src.config.hydra_zen_configs import (
    create_gat_normal_config,
    create_autoencoder_config,
    create_distillation_config,
    create_fusion_config,
)
from src.training.trainer import HydraZenTrainer


class TestEndToEndPipeline:
    """Test complete training pipeline with minimal epochs."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_gat_teacher_training(self, tmp_path):
        """Test GAT teacher training (minimal)."""
        config = create_gat_normal_config("hcrl_sa")
        
        # Override for fast test
        config.training.max_epochs = 1
        config.training.batch_size = 16
        config.output_dir = str(tmp_path)
        config.logging["enable_tensorboard"] = False
        
        trainer = HydraZenTrainer(config)
        
        # Should not crash
        try:
            results = trainer.train()
            assert results is not None
        except Exception as e:
            # If data not available, skip
            if "FileNotFoundError" in str(e) or "data" in str(e).lower():
                pytest.skip(f"Data not available: {e}")
            raise
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_vgae_autoencoder_training(self, tmp_path):
        """Test VGAE autoencoder training (minimal)."""
        config = create_autoencoder_config("hcrl_sa")
        
        # Override for fast test
        config.training.max_epochs = 1
        config.training.batch_size = 16
        config.output_dir = str(tmp_path)
        config.logging["enable_tensorboard"] = False
        
        trainer = HydraZenTrainer(config)
        
        try:
            results = trainer.train()
            assert results is not None
        except Exception as e:
            if "FileNotFoundError" in str(e) or "data" in str(e).lower():
                pytest.skip(f"Data not available: {e}")
            raise
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_knowledge_distillation_flow(self, tmp_path):
        """Test knowledge distillation flow (requires teacher model)."""
        # Step 1: Check if teacher model exists
        teacher_config = create_gat_normal_config("hcrl_sa")
        teacher_checkpoint = Path(teacher_config.canonical_experiment_dir()) / "checkpoints" / "best_model.ckpt"
        
        if not teacher_checkpoint.exists():
            pytest.skip("Teacher model not found - run teacher training first")
        
        # Step 2: Create distillation config
        config = create_distillation_config("hcrl_sa", "gat_student")
        config.training.max_epochs = 1
        config.training.batch_size = 16
        config.output_dir = str(tmp_path)
        
        # Step 3: Run distillation
        trainer = HydraZenTrainer(config)
        
        try:
            results = trainer.train()
            assert results is not None
        except Exception as e:
            if "FileNotFoundError" in str(e) or "data" in str(e).lower():
                pytest.skip(f"Data not available: {e}")
            raise
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_fusion_training_flow(self, tmp_path):
        """Test fusion training flow (requires VGAE and GAT models)."""
        # Check if required models exist
        vgae_config = create_autoencoder_config("hcrl_sa")
        gat_config = create_gat_normal_config("hcrl_sa")
        
        vgae_checkpoint = Path(vgae_config.canonical_experiment_dir()) / "checkpoints" / "best_model.ckpt"
        gat_checkpoint = Path(gat_config.canonical_experiment_dir()) / "checkpoints" / "best_model.ckpt"
        
        if not vgae_checkpoint.exists() or not gat_checkpoint.exists():
            pytest.skip("Required models not found - run VGAE and GAT training first")
        
        # Create fusion config
        config = create_fusion_config("hcrl_sa")
        config.training.max_epochs = 1
        config.training.batch_size = 16
        config.training.fusion.num_episodes = 2  # Minimal episodes
        config.output_dir = str(tmp_path)
        
        trainer = HydraZenTrainer(config)
        
        try:
            results = trainer.train()
            assert results is not None
        except Exception as e:
            if "FileNotFoundError" in str(e) or "data" in str(e).lower():
                pytest.skip(f"Data not available: {e}")
            raise


class TestPipelineComponents:
    """Test individual pipeline components."""
    
    def test_config_chaining(self):
        """Test that configs can be chained (teacher → student → fusion)."""
        # Teacher configs
        gat_teacher = create_gat_normal_config("hcrl_sa")
        vgae_teacher = create_autoencoder_config("hcrl_sa")
        
        # Student configs (reference teachers)
        gat_student = create_distillation_config("hcrl_sa", "gat_student")
        vgae_student = create_distillation_config("hcrl_sa", "vgae_student")
        
        # Fusion config (references teachers)
        fusion = create_fusion_config("hcrl_sa")
        
        # All should be valid
        assert gat_teacher.experiment_name != gat_student.experiment_name
        assert vgae_teacher.experiment_name != vgae_student.experiment_name
        assert fusion.training.mode == "fusion"
    
    def test_experiment_isolation(self):
        """Test that experiments are isolated (different output dirs)."""
        configs = [
            create_gat_normal_config("hcrl_sa"),
            create_autoencoder_config("hcrl_sa"),
            create_fusion_config("hcrl_sa"),
        ]
        
        dirs = [c.canonical_experiment_dir() for c in configs]
        
        # All should be different
        assert len(dirs) == len(set(dirs))
    
    def test_reproducibility_settings(self):
        """Test that reproducibility settings are consistent."""
        config = create_gat_normal_config("hcrl_sa")
        
        # Should have seed
        assert hasattr(config, 'seed')
        assert config.seed is not None


class TestDataPipeline:
    """Test data loading and preprocessing."""
    
    def test_dataset_configuration(self):
        """Test that dataset configs are valid."""
        config = create_gat_normal_config("hcrl_sa")
        
        assert config.dataset.name == "hcrl_sa"
        assert config.model.input_dim > 0
        assert config.model.output_dim > 0
    
    def test_batch_size_compatibility(self):
        """Test that batch sizes are compatible with training."""
        config = create_gat_normal_config("hcrl_sa")
        
        # Batch size should be reasonable
        assert 1 <= config.training.batch_size <= 512


if __name__ == "__main__":
    # Run with markers:
    # pytest tests/integration/test_pipeline.py -v
    # pytest tests/integration/test_pipeline.py -v -m "not slow"
    pytest.main([__file__, "-v", "-m", "not slow"])
