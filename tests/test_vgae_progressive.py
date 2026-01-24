import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import importlib


def test_vgae_progressive_instantiation():
    # Require real torch in the test environment (run via scripts/run_tests_in_conda.sh)
    torch = pytest.importorskip('torch')

    # Define minimal config stubs to avoid importing heavy project config modules
    class VGAEConfig:
        type = 'vgae'
        input_dim = 11
        hidden_dims = [256, 128, 96, 48]
        latent_dim = 48
        attention_heads = 8
        dropout = 0.15
        batch_norm = True
        embedding_dim = 32

    class StudentVGAEConfig:
        type = 'vgae_student'
        input_dim = 11
        encoder_dims = [128, 64, 24]
        latent_dim = 24
        attention_heads = 2
        dropout = 0.1
        batch_norm = True
        embedding_dim = 8
    from src.training.lightning_modules import CANGraphLightningModule

    vcfg = VGAEConfig()
    teacher = CANGraphLightningModule(model_config=vcfg, training_config=type('T', (), {'batch_size':32, 'mode':'autoencoder'})(), model_type='vgae', training_mode='autoencoder', num_ids=50).model
    # VGAEConfig.hidden_dims default [256,128,96,48] => encoder targets length = 3
    assert len(teacher.encoder_layers) == 3
    assert len(teacher.decoder_layers) == 3
    assert teacher.latent_dim == vcfg.latent_dim

    scfg = StudentVGAEConfig()
    student = CANGraphLightningModule(model_config=scfg, training_config=type('T', (), {'batch_size':32, 'mode':'autoencoder'})(), model_type='vgae_student', training_mode='autoencoder', num_ids=50).model
    # Student encoder_dims default [128,64,24] => encoder targets length = 2
    assert len(student.encoder_layers) == 2
    assert len(student.decoder_layers) == 2
    assert student.latent_dim == scfg.latent_dim
