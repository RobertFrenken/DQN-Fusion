#!/usr/bin/env python3
"""
Quick diagnostic script to test VGAE dimension configuration.

Run with: python test_vgae_dimensions.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_vgae_config():
    """Test that VGAEConfig has correct defaults."""
    print("=" * 60)
    print("Testing VGAEConfig defaults")
    print("=" * 60)

    from src.config.hydra_zen_configs import VGAEConfig

    config = VGAEConfig()

    print(f"VGAEConfig created:")
    print(f"  hidden_dims: {config.hidden_dims}")
    print(f"  latent_dim: {config.latent_dim}")
    print(f"  input_dim: {config.input_dim}")
    print(f"  embedding_dim: {config.embedding_dim}")
    print(f"  attention_heads: {config.attention_heads}")

    # Verify expected values
    assert config.hidden_dims == [1024, 512], f"Expected [1024, 512], got {config.hidden_dims}"
    assert config.latent_dim == 96, f"Expected 96, got {config.latent_dim}"
    assert config.input_dim == 11, f"Expected 11, got {config.input_dim}"
    assert config.embedding_dim == 64, f"Expected 64, got {config.embedding_dim}"

    print("\n✅ VGAEConfig defaults are correct!")
    return config


def test_vgae_construction(config):
    """Test VGAE model construction."""
    print("\n" + "=" * 60)
    print("Testing VGAE model construction")
    print("=" * 60)

    from src.models.vgae import GraphAutoencoderNeighborhood

    # Build VGAE with config parameters
    hidden_dims = list(config.hidden_dims)
    latent_dim = config.latent_dim

    print(f"\nBuilding VGAE with:")
    print(f"  num_ids: 2049 (typical for hcrl_sa)")
    print(f"  in_channels: {config.input_dim}")
    print(f"  hidden_dims: {hidden_dims}")
    print(f"  latent_dim: {latent_dim}")
    print(f"  embedding_dim: {config.embedding_dim}")
    print(f"  encoder_heads: {config.attention_heads}")
    print(f"  decoder_heads: {config.attention_heads}")

    model = GraphAutoencoderNeighborhood(
        num_ids=2049,
        in_channels=config.input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        encoder_heads=config.attention_heads,
        decoder_heads=config.attention_heads,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
        batch_norm=config.batch_norm
    )

    print(f"\n✅ VGAE model created!")
    print(f"\nEncoder layers: {len(model.encoder_layers)}")
    for i, layer in enumerate(model.encoder_layers):
        print(f"  Layer {i}: in={layer.in_channels}, out={layer.out_channels}, heads={layer.heads}")

    print(f"\nDecoder layers: {len(model.decoder_layers)}")
    for i, layer in enumerate(model.decoder_layers):
        print(f"  Layer {i}: in={layer.in_channels}, out={layer.out_channels}, heads={layer.heads}")

    print(f"\nLatent dimension: {model.latent_dim}")
    print(f"z_mean: Linear({model.latent_in_dim}, {model.latent_dim})")

    # Verify decoder first layer matches latent_dim
    first_decoder = model.decoder_layers[0]
    if first_decoder.in_channels != model.latent_dim:
        print(f"\n❌ ERROR: First decoder layer expects {first_decoder.in_channels} features")
        print(f"         but latent_dim is {model.latent_dim}!")
        return None

    print(f"\n✅ First decoder layer input ({first_decoder.in_channels}) matches latent_dim ({model.latent_dim})")

    return model


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\n" + "=" * 60)
    print("Testing forward pass")
    print("=" * 60)

    import torch
    from torch_geometric.data import Data, Batch

    # Create dummy graph data (similar to CAN data)
    num_nodes = 100
    num_edges = 200

    # x: [num_nodes, 11] - first column is CAN ID, rest are features
    x = torch.randn(num_nodes, 11)
    x[:, 0] = torch.randint(0, 2049, (num_nodes,))  # CAN IDs

    # edge_index: [2, num_edges]
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Create batch
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data])

    print(f"\nInput shape: x={batch.x.shape}, edge_index={batch.edge_index.shape}")

    try:
        # Forward pass
        cont_out, canid_logits, neighbor_logits, z, kl_loss = model(batch.x, batch.edge_index, batch.batch)

        print(f"\nForward pass successful!")
        print(f"  z shape: {z.shape}")
        print(f"  cont_out shape: {cont_out.shape}")
        print(f"  canid_logits shape: {canid_logits.shape}")
        print(f"  neighbor_logits shape: {neighbor_logits.shape}")
        print(f"  kl_loss: {kl_loss.item():.4f}")

        # Verify z has correct latent dimension
        assert z.shape[-1] == model.latent_dim, f"z has {z.shape[-1]} features, expected {model.latent_dim}"

        print(f"\n✅ Forward pass verified!")
        return True

    except Exception as e:
        print(f"\n❌ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n🔍 VGAE Dimension Diagnostic Test\n")

    # Test 1: Config defaults
    config = test_vgae_config()
    if not config:
        return 1

    # Test 2: Model construction
    model = test_vgae_construction(config)
    if not model:
        return 1

    # Test 3: Forward pass
    if not test_forward_pass(model):
        return 1

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nIf this test passes but training fails, the issue may be:")
    print("  1. Different config values passed via CLI")
    print("  2. Config serialization issues")
    print("  3. Data preprocessing issues")
    print("\nRun the training with --dry-run to see actual config values.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
