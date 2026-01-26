#!/users/PAS2022/rf15/.conda/envs/gnn-experiments/bin/python
"""
Test script for config_builder integration with CANGraphConfig.

Tests the complete flow from bucket strings to actual config objects.
"""

import sys
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Import config_builder directly without triggering src/__init__.py
config_builder_path = project_root / 'src' / 'cli' / 'config_builder.py'
spec = importlib.util.spec_from_file_location("config_builder", config_builder_path)
config_builder = importlib.util.module_from_spec(spec)
sys.modules['config_builder'] = config_builder
spec.loader.exec_module(config_builder)

parse_bucket = config_builder.parse_bucket
expand_sweep = config_builder.expand_sweep
build_config_from_buckets = config_builder.build_config_from_buckets
create_can_graph_config = config_builder.create_can_graph_config
format_config_summary = config_builder.format_config_summary


def test_simple_config():
    """Test creating a simple GAT normal config."""
    print("\n" + "="*70)
    print("TEST 1: Simple GAT Normal Config")
    print("="*70)

    run_type_str = "model=gat,model_size=teacher,dataset=hcrl_ch,mode=normal"
    model_args_str = "epochs=100,learning_rate=0.001"
    slurm_args_str = "walltime=12:00:00,memory=128G"

    configs = build_config_from_buckets(run_type_str, model_args_str, slurm_args_str)

    assert len(configs) == 1, f"Expected 1 config, got {len(configs)}"

    run_type, model_args, slurm_args = configs[0]
    config = create_can_graph_config(run_type, model_args, slurm_args)

    print(f"✓ Created config: {config.experiment_name}")
    print(f"  Model type: {config.model.type}")
    print(f"  Dataset: {config.dataset.name}")
    print(f"  Training mode: {config.training.mode}")
    print(f"  Max epochs: {config.training.max_epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  SLURM args attached: {hasattr(config, '_slurm_args')}")

    # Verify overrides applied
    assert config.training.max_epochs == 100, f"Expected epochs=100, got {config.training.max_epochs}"
    assert config.training.learning_rate == 0.001, f"Expected lr=0.001, got {config.training.learning_rate}"
    assert config._slurm_args['walltime'] == '12:00:00'

    print("✓ All assertions passed!")


def test_vgae_autoencoder():
    """Test creating VGAE autoencoder config."""
    print("\n" + "="*70)
    print("TEST 2: VGAE Autoencoder Config")
    print("="*70)

    run_type_str = "model=vgae,model_size=teacher,dataset=set_01,mode=autoencoder"
    model_args_str = "latent_dim=32,epochs=50"
    slurm_args_str = ""

    configs = build_config_from_buckets(run_type_str, model_args_str, slurm_args_str)
    run_type, model_args, slurm_args = configs[0]
    config = create_can_graph_config(run_type, model_args, slurm_args)

    print(f"✓ Created config: {config.experiment_name}")
    print(f"  Model type: {config.model.type}")
    print(f"  Latent dim: {config.model.latent_dim}")
    print(f"  Training mode: {config.training.mode}")

    # Verify VGAE-specific override
    assert config.model.latent_dim == 32, f"Expected latent_dim=32, got {config.model.latent_dim}"
    assert config.model.type == "vgae"
    assert config.training.mode == "autoencoder"

    print("✓ All assertions passed!")


def test_student_distillation():
    """Test creating student GAT with distillation mode."""
    print("\n" + "="*70)
    print("TEST 3: Student GAT Distillation Config")
    print("="*70)

    run_type_str = "model=gat,model_size=student,dataset=hcrl_sa,mode=distillation"
    model_args_str = "hidden_channels=32,dropout=0.1"
    slurm_args_str = "gpus=2"

    configs = build_config_from_buckets(run_type_str, model_args_str, slurm_args_str)
    run_type, model_args, slurm_args = configs[0]
    config = create_can_graph_config(run_type, model_args, slurm_args)

    print(f"✓ Created config: {config.experiment_name}")
    print(f"  Model type: {config.model.type}")
    print(f"  Hidden channels: {config.model.hidden_channels}")
    print(f"  Dropout: {config.model.dropout}")
    print(f"  Training mode: {config.training.mode}")

    # Verify student config
    assert config.model.type == "gat_student", f"Expected gat_student, got {config.model.type}"
    assert config.model.hidden_channels == 32
    assert config.model.dropout == 0.1
    assert config.training.mode == "knowledge_distillation"

    print("✓ All assertions passed!")


def test_dataset_sweep():
    """Test dataset sweep expansion."""
    print("\n" + "="*70)
    print("TEST 4: Dataset Sweep (2 configs)")
    print("="*70)

    run_type_str = "model=gat,model_size=teacher,dataset=[hcrl_ch,hcrl_sa],mode=curriculum"
    model_args_str = ""
    slurm_args_str = ""

    configs = build_config_from_buckets(run_type_str, model_args_str, slurm_args_str)

    print(f"✓ Generated {len(configs)} configs from sweep")
    assert len(configs) == 2, f"Expected 2 configs, got {len(configs)}"

    for idx, (run_type, model_args, slurm_args) in enumerate(configs):
        config = create_can_graph_config(run_type, model_args, slurm_args)
        print(f"  Config {idx+1}: {config.experiment_name}")

    # Verify both datasets present
    datasets = [configs[i][0]['dataset'] for i in range(len(configs))]
    assert 'hcrl_ch' in datasets
    assert 'hcrl_sa' in datasets

    print("✓ All assertions passed!")


def test_hyperparameter_sweep():
    """Test multi-dimensional hyperparameter sweep."""
    print("\n" + "="*70)
    print("TEST 5: Hyperparameter Sweep (4 configs)")
    print("="*70)

    run_type_str = "model=gat,model_size=teacher,dataset=set_02,mode=normal"
    model_args_str = "learning_rate=[0.001,0.003],batch_size=[32,64]"
    slurm_args_str = ""

    configs = build_config_from_buckets(run_type_str, model_args_str, slurm_args_str)

    print(f"✓ Generated {len(configs)} configs from sweep")
    assert len(configs) == 4, f"Expected 4 configs (2 lr × 2 batch), got {len(configs)}"

    lrs = []
    batches = []
    for idx, (run_type, model_args, slurm_args) in enumerate(configs):
        config = create_can_graph_config(run_type, model_args, slurm_args)
        print(f"  Config {idx+1}: lr={config.training.learning_rate}, batch={config.training.batch_size}")
        lrs.append(config.training.learning_rate)
        batches.append(config.training.batch_size)

    # Verify Cartesian product
    assert 0.001 in lrs and 0.003 in lrs
    assert 32 in batches and 64 in batches

    print("✓ All assertions passed!")


def test_config_summary_formatting():
    """Test the config summary output."""
    print("\n" + "="*70)
    print("TEST 6: Config Summary Formatting")
    print("="*70)

    run_type_str = "model=vgae,model_size=student,dataset=set_03,mode=autoencoder"
    model_args_str = "epochs=200,latent_dim=8"
    slurm_args_str = "walltime=24:00:00,memory=256G,gpus=4"

    configs = build_config_from_buckets(run_type_str, model_args_str, slurm_args_str)
    run_type, model_args, slurm_args = configs[0]

    summary = format_config_summary(run_type, model_args, slurm_args)
    print(summary)

    # Verify summary contains key info
    assert 'vgae' in summary
    assert 'set_03' in summary
    assert 'autoencoder' in summary
    assert '200' in summary  # epochs
    assert '24:00:00' in summary  # walltime

    print("✓ Summary formatting works!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CONFIG BUILDER INTEGRATION TESTS")
    print("="*70)

    try:
        test_simple_config()
        test_vgae_autoencoder()
        test_student_distillation()
        test_dataset_sweep()
        test_hyperparameter_sweep()
        test_config_summary_formatting()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nConfigManager integration complete. Ready for:")
        print("  1. Pre-flight Validator")
        print("  2. JobManager for SLURM submission")
        print("  3. Execution Router")
        print("="*70)

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
