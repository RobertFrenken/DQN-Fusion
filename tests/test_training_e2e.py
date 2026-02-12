"""End-to-end tests for training stages.

Tests the full train_autoencoder -> train_curriculum -> train_fusion pipeline
with synthetic data on CPU. Verifies that checkpoints and configs are saved
correctly and that downstream stages can load upstream outputs.

Run:  python -m pytest tests/test_training_e2e.py -v -m "not slow"
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from tests.conftest import _make_dataset, NUM_IDS, IN_CHANNELS, E2E_OVERRIDES


def _patch_load_data(data):
    """Return a monkeypatch-ready load_data that returns synthetic data."""
    def _fake_load_data(cfg):
        train = data[:35]
        val = data[35:]
        return train, val, NUM_IDS, IN_CHANNELS
    return _fake_load_data


@pytest.fixture()
def synth_data():
    return _make_dataset(50)


@pytest.fixture()
def exp_root(tmp_path):
    return str(tmp_path / "experimentruns")


def _apply_load_data_patches(stack, data):
    """Patch load_data in all modules that import it."""
    fake = _patch_load_data(data)
    stack.enter_context(patch("pipeline.stages.training.load_data", fake))
    stack.enter_context(patch("pipeline.stages.fusion.load_data", fake))
    stack.enter_context(patch("pipeline.stages.evaluation.load_data", fake))


class TestAutoencoderE2E:
    """train_autoencoder produces checkpoint + config that load correctly."""

    def test_autoencoder_e2e(self, synth_data, exp_root):
        from pipeline.config import PipelineConfig
        from pipeline.stages.training import train_autoencoder
        from pipeline.paths import config_path

        cfg = PipelineConfig.from_preset(
            "vgae", "teacher",
            dataset="test_ds", experiment_root=exp_root,
            **E2E_OVERRIDES,
        )

        with ExitStack() as stack:
            _apply_load_data_patches(stack, synth_data)
            ckpt = train_autoencoder(cfg)

        assert ckpt.exists(), "Checkpoint not saved"
        assert config_path(cfg, "autoencoder").exists(), "Config not saved"

        # Verify checkpoint loads back
        loaded_cfg = PipelineConfig.load(config_path(cfg, "autoencoder"))
        from src.models.vgae import GraphAutoencoderNeighborhood
        model = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_dims=list(loaded_cfg.vgae_hidden_dims),
            latent_dim=loaded_cfg.vgae_latent_dim,
            encoder_heads=loaded_cfg.vgae_heads,
            embedding_dim=loaded_cfg.vgae_embedding_dim,
            dropout=loaded_cfg.vgae_dropout,
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))


class TestCurriculumE2E:
    """train_curriculum loads VGAE, trains GAT, saves checkpoint."""

    def test_curriculum_e2e(self, synth_data, exp_root):
        from pipeline.config import PipelineConfig
        from pipeline.stages.training import train_autoencoder, train_curriculum
        from pipeline.paths import config_path

        vgae_cfg = PipelineConfig.from_preset(
            "vgae", "teacher",
            dataset="test_ds", experiment_root=exp_root,
            **E2E_OVERRIDES,
        )
        gat_cfg = PipelineConfig.from_preset(
            "gat", "teacher",
            dataset="test_ds", experiment_root=exp_root,
            **E2E_OVERRIDES,
        )

        with ExitStack() as stack:
            _apply_load_data_patches(stack, synth_data)
            train_autoencoder(vgae_cfg)
            ckpt = train_curriculum(gat_cfg)

        assert ckpt.exists(), "GAT checkpoint not saved"
        assert config_path(gat_cfg, "curriculum").exists(), "GAT config not saved"

        loaded_cfg = PipelineConfig.load(config_path(gat_cfg, "curriculum"))
        from src.models.gat import GATWithJK
        model = GATWithJK(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_channels=loaded_cfg.gat_hidden, out_channels=2,
            num_layers=loaded_cfg.gat_layers, heads=loaded_cfg.gat_heads,
            dropout=loaded_cfg.gat_dropout,
            num_fc_layers=getattr(loaded_cfg, 'gat_fc_layers', 3),
            embedding_dim=loaded_cfg.gat_embedding_dim,
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))


class TestFusionE2E:
    """train_fusion loads VGAE+GAT, trains DQN, saves checkpoint."""

    def test_fusion_e2e(self, synth_data, exp_root):
        from pipeline.config import PipelineConfig
        from pipeline.stages.training import train_autoencoder, train_curriculum
        from pipeline.stages.fusion import train_fusion
        from pipeline.paths import config_path

        vgae_cfg = PipelineConfig.from_preset(
            "vgae", "teacher",
            dataset="test_ds", experiment_root=exp_root,
            **E2E_OVERRIDES,
        )
        gat_cfg = PipelineConfig.from_preset(
            "gat", "teacher",
            dataset="test_ds", experiment_root=exp_root,
            **E2E_OVERRIDES,
        )
        dqn_cfg = PipelineConfig.from_preset(
            "dqn", "teacher",
            dataset="test_ds", experiment_root=exp_root,
            fusion_episodes=5, episode_sample_size=20,
            fusion_max_samples=50, max_val_samples=15,
            gpu_training_steps=2,
            **E2E_OVERRIDES,
        )

        with ExitStack() as stack:
            _apply_load_data_patches(stack, synth_data)
            train_autoencoder(vgae_cfg)
            train_curriculum(gat_cfg)
            ckpt = train_fusion(dqn_cfg)

        assert ckpt.exists(), "DQN checkpoint not saved"
        assert config_path(dqn_cfg, "fusion").exists(), "DQN config not saved"

        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        assert "q_network" in sd


@pytest.mark.slow
class TestFullPipelineE2E:
    """Full 3-stage pipeline + evaluation in sequence."""

    def test_full_pipeline(self, synth_data, exp_root):
        from pipeline.config import PipelineConfig
        from pipeline.stages.training import train_autoencoder, train_curriculum
        from pipeline.stages.fusion import train_fusion
        from pipeline.stages.evaluation import evaluate
        from pipeline.paths import stage_dir

        overrides = dict(
            dataset="test_ds", experiment_root=exp_root,
            fusion_episodes=3, episode_sample_size=20,
            fusion_max_samples=50, max_val_samples=15,
            gpu_training_steps=2,
            **E2E_OVERRIDES,
        )

        with ExitStack() as stack:
            _apply_load_data_patches(stack, synth_data)
            train_autoencoder(PipelineConfig.from_preset("vgae", "teacher", **overrides))
            train_curriculum(PipelineConfig.from_preset("gat", "teacher", **overrides))
            train_fusion(PipelineConfig.from_preset("dqn", "teacher", **overrides))

            eval_cfg = PipelineConfig(
                dataset="test_ds", model_size="teacher",
                experiment_root=exp_root, **E2E_OVERRIDES,
            )

            stack.enter_context(
                patch("pipeline.stages.evaluation._load_test_data", return_value={})
            )
            metrics = evaluate(eval_cfg)

        assert "gat" in metrics, "GAT metrics missing"
        assert "vgae" in metrics, "VGAE metrics missing"
        assert "fusion" in metrics, "Fusion metrics missing"
        assert metrics["gat"]["core"]["accuracy"] >= 0.0
        assert (stage_dir(eval_cfg, "evaluation") / "metrics.json").exists()
