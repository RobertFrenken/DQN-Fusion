"""Module-level smoke tests for VGAE, GAT, and DQN training.

Fast tests (< 15s each) that verify models train for 2 epochs without crashing,
produce finite losses, and respect config parameters.

Run:  python -m pytest tests/test_training_smoke.py -v -m "not slow"
"""
from __future__ import annotations

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from tests.conftest import _make_graph, _make_dataset, NUM_IDS, IN_CHANNELS, SMOKE_OVERRIDES


class TestVGAESmoke:
    """VGAE module trains and produces finite loss."""

    def test_trains(self):
        from pipeline.config import PipelineConfig
        from pipeline.stages.modules import VGAEModule

        cfg = PipelineConfig.from_preset("vgae", "teacher", **SMOKE_OVERRIDES)
        data = _make_dataset(20)
        module = VGAEModule(cfg, NUM_IDS, IN_CHANNELS)

        trainer = pl.Trainer(
            max_epochs=2, accelerator="cpu",
            enable_checkpointing=False, logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(module, DataLoader(data[:15], batch_size=4), DataLoader(data[15:], batch_size=4))

        assert trainer.callback_metrics.get("train_loss") is not None
        assert torch.isfinite(trainer.callback_metrics["train_loss"])

    def test_kd_trains(self):
        from pipeline.config import PipelineConfig
        from pipeline.stages.modules import VGAEModule
        from pipeline.stages.utils import make_projection
        from src.models.vgae import GraphAutoencoderNeighborhood

        teacher_cfg = PipelineConfig.from_preset("vgae", "teacher", **SMOKE_OVERRIDES)
        student_cfg = PipelineConfig.from_preset(
            "vgae", "student", **SMOKE_OVERRIDES,
            use_kd=True, kd_alpha=0.5,
        )

        teacher = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_dims=list(teacher_cfg.vgae_hidden_dims),
            latent_dim=teacher_cfg.vgae_latent_dim,
            encoder_heads=teacher_cfg.vgae_heads,
            embedding_dim=teacher_cfg.vgae_embedding_dim,
            dropout=teacher_cfg.vgae_dropout,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student_model = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_dims=list(student_cfg.vgae_hidden_dims),
            latent_dim=student_cfg.vgae_latent_dim,
            encoder_heads=student_cfg.vgae_heads,
            embedding_dim=student_cfg.vgae_embedding_dim,
            dropout=student_cfg.vgae_dropout,
        )
        projection = make_projection(student_model, teacher, "vgae", torch.device("cpu"))
        del student_model

        module = VGAEModule(student_cfg, NUM_IDS, IN_CHANNELS, teacher=teacher, projection=projection)
        data = _make_dataset(20)

        trainer = pl.Trainer(
            max_epochs=2, accelerator="cpu",
            enable_checkpointing=False, logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(module, DataLoader(data[:15], batch_size=4), DataLoader(data[15:], batch_size=4))

        assert torch.isfinite(trainer.callback_metrics["train_loss"])


class TestGATSmoke:
    """GAT module trains and produces finite loss."""

    def test_trains(self):
        from pipeline.config import PipelineConfig
        from pipeline.stages.modules import GATModule

        cfg = PipelineConfig.from_preset("gat", "teacher", **SMOKE_OVERRIDES)
        data = _make_dataset(20)
        module = GATModule(cfg, NUM_IDS, IN_CHANNELS)

        trainer = pl.Trainer(
            max_epochs=2, accelerator="cpu",
            enable_checkpointing=False, logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(module, DataLoader(data[:15], batch_size=4), DataLoader(data[15:], batch_size=4))

        assert trainer.callback_metrics.get("train_loss") is not None
        assert torch.isfinite(trainer.callback_metrics["train_loss"])

    def test_kd_trains(self):
        from pipeline.config import PipelineConfig
        from pipeline.stages.modules import GATModule
        from src.models.gat import GATWithJK

        teacher_cfg = PipelineConfig.from_preset("gat", "teacher", **SMOKE_OVERRIDES)
        student_cfg = PipelineConfig.from_preset(
            "gat", "student", **SMOKE_OVERRIDES,
            use_kd=True, kd_alpha=0.5,
        )

        teacher = GATWithJK(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
            hidden_channels=teacher_cfg.gat_hidden, out_channels=2,
            num_layers=teacher_cfg.gat_layers, heads=teacher_cfg.gat_heads,
            dropout=teacher_cfg.gat_dropout,
            num_fc_layers=teacher_cfg.gat_fc_layers,
            embedding_dim=teacher_cfg.gat_embedding_dim,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        module = GATModule(student_cfg, NUM_IDS, IN_CHANNELS, teacher=teacher)
        data = _make_dataset(20)

        trainer = pl.Trainer(
            max_epochs=2, accelerator="cpu",
            enable_checkpointing=False, logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(module, DataLoader(data[:15], batch_size=4), DataLoader(data[15:], batch_size=4))

        assert torch.isfinite(trainer.callback_metrics["train_loss"])

    def test_fc_layers_config(self):
        """GATWithJK should work with different num_fc_layers values."""
        from src.models.gat import GATWithJK

        g = _make_graph()
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)

        for fc_layers in [1, 2, 3]:
            model = GATWithJK(
                num_ids=NUM_IDS, in_channels=IN_CHANNELS,
                hidden_channels=24, out_channels=2,
                num_layers=2, heads=4, dropout=0.1,
                num_fc_layers=fc_layers, embedding_dim=8,
            )
            model.eval()
            with torch.no_grad():
                out = model(g)
            assert out.shape == (1, 2), f"fc_layers={fc_layers} gave wrong shape: {out.shape}"


class TestDQNSmoke:
    """DQN trains and produces finite loss."""

    def test_trains(self):
        import numpy as np
        from src.models.dqn import EnhancedDQNFusionAgent

        agent = EnhancedDQNFusionAgent(
            alpha_steps=21, lr=1e-3, gamma=0.99,
            epsilon=0.5, epsilon_decay=0.99, min_epsilon=0.01,
            buffer_size=500, batch_size=32,
            target_update_freq=10, device="cpu",
            hidden_dim=64, num_layers=2,
        )

        # Fill replay buffer
        for _ in range(100):
            state = np.random.randn(15).astype(np.float32)
            alpha, action_idx, proc_state = agent.select_action(state, training=True)
            reward = 1.0 if np.random.random() > 0.5 else -1.0
            agent.store_experience(proc_state, action_idx, reward, proc_state, False)

        # Train a few steps
        losses = []
        for _ in range(10):
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        assert len(losses) > 0, "DQN did not produce any training losses"
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), "DQN loss is not finite"
