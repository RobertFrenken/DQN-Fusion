"""Shared fixtures for training smoke and e2e tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data import Data


NUM_IDS = 20
IN_CHANNELS = 11


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_graph(num_nodes=10, num_edges=20, label=0):
    """Create a single synthetic graph matching real data shape."""
    x = torch.randn(num_nodes, IN_CHANNELS)
    x[:, 0] = torch.randint(0, NUM_IDS, (num_nodes,)).float()
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.tensor([label])
    return Data(x=x, edge_index=edge_index, y=y)


def _make_dataset(n=50):
    """Create a small dataset with mix of normal/attack graphs."""
    return [_make_graph(label=i % 2) for i in range(n)]


# ---------------------------------------------------------------------------
# Shared smoke-test config overrides
# ---------------------------------------------------------------------------

SMOKE_OVERRIDES = dict(
    max_epochs=2,
    batch_size=16,
    device="cpu",
    precision="32-true",
    patience=2,
    optimize_batch_size=False,
    num_workers=0,
    gradient_checkpointing=False,
    log_every_n_steps=1,
    safety_factor=1.0,
    mp_start_method="fork",
)

# E2E tests need tiny architectures to finish in reasonable time on CPU
E2E_OVERRIDES = dict(
    **SMOKE_OVERRIDES,
    # Tiny VGAE
    vgae_hidden_dims=(32, 16, 8), vgae_latent_dim=8,
    vgae_heads=1, vgae_embedding_dim=4, vgae_dropout=0.1,
    # Tiny GAT
    gat_hidden=8, gat_layers=2, gat_heads=2,
    gat_embedding_dim=4, gat_fc_layers=2, gat_dropout=0.1,
    # Tiny DQN
    dqn_hidden=32, dqn_layers=2,
    dqn_buffer_size=500, dqn_batch_size=32, dqn_target_update=10,
)


# ---------------------------------------------------------------------------
# MLflow mock (autouse in smoke/e2e tests)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    """Disable all MLflow calls during tests."""
    mock = MagicMock()
    monkeypatch.setattr("mlflow.start_run", mock)
    monkeypatch.setattr("mlflow.end_run", mock)
    monkeypatch.setattr("mlflow.log_metric", mock)
    monkeypatch.setattr("mlflow.log_metrics", mock)
    monkeypatch.setattr("mlflow.log_param", mock)
    monkeypatch.setattr("mlflow.log_params", mock)
    monkeypatch.setattr("mlflow.log_artifact", mock)
    monkeypatch.setattr("mlflow.set_tag", mock)
    monkeypatch.setattr("mlflow.set_tags", mock)
    monkeypatch.setattr("mlflow.set_experiment", mock)
    monkeypatch.setattr("mlflow.pytorch.autolog", mock)
    return mock
