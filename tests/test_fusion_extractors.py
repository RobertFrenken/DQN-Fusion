"""Tests for fusion feature extractors (src/models/fusion_features.py).

Uses synthetic graphs and tiny models to verify output dimensions,
value ranges, and determinism.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import Data

from config import resolve
from src.models.fusion_features import (
    FusionFeatureExtractor,
    VGAEFusionExtractor,
    GATFusionExtractor,
)
from src.models.registry import get as registry_get
from tests.conftest import NUM_IDS, IN_CHANNELS, SMOKE_OVERRIDES, _make_graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_graph():
    """A single synthetic graph with 10 nodes, 20 edges, 11 features."""
    return _make_graph(num_nodes=10, num_edges=20, label=0)


@pytest.fixture
def tiny_vgae():
    """A small VGAE model for testing."""
    cfg = resolve("vgae", "small", **SMOKE_OVERRIDES)
    model = registry_get("vgae").factory(cfg, NUM_IDS, IN_CHANNELS)
    model.eval()
    return model


@pytest.fixture
def tiny_gat():
    """A small GAT model for testing."""
    cfg = resolve("gat", "small", **SMOKE_OVERRIDES)
    model = registry_get("gat").factory(cfg, NUM_IDS, IN_CHANNELS)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:

    def test_vgae_extractor_is_protocol(self):
        assert isinstance(VGAEFusionExtractor(), FusionFeatureExtractor)

    def test_gat_extractor_is_protocol(self):
        assert isinstance(GATFusionExtractor(), FusionFeatureExtractor)


# ---------------------------------------------------------------------------
# VGAE extractor
# ---------------------------------------------------------------------------

class TestVGAEExtractor:

    def test_output_dim(self, tiny_vgae, synthetic_graph):
        ext = VGAEFusionExtractor()
        g = synthetic_graph.clone()
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            features = ext.extract(tiny_vgae, g, batch_idx, torch.device("cpu"))
        assert features.shape == (8,)

    def test_all_finite(self, tiny_vgae, synthetic_graph):
        ext = VGAEFusionExtractor()
        g = synthetic_graph.clone()
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            features = ext.extract(tiny_vgae, g, batch_idx, torch.device("cpu"))
        assert torch.isfinite(features).all()

    def test_errors_non_negative(self, tiny_vgae, synthetic_graph):
        ext = VGAEFusionExtractor()
        g = synthetic_graph.clone()
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            features = ext.extract(tiny_vgae, g, batch_idx, torch.device("cpu"))
        # features[0:3] are error values (MSE, CE, BCE) — always >= 0
        assert (features[:3] >= 0).all()

    def test_confidence_in_range(self, tiny_vgae, synthetic_graph):
        ext = VGAEFusionExtractor()
        g = synthetic_graph.clone()
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            features = ext.extract(tiny_vgae, g, batch_idx, torch.device("cpu"))
        # features[7] = 1/(1+recon_err), always in (0, 1]
        assert 0 < features[7].item() <= 1.0

    def test_deterministic_eval_with_seed(self, tiny_vgae, synthetic_graph):
        """VGAE uses variational sampling — determinism requires seeding."""
        ext = VGAEFusionExtractor()
        g = synthetic_graph.clone()
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            torch.manual_seed(42)
            f1 = ext.extract(tiny_vgae, g.clone(), batch_idx, torch.device("cpu"))
            torch.manual_seed(42)
            f2 = ext.extract(tiny_vgae, g.clone(), batch_idx, torch.device("cpu"))
        assert torch.allclose(f1, f2)


# ---------------------------------------------------------------------------
# GAT extractor
# ---------------------------------------------------------------------------

class TestGATExtractor:

    def test_output_dim(self, tiny_gat, synthetic_graph):
        ext = GATFusionExtractor()
        g = synthetic_graph.clone()
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            features = ext.extract(tiny_gat, g, batch_idx, torch.device("cpu"))
        assert features.shape == (7,)

    def test_all_finite(self, tiny_gat, synthetic_graph):
        ext = GATFusionExtractor()
        g = synthetic_graph.clone()
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            features = ext.extract(tiny_gat, g, batch_idx, torch.device("cpu"))
        assert torch.isfinite(features).all()

    def test_probabilities_sum_to_one(self, tiny_gat, synthetic_graph):
        ext = GATFusionExtractor()
        g = synthetic_graph.clone()
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            features = ext.extract(tiny_gat, g, batch_idx, torch.device("cpu"))
        # features[0:2] are softmax probabilities
        prob_sum = features[0].item() + features[1].item()
        assert abs(prob_sum - 1.0) < 1e-5

    def test_confidence_in_range(self, tiny_gat, synthetic_graph):
        ext = GATFusionExtractor()
        g = synthetic_graph.clone()
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            features = ext.extract(tiny_gat, g, batch_idx, torch.device("cpu"))
        # features[6] = confidence = 1 - normalized_entropy, in [0, 1]
        assert 0.0 <= features[6].item() <= 1.0

    def test_deterministic_eval(self, tiny_gat, synthetic_graph):
        ext = GATFusionExtractor()
        g = synthetic_graph.clone()
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            f1 = ext.extract(tiny_gat, g.clone(), batch_idx, torch.device("cpu"))
            f2 = ext.extract(tiny_gat, g.clone(), batch_idx, torch.device("cpu"))
        assert torch.allclose(f1, f2)
