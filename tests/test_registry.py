"""Tests for the model registry (src/models/registry.py).

Validates default registrations, public API, and factory integration.
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from graphids.core.models.registry import (
    ModelEntry, register, get, fusion_state_dim, extractors, _REGISTRY,
)
from graphids.core.models.fusion_features import (
    FusionFeatureExtractor, VGAEFusionExtractor, GATFusionExtractor,
)


class TestDefaultRegistrations:
    """Verify the three default registrations exist and are well-formed."""

    def test_vgae_registered(self):
        entry = get("vgae")
        assert entry.model_type == "vgae"
        assert entry.extractor is not None

    def test_gat_registered(self):
        entry = get("gat")
        assert entry.model_type == "gat"
        assert entry.extractor is not None

    def test_dqn_registered(self):
        entry = get("dqn")
        assert entry.model_type == "dqn"
        assert entry.extractor is None

    def test_nonexistent_raises_keyerror(self):
        with pytest.raises(KeyError, match="not registered"):
            get("nonexistent")


class TestFusionDimensions:
    """Verify fusion state dimensions match the expected 15-D layout."""

    def test_fusion_state_dim_is_15(self):
        assert fusion_state_dim() == 15

    def test_vgae_extractor_is_8d(self):
        assert get("vgae").extractor.feature_dim == 8

    def test_gat_extractor_is_7d(self):
        assert get("gat").extractor.feature_dim == 7


class TestExtractors:
    """Verify extractor ordering and types."""

    def test_extractors_order_vgae_before_gat(self):
        pairs = extractors()
        names = [name for name, _ in pairs]
        assert names == ["vgae", "gat"]

    def test_dqn_excluded_from_extractors(self):
        names = [name for name, _ in extractors()]
        assert "dqn" not in names

    def test_extractors_are_protocol_compliant(self):
        for _, ext in extractors():
            assert isinstance(ext, FusionFeatureExtractor)


class TestFactory:
    """Verify factory functions produce nn.Module instances."""

    def test_vgae_factory(self):
        from graphids.config import resolve
        from tests.conftest import NUM_IDS, IN_CHANNELS, SMOKE_OVERRIDES

        cfg = resolve("vgae", "large", **SMOKE_OVERRIDES)
        model = get("vgae").factory(cfg, NUM_IDS, IN_CHANNELS)
        assert isinstance(model, nn.Module)

    def test_gat_factory(self):
        from graphids.config import resolve
        from tests.conftest import NUM_IDS, IN_CHANNELS, SMOKE_OVERRIDES

        cfg = resolve("gat", "large", **SMOKE_OVERRIDES)
        model = get("gat").factory(cfg, NUM_IDS, IN_CHANNELS)
        assert isinstance(model, nn.Module)

    def test_dqn_factory(self):
        from graphids.config import resolve
        from tests.conftest import SMOKE_OVERRIDES

        cfg = resolve("dqn", "large", **SMOKE_OVERRIDES)
        model = get("dqn").factory(cfg, 0, 0)
        assert isinstance(model, nn.Module)
