"""Preprocessing module for CAN-Graph.

New architecture (Phase 3):
    schema.py       — IR column layout and validation
    vocabulary.py   — EntityVocabulary (ID ↔ dense index)
    engine.py       — Domain-agnostic graph construction
    dataset.py      — GraphDataset wrapper
    adapters/       — Domain-specific adapters (CAN bus, network flow)

Legacy:
    preprocessing.py — Monolithic predecessor (still used by datamodules.py)
"""
from .dataset import GraphDataset
from .schema import IRSchema, CAN_BUS_SCHEMA
from .vocabulary import EntityVocabulary
from .engine import GraphEngine

__all__ = [
    "GraphDataset",
    "IRSchema",
    "CAN_BUS_SCHEMA",
    "EntityVocabulary",
    "GraphEngine",
]
