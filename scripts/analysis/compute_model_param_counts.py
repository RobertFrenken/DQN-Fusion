#!/usr/bin/env python3
"""Compute parameter counts for each model type using project configs.

Usage: scripts/compute_model_param_counts.py

Runs for: GAT teacher/student, VGAE teacher/student, DQN teacher/student
Prints parameter counts and compares to config target_parameters.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import hydra_zen_configs as cfg_mod
from src.models.vgae import GraphAutoencoderNeighborhood
from src.models.models import GATWithJK, create_dqn_teacher, create_dqn_student


def count_parameters(model):
    return sum(int(p.numel()) for p in model.parameters() if p is not None)


def instantiate_model_from_config(cfg, model_type, num_ids=50):
    """Instantiate model directly from config."""
    if model_type in ['vgae', 'vgae_student']:
        hidden_dims = getattr(cfg, 'hidden_dims', None) or getattr(cfg, 'encoder_dims', None)
        return GraphAutoencoderNeighborhood(
            num_ids=num_ids,
            in_channels=cfg.input_dim,
            hidden_dims=list(hidden_dims) if hidden_dims else None,
            latent_dim=cfg.latent_dim,
            encoder_heads=cfg.attention_heads,
            decoder_heads=cfg.attention_heads,
            embedding_dim=cfg.embedding_dim,
            dropout=cfg.dropout,
            batch_norm=getattr(cfg, 'batch_norm', True)
        )
    elif model_type in ['gat', 'gat_student']:
        return GATWithJK(
            num_ids=num_ids,
            in_channels=cfg.input_dim,
            hidden_channels=cfg.hidden_channels,
            out_channels=cfg.output_dim,
            num_layers=cfg.num_layers,
            heads=cfg.heads,
            dropout=cfg.dropout,
            num_fc_layers=cfg.num_fc_layers,
            embedding_dim=cfg.embedding_dim
        )
    elif model_type == 'dqn':
        return create_dqn_teacher(cfg, num_ids=num_ids)
    elif model_type == 'dqn_student':
        return create_dqn_student(cfg, num_ids=num_ids)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def print_report():
    pairs = [
        (cfg_mod.GATConfig, 'gat'),
        (cfg_mod.StudentGATConfig, 'gat_student'),
        (cfg_mod.VGAEConfig, 'vgae'),
        (cfg_mod.StudentVGAEConfig, 'vgae_student'),
        (cfg_mod.DQNConfig, 'dqn'),
        (cfg_mod.StudentDQNConfig, 'dqn_student'),
    ]

    for cfg_cls, model_type in pairs:
        cfg = cfg_cls()
        model = instantiate_model_from_config(cfg, model_type)
        total = count_parameters(model)
        target = getattr(cfg, 'target_parameters', None)
        print(f"{model_type:12s}: params={total:10d}", end='')
        if target is not None:
            pct = (total / target) * 100 if target > 0 else float('inf')
            print(f"  target={target:10d}  ({pct:6.2f}%)")
        else:
            print("  (no target specified)")


if __name__ == '__main__':
    print_report()
