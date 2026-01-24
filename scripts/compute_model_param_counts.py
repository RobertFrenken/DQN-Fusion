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
from src.training.can_graph_module import CANGraphLightningModule


def count_parameters(model):
    return sum(int(p.numel()) for p in model.parameters() if p is not None)


def instantiate_model_from_config(cfg, model_type, num_ids=50):
    training_config = type('T', (), {'batch_size': 32, 'mode': 'normal'})()
    lm = CANGraphLightningModule(model_config=cfg, training_config=training_config, model_type=model_type, training_mode='normal', num_ids=num_ids)
    return lm.model


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
