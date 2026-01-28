"""
CAN-Graph Publication Visualizations Package

This package contains all visualization code for generating publication-quality
figures for the CAN-Graph paper.

Modules:
    utils: Data loading and common utilities
    architecture_diagram: System architecture visualization
    embedding_umap: Embedding space analysis
    vgae_reconstruction: VGAE reconstruction error analysis
    dqn_policy_analysis: DQN policy visualization (novel contribution)
    performance_comparison: Main results comparison
    roc_pr_curves: ROC and Precision-Recall curves
    ablation_state_space: 15D state space ablation study
    kd_impact: Knowledge distillation impact analysis
    per_attack_analysis: Per-attack-type performance breakdown
    confusion_matrices: Confusion matrix visualizations
    training_dynamics: Training curves and dynamics
    computational_cost: Efficiency analysis

Usage:
    import matplotlib.pyplot as plt
    plt.style.use('../paper_style.mplstyle')
    from visualizations.utils import load_evaluation_results
"""

__version__ = '1.0.0'
__author__ = 'CAN-Graph Team'

# Import common utilities
from .utils import (
    load_evaluation_results,
    load_dqn_predictions,
    get_color_palette,
    save_figure
)

__all__ = [
    'load_evaluation_results',
    'load_dqn_predictions',
    'get_color_palette',
    'save_figure'
]
