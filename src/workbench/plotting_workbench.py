"""Temporary plotting workbench for rapid iteration on visualization helpers.

This module re-exports selected plotting utilities from `src.utils.plotting_utils`
and adds small harnesses to run quick demos and save plots to a workbench output
folder (outputs/workbench_plots).

Keep this file small and experimental; we can iterate quickly here without
changing the main plotting utilities used in production pipelines.
"""

from pathlib import Path
from typing import Optional

# Re-export commonly used plotting helpers
from src.utils.plotting_utils import (
    plot_latent_space,
    plot_feature_histograms,
    plot_recon_error_hist,
    plot_fusion_score_distributions,
    plot_node_recon_errors,
    plot_graph_reconstruction,
)

WORKBENCH_DIR = Path.cwd() / "outputs" / "workbench_plots"
WORKBENCH_DIR.mkdir(parents=True, exist_ok=True)


def save_latent_space_demo(latent_vectors, labels, name: Optional[str] = None):
    save_path = WORKBENCH_DIR / (name or "latent_space_demo.png")
    plot_latent_space(latent_vectors, labels, save_path=save_path)
    print(f"Saved latent space demo: {save_path}")
    return save_path


def save_recon_hist_demo(errors_normal, errors_attack, threshold, name: Optional[str] = None):
    save_path = WORKBENCH_DIR / (name or "recon_hist_demo.png")
    plot_recon_error_hist(errors_normal, errors_attack, threshold, save_path=save_path)
    print(f"Saved recon histogram demo: {save_path}")
    return save_path


def save_fusion_score_demo(anomaly_scores, gat_probs, labels, name: Optional[str] = None):
    save_path = WORKBENCH_DIR / (name or "fusion_score_demo.png")
    plot_fusion_score_distributions(anomaly_scores, gat_probs, labels, save_path=save_path)
    print(f"Saved fusion score demo: {save_path}")
    return save_path


if __name__ == "__main__":
    # Quick smoke test: create dummy data to ensure plotting functions run
    import numpy as np

    print("Running plotting workbench smoke tests...")

    # Latent demo
    latent = np.random.randn(200, 2)
    labels = np.random.randint(0, 2, size=(200,))
    save_latent_space_demo(latent, labels, name="latent_demo.png")

    # Recon hist demo
    errors_normal = np.random.gamma(2.0, 0.5, size=500)
    errors_attack = np.random.gamma(5.0, 0.7, size=100)
    save_recon_hist_demo(errors_normal, errors_attack, threshold=3.0, name="recon_demo.png")

    # Fusion score demo
    anomaly_scores = np.concatenate([np.random.normal(0.2, 0.05, size=400), np.random.normal(0.8, 0.1, size=100)])
    gat_probs = np.concatenate([np.random.beta(2, 8, size=400), np.random.beta(8, 2, size=100)])
    labels = np.concatenate([np.zeros(400), np.ones(100)])
    save_fusion_score_demo(anomaly_scores, gat_probs, labels, name="fusion_demo.png")

    print("Workbench demo complete. Files saved to:", WORKBENCH_DIR)
