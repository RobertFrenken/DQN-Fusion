"""Fusion stage: combines VGAE + GAT predictions via configurable method (DQN, MLP, weighted_avg)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import pytorch_lightning as pl

from config import PipelineConfig, stage_dir, checkpoint_path, config_path
from .utils import load_data, load_model, cache_predictions, cleanup

log = logging.getLogger(__name__)


def _train_dqn_fusion(cfg, train_cache, val_cache, device, out) -> float:
    """Original DQN RL fusion training loop. Returns best validation accuracy."""
    from src.models.dqn import EnhancedDQNFusionAgent
    from src.models.registry import fusion_state_dim

    agent = EnhancedDQNFusionAgent(
        lr=cfg.fusion.lr,
        gamma=cfg.dqn.gamma,
        epsilon=cfg.dqn.epsilon,
        epsilon_decay=cfg.dqn.epsilon_decay,
        min_epsilon=cfg.dqn.min_epsilon,
        buffer_size=cfg.dqn.buffer_size,
        batch_size=cfg.dqn.batch_size,
        target_update_freq=cfg.dqn.target_update,
        device=str(device),
        state_dim=fusion_state_dim(),
        hidden_dim=cfg.dqn.hidden,
        num_layers=cfg.dqn.layers,
    )

    best_acc = 0.0

    for ep in range(cfg.fusion.episodes):
        idx = torch.randperm(len(train_cache["states"]))[: cfg.fusion.episode_sample_size]
        batch_states = train_cache["states"][idx]
        batch_labels = train_cache["labels"][idx]

        total_reward = 0.0
        for i in range(len(batch_states)):
            state_np = batch_states[i].numpy()
            alpha, action_idx, proc_state = agent.select_action(state_np, training=True)
            pred = 1 if alpha > 0.5 else 0
            reward = agent.compute_fusion_reward(
                prediction=pred,
                true_label=batch_labels[i].item(),
                state_features=state_np,
                alpha=alpha,
            )
            agent.store_experience(proc_state, action_idx, reward, proc_state, False)
            total_reward += reward

        if len(agent.replay_buffer) >= cfg.dqn.batch_size:
            for _ in range(cfg.fusion.gpu_training_steps):
                agent.train_step()

        if (ep + 1) % 50 == 0:
            val_pairs = [
                (val_cache["states"][i].numpy(), val_cache["labels"][i].item())
                for i in range(min(5000, len(val_cache["states"])))
            ]
            metrics = agent.validate_agent(val_pairs, num_samples=len(val_pairs))
            acc = metrics.get("accuracy", 0)
            log.info(
                "Episode %d/%d  reward=%.1f  val_acc=%.4f",
                ep + 1,
                cfg.fusion.episodes,
                total_reward,
                acc,
            )

            if acc > best_acc:
                best_acc = acc
                torch.save(
                    {
                        "q_network": agent.q_network.state_dict(),
                        "target_network": agent.target_network.state_dict(),
                        "epsilon": agent.epsilon,
                    },
                    checkpoint_path(cfg, "fusion"),
                )

    # Ensure we always save something
    ckpt = checkpoint_path(cfg, "fusion")
    if not ckpt.exists():
        torch.save(
            {
                "q_network": agent.q_network.state_dict(),
                "target_network": agent.target_network.state_dict(),
                "epsilon": agent.epsilon,
            },
            ckpt,
        )

    return best_acc


def _train_mlp_fusion(cfg, train_cache, val_cache, device) -> float:
    """MLP supervised fusion. Returns best validation accuracy."""
    from src.models.dqn import MLPFusionAgent
    from src.models.registry import fusion_state_dim

    agent = MLPFusionAgent(
        state_dim=fusion_state_dim(),
        hidden_dims=cfg.fusion.mlp_hidden_dims,
        lr=cfg.fusion.lr,
        device=str(device),
    )
    best_acc = agent.train_on_cache(
        train_cache["states"],
        train_cache["labels"],
        val_cache["states"],
        val_cache["labels"],
        cfg,
    )
    torch.save(agent.state_dict(), checkpoint_path(cfg, "fusion"))
    return best_acc


def _train_weighted_avg_fusion(cfg, train_cache, val_cache, device) -> float:
    """Weighted average fusion. Returns best validation accuracy."""
    from src.models.dqn import WeightedAvgFusionAgent

    agent = WeightedAvgFusionAgent(device=str(device))
    best_acc = agent.train_on_cache(
        train_cache["states"],
        train_cache["labels"],
        val_cache["states"],
        val_cache["labels"],
        cfg,
    )
    torch.save(agent.state_dict(), checkpoint_path(cfg, "fusion"))
    return best_acc


def train_fusion(cfg: PipelineConfig) -> Path:
    """Train fusion agent on cached VGAE+GAT predictions. Returns checkpoint path."""
    log.info(
        "=== FUSION (%s): %s / %s_%s ===", cfg.fusion.method, cfg.dataset, cfg.model_type, cfg.scale
    )
    pl.seed_everything(cfg.seed)

    train_data, val_data, num_ids, in_ch = load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load frozen VGAE + GAT
    vgae = load_model(cfg, "vgae", "autoencoder", num_ids, in_ch, device)
    gat = load_model(cfg, "gat", "curriculum", num_ids, in_ch, device)

    # Cache predictions
    log.info("Caching VGAE + GAT predictions ...")
    models = {"vgae": vgae, "gat": gat}
    train_cache = cache_predictions(models, train_data, device, cfg.fusion.max_samples)
    val_cache = cache_predictions(models, val_data, device, cfg.fusion.max_val_samples)
    del vgae, gat
    cleanup()

    out = stage_dir(cfg, "fusion")
    out.mkdir(parents=True, exist_ok=True)

    # Dispatch on fusion method
    method = cfg.fusion.method
    if method == "dqn":
        best_acc = _train_dqn_fusion(cfg, train_cache, val_cache, device, out)
    elif method == "mlp":
        best_acc = _train_mlp_fusion(cfg, train_cache, val_cache, device)
    elif method == "weighted_avg":
        best_acc = _train_weighted_avg_fusion(cfg, train_cache, val_cache, device)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    ckpt = checkpoint_path(cfg, "fusion")
    cfg.save(config_path(cfg, "fusion"))
    log.info("Saved %s fusion: %s (best_acc=%.4f)", method, ckpt, best_acc)
    cleanup()
    return ckpt
