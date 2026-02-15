"""DQN fusion stage: combines VGAE + GAT predictions."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import pytorch_lightning as pl

from config import PipelineConfig, stage_dir, checkpoint_path, config_path
from .utils import load_data, load_model, cache_predictions, cleanup

log = logging.getLogger(__name__)


def train_fusion(cfg: PipelineConfig) -> Path:
    """Train DQN fusion agent on cached VGAE+GAT predictions. Returns checkpoint path."""
    log.info("=== FUSION: %s / %s_%s ===", cfg.dataset, cfg.model_type, cfg.scale)
    pl.seed_everything(cfg.seed)

    train_data, val_data, num_ids, in_ch = load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Determine prerequisite stage names (run_id() adds aux suffix automatically)
    vgae_stage = "autoencoder"
    gat_stage = "curriculum"

    # Load frozen VGAE + GAT
    vgae = load_model(cfg, "vgae", vgae_stage, num_ids, in_ch, device)
    gat = load_model(cfg, "gat", gat_stage, num_ids, in_ch, device)

    # Cache predictions
    log.info("Caching VGAE + GAT predictions ...")
    models = {"vgae": vgae, "gat": gat}
    train_cache = cache_predictions(models, train_data, device, cfg.fusion.max_samples)
    val_cache = cache_predictions(models, val_data, device, cfg.fusion.max_val_samples)
    del vgae, gat
    cleanup()

    # DQN agent
    from src.models.dqn import EnhancedDQNFusionAgent
    from src.models.registry import fusion_state_dim

    agent = EnhancedDQNFusionAgent(
        lr=cfg.fusion.lr, gamma=cfg.dqn.gamma,
        epsilon=cfg.dqn.epsilon, epsilon_decay=cfg.dqn.epsilon_decay,
        min_epsilon=cfg.dqn.min_epsilon,
        buffer_size=cfg.dqn.buffer_size, batch_size=cfg.dqn.batch_size,
        target_update_freq=cfg.dqn.target_update, device=str(device),
        state_dim=fusion_state_dim(),
        hidden_dim=cfg.dqn.hidden, num_layers=cfg.dqn.layers,
    )

    out = stage_dir(cfg, "fusion")
    out.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for ep in range(cfg.fusion.episodes):
        # Sample a batch of cached states
        idx = torch.randperm(len(train_cache["states"]))[:cfg.fusion.episode_sample_size]
        batch_states = train_cache["states"][idx]
        batch_labels = train_cache["labels"][idx]

        total_reward = 0.0
        for i in range(len(batch_states)):
            state_np = batch_states[i].numpy()
            alpha, action_idx, proc_state = agent.select_action(state_np, training=True)
            pred = 1 if alpha > 0.5 else 0
            reward = 3.0 if pred == batch_labels[i].item() else -3.0
            agent.store_experience(proc_state, action_idx, reward, proc_state, False)
            total_reward += reward

        # DQN training steps
        if len(agent.replay_buffer) >= cfg.dqn.batch_size:
            for _ in range(cfg.fusion.gpu_training_steps):
                agent.train_step()

        # Periodic validation
        if (ep + 1) % 50 == 0:
            val_pairs = [
                (val_cache["states"][i].numpy(), val_cache["labels"][i].item())
                for i in range(min(5000, len(val_cache["states"])))
            ]
            metrics = agent.validate_agent(val_pairs, num_samples=len(val_pairs))
            acc = metrics.get("accuracy", 0)
            log.info("Episode %d/%d  reward=%.1f  val_acc=%.4f",
                     ep + 1, cfg.fusion.episodes, total_reward, acc)

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "q_network": agent.q_network.state_dict(),
                    "target_network": agent.target_network.state_dict(),
                    "epsilon": agent.epsilon,
                }, checkpoint_path(cfg, "fusion"))

    # Ensure we always save something
    ckpt = checkpoint_path(cfg, "fusion")
    if not ckpt.exists():
        torch.save({
            "q_network": agent.q_network.state_dict(),
            "target_network": agent.target_network.state_dict(),
            "epsilon": agent.epsilon,
        }, ckpt)

    cfg.save(config_path(cfg, "fusion"))
    log.info("Saved DQN: %s (best_acc=%.4f)", ckpt, best_acc)
    cleanup()
    return ckpt
