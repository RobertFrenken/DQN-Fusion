"""Evaluation stage: runs inference on validation and test data."""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from ..config import PipelineConfig
from ..paths import stage_dir, checkpoint_path, config_path, data_dir, cache_dir
from .utils import (
    load_data,
    load_vgae,
    load_gat,
    load_frozen_cfg,
    cache_predictions,
    cleanup,
)

log = logging.getLogger(__name__)


def evaluate(cfg: PipelineConfig) -> dict:
    """Evaluate trained model(s) on validation and held-out test data.

    Output metrics.json layout:
        {
            "gat":    {"core": {...}, "additional": {...}},
            "vgae":   {"core": {...}, "additional": {...}},
            "fusion": {"core": {...}, "additional": {...}},
            "test": {
                "gat":    {"test_01_...": {"core": ...}, ...},
                "vgae":   {"test_01_...": {"core": ...}, ...},
                "fusion": {"test_01_...": {"core": ...}, ...}
            }
        }
    """
    log.info("=== EVALUATION: %s / %s ===", cfg.dataset, cfg.model_size)
    pl.seed_everything(cfg.seed)

    train_data, val_data, num_ids, in_ch = load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    test_scenarios = _load_test_data(cfg)

    all_metrics: dict = {}
    test_metrics: dict = {}

    # Determine stage names based on use_kd flag
    gat_stage = "curriculum_kd" if cfg.use_kd else "curriculum"
    vgae_stage = "autoencoder_kd" if cfg.use_kd else "autoencoder"

    # ---- GAT evaluation ----
    gat_ckpt = checkpoint_path(cfg, gat_stage)
    if gat_ckpt.exists():
        gat = load_gat(cfg, num_ids, in_ch, device, stage=gat_stage)

        p, l, s = _run_gat_inference(gat, val_data, device)
        all_metrics["gat"] = _compute_metrics(l, p, s)
        log.info("GAT val metrics: %s",
                 {k: f"{v:.4f}" for k, v in all_metrics["gat"]["core"].items()
                  if isinstance(v, float)})

        if test_scenarios:
            test_metrics["gat"] = {}
            for scenario, tdata in test_scenarios.items():
                tp, tl, ts = _run_gat_inference(gat, tdata, device)
                test_metrics["gat"][scenario] = _compute_metrics(tl, tp, ts)
                log.info("GAT %s  acc=%.4f f1=%.4f",
                         scenario,
                         test_metrics["gat"][scenario]["core"]["accuracy"],
                         test_metrics["gat"][scenario]["core"]["f1"])

        del gat
        cleanup()

    # ---- VGAE evaluation ----
    vgae_ckpt = checkpoint_path(cfg, vgae_stage)
    if vgae_ckpt.exists():
        vgae = load_vgae(cfg, num_ids, in_ch, device, stage=vgae_stage)

        errors_np, labels_np = _run_vgae_inference(vgae, val_data, device)
        best_thresh, youden_j, vgae_preds = _vgae_threshold(labels_np, errors_np)
        all_metrics["vgae"] = _compute_metrics(labels_np, vgae_preds, errors_np)
        all_metrics["vgae"]["core"]["optimal_threshold"] = best_thresh
        all_metrics["vgae"]["core"]["youden_j"] = youden_j
        log.info("VGAE val metrics: %s",
                 {k: f"{v:.4f}" for k, v in all_metrics["vgae"]["core"].items()
                  if isinstance(v, float)})

        if test_scenarios:
            test_metrics["vgae"] = {}
            for scenario, tdata in test_scenarios.items():
                te, tl = _run_vgae_inference(vgae, tdata, device)
                tp = (te > best_thresh).astype(int)
                test_metrics["vgae"][scenario] = _compute_metrics(tl, tp, te)
                test_metrics["vgae"][scenario]["core"]["threshold_from_val"] = best_thresh
                log.info("VGAE %s  acc=%.4f f1=%.4f",
                         scenario,
                         test_metrics["vgae"][scenario]["core"]["accuracy"],
                         test_metrics["vgae"][scenario]["core"]["f1"])

        del vgae
        cleanup()

    # ---- DQN Fusion evaluation ----
    fusion_ckpt = checkpoint_path(cfg, "fusion")
    if fusion_ckpt.exists() and vgae_ckpt.exists() and gat_ckpt.exists():
        vgae = load_vgae(cfg, num_ids, in_ch, device, stage=vgae_stage)
        gat = load_gat(cfg, num_ids, in_ch, device, stage=gat_stage)

        val_cache = cache_predictions(vgae, gat, val_data, device, cfg.max_val_samples)

        from src.models.dqn import EnhancedDQNFusionAgent

        fusion_cfg = load_frozen_cfg(cfg, "fusion")
        agent = EnhancedDQNFusionAgent(
            lr=fusion_cfg.fusion_lr, gamma=fusion_cfg.dqn_gamma,
            epsilon=0.0, epsilon_decay=1.0, min_epsilon=0.0,
            buffer_size=fusion_cfg.dqn_buffer_size,
            batch_size=fusion_cfg.dqn_batch_size,
            target_update_freq=fusion_cfg.dqn_target_update,
            device=str(device),
        )
        fusion_sd = torch.load(fusion_ckpt, map_location="cpu", weights_only=True)
        agent.q_network.load_state_dict(fusion_sd["q_network"])
        agent.target_network.load_state_dict(fusion_sd["target_network"])

        fp, fl, fs = _run_fusion_inference(agent, val_cache)
        all_metrics["fusion"] = _compute_metrics(fl, fp, fs)
        log.info("Fusion val metrics: %s",
                 {k: f"{v:.4f}" for k, v in all_metrics["fusion"]["core"].items()
                  if isinstance(v, float)})

        if test_scenarios:
            test_metrics["fusion"] = {}
            for scenario, tdata in test_scenarios.items():
                tc = cache_predictions(vgae, gat, tdata, device, cfg.max_val_samples)
                tp, tl, ts = _run_fusion_inference(agent, tc)
                test_metrics["fusion"][scenario] = _compute_metrics(tl, tp, ts)
                log.info("Fusion %s  acc=%.4f f1=%.4f",
                         scenario,
                         test_metrics["fusion"][scenario]["core"]["accuracy"],
                         test_metrics["fusion"][scenario]["core"]["f1"])

        del vgae, gat
        cleanup()

    if test_metrics:
        all_metrics["test"] = test_metrics

    # Log core metrics to MLflow
    for model_name, model_metrics in all_metrics.items():
        if model_name == "test":
            continue
        if "core" in model_metrics:
            for k, v in model_metrics["core"].items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{model_name}_{k}", v)

    # Save all metrics
    out = stage_dir(cfg, "evaluation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    cfg.save(config_path(cfg, "evaluation"))
    log.info("All metrics saved to %s/metrics.json", out)
    cleanup()
    return all_metrics


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _graph_label(g) -> int:
    """Extract scalar graph-level label consistently."""
    return g.y.item() if g.y.dim() == 0 else int(g.y[0].item())


def _load_test_data(cfg: PipelineConfig) -> dict:
    """Load held-out test graphs per scenario."""
    from src.preprocessing.preprocessing import graph_creation

    mapping_file = cache_dir(cfg) / "id_mapping.pkl"
    if not mapping_file.exists():
        log.warning("No id_mapping at %s — skipping test data", mapping_file)
        return {}

    with open(mapping_file, "rb") as f:
        id_mapping = pickle.load(f)

    ds_path = data_dir(cfg)
    if not ds_path.exists():
        log.warning("Dataset path %s not found — skipping test data", ds_path)
        return {}

    scenarios: dict[str, list] = {}
    for folder in sorted(ds_path.iterdir()):
        if folder.is_dir() and folder.name.startswith("test_"):
            name = folder.name
            log.info("Loading test scenario: %s", name)
            graphs = graph_creation(
                str(ds_path), folder_type=name,
                id_mapping=id_mapping, return_id_mapping=False,
            )
            if graphs:
                scenarios[name] = graphs
                log.info("  %s: %d graphs", name, len(graphs))

    return scenarios


def _run_gat_inference(gat, data, device):
    """Run GAT inference. Returns (preds, labels, scores) as numpy arrays."""
    preds, labels, scores = [], [], []
    with torch.no_grad():
        for g in data:
            g = g.clone().to(device)
            logits = gat(g)
            probs = F.softmax(logits, dim=1)
            preds.append(logits.argmax(1)[0].item())
            labels.append(_graph_label(g))
            scores.append(probs[0, 1].item())
    return np.array(preds), np.array(labels), np.array(scores)


def _run_vgae_inference(vgae, data, device):
    """Run VGAE reconstruction-error inference. Returns (errors, labels)."""
    errors, labels = [], []
    with torch.no_grad():
        for g in data:
            g = g.clone().to(device)
            batch_idx = (g.batch if hasattr(g, "batch") and g.batch is not None
                         else torch.zeros(g.x.size(0), dtype=torch.long, device=device))
            cont, canid_logits, _, _, _ = vgae(g.x, g.edge_index, batch_idx)
            err = F.mse_loss(cont, g.x[:, 1:]).item()
            errors.append(err)
            labels.append(_graph_label(g))
    return np.array(errors), np.array(labels)


def _run_fusion_inference(agent, cache):
    """Run DQN fusion inference. Returns (preds, labels, scores)."""
    preds, labels, scores = [], [], []
    for i in range(len(cache["states"])):
        state_np = cache["states"][i].numpy()
        alpha, _, _ = agent.select_action(state_np, training=False)
        preds.append(1 if alpha > 0.5 else 0)
        labels.append(cache["labels"][i].item())
        scores.append(float(alpha))
    return np.array(preds), np.array(labels), np.array(scores)


def _vgae_threshold(labels, errors):
    """Find optimal anomaly-detection threshold via Youden's J statistic."""
    from sklearn.metrics import roc_curve as _roc_curve

    fpr_v, tpr_v, thresholds_v = _roc_curve(labels, errors)
    j_scores = tpr_v - fpr_v
    best_idx = np.argmax(j_scores)
    best_thresh = (float(thresholds_v[best_idx])
                   if best_idx < len(thresholds_v)
                   else float(np.median(errors)))
    preds = (errors > best_thresh).astype(int)
    return best_thresh, float(j_scores[best_idx]), preds


def _compute_metrics(labels, preds, scores=None) -> dict:
    """Compute comprehensive classification metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
        confusion_matrix, cohen_kappa_score, precision_recall_curve,
        auc as sk_auc, roc_curve,
    )

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    core = {
        "accuracy":          float(accuracy_score(labels, preds)),
        "precision":         float(precision_score(labels, preds, zero_division=0)),
        "recall":            float(recall_score(labels, preds, zero_division=0)),
        "f1":                float(f1_score(labels, preds, zero_division=0)),
        "specificity":       specificity,
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "mcc":               float(matthews_corrcoef(labels, preds)),
        "fpr":               fpr,
        "fnr":               fnr,
        "n_samples":         int(len(labels)),
    }

    additional = {
        "kappa":          float(cohen_kappa_score(labels, preds)),
        "tpr":            tpr,
        "tnr":            specificity,
        "detection_rate": tpr,
        "miss_rate":      fnr,
    }

    if scores is not None and len(set(labels)) > 1:
        core["auc"] = float(roc_auc_score(labels, scores))

        try:
            prec_vals, rec_vals, _ = precision_recall_curve(labels, scores)
            additional["pr_auc"] = float(sk_auc(rec_vals, prec_vals))
        except ValueError:
            additional["pr_auc"] = 0.0

        try:
            fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
            det_at_fpr = {}
            for fpr_target in [0.05, 0.01, 0.001]:
                idx = np.argmin(np.abs(fpr_curve - fpr_target))
                det_at_fpr[str(fpr_target)] = float(tpr_curve[idx])
            additional["detection_at_fpr"] = det_at_fpr
        except ValueError:
            additional["detection_at_fpr"] = {}

    return {"core": core, "additional": additional}
