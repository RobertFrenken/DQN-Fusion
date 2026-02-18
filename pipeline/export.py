"""Export experiment results to static JSON for the GitHub Pages dashboard.

Data sources:
  - Filesystem: experimentruns/{ds}/{run}/metrics.json, config.json
  - Catalog: config/datasets.yaml
  - Artifacts: embeddings.npz, attention_weights.npz, dqn_policy.json, cka_matrix.json

Usage:
    python -m pipeline.export [--output-dir docs/dashboard/data]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("docs/dashboard/data")
EXPERIMENT_ROOT = Path("experimentruns")


def _versioned_envelope(data: list | dict) -> dict:
    """Wrap export data with schema version and timestamp."""
    from datetime import datetime, timezone
    return {
        "schema_version": "1.0.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }


# ---------------------------------------------------------------------------
# Filesystem scanning helpers
# ---------------------------------------------------------------------------

def _scan_runs() -> list[dict]:
    """Scan experimentruns/ for completed runs with config.json.

    Returns list of dicts with run metadata extracted from config.json
    and filesystem structure.
    """
    runs = []
    if not EXPERIMENT_ROOT.is_dir():
        return runs

    for ds_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            cfg_path = run_dir / "config.json"
            if not cfg_path.exists():
                continue
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                continue

            # Parse run_id components from directory name
            parts = run_dir.name.split("_")
            model_type = parts[0] if parts else ""
            scale = parts[1] if len(parts) > 1 else ""
            stage = parts[2] if len(parts) > 2 else ""
            has_kd = "_kd" in run_dir.name and "nokd" not in run_dir.name

            run_id = f"{ds_dir.name}/{run_dir.name}"
            has_metrics = (run_dir / "metrics.json").exists()
            has_checkpoint = (run_dir / "best_model.pt").exists()

            runs.append({
                "run_id": run_id,
                "dataset": ds_dir.name,
                "model_type": model_type,
                "scale": scale,
                "stage": stage,
                "has_kd": 1 if has_kd else 0,
                "status": "complete" if has_metrics or has_checkpoint else "running",
                "config": cfg,
                "dir": run_dir,
            })
    return runs


def _load_eval_metrics(run_dir: Path) -> dict | None:
    """Load metrics.json from an evaluation run directory."""
    mp = run_dir / "metrics.json"
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Export functions (DB-free: read from filesystem)
# ---------------------------------------------------------------------------

def export_leaderboard(output_dir: Path) -> Path:
    """Best F1/accuracy per model x dataset x scale from evaluation metrics.json files."""
    target_metrics = {"f1", "accuracy", "precision", "recall", "auc", "mcc"}
    rows = []

    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        metrics = _load_eval_metrics(run["dir"])
        if not metrics:
            continue

        for model_key in ("gat", "vgae", "fusion"):
            model_data = metrics.get(model_key, {})
            core = model_data.get("core", {})
            for metric_name in target_metrics:
                if metric_name in core and isinstance(core[metric_name], (int, float)):
                    rows.append({
                        "dataset": run["dataset"],
                        "model_type": run["model_type"],
                        "scale": run["scale"],
                        "has_kd": run["has_kd"],
                        "model": model_key,
                        "metric_name": metric_name,
                        "best_value": round(core[metric_name], 6),
                    })

    out = output_dir / "leaderboard.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d leaderboard entries → %s", len(rows), out)
    return out


def export_runs(output_dir: Path) -> Path:
    """All runs with status."""
    rows = []
    for run in _scan_runs():
        rows.append({
            "run_id": run["run_id"],
            "dataset": run["dataset"],
            "model_type": run["model_type"],
            "scale": run["scale"],
            "stage": run["stage"],
            "has_kd": run["has_kd"],
            "status": run["status"],
            "teacher_run": "",
            "started_at": None,
            "completed_at": None,
        })

    out = output_dir / "runs.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d runs → %s", len(rows), out)
    return out


def export_metrics(output_dir: Path) -> Path:
    """Per-run flattened metrics from evaluation metrics.json files."""
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        metrics = _load_eval_metrics(run["dir"])
        if not metrics:
            continue

        rows = []
        for model_key in ("gat", "vgae", "fusion"):
            model_data = metrics.get(model_key, {})
            for scenario_type in ("core", "additional"):
                section = model_data.get(scenario_type, {})
                for metric_name, value in section.items():
                    if isinstance(value, (int, float)):
                        rows.append({
                            "model": model_key,
                            "scenario": "val",
                            "metric_name": metric_name,
                            "value": value,
                        })

        fname = run["run_id"].replace("/", "_") + ".json"
        (metrics_dir / fname).write_text(json.dumps(_versioned_envelope(rows), indent=2))
        count += 1

    log.info("Exported metrics for %d runs → %s", count, metrics_dir)
    return metrics_dir


def export_metric_catalog(output_dir: Path) -> Path:
    """Export distinct metric names for dynamic dashboard dropdown."""
    all_names: set[str] = set()

    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        metrics = _load_eval_metrics(run["dir"])
        if not metrics:
            continue
        for model_key in ("gat", "vgae", "fusion"):
            model_data = metrics.get(model_key, {})
            for section in ("core", "additional"):
                all_names.update(
                    k for k, v in model_data.get(section, {}).items()
                    if isinstance(v, (int, float))
                )

    catalog = sorted(all_names)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / "metric_catalog.json"
    out.write_text(json.dumps(_versioned_envelope(catalog), indent=2))
    log.info("Exported %d metric names → %s", len(catalog), out)
    return out


def export_datasets(output_dir: Path) -> Path:
    """Dataset metadata from config/datasets.yaml catalog."""
    from config.catalog import load_catalog

    catalog = load_catalog()
    rows = []
    for name, entry in catalog.items():
        rows.append({
            "name": name,
            "domain": getattr(entry, "domain", "automotive"),
            "protocol": getattr(entry, "protocol", "CAN"),
            "source": getattr(entry, "source", ""),
            "description": getattr(entry, "description", ""),
        })

    out = output_dir / "datasets.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d datasets → %s", len(rows), out)
    return out


def export_kd_transfer(output_dir: Path) -> Path:
    """Teacher vs student metric pairs for KD analysis.

    Scans evaluation runs, pairs large (teacher) with small+kd (student)
    on the same dataset.
    """
    target_metrics = {"f1", "accuracy", "auc"}
    rows = []

    # Group evaluation runs by dataset
    eval_runs: dict[str, list[dict]] = {}
    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        eval_runs.setdefault(run["dataset"], []).append(run)

    for ds, runs in eval_runs.items():
        # Find teacher (large, no KD) and student (small, KD) runs
        teachers = [r for r in runs if r["scale"] == "large" and not r["has_kd"]]
        students = [r for r in runs if r["scale"] == "small" and r["has_kd"]]

        if not teachers or not students:
            continue

        teacher = teachers[0]
        student = students[0]
        t_metrics = _load_eval_metrics(teacher["dir"])
        s_metrics = _load_eval_metrics(student["dir"])
        if not t_metrics or not s_metrics:
            continue

        for model_key in ("gat", "vgae", "fusion"):
            t_core = t_metrics.get(model_key, {}).get("core", {})
            s_core = s_metrics.get(model_key, {}).get("core", {})
            for mn in target_metrics:
                if mn in t_core and mn in s_core:
                    rows.append({
                        "student_run": student["run_id"],
                        "dataset": ds,
                        "model_type": teacher["model_type"],
                        "student_scale": "small",
                        "teacher_run": teacher["run_id"],
                        "metric_name": mn,
                        "student_value": round(s_core[mn], 6),
                        "teacher_value": round(t_core[mn], 6),
                    })

    out = output_dir / "kd_transfer.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d KD transfer pairs → %s", len(rows), out)
    return out


def export_training_curves(output_dir: Path) -> Path:
    """Per-run training curves from Lightning CSV logs."""
    curves_dir = output_dir / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    if not EXPERIMENT_ROOT.is_dir():
        return curves_dir

    for ds_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            # Find Lightning CSV log
            csv_logs = list(run_dir.glob("csv_logs/*/metrics.csv")) + \
                       list(run_dir.glob("lightning_logs/*/metrics.csv"))
            if not csv_logs:
                continue

            try:
                import csv
                rows = []
                with open(csv_logs[0]) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        epoch = row.get("epoch")
                        if epoch is None:
                            continue
                        for key, val in row.items():
                            if key == "epoch" or key == "step" or val == "":
                                continue
                            try:
                                rows.append({
                                    "epoch": int(float(epoch)),
                                    "metric_name": key,
                                    "value": float(val),
                                })
                            except (ValueError, TypeError):
                                continue

                if rows:
                    run_id = f"{ds_dir.name}/{run_dir.name}"
                    fname = run_id.replace("/", "_") + ".json"
                    (curves_dir / fname).write_text(
                        json.dumps(_versioned_envelope(rows), indent=2)
                    )
                    count += 1
            except Exception as e:
                log.warning("Failed to parse CSV log in %s: %s", run_dir, e)

    log.info("Exported training curves for %d runs → %s", count, curves_dir)
    return curves_dir


# ---------------------------------------------------------------------------
# Filesystem-only exports (unchanged from original)
# ---------------------------------------------------------------------------

def export_graph_samples(output_dir: Path) -> Path:
    """Export a few cached PyG graphs per dataset as D3-compatible JSON."""
    import torch

    cache_root = Path("data/cache")
    samples: list[dict] = []

    for ds_dir in sorted(cache_root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name.endswith(".dvc"):
            continue
        graphs_path = ds_dir / "processed_graphs.pt"
        if not graphs_path.exists():
            continue

        try:
            graphs = torch.load(graphs_path, map_location="cpu", weights_only=False)
        except Exception as e:
            log.warning("Could not load graphs from %s: %s", graphs_path, e)
            continue

        import random as _random
        rng = _random.Random(42)
        normal_idx = [i for i, g in enumerate(graphs)
                      if hasattr(g, 'y') and g.y is not None and int(g.y.item()) == 0]
        attack_idx = [i for i, g in enumerate(graphs)
                      if hasattr(g, 'y') and g.y is not None and int(g.y.item()) == 1]
        selected_indices = set()
        selected = []
        if normal_idx:
            selected.append(graphs[normal_idx[0]])
            selected_indices.add(normal_idx[0])
        if attack_idx:
            selected.append(graphs[attack_idx[0]])
            selected_indices.add(attack_idx[0])
        remaining = [i for i in range(len(graphs)) if i not in selected_indices]
        if remaining and len(selected) < 3:
            selected.append(graphs[rng.choice(remaining)])
        for idx, g in enumerate(selected):
            edge_index = g.edge_index.tolist()
            num_nodes = g.x.size(0) if hasattr(g, "x") else g.num_nodes
            nodes = []
            for nid in range(num_nodes):
                node = {"id": nid}
                if hasattr(g, "x"):
                    node["features"] = g.x[nid].tolist()
                nodes.append(node)
            links = [
                {"source": edge_index[0][i], "target": edge_index[1][i]}
                for i in range(len(edge_index[0]))
            ]
            label = int(g.y.item()) if hasattr(g, "y") and g.y is not None else None
            samples.append({
                "dataset": ds_dir.name,
                "sample_idx": idx,
                "label": label,
                "nodes": nodes,
                "links": links,
            })

    out = output_dir / "graph_samples.json"
    out.write_text(json.dumps(_versioned_envelope(samples), indent=2))
    log.info("Exported %d graph samples → %s", len(samples), out)
    return out


def export_model_sizes(output_dir: Path) -> Path:
    """Export parameter counts per model_type x scale from config resolution."""
    from config import resolve
    from config.constants import NODE_FEATURE_COUNT

    sizes: list[dict] = []
    num_ids = 30
    in_ch = NODE_FEATURE_COUNT

    for model_type in ("vgae", "gat", "dqn"):
        for scale in ("large", "small"):
            try:
                cfg = resolve(model_type, scale, dataset="hcrl_sa")
                from src.models.registry import get as get_model
                entry = get_model(model_type)
                model = entry.factory(cfg, num_ids, in_ch)
                param_count = sum(p.numel() for p in model.parameters())
                sizes.append({
                    "model_type": model_type,
                    "scale": scale,
                    "param_count": param_count,
                    "param_count_M": round(param_count / 1e6, 3),
                })
                del model
            except Exception as e:
                log.warning("Could not instantiate %s/%s for param count: %s",
                            model_type, scale, e)

    out = output_dir / "model_sizes.json"
    out.write_text(json.dumps(_versioned_envelope(sizes), indent=2))
    log.info("Exported %d model size entries → %s", len(sizes), out)
    return out


EMBEDDING_MAX_SAMPLES = 2000


def _stratified_sample(coords_2d, labels, errors, max_samples: int):
    """Stratified sampling to preserve class distribution."""
    import numpy as np

    n = len(coords_2d)
    if n <= max_samples:
        return coords_2d, labels, errors, False

    unique_labels = np.unique(labels[:n]) if len(labels) >= n else np.array([0])
    indices = []
    for label in unique_labels:
        label_mask = labels[:n] == label if len(labels) >= n else np.ones(n, dtype=bool)
        label_indices = np.where(label_mask)[0]
        n_samples = max(1, int(max_samples * len(label_indices) / n))
        rng = np.random.default_rng(42)
        sampled = rng.choice(label_indices, size=min(n_samples, len(label_indices)), replace=False)
        indices.extend(sampled)

    indices = sorted(indices)[:max_samples]
    return (
        coords_2d[indices],
        labels[indices] if len(labels) >= n else labels,
        errors[indices] if len(errors) >= n else errors,
        True,
    )


def export_embeddings(output_dir: Path) -> Path:
    """Export dimensionality-reduced embeddings from evaluation runs."""
    embed_dir = output_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    exported_files: list[str] = []

    if not EXPERIMENT_ROOT.is_dir():
        index_path = embed_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        log.info("No experimentruns directory — wrote empty embeddings index")
        return embed_dir

    for ds_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            npz_path = run_dir / "embeddings.npz"
            if not npz_path.exists():
                continue

            import numpy as np
            data = np.load(npz_path, allow_pickle=True)
            run_id = f"{ds_dir.name}_{run_dir.name}"

            for model_key in ("vgae_z", "gat_emb"):
                if model_key not in data:
                    continue
                embeddings = data[model_key]
                model_name = model_key.split("_")[0]
                labels = data.get(f"{model_name}_labels", np.array([]))
                errors = data.get(f"{model_name}_errors", np.array([]))

                if embeddings.shape[0] < 3:
                    continue

                # Downsample BEFORE reduction to keep UMAP/PyMDE tractable
                total_original = embeddings.shape[0]
                embeddings, labels, errors, was_sampled = _stratified_sample(
                    embeddings, labels, errors, EMBEDDING_MAX_SAMPLES
                )
                if was_sampled:
                    log.info("Pre-sampled embeddings %s/%s: %d → %d",
                             run_id, model_name, total_original, len(embeddings))

                for method in ("umap", "pymde"):
                    try:
                        coords_2d = _reduce_embeddings(embeddings, method)
                    except Exception as e:
                        log.warning("Dimensionality reduction (%s) failed for %s/%s: %s",
                                    method, run_id, model_name, e)
                        continue

                    # _stratified_sample already applied before reduction
                    s_labels, s_errors = labels, errors

                    records = []
                    for i in range(len(coords_2d)):
                        rec = {
                            "dim0": float(coords_2d[i, 0]),
                            "dim1": float(coords_2d[i, 1]),
                        }
                        if i < len(s_labels):
                            rec["label"] = int(s_labels[i])
                        if i < len(s_errors):
                            rec["error"] = float(s_errors[i])
                        records.append(rec)

                    metadata = {
                        "n_points": len(records),
                        "sampled": was_sampled,
                        "total_original": total_original,
                    }

                    fname = f"{run_id}_{model_name}_{method}.json"
                    envelope = _versioned_envelope(records)
                    envelope["metadata"] = metadata
                    (embed_dir / fname).write_text(json.dumps(envelope, indent=2))
                    exported_files.append(fname)
                    count += 1

    index_path = embed_dir / "index.json"
    index_path.write_text(json.dumps(_versioned_envelope(sorted(exported_files)), indent=2))

    log.info("Exported %d embedding projections → %s", count, embed_dir)
    return embed_dir


def _reduce_embeddings(embeddings, method: str):
    """Reduce high-dimensional embeddings to 2D.

    Applies PCA pre-reduction to 50 dimensions when input dimensionality
    exceeds 50 — standard practice that makes UMAP/PyMDE tractable on
    high-dimensional latent spaces (e.g. 2049-D VGAE z-vectors).
    """
    import numpy as np
    from sklearn.decomposition import PCA

    PCA_TARGET = 50
    if embeddings.shape[1] > PCA_TARGET:
        n_components = min(PCA_TARGET, embeddings.shape[0] - 1)
        embeddings = PCA(n_components=n_components, random_state=42).fit_transform(embeddings)

    if method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(embeddings)
    elif method == "pymde":
        import pymde
        import torch
        data_t = torch.tensor(embeddings, dtype=torch.float32)
        mde = pymde.preserve_neighbors(data_t, embedding_dim=2)
        return mde.embed().numpy()
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def export_dqn_policy(output_dir: Path) -> Path:
    """Export DQN alpha distributions from evaluation runs."""
    policy_dir = output_dir / "dqn_policy"
    policy_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    exported_files: list[str] = []

    if not EXPERIMENT_ROOT.is_dir():
        index_path = policy_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        return policy_dir

    for ds_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            policy_path = run_dir / "dqn_policy.json"
            if not policy_path.exists():
                continue

            run_id = f"{ds_dir.name}_{run_dir.name}"
            fname = f"{run_id}.json"
            import shutil
            shutil.copy2(policy_path, policy_dir / fname)
            exported_files.append(fname)
            count += 1

    index_path = policy_dir / "index.json"
    index_path.write_text(json.dumps(_versioned_envelope(sorted(exported_files)), indent=2))

    log.info("Exported %d DQN policy files → %s", count, policy_dir)
    return policy_dir


def export_roc_curves(output_dir: Path) -> Path:
    """Export ROC/PR curve arrays from evaluation metrics.json files."""
    curves_dir = output_dir / "roc_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    exported_files: list[str] = []

    if not EXPERIMENT_ROOT.is_dir():
        index_path = curves_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        return curves_dir

    for ds_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            mp = run_dir / "metrics.json"
            if not mp.exists():
                continue

            metrics = json.loads(mp.read_text())
            run_id = f"{ds_dir.name}_{run_dir.name}"

            for model_key in ("vgae", "gat", "fusion"):
                model_metrics = metrics.get(model_key, {})
                additional = model_metrics.get("additional", {})
                roc = additional.get("roc_curve")
                pr = additional.get("pr_curve")
                if not roc and not pr:
                    continue
                curve_data = {"model": model_key}
                if roc:
                    curve_data["roc_curve"] = roc
                    curve_data["auc"] = model_metrics.get("core", {}).get("auc")
                if pr:
                    curve_data["pr_curve"] = pr
                    curve_data["pr_auc"] = additional.get("pr_auc")

                fname = f"{run_id}_{model_key}.json"
                (curves_dir / fname).write_text(json.dumps(_versioned_envelope(curve_data), indent=2))
                exported_files.append(fname)
                count += 1

    index_path = curves_dir / "index.json"
    index_path.write_text(json.dumps(_versioned_envelope(sorted(exported_files)), indent=2))
    log.info("Exported %d ROC/PR curve files → %s", count, curves_dir)
    return curves_dir


def export_attention(output_dir: Path) -> Path:
    """Export GAT attention weights from evaluation runs as D3-friendly JSON."""
    attn_dir = output_dir / "attention"
    attn_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    count = 0
    exported_files: list[str] = []

    if not EXPERIMENT_ROOT.is_dir():
        index_path = attn_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        return attn_dir

    for ds_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            npz_path = run_dir / "attention_weights.npz"
            if not npz_path.exists():
                continue

            data = np.load(npz_path, allow_pickle=True)
            n_samples = int(data["n_samples"])
            run_id = f"{ds_dir.name}_{run_dir.name}"

            samples = []
            for i in range(n_samples):
                prefix = f"sample_{i}"
                graph_idx = int(data[f"{prefix}_graph_idx"])
                label = int(data[f"{prefix}_label"])
                edge_index = data[f"{prefix}_edge_index"].tolist()
                node_features = data[f"{prefix}_node_features"].tolist()

                layers = []
                layer_idx = 0
                while f"{prefix}_layer_{layer_idx}_alpha" in data:
                    alpha = data[f"{prefix}_layer_{layer_idx}_alpha"]
                    layers.append({
                        "alpha_mean": alpha.mean(axis=-1).tolist() if alpha.ndim > 1 else alpha.tolist(),
                        "n_heads": int(alpha.shape[-1]) if alpha.ndim > 1 else 1,
                    })
                    layer_idx += 1

                samples.append({
                    "graph_idx": graph_idx,
                    "label": label,
                    "edge_index": edge_index,
                    "node_features": node_features,
                    "layers": layers,
                })

            fname = f"{run_id}.json"
            (attn_dir / fname).write_text(json.dumps(_versioned_envelope(samples), indent=2))
            exported_files.append(fname)
            count += 1

    index_path = attn_dir / "index.json"
    index_path.write_text(json.dumps(_versioned_envelope(sorted(exported_files)), indent=2))
    log.info("Exported %d attention files → %s", count, attn_dir)
    return attn_dir


def export_recon_errors(output_dir: Path) -> Path:
    """Export VGAE reconstruction errors from evaluation embeddings.npz files."""
    recon_dir = output_dir / "recon_errors"
    recon_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    count = 0
    exported_files: list[str] = []

    if not EXPERIMENT_ROOT.is_dir():
        index_path = recon_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        return recon_dir

    for ds_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            npz_path = run_dir / "embeddings.npz"
            if not npz_path.exists():
                continue

            data = np.load(npz_path, allow_pickle=True)
            if "vgae_errors" not in data:
                continue

            errors = data["vgae_errors"].tolist()
            labels = data["vgae_labels"].tolist() if "vgae_labels" in data else []

            threshold = None
            mp = run_dir / "metrics.json"
            if mp.exists():
                metrics = json.loads(mp.read_text())
                threshold = metrics.get("vgae", {}).get("core", {}).get("optimal_threshold")

            recon_data = {
                "errors": errors,
                "labels": labels,
                "optimal_threshold": threshold,
            }

            run_id = f"{ds_dir.name}_{run_dir.name}"
            fname = f"{run_id}.json"
            (recon_dir / fname).write_text(json.dumps(_versioned_envelope(recon_data), indent=2))
            exported_files.append(fname)
            count += 1

    index_path = recon_dir / "index.json"
    index_path.write_text(json.dumps(_versioned_envelope(sorted(exported_files)), indent=2))
    log.info("Exported %d recon error files → %s", count, recon_dir)
    return recon_dir


def export_cka(output_dir: Path) -> Path:
    """Export CKA matrices from KD evaluation runs."""
    cka_dir = output_dir / "cka"
    cka_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    exported_files: list[str] = []

    if not EXPERIMENT_ROOT.is_dir():
        index_path = cka_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        return cka_dir

    for ds_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            cka_path = run_dir / "cka_matrix.json"
            if not cka_path.exists():
                continue

            import shutil
            run_id = f"{ds_dir.name}_{run_dir.name}"
            fname = f"{run_id}.json"
            shutil.copy2(cka_path, cka_dir / fname)
            exported_files.append(fname)
            count += 1

    index_path = cka_dir / "index.json"
    index_path.write_text(json.dumps(_versioned_envelope(sorted(exported_files)), indent=2))
    log.info("Exported %d CKA matrices → %s", count, cka_dir)
    return cka_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export_all(output_dir: Path) -> None:
    """Run all exports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lb = export_leaderboard(output_dir)
    runs = export_runs(output_dir)
    export_metrics(output_dir)
    ds = export_datasets(output_dir)
    kd = export_kd_transfer(output_dir)
    export_training_curves(output_dir)
    export_metric_catalog(output_dir)

    for name, func in [
        ("graph_samples", export_graph_samples),
        ("model_sizes", export_model_sizes),
        ("embeddings", export_embeddings),
        ("dqn_policy", export_dqn_policy),
        ("roc_curves", export_roc_curves),
        ("attention", export_attention),
        ("recon_errors", export_recon_errors),
        ("cka", export_cka),
    ]:
        try:
            func(output_dir)
        except Exception as e:
            log.warning("Export %s failed (non-fatal): %s", name, e)

    for name, path in [
        ("leaderboard", lb), ("runs", runs), ("datasets", ds), ("kd_transfer", kd),
    ]:
        if path.stat().st_size < 10:
            log.warning("EMPTY EXPORT: %s (%s)", name, path)

    log.info("All exports complete → %s", output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline.export",
        description="Export experiment results to static JSON for dashboard",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )

    export_all(args.output_dir)


if __name__ == "__main__":
    main()
