"""Export project DB to static JSON for the GitHub Pages dashboard.

Usage:
    python -m pipeline.export [--output-dir docs/dashboard/data]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config.constants import SCHEMA_VERSION
from .db import get_connection

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("docs/dashboard/data")


def _versioned_envelope(data: list | dict) -> dict:
    """Wrap export data with schema version and timestamp."""
    from datetime import datetime, timezone
    return {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }


REQUIRED_COLUMNS = {
    "runs": {"run_id", "dataset", "model_type", "scale", "has_kd", "status", "started_at"},
    "metrics": {"run_id", "model", "scenario", "metric_name", "value"},
}


def _validate_schema(conn):
    """Verify DB schema has all required columns before exporting."""
    for table, expected in REQUIRED_COLUMNS.items():
        actual = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        missing = expected - actual
        if missing:
            raise RuntimeError(f"DB schema mismatch: {table} missing columns {missing}")


def _validate_data(conn) -> None:
    """Pre-export data quality checks. Warns but does not abort."""
    epoch_count = conn.execute("SELECT COUNT(*) FROM epoch_metrics").fetchone()[0]
    if epoch_count == 0:
        log.warning("VALIDATION: epoch_metrics table has 0 rows — training curves will be empty")

    null_started = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE started_at IS NULL"
    ).fetchone()[0]
    total_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    if null_started > 0:
        log.warning("VALIDATION: %d/%d runs have NULL started_at — run timeline will be sparse",
                     null_started, total_runs)

    metric_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    if metric_count == 0:
        log.warning("VALIDATION: metrics table has 0 rows — leaderboard will be empty")


def export_metric_catalog(output_dir: Path) -> Path:
    """Export distinct metric names for dynamic dashboard dropdown."""
    conn = get_connection()
    conn.row_factory = _dict_factory
    rows = conn.execute(
        "SELECT DISTINCT metric_name FROM metrics ORDER BY metric_name"
    ).fetchall()
    conn.close()

    catalog = [r["metric_name"] for r in rows]
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / "metric_catalog.json"
    out.write_text(json.dumps(_versioned_envelope(catalog), indent=2))
    log.info("Exported %d metric names → %s", len(catalog), out)
    return out


def export_leaderboard(output_dir: Path) -> Path:
    """Best F1/accuracy per model x dataset x scale."""
    conn = get_connection()
    conn.row_factory = _dict_factory
    rows = conn.execute("""
        SELECT
            r.dataset, r.model_type, r.scale, r.has_kd,
            m.model, m.metric_name,
            ROUND(MAX(m.value), 6) AS best_value
        FROM metrics m
        JOIN runs r ON r.run_id = m.run_id
        WHERE m.scenario = 'val'
          AND m.metric_name IN ('f1', 'accuracy', 'precision', 'recall', 'auc', 'mcc')
          AND r.status = 'complete'
        GROUP BY r.dataset, r.model_type, r.scale, r.has_kd, m.model, m.metric_name
        ORDER BY r.dataset, m.model, m.metric_name
    """).fetchall()
    conn.close()

    out = output_dir / "leaderboard.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d leaderboard entries → %s", len(rows), out)
    return out


def export_runs(output_dir: Path) -> Path:
    """All completed runs with config and status."""
    conn = get_connection()
    conn.row_factory = _dict_factory
    rows = conn.execute("""
        SELECT
            run_id, dataset, model_type, scale, stage,
            has_kd, status, teacher_run, started_at, completed_at
        FROM runs
        ORDER BY started_at DESC
    """).fetchall()
    conn.close()

    out = output_dir / "runs.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d runs → %s", len(rows), out)
    return out


def export_metrics(output_dir: Path) -> Path:
    """Per-run flattened metrics."""
    conn = get_connection()
    conn.row_factory = _dict_factory

    run_ids = [r["run_id"] for r in conn.execute(
        "SELECT DISTINCT run_id FROM metrics"
    ).fetchall()]

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for rid in run_ids:
        rows = conn.execute(
            "SELECT model, scenario, metric_name, value FROM metrics WHERE run_id = ?",
            (rid,),
        ).fetchall()
        # Use sanitized filename
        fname = rid.replace("/", "_") + ".json"
        (metrics_dir / fname).write_text(json.dumps(_versioned_envelope(rows), indent=2))

    conn.close()
    log.info("Exported metrics for %d runs → %s", len(run_ids), metrics_dir)
    return metrics_dir


def export_datasets(output_dir: Path) -> Path:
    """Dataset metadata."""
    conn = get_connection()
    conn.row_factory = _dict_factory
    rows = conn.execute("""
        SELECT name, domain, protocol, source, description,
               num_files, num_samples, num_graphs, num_unique_ids,
               attack_types
        FROM datasets
        ORDER BY name
    """).fetchall()
    conn.close()

    out = output_dir / "datasets.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d datasets → %s", len(rows), out)
    return out


def export_kd_transfer(output_dir: Path) -> Path:
    """Teacher vs student metric pairs for KD analysis."""
    conn = get_connection()
    conn.row_factory = _dict_factory

    rows = conn.execute("""
        SELECT
            student.run_id   AS student_run,
            student.dataset,
            student.model_type,
            student.scale     AS student_scale,
            teacher.run_id    AS teacher_run,
            sm.metric_name,
            ROUND(sm.value, 6) AS student_value,
            ROUND(tm.value, 6) AS teacher_value
        FROM runs student
        JOIN metrics sm ON sm.run_id = student.run_id AND sm.scenario = 'val'
        JOIN runs teacher ON teacher.dataset = student.dataset
                          AND teacher.stage = 'evaluation'
                          AND teacher.has_kd = 0
        JOIN metrics tm ON tm.run_id = teacher.run_id
                       AND tm.scenario = 'val'
                       AND tm.metric_name = sm.metric_name
                       AND tm.model = sm.model
        WHERE student.has_kd = 1
          AND student.stage = 'evaluation'
          AND sm.metric_name IN ('f1', 'accuracy', 'auc')
          AND teacher.model_type = student.model_type
          AND student.scale = 'small' AND teacher.scale = 'large'
        ORDER BY student.dataset, sm.metric_name
    """).fetchall()
    conn.close()

    out = output_dir / "kd_transfer.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d KD transfer pairs → %s", len(rows), out)
    return out


def export_training_curves(output_dir: Path) -> Path:
    """Per-run training curves from epoch_metrics table."""
    conn = get_connection()
    conn.row_factory = _dict_factory

    run_ids = [r["run_id"] for r in conn.execute(
        "SELECT DISTINCT run_id FROM epoch_metrics"
    ).fetchall()]

    curves_dir = output_dir / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    for rid in run_ids:
        rows = conn.execute(
            "SELECT epoch, metric_name, value FROM epoch_metrics WHERE run_id = ? ORDER BY epoch, metric_name",
            (rid,),
        ).fetchall()
        fname = rid.replace("/", "_") + ".json"
        (curves_dir / fname).write_text(json.dumps(_versioned_envelope(rows), indent=2))

    conn.close()
    log.info("Exported training curves for %d runs → %s", len(run_ids), curves_dir)
    return curves_dir


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

        # Stratify by label: 1 normal + 1 attack + 1 random (seeded)
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
        # Fill remaining slots (up to 3) with random picks
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
    # Representative num_ids and in_ch for param counting
    num_ids = 30  # typical across datasets
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
        # Proportional allocation
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

    exp_root = Path("experimentruns")
    count = 0
    exported_files: list[str] = []

    if not exp_root.is_dir():
        index_path = embed_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        log.info("No experimentruns directory — wrote empty embeddings index")
        return embed_dir

    for ds_dir in sorted(exp_root.iterdir()):
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
                model_name = model_key.split("_")[0]  # "vgae" or "gat"
                labels = data.get(f"{model_name}_labels", np.array([]))
                errors = data.get(f"{model_name}_errors", np.array([]))

                if embeddings.shape[0] < 3:
                    continue

                for method in ("umap", "pymde"):
                    try:
                        coords_2d = _reduce_embeddings(embeddings, method)
                    except Exception as e:
                        log.warning("Dimensionality reduction (%s) failed for %s/%s: %s",
                                    method, run_id, model_name, e)
                        continue

                    total_original = len(coords_2d)
                    coords_2d, s_labels, s_errors, was_sampled = _stratified_sample(
                        coords_2d, labels, errors, EMBEDDING_MAX_SAMPLES
                    )
                    if was_sampled:
                        log.info("Sampled embeddings %s/%s/%s: %d → %d",
                                 run_id, model_name, method, total_original, len(coords_2d))

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

    # Write index file
    index_path = embed_dir / "index.json"
    index_path.write_text(json.dumps(_versioned_envelope(sorted(exported_files)), indent=2))

    log.info("Exported %d embedding projections → %s", count, embed_dir)
    return embed_dir


def _reduce_embeddings(embeddings, method: str):
    """Reduce high-dimensional embeddings to 2D."""
    import numpy as np

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

    exp_root = Path("experimentruns")
    count = 0
    exported_files: list[str] = []

    if not exp_root.is_dir():
        index_path = policy_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        log.info("No experimentruns directory — wrote empty dqn_policy index")
        return policy_dir

    for ds_dir in sorted(exp_root.iterdir()):
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

    # Write index file
    index_path = policy_dir / "index.json"
    index_path.write_text(json.dumps(_versioned_envelope(sorted(exported_files)), indent=2))

    log.info("Exported %d DQN policy files → %s", count, policy_dir)
    return policy_dir


def export_roc_curves(output_dir: Path) -> Path:
    """Export ROC/PR curve arrays from evaluation metrics.json files."""
    curves_dir = output_dir / "roc_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    exp_root = Path("experimentruns")
    count = 0
    exported_files: list[str] = []

    if not exp_root.is_dir():
        index_path = curves_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        return curves_dir

    for ds_dir in sorted(exp_root.iterdir()):
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

    exp_root = Path("experimentruns")
    count = 0
    exported_files: list[str] = []

    if not exp_root.is_dir():
        index_path = attn_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        return attn_dir

    for ds_dir in sorted(exp_root.iterdir()):
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

                # Collect attention per layer
                layers = []
                layer_idx = 0
                while f"{prefix}_layer_{layer_idx}_alpha" in data:
                    alpha = data[f"{prefix}_layer_{layer_idx}_alpha"]
                    # alpha shape: [n_edges, n_heads] or [n_edges]
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

    exp_root = Path("experimentruns")
    count = 0
    exported_files: list[str] = []

    if not exp_root.is_dir():
        index_path = recon_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        return recon_dir

    for ds_dir in sorted(exp_root.iterdir()):
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

            # Read optimal threshold from metrics.json if available
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

    exp_root = Path("experimentruns")
    count = 0
    exported_files: list[str] = []

    if not exp_root.is_dir():
        index_path = cka_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        return cka_dir

    for ds_dir in sorted(exp_root.iterdir()):
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


def export_all(output_dir: Path) -> None:
    """Run all exports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight: validate DB schema and data quality
    conn = get_connection()
    try:
        _validate_schema(conn)
        _validate_data(conn)
    finally:
        conn.close()

    lb = export_leaderboard(output_dir)
    runs = export_runs(output_dir)
    metrics = export_metrics(output_dir)
    ds = export_datasets(output_dir)
    kd = export_kd_transfer(output_dir)
    curves = export_training_curves(output_dir)
    export_metric_catalog(output_dir)

    # New exports (graceful — failures don't block other exports)
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

    # Validation summary — check file sizes
    for name, path in [
        ("leaderboard", lb), ("runs", runs), ("datasets", ds), ("kd_transfer", kd),
    ]:
        if path.stat().st_size < 10:
            log.warning("EMPTY EXPORT: %s (%s)", name, path)

    log.info("All exports complete → %s", output_dir)


def _dict_factory(cursor, row):
    """SQLite row factory that returns dicts."""
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline.export",
        description="Export project DB to static JSON for dashboard",
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
