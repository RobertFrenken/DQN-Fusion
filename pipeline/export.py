"""Export experiment results to static JSON for the GitHub Pages dashboard.

Data sources:
  - Datalake: data/datalake/*.parquet (primary — metadata + metrics)
  - Filesystem: experimentruns/{ds}/{run}/ (binary artifacts)
  - Catalog: config/datasets.yaml

Usage:
    python -m pipeline.export [--output-dir docs/dashboard/data]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("docs/dashboard/data")
EXPERIMENT_ROOT = Path("experimentruns")
_DATALAKE_ROOT = Path("data/datalake")


def _versioned_envelope(data: list | dict) -> dict:
    """Wrap export data with schema version and timestamp."""
    return {
        "schema_version": "1.0.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }


# ---------------------------------------------------------------------------
# Data source: datalake Parquet (primary) with filesystem fallback
# ---------------------------------------------------------------------------

def _scan_runs() -> list[dict]:
    """Load run metadata from datalake Parquet, with filesystem dir paths.

    Falls back to filesystem scan if datalake doesn't exist.
    """
    runs_parquet = _DATALAKE_ROOT / "runs.parquet"
    if runs_parquet.exists():
        return _scan_runs_from_datalake()
    return _scan_runs_from_filesystem()


def _scan_runs_from_datalake() -> list[dict]:
    """Read run metadata from datalake Parquet, attach filesystem paths."""
    import duckdb

    datalake = str(_DATALAKE_ROOT)
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT run_id, dataset, model_type, scale, stage, has_kd, success
        FROM '{datalake}/runs.parquet'
        ORDER BY dataset, run_id
    """).fetchall()
    con.close()

    runs = []
    for run_id, dataset, model_type, scale, stage, has_kd, success in rows:
        run_dir = EXPERIMENT_ROOT / run_id
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        try:
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        except Exception:
            cfg = {}

        has_metrics = (run_dir / "metrics.json").exists()
        has_checkpoint = (run_dir / "best_model.pt").exists()

        runs.append({
            "run_id": run_id,
            "dataset": dataset,
            "model_type": model_type,
            "scale": scale,
            "stage": stage,
            "has_kd": 1 if has_kd else 0,
            "status": "complete" if has_metrics or has_checkpoint else "running",
            "config": cfg,
            "dir": run_dir,
        })
    return runs


def _scan_runs_from_filesystem() -> list[dict]:
    """Legacy filesystem scan (fallback when datalake doesn't exist)."""
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

            parts = run_dir.name.split("_")
            model_type = cfg.get("model_type") or (parts[0] if parts else "")
            scale = cfg.get("scale") or (parts[1] if len(parts) > 1 else "")

            _AUX_SUFFIXES = {"kd", "nokd"}
            if cfg.get("stage"):
                stage = cfg["stage"]
            else:
                remaining = parts[2:]
                if remaining and remaining[-1] in _AUX_SUFFIXES:
                    remaining = remaining[:-1]
                stage = "_".join(remaining)

            has_kd = bool(cfg.get("auxiliaries")) or ("_kd" in run_dir.name and "nokd" not in run_dir.name)

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
# Export functions
# ---------------------------------------------------------------------------

_MODEL_KEYS = ("gat", "vgae", "fusion")


def _extract_core_metrics(metrics: dict, base: dict, target_metrics: set) -> list[dict]:
    """Extract core metrics across model keys, returning rows with base fields merged in."""
    rows = []
    for model_key in _MODEL_KEYS:
        core = metrics.get(model_key, {}).get("core", {})
        for name in target_metrics:
            val = core.get(name)
            if isinstance(val, (int, float)):
                rows.append({**base, "model": model_key, "metric_name": name, "best_value": round(val, 6)})
    return rows


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

        base = {
            "dataset": run["dataset"], "model_type": run["model_type"],
            "scale": run["scale"], "has_kd": run["has_kd"],
        }
        rows.extend(_extract_core_metrics(metrics, base, target_metrics))

    out = output_dir / "leaderboard.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d leaderboard entries → %s", len(rows), out)
    return out


def export_runs(output_dir: Path) -> Path:
    """All runs with status."""
    rows = []
    for run in _scan_runs():
        started_at = None
        completed_at = None
        cfg_path = run["dir"] / "config.json"
        if cfg_path.exists():
            started_at = datetime.fromtimestamp(
                cfg_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        for end_file in ("best_model.pt", "metrics.json"):
            p = run["dir"] / end_file
            if p.exists():
                completed_at = datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc
                ).isoformat()
                break

        rows.append({
            "run_id": run["run_id"],
            "dataset": run["dataset"],
            "model_type": run["model_type"],
            "scale": run["scale"],
            "stage": run["stage"],
            "has_kd": run["has_kd"],
            "status": run["status"],
            "teacher_run": "",
            "started_at": started_at,
            "completed_at": completed_at,
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
        for model_key in _MODEL_KEYS:
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
        for model_key in _MODEL_KEYS:
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

    eval_runs: dict[str, list[dict]] = {}
    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        eval_runs.setdefault(run["dataset"], []).append(run)

    for ds, runs in eval_runs.items():
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

        for model_key in _MODEL_KEYS:
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

            csv_logs = list(run_dir.glob("csv_logs/*/metrics.csv")) + \
                       list(run_dir.glob("lightning_logs/*/metrics.csv"))
            if not csv_logs:
                continue

            try:
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
                    exported_files.append(fname)
                    count += 1
            except Exception as e:
                log.warning("Failed to parse CSV log in %s: %s", run_dir, e)

    index_path = curves_dir / "index.json"
    index_path.write_text(json.dumps(_versioned_envelope(sorted(exported_files)), indent=2))

    log.info("Exported training curves for %d runs → %s", count, curves_dir)
    return curves_dir


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


# ---------------------------------------------------------------------------
# Artifact Parquet exports (for Quarto dashboard / DuckDB-WASM)
# ---------------------------------------------------------------------------

_ARTIFACTS_DIR = _DATALAKE_ROOT / "artifacts"


def export_embeddings_parquet() -> Path | None:
    """Export UMAP-reduced embeddings from all eval runs to a single Parquet.

    Reduces high-dim embeddings to 2D via UMAP for scatter plots.
    Output: artifacts/embeddings.parquet with columns:
        run_id, model (vgae|gat), x, y, label
    """
    import numpy as np

    try:
        from umap import UMAP
    except ImportError:
        log.warning("umap-learn not installed; skipping embeddings export")
        return None

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        emb_path = run["dir"] / "embeddings.npz"
        if not emb_path.exists():
            continue

        data = np.load(emb_path)
        for prefix, model_name in [("vgae_z", "vgae"), ("gat_emb", "gat")]:
            if prefix not in data:
                continue
            embeddings = data[prefix]
            labels = data.get(f"{model_name}_labels", data.get("vgae_labels"))
            if labels is None:
                continue

            # Subsample for browser performance (max 2000 points per model per run)
            n = len(embeddings)
            if n > 2000:
                idx = np.random.default_rng(42).choice(n, 2000, replace=False)
                embeddings = embeddings[idx]
                labels = labels[idx]

            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
            xy = reducer.fit_transform(embeddings)

            for i in range(len(xy)):
                rows.append({
                    "run_id": run["run_id"],
                    "model": model_name,
                    "x": float(xy[i, 0]),
                    "y": float(xy[i, 1]),
                    "label": int(labels[i]),
                })

    if not rows:
        return None

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    out = _ARTIFACTS_DIR / "embeddings.parquet"
    pq.write_table(table, out)
    log.info("Exported %d embedding points → %s", len(rows), out)
    return out


def export_attention_parquet() -> Path | None:
    """Export attention weight summaries from all eval runs to Parquet.

    For each sample, computes mean attention per layer across heads.
    Output: artifacts/attention_weights.parquet with columns:
        run_id, sample_idx, label, layer, head, mean_alpha
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        att_path = run["dir"] / "attention_weights.npz"
        if not att_path.exists():
            continue

        data = np.load(att_path)
        n_samples = int(data.get("n_samples", 0))
        # Limit to first 10 samples per run for dashboard
        for si in range(min(n_samples, 10)):
            label = int(data.get(f"sample_{si}_label", -1))
            for layer in range(3):
                key = f"sample_{si}_layer_{layer}_alpha"
                if key not in data:
                    continue
                alpha = data[key]  # (n_edges, n_heads)
                # Mean attention per head across edges
                head_means = alpha.mean(axis=0)  # (n_heads,)
                for head_idx, val in enumerate(head_means):
                    rows.append({
                        "run_id": run["run_id"],
                        "sample_idx": si,
                        "label": label,
                        "layer": layer,
                        "head": head_idx,
                        "mean_alpha": float(val),
                    })

    if not rows:
        return None

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    out = _ARTIFACTS_DIR / "attention_weights.parquet"
    pq.write_table(table, out)
    log.info("Exported %d attention rows → %s", len(rows), out)
    return out


def export_recon_errors_parquet() -> Path | None:
    """Export VGAE reconstruction errors from embeddings.npz files.

    Output: artifacts/recon_errors.parquet with columns:
        run_id, error, label
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        emb_path = run["dir"] / "embeddings.npz"
        if not emb_path.exists():
            continue

        data = np.load(emb_path)
        if "vgae_errors" not in data:
            continue

        errors = data["vgae_errors"]
        labels = data.get("vgae_labels")
        if labels is None:
            continue

        for i in range(len(errors)):
            rows.append({
                "run_id": run["run_id"],
                "error": float(errors[i]),
                "label": int(labels[i]),
            })

    if not rows:
        return None

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    out = _ARTIFACTS_DIR / "recon_errors.parquet"
    pq.write_table(table, out)
    log.info("Exported %d recon error rows → %s", len(rows), out)
    return out


def export_dqn_policy_parquet() -> Path | None:
    """Export DQN alpha policies from all eval runs to Parquet.

    Output: artifacts/dqn_policy.parquet with columns:
        run_id, dataset, scale, has_kd, action_idx, alpha
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        policy_path = run["dir"] / "dqn_policy.json"
        if not policy_path.exists():
            continue

        try:
            policy = json.loads(policy_path.read_text())
        except Exception:
            continue

        alphas = policy.get("alphas", [])
        for idx, alpha in enumerate(alphas):
            rows.append({
                "run_id": run["run_id"],
                "dataset": run["dataset"],
                "scale": run["scale"],
                "has_kd": 1 if run["has_kd"] else 0,
                "action_idx": idx,
                "alpha": float(alpha),
            })

    if not rows:
        return None

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    out = _ARTIFACTS_DIR / "dqn_policy.parquet"
    pq.write_table(table, out)
    log.info("Exported %d DQN policy rows → %s", len(rows), out)
    return out


def export_cka_parquet() -> Path | None:
    """Export CKA similarity matrices from KD eval runs to Parquet.

    Output: artifacts/cka_similarity.parquet with columns:
        run_id, dataset, teacher_layer, student_layer, similarity
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        cka_path = run["dir"] / "cka_matrix.json"
        if not cka_path.exists():
            continue

        try:
            cka = json.loads(cka_path.read_text())
        except Exception:
            continue

        matrix = cka.get("matrix", [])
        teacher_layers = cka.get("teacher_layers", [])
        student_layers = cka.get("student_layers", [])

        for ti, t_layer in enumerate(teacher_layers):
            for si, s_layer in enumerate(student_layers):
                if ti < len(matrix) and si < len(matrix[ti]):
                    rows.append({
                        "run_id": run["run_id"],
                        "dataset": run["dataset"],
                        "teacher_layer": t_layer,
                        "student_layer": s_layer,
                        "similarity": float(matrix[ti][si]),
                    })

    if not rows:
        return None

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    out = _ARTIFACTS_DIR / "cka_similarity.parquet"
    pq.write_table(table, out)
    log.info("Exported %d CKA rows → %s", len(rows), out)
    return out


def export_artifacts_parquet() -> list[Path]:
    """Run all artifact Parquet exports. Returns list of created files."""
    results = []
    for fn in [
        export_recon_errors_parquet,
        export_attention_parquet,
        export_dqn_policy_parquet,
        export_cka_parquet,
    ]:
        try:
            path = fn()
            if path:
                results.append(path)
        except Exception as e:
            log.warning("%s failed: %s", fn.__name__, e)

    # Embeddings export needs umap-learn and is slow; run separately
    try:
        path = export_embeddings_parquet()
        if path:
            results.append(path)
    except Exception as e:
        log.warning("export_embeddings_parquet failed: %s", e)

    return results


def export_data_for_reports(reports_data_dir: Path | None = None) -> None:
    """Copy datalake Parquet + artifact Parquet to reports/data/ for Quarto.

    This is the bridge between the pipeline datalake and the Quarto site.
    FileAttachment in OJS cells loads from reports/data/.
    """
    import shutil

    if reports_data_dir is None:
        reports_data_dir = Path("reports/data")
    reports_data_dir.mkdir(parents=True, exist_ok=True)

    # Core datalake files
    for name in ["metrics.parquet", "runs.parquet", "datasets.parquet"]:
        src = _DATALAKE_ROOT / name
        if src.exists():
            shutil.copy2(src, reports_data_dir / name)
            log.info("Copied %s → reports/data/", name)

    # Training curves: merge all into a single file for easy DuckDB-WASM loading
    tc_dir = _DATALAKE_ROOT / "training_curves"
    if tc_dir.is_dir():
        import pyarrow.parquet as pq
        import pyarrow as pa

        tables = []
        for f in sorted(tc_dir.glob("*.parquet")):
            tables.append(pq.read_table(f))
        if tables:
            merged = pa.concat_tables(tables)
            out = reports_data_dir / "training_curves.parquet"
            pq.write_table(merged, out)
            log.info("Merged %d training curve files → %s", len(tables), out)

    # Artifact Parquet files
    if _ARTIFACTS_DIR.is_dir():
        for f in _ARTIFACTS_DIR.glob("*.parquet"):
            shutil.copy2(f, reports_data_dir / f.name)
            log.info("Copied artifact %s → reports/data/", f.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export_all(output_dir: Path, *, include_reports: bool = False) -> None:
    """Run all exports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lb = export_leaderboard(output_dir)
    runs = export_runs(output_dir)
    export_metrics(output_dir)
    ds = export_datasets(output_dir)
    kd = export_kd_transfer(output_dir)
    export_training_curves(output_dir)
    export_metric_catalog(output_dir)

    for name, path in [
        ("leaderboard", lb), ("runs", runs), ("datasets", ds), ("kd_transfer", kd),
    ]:
        if path.stat().st_size < 10:
            log.warning("EMPTY EXPORT: %s (%s)", name, path)

    try:
        export_model_sizes(output_dir)
    except Exception as e:
        log.warning("Export model_sizes failed (non-fatal): %s", e)

    # Artifact Parquet exports (skip embeddings by default — needs umap-learn)
    artifact_files = export_artifacts_parquet()
    log.info("Exported %d artifact Parquet files", len(artifact_files))

    # Optionally copy everything to reports/data/ for Quarto
    if include_reports:
        export_data_for_reports()

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
    parser.add_argument(
        "--reports", action="store_true",
        help="Also copy Parquet data to reports/data/ for Quarto site",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )

    export_all(args.output_dir, include_reports=args.reports)


if __name__ == "__main__":
    main()
