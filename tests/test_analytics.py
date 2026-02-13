"""Tests for pipeline.analytics query functions.

Uses a populated_db fixture with synthetic runs, configs, and metrics.
Run with:  python -m pytest tests/test_analytics.py -v
"""
from __future__ import annotations

import json

import pytest

from pipeline.db import get_connection
from pipeline.analytics import (
    sweep,
    leaderboard,
    compare,
    config_diff,
    dataset_summary,
    _validate_param,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def populated_db(tmp_path):
    """Create a temporary SQLite DB with sample runs, configs, and metrics."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)

    # Insert datasets
    conn.execute(
        "INSERT INTO datasets (name, domain) VALUES (?, ?)",
        ("ds_a", "automotive"),
    )
    conn.execute(
        "INSERT INTO datasets (name, domain) VALUES (?, ?)",
        ("ds_b", "automotive"),
    )

    # Insert runs with config_json
    runs = [
        ("ds_a/vgae_large_autoencoder", "ds_a", "vgae", "large", "autoencoder", 0, "complete",
         json.dumps({"lr": 0.002, "batch_size": 4096, "gat_hidden": 48, "seed": 42})),
        ("ds_a/vgae_small_autoencoder_kd", "ds_a", "vgae", "small", "autoencoder", 1, "complete",
         json.dumps({"lr": 0.001, "batch_size": 2048, "gat_hidden": 24, "seed": 42})),
        ("ds_a/gat_large_curriculum", "ds_a", "gat", "large", "curriculum", 0, "complete",
         json.dumps({"lr": 0.002, "batch_size": 4096, "gat_hidden": 48, "seed": 42})),
        ("ds_a/vgae_large_evaluation", "ds_a", "vgae", "large", "evaluation", 0, "complete",
         json.dumps({"lr": 0.002, "batch_size": 4096, "gat_hidden": 48, "seed": 42})),
        ("ds_a/vgae_small_evaluation_kd", "ds_a", "vgae", "small", "evaluation", 1, "complete",
         json.dumps({"lr": 0.001, "batch_size": 2048, "gat_hidden": 24, "seed": 42})),
        ("ds_b/vgae_large_autoencoder", "ds_b", "vgae", "large", "autoencoder", 0, "complete",
         json.dumps({"lr": 0.005, "batch_size": 4096, "gat_hidden": 48, "seed": 99})),
    ]
    for run_id, dataset, model_type, scale, stage, has_kd, status, config_json in runs:
        conn.execute(
            """INSERT INTO runs (run_id, dataset, model_type, scale, stage, has_kd, status, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, dataset, model_type, scale, stage, has_kd, status, config_json),
        )

    # Insert metrics
    metrics = [
        # ds_a large eval
        ("ds_a/vgae_large_evaluation", "gat", "val", "f1", 0.95),
        ("ds_a/vgae_large_evaluation", "gat", "val", "accuracy", 0.96),
        ("ds_a/vgae_large_evaluation", "gat", "val", "auc", 0.98),
        ("ds_a/vgae_large_evaluation", "vgae", "val", "f1", 0.88),
        # ds_a small+kd eval
        ("ds_a/vgae_small_evaluation_kd", "gat", "val", "f1", 0.92),
        ("ds_a/vgae_small_evaluation_kd", "gat", "val", "accuracy", 0.93),
        ("ds_a/vgae_small_evaluation_kd", "gat", "val", "auc", 0.96),
        ("ds_a/vgae_small_evaluation_kd", "vgae", "val", "f1", 0.85),
        # ds_a large autoencoder (training metrics)
        ("ds_a/vgae_large_autoencoder", "vgae", "val", "f1", 0.87),
        # ds_b large autoencoder
        ("ds_b/vgae_large_autoencoder", "vgae", "val", "f1", 0.90),
    ]
    for run_id, model, scenario, metric_name, value in metrics:
        conn.execute(
            """INSERT INTO metrics (run_id, model, scenario, metric_name, value)
               VALUES (?, ?, ?, ?, ?)""",
            (run_id, model, scenario, metric_name, value),
        )

    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestValidateParam:

    def test_valid_names(self):
        assert _validate_param("lr") == "lr"
        assert _validate_param("batch_size") == "batch_size"
        assert _validate_param("gat_hidden") == "gat_hidden"
        assert _validate_param("_private") == "_private"

    def test_rejects_sql_injection(self):
        with pytest.raises(ValueError, match="Invalid parameter name"):
            _validate_param("lr; DROP TABLE runs")

    def test_rejects_json_path_traversal(self):
        with pytest.raises(ValueError, match="Invalid parameter name"):
            _validate_param("$.lr")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Invalid parameter name"):
            _validate_param("")

    def test_rejects_numeric_start(self):
        with pytest.raises(ValueError, match="Invalid parameter name"):
            _validate_param("1lr")


# ---------------------------------------------------------------------------
# sweep()
# ---------------------------------------------------------------------------

class TestSweep:

    def test_groups_by_param(self, populated_db):
        rows = sweep("lr", "f1", db_path=populated_db)
        assert len(rows) >= 2
        lr_values = {r["param_value"] for r in rows}
        assert 0.002 in lr_values
        assert 0.001 in lr_values

    def test_returns_stats(self, populated_db):
        rows = sweep("lr", "f1", db_path=populated_db)
        for row in rows:
            assert "count" in row
            assert "min" in row
            assert "max" in row
            assert "mean" in row
            assert row["count"] >= 1

    def test_filter_by_dataset(self, populated_db):
        rows = sweep("lr", "f1", dataset="ds_b", db_path=populated_db)
        assert len(rows) >= 1
        assert all(r["param_value"] == 0.005 for r in rows)

    def test_missing_param_returns_empty(self, populated_db):
        rows = sweep("nonexistent_param", "f1", db_path=populated_db)
        assert rows == []


# ---------------------------------------------------------------------------
# leaderboard()
# ---------------------------------------------------------------------------

class TestLeaderboard:

    def test_descending_order(self, populated_db):
        rows = leaderboard("f1", top=10, db_path=populated_db)
        values = [r["value"] for r in rows]
        assert values == sorted(values, reverse=True)

    def test_top_limit(self, populated_db):
        rows = leaderboard("f1", top=3, db_path=populated_db)
        assert len(rows) <= 3

    def test_includes_run_info(self, populated_db):
        rows = leaderboard("f1", top=1, db_path=populated_db)
        assert len(rows) == 1
        row = rows[0]
        assert "run_id" in row
        assert "dataset" in row
        assert "stage" in row
        assert row["value"] == 0.95  # best f1 in fixture

    def test_filter_by_model(self, populated_db):
        rows = leaderboard("f1", model="vgae", db_path=populated_db)
        assert all(r["model"] == "vgae" for r in rows)


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------

class TestCompare:

    def test_shows_deltas(self, populated_db):
        rows = compare(
            "ds_a/vgae_large_evaluation", "ds_a/vgae_small_evaluation_kd",
            db_path=populated_db,
        )
        assert len(rows) > 0
        f1_rows = [r for r in rows if r["metric_name"] == "f1" and r["model"] == "gat"]
        assert len(f1_rows) == 1
        assert f1_rows[0]["value_a"] == 0.95
        assert f1_rows[0]["value_b"] == 0.92
        assert abs(f1_rows[0]["delta"] - (-0.03)) < 1e-5

    def test_missing_run_raises(self, populated_db):
        with pytest.raises(KeyError, match="Run not found"):
            compare("ds_a/vgae_large_evaluation", "nonexistent/run", db_path=populated_db)

    def test_both_missing_raises(self, populated_db):
        with pytest.raises(KeyError, match="Run not found"):
            compare("no/run_a", "no/run_b", db_path=populated_db)


# ---------------------------------------------------------------------------
# config_diff()
# ---------------------------------------------------------------------------

class TestConfigDiff:

    def test_finds_differences(self, populated_db):
        diffs = config_diff(
            "ds_a/vgae_large_autoencoder", "ds_a/vgae_small_autoencoder_kd",
            db_path=populated_db,
        )
        diff_params = {d["param"] for d in diffs}
        assert "lr" in diff_params
        assert "batch_size" in diff_params
        assert "gat_hidden" in diff_params

    def test_excludes_same_values(self, populated_db):
        diffs = config_diff(
            "ds_a/vgae_large_autoencoder", "ds_a/vgae_small_autoencoder_kd",
            db_path=populated_db,
        )
        diff_params = {d["param"] for d in diffs}
        assert "seed" not in diff_params  # both have seed=42

    def test_missing_run_raises(self, populated_db):
        with pytest.raises(KeyError, match="Run not found"):
            config_diff("ds_a/vgae_large_autoencoder", "nonexistent/run", db_path=populated_db)

    def test_identical_configs_empty(self, populated_db):
        diffs = config_diff(
            "ds_a/vgae_large_autoencoder", "ds_a/gat_large_curriculum",
            db_path=populated_db,
        )
        assert diffs == []


# ---------------------------------------------------------------------------
# dataset_summary()
# ---------------------------------------------------------------------------

class TestDatasetSummary:

    def test_returns_all_runs(self, populated_db):
        result = dataset_summary("ds_a", db_path=populated_db)
        assert result["dataset"] == "ds_a"
        assert len(result["runs"]) == 5

    def test_returns_best_metrics(self, populated_db):
        result = dataset_summary("ds_a", db_path=populated_db)
        assert "f1" in result["best_metrics"]
        assert result["best_metrics"]["f1"]["best_value"] == 0.95

    def test_empty_dataset(self, populated_db):
        result = dataset_summary("nonexistent_ds", db_path=populated_db)
        assert result["runs"] == []
        assert result["best_metrics"] == {}
