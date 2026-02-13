# Datasette: Interactive DB Browsing

Datasette provides a web UI for exploring the project SQLite database without writing SQL.

## Installation

```bash
pip install datasette
```

## Quick Start

```bash
# On login node (inside tmux):
datasette data/project.db --port 8001

# Local machine:
ssh -L 8001:localhost:8001 rf15@pitzer.osc.edu
# Open http://localhost:8001
```

## What You Can Do

- **Browse tables**: datasets, runs, metrics — click through with no SQL
- **Filter and facet**: point-and-click filtering by dataset, stage, model_size
- **Export**: CSV, JSON, or API endpoints for any query result
- **Custom SQL**: write and share SQL queries via URL

## Tables Available

| Table | Contents |
|-------|----------|
| `datasets` | Registered datasets with metadata (name, domain, num_samples, etc.) |
| `runs` | Pipeline runs with config_json, status, timestamps |
| `metrics` | Per-run metrics (accuracy, f1, auc, etc.) by model and scenario |

## Example Queries

Browse to `http://localhost:8001/project` and use the SQL query box:

```sql
-- Best F1 by dataset
SELECT r.dataset, MAX(m.value) AS best_f1
FROM metrics m JOIN runs r ON r.run_id = m.run_id
WHERE m.metric_name = 'f1'
GROUP BY r.dataset ORDER BY best_f1 DESC;

-- All config values for a hyperparameter
SELECT r.run_id, json_extract(r.config_json, '$.lr') AS lr, m.value AS f1
FROM runs r JOIN metrics m ON m.run_id = r.run_id
WHERE m.metric_name = 'f1' AND lr IS NOT NULL
ORDER BY f1 DESC;
```

## When to Use Datasette vs Other Tools

| Task | Tool |
|------|------|
| Quick data exploration, sharing links | Datasette |
| Visual metric plots, artifact browsing | MLflow UI |
| Scripted sweep analysis | `pipeline.analytics` CLI |
| Programmatic queries in Python | `pipeline.db.get_connection()` |
