# P3 Infrastructure Implementation + Code Review

## Context

P0–P2 are done (W&B synced, docs updated, dashboard verified, export bugs fixed). This plan covers the four remaining P3 infrastructure items plus a code review/cleanup pass. The goal is to add CI guardrails, speed up heavy exports, validate the inference server, and clean up the codebase.

---

## Implementation Order

```
Step 1: pipeline/export.py  — --methods flag + parallelization (items 14 + 12)
Step 2: tests/test_serve.py — new mock-based serve tests (item 13)
Step 3: .github/workflows/  — CI workflow (item 11)
Step 4: Code review + cleanup (serve.py security, pyproject.toml, dead code)
Step 5: Commit
```

---

## Step 1: Export Changes (`pipeline/export.py`)

### 1a: `--methods` flag (item 14 — drop PyMDE from default)

- Add `--methods` CLI arg: default `"umap"`, accepts comma-separated `"umap,pymde"`
- Add `methods: tuple[str, ...]` param to `export_all()` and `export_embeddings()`
- Replace hardcoded `for method in ("umap", "pymde"):` (line 555) with `for method in methods:`
- Thread through: `main()` parses → `export_all(methods=...)` → special-case embeddings call in artifact loop

### 1b: Parallelization (item 12)

- Add `--workers` CLI arg (default `1`)
- Extract `_process_embedding_run()` as **module-level** function (must be pickle-serializable):
  - Signature: `(npz_path: Path, run_id: str, embed_dir: Path, methods: tuple[str, ...]) -> list[str]`
  - Contains: load npz → iterate model keys → `_stratified_sample` → `_reduce_embeddings` → write JSON
  - Returns list of exported filenames
- In `export_embeddings()`: collect `(npz_path, run_id)` pairs, then dispatch:
  - `workers > 1`: `ProcessPoolExecutor(max_workers)` with `as_completed`
  - `workers == 1`: sequential loop (current behavior)
- Update `scripts/export_dashboard_slurm.sh`: add `--workers 4` (matches 4 CPUs in SBATCH)

---

## Step 2: Serve Tests (`tests/test_serve.py`) — New File

All mock-based, no GPU/data/SLURM needed. Uses `fastapi.testclient.TestClient`.

**Helper:** `_make_mock_models()` → dict with:
- Mock GAT: returns `torch.tensor([[0.1, 0.9]])` (attack logits)
- Mock VGAE: returns `(torch.randn(10, 10), None, None, None, None)`
- Mock DQN: `select_action` returns `(0.7, None, None)`, `q_network[0].in_features = 15`

**Tests (7):**
1. `test_health_empty` — GET `/health` → 200, empty models_loaded
2. `test_health_schema` — response has status, models_loaded, device
3. `test_predict_no_models_503` — POST `/predict` → 503 when checkpoints missing
4. `test_predict_mocked` — patch `_load_models`, verify 200 + all response fields
5. `test_predict_without_dqn` — no DQN → alpha defaults to 0.5
6. `test_predict_edge_index_transpose` — [N,2] input gets transposed to [2,N]
7. `test_request_validation` — malformed body → 422

**Fixture:** `client()` creates `TestClient(app)`, clears `_models` before/after.

---

## Step 3: GitHub Actions CI (`.github/workflows/ci.yml`) — New File

Three parallel jobs:

**`lint`:** `pip install ruff` → `ruff check config/ pipeline/ src/ tests/`

**`js-syntax`:** `npx -y acorn --ecma2022 --module` on dashboard JS files

**`test`:**
- CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- `pip install torch-geometric && pip install -e ".[dev]"`
- `pytest tests/test_layer_boundaries.py tests/test_serve.py tests/test_registry.py tests/test_pipeline_integration.py -v -m "not slurm"`
- slurm-marked tests auto-skip (no `SLURM_JOB_ID` in CI env)

**Triggers:** push to main, PRs to main.

---

## Step 4: Code Review + Cleanup

### `pyproject.toml`
- Add `httpx>=0.27` to `[dev]` deps (TestClient needs it)
- Add `slurm` to pytest markers (suppress warning)
- Add `[tool.ruff]` config: target-version py311, select E/F/W/I, ignore E501
- Remove `mlflow` from base deps (legacy — code references already cleaned)

### `pipeline/serve.py` — Security
- Add `max_length` on `PredictRequest.node_features` (1000) and `edge_index` (10000) to prevent OOM
- Sanitize error message in `_load_models` except block (don't leak internal paths)

### General
- Verify `_process_embedding_run` is pickle-safe (no closures)
- Run `ruff check` and fix any issues surfaced
- Run `test_layer_boundaries.py` to confirm no import violations

---

## Files Modified/Created

| File | Action |
|------|--------|
| `pipeline/export.py` | Edit: `--methods`, `--workers`, `_process_embedding_run()` |
| `pipeline/serve.py` | Edit: input size limits, error sanitization |
| `tests/test_serve.py` | **New**: 7 mock-based tests |
| `.github/workflows/ci.yml` | **New**: lint + js-syntax + test jobs |
| `pyproject.toml` | Edit: dev deps, ruff config, markers, remove mlflow |
| `scripts/export_dashboard_slurm.sh` | Edit: add `--workers 4` |

## Verification

- `ruff check config/ pipeline/ src/ tests/` passes
- `pytest tests/test_layer_boundaries.py tests/test_serve.py -v` passes
- `python -m pipeline.export --skip-heavy` runs (methods flag defaults to umap)
- `python -m pipeline.export --methods umap,pymde --only-heavy --workers 1` runs
- Dashboard JS files pass acorn syntax check
