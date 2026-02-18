# TODO — KD-GAT

**Last updated**: 2026-02-18

---

## P0 — Verify & Commit

| # | Task | Category | Status | Notes |
|---|------|----------|--------|-------|
| 1 | ~~Sync W&B offline runs~~ | ops | **done** | Synced 2026-02-18. All 80 runs now online. |
| 2 | ~~Hard-refresh dashboard and verify panels~~ | dashboard | **done** | All panels have data except Confusion Matrix (no export). S3 sync verified (270 objects). |
| 3 | ~~Verify embedding panel fix end-to-end~~ | dashboard | **done** | Embedding data in S3. Model Comparison has data (`leaderboard.json` + `model_sizes.json`). |

## P1 — Documentation

| # | Task | Category | Status | Notes |
|---|------|----------|--------|-------|
| 4 | ~~Update STATE.md~~ | docs | **done** | Rewritten 2026-02-18 with full ecosystem state. |
| 5 | ~~Update CLAUDE.md export docs~~ | docs | **done** | Added `--skip-heavy`/`--only-heavy`, SLURM export, S3 sourcing. |
| 6 | ~~Update TODO.md~~ | docs | **done** | This file. Marked dashboard rework phases complete, added assessment items. |

## P2 — Dashboard Polish

| # | Task | Category | Status | Notes |
|---|------|----------|--------|-------|
| 7 | ~~Verify all 27 panels systematically~~ | dashboard | **done** | All panels have S3 data except Confusion Matrix (needs eval export). Not worth fixing for static site. |
| 8 | Add loading/error states for S3 fetch failures | dashboard | pending | `_loadJSON` returns `[]` on failure with console.warn. Show user-visible message. |
| 9 | Add S3 cache headers | infra | pending | `Cache-Control: max-age=3600` on `aws s3 sync` to reduce load times. |
| 10 | Test local dev workflow (`python -m http.server`) | dashboard | pending | Confirm `BASE` falls back to `data/` on localhost. |

## P3 — Infrastructure

| # | Task | Category | Status | Notes |
|---|------|----------|--------|-------|
| 11 | GitHub Actions CI | ci | pending | Lint JS, run lightweight pytest subset, validate config schema. |
| 12 | Export parallelization | perf | pending | `ProcessPoolExecutor` for UMAP/PyMDE. Cut heavy export from ~10min to ~2-3min. |
| 13 | Inference server validation | infra | pending | Test `/predict` endpoint with current checkpoints end-to-end. |
| 14 | Drop PyMDE from default export | perf | pending | Make PyMDE opt-in (`--methods umap,pymde`) to halve embedding export time. |

## P4 — Research

| # | Task | Category | Status | Notes |
|---|------|----------|--------|-------|
| 15 | OOD threshold calibration | research | pending | VGAE reconstruction thresholds don't transfer across datasets. Research question. |
| 16 | JumpReLU SAE integration | research | pending | Sparse autoencoder for GAT attention interpretability. |
| 17 | Cascading KD (VGAE → GAT → DQN) | research | pending | Propagate knowledge distillation across full pipeline. |
| 18 | Additional datasets (ROAD, SynCAN) | data | pending | Expand beyond 6 automotive CAN datasets. |

## Completed (Archive)

| Task | Date | Notes |
|------|------|-------|
| Dashboard rework Phase 1-4 | 2026-02-17 | S3 data source, embedding fix, timestamps, color theme |
| Export bug: `_scan_runs` config.json parsing | 2026-02-18 | Prefer config fields over dir name parsing |
| Export bug: training_curves index.json | 2026-02-18 | Generate index.json listing exported curve files |
| PR #2: Platform migration | 2026-02-18 | W&B/Prefect/S3 replaces Snakemake/SQLite/MLflow |
| Legacy cleanup | 2026-02-18 | Removed all Snakemake/SQLite/MLflow references |
| 72 training runs | 2026-02-17 | All 6 datasets × 12 configs complete |
| S3 bucket policy for public read | 2026-02-17 | `aws s3api put-bucket-policy` applied |
| W&B offline runs synced | 2026-02-18 | 3 offline runs synced, all 80 runs now online |
| Dashboard panels verified | 2026-02-18 | All panels have S3 data (270 objects). Confusion Matrix excluded (no export). |
| Model Comparison verified | 2026-02-18 | `leaderboard.json` + `model_sizes.json` both in S3, panel has data. |

---

## Categories

- **infra**: S3, IAM, deployment
- **dashboard**: D3.js panels, charts, UX
- **ops**: W&B, SLURM, routine maintenance
- **docs**: CLAUDE.md, STATE.md, README
- **perf**: Performance optimization
- **research**: ML experiments, new methods
- **data**: Dataset ingestion, preprocessing
- **ci**: Continuous integration
