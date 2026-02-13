# Snakemake Features for ML Research Workflows

*Audit of underutilized Snakemake features for GNN/DRL pipelines on OSC Pitzer (Snakemake 9.x)*

---

## Implemented Features

### Benchmark Directive (**DONE**)

All training and evaluation rules have `benchmark:` directives logging wall time, max RSS/VMS, and I/O to TSV files under `experimentruns/{ds}/{run}/benchmark.tsv`.

### Retries with Attempt-Based Resource Scaling (**DONE**)

All 9 training rules have `retries: 2` with dynamic memory: `mem_mb=lambda wc, attempt: 128000 * attempt`. First attempt uses 128GB; OOM auto-retries with 256GB.

### Between-Workflow Caching (**DONE**)

Dedicated `preprocess` rule with `cache: True` warms the graph cache per dataset. All training rules depend on it. Cache location: `SNAKEMAKE_OUTPUT_CACHE=/fs/scratch/PAS1266/snakemake-cache/`. Profile has `cache: all`.

### Group Jobs — Batch SLURM Submissions (**DONE**)

Three evaluation rules (`eval_large`, `eval_small_kd`, `eval_small_nokd`) use `group: "evaluation"` to bundle into single SLURM submissions per dataset, reducing scheduler overhead.

---

## Not Yet Implemented

### Paramspace — Hyperparameter Sweep Management

Replaces manual wildcard construction for HP sweeps. Takes a Pandas DataFrame and auto-generates wildcard patterns and directory structures.

```python
from snakemake.utils import Paramspace
import pandas as pd

paramspace = Paramspace(pd.read_csv("params.tsv", sep="\t"))

rule train_gnn:
    input: "data/can_bus/{dataset}.parquet"
    output: f"results/{paramspace.wildcard_pattern}/model.pt"
    params: **paramspace.instance
    script: "scripts/train.py"
```

- `single_wildcard` argument encodes the entire paramspace into one wildcard for cleaner paths
- Create sub-spaces with standard Pandas ops (`.loc`, `.filter()`) for ablation studies
- Each `paramspace.instance` maps naturally to a unique MLflow run

### Jupyter Notebook Integration

Use `notebook:` in rules for interactive development, then `snakemake --edit-notebook` to draft:

```python
rule analyze_results:
    input: expand("results/{p}/metrics.json", p=paramspace.instance_patterns)
    output: "reports/analysis.ipynb"
    log: notebook="logs/analysis.ipynb"
    notebook: "notebooks/analyze.py.ipynb"
```

The `snakemake` object is injected with access to `.input`, `.output`, `.params`, etc.

### Module Pattern for Preprocessing

Reusable CAN bus preprocessing module via Snakemake's `module` system:

```python
module can_preprocess:
    snakefile: "modules/can_preprocess/Snakefile"
    config: config

use rule * from can_preprocess
```

Overkill at current scale (single pipeline), but useful if experiment branches diverge significantly.

---

## Priority Summary

| Feature | Status | Effort | Impact |
|---------|--------|--------|--------|
| Benchmark directive | **Done** | — | Solves GPU memory estimation |
| Retries + resource scaling | **Done** | — | Auto-recovers from OOM |
| Between-workflow caching | **Done** | — | Saves compute across experiments |
| Group jobs | **Done** | — | Reduces SLURM overhead |
| Paramspace | Not started | ~1 hr | Eliminates HP sweep boilerplate |
| `--rulegraph` for docs | Trivial | ~5 min | Immediate onboarding value |
| Notebook integration | Not started | ~30 min | Self-documenting analysis |
| Module pattern | Not started | ~half day | DRY shared pipeline code |
