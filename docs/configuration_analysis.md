# KD-GAT Configuration, Outputs & Bugs Analysis

Comprehensive audit of the pipeline's configuration system, experiment outputs, and known bugs.

---

## Part 1: Configuration & SLURM

### 1.1 SLURM Job Creation Flow

Snakemake submits SLURM jobs via `profiles/slurm/config.yaml`, which templates an `sbatch` command using `{resources.*}` placeholders from each rule:

```yaml
# profiles/slurm/config.yaml
cluster: >-
  sbatch
  --account={resources.slurm_account}
  --partition={resources.slurm_partition}
  --time={resources.time_min}
  --mem={resources.mem_mb}M
  --cpus-per-task={resources.cpus_per_task}
  --gpus-per-node={resources.gpus}
  --job-name=smk-{rule}-{wildcards}
  --output=slurm_logs/%j-{rule}.out
  --error=slurm_logs/%j-{rule}.err
  --parsable
```

Two resource dictionaries define all SLURM allocations (`Snakefile:68-75`):

| Resource | Training (`_TRAIN_RES`) | Evaluation (`_EVAL_RES`) |
|----------|------------------------|--------------------------|
| Wall time | 360 min (6 hr) | 60 min (1 hr) |
| Memory | 128 GB | 32 GB |
| CPUs | 16 | 8 |
| GPUs | 1x V100 | 1x V100 |
| Account | PAS3209 | PAS3209 |
| Partition | gpu | gpu |

All 9 training rules (3 stages x 3 variants) get identical `_TRAIN_RES`. All 3 evaluation rules (teacher, student_kd, student_nokd) get identical `_EVAL_RES`.

**Hardcoded values in Snakefile:**
- `CLI` path: `/users/PAS2022/rf15/.conda/envs/gnn-experiments/bin/python -m pipeline.cli` (line 25)
- `PY` path: `/users/PAS2022/rf15/.conda/envs/gnn-experiments/bin/python` (line 26)
- Account `PAS3209` and partition `gpu` embedded in resource dicts
- `jobs: 20` and `latency-wait: 120` in the profile config

**onstart hook** (`Snakefile:36-37`): Pre-initializes the MLflow SQLite database on the login node before any SLURM jobs launch, preventing race conditions when multiple jobs try to CREATE TABLE concurrently.

**Job status polling** (`profiles/slurm/status.sh`): Uses `sacct` to check job state. Maps SLURM states to Snakemake's expected `running`/`success`/`failed` vocabulary. Empty `sacct` output (lag) defaults to `running`.

### 1.2 Configuration Cascade

Parameters flow through 5 layers:

```
PRESETS dict  -->  PipelineConfig (frozen dataclass)  -->  CLI --flag overrides
     -->  frozen config.json  -->  stage function  -->  model constructor
```

**Step 1: Preset selection** (`config.py:18-62`). Six presets define model-specific overrides:

| Preset | Key overrides |
|--------|---------------|
| `(vgae, teacher)` | lr=0.002, hidden_dims=(1024,512,96), latent=96, heads=4, emb=64, dropout=0.15 |
| `(vgae, student)` | lr=0.002, hidden_dims=(80,40,16), latent=16, heads=1, emb=4, dropout=0.1, use_kd=True |
| `(gat, teacher)` | lr=0.003, hidden=64, layers=5, heads=8, dropout=0.2, emb=32 |
| `(gat, student)` | lr=0.001, hidden=24, layers=2, heads=4, dropout=0.1, emb=8, patience=50, use_kd=True |
| `(dqn, teacher)` | hidden=576, layers=3, gamma=0.99, buffer=100K, batch=128, target_update=100, episodes=500 |
| `(dqn, student)` | hidden=160, layers=2, gamma=0.99, buffer=50K, batch=64, target_update=50, episodes=500 |

**Step 2: PipelineConfig** (`config.py:65-178`). A frozen dataclass with ~65 fields covering identity, training, LR scheduling, memory optimization, batch sizing, checkpointing, reproducibility, GAT/VGAE/DQN architecture, KD, curriculum, fusion, and infrastructure.

**Step 3: CLI overrides** (`cli.py:54-69`). Every dataclass field is auto-registered as `--field-name`. Type coercion is automatic (bool, tuple, int, float, str).

**Step 4: Frozen config** (`cli.py:110-112`). The complete config is serialized to `config.json` in the stage output directory before training begins. This is the record of exactly what ran.

**Step 5: Stage dispatch** (`cli.py:121-122`). The stage function receives the frozen config and constructs models from it.

### 1.3 The Three Tiers Problem

Pipeline parameters exist at three visibility levels, creating confusion about what's tunable.

#### Tier 1: Centralized in PipelineConfig (~65 fields)

All architecture dimensions, learning rates, KD weights, fusion parameters, batch sizing, and scheduling. Fully visible, documented, JSON-serialized, and overridable from CLI.

#### Tier 2: Hardcoded in `pipeline/` code (~20 values)

These are invisible to the config system. Changing them requires editing source code.

| Value | Location | Description |
|-------|----------|-------------|
| `recon + 0.1*canid + 0.05*nbr + 0.01*kl` | `modules.py:66` | VGAE task loss weights |
| `recon + 0.1*canid` | `training.py:173` | Difficulty score formula |
| `95` | `modules.py:237` | Curriculum end percentile |
| `3.0 / -3.0` | `fusion.py:68` | Training reward (correct/incorrect) |
| `50` | `fusion.py:78` | Validation interval (episodes) |
| `alpha > 0.5` | `fusion.py:67`, `evaluation.py:257` | Decision threshold (3 locations) |
| `[0.05, 0.01, 0.001]` | `evaluation.py:327` | FPR targets for detection-at-FPR |
| `5000` | `fusion.py:81` | Max validation samples per interval |
| `500` | `training.py:150` | Difficulty scoring chunk size |
| `15` | `utils.py:527-585` | State vector dimensionality |
| `1/(1+err)` | `utils.py:558` | VGAE confidence formula |
| `1 - entropy/0.693` | `utils.py:572` | GAT confidence formula (0.693 = ln(2)) |
| `chunk_size=5000` | `preprocessing.py:549` | CSV streaming chunk size |
| `0.8 / 0.2` | `datamodules.py:146` | Train/val split ratio |

#### Tier 3: Shadow defaults in `src/` models

These defaults in model constructors are always overridden by PipelineConfig, but their existence creates confusion when reading model code in isolation.

| Parameter | `src/` default | PipelineConfig default | Preset override |
|-----------|---------------|----------------------|-----------------|
| VGAE dropout | 0.35 (`vgae.py:33`) | 0.15 (`config.py:131`) | 0.15/0.1 |
| VGAE latent_dim | 32 (`vgae.py:32`) | 96 (`config.py:128`) | 96/16 |
| VGAE embedding_dim | 8 (`vgae.py:33`) | 64 (`config.py:130`) | 64/4 |
| GAT num_fc_layers | 3 (`models.py:12`) | **NOT IN CONFIG** | **NOT TUNABLE** |
| GAT num_layers | 3 (`models.py:12`) | 5 (`config.py:121`) | 5/2 |
| GAT heads | 4 (`models.py:12`) | 8 (`config.py:122`) | 8/4 |
| DQN gamma | 0.9 (`dqn.py:40`) | 0.99 (`config.py:136`) | 0.99 |
| DQN epsilon | 0.2 (`dqn.py:40`) | 0.1 (`config.py:137`) | 0.1 |
| DQN hidden_dim | 128 (`dqn.py:43`) | 576 (`config.py:134`) | 576/160 |
| QNetwork Dropout | 0.2 (`dqn.py:25`) | **NOT IN CONFIG** | **NOT TUNABLE** |
| AdamW weight_decay | 1e-5 (`dqn.py:67`) | **NOT IN CONFIG** | **NOT TUNABLE** |

### 1.4 Dual Reward Functions

The DQN fusion has **two different reward functions** used in different contexts:

**Training reward** (`fusion.py:68`): Simple binary reward used during episode training.
```python
reward = 3.0 if pred == batch_labels[i].item() else -3.0
```

**Validation reward** (`dqn.py:146-212`): Elaborate 9-weight shaped reward used only during `validate_agent()`. Component weights:

| Component | Correct prediction | Wrong prediction |
|-----------|-------------------|------------------|
| Base reward | +3.0 | -3.0 |
| Model agreement bonus | +1.0 * agreement | -- |
| Confidence bonus (attack) | +0.5 * max(anomaly, gat_prob) | -- |
| Confidence bonus (normal) | +0.5 * (1-max(anomaly, gat_prob)) | -- |
| Combined confidence | +0.3 * combined_confidence | -- |
| Disagreement penalty | -- | -1.0 * (1-agreement) |
| FP overconfidence penalty | -- | -1.5 * fused_confidence |
| FN overconfidence penalty | -- | -1.5 * (1-fused_confidence) |
| Balance bonus (always) | +0.3 * (1 - \|alpha-0.5\| * 2) | +0.3 * (1 - \|alpha-0.5\| * 2) |

The VGAE error weights within the reward: `[0.4, 0.35, 0.25]` for `[node_recon, neighbor, canid]`.

The validation reward also fuses scores differently than training. During training, `alpha > 0.5` is a direct threshold. During validation, `fused_score = (1-alpha)*anomaly + alpha*gat_prob` is computed and thresholded at 0.5.

### 1.5 Preprocessing Constants (not in config)

Defined in `src/preprocessing/preprocessing.py:34-53`:

| Constant | Value | Location |
|----------|-------|----------|
| `DEFAULT_WINDOW_SIZE` | 100 | `preprocessing.py:35` |
| `DEFAULT_STRIDE` | 100 | `preprocessing.py:36` |
| `EXCLUDED_ATTACK_TYPES` | `['suppress', 'masquerade']` | `preprocessing.py:37` |
| `MAX_DATA_BYTES` | 8 | `preprocessing.py:38` |
| `NODE_FEATURE_COUNT` | 11 (CAN_ID + 8 bytes + count + position) | `preprocessing.py:42` |
| `EDGE_FEATURE_COUNT` | 11 | `preprocessing.py:43` |
| `MMAP_TENSOR_LIMIT` | 60,000 | `datamodules.py:38` |
| Train/val split | 0.8 | `datamodules.py:146` |

### 1.6 Recommendations

**Should centralize into PipelineConfig:**
- VGAE task loss weights (0.1, 0.05, 0.01) -- these materially affect training
- Decision threshold (0.5) -- currently hardcoded in 3 places
- Fusion validation interval (50 episodes)
- Training reward magnitude (3.0/-3.0)

**Should parameterize in Snakefile:**
- Python path (`CLI`, `PY`) -- use `$(which python)` or a config variable
- SLURM account and partition -- make them `config.get()` variables

**Fine as-is:**
- State vector layout (15-dim) -- structural, not a tunable
- Confidence formulas -- derived, not parameters
- FPR targets -- analysis convention, not training param
- Preprocessing constants (window, stride, features) -- rarely changed, would add complexity

**Needs investigation:**
- Dual reward functions -- training and validation rewards disagree; unclear if intentional
- QNetwork `Dropout(0.2)` -- not tunable, should at least match config's `gat_dropout`
- DQN `weight_decay=1e-5` -- differs from PipelineConfig's `weight_decay=1e-4`
- `num_fc_layers=3` -- not in PipelineConfig, interacts badly with JK cat mode (see Bug 3.7)

---

## Part 2: Experiment Outputs

### 2.1 Run Inventory

Expected: 6 datasets x 13 stages = 78 output directories.
Actual: 78 directories exist (all created). 66 `config.json` files found. 54 `best_model.pt` files found (training stages only; evaluation stages produce `metrics.json`).

All 54 training checkpoints are present:
- 6 datasets x 9 training stages = 54 `best_model.pt` files

### 2.2 Model Compression Results

Measured from `hcrl_sa` checkpoints (teacher vs student_kd):

| Component | Teacher Size | Student Size | Compression Ratio | Expected |
|-----------|-------------|-------------|-------------------|----------|
| VGAE (`best_model.pt`) | 5.1 MB | 352 KB | 14.7x | ~15x OK |
| GAT (`best_model.pt`) | **55 MB** | **406 KB** | **137x** | **~20x -- see Bug 3.7** |
| DQN (`best_model.pt`) | 5.3 MB | 260 KB | 20.8x | ~20x OK |

The GAT teacher is 137x larger than its student (expected ~20x). Root cause analysis in Bug 3.7 below.

### 2.3 Evaluation Completion Status

| Dataset | Teacher eval | Student eval (no KD) | Student eval (KD) |
|---------|-------------|---------------------|-------------------|
| **hcrl_sa** | metrics.json present | metrics.json present | CRASHED (bug 3.4 -- now fixed) |
| **hcrl_ch** | metrics.json present | metrics.json present | CRASHED (bug 3.4 -- now fixed) |
| **set_01** | CRASHED (bug 3.1 -- now fixed) | not attempted | not attempted |
| **set_02** | TIMED OUT (bug 3.2 -- now fixed) | not attempted | not attempted |
| **set_03** | CRASHED (bug 3.1 -- now fixed) | not attempted | not attempted |
| **set_04** | TIMED OUT (bugs 3.2, 3.5 -- now fixed) | not attempted | not attempted |

Total: 4 of 18 evaluation runs completed. All blocking bugs (3.1-3.5) now fixed -- re-run needed.

### 2.4 Metrics Summary

#### In-distribution validation (near-perfect)

**hcrl_sa** (1,873 validation graphs):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| GAT teacher | 0.9995 | 1.0000 | 0.9960 | 0.9980 | 1.0000 |
| GAT student (no KD) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| VGAE teacher | 0.8190 | 0.4127 | 0.8287 | 0.5510 | 0.9086 |
| VGAE student (no KD) | 0.6716 | 0.2764 | 0.8964 | 0.4225 | 0.8776 |
| Fusion teacher | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Fusion student (no KD) | 0.9995 | 1.0000 | 0.9960 | 0.9980 | 0.9970 |

**hcrl_ch** (29,088 validation graphs):

| Model | Accuracy | F1 | AUC |
|-------|----------|-----|-----|
| GAT teacher | 0.9997 | 0.9995 | 1.0000 |
| GAT student (no KD) | 0.9997 | 0.9995 | 1.0000 |
| VGAE teacher | 0.7568 | 0.6373 | 0.8342 |
| Fusion teacher | 0.9998 | 0.9997 | 0.9999 |

#### Out-of-distribution test results (hcrl_sa)

**Teacher:**

| Test Scenario | GAT Acc | GAT F1 | Fusion Acc | Fusion F1 |
|---------------|---------|--------|------------|-----------|
| test_01: known vehicle, known attack (n=3994) | 0.9987 | 0.9987 | 0.9990 | 0.9990 |
| test_02: unknown vehicle, known attack (n=1861) | 0.4052 | 0.5767 | 0.4052 | 0.5767 |
| test_03: known vehicle, unknown attack (n=3060) | 0.6212 | 0.0000 | 0.6425 | 0.1062 |
| test_04: unknown vehicle, unknown attack (n=797) | 0.3915 | 0.5627 | 0.3915 | 0.5627 |

**Student (no KD):**

| Test Scenario | GAT Acc | GAT F1 | GAT AUC |
|---------------|---------|--------|---------|
| test_01 | 0.9990 | 0.9990 | 0.9995 |
| test_02 | 0.4052 | 0.5767 | 0.9992 |
| test_03 | 0.6719 | 0.2359 | 0.9361 |
| test_04 | 0.3915 | 0.5627 | 0.9852 |

**Key observations:**
- In-distribution: all models achieve F1 > 0.99 (except VGAE, which is reconstruction-based and expected lower)
- test_01 (known vehicle, known attack): F1 ~0.999 -- good generalization
- test_02 (unknown vehicle, known attack): accuracy ~0.40, F1 ~0.58 -- model predicts most samples as attack (low specificity)
- test_03 (known vehicle, unknown attack): teacher GAT F1 = 0.0 (fails to detect novel attacks entirely); student slightly better at F1 = 0.24
- test_04 (unknown vehicle, unknown attack): same collapse as test_02
- Fusion does NOT improve over standalone GAT on any OOD scenario
- Student has substantially higher AUC scores on OOD (test_02: 0.9992 vs 0.9666; test_04: 0.9852 vs 0.8979), suggesting the ranking is preserved even when the threshold fails

**hcrl_ch test results** (same vehicle, different attack types):

| Test | GAT Teacher F1 | Fusion Teacher F1 |
|------|----------------|-------------------|
| test_01 (DoS) | 0.9998 | 0.9998 |
| test_02 (fuzzing) | 0.9995 | 1.0000 |
| test_03 (gear spoofing) | 0.9991 | 0.9991 |
| test_04 (rpm spoofing) | 0.9999 | 0.9997 |

hcrl_ch tests are all same-vehicle known-attack scenarios, so near-perfect results are expected.

### 2.5 Data Cache Status

| Dataset | Cache Size | Status |
|---------|-----------|--------|
| hcrl_sa | 47 MB | OK |
| hcrl_ch | 624 MB | OK |
| set_01 | 922 MB | OK |
| set_02 | 1.3 GB | OK |
| set_03 | 1.4 GB | OK |
| set_04 | 702 MB | Cache save failed on first attempt (NFS race), eventually cached |
| **Total** | **~5 GB** | |

Cache format: `processed_graphs.pt` (list of PyG Data objects) + `id_mapping.pkl` (dict).

Evaluation runs log `WARNING Invalid cache format` because the loaded object fails the `isinstance(graphs, list)` check -- mmap-loaded PyTorch objects may return a different container type. Data gets reprocessed from CSVs each eval run.

### 2.6 MLflow Status

- SQLite database: 9.1 MB at `/fs/scratch/PAS1266/kd_gat_mlflow/mlflow.db`
- Metadata-only tracking (model artifacts live in `experimentruns/`)
- `onstart` hook in Snakefile pre-initializes the DB to prevent SQLite race conditions
- All training runs logged; evaluation runs that crash still log partial metrics

---

## Part 3: Bugs & Issues

### Bug 3.1 -- CAN ID Embedding Out-of-Bounds (CRITICAL) -- FIXED 2026-02-05

**Severity:** Critical -- crashes evaluation for set_01 and set_03.

**Symptom:** `CUDA error: device-side assert triggered` in `GATConv.forward` during test data inference. The error manifests as `vectorized_gather_kernel: Assertion 'ind >= 0 && ind < ind_dim_size'`.

**Root cause:** `apply_dynamic_id_mapping()` (`preprocessing.py:81-117`) mutated the ID mapping dictionary when test data contained CAN IDs not seen during training. It inserted new entries and repositioned the OOV index, pushing mapped values beyond the fixed-size `nn.Embedding` table.

```
Training: build_lightweight_id_mapping() --> num_ids=501 --> nn.Embedding(501, dim)
Test:     apply_dynamic_id_mapping()      --> mapping grows to 502+
          --> embedding lookup(502) crashes: index >= table size
```

**Affected datasets:** set_01, set_03 (different vehicle networks with CAN IDs not present in training data).

**Fix applied:** Removed the first pass that expanded the mapping. Unseen IDs now fall through to the `.map().fillna(oov_index)` in the second pass, correctly mapping to OOV without growing the dictionary. Test added: `test_apply_dynamic_id_mapping_no_expansion`.

### Bug 3.2 -- Eval Time Budget Insufficient -- FIXED 2026-02-05

**Severity:** High -- prevents evaluation of set_02 and set_04.

**Symptom:** Evaluation runs for large datasets hit the 60-minute SLURM wall time limit and are killed before producing `metrics.json`.

**Root cause:** `_EVAL_RES` allocated 60 minutes (`Snakefile:72`). For set_01-04:
- Data processing from scratch (cache format mismatch): 10-20 min
- Loading 4 test scenarios from CSVs: 20-40 min
- Running inference across 3 models x 4 scenarios: remaining time insufficient

set_02 is the largest: 203,496 training graphs + 4 test scenarios totaling 385,981 graphs.

**Fix applied:** Bumped `_EVAL_RES` time_min from 60 to 120.

### Bug 3.3 -- Cache Format Invalidation -- FIXED 2026-02-05

**Severity:** Medium -- causes redundant computation, no data loss.

**Symptom:** Every evaluation run logs: `WARNING Invalid cache format. Expected list of graphs and dict mapping.`

**Root cause:** The cache validation check (`datamodules.py:204`) tested `isinstance(graphs, list)`. When loaded with `mmap=True` (PyTorch 2.1+), the deserialized object may not pass this check. The data was then reprocessed from CSVs, adding 10-20 minutes to eval runs.

**Fix applied:** Validation now accepts `GraphDataset` objects (extracts `.data_list`) and any iterable (converts to list). Only `id_mapping` type is strictly checked.

### Bug 3.4 -- eval_student_kd Fails Validation -- FIXED (prior session)

**Severity:** High -- blocks all student KD evaluation runs.

**Symptom:** All `student_evaluation_kd` runs crash immediately with:
```
ValueError: Config validation failed: use_kd=True but teacher_path is empty
```

**Root cause:** The `validate()` function required `teacher_path` whenever `use_kd=True`, even though evaluation doesn't use the teacher for inference.

**Fix applied:** `validate.py:40` now reads `if cfg.use_kd and not cfg.teacher_path and stage != "evaluation":` — evaluation stage is exempt from the teacher_path requirement.

### Bug 3.5 -- NFS Cache Race (set_04) -- FIXED 2026-02-05

**Severity:** Low -- transient, resolved on retry.

**Symptom:** `Failed to save cache: [Errno 2] No such file or directory: 'data/cache/set_04/processed_graphs.tmp'`

**Root cause:** The atomic rename (`temp_cache.rename(cache_file)` in `datamodules.py:299`) failed because NFS didn't guarantee the `.tmp` file was visible after creation.

**Fix applied:** Added `os.fsync()` after writing temp files to force NFS write-through, plus retry-with-backoff (3 attempts, 1s delay) on the rename operation.

### Bug 3.6 -- Generalization Collapse (Research Issue)

**Severity:** Research -- not a code bug, but critical for the project's goals.

**Symptom:** Models achieve 99%+ F1 in-distribution but collapse on out-of-distribution data:
- Unknown vehicles (test_02, test_04): accuracy ~0.40, models predict nearly everything as attack
- Unknown attacks (test_03): teacher GAT F1 = 0.0, student GAT F1 = 0.24 -- misses most novel attacks
- Fusion provides no improvement over standalone GAT on any OOD scenario

**Interesting finding:** Student models (no KD) achieve much higher AUC on OOD scenarios than teachers (test_02: 0.9992 vs 0.9666). This suggests the ranking/scoring is informative but the decision threshold (hardcoded at 0.5) is miscalibrated for OOD data.

**Possible research directions:**
- Threshold calibration per deployment domain
- Domain adaptation or adversarial training for vehicle-invariant features
- Data augmentation with synthetic CAN ID remapping
- Using AUC-optimized thresholds instead of fixed 0.5

### Bug 3.7 -- GAT Teacher 137x Larger Than Student (Expected ~20x)

**Severity:** Medium -- not a crash, but indicates architectural inefficiency.

**Symptom:** GAT teacher checkpoint is 55 MB vs student 406 KB = 137x compression ratio. VGAE (14.7x) and DQN (20.8x) have expected ratios.

**Root cause:** JumpingKnowledge "cat" mode concatenates all layer outputs, making `fc_input_dim = hidden * heads * num_layers`. The FC layers maintain this full width.

From `models.py:48-54`:
```python
if self.jk.mode == "cat":
    fc_input_dim = hidden_channels * heads * num_layers  # concatenation

for _ in range(num_fc_layers - 1):
    self.fc_layers.append(nn.Linear(fc_input_dim, fc_input_dim))  # full-width intermediate
```

**Teacher:** `64 * 8 * 5 = 2560` --> FC layers are `Linear(2560, 2560)` x 2 = **13.1M params** in FC alone.
**Student:** `24 * 4 * 2 = 192` --> FC layers are `Linear(192, 192)` x 2 = **74K params** in FC.

FC parameter ratio: 13.1M / 74K = **177x** (dominates total model size).

Compounding factors:
- `num_fc_layers=3` is hardcoded in `models.py:12`, not exposed in PipelineConfig (Tier 3 shadow default)
- The intermediate FC layers have no bottleneck -- they maintain the full JK concatenation width
- The width grows **quadratically** with attention heads and linearly with layer count

**Fix options (document only):**
1. Add `gat_fc_hidden` config field as bottleneck: `Linear(fc_input_dim, gat_fc_hidden)` then `Linear(gat_fc_hidden, out_channels)`
2. Reduce `num_fc_layers` from 3 to 2 (less impact, still quadratic)
3. Switch JK mode from "cat" to "max" (reduces `fc_input_dim` from `h*heads*layers` to `h*heads`)
4. Add `num_fc_layers` to PipelineConfig so it's tunable per preset

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Training stages completed | 54/54 | All checkpoints present |
| Evaluation stages completed | 4/18 | Re-run needed (blockers fixed) |
| Bugs fixed (2026-02-05) | 5 | 3.1, 3.2, 3.3, 3.4, 3.5 all resolved |
| Remaining bugs | 1 | 3.7 (GAT size — needs retrain) |
| Research issues | 1 | 3.6 (OOD collapse) |
| Tier 2 hardcoded values | ~20 | See section 1.3 |
| Tier 3 shadow defaults | ~10 | See section 1.3 |
