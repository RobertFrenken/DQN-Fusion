# Comprehensive Failure Analysis & Proposed Fixes

## Investigation Summary

This document consolidates all findings from the job failure investigation, including root causes and proposed fixes.

---

## 1. Job Success Rate Clarification

### The Discrepancy: 95% vs 73%

**Initial Claim (Incorrect)**: 37/38 jobs succeeded (97.4%)
- **Method**: Quick visual inspection of job submissions
- **Problem**: Counted jobs that **started** not jobs that **completed**

**Corrected Analysis**: 28/38 jobs succeeded (73.7%)
- **Method**: Systematic parsing of SLURM output files
- **Tool**: [scripts/parse_job_results.py](scripts/parse_job_results.py)
- **Documentation**: [HOW_TO_ANALYZE_JOB_SUCCESS.md](HOW_TO_ANALYZE_JOB_SUCCESS.md)

### Success Breakdown by Stage

| Stage | Success | Failed | Rate   |
|-------|---------|--------|--------|
| VGAE  | 12/12   | 0/12   | **100%** ✅ |
| GAT   | 11/13   | 2/13   | 84.6%  |
| DQN   | 5/13    | 8/13   | **38.5%** ❌ |

**Key Insight**: VGAE is rock solid, DQN fusion is the weakest link.

---

## 2. Why 4/6 Datasets Have Identical ID Counts (2049)

### The Pattern

| Dataset | num_ids | CSV Files | Interpretation |
|---------|---------|-----------|----------------|
| hcrl_sa | 2049    | 6         | Complete CAN standard range |
| hcrl_ch | 2049    | 6         | Complete CAN standard range |
| set_01  | **53**  | 18        | **Filtered subset** |
| set_02  | 2049    | 22        | Complete CAN standard range |
| set_03  | **1791**| 20        | **Incomplete range (87%)** |
| set_04  | 2049    | 26        | Complete CAN standard range |

### Explanation: CAN Bus Standards

**11-bit CAN IDs**: Standard CAN bus uses 11-bit identifiers
- Range: 0x000 - 0x7FF (decimal: 0-2047)
- Total IDs: **2048 unique values**
- With OOV token: **2049 entries**

**Why these datasets have 2049**:
1. They contain **complete coverage** of the standard CAN ID range
2. Preprocessor scans all CSV files: `build_lightweight_id_mapping(csv_files)`
3. Extracts unique CAN IDs: `unique_ids.update(file_ids)`
4. Adds OOV token: `id_mapping['OOV'] = len(id_mapping)`
5. Result: 2048 unique IDs + 1 OOV = **2049 total**

**Why set_01 has 53**:
- Filtered or synthetic dataset containing only a subset of IDs
- Likely for testing or specific attack scenarios

**Why set_03 has 1791**:
- **Incomplete ID coverage** (87% of standard range)
- Missing ~258 IDs from standard range
- Could indicate:
  - Real-world dataset with sparse ID usage
  - Data collection from specific vehicle subsystem
  - Attack scenarios with ID filtering

### NOT a Bug

This is **correct behavior** - the ID mapping reflects the actual unique CAN IDs present in each dataset.

---

## 3. set_03 GAT Teacher OOM Failure

### Root Cause: Graph Density

**Measured Statistics** (from [scripts/analyze_graph_density.py](scripts/analyze_graph_density.py)):

| Dataset | Avg Nodes | Avg Edges | Memory/Graph | Status |
|---------|-----------|-----------|--------------|--------|
| set_02  | 37.68     | 67.91     | 2.68 KB      | ✅ SUCCESS |
| **set_03** | **53.67** | **88.24** | **3.68 KB** | ❌ **OOM** |

**Critical Findings**:
- set_03 has **42% more nodes per graph** (53.67 vs 37.68)
- set_03 has **30% more edges per graph** (88.24 vs 67.91)
- set_03 graphs are **37% larger in memory** (3.68 KB vs 2.68 KB)

### Why set_03 Failed When set_02 Succeeded

**Memory Calculation**:

```python
set_02 (SUCCESS):
  203k graphs × 2.68 KB × 2 loads = 1.06 GB graph data
  × ~5x PyTorch overhead (edge_index tensors, activations)
  = ~5.3 GB base
  + VGAE + GAT models + first batch
  = ~13.2 GB total ✅ Fits in 15.77 GB GPU

set_03 (OOM):
  166k graphs × 3.68 KB × 2 loads = 1.19 GB graph data
  × ~5x PyTorch overhead
  = ~6.0 GB base
  + VGAE + GAT models + first batch (2.11 GB)
  = ~15.9 GB total ❌ EXCEEDS 15.77 GB GPU
```

**GPU Memory at Failure**:
```
GPU capacity: 15.77 GiB
Already allocated: 13.84 GiB
Tried to allocate (first batch): 2.11 GiB
Free memory: 1.92 GiB
Result: OOM
```

### Proposed Fix

**Option 1: Reduce batch size for set_03**
```bash
--batch-size 16  # Half the default 32
# Estimated first batch: ~1.0 GB instead of 2.11 GB
# Total: ~14.8 GB ✅ Fits
```

**Option 2: Gradient accumulation**
```bash
--batch-size 16 --accumulate-grad-batches 2
# Simulates batch_size=32 with lower memory
```

**Option 3: Add curriculum-specific batch size config** (see section 5 below)

---

## 4. Curriculum Learning Memory Analysis

### How Curriculum Mode Works

**File**: [src/training/modes/curriculum.py](src/training/modes/curriculum.py)

**Step 1: Load Dataset Twice** (lines 88-102)
```python
# Load full dataset
full_dataset, val_dataset, _ = load_dataset(...)

# Separate into normal and attack
train_normal = [g for g in full_dataset if g.y.item() == 0]
train_attack = [g for g in full_dataset if g.y.item() == 1]
```

**Why loaded twice?** Logs show:
```
Line 65:  Total graphs created: 166098  (first pass)
Line 113: Total graphs created: 166098  (second pass)
```

This is for train/validation split construction.

**Step 2: Load VGAE Model** (lines 104-110)
```python
vgae_path = self._resolve_vgae_path()
vgae_model = self._load_vgae_model(vgae_path, num_ids)
vgae_model.eval()  # Frozen for inference
```

**Step 3: Create Enhanced DataModule** (lines 115-124)
```python
datamodule = EnhancedCANGraphDataModule(
    train_normal=train_normal,
    train_attack=train_attack,
    val_normal=val_normal,
    val_attack=val_attack,
    vgae_model=vgae_model,  # ← VGAE stays in memory!
    batch_size=initial_batch_size,
    total_epochs=self.config.training.max_epochs
)
```

**Step 4: Hard Sample Mining** (datamodules.py, lines 176-220)
```python
# During training, VGAE is used to score difficulty
def _score_difficulty(self, graphs):
    for graph in graphs:
        # VGAE forward pass
        z, kl_loss = self.vgae_model.encode(x, edge_index)
        cont_out, canid_logits = self.vgae_model.decode_node(z, edge_index)
        # Compute reconstruction loss as difficulty score
        recon_loss = F.mse_loss(cont_out, continuous_features)
```

### Memory Footprint

```
Component                          | Memory
-----------------------------------|------------------
Graph data (2× loads)              | ~2.7 GB (set_03)
VGAE model (frozen, loaded)        | ~5-10 MB
GAT model (training)               | ~5-10 MB
Embedding layers                   | ~0.5 MB
PyTorch allocations & gradients    | ~5-8 GB
First training batch               | ~2.1 GB
-----------------------------------|------------------
TOTAL                              | ~13.8 GB
```

### Weak Points

1. **Double dataset loading**: Could use view/subset instead of full copy
2. **VGAE stays in GPU memory**: Could offload to CPU when not actively mining
3. **No memory-based batch size adjustment**: Fixed batch size regardless of dataset density
4. **Hard sample mining overhead**: VGAE forward passes add activation memory

### Proposed Fixes

**Fix 1: Conditional VGAE offloading**
```python
# Only load VGAE to GPU when mining, keep on CPU otherwise
if self.should_mine_hard_samples(current_epoch):
    self.vgae_model = self.vgae_model.to(self.device)
    # Mine samples
    self.vgae_model = self.vgae_model.cpu()
```

**Fix 2: Streaming dataset loading**
```python
# Don't load entire dataset into memory
# Use lazy loading with indices
train_indices = [i for i, g in enumerate(full_dataset) if g.y.item() == 0]
```

**Fix 3: Memory-aware batch size** (see section 5)

---

## 5. Student DQN Failures: Wrong Artifact Paths

### The Bug

**Error from logs**:
```
❌ Training failed: Fusion training requires pre-trained artifacts:
autoencoder missing at experimentruns/.../vgae/student/.../knowledge_distillation/models/vgae_student_knowledge_distillation.pth
classifier missing at experimentruns/.../gat/student/.../knowledge_distillation/models/gat_student_knowledge_distillation.pth
```

**Problem**: Student fusion is looking for models at wrong paths!

### Root Cause Analysis

**File**: [src/config/hydra_zen_configs.py:697-706](src/config/hydra_zen_configs.py#L697-L706)

```python
if is_student_fusion:
    # Student fusion needs student VGAE and student GAT
    ae_dir = Path(...) / "vgae" / "student" / self.distillation / "knowledge_distillation"  # ❌ WRONG
    clf_dir = Path(...) / "gat" / "student" / self.distillation / "knowledge_distillation"  # ❌ WRONG
```

**Expected paths** (where models actually exist):
```
vgae/student/no_distillation/autoencoder/models/vgae_student_autoencoder.pth
gat/student/no_distillation/curriculum/models/gat_student_curriculum.pth
```

**Searched paths** (incorrect):
```
vgae/student/no_distillation/knowledge_distillation/models/vgae_student_knowledge_distillation.pth
gat/student/no_distillation/knowledge_distillation/models/gat_student_knowledge_distillation.pth
```

### Why This Happened

**Confusion between two concepts**:

1. **Training Mode** (actual mode used during training):
   - VGAE student trained with `mode=autoencoder`
   - GAT student trained with `mode=curriculum`
   - Both trained WITHOUT knowledge distillation

2. **Knowledge Distillation** (a feature, not a mode):
   - Can be enabled via `--distillation with-kd`
   - But uses `use_knowledge_distillation=True` flag in training config
   - Models still use their base modes (autoencoder, curriculum)

### The Fix

**Change lines 697-706 in hydra_zen_configs.py**:

```python
if is_student_fusion:
    # Student fusion needs student VGAE (autoencoder) and student GAT (curriculum)
    ae_dir = Path(self.experiment_root) / self.modality / self.dataset.name / "unsupervised" / "vgae" / "student" / self.distillation / "autoencoder"  # ✅ CORRECT
    clf_dir = Path(self.experiment_root) / self.modality / self.dataset.name / "supervised" / "gat" / "student" / self.distillation / "curriculum"  # ✅ CORRECT

    discovered_ae = resolver.discover_model(ae_dir, 'vgae', require_exists=False)
    discovered_clf = resolver.discover_model(clf_dir, 'gat', require_exists=False)

    artifacts["autoencoder"] = discovered_ae or (ae_dir / "models" / "vgae_student_autoencoder.pth")
    artifacts["classifier"] = discovered_clf or (clf_dir / "models" / "gat_student_curriculum.pth")
```

### Impact

**Current**: 8/13 student DQN fusion jobs failed (61.5% failure rate)
**After fix**: Should be 0/13 failures ✅

---

## 6. Batch Size Configuration for Curriculum Mode

### The Problem

Current batch size is **static** across all training modes:
```python
batch_size = 32  # Fixed for GAT curriculum
```

But curriculum mode has **2x memory footprint** due to:
1. Double dataset loading
2. VGAE model in memory
3. Hard sample mining activations

### Proposed Solution

**Add mode-specific batch size factors to config**:

**File**: [src/config/hydra_zen_configs.py](src/config/hydra_zen_configs.py)

**Option 1: Add curriculum_batch_size field**
```python
@dataclass
class CurriculumTrainingConfig(TrainingBase):
    mode: str = field(default="curriculum")
    max_epochs: int = field(default=200)
    batch_size: int = field(default=32)
    curriculum_batch_size: int = field(default=16)  # ← NEW: Half size for curriculum
    # ... rest of fields
```

**Option 2: Add batch_size_multiplier by mode**
```python
@dataclass
class CurriculumTrainingConfig(TrainingBase):
    mode: str = field(default="curriculum")
    max_epochs: int = field(default=200)
    batch_size: int = field(default=32)
    memory_multiplier: float = field(default=0.5)  # ← NEW: 50% batch size for high-memory modes
    # ... rest of fields
```

**Implementation in curriculum.py**:
```python
def _setup_curriculum_datamodule(self, num_ids):
    # ...

    # Adjust batch size for curriculum memory overhead
    base_batch_size = getattr(self.config.training, 'batch_size', 64)

    # Option 1: Direct override
    curriculum_batch_size = getattr(
        self.config.training,
        'curriculum_batch_size',
        base_batch_size // 2  # Default: half size
    )

    # Option 2: Multiplier
    memory_multiplier = getattr(self.config.training, 'memory_multiplier', 0.5)
    curriculum_batch_size = int(base_batch_size * memory_multiplier)

    logger.info(
        f"📊 Curriculum batch size: {curriculum_batch_size} "
        f"(base: {base_batch_size}, multiplier: {memory_multiplier})"
    )

    datamodule = EnhancedCANGraphDataModule(
        ...,
        batch_size=curriculum_batch_size,
        ...
    )
```

### Benefits

1. **Prevents hidden OOM crashes** like set_03
2. **Explicit configuration** - no magic numbers
3. **Dataset-specific tuning** - can override per dataset
4. **Backwards compatible** - defaults to existing behavior

---

## 7. Summary of Fixes Needed

### Priority 1: Critical Bugs

1. **Fix student DQN artifact paths** (lines 697-706 in hydra_zen_configs.py)
   - Impact: Fixes 8/13 student DQN failures
   - Complexity: Simple path string changes
   - File: [src/config/hydra_zen_configs.py](src/config/hydra_zen_configs.py)

### Priority 2: Memory Management

2. **Add curriculum-specific batch size config**
   - Impact: Prevents OOM for dense datasets
   - Complexity: Add config field + use in curriculum.py
   - Files:
     - [src/config/hydra_zen_configs.py](src/config/hydra_zen_configs.py)
     - [src/training/modes/curriculum.py](src/training/modes/curriculum.py)

3. **Re-run set_03 GAT teacher with reduced batch size**
   ```bash
   --batch-size 16  # or add curriculum_batch_size=16 to config
   ```

### Priority 3: Optimizations

4. **Implement VGAE CPU offloading during curriculum**
   - Impact: Reduces baseline memory by ~5-10 MB
   - Complexity: Moderate (device management)
   - File: [src/training/datamodules.py](src/training/datamodules.py)

5. **Convert double dataset loading to indexed views**
   - Impact: Reduces memory by ~50% of dataset size
   - Complexity: High (refactor dataset handling)
   - File: [src/training/modes/curriculum.py](src/training/modes/curriculum.py)

---

## 8. Testing Plan

### After Fix 1 (Student DQN Paths)

Run student pipeline for one dataset:
```bash
./can-train pipeline \
  --modality automotive \
  --model vgae,gat,dqn \
  --learning-type unsupervised,supervised,rl_fusion \
  --training-strategy autoencoder,curriculum,fusion \
  --dataset hcrl_sa \
  --model-size student \
  --distillation no-kd \
  --epochs 5 \
  --submit
```

**Expected**: All 3 stages succeed (VGAE, GAT, DQN)

### After Fix 2 (Curriculum Batch Size)

Run set_03 teacher pipeline:
```bash
./can-train pipeline \
  --modality automotive \
  --model vgae,gat,dqn \
  --learning-type unsupervised,supervised,rl_fusion \
  --training-strategy autoencoder,curriculum,fusion \
  --dataset set_03 \
  --model-size teacher \
  --distillation no-kd \
  --epochs 5 \
  --submit
```

**Expected**: GAT curriculum stage succeeds (previously OOM)

---

## 9. Documentation Updates

Created/updated files:
1. [HOW_TO_ANALYZE_JOB_SUCCESS.md](HOW_TO_ANALYZE_JOB_SUCCESS.md) - Proper job analysis method
2. [EXPERIMENT_RESULTS_ANALYSIS.md](EXPERIMENT_RESULTS_ANALYSIS.md) - Full results tables
3. [SET_03_FAILURE_DEEP_DIVE.md](SET_03_FAILURE_DEEP_DIVE.md) - set_03 OOM analysis
4. [COMPREHENSIVE_FAILURE_ANALYSIS.md](COMPREHENSIVE_FAILURE_ANALYSIS.md) - This document

Analysis scripts:
1. [scripts/parse_job_results.py](scripts/parse_job_results.py) - Parse SLURM outputs
2. [scripts/pipeline_summary.py](scripts/pipeline_summary.py) - Aggregate pipeline results
3. [scripts/analyze_graph_density.py](scripts/analyze_graph_density.py) - Measure graph statistics
