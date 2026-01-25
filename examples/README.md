# Configuration Examples

**Ready-to-run examples for common training scenarios**

All examples are self-contained Python scripts that demonstrate how to configure and train models.

---

## Quick Start

```bash
# Make examples executable
chmod +x examples/*.py

# Run an example
python examples/simple_gat_training.py
```

---

## Available Examples

### 1. Simple GAT Training
**File**: [simple_gat_training.py](simple_gat_training.py)

Train a teacher GAT model for supervised classification.

```bash
python examples/simple_gat_training.py
```

**What it does**:
- Creates GAT config with `create_gat_normal_config()`
- Trains teacher model
- Saves to canonical experiment path

---

### 2. VGAE Autoencoder
**File**: [vgae_autoencoder_training.py](vgae_autoencoder_training.py)

Train unsupervised VGAE for anomaly detection.

```bash
python examples/vgae_autoencoder_training.py
```

**What it does**:
- Creates VGAE config with `create_autoencoder_config()`
- Trains on normal samples only
- Produces reconstruction-based anomaly scores

**Use for**:
- Anomaly detection
- Curriculum learning guidance
- Fusion component

---

### 3. Knowledge Distillation
**File**: [knowledge_distillation.py](knowledge_distillation.py)

Compress teacher model to lightweight student.

```bash
# Requires trained teacher first
python examples/simple_gat_training.py  # Train teacher
python examples/knowledge_distillation.py  # Distill to student
```

**What it does**:
- Loads trained teacher model
- Creates student config
- Trains student with soft labels
- Compresses ~1.1M params → ~55K params

**Key parameters**:
- `temperature`: Softens probability distributions (default: 4.0)
- `alpha`: Balance between KD loss and task loss (default: 0.7)

---

### 4. Fusion Training
**File**: [fusion_training.py](fusion_training.py)

Train DQN agent to fuse VGAE + GAT predictions.

```bash
# Requires both VGAE and GAT models
python examples/vgae_autoencoder_training.py  # Train VGAE
python examples/simple_gat_training.py  # Train GAT
python examples/fusion_training.py  # Train fusion
```

**What it does**:
- Validates required artifacts exist
- Creates fusion config
- Trains DQN to learn optimal fusion weights
- Combines anomaly scores + classification

---

### 5. Curriculum Learning
**File**: [curriculum_learning.py](curriculum_learning.py)

Train with VGAE-guided hard sample mining.

```bash
# Requires VGAE model for guidance
python examples/vgae_autoencoder_training.py  # Train VGAE
python examples/curriculum_learning.py  # Train with curriculum
```

**What it does**:
- Uses VGAE reconstruction errors to identify hard samples
- Progressive difficulty schedule (1:1 → 10:1 class imbalance)
- Hard negative mining for better generalization

**Key parameters**:
- `start_ratio`: Initial normal:attack ratio (default: 1.0)
- `end_ratio`: Final normal:attack ratio (default: 10.0)
- `difficulty_percentile`: Top % difficult samples (default: 75.0)

---

## Customization

### Override Config Values

All examples allow overriding defaults:

```python
config = create_gat_normal_config("hcrl_sa")

# Override training parameters
config.training.max_epochs = 200
config.training.batch_size = 128
config.training.learning_rate = 0.001

# Override model parameters
config.model.hidden_channels = 128
config.model.num_layers = 4
config.model.dropout = 0.2
```

### Using Config Store Directly

```python
from src.config.hydra_zen_configs import CANGraphConfigStore

store = CANGraphConfigStore()

# Create any config
config = store.create_config(
    model_type="gat",
    dataset_name="hcrl_sa",
    training_mode="normal",
    # Pass overrides directly
    max_epochs=200,
    batch_size=128
)
```

---

## Common Patterns

### Check Required Artifacts

Before training modes that need pre-trained models:

```python
artifacts = config.required_artifacts()

for name, path in artifacts.items():
    if not path.exists():
        print(f"Missing: {name} at {path}")
```

### Validate Config

Always validate before training:

```python
from src.config.hydra_zen_configs import validate_config

try:
    validate_config(config)
    print("✓ Config valid")
except (ValueError, FileNotFoundError) as e:
    print(f"✗ Config error: {e}")
```

### Get Canonical Paths

```python
# Where model will be saved
exp_dir = config.canonical_experiment_dir()
print(f"Output: {exp_dir}")

# Required artifacts for complex modes
artifacts = config.required_artifacts()
print(f"Required: {list(artifacts.keys())}")
```

---

## Training Pipeline Examples

### Sequential Pipeline

```bash
#!/bin/bash
# train_pipeline.sh - Complete training pipeline

DATASET="hcrl_sa"

echo "=== Training Pipeline for $DATASET ==="

# 1. Train VGAE (unsupervised)
echo "1. Training VGAE autoencoder..."
python examples/vgae_autoencoder_training.py

# 2. Train GAT (supervised)
echo "2. Training GAT classifier..."
python examples/simple_gat_training.py

# 3. Knowledge distillation
echo "3. Distilling to student..."
python examples/knowledge_distillation.py

# 4. Fusion
echo "4. Training fusion agent..."
python examples/fusion_training.py

echo "✓ Pipeline complete!"
```

### Parallel Training

```bash
#!/bin/bash
# train_parallel.sh - Train teachers in parallel

# Launch both teachers simultaneously
python examples/vgae_autoencoder_training.py &
VGAE_PID=$!

python examples/simple_gat_training.py &
GAT_PID=$!

# Wait for both
wait $VGAE_PID
wait $GAT_PID

echo "Both teachers trained!"

# Now train fusion
python examples/fusion_training.py
```

---

## Troubleshooting

### Dataset Not Found

**Error**: `FileNotFoundError: Dataset path does not exist`

**Solution**:
```python
config.dataset.data_path = "/path/to/your/dataset"
```

Or set environment variable:
```bash
export CAN_DATA_PATH="/path/to/datasets"
```

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Reduce batch size
config.training.batch_size = 32  # From 64

# Or enable auto batch size optimization
config.training.optimize_batch_size = True
```

### Missing Teacher Model

**Error**: `FileNotFoundError: Teacher model not found`

**Solution**: Train teacher first
```bash
python examples/simple_gat_training.py
```

---

## Next Steps

- **Modify examples** for your datasets
- **Create custom training scripts** using these as templates
- **Submit to cluster** using [docs/WORKFLOW_GUIDE.md](../docs/WORKFLOW_GUIDE.md)
- **Add logging** with MLflow (see [docs/MLflow_SETUP.md](../docs/MLflow_SETUP.md))

See [docs/CODE_TEMPLATES.md](../docs/CODE_TEMPLATES.md) for more code snippets.
