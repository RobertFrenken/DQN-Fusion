# Multi-Stage Knowledge-Distilled VGAE and GAT for Robust Controller-Area-Network Intrusion Detection

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

**Production-ready CAN intrusion detection with knowledge distillation for edge deployment**

---

## 🚀 Quick Start

**New to this project?** → See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)

```bash
# 1. Install environment
conda env create -f environment.yml
conda activate gnn-experiments

# 2. Train your first model (GAT classifier)
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal

# 3. Train VGAE autoencoder
python train_with_hydra_zen.py --model vgae --dataset hcrl_sa --training autoencoder

# 4. Train with knowledge distillation
python train_with_hydra_zen.py --model gat_student --dataset hcrl_sa --training knowledge_distillation
```

**Complete documentation**: [docs/](docs/)

**System Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Detailed system architecture, training pipeline, and implementation patterns

---

## Table of Contents

- [Description](#description)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Datasets](#datasets)
- [Model Components](#model-components)
- [Configuration](#configuration)
- [License](#license)

---

## Description

This framework implements a **multi-stage anomaly detection system** for Controller Area Network (CAN) intrusion detection using **Variational Graph Autoencoders (VGAE)** and **Graph Attention Networks (GAT)**. The system combines unsupervised anomaly detection with supervised classification in a two-stage pipeline, enhanced by knowledge distillation for efficient deployment.

The framework addresses CAN bus cybersecurity by:
- **Stage 1**: Unsupervised anomaly detection using VGAE with multiple reconstruction components
- **Stage 2**: Supervised classification of detected anomalies using GAT
- **Knowledge Distillation**: Training lightweight student models for edge device deployment

---

## Architecture

📐 **Full Documentation**: See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for comprehensive architecture diagrams and implementation details.

### Three-Stage Pipeline

**Stage 1: Feature Learning (VGAE)**
- Unsupervised learning of latent CAN message representations
- Graph autoencoder with variational inference
- ~1.3M parameters (teacher) / ~300K (student)

**Stage 2: Classification (GAT)**
- Supervised binary classification (normal vs attack)
- Curriculum learning: easy → hard samples
- Multi-head graph attention
- ~1.1M parameters (teacher) / ~250K (student)

**Stage 3: Fusion (DQN)**
- Reinforcement learning agent
- Fuses VGAE + GAT predictions
- Learns optimal combination strategy
- ~50K parameters

### Knowledge Distillation
- **Teacher Models**: Full-size VGAE + GAT pipeline
- **Student Models**: Compressed versions for resource-constrained environments
- **Distillation Process**: Soft label transfer with temperature scaling (α=0.7, T=4.0)

**Visual Diagrams**:
- [Training Pipeline Flow](docs/ARCHITECTURE.md#training-pipeline)
- [Parameter Flow (CLI → SLURM → Training)](docs/diagrams/parameter_flow.md)
- [Directory Structure](docs/diagrams/directory_structure.md)

---

## Features

- **Multi-Component Reconstruction**: Node features, CAN IDs, and graph neighborhoods
- **Composite Error Analysis**: Weighted combination of reconstruction errors
- **Two-Stage Detection**: Anomaly detection followed by classification
- **Knowledge Distillation**: Teacher-student model compression
- **Graph-Based Learning**: CAN messages represented as temporal graphs
- **Multiple Error Metrics**: Node, neighborhood, and structural reconstruction errors
- **Publication-Quality Visualizations**: Histogram analysis and error distribution plots
- **Configurable Thresholds**: Adaptive threshold setting based on training data

---

## Model Components

### Core Models
- **[`GraphAutoencoderNeighborhood`](models/models.py)**: VGAE with neighborhood prediction
- **[`GATWithJK`](models/models.py)**: Graph Attention Network with Jumping Knowledge
- **[`GATPipeline`](osc_training_AD.py)**: End-to-end two-stage pipeline

### Key Methods
- **Reconstruction Error Computation**: [`_compute_reconstruction_errors`](training-AD.py)
- **Threshold Setting**: Percentile-based adaptive thresholding
- **Graph Creation**: [`graph_creation`](preprocessing.py) - CAN frames to temporal graphs
- **Visualization**: [`plotting_utils.py`](plotting_utils.py) - Error analysis and distribution plots

---

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- Other dependencies listed in `requirements.txt`

## Installation

### Quick Install (Conda - Recommended)

```bash
git clone <your-repo-url>
cd KD-GAT

# Create environment
conda env create -f environment.yml
conda activate gnn-experiments

# Verify installation
python -c "from src.config.hydra_zen_configs import CANGraphConfigStore; print('✓ Setup complete')"
```

### Alternative: Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Development workflows (Just + uv/conda) ✅

We provide a `Justfile` with common development tasks to make experiments repeatable locally.

- Install `just` (platform dependent): https://github.com/casey/just

Common tasks:
```bash
# Quick environment check
just check-env

# Check dataset availability (uses hydra config store if available)
just check-data

# Test dataset loader and force-rebuild caches on a specific folder
just check-data-load

# Install dependencies using uv (if you prefer uv):
just install-uv

# Create or update conda env (for cluster/HPC runs):
just install-conda

# Run a small synthetic smoke experiment (safe and fast):
just smoke-synthetic

# Run a smoke experiment using local dataset (requires dataset path in config or --data-path):
just smoke

# Start MLflow UI
just mlflow

# Preview a Slurm sweep (JSON)
just preview

# Collect experiment summaries
just collect-summaries
```

Notes:
- Use `just check-data` to verify datasets are present before running experiments.
- Use `just check-data-load` to call the project loader and validate cache/CSV discovery (may take time for large datasets).

Notes on uv vs conda:
- `uv` is a lightweight package manager for local development. Use it for fast installs and lockfile-based reproducibility on developer machines.
- `conda` is recommended for cluster/HPC runs (OSC) because it handles binary packages (CUDA, NCCL) better.
- Recommended approach: develop locally with `uv`, run experiments on OSC using a `conda` environment configured in cluster submission.

If you'd like, I can scaffold a `uv` lockfile in this repo; tell me which `uv` commands you use (e.g., `uv install`, `uv lock`).

## Usage

### New Unified Training System (Recommended)

**All training now uses `train_with_hydra_zen.py` with type-safe configurations:**

```bash
# Train teacher models
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal
python train_with_hydra_zen.py --model vgae --dataset hcrl_sa --training autoencoder

# Knowledge distillation (compress to student)
python train_with_hydra_zen.py \
  --model gat_student \
  --dataset hcrl_sa \
  --training knowledge_distillation \
  --teacher-path experiment_runs/.../best_teacher_model.pth

# Curriculum learning (hard sample mining)
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training curriculum

# Multi-model fusion
python train_with_hydra_zen.py --model dqn --dataset hcrl_sa --training fusion
```

### Submit to SLURM (OSC)

---

## Documentation

**Complete guide**: [docs/README.md](docs/README.md)

### Essential Guides

- **[Getting Started](docs/GETTING_STARTED.md)** - Quick setup & first training run
- **[Code Templates](docs/CODE_TEMPLATES.md)** - Copy-paste working examples
- **[Workflow Guide](docs/WORKFLOW_GUIDE.md)** - Job submission & pipelines
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common errors & solutions

### Reference Docs

- **[Architecture Summary](docs/ARCHITECTURE_SUMMARY.md)** - System architecture
- **[Quick References](docs/QUICK_REFERENCES.md)** - Fast command lookup
- **[Job Templates](docs/JOB_TEMPLATES.md)** - Complete job configurations
- **[Model Calculations](docs/MODEL_SIZE_CALCULATIONS.md)** - Parameter budgets
A small convenience script helps create canonical experiment directories and optionally run a one-epoch smoke training (safe by default). Outputs are written to `experimentruns_test/` unless you set `--experiment-root`.

```bash
# Create canonical directories and inspect paths (no training)
python scripts/local_smoke_experiment.py --model vgae_student --dataset hcrl_ch --training autoencoder --epochs 1

# Attempt a one-epoch training run (may require dataset files)
python scripts/local_smoke_experiment.py --model vgae_student --dataset hcrl_ch --training autoencoder --epochs 1 --run

# Change where experiment outputs are placed
python scripts/local_smoke_experiment.py --experiment-root experimentruns_mytest --run
```

> Note: If your datasets live outside the repo defaults, set `--data-path` to point to the local dataset directory.

### Evaluation and Analysis
```bash
# Evaluate trained models
python evaluation.py

# Generate publication visualizations
jupyter notebook visuals_histogram.ipynb
```

### Key Training Files
- **[`osc_training_AD.py`](osc_training_AD.py)**: Main training pipeline
- **[`training-AD.py`](training-AD.py)**: Core GATPipeline implementation
- **[`AD_KD_GPU.py`](AD_KD_GPU.py)**: Knowledge distillation training
- **[`visuals_histogram.ipynb`](visuals_histogram.ipynb)**: Error analysis and visualization

---

## Datasets

The framework supports multiple CAN intrusion detection datasets with preprocessing pipelines:

| Name      | Description                  | Link                                      |
|-----------|-----------------------------|----------------------------------------------------|
| can-train-and-test | Primary dataset with HCRL and SET variants | [Link](https://bitbucket.org/brooke-lampe/can-train-and-test-v1.5/src/master/)  |
| HCRL-SA/CH | High-frequency CAN datasets     | Included in can-train-and-test      |
| SET_01-04 | Standardized evaluation datasets | Included in can-train-and-test      |

### Dataset Structure
```
datasets/
├── can-train-and-test-v1.5/
│   ├── hcrl-sa/
│   ├── hcrl-ch/
│   ├── set_01/
│   ├── set_02/
│   ├── set_03/
│   └── set_04/
```

---

## Model Architecture Details

### Stage 1: VGAE Anomaly Detection
- **Input**: CAN message graphs with temporal windows
- **Reconstruction Components**:
  - Continuous features (payload data)
  - CAN ID classification
  - Neighborhood structure prediction
- **Loss Function**: `L_total = L_node + L_canid + L_neighbor + β·L_kl`
- **Threshold**: Adaptive percentile-based on training reconstruction errors

### Stage 2: GAT Classification
- **Input**: Graphs flagged as anomalous by Stage 1
- **Architecture**: Multi-head attention with Jumping Knowledge
- *Configuration

**Single source of truth**: `src/config/hydra_zen_configs.py`

All model, dataset, and training configurations are consolidated in one place:

```python
from src.config.hydra_zen_configs import CANGraphConfigStore

store = CANGraphConfigStore()
config = store.create_config(
    model_type="gat",
    dataset_name="hcrl_sa", 
    training_mode="normal"
)
```

### Available Models

| Model | Parameters | Purpose |
|-------|------------|---------|
| `gat` | ~1.1M | Teacher classifier |
| `gat_student` | ~55K | Student classifier |
| `vgae` | ~1.74M | Teacher autoencoder |
| `vgae_student` | ~87K | Student autoencoder |
| `dqn` | ~687K | Fusion agent |
| `dqn_student` | ~32K | Student fusion |

### Training Modes

- `normal` - Standard supervised training
- `autoencoder` - Unsupervised VGAE training
- `curriculum` - Hard sample mining with VGAE guidance
- `knowledge_distillation` - Teacher→Student compression
- `fusion` - Multi-model ensemble with DQN

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for complete configuration guide.

---

## Project Structure

```
KD-GAT/
├── data/                         # CAN datasets
├── experiment_runs/              # Training outputs (canonical paths)
├── src/
│   ├── config/
│   │   └── hydra_zen_configs.py  # ⭐ Single config source
│   ├── models/                   # GAT, VGAE, DQN architectures
│   ├── training/
│   │   ├── trainer.py            # Unified trainer
│   │   ├── lightning_modules.py  # PyTorch Lightning wrappers
│   │   └── modes/                # Training modes (fusion, curriculum)
│   ├── paths.py                  # PathResolver for canonical paths
│   └── utils/                    # Utilities
├── train_with_hydra_zen.py       # ⭐ Main training script
├── oscjobmanager.py              # ⭐ SLURM job submission
├── docs/                         # ⭐ Complete documentation
└── examples/                     # Configuration examples
```

---

## Recent Updates (2026-01-24)

- ✅ **Configuration consolidated** - Single source of truth in `hydra_zen_configs.py`
- ✅ **Documentation streamlined** - Reduced from 30 to 12 essential files
- ✅ **Training unified** - All modes through `train_with_hydra_zen.py`
- ✅ **Paths standardized** - Canonical experiment directory structure
- ✅ **No backward compatibility** - Clean, maintainable codebase

See [CLEANUP_COMPLETE.md](CLEANUP_COMPLETE.md) for details.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{frenken2026kdgat,
  title={Multi-Stage Knowledge-Distilled VGAE and GAT for Robust CAN Intrusion Detection},
  author={Frenken, Robert},
  journal={TBD},
  year={2026}
}
```

---

## Support

- **Documentation**: [docs/README.md](docs/README.md)
- **Issues**: Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **Questions**: Open an issue on GitHub

**Ready to start?** → [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- [ ] Add pre-trained model releases
- [ ] Extend knowledge distillation to multi-teacher setups
- [ ] Add support for federated learning scenarios
- [ ] Implement additional graph construction methods
- [ ] Add comprehensive benchmarking against baseline methods