# Multi-Stage Knowledge-Distilled VGAE and GAT for Robust Controller-Area-Network Intrusion Detection

![License](https://img.shields.io/badge/license-MIT-blue.svg)
<!-- ![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg) -->
---

## Table of Contents

- [Description](#description)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Model Components](#model-components)
- [License](#license)
- [TODO](#TODO)

---

## Description

This framework implements a **multi-stage anomaly detection system** for Controller Area Network (CAN) intrusion detection using **Variational Graph Autoencoders (VGAE)** and **Graph Attention Networks (GAT)**. The system combines unsupervised anomaly detection with supervised classification in a two-stage pipeline, enhanced by knowledge distillation for efficient deployment.

The framework addresses CAN bus cybersecurity by:
- **Stage 1**: Unsupervised anomaly detection using VGAE with multiple reconstruction components
- **Stage 2**: Supervised classification of detected anomalies using GAT
- **Knowledge Distillation**: Training lightweight student models for edge device deployment

---

## Architecture

### Two-Stage Pipeline
1. **Anomaly Detection Stage**: 
   - Uses [`GraphAutoencoderNeighborhood`](models/models.py) to reconstruct:
     - Node features (continuous CAN data)
     - CAN ID predictions
     - Neighborhood structures
   - Computes composite reconstruction errors with weighted combination
   - Applies threshold-based filtering

2. **Classification Stage**:
   - Uses [`GATWithJK`](models/models.py) (Jumping Knowledge GAT) for attack type classification
   - Processes only graphs flagged as anomalous by Stage 1
   - Provides fine-grained attack classification

### Knowledge Distillation
- **Teacher Models**: Full-size VGAE + GAT pipeline
- **Student Models**: Compressed versions for resource-constrained environments
- **Distillation Process**: Soft label transfer with temperature scaling

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
```bash
git clone https://github.com/robertfrenken/CAN-Graph.git
cd CAN-Graph
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
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

### Training the Multi-Stage Pipeline
```bash
# Train both stages of the pipeline
python osc_training_AD.py

# Alternative training with knowledge distillation
python AD_KD_GPU.py
```

### Quick local smoke test ✅
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
- **Output**: Binary classification (normal vs. attack)
- **Training**: Only on filtered anomalous graphs

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## TODO

- [ ] Add configuration details for `base.yaml` files
- [ ] Implement real-time CAN data processing pipeline
- [ ] Add pre-trained model releases
- [ ] Extend knowledge distillation to multi-teacher setups
- [ ] Add support for federated learning scenarios
- [ ] Implement additional graph construction methods
- [ ] Add comprehensive benchmarking against baseline methods