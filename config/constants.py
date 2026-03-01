"""Domain and infrastructure constants.

These are NOT hyperparameters (those live in PipelineConfig).
These are structural/environmental constants that rarely change.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
PREPROCESSING_VERSION = "1.2.0"  # Bump when graph construction logic changes

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------
CATALOG_PATH = Path(__file__).parent / "datasets.yaml"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_SIZE = 100
DEFAULT_STRIDE = 100
EXCLUDED_ATTACK_TYPES = ["suppress", "masquerade"]
MAX_DATA_BYTES = 8
NODE_FEATURE_COUNT = 11  # CAN_ID + 8 data bytes + count + position
EDGE_FEATURE_COUNT = 11  # Streamlined edge features

# ---------------------------------------------------------------------------
# DataLoader / memory mapping
# ---------------------------------------------------------------------------
# vm.max_map_count is typically 65530 on Linux.
# Both spawn workers and share_memory_() create mmap entries per tensor,
# so datasets exceeding this limit must use num_workers=0.
MMAP_TENSOR_LIMIT = 60000

# ---------------------------------------------------------------------------
# GPU memory estimation
# ---------------------------------------------------------------------------
CUDA_CONTEXT_MB = 500.0
FRAGMENTATION_BUFFER = 0.10

# ---------------------------------------------------------------------------
# SLURM defaults (override via environment for cluster migration)
# ---------------------------------------------------------------------------
import os

SLURM_ACCOUNT = os.getenv("KD_GAT_SLURM_ACCOUNT", "PAS3209")
SLURM_PARTITION = os.getenv("KD_GAT_SLURM_PARTITION", "gpu")
SLURM_GPU_TYPE = os.getenv("KD_GAT_GPU_TYPE", "v100")
