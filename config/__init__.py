"""Configuration layer: inert, declarative, no imports from pipeline/ or src/.

Usage:
    from config import PipelineConfig, STAGES, checkpoint_path
    from config.resolver import resolve, list_models, list_auxiliaries
    from config.constants import NODE_FEATURE_COUNT, MMAP_TENSOR_LIMIT
"""
from .schema import (
    PipelineConfig,
    VGAEArchitecture,
    GATArchitecture,
    DQNArchitecture,
    AuxiliaryConfig,
    TrainingConfig,
    FusionConfig,
    PreprocessingConfig,
)
from .resolver import (
    resolve,
    list_models,
    list_auxiliaries,
)
from .paths import (
    EXPERIMENT_ROOT,
    STAGES,
    CATALOG_PATH,
    get_datasets,
    run_id,
    stage_dir,
    checkpoint_path,
    config_path,
    log_dir,
    data_dir,
    metrics_path,
    cache_dir,
    run_id_str,
    checkpoint_path_str,
    metrics_path_str,
    benchmark_path_str,
    log_path_str,
)
from .constants import (
    DB_PATH,
    PARQUET_ROOT,
    ROW_GROUP_SIZE,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STRIDE,
    EXCLUDED_ATTACK_TYPES,
    MAX_DATA_BYTES,
    NODE_FEATURE_COUNT,
    EDGE_FEATURE_COUNT,
    MMAP_TENSOR_LIMIT,
    CUDA_CONTEXT_MB,
    FRAGMENTATION_BUFFER,
    SLURM_ACCOUNT,
    SLURM_PARTITION,
    SLURM_GPU_TYPE,
    TRACKING_URI,
    EXPERIMENT_NAME,
)
