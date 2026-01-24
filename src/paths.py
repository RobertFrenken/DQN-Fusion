"""
Unified Path Management for KD-GAT Project

This module centralizes all path resolution logic, providing a single source
of truth for:
- Dataset paths (with fallback resolution)
- Cache paths
- Experiment/checkpoint/model paths
- MLflow paths
- Artifact paths

Usage:
    from src.paths import PathResolver
    
    resolver = PathResolver(config)
    dataset_path = resolver.resolve_dataset_path('hcrl_sa')
    cache_paths = resolver.get_cache_paths('hcrl_sa')
    exp_dir = resolver.get_experiment_dir()
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Standard dataset paths (relative to project root)
DATASET_PATHS = {
    'hcrl_ch': 'datasets/can-train-and-test-v1.5/hcrl-ch',
    'hcrl_sa': 'datasets/can-train-and-test-v1.5/hcrl-sa',
    'set_01': 'datasets/can-train-and-test-v1.5/set_01',
    'set_02': 'datasets/can-train-and-test-v1.5/set_02',
    'set_03': 'datasets/can-train-and-test-v1.5/set_03',
    'set_04': 'datasets/can-train-and-test-v1.5/set_04',
}

# Environment variables to check for dataset paths
DATASET_ENV_VARS = ['CAN_DATA_PATH', 'DATA_PATH', 'EXPERIMENT_DATA_PATH']

# Project root (computed once at module import)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# Path Resolver Class
# ============================================================================

class PathResolver:
    """
    Unified path resolver for all project paths.
    
    Provides consistent path resolution with fallbacks for:
    - Dataset paths (config → env vars → standard locations)
    - Cache paths (config → default locations)
    - Experiment paths (hierarchical structure)
    - Model artifact paths
    
    Args:
        config: Configuration object (Hydra-Zen or OmegaConf)
    """
    
    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize resolver with optional config."""
        self.config = config
        self._project_root = PROJECT_ROOT
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self._project_root
    
    # ========================================================================
    # Dataset Paths
    # ========================================================================
    
    def resolve_dataset_path(
        self, 
        dataset_name: str,
        explicit_path: Optional[str] = None
    ) -> Path:
        """
        Resolve dataset path with multiple fallbacks.
        
        Resolution order:
        1. Explicit path parameter
        2. Config dataset.data_path
        3. Environment variables (CAN_DATA_PATH, DATA_PATH, EXPERIMENT_DATA_PATH)
        4. Standard locations (project_root/datasets/..., project_root/data/automotive/...)
        
        Args:
            dataset_name: Name of dataset (e.g., 'hcrl_sa', 'set_01')
            explicit_path: Optional explicit path override
            
        Returns:
            Resolved dataset path
            
        Raises:
            FileNotFoundError: If dataset cannot be found in any location
        """
        candidates = []
        
        # 1) Explicit path parameter
        if explicit_path:
            candidates.append(Path(explicit_path))
        
        # 2) Config value
        if self.config and hasattr(self.config, 'dataset'):
            if hasattr(self.config.dataset, 'data_path') and self.config.dataset.data_path:
                candidates.append(Path(self.config.dataset.data_path))
        
        # 3) Environment variables
        for env_var in DATASET_ENV_VARS:
            env_path = os.environ.get(env_var)
            if env_path:
                candidates.append(Path(env_path))
        
        # 4) Standard locations
        standard_paths = [
            self._project_root / 'datasets' / 'can-train-and-test-v1.5' / dataset_name,
            self._project_root / 'data' / 'automotive' / dataset_name,
            self._project_root / 'datasets' / dataset_name,
        ]
        candidates.extend(standard_paths)
        
        # Find first existing path
        for candidate in candidates:
            try:
                if candidate and candidate.exists():
                    logger.info(f"✅ Dataset path resolved: {candidate}")
                    return candidate
            except Exception:
                continue
        
        # Not found - raise informative error
        msg_lines = [f"Dataset '{dataset_name}' not found. Tried the following locations:"]
        for c in candidates:
            msg_lines.append(f"  - {c}")
        msg_lines.extend([
            "\nSuggestions:",
            "  - Set `config.dataset.data_path` in your config",
            "  - Pass explicit_path to resolve_dataset_path()",
            "  - Set CAN_DATA_PATH environment variable",
            f"  - Place dataset at: {standard_paths[0]}"
        ])
        raise FileNotFoundError("\n".join(msg_lines))
    
    # ========================================================================
    # Cache Paths
    # ========================================================================
    
    def get_cache_paths(
        self, 
        dataset_name: str
    ) -> Tuple[bool, Path, Path]:
        """
        Get cache file paths for a dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Tuple of (cache_enabled, cache_file_path, id_mapping_file_path)
        """
        # Determine if caching is enabled
        cache_enabled = True
        if self.config and hasattr(self.config, 'dataset'):
            if hasattr(self.config.dataset, 'get'):
                # Dict-like config
                cache_enabled = self.config.dataset.get('preprocessing', {}).get('cache_processed_data', True)
            else:
                # Dataclass config
                cache_enabled = getattr(self.config.dataset, 'cache_processed_data', True)
        
        # Determine cache directory
        cache_dir = None
        if self.config and hasattr(self.config, 'dataset'):
            if hasattr(self.config.dataset, 'get'):
                cache_dir = self.config.dataset.get('cache_dir')
            else:
                cache_dir = getattr(self.config.dataset, 'cache_dir', None)
        
        # Default cache directory
        if not cache_dir:
            cache_dir = self._project_root / 'experiment_runs' / 'automotive' / dataset_name / 'cache'
        else:
            cache_dir = Path(cache_dir)
        
        cache_file = cache_dir / 'processed_graphs.pt'
        id_mapping_file = cache_dir / 'id_mapping.pkl'
        
        return cache_enabled, cache_file, id_mapping_file
    
    # ========================================================================
    # Experiment Paths
    # ========================================================================
    
    def get_experiment_root(self) -> Path:
        """
        Get experiment root directory from config.
        
        Returns:
            Experiment root path
            
        Raises:
            ValueError: If experiment_root not properly configured
        """
        if not self.config:
            return self._project_root / 'experiment_runs'
        
        experiment_root = getattr(self.config, 'experiment_root', None)
        if not experiment_root or str(experiment_root) == 'None':
            # Try to get from dataset config
            if hasattr(self.config, 'dataset'):
                experiment_root = getattr(self.config.dataset, 'experiment_root', None)
        
        if not experiment_root or str(experiment_root) == 'None':
            experiment_root = self._project_root / 'experiment_runs'
        
        return Path(experiment_root)
    
    def get_experiment_dir(self, create: bool = False) -> Path:
        """
        Get full hierarchical experiment directory path.
        
        Structure:
        experiment_root / modality / dataset / learning_type / model_architecture /
        model_size / distillation / training_mode
        
        Args:
            create: If True, create directory (with parents)
            
        Returns:
            Experiment directory path
            
        Raises:
            ValueError: If required config fields are missing
        """
        if not self.config:
            raise ValueError("Config required to determine experiment directory")
        
        # Required fields
        required = ['modality', 'dataset', 'learning_type', 'model_architecture',
                   'model_size', 'distillation', 'training_mode']
        
        missing = []
        for field in required:
            if not hasattr(self.config, field) or getattr(self.config, field) is None:
                missing.append(field)
        
        if missing:
            raise ValueError(
                f"Missing required config fields for experiment path: {missing}"
            )
        
        # Build hierarchical path
        exp_root = self.get_experiment_root()
        
        # Handle dataset - may be string or object with .name attribute
        dataset_name = self.config.dataset
        if hasattr(dataset_name, 'name'):
            dataset_name = dataset_name.name
        
        path = (
            exp_root
            / self.config.modality
            / str(dataset_name)
            / self.config.learning_type
            / self.config.model_architecture
            / self.config.model_size
            / self.config.distillation
            / self.config.training_mode
        )
        
        if create:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Experiment directory: {path}")
        
        return path
    
    def get_checkpoint_dir(self, create: bool = False) -> Path:
        """Get checkpoint directory within experiment."""
        exp_dir = self.get_experiment_dir(create=create)
        checkpoint_dir = exp_dir / 'checkpoints'
        if create:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    
    def get_model_save_dir(self, create: bool = False) -> Path:
        """Get model save directory within experiment."""
        exp_dir = self.get_experiment_dir(create=create)
        model_dir = exp_dir / 'models'
        if create:
            model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def get_log_dir(self, create: bool = False) -> Path:
        """Get log directory within experiment."""
        exp_dir = self.get_experiment_dir(create=create)
        log_dir = exp_dir / 'logs'
        if create:
            log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def get_mlruns_dir(self, create: bool = False) -> Path:
        """Get MLflow runs directory within experiment."""
        exp_dir = self.get_experiment_dir(create=create)
        mlruns_dir = exp_dir / 'mlruns'
        if create:
            mlruns_dir.mkdir(parents=True, exist_ok=True)
        return mlruns_dir
    
    def get_all_experiment_dirs(self, create: bool = False) -> Dict[str, Path]:
        """
        Get all experiment-related directories.
        
        Args:
            create: If True, create all directories
            
        Returns:
            Dictionary with keys: experiment_dir, checkpoint_dir, model_save_dir,
            log_dir, mlruns_dir
        """
        exp_dir = self.get_experiment_dir(create=create)
        
        dirs = {
            'experiment_dir': exp_dir,
            'checkpoint_dir': self.get_checkpoint_dir(create=create),
            'model_save_dir': self.get_model_save_dir(create=create),
            'log_dir': self.get_log_dir(create=create),
            'mlruns_dir': self.get_mlruns_dir(create=create),
        }
        
        return dirs
    
    # ========================================================================
    # Model Artifact Paths
    # ========================================================================
    
    def resolve_teacher_path(
        self, 
        explicit_path: Optional[str] = None
    ) -> Optional[Path]:
        """
        Resolve path to teacher model for knowledge distillation.
        
        Args:
            explicit_path: Optional explicit path override
            
        Returns:
            Path to teacher model, or None if not applicable
        """
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Teacher model not found: {path}")
        
        # Try to get from config
        if not self.config:
            return None
        
        if hasattr(self.config, 'training'):
            teacher_path = getattr(self.config.training, 'teacher_path', None)
            if teacher_path:
                path = Path(teacher_path)
                if path.exists():
                    return path
        
        # Try to get from artifacts
        if hasattr(self.config, 'get_required_artifacts'):
            artifacts = self.config.get_required_artifacts()
            if 'teacher_model' in artifacts:
                return Path(artifacts['teacher_model'])
        
        return None
    
    def resolve_autoencoder_path(
        self,
        explicit_path: Optional[str] = None
    ) -> Optional[Path]:
        """
        Resolve path to autoencoder (VGAE) model.
        
        Args:
            explicit_path: Optional explicit path override
            
        Returns:
            Path to autoencoder model, or None if not applicable
        """
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Autoencoder model not found: {path}")
        
        # Try to get from config
        if not self.config:
            return None
        
        if hasattr(self.config, 'training'):
            ae_path = getattr(self.config.training, 'autoencoder_path', None)
            if ae_path:
                path = Path(ae_path)
                if path.exists():
                    return path
        
        return None
    
    def resolve_classifier_path(
        self,
        explicit_path: Optional[str] = None
    ) -> Optional[Path]:
        """
        Resolve path to classifier (GAT) model.
        
        Args:
            explicit_path: Optional explicit path override
            
        Returns:
            Path to classifier model, or None if not applicable
        """
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Classifier model not found: {path}")
        
        # Try to get from config
        if not self.config:
            return None
        
        if hasattr(self.config, 'training'):
            clf_path = getattr(self.config.training, 'classifier_path', None)
            if clf_path:
                path = Path(clf_path)
                if path.exists():
                    return path
        
        return None
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def print_structure(self):
        """Print the full hierarchical path structure."""
        print("\n" + "=" * 70)
        print("PATH RESOLVER CONFIGURATION")
        print("=" * 70)
        print(f"Project Root:     {self.project_root}")
        
        if self.config:
            try:
                exp_root = self.get_experiment_root()
                print(f"Experiment Root:  {exp_root}")
                
                exp_dir = self.get_experiment_dir()
                print(f"Experiment Dir:   {exp_dir}")
                print(f"  ├─ checkpoints/ {self.get_checkpoint_dir()}")
                print(f"  ├─ models/      {self.get_model_save_dir()}")
                print(f"  ├─ logs/        {self.get_log_dir()}")
                print(f"  └─ mlruns/      {self.get_mlruns_dir()}")
            except Exception as e:
                print(f"Cannot determine experiment paths: {e}")
        
        print("=" * 70 + "\n")


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def resolve_dataset_path(
    dataset_name: str, 
    config=None, 
    explicit_path: Optional[str] = None
) -> Path:
    """
    Backward-compatible function for dataset path resolution.
    
    Args:
        dataset_name: Name of dataset
        config: Optional config object
        explicit_path: Optional explicit path
        
    Returns:
        Resolved dataset path
    """
    resolver = PathResolver(config)
    return resolver.resolve_dataset_path(dataset_name, explicit_path)


def get_cache_paths(
    dataset_name: str,
    config=None
) -> Tuple[bool, Path, Path]:
    """
    Backward-compatible function for cache path resolution.
    
    Args:
        dataset_name: Name of dataset
        config: Optional config object
        
    Returns:
        Tuple of (cache_enabled, cache_file, id_mapping_file)
    """
    resolver = PathResolver(config)
    return resolver.get_cache_paths(dataset_name)
