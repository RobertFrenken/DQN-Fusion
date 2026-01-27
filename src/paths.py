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
# CORRECT PATHS: data/automotive/{dataset}
DATASET_PATHS = {
    'hcrl_ch': 'data/automotive/hcrl_ch',
    'hcrl_sa': 'data/automotive/hcrl_sa',
    'set_01': 'data/automotive/set_01',
    'set_02': 'data/automotive/set_02',
    'set_03': 'data/automotive/set_03',
    'set_04': 'data/automotive/set_04',
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
        
        # 4) Standard locations - data/automotive is the CORRECT primary location
        standard_paths = [
            self._project_root / 'data' / 'automotive' / dataset_name,
            self._project_root / 'datasets' / dataset_name,
            self._project_root / 'datasets' / 'can-train-and-test-v1.5' / dataset_name,
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
            cache_dir = self._project_root / 'experimentruns' / 'automotive' / dataset_name / 'cache'
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
            return self._project_root / 'experimentruns'

        experiment_root = getattr(self.config, 'experiment_root', None)
        if not experiment_root or str(experiment_root) == 'None':
            # Try to get from dataset config
            if hasattr(self.config, 'dataset'):
                experiment_root = getattr(self.config.dataset, 'experiment_root', None)

        if not experiment_root or str(experiment_root) == 'None':
            experiment_root = self._project_root / 'experimentruns'

        return Path(experiment_root)
    
    def get_experiment_dir(self, create: bool = False) -> Path:
        """
        Get full hierarchical experiment directory path.
        
        Uses the config's canonical_experiment_dir() method to compute the path,
        which handles all the logic for determining learning_type, model_architecture, etc.
        
        Args:
            create: If True, create directory (with parents)
            
        Returns:
            Experiment directory path
            
        Raises:
            ValueError: If config is missing or doesn't support canonical_experiment_dir()
        """
        if not self.config:
            raise ValueError("Config required to determine experiment directory")
        
        # Use the config's method to compute canonical experiment directory
        if hasattr(self.config, 'canonical_experiment_dir'):
            path = self.config.canonical_experiment_dir()
        else:
            raise ValueError(
                "Config must have canonical_experiment_dir() method"
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

    def get_run_counter(self) -> int:
        """Get next run number and increment counter.

        Returns the current run number (1-indexed) and increments the counter
        for the next run. Stores state in experiment_dir/run_counter.txt.

        First call returns 1, creates file with "2"
        Second call returns 2, updates file to "3"
        etc.

        Returns:
            int: Current run number (1-indexed)

        Example:
            Run 1: get_run_counter() → 1 (file now contains "2")
            Run 2: get_run_counter() → 2 (file now contains "3")
        """
        exp_dir = self.get_experiment_dir(create=True)
        counter_file = exp_dir / 'run_counter.txt'

        # Read current value or default to 1
        if counter_file.exists():
            try:
                with open(counter_file, 'r') as f:
                    next_run = int(f.read().strip())
            except (ValueError, IOError):
                logger.warning(f"Could not read run_counter.txt, resetting to 1")
                next_run = 1
        else:
            next_run = 1

        # Write incremented value for future runs
        try:
            with open(counter_file, 'w') as f:
                f.write(str(next_run + 1))
        except IOError as e:
            logger.error(f"Could not write run_counter.txt: {e}")

        return next_run

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
    # Flexible Model Discovery (glob-based)
    # ========================================================================
    
    def discover_model(
        self,
        base_dir: Path,
        model_type: str,
        patterns: Optional[List[str]] = None,
        require_exists: bool = True
    ) -> Optional[Path]:
        """
        Discover a model file using flexible glob patterns.
        
        This is the PREFERRED method for finding models - it navigates to the 
        canonical directory and uses pattern matching rather than hardcoded paths.
        
        Args:
            base_dir: Base directory to search in (typically from canonical_experiment_dir())
            model_type: Type of model ('vgae', 'gat', 'dqn', 'fusion')
            patterns: Optional list of glob patterns to try. If None, uses defaults.
            require_exists: If True, raises FileNotFoundError when not found
            
        Returns:
            Path to discovered model, or None if not found (and require_exists=False)
            
        Raises:
            FileNotFoundError: If require_exists=True and no model found
            
        Example:
            >>> resolver = PathResolver(config)
            >>> vgae_dir = resolver.get_experiment_dir() / ".." / "autoencoder"
            >>> model = resolver.discover_model(vgae_dir, "vgae")
        """
        # Default patterns for each model type
        default_patterns = {
            'vgae': ['models/vgae*.pth', 'vgae*.pth', '**/*vgae*.pth'],
            'gat': ['models/gat*.pth', 'gat*.pth', 'best_*.pth', '**/*gat*.pth'],
            'dqn': ['models/dqn*.pth', 'dqn*.pth', 'agent*.pth', '**/*dqn*.pth'],
            'fusion': ['models/fusion*.pth', 'fusion*.pth', '**/*fusion*.pth'],
            'teacher': ['models/*teacher*.pth', '*teacher*.pth', 'best_*.pth'],
            'student': ['models/*student*.pth', '*student*.pth'],
        }
        
        search_patterns = patterns or default_patterns.get(model_type, ['models/*.pth', '*.pth'])
        base_dir = Path(base_dir)
        
        if not base_dir.exists():
            if require_exists:
                raise FileNotFoundError(
                    f"Base directory does not exist: {base_dir}\n"
                    f"Cannot discover {model_type} model."
                )
            return None
        
        # Try each pattern in order
        for pattern in search_patterns:
            matches = list(base_dir.glob(pattern))
            if matches:
                # If multiple matches, prefer the most recently modified
                if len(matches) > 1:
                    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    logger.info(f"Multiple {model_type} models found, using most recent: {matches[0].name}")
                return matches[0]
        
        if require_exists:
            raise FileNotFoundError(
                f"No {model_type} model found in {base_dir}\n"
                f"Searched patterns: {search_patterns}\n"
                f"Directory contents: {list(base_dir.iterdir()) if base_dir.exists() else 'N/A'}"
            )
        return None
    
    def discover_artifact(
        self,
        artifact_type: str,
        config_override: Optional['DictConfig'] = None,
        require_exists: bool = True
    ) -> Optional[Path]:
        """
        High-level artifact discovery using config semantics.
        
        This combines canonical directory resolution with flexible model discovery.
        It's the single entry point for "find me the VGAE for this config".
        
        Args:
            artifact_type: One of 'vgae', 'gat_teacher', 'gat_curriculum', 'classifier', 'autoencoder'
            config_override: Optional config to use instead of self.config
            require_exists: If True, raises FileNotFoundError when not found
            
        Returns:
            Path to artifact, or None if not found (and require_exists=False)
        """
        cfg = config_override or self.config
        if not cfg:
            raise ValueError("Config required for artifact discovery")
        
        exp_root = Path(getattr(cfg, 'experiment_root', self._project_root / 'experimentruns'))
        modality = getattr(cfg, 'modality', 'automotive')
        dataset_name = getattr(cfg.dataset, 'name', 'unknown') if hasattr(cfg, 'dataset') else 'unknown'
        
        # Map artifact types to their canonical locations
        artifact_dirs = {
            'vgae': exp_root / modality / dataset_name / "unsupervised" / "vgae" / "teacher" / "no_distillation" / "autoencoder",
            'vgae_student': exp_root / modality / dataset_name / "unsupervised" / "vgae" / "student" / "distilled" / "knowledge_distillation",
            'gat_teacher': exp_root / modality / dataset_name / "supervised" / "gat" / "teacher" / "no_distillation" / "normal",
            'gat_curriculum': exp_root / modality / dataset_name / "supervised" / "gat" / "teacher" / "no_distillation" / "curriculum",
            'gat_student': exp_root / modality / dataset_name / "supervised" / "gat" / "student" / "distilled" / "knowledge_distillation",
        }
        
        # Determine model_type for pattern matching
        model_type_map = {
            'vgae': 'vgae', 'vgae_student': 'vgae',
            'gat_teacher': 'gat', 'gat_curriculum': 'gat', 'gat_student': 'gat',
        }
        
        if artifact_type not in artifact_dirs:
            raise ValueError(f"Unknown artifact type: {artifact_type}. Valid: {list(artifact_dirs.keys())}")
        
        base_dir = artifact_dirs[artifact_type]
        model_type = model_type_map.get(artifact_type, artifact_type)
        
        return self.discover_model(base_dir, model_type, require_exists=require_exists)
    
    def discover_artifact_for_mode(
        self,
        mode: str,
        artifact_name: str,
        experiment_root: Optional[Path] = None,
        modality: str = "automotive",
        dataset_name: str = "hcrl_sa",
        model_size: str = "teacher",
        distillation: str = "no_distillation",
        require_exists: bool = False
    ) -> Optional[Path]:
        """
        Mode-aware artifact discovery - the canonical way to find any required model.
        
        This is the PREFERRED method for finding dependent models. Given a training
        mode and the artifact needed, it navigates to the canonical directory and
        uses glob-based discovery to find the model regardless of exact filename.
        
        Args:
            mode: Training mode that needs the artifact ('curriculum', 'fusion', 'knowledge_distillation')
            artifact_name: Name of artifact ('vgae', 'gat', 'teacher_model', 'autoencoder', 'classifier')
            experiment_root: Base experiment directory
            modality: Data modality (default 'automotive')
            dataset_name: Dataset name (default 'hcrl_sa')
            model_size: 'teacher' or 'student'
            distillation: 'no_distillation' or 'distilled'
            require_exists: If True and not found, raises FileNotFoundError
            
        Returns:
            Path to discovered artifact, or None if not found (and require_exists=False)
            
        Example:
            >>> resolver = PathResolver(config)
            >>> vgae = resolver.discover_artifact_for_mode(
            ...     mode='curriculum',
            ...     artifact_name='vgae',
            ...     experiment_root=Path('experimentruns/automotive/hcrl_sa')
            ... )
        """
        exp_root = experiment_root or (self._project_root / 'experimentruns')
        base = exp_root / modality / dataset_name if modality not in str(exp_root) else exp_root
        
        # Define canonical locations for each mode's required artifacts
        artifact_locations = {
            # CURRICULUM mode needs VGAE from autoencoder training
            ('curriculum', 'vgae'): (
                base / "unsupervised" / "vgae" / "teacher" / "no_distillation" / "autoencoder",
                'vgae'
            ),
            
            # FUSION mode (teacher) needs VGAE and GAT
            ('fusion', 'autoencoder'): (
                base / "unsupervised" / "vgae" / model_size / distillation / "autoencoder",
                'vgae'
            ),
            ('fusion', 'classifier'): (
                base / "supervised" / "gat" / model_size / distillation / "curriculum",
                'gat'
            ),
            
            # FUSION mode (student) 
            ('fusion_student', 'autoencoder'): (
                base / "unsupervised" / "vgae" / "student" / "distilled" / "knowledge_distillation",
                'vgae'
            ),
            ('fusion_student', 'classifier'): (
                base / "supervised" / "gat" / "student" / "distilled" / "knowledge_distillation",
                'gat'
            ),
            
            # KNOWLEDGE_DISTILLATION needs teacher models
            ('knowledge_distillation', 'teacher_vgae'): (
                base / "unsupervised" / "vgae" / "teacher" / "no_distillation" / "autoencoder",
                'vgae'
            ),
            ('knowledge_distillation', 'teacher_gat'): (
                # Try curriculum first, then normal
                [
                    base / "supervised" / "gat" / "teacher" / "no_distillation" / "curriculum",
                    base / "supervised" / "gat" / "teacher" / "no_distillation" / "normal",
                ],
                'gat'
            ),
        }
        
        key = (mode, artifact_name)
        if key not in artifact_locations:
            # Try without mode prefix for generic artifacts
            for (m, a), (loc, model_type) in artifact_locations.items():
                if a == artifact_name:
                    key = (m, a)
                    break
            else:
                if require_exists:
                    raise ValueError(f"Unknown artifact: mode={mode}, artifact={artifact_name}")
                return None
        
        location_info, model_type = artifact_locations[key]
        
        # Handle multiple possible locations (e.g., curriculum OR normal for GAT teacher)
        if isinstance(location_info, list):
            for loc in location_info:
                result = self.discover_model(loc, model_type, require_exists=False)
                if result:
                    return result
            if require_exists:
                raise FileNotFoundError(
                    f"Artifact '{artifact_name}' for mode '{mode}' not found.\n"
                    f"Searched locations:\n" + "\n".join(f"  - {loc}" for loc in location_info)
                )
            return None
        else:
            return self.discover_model(location_info, model_type, require_exists=require_exists)
    
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
