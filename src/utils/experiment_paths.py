# ============================================================================
# Experiment Path Management for KD-GAT
# Strict, deterministic path generation with NO FALLBACKS
# ============================================================================

from pathlib import Path
from typing import Optional
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)

class ExperimentPathManager:
    """
    Manages experiment directory structure according to specification:
    L1: experimentruns (parent)
    L2: modality (automotive, internet, watertreatment)
    L3: dataset (hcrlch, set01, set02, set03, set04)
    L4: learning_type (unsupervised, classifier, fusion)
    L5: model_architecture (VGAE, GAT, DQN)
    L6: model_size (teacher, student, intermediate, huge, tiny)
    L7: distillation (yes, no, standard, topology_preserving)
    L8: training_type (all_samples, normals_only, curriculum_classifier, curriculum_fusion)
    L9: run_id (saved models, training metrics, validation metrics, evaluation results)
    """
    
    def __init__(self, cfg: DictConfig):
        """Initialize path manager with experiment config"""
        self.cfg = cfg
        self.experiment_root = Path(cfg.experiment_root)
        
        # Validate that experiment_root is set
        if str(self.experiment_root) == "None" or not self.experiment_root.parent.exists():
            raise ValueError(
                f"❌ experiment_root not properly configured: {self.experiment_root}\n"
                f"   Set 'experiment_root' in your config to a valid path"
            )
    
    def get_experiment_dir(self) -> Path:
        """
        Get the full experiment directory path (up to training_type level).
        NO CREATION - use get_experiment_dir_safe() for creation with error checking.
        """
        path = (
            self.experiment_root
            / self.cfg.modality
            / self.cfg.dataset
            / self.cfg.learning_type
            / self.cfg.model_architecture
            / self.cfg.model_size
            / self.cfg.distillation
            / self.cfg.training_mode
        )
        return path
    
    def get_experiment_dir_safe(self) -> Path:
        """
        Get experiment directory and create it (with error handling).
        Raises informative error if any component is misconfigured.
        """
        required_fields = [
            'modality', 'dataset', 'learning_type', 'model_architecture',
            'model_size', 'distillation', 'training_mode', 'experiment_root'
        ]
        
        # Validate all required fields
        missing = [f for f in required_fields if not hasattr(self.cfg, f) or getattr(self.cfg, f) is None]
        if missing:
            raise ValueError(
                f"❌ Missing required configuration fields: {missing}\n"
                f"   These must be set in your experiment config to determine save paths"
            )
        
        exp_dir = self.get_experiment_dir()
        
        try:
            exp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Experiment directory: {exp_dir.absolute()}")
        except Exception as e:
            raise IOError(
                f"❌ Failed to create experiment directory: {exp_dir}\n"
                f"   Error: {e}\n"
                f"   Check that {self.experiment_root} is accessible"
            )
        
        return exp_dir
    
    def get_run_dir(self, run_id: Optional[int] = None) -> Path:
        """
        Get run directory within experiment.
        If run_id not provided, finds next available run number.
        """
        exp_dir = self.get_experiment_dir()
        
        if run_id is None:
            # Find next available run number
            existing_runs = list(exp_dir.glob("run_*"))
            run_id = len(existing_runs)
        
        run_dir = exp_dir / f"run_{run_id:03d}"
        return run_dir
    
    def get_run_dir_safe(self, run_id: Optional[int] = None) -> Path:
        """
        Get run directory and create it (with error handling).
        """
        exp_dir = self.get_experiment_dir_safe()
        run_dir = self.get_run_dir(run_id)
        
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Run directory: {run_dir.absolute()}")
        except Exception as e:
            raise IOError(
                f"❌ Failed to create run directory: {run_dir}\n"
                f"   Error: {e}"
            )
        
        return run_dir
    
    def get_model_path(self, run_id: Optional[int] = None) -> Path:
        """Get path where trained model will be saved"""
        run_dir = self.get_run_dir(run_id)
        return run_dir / "model.pt"
    
    def get_checkpoint_dir(self, run_id: Optional[int] = None) -> Path:
        """Get directory for training checkpoints"""
        run_dir = self.get_run_dir(run_id)
        checkpoint_dir = run_dir / "checkpoints"
        return checkpoint_dir
    
    def get_training_metrics_path(self, run_id: Optional[int] = None) -> Path:
        """Get path where training metrics are saved"""
        run_dir = self.get_run_dir(run_id)
        return run_dir / "training_metrics.json"
    
    def get_validation_metrics_path(self, run_id: Optional[int] = None) -> Path:
        """Get path where validation metrics are saved"""
        run_dir = self.get_run_dir(run_id)
        return run_dir / "validation_metrics.json"
    
    def get_evaluation_dir(self, run_id: Optional[int] = None) -> Path:
        """
        Get evaluation results directory.
        Structure:
        - eval/
          - test_set/
          - known_vehicle_known_attacks/
          - known_vehicle_unknown_attacks/
          - unknown_vehicle_known_attacks/
          - unknown_vehicle_unknown_attacks/
        """
        run_dir = self.get_run_dir(run_id)
        eval_dir = run_dir / "evaluation"
        return eval_dir
    
    def get_config_path(self, run_id: Optional[int] = None) -> Path:
        """Get path where experiment config is saved (for reproducibility)"""
        run_dir = self.get_run_dir(run_id)
        return run_dir / "config.yaml"
    
    def print_structure(self):
        """Print the full hierarchical structure of this experiment"""
        exp_path = self.get_experiment_dir()
        print("\n" + "=" * 70)
        print("EXPERIMENT DIRECTORY STRUCTURE")
        print("=" * 70)
        print(f"Root:            {self.experiment_root.absolute()}")
        print(f"Modality:        {self.cfg.modality}")
        print(f"Dataset:         {self.cfg.dataset}")
        print(f"Learning Type:   {self.cfg.learning_type}")
        print(f"Model Arch:      {self.cfg.model_architecture}")
        print(f"Model Size:      {self.cfg.model_size}")
        print(f"Distillation:    {self.cfg.distillation}")
        print(f"Training Mode:   {self.cfg.training_mode}")
        print(f"\nFull Path:       {exp_path.absolute()}")
        print("=" * 70 + "\n")
    
    def validate_structure(self):
        """Validate that the experiment directory has proper structure"""
        exp_dir = self.get_experiment_dir_safe()
        
        # Count existing runs
        runs = list(exp_dir.glob("run_*"))
        
        print(f"\n✅ Experiment is ready")
        print(f"   Location: {exp_dir.absolute()}")
        print(f"   Existing runs: {len(runs)}")
        print(f"   Next run ID: run_{len(runs):03d}")


def build_experiment_path_from_config(cfg: DictConfig) -> Path:
    """
    Helper function to quickly get experiment path from config.
    Usage:
        path_manager = ExperimentPathManager(cfg)
        exp_dir = path_manager.get_experiment_dir_safe()
    """
    pm = ExperimentPathManager(cfg)
    return pm.get_experiment_dir_safe()


if __name__ == "__main__":
    # Example usage
    from omegaconf import OmegaConf
    
    sample_cfg = OmegaConf.create({
        "experiment_root": "/tmp/experimentruns",
        "modality": "automotive",
        "dataset": "hcrlch",
        "learning_type": "unsupervised",
        "model_architecture": "VGAE",
        "model_size": "student",
        "distillation": "no",
        "training_mode": "all_samples"
    })
    
    pm = ExperimentPathManager(sample_cfg)
    pm.print_structure()
    pm.validate_structure()
