"""
Training Presets and Common Hyperparameter Configurations

This module provides ready-to-use training presets for different scenarios:
- Teacher training (normal, autoencoder, curriculum)
- Student baselines (individual learning without KD)
- Knowledge distillation (with various temperatures and alphas)
- Fusion training (multi-model DQN agent)

Usage:
    from src.config.training_presets import get_preset
    config = get_preset("aggressive_distillation_hcrl_sa")
    
    # Or programmatically:
    from src.config.hydra_zen_configs import CANGraphConfigStore
    store = CANGraphConfigStore()
    config = store.create_config(**preset_params)
"""

from typing import Dict, Any
from dataclasses import asdict


# ============================================================================
# Preset Categories and Configurations
# ============================================================================

class TeacherPresets:
    """Teacher model training presets."""
    
    @staticmethod
    def gat_normal(dataset: str = "hcrl_sa") -> Dict[str, Any]:
        """Teacher GAT with standard supervised training."""
        return {
            "model_type": "gat",
            "dataset_name": dataset,
            "training_mode": "normal",
            "overrides": {
                "max_epochs": 400,
                "batch_size": 64,
                "learning_rate": 0.001,
                "early_stopping_patience": 100,
            }
        }
    
    @staticmethod
    def vgae_autoencoder(dataset: str = "hcrl_sa") -> Dict[str, Any]:
        """Teacher VGAE autoencoder on normal samples only."""
        return {
            "model_type": "vgae",
            "dataset_name": dataset,
            "training_mode": "autoencoder",
            "overrides": {
                "max_epochs": 400,
                "batch_size": 64,
                "learning_rate": 0.0005,
                "early_stopping_patience": 100,
            }
        }
    
    @staticmethod
    def gat_curriculum(dataset: str = "hcrl_sa") -> Dict[str, Any]:
        """Teacher GAT with curriculum learning and hard mining."""
        return {
            "model_type": "gat",
            "dataset_name": dataset,
            "training_mode": "curriculum",
            "overrides": {
                "max_epochs": 400,
                "batch_size": 32,
                "learning_rate": 0.001,
                "early_stopping_patience": 150,
                "optimize_batch_size": True,
            }
        }


class StudentBaselinePresets:
    """Student-only training (no knowledge distillation)."""
    
    @staticmethod
    def gat_student_baseline(dataset: str = "hcrl_sa") -> Dict[str, Any]:
        """Student GAT trained independently (baseline for KD comparison)."""
        return {
            "model_type": "gat_student",
            "dataset_name": dataset,
            "training_mode": "student_baseline",
            "overrides": {
                "max_epochs": 300,
                "batch_size": 64,
                "learning_rate": 0.001,
                "early_stopping_patience": 50,
                "precision": "16-mixed",
            }
        }
    
    @staticmethod
    def vgae_student_baseline(dataset: str = "hcrl_sa") -> Dict[str, Any]:
        """Student VGAE trained independently."""
        return {
            "model_type": "vgae_student",
            "dataset_name": dataset,
            "training_mode": "student_baseline",
            "overrides": {
                "max_epochs": 300,
                "batch_size": 64,
                "learning_rate": 0.001,
                "early_stopping_patience": 50,
                "precision": "16-mixed",
            }
        }


class DistillationPresets:
    """Knowledge distillation presets with different aggressiveness levels."""
    
    @staticmethod
    def conservative_distillation(dataset: str = "hcrl_sa", 
                                 teacher_path: str = None) -> Dict[str, Any]:
        """Conservative KD: prioritize student task loss."""
        if teacher_path is None:
            teacher_path = str(Path(__file__).parent.resolve() / "experiment_runs" / "automotive" / dataset / "supervised" / "gat" / "teacher" / "no_distillation" / "normal" / f"best_teacher_model_{dataset}.pth")
        
        return {
            "model_type": "gat_student",
            "dataset_name": dataset,
            "training_mode": "knowledge_distillation",
            "overrides": {
                "teacher_model_path": teacher_path,
                "distillation_temperature": 2.0,  # Lower temp = sharper soft targets
                "distillation_alpha": 0.3,        # More weight on hard targets
                "max_epochs": 400,
                "batch_size": 64,
                "learning_rate": 0.002,
                "precision": "16-mixed",
            }
        }
    
    @staticmethod
    def balanced_distillation(dataset: str = "hcrl_sa",
                             teacher_path: str = None) -> Dict[str, Any]:
        """Balanced KD: equal weight on hard and soft targets."""
        if teacher_path is None:
            teacher_path = str(Path(__file__).parent.resolve() / "experiment_runs" / "automotive" / dataset / "supervised" / "gat" / "teacher" / "no_distillation" / "normal" / f"best_teacher_model_{dataset}.pth")
        
        return {
            "model_type": "gat_student",
            "dataset_name": dataset,
            "training_mode": "knowledge_distillation",
            "overrides": {
                "teacher_model_path": teacher_path,
                "distillation_temperature": 4.0,  # Moderate temp
                "distillation_alpha": 0.5,        # Equal weight
                "max_epochs": 400,
                "batch_size": 64,
                "learning_rate": 0.002,
                "precision": "16-mixed",
            }
        }
    
    @staticmethod
    def aggressive_distillation(dataset: str = "hcrl_sa",
                               teacher_path: str = None) -> Dict[str, Any]:
        """Aggressive KD: prioritize knowledge transfer."""
        if teacher_path is None:
            teacher_path = str(Path(__file__).parent.resolve() / "experiment_runs" / "automotive" / dataset / "supervised" / "gat" / "teacher" / "no_distillation" / "normal" / f"best_teacher_model_{dataset}.pth")
        
        return {
            "model_type": "gat_student",
            "dataset_name": dataset,
            "training_mode": "knowledge_distillation",
            "overrides": {
                "teacher_model_path": teacher_path,
                "distillation_temperature": 8.0,  # Higher temp = softer targets
                "distillation_alpha": 0.7,        # More weight on soft targets
                "max_epochs": 400,
                "batch_size": 32,  # Smaller for more stable KD
                "learning_rate": 0.001,
                "precision": "16-mixed",
                "accumulate_grad_batches": 2,
            }
        }
    
    @staticmethod
    def vgae_student_distillation(dataset: str = "hcrl_sa",
                                 teacher_path: str = None) -> Dict[str, Any]:
        """VGAE student with knowledge distillation from teacher VGAE."""
        if teacher_path is None:
            teacher_path = str(Path(__file__).parent.resolve() / "experiment_runs" / "automotive" / dataset / "unsupervised" / "vgae" / "teacher" / "no_distillation" / "autoencoder" / f"autoencoder_{dataset}.pth")
        
        return {
            "model_type": "vgae_student",
            "dataset_name": dataset,
            "training_mode": "knowledge_distillation",
            "overrides": {
                "teacher_model_path": teacher_path,
                "distillation_temperature": 4.0,
                "distillation_alpha": 0.7,
                "max_epochs": 400,
                "batch_size": 64,
                "learning_rate": 0.001,
                "precision": "16-mixed",
            }
        }


class FusionPresets:
    """Fusion training (multi-model DQN agent)."""
    
    @staticmethod
    def fusion_standard(dataset: str = "hcrl_sa") -> Dict[str, Any]:
        """Standard fusion training with cached predictions."""
        return {
            "model_type": "gat",
            "dataset_name": dataset,
            "training_mode": "fusion",
            "overrides": {
                "fusion_episodes": 500,
                "max_epochs": 500,
                "batch_size": 32768,  # Large for replay buffer
                "learning_rate": 0.001,
            }
        }
    
    @staticmethod
    def fusion_lightweight(dataset: str = "hcrl_sa") -> Dict[str, Any]:
        """Lightweight fusion for quick iteration."""
        return {
            "model_type": "gat",
            "dataset_name": dataset,
            "training_mode": "fusion",
            "overrides": {
                "fusion_episodes": 200,
                "max_epochs": 200,
                "batch_size": 8192,
                "learning_rate": 0.002,
            }
        }


# ============================================================================
# Preset Lookup and Helper Functions
# ============================================================================

PRESET_REGISTRY = {
    # Teacher presets
    "teacher_gat_normal": TeacherPresets.gat_normal,
    "teacher_vgae_autoencoder": TeacherPresets.vgae_autoencoder,
    "teacher_gat_curriculum": TeacherPresets.gat_curriculum,
    
    # Student baseline presets
    "student_gat_baseline": StudentBaselinePresets.gat_student_baseline,
    "student_vgae_baseline": StudentBaselinePresets.vgae_student_baseline,
    
    # Distillation presets
    "distillation_conservative": DistillationPresets.conservative_distillation,
    "distillation_balanced": DistillationPresets.balanced_distillation,
    "distillation_aggressive": DistillationPresets.aggressive_distillation,
    "distillation_vgae_student": DistillationPresets.vgae_student_distillation,
    
    # Fusion presets
    "fusion_standard": FusionPresets.fusion_standard,
    "fusion_lightweight": FusionPresets.fusion_lightweight,
}


def list_presets() -> None:
    """Print all available presets."""
    print("📋 AVAILABLE TRAINING PRESETS")
    print("=" * 60)
    
    categories = {
        "🎓 Teacher Training": [k for k in PRESET_REGISTRY if k.startswith("teacher_")],
        "📚 Student Baselines": [k for k in PRESET_REGISTRY if k.startswith("student_")],
        "🔬 Knowledge Distillation": [k for k in PRESET_REGISTRY if k.startswith("distillation_")],
        "🔀 Fusion Training": [k for k in PRESET_REGISTRY if k.startswith("fusion_")],
    }
    
    for category, presets in categories.items():
        if presets:
            print(f"\n{category}:")
            for preset in presets:
                print(f"  • {preset}")


def get_preset(preset_name: str, dataset: str = "hcrl_sa") -> Dict[str, Any]:
    """
    Get a training preset by name.
    
    Args:
        preset_name: Name of the preset (e.g., 'distillation_aggressive')
        dataset: Dataset name to use (default: hcrl_sa)
    
    Returns:
        Dictionary with model_type, dataset_name, training_mode, and overrides
    
    Raises:
        ValueError: If preset not found
    """
    if preset_name not in PRESET_REGISTRY:
        available = ", ".join(PRESET_REGISTRY.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    preset_func = PRESET_REGISTRY[preset_name]
    return preset_func(dataset=dataset)


def get_preset_with_overrides(preset_name: str, dataset: str = "hcrl_sa",
                             **custom_overrides) -> Dict[str, Any]:
    """
    Get a preset and merge custom overrides.
    
    Example:
        config = get_preset_with_overrides(
            "distillation_balanced", 
            dataset="hcrl_ch",
            max_epochs=200,
            learning_rate=0.0005
        )
    """
    preset = get_preset(preset_name, dataset=dataset)
    preset["overrides"].update(custom_overrides)
    return preset


# ============================================================================
# Quick Reference Documentation
# ============================================================================

PRESET_DOCUMENTATION = """
TRAINING PRESETS QUICK REFERENCE
=================================

Teacher Training:
  teacher_gat_normal        → GAT on full dataset with supervised loss
  teacher_vgae_autoencoder  → VGAE on normal samples only (reconstruction)
  teacher_gat_curriculum    → GAT with momentum curriculum + hard mining

Student Baselines (no teacher):
  student_gat_baseline      → Small GAT trained independently  
  student_vgae_baseline     → Small VGAE trained independently

Knowledge Distillation:
  distillation_conservative → Prioritize task loss (α=0.3, T=2.0)
  distillation_balanced     → Equal weight (α=0.5, T=4.0)  
  distillation_aggressive   → Prioritize knowledge transfer (α=0.7, T=8.0)
  distillation_vgae_student → VGAE student from VGAE teacher

Fusion Training:
  fusion_standard           → Full training loop (500 episodes)
  fusion_lightweight        → Quick iteration (200 episodes)

USAGE EXAMPLES:
  # Use preset directly
  python train_with_hydra_zen.py --preset distillation_aggressive --dataset hcrl_sa
  
  # Or get from Python
  from src.config.training_presets import get_preset
  config = get_preset("distillation_balanced", dataset="set_01")
  
  # With custom overrides
  config = get_preset_with_overrides(
      "distillation_aggressive",
      dataset="hcrl_ch",
      max_epochs=200,
      learning_rate=0.0005
  )
"""

if __name__ == "__main__":
    list_presets()
    print("\n" + PRESET_DOCUMENTATION)
