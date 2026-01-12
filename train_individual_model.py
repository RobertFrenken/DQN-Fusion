"""
Individual Model Training Script

UPDATED: This script now serves as a compatibility wrapper around the new PyTorch Lightning
Fabric-based training system. For full features and better performance, use train_fabric_models.py directly.

Train specific models one at a time with full resource management and error isolation.
This is safer than batch training as it prevents one model's failure from affecting others.

Usage:
    python train_individual_model.py --dataset hcrl_ch --model teacher
    python train_individual_model.py --dataset set_01 --model student
    python train_individual_model.py --dataset hcrl_sa --model fusion
    python train_individual_model.py --list-available  # Show available options
    python train_individual_model.py --use-fabric      # Use new Fabric training (recommended)

New Fabric Usage (Recommended):
    python train_fabric_models.py --model gat --dataset hcrl_ch --type teacher
    python train_fabric_models.py --model vgae --dataset set_01 --type student --teacher saved_models/best_vgae_model_hcrl_sa.pth
"""

import os
import sys
import argparse
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from omegaconf import DictConfig, OmegaConf

class IndividualModelTrainer:
    """Handles training of individual models with comprehensive resource management."""
    
    def __init__(self, use_fabric: bool = False):
        self.use_fabric = use_fabric
        self.available_datasets = ['hcrl_ch', 'hcrl_sa', 'set_01', 'set_02', 'set_03', 'set_04']
        self.available_models = ['teacher', 'student', 'fusion']
        
        
        # Initialize orchestrator for resource management (legacy mode)
        if not self.use_fabric:
            try:
                from src.training.resource_aware_orchestrator import ResourceAwareOrchestrator, TrainingPhase
                from src.utils.gpu_utils import setup_gpu_optimization, cleanup_memory, log_memory_usage
                self.orchestrator = ResourceAwareOrchestrator()
                self.TrainingPhase = TrainingPhase  # Store for later use
            except ImportError:
                print("⚠️  Legacy training modules not available. Please use Fabric training.")
                print("Run: python train_fabric_models.py --help")
                self.orchestrator = None
        
        # Results tracking
        self.results_dir = Path("outputs/individual_training_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def train_with_fabric(self, dataset_key: str, model_type: str) -> Dict[str, Any]:
        """
        Train model using the new Fabric-based training system.
        
        Args:
            dataset_key: Dataset identifier
            model_type: Model type ('teacher', 'student', 'fusion')
            
        Returns:
            Training results
        """
        print(f"🚀 Using PyTorch Lightning Fabric training for {model_type} on {dataset_key}")
        
        # Map legacy model types to new system
        if model_type == 'fusion':
            print("⚠️  Fusion training not yet implemented in Fabric system.")
            print("Use legacy training or implement DQN fusion with Fabric.")
            return {"status": "not_implemented", "message": "Fusion training pending"}
        
        # Determine which models to train (GAT and VGAE for teacher/student)
        models_to_train = ['gat', 'vgae']
        results = {}
        
        for model in models_to_train:
            print(f"\n📊 Training {model.upper()} {model_type} on {dataset_key}")
            
            try:
                # Construct command
                cmd = [
                    sys.executable, "train_fabric_models.py",
                    "--model", model,
                    "--dataset", dataset_key,
                    "--type", model_type
                ]
                
                # Add teacher model path for student training
                if model_type == 'student':
                    if model == 'gat':
                        teacher_path = f"saved_models/best_teacher_model_{dataset_key}.pth"
                    else:  # vgae
                        teacher_path = f"saved_models/autoencoder_{dataset_key}.pth"
                    
                    if Path(teacher_path).exists():
                        cmd.extend(["--teacher", teacher_path])
                    else:
                        print(f"⚠️  Teacher model not found: {teacher_path}")
                        print("Training student without knowledge distillation")
                
                # Run Fabric training
                print(f"Executing: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
                
                if result.returncode == 0:
                    print(f"✅ {model.upper()} {model_type} training completed successfully")
                    results[model] = {"status": "success", "stdout": result.stdout}
                else:
                    print(f"❌ {model.upper()} {model_type} training failed")
                    print(f"Error: {result.stderr}")
                    results[model] = {"status": "failed", "stderr": result.stderr}
                
            except Exception as e:
                print(f"❌ Failed to execute Fabric training for {model}: {e}")
                results[model] = {"status": "error", "error": str(e)}
        
        return results
    
    def list_available_options(self):
        """Display all available datasets and model types."""
        print("🎯 Available Training Options:")
        print(f"\n📊 Datasets ({len(self.available_datasets)}):")
        for dataset in self.available_datasets:
            model_path = Path(f"saved_models/best_teacher_model_{dataset}.pth")
            status = "✅ Exists" if model_path.exists() else "❌ Missing"
            print(f"  • {dataset:<12} {status}")
        
        print(f"\n🧠 Model Types ({len(self.available_models)}):")
        print("  • teacher    - Base model (autoencoder + classifier)")
        print("  • student    - Knowledge distillation model")
        print("  • fusion     - DQN fusion agent")
        
        # Check system resources
        profile = self.orchestrator.resource_profile
        print(f"\n💻 System Resources:")
        print(f"  • GPU Memory: {profile.gpu_memory_available_gb:.1f}GB available / {profile.gpu_memory_total_gb:.1f}GB total")
        print(f"  • Optimal Batch Size: {profile.optimal_batch_size}")
        print(f"  • Training Capacity: {profile.training_capacity:.1%}")
    
    def get_training_phase(self, dataset: str, model_type: str) -> TrainingPhase:
        """Create a training phase for the specified dataset and model type."""
        if dataset not in self.available_datasets:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {self.available_datasets}")
        if model_type not in self.available_models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {self.available_models}")
        
        profile = self.orchestrator.resource_profile
        
        # Define phase configurations
        if model_type == "teacher":
            return TrainingPhase(
                phase_name=f"teacher_{dataset}",
                dataset_key=dataset,
                model_type="teacher",
                script_path="src/training/AD_KD_GPU.py",
                config_override={
                    'root_folder': dataset,
                    'epochs': 10,
                    'batch_size': profile.optimal_batch_size,
                    'device': str(self.orchestrator.device),
                    'mode': 'teacher_only'
                },
                dependencies=[],
                estimated_time_minutes=45,
                memory_requirement_gb=8.0,
                priority=3
            )
        
        elif model_type == "student":
            # Check if teacher model exists
            teacher_path = Path(f"saved_models/best_teacher_model_{dataset}.pth")
            if not teacher_path.exists():
                raise FileNotFoundError(f"Teacher model required: {teacher_path}")
            
            return TrainingPhase(
                phase_name=f"student_{dataset}",
                dataset_key=dataset,
                model_type="student",
                script_path="src/training/AD_KD_GPU.py",
                config_override={
                    'root_folder': dataset,
                    'epochs': 8,
                    'batch_size': profile.optimal_batch_size,
                    'device': str(self.orchestrator.device),
                    'mode': 'student_only',
                    'teacher_model_path': str(teacher_path)
                },
                dependencies=[f"teacher_{dataset}"],
                estimated_time_minutes=35,
                memory_requirement_gb=10.0,
                priority=2
            )
        
        elif model_type == "fusion":
            # Check if both teacher and student models exist
            teacher_path = Path(f"saved_models/best_teacher_model_{dataset}.pth")
            student_path = Path(f"saved_models/final_student_model_{dataset}.pth")
            
            missing = []
            if not teacher_path.exists():
                missing.append(f"Teacher: {teacher_path}")
            if not student_path.exists():
                missing.append(f"Student: {student_path}")
            
            if missing:
                raise FileNotFoundError(f"Required models missing: {', '.join(missing)}")
            
            return TrainingPhase(
                phase_name=f"fusion_{dataset}",
                dataset_key=dataset,
                model_type="fusion",
                script_path="src/training/fusion_training.py",
                config_override={
                    'root_folder': dataset,
                    'episodes': 500,
                    'batch_size': min(profile.optimal_batch_size, 512),  # Fusion needs smaller batches
                    'device': str(self.orchestrator.device),
                    'teacher_model_path': str(teacher_path),
                    'student_model_path': str(student_path)
                },
                dependencies=[f"teacher_{dataset}", f"student_{dataset}"],
                estimated_time_minutes=60,
                memory_requirement_gb=12.0,
                priority=1
            )
    
    def check_prerequisites(self, dataset: str, model_type: str) -> Dict[str, Any]:
        """Check if all prerequisites are met for training."""
        checks = {
            'dataset_exists': False,
            'gpu_available': False,
            'memory_sufficient': False,
            'dependencies_met': True,
            'messages': []
        }
        
        # Check dataset
        dataset_paths = {
            'hcrl_ch': 'datasets/can-train-and-test-v1.5/hcrl-ch',
            'hcrl_sa': 'datasets/can-train-and-test-v1.5/hcrl-sa', 
            'set_01': 'datasets/can-train-and-test-v1.5/set_01',
            'set_02': 'datasets/can-train-and-test-v1.5/set_02',
            'set_03': 'datasets/can-train-and-test-v1.5/set_03',
            'set_04': 'datasets/can-train-and-test-v1.5/set_04'
        }
        
        dataset_path = Path(dataset_paths.get(dataset, ''))
        if dataset_path.exists():
            checks['dataset_exists'] = True
            checks['messages'].append(f"✅ Dataset found: {dataset_path}")
        else:
            checks['messages'].append(f"❌ Dataset missing: {dataset_path}")
        
        # Check GPU
        if torch.cuda.is_available():
            checks['gpu_available'] = True
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            checks['messages'].append(f"✅ GPU available: {gpu_name} ({memory_gb:.1f}GB)")
        else:
            checks['messages'].append("❌ No CUDA GPU available")
        
        # Check memory requirements
        phase = self.get_training_phase(dataset, model_type)
        profile = self.orchestrator.resource_profile
        if profile.gpu_memory_available_gb >= phase.memory_requirement_gb:
            checks['memory_sufficient'] = True
            checks['messages'].append(f"✅ Sufficient GPU memory: {profile.gpu_memory_available_gb:.1f}GB >= {phase.memory_requirement_gb:.1f}GB required")
        else:
            checks['messages'].append(f"❌ Insufficient GPU memory: {profile.gpu_memory_available_gb:.1f}GB < {phase.memory_requirement_gb:.1f}GB required")
        
        # Check dependencies
        if model_type == "student":
            teacher_path = Path(f"saved_models/best_teacher_model_{dataset}.pth")
            if not teacher_path.exists():
                checks['dependencies_met'] = False
                checks['messages'].append(f"❌ Missing dependency: {teacher_path}")
            else:
                checks['messages'].append(f"✅ Teacher model found: {teacher_path}")
        
        elif model_type == "fusion":
            teacher_path = Path(f"saved_models/best_teacher_model_{dataset}.pth")
            student_path = Path(f"saved_models/final_student_model_{dataset}.pth")
            
            if not teacher_path.exists():
                checks['dependencies_met'] = False
                checks['messages'].append(f"❌ Missing dependency: {teacher_path}")
            else:
                checks['messages'].append(f"✅ Teacher model found: {teacher_path}")
                
            if not student_path.exists():
                checks['dependencies_met'] = False
                checks['messages'].append(f"❌ Missing dependency: {student_path}")
            else:
                checks['messages'].append(f"✅ Student model found: {student_path}")
        
        return checks
    
    def train_model(self, dataset: str, model_type: str, 
                   force_retrain: bool = False, 
                   dry_run: bool = False) -> Dict[str, Any]:
        """Train a specific model with full resource management."""
        
        print(f"\n🚀 Individual Model Training")
        print(f"{'='*50}")
        print(f"Dataset: {dataset}")
        print(f"Model Type: {model_type}")
        print(f"{'='*50}")
        
        # Check prerequisites
        print("🔍 Checking prerequisites...")
        checks = self.check_prerequisites(dataset, model_type)
        
        for message in checks['messages']:
            print(f"  {message}")
        
        # Check if all prerequisites are met
        all_good = all([
            checks['dataset_exists'],
            checks['gpu_available'], 
            checks['memory_sufficient'],
            checks['dependencies_met']
        ])
        
        if not all_good:
            print(f"\n❌ Prerequisites not met. Cannot proceed with training.")
            return {'status': 'failed', 'reason': 'prerequisites_not_met', 'checks': checks}
        
        # Check if model already exists
        model_path_map = {
            'teacher': f"saved_models/best_teacher_model_{dataset}.pth",
            'student': f"saved_models/final_student_model_{dataset}.pth",
            'fusion': f"saved_models/fusion_agent_{dataset}.pth"
        }
        
        model_path = Path(model_path_map[model_type])
        
        if model_path.exists() and not force_retrain:
            print(f"\n⚠️  Model already exists: {model_path}")
            print("Use --force-retrain to overwrite existing model")
            return {'status': 'skipped', 'reason': 'model_exists', 'model_path': str(model_path)}
        
        if dry_run:
            print(f"\n🧪 DRY RUN - Would train {model_type} model for {dataset}")
            print(f"  Model path: {model_path}")
            print(f"  Estimated time: {self.get_training_phase(dataset, model_type).estimated_time_minutes} minutes")
            return {'status': 'dry_run', 'would_train': True}
        
        # Execute training
        print(f"\n🏋️ Starting training...")
        phase = self.get_training_phase(dataset, model_type)
        
        start_time = time.time()
        try:
            result = self.orchestrator._execute_training_phase(phase)
            elapsed_time = time.time() - start_time
            
            if result['status'] == 'completed':
                print(f"\n✅ Training completed successfully!")
                print(f"   Time: {elapsed_time/60:.1f} minutes")
                print(f"   Model saved to: {model_path}")
                
                # Save detailed results
                result_file = self.results_dir / f"{dataset}_{model_type}_results.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"   Results: {result_file}")
                
            else:
                print(f"\n❌ Training failed!")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                print(f"   Time: {elapsed_time/60:.1f} minutes")
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"\n💥 Training crashed!")
            print(f"   Error: {str(e)}")
            print(f"   Time: {elapsed_time/60:.1f} minutes")
            
            return {
                'status': 'crashed',
                'error': str(e),
                'elapsed_time': elapsed_time,
                'dataset': dataset,
                'model_type': model_type
            }
        
        finally:
            # Always cleanup
            cleanup_memory()

def main():
    parser = argparse.ArgumentParser(
        description="Train individual models with resource management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available options
    python train_individual_model.py --list-available
    
    # Train using new Fabric system (recommended)
    python train_individual_model.py --dataset hcrl_ch --model teacher --use-fabric
    python train_individual_model.py --dataset hcrl_ch --model student --use-fabric
    
    # Train using legacy system
    python train_individual_model.py --dataset hcrl_ch --model teacher
    python train_individual_model.py --dataset hcrl_ch --model student
    
    # Force retrain existing model (legacy only)
    python train_individual_model.py --dataset hcrl_ch --model teacher --force-retrain
    
    # Check what would be done without training (legacy only)
    python train_individual_model.py --dataset hcrl_ch --model teacher --dry-run
    
    Note: --use-fabric is recommended for better performance and HPC compatibility
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        choices=['hcrl_ch', 'hcrl_sa', 'set_01', 'set_02', 'set_03', 'set_04'],
        help="Dataset to train on"
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=['teacher', 'student', 'fusion'],
        help="Model type to train"
    )
    
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain even if model already exists"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Show what would be done without actually training"
    )
    
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="List all available datasets and model types"
    )
    
    parser.add_argument(
        "--use-fabric",
        action="store_true", 
        help="Use PyTorch Lightning Fabric training (recommended for better performance)"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    try:
        trainer = IndividualModelTrainer(use_fabric=args.use_fabric)
    except Exception as e:
        print(f"❌ Failed to initialize trainer: {e}")
        return 1
    
    # Handle list command
    if args.list_available:
        trainer.list_available_options()
        return 0
    
    # Validate required arguments
    if not args.dataset or not args.model:
        print("❌ Both --dataset and --model are required (or use --list-available)")
        parser.print_help()
        return 1
    
    # Run training
    if args.use_fabric:
        print("🚀 Using PyTorch Lightning Fabric training system")
        result = trainer.train_with_fabric(args.dataset, args.model)
        success = all(r.get('status') == 'success' for r in result.values() if isinstance(r, dict))
        return 0 if success else 1
    else:
        print("⚠️  Using legacy training system")
        result = trainer.train_model(
            dataset=args.dataset,
            model_type=args.model,
            force_retrain=args.force_retrain,
            dry_run=args.dry_run
        )
        
        # Return appropriate exit code
        if result['status'] in ['completed', 'skipped', 'dry_run']:
            return 0
        else:
            return 1

if __name__ == "__main__":
    sys.exit(main())