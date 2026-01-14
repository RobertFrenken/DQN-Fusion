#!/usr/bin/env python3
"""
HPC Training Setup and Optimization Script

Prepares CAN-Graph training for high-performance computing environments.
Includes configuration for SLURM, multi-GPU, and resource optimization.

Usage:
    python scripts/hpc_training_setup.py --check-setup
    python scripts/hpc_training_setup.py --generate-slurm --job-type fusion --dataset hcrl_sa
    python scripts/hpc_training_setup.py --optimize-config --target gpu-nodes
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HPCSetupOptimizer:
    """HPC training setup and optimization."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.slurm_dir = self.project_root / "scripts" / "slurm"
        self.slurm_dir.mkdir(exist_ok=True)
        
    def check_hpc_readiness(self):
        """Check if setup is ready for HPC training."""
        logger.info("🔍 Checking HPC Readiness...")
        
        checks = {
            "Lightning Integration": self._check_lightning_integration(),
            "Hydra-Zen Configs": self._check_hydra_zen_configs(),
            "Model Checkpoints": self._check_saved_models(),
            "SLURM Scripts": self._check_slurm_scripts(),
            "Multi-GPU Support": self._check_multi_gpu_support(),
            "Memory Optimization": self._check_memory_optimization()
        }
        
        passed = sum(checks.values())
        total = len(checks)
        
        logger.info(f"\n📊 HPC Readiness Score: {passed}/{total}")
        
        for check, status in checks.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"  {status_icon} {check}")
            
        if passed == total:
            logger.info("\n🚀 Ready for HPC training!")
        else:
            logger.info(f"\n⚠️  {total - passed} improvements needed before HPC deployment")
            
        return passed == total
    
    def _check_lightning_integration(self) -> bool:
        """Check Lightning integration."""
        train_files = [
            "train_with_hydra_zen.py",
            "train_fusion_lightning.py", 
            "train_knowledge_distillation.py",
            "src/training/fusion_lightning.py"
        ]
        return all((self.project_root / f).exists() for f in train_files)
    
    def _check_hydra_zen_configs(self) -> bool:
        """Check Hydra-Zen configuration files."""
        config_files = [
            "src/config/hydra_zen_configs.py",
            "src/config/fusion_config.py"
        ]
        return all((self.project_root / f).exists() for f in config_files)
    
    def _check_saved_models(self) -> bool:
        """Check if required pre-trained models exist."""
        model_patterns = [
            "saved_models/*teacher*.pth",
            "saved_models/*fusion*.pth", 
            "saved_models/*autoencoder*.pth"
        ]
        saved_models = list(self.project_root.glob("saved_models/*.pth"))
        return len(saved_models) >= 10  # Should have multiple models
    
    def _check_slurm_scripts(self) -> bool:
        """Check SLURM script availability."""
        return (self.project_root / "scripts" / "fusion_slurm.sh").exists()
    
    def _check_multi_gpu_support(self) -> bool:
        """Check multi-GPU configuration."""
        # Check if training scripts use Lightning's auto device detection
        with open(self.project_root / "train_models.py") as f:
            content = f.read()
            return "accelerator='auto'" in content and "devices='auto'" in content
    
    def _check_memory_optimization(self) -> bool:
        """Check memory optimization features."""
        fusion_file = self.project_root / "src/training/fusion_lightning.py"
        if not fusion_file.exists():
            return False
            
        with open(fusion_file) as f:
            content = f.read()
            return "teacher_cache" in content or "memory_optimization" in content

    def generate_slurm_scripts(self, job_type: str, dataset: str):
        """Generate optimized SLURM scripts for different training types."""
        
        scripts = {
            "individual": self._generate_individual_slurm(dataset),
            "distillation": self._generate_distillation_slurm(dataset),
            "fusion": self._generate_fusion_slurm(dataset),
            "all": self._generate_pipeline_slurm(dataset)
        }
        
        if job_type in scripts:
            script_content = scripts[job_type]
            script_path = self.slurm_dir / f"{job_type}_{dataset}.sh"
            
            with open(script_path, 'w') as f:
                f.write(script_content)
                
            logger.info(f"✅ Generated SLURM script: {script_path}")
            logger.info(f"📋 Submit with: sbatch {script_path}")
            
        elif job_type == "all":
            for script_name, script_content in scripts.items():
                if script_name == "all":
                    continue
                    
                script_path = self.slurm_dir / f"{script_name}_{dataset}.sh"
                with open(script_path, 'w') as f:
                    f.write(script_content)
                    
                logger.info(f"✅ Generated {script_name} SLURM script: {script_path}")
        else:
            logger.error(f"Unknown job type: {job_type}")
    
    def _generate_individual_slurm(self, dataset: str) -> str:
        """Generate SLURM script for individual model training."""
        return f'''#!/bin/bash
#SBATCH --job-name=can-individual-{dataset}
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --output=logs/individual_{dataset}_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# Environment setup
module load miniconda3/24.1.2-py310
source activate gnn-gpu
module load cuda/11.8.0

# Optimizations
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd $SLURM_SUBMIT_DIR

echo "=== Individual Training: {dataset} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Train GAT (supervised)
python train_with_hydra_zen.py \\
    --config-path conf \
    --config-name hpc_optimized \
    --model gat \
    --dataset {dataset} \
    --training normal

# Train VGAE (unsupervised)  
python train_with_hydra_zen.py \
    --config-path conf \
    --config-name hpc_optimized \
    --model vgae \
    --dataset {dataset} \
    --training autoencoder

echo "=== Individual Training Complete ==="
'''

    def _generate_distillation_slurm(self, dataset: str) -> str:
        """Generate SLURM script for knowledge distillation."""
        return f'''#!/bin/bash
#SBATCH --job-name=can-distill-{dataset}
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --output=logs/distillation_{dataset}_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# Environment setup
module load miniconda3/24.1.2-py310
source activate gnn-gpu
module load cuda/11.8.0

# Memory optimizations for teacher-student training
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $SLURM_SUBMIT_DIR

echo "=== Knowledge Distillation: {dataset} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Teacher Model: saved_models/best_teacher_model_{dataset}.pth"

# Knowledge Distillation with optimizations
python train_with_hydra_zen.py \\
    --config-path conf \
    --config-name hpc_optimized \
    --training knowledge_distillation \
    --dataset {dataset} \
    --teacher_path saved_models/best_teacher_model_{dataset}.pth \
    --student_scale 0.5

echo "=== Knowledge Distillation Complete ==="
'''

    def _generate_fusion_slurm(self, dataset: str) -> str:
        """Generate SLURM script for fusion training."""
        return f'''#!/bin/bash
#SBATCH --job-name=can-fusion-{dataset}
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --output=logs/fusion_{dataset}_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# Environment setup
module load miniconda3/24.1.2-py310
source activate gnn-gpu
module load cuda/11.8.0

# GPU optimizations for DQN fusion
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd $SLURM_SUBMIT_DIR

echo "=== Fusion Training: {dataset} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Required Models:"
echo "  - Autoencoder: saved_models/autoencoder_{dataset}.pth"
echo "  - Classifier: saved_models/classifier_{dataset}.pth"

# UNIFIED: Fusion Training with single train_with_hydra_zen.py
python train_with_hydra_zen.py \\
    --config-path conf \\
    --config-name hpc_optimized \\
    --training fusion \\
    --dataset {dataset} \\
    --autoencoder_path saved_models/autoencoder_{dataset}.pth \\
    --classifier_path saved_models/classifier_{dataset}.pth

echo "=== Fusion Training Complete ==="
'''

    def _generate_pipeline_slurm(self, dataset: str) -> str:
        """Generate full pipeline SLURM script."""
        return f'''#!/bin/bash
#SBATCH --job-name=can-pipeline-{dataset}
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --output=logs/pipeline_{dataset}_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# Environment setup
module load miniconda3/24.1.2-py310
source activate gnn-gpu
module load cuda/11.8.0

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $SLURM_SUBMIT_DIR

echo "=== Full CAN-Graph Pipeline: {dataset} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Estimated time: 10-12 hours"

# Step 1: Train teacher models
echo "Step 1/4: Training teacher models..."
python train_with_hydra_zen.py --config-path conf --config-name hpc_optimized --model gat --dataset {dataset} --training normal
python train_with_hydra_zen.py --config-path conf --config-name hpc_optimized --model vgae --dataset {dataset} --training autoencoder

# Step 2: Knowledge distillation  
echo "Step 2/4: Knowledge distillation..."
python train_with_hydra_zen.py \
    --config-path conf \
    --config-name hpc_optimized \
    --training knowledge_distillation \
    --dataset {dataset} \
    --teacher_path saved_models/best_teacher_model_{dataset}.pth \
    --student_scale 0.5

# Step 3: Fusion training
echo "Step 3/4: Fusion training..."
python train_with_hydra_zen.py \
    --config-path conf \
    --config-name hpc_optimized \
    --training fusion \
    --dataset {dataset} \
    --autoencoder_path saved_models/autoencoder_{dataset}.pth \
    --classifier_path saved_models/classifier_{dataset}.pth

# Step 4: Evaluation
echo "Step 4/4: Model evaluation..."
python scripts/evaluate_models.py --dataset {dataset} --all-models

echo "=== Full Pipeline Complete ==="
'''

    def optimize_configs_for_hpc(self, target: str = "gpu-nodes"):
        """Optimize configuration files for HPC environments."""
        
        logger.info(f"🔧 Optimizing configs for: {target}")
        
        # HPC-optimized configurations
        hpc_optimizations = {
            "training": {
                "precision": "16-mixed",  # Mixed precision for A100s
                "optimize_batch_size": True,
                "batch_size_mode": "power",
                "accumulate_grad_batches": 4,
                "gradient_clip_val": 1.0,
                "find_unused_parameters": False,  # DDP optimization
                "sync_batchnorm": True  # Multi-GPU training
            },
            "hardware": {
                "accelerator": "auto",
                "devices": "auto", 
                "strategy": "ddp_find_unused_parameters_false",
                "num_workers": 8,
                "pin_memory": True,
                "persistent_workers": True
            },
            "logging": {
                "log_every_n_steps": 100,  # Reduce I/O
                "save_top_k": 2,  # Reduce storage
                "enable_progress_bar": False  # Reduce output for SLURM
            }
        }
        
        # Save optimized config
        config_path = self.project_root / "conf" / "hpc_optimized.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(hpc_optimizations, f, default_flow_style=False)
            
        logger.info(f"✅ Saved HPC-optimized config: {config_path}")
        
        # Generate usage examples
        examples_path = self.project_root / "scripts" / "hpc_usage_examples.sh"
        examples_content = f'''#!/bin/bash
# HPC Training Usage Examples

# Use optimized config
export HYDRA_FULL_ERROR=1

# Individual training with HPC optimizations
python train_with_hydra_zen.py \\
    --config-path conf \\
    --config-name hpc_optimized \\
    --model gat \\
    --dataset hcrl_sa

# Knowledge distillation with HPC optimizations  
python train_with_hydra_zen.py \\
    --config-path conf \\
    --config-name hpc_optimized \\
    --training knowledge_distillation \\
    --teacher_path saved_models/best_teacher_model_hcrl_sa.pth

# Fusion training with HPC optimizations
python train_fusion_lightning.py \\
    --config conf/hpc_optimized.yaml \\
    --dataset hcrl_sa \\
    --precision 16-mixed

# Multi-GPU distributed training
torchrun --nproc_per_node=4 train_with_hydra_zen.py \\
    --config-name hpc_optimized \\
    --model gat \\
    --dataset hcrl_sa
'''
        
        with open(examples_path, 'w') as f:
            f.write(examples_content)
            
        os.chmod(examples_path, 0o755)  # Make executable
        logger.info(f"✅ Generated usage examples: {examples_path}")

def main():
    parser = argparse.ArgumentParser(description="HPC Training Setup for CAN-Graph")
    parser.add_argument("--check-setup", action="store_true", help="Check HPC readiness")
    parser.add_argument("--generate-slurm", action="store_true", help="Generate SLURM scripts")
    parser.add_argument("--job-type", choices=["individual", "distillation", "fusion", "all"], 
                       default="all", help="Type of SLURM job to generate")
    parser.add_argument("--dataset", default="hcrl_sa", help="Dataset to use")
    parser.add_argument("--optimize-config", action="store_true", help="Optimize configs for HPC")
    parser.add_argument("--target", default="gpu-nodes", help="Target HPC environment")
    
    args = parser.parse_args()
    
    optimizer = HPCSetupOptimizer()
    
    if args.check_setup:
        ready = optimizer.check_hpc_readiness()
        sys.exit(0 if ready else 1)
        
    if args.generate_slurm:
        optimizer.generate_slurm_scripts(args.job_type, args.dataset)
        
    if args.optimize_config:
        optimizer.optimize_configs_for_hpc(args.target)
        
    if not any([args.check_setup, args.generate_slurm, args.optimize_config]):
        parser.print_help()

if __name__ == "__main__":
    main()