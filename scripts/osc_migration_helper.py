#!/usr/bin/env python3
"""
OSC Migration Helper

Helps transition from Job Composer GUI to automated job management.
Creates organized job structure and provides migration commands.

Usage:
    python scripts/osc_migration_helper.py --setup
    python scripts/osc_migration_helper.py --create-examples
    python scripts/osc_migration_helper.py --migrate-existing
"""

import os
import sys
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class OSCMigrationHelper:
    """Help migrate from Job Composer GUI to automated management."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.osc_jobs_dir = self.project_root / "osc_jobs"
        
    def setup_directory_structure(self):
        """Create organized directory structure for OSC jobs."""
        
        logger.info("🔧 Setting up OSC job directory structure...")
        
        # Create main directories
        directories = [
            "osc_jobs",
            "osc_jobs/scripts",
            "osc_jobs/outputs", 
            "osc_jobs/logs",
            "osc_jobs/completed",
            "osc_jobs/templates"
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"  ✅ Created: {dir_name}/")
            
        # Create .gitignore for job outputs
        gitignore_content = """# OSC Job outputs
osc_jobs/outputs/*
osc_jobs/logs/*
osc_jobs/completed/*
*.out
*.err
slurm-*.out

# But keep the directories
!osc_jobs/outputs/.gitkeep
!osc_jobs/logs/.gitkeep
!osc_jobs/completed/.gitkeep
"""
        
        with open(self.project_root / "osc_jobs" / ".gitignore", 'w') as f:
            f.write(gitignore_content)
            
        # Create .gitkeep files
        for subdir in ["outputs", "logs", "completed"]:
            (self.project_root / "osc_jobs" / subdir / ".gitkeep").touch()
            
        logger.info("✅ Directory structure created!")
        
    def create_example_jobs(self):
        """Create example job scripts for different scenarios."""
        
        logger.info("📝 Creating example job scripts...")
        
        examples = {
            "individual_gat_example.sh": self._generate_individual_example("gat"),
            "individual_vgae_example.sh": self._generate_individual_example("vgae"), 
            "knowledge_distillation_example.sh": self._generate_kd_example(),
            "fusion_example.sh": self._generate_fusion_example(),
            "parameter_sweep_example.sh": self._generate_sweep_example()
        }
        
        templates_dir = self.osc_jobs_dir / "templates"
        
        for filename, content in examples.items():
            file_path = templates_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
            os.chmod(file_path, 0o755)  # Make executable
            logger.info(f"  ✅ Created: {filename}")
            
        logger.info("✅ Example scripts created!")
        
    def _generate_individual_example(self, model_type: str) -> str:
        """Generate individual training example."""
        
        mode = "normal" if model_type == "gat" else "autoencoder"
        time_limit = "2:00:00"
        
        return f'''#!/bin/bash
#SBATCH --job-name=individual-{model_type}-hcrl_sa
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS3209
#SBATCH --output=osc_jobs/outputs/individual_{model_type}_hcrl_sa_%j.out
#SBATCH --error=osc_jobs/outputs/individual_{model_type}_hcrl_sa_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frenken.2@osu.edu

echo "=== Individual {model_type.upper()} Training ==="
echo "Dataset: hcrl_sa"
echo "Started: $(date)"

# Environment setup
module load miniconda3/24.1.2-py310
source activate gnn-gpu
module load cuda/11.8.0

# Optimizations
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd /users/PAS2022/rf15/CAN-Graph

# UNIFIED TRAINING COMMAND
python train_with_hydra_zen.py \\
    --config-path conf \\
    --config-name hpc_optimized \\
    --model {model_type} \\
    --dataset hcrl_sa \\
    --training {mode}

echo "✅ Individual {model_type.upper()} training completed!"
'''
        
    def _generate_kd_example(self) -> str:
        """Generate knowledge distillation example."""
        
        return '''#!/bin/bash
#SBATCH --job-name=kd-hcrl_sa
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS3209
#SBATCH --output=osc_jobs/outputs/kd_hcrl_sa_%j.out
#SBATCH --error=osc_jobs/outputs/kd_hcrl_sa_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frenken.2@osu.edu

echo "=== Knowledge Distillation Training ==="
echo "Dataset: hcrl_sa"
echo "Started: $(date)"

# Environment setup
module load miniconda3/24.1.2-py310
source activate gnn-gpu
module load cuda/11.8.0

# Optimizations
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/PAS2022/rf15/CAN-Graph

# UNIFIED TRAINING COMMAND
python train_with_hydra_zen.py \\
    --config-path conf \\
    --config-name hpc_optimized \\
    --training knowledge_distillation \\
    --dataset hcrl_sa \\
    --teacher_path saved_models/best_teacher_model_hcrl_sa.pth \\
    --student_scale 0.5

echo "✅ Knowledge distillation completed!"
'''
        
    def _generate_fusion_example(self) -> str:
        """Generate fusion training example."""
        
        return '''#!/bin/bash
#SBATCH --job-name=fusion-hcrl_sa
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS3209
#SBATCH --output=osc_jobs/outputs/fusion_hcrl_sa_%j.out
#SBATCH --error=osc_jobs/outputs/fusion_hcrl_sa_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frenken.2@osu.edu

echo "=== Fusion Training ==="
echo "Dataset: hcrl_sa"
echo "Started: $(date)"

# Environment setup
module load miniconda3/24.1.2-py310
source activate gnn-gpu
module load cuda/11.8.0

# Optimizations
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd /users/PAS2022/rf15/CAN-Graph

# UNIFIED TRAINING COMMAND
python train_with_hydra_zen.py \\
    --config-path conf \\
    --config-name hpc_optimized \\
    --training fusion \\
    --dataset hcrl_sa \\
    --autoencoder_path saved_models/autoencoder_hcrl_sa.pth \\
    --classifier_path saved_models/classifier_hcrl_sa.pth

echo "✅ Fusion training completed!"
'''
        
    def _generate_sweep_example(self) -> str:
        """Generate parameter sweep example."""
        
        return '''#!/bin/bash
#SBATCH --job-name=sweep-fusion-lr001
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --account=PAS3209
#SBATCH --output=osc_jobs/outputs/sweep_fusion_lr001_%j.out
#SBATCH --error=osc_jobs/outputs/sweep_fusion_lr001_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frenken.2@osu.edu

echo "=== Parameter Sweep: Fusion LR=0.001 ==="
echo "Dataset: hcrl_sa"
echo "Started: $(date)"

# Environment setup
module load miniconda3/24.1.2-py310
source activate gnn-gpu
module load cuda/11.8.0

# Optimizations
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

cd /users/PAS2022/rf15/CAN-Graph

# PARAMETER SWEEP COMMAND
python train_with_hydra_zen.py \\
    --config-path conf \\
    --config-name hpc_optimized \\
    --training fusion \\
    --dataset hcrl_sa \\
    --autoencoder_path saved_models/autoencoder_hcrl_sa.pth \\
    --classifier_path saved_models/classifier_hcrl_sa.pth \\
    training.fusion_agent_config.fusion_lr=0.001 \\
    training.fusion_agent_config.fusion_batch_size=512

echo "✅ Parameter sweep job completed!"
'''
        
    def create_migration_guide(self):
        """Create migration guide from Job Composer to automated management."""
        
        guide_content = '''# OSC Migration Guide: From Job Composer to Automated Management

## 🔄 Migration Overview

**Before:** 20+ Job Composer folders with manual management
**After:** Automated job submission with organized outputs

## 🚀 Quick Start

### 1. Setup (One-time)
```bash
# Run this once to set up directory structure
python scripts/osc_migration_helper.py --setup
python scripts/osc_migration_helper.py --create-examples
```

### 2. Basic Usage
```bash
# Instead of Job Composer, use command line:

# Submit individual jobs for all datasets
python scripts/osc_job_manager.py --submit-individual

# Submit specific dataset
python scripts/osc_job_manager.py --submit-fusion --datasets hcrl_sa

# Submit complete pipeline
python scripts/osc_job_manager.py --submit-pipeline --datasets set_04

# Monitor jobs
python scripts/osc_job_manager.py --monitor
```

## 📊 Comparison

| Task | Old Way (Job Composer) | New Way (Automated) |
|------|------------------------|---------------------|
| **Submit 6 datasets** | Create 6 folders manually | `--submit-individual` (1 command) |
| **Parameter sweep** | Create 20+ folders | `--parameter-sweep` (1 command) |
| **Monitor jobs** | Check GUI repeatedly | `--monitor` (instant status) |
| **Output organization** | Scattered across folders | Organized in `osc_jobs/` |
| **Job dependencies** | Manual timing | Automatic `--submit-pipeline` |

## 🎯 Migration Steps

### Step 1: Test New Approach
```bash
# Test with one dataset first
python scripts/osc_job_manager.py --submit-fusion --datasets hcrl_sa
```

### Step 2: Compare Results
- Old output: Scattered in Job Composer folders
- New output: Organized in `osc_jobs/outputs/`

### Step 3: Migrate Gradually
```bash
# Start with individual jobs
python scripts/osc_job_manager.py --submit-individual --datasets hcrl_sa,set_04

# Then try parameter sweeps
python scripts/osc_job_manager.py --parameter-sweep --training fusion --datasets hcrl_sa
```

### Step 4: Full Migration
```bash
# Submit all your typical jobs at once
python scripts/osc_job_manager.py --submit-pipeline --datasets all
```

## 📁 Directory Structure

```
CAN-Graph/
├── osc_jobs/                    # New organized structure
│   ├── scripts/                 # Generated SLURM scripts
│   ├── outputs/                 # Job outputs (replaces Job Composer folders)
│   ├── logs/                    # Centralized logs
│   ├── completed/               # Archive completed jobs
│   └── templates/               # Example scripts
├── train_with_hydra_zen.py     # Unified training script
└── conf/hpc_optimized.yaml     # Single config file
```

## 🔧 Customization

### Update Your Settings
Edit `scripts/osc_job_manager.py`:
```python
self.osc_settings = {
    "account": "PAS3209",                    # Your account
    "email": "frenken.2@osu.edu",          # Your email  
    "project_path": "/users/PAS2022/rf15/CAN-Graph",  # Your path
    "conda_env": "gnn-gpu",                 # Your environment
}
```

### Add Custom Parameters
```bash
# Override any training parameter
python scripts/osc_job_manager.py --submit-fusion --datasets hcrl_sa \\
    training.max_epochs=200 \\
    training.learning_rate=0.0005
```

## ✅ Benefits

1. **No more manual folder creation** - Everything automated
2. **Consistent job structure** - All use same optimized template
3. **Organized outputs** - No more scattered files
4. **Easy parameter sweeps** - Generate 100s of jobs with one command
5. **Job dependencies** - Automatic pipeline management
6. **Better monitoring** - Real-time status checking
7. **Version control** - All job scripts saved and tracked

## 🎯 Next Steps

1. **Test**: Submit one job using new system
2. **Compare**: Verify outputs match your expectations  
3. **Migrate**: Gradually replace Job Composer workflows
4. **Optimize**: Add custom parameters or job types as needed

Your 20+ Job Composer folders become **3-4 simple commands**! 🎉
'''
        
        guide_path = self.osc_jobs_dir / "MIGRATION_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
            
        logger.info(f"✅ Migration guide created: {guide_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="OSC Migration Helper")
    parser.add_argument("--setup", action="store_true", help="Set up directory structure")
    parser.add_argument("--create-examples", action="store_true", help="Create example job scripts")
    parser.add_argument("--migrate-existing", action="store_true", help="Help migrate existing jobs")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    
    args = parser.parse_args()
    
    helper = OSCMigrationHelper()
    
    if args.all or args.setup:
        helper.setup_directory_structure()
        
    if args.all or args.create_examples:
        helper.create_example_jobs()
        
    if args.all or args.migrate_existing:
        helper.create_migration_guide()
        
    if args.all:
        logger.info("\\n🎉 OSC Migration Setup Complete!")
        logger.info("\\n📋 Next steps:")
        logger.info("1. Review: cat osc_jobs/MIGRATION_GUIDE.md")
        logger.info("2. Test: python scripts/osc_job_manager.py --submit-fusion --datasets hcrl_sa")
        logger.info("3. Monitor: python scripts/osc_job_manager.py --monitor")
        logger.info("\\n✅ Ready to replace Job Composer workflows!")
        
    if not any([args.setup, args.create_examples, args.migrate_existing, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()