#!/usr/bin/env python3
"""
OSC Job Manager for CAN-Graph Training

Automates SLURM job submission on Ohio Supercomputer Center with:
- Parameterized job generation
- Batch job submission  
- Organized output management
- Job status monitoring
- Easy parameter sweeps

Usage:
    python scripts/osc_job_manager.py --submit-individual --datasets hcrl_sa,set_04
    python scripts/osc_job_manager.py --submit-fusion --all-datasets
    python scripts/osc_job_manager.py --parameter-sweep --training fusion
    python scripts/osc_job_manager.py --monitor-jobs
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OSCJobManager:
    """Manage SLURM jobs on Ohio Supercomputer Center."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.jobs_dir = self.project_root / "osc_jobs"
        self.jobs_dir.mkdir(exist_ok=True)
        
        # OSC-specific settings (customize for your account)
        self.osc_settings = {
            "account": "PAS3209",  # Your account
            "email": "frenken.2@osu.edu",  # Your email
            "project_path": "/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT",  # Your project path
            "conda_env": "gnn-gpu",  # Your conda environment
        }
        
        # Available datasets
        self.datasets = ["hcrl_sa", "hcrl_ch", "set_01", "set_02", "set_03", "set_04"]
        
        # Complex datasets requiring longer training
        self.complex_datasets = ["set_01", "set_02", "set_03", "set_04"]
        
        # Training configurations
        self.training_configs = {
            "individual_gat": {
                "time": "2:00:00",
                "time_complex": "8:00:00",  # Longer for complex datasets
                "mem": "32G", 
                "mem_complex": "64G",  # More memory for complex datasets
                "cpus": 8,
                "gpus": 1,
                "mode": "normal",
                "model": "gat"
            },
            "individual_vgae": {
                "time": "2:00:00", 
                "time_complex": "8:00:00",  # Longer for complex datasets
                "mem": "32G",
                "mem_complex": "64G",  # More memory for complex datasets
                "cpus": 8,
                "gpus": 1,
                "mode": "autoencoder", 
                "model": "vgae"
            },
            "knowledge_distillation": {
                "time": "4:00:00",
                "time_complex": "12:00:00",  # Much longer for complex datasets
                "mem": "48G",
                "mem_complex": "80G",  # More memory for complex datasets
                "cpus": 8, 
                "gpus": 1,
                "mode": "knowledge_distillation",
                "model": "gat"
            },
            "fusion": {
                "time": "3:00:00",
                "mem": "48G",
                "cpus": 8,
                "gpus": 1, 
                "mode": "fusion",
                "model": "gat"  # Not used for fusion
            }
        }
    
    def generate_slurm_script(self, job_name: str, training_type: str, dataset: str, 
                            extra_args: Dict[str, Any] = None) -> str:
        """Generate optimized SLURM script for unified training approach."""
        
        config = self.training_configs[training_type]
        extra_args = extra_args or {}
        
        # Use complex dataset configurations if needed
        is_complex = dataset in self.complex_datasets
        
        # Select appropriate time and memory based on dataset complexity
        time_limit = config.get("time_complex" if is_complex else "time", config["time"])
        memory = config.get("mem_complex" if is_complex else "mem", config["mem"])
        
        # Create job-specific output directory
        output_dir = f"{self.osc_settings['project_path']}/osc_jobs/{job_name}"
        
        # Build training command
        training_cmd = self._build_training_command(training_type, dataset, extra_args)
        
        # Generate SLURM script content with dynamic resource allocation
        script_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={config["cpus"]}
#SBATCH --mem={memory}
#SBATCH --gpus-per-node={config["gpus"]}
#SBATCH --account={self.osc_settings["account"]}
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err
#SBATCH --mail-user={self.osc_settings["email"]}

# Job information
echo "=== JOB INFORMATION ==="
echo "Job Name: {job_name}"
echo "Training Type: {training_type}"
echo "Dataset: {dataset} {'(COMPLEX)' if is_complex else '(STANDARD)'}"
echo "Time Limit: {time_limit}"
echo "Memory: {memory}"
echo "Job ID: $SLURM_JOB_ID" 
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "======================="

# Environment setup
module load miniconda3/24.1.2-py310
source activate {self.osc_settings["conda_env"]}
module load cuda/11.8.0

# GPU optimizations
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS={config["cpus"]}
export MKL_NUM_THREADS={config["cpus"]}
export NUMEXPR_NUM_THREADS={config["cpus"]}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directory on compute node
mkdir -p {output_dir}

# Dataset diagnostics
echo "=== DATASET DIAGNOSTICS ==="
echo "Checking dataset structure for {dataset}..."
ls -la datasets/ 2>/dev/null || echo "datasets/ not found"
ls -la datasets/can-train-and-test-v1.5/{dataset}/ 2>/dev/null || echo "Dataset folder not found"
find datasets/can-train-and-test-v1.5/{dataset}/ -name "*train_*.csv" | wc -l 2>/dev/null || echo "No train CSV files found"
echo "=============================="

# Clear dataset cache before processing to ensure fresh data
rm -rf datasets/cache/{dataset} 2>/dev/null || true
echo "Cleared cache for {dataset}"

# Change to project directory
cd {self.osc_settings["project_path"]}

# GPU information
echo "=== GPU INFORMATION ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "======================="

# Record start time
start_time=$(date +%s)

# Run training with unified script
echo "=== TRAINING COMMAND ==="
echo "{training_cmd}"
echo "========================"

{training_cmd}

# Record completion
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "=== JOB COMPLETED ==="
echo "Elapsed: ${{elapsed}}s ($((elapsed / 3600))h $((elapsed % 3600 / 60))m)"
echo "Finished: $(date)"
echo "===================="

# Copy important outputs to job directory
cp -r outputs/lightning_logs/* {output_dir}/ 2>/dev/null || true
cp saved_models/*{dataset}* {output_dir}/ 2>/dev/null || true

echo "✅ Job {job_name} completed successfully!"
'''
        
        return script_content
    
    def parse_extra_args(self, extra_args_str: str) -> Dict[str, Any]:
        """Parse extra arguments string into dictionary.
        
        Args:
            extra_args_str: String in format 'key1=value1' or 'key1=value1,key2=value2'
            
        Returns:
            Dictionary of parsed arguments
        """
        if not extra_args_str:
            return {}
        
        extra_args = {}
        # Handle both single argument and comma-separated arguments
        for arg in extra_args_str.split(','):
            if '=' in arg:
                key, value = arg.split('=', 1)
                extra_args[key.strip()] = value.strip()
        
        return extra_args
    
    def _build_training_command(self, training_type: str, dataset: str, 
                              extra_args: Dict[str, Any]) -> str:
        """Build the unified training command."""
        
        config = self.training_configs[training_type]
        
        # Base command using unified script
        cmd_parts = [
            "python train_with_hydra_zen.py",
            f"--dataset {dataset}",
            f"--training {config['mode']}"
        ]
        
        # Add model type for non-fusion modes
        if training_type != "fusion":
            cmd_parts.append(f"--model {config['model']}")
        
        # Add mode-specific arguments
        if training_type == "knowledge_distillation":
            teacher_path = f"saved_models/best_teacher_model_{dataset}.pth"
            cmd_parts.extend([
                f"--teacher_path {teacher_path}",
                "--student_scale 0.5"
            ])
        elif training_type == "fusion":
            cmd_parts.extend([
                f"--autoencoder_path saved_models/autoencoder_{dataset}.pth",
                f"--classifier_path saved_models/classifier_{dataset}.pth"
            ])
        
        # Add extra arguments
        for key, value in extra_args.items():
            cmd_parts.append(f"--{key} {value}")
            
        # Add standard debugging for dataset loading
        cmd_parts.append("--debug-graph-count")
        
        return " ".join(cmd_parts)
    
    def submit_individual_jobs(self, datasets: List[str] = None, 
                             training_types: List[str] = None,
                             extra_args: Dict[str, Any] = None) -> List[str]:
        """Submit individual training jobs."""
        
        datasets = datasets or self.datasets
        training_types = training_types or ["individual_gat", "individual_vgae"]
        extra_args = extra_args or {}
        
        submitted_jobs = []
        
        for dataset in datasets:
            for training_type in training_types:
                job_name = f"{training_type}_{dataset}"
                
                # Generate script
                script_content = self.generate_slurm_script(job_name, training_type, dataset, extra_args)
                
                # Save script
                script_path = self.jobs_dir / f"{job_name}.sh"
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # Submit job
                job_id = self._submit_slurm_job(script_path)
                if job_id:
                    submitted_jobs.append((job_name, job_id))
                    logger.info(f"✅ Submitted {job_name}: Job ID {job_id}")
                else:
                    logger.error(f"❌ Failed to submit {job_name}")
        
        return submitted_jobs
    
    def submit_pipeline_jobs(self, datasets: List[str] = None) -> List[str]:
        """Submit complete pipeline jobs (individual -> distillation -> fusion)."""
        
        datasets = datasets or self.datasets
        submitted_jobs = []
        
        for dataset in datasets:
            # Stage 1: Individual models (parallel)
            gat_job_id = self._submit_single_job("individual_gat", dataset)
            vgae_job_id = self._submit_single_job("individual_vgae", dataset)
            
            if not (gat_job_id and vgae_job_id):
                logger.error(f"❌ Failed to submit individual jobs for {dataset}")
                continue
            
            # Stage 2: Knowledge distillation (depends on GAT)
            kd_job_id = self._submit_single_job("knowledge_distillation", dataset, 
                                              dependency=gat_job_id)
            
            # Stage 3: Fusion (depends on both individual models)
            fusion_job_id = self._submit_single_job("fusion", dataset,
                                                  dependency=f"{gat_job_id}:{vgae_job_id}")
            
            submitted_jobs.extend([
                (f"pipeline_{dataset}_gat", gat_job_id),
                (f"pipeline_{dataset}_vgae", vgae_job_id), 
                (f"pipeline_{dataset}_kd", kd_job_id),
                (f"pipeline_{dataset}_fusion", fusion_job_id)
            ])
            
            logger.info(f"✅ Submitted pipeline for {dataset}: "
                       f"GAT({gat_job_id}), VGAE({vgae_job_id}), "
                       f"KD({kd_job_id}), Fusion({fusion_job_id})")
        
        return submitted_jobs
    
    def submit_parameter_sweep(self, training_type: str, dataset: str,
                             param_grid: Dict[str, List[Any]]) -> List[str]:
        """Submit parameter sweep jobs."""
        
        import itertools
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        submitted_jobs = []
        
        for i, combo in enumerate(combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, combo))
            
            # Create job name with parameters
            param_str = "_".join(f"{k}{v}" for k, v in params.items())
            job_name = f"{training_type}_{dataset}_sweep_{i:03d}_{param_str}"
            
            # Generate and submit job
            script_content = self.generate_slurm_script(job_name, training_type, 
                                                      dataset, params)
            script_path = self.jobs_dir / f"{job_name}.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            job_id = self._submit_slurm_job(script_path)
            if job_id:
                submitted_jobs.append((job_name, job_id))
                logger.info(f"✅ Submitted sweep job {i+1}/{len(combinations)}: {job_name}")
        
        return submitted_jobs
    
    def _submit_single_job(self, training_type: str, dataset: str, 
                          dependency: str = None) -> str:
        """Submit a single job with optional dependency."""
        
        job_name = f"{training_type}_{dataset}"
        script_content = self.generate_slurm_script(job_name, training_type, dataset)
        
        script_path = self.jobs_dir / f"{job_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return self._submit_slurm_job(script_path, dependency)
    
    def _submit_slurm_job(self, script_path: Path, dependency: str = None) -> str:
        """Submit SLURM job and return job ID."""
        
        cmd = ["sbatch"]
        
        if dependency:
            cmd.extend(["--dependency", f"afterok:{dependency}"])
        
        cmd.append(str(script_path))
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Extract job ID from "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]
                
                # Clean up old failed jobs to prevent buildup
                self._cleanup_old_jobs()
                
                return job_id
            else:
                logger.error(f"SLURM submission failed: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return None
    
    def _cleanup_old_jobs(self):
        """Clean up old failed job directories to prevent buildup."""
        try:
            job_dirs = list(self.jobs_dir.glob("*/"))
            if len(job_dirs) > 10:  # Keep only recent jobs
                # Sort by creation time, remove oldest failed jobs
                job_dirs.sort(key=lambda x: x.stat().st_mtime)
                
                for job_dir in job_dirs[:-10]:  # Keep 10 most recent
                    # Check if job failed (no success message)
                    out_files = list(job_dir.glob("slurm_*.out"))
                    failed = True
                    
                    for out_file in out_files:
                        try:
                            with open(out_file, 'r') as f:
                                if "✅" in f.read() and "completed successfully" in f.read():
                                    failed = False
                                    break
                        except:
                            pass
                    
                    if failed:
                        import shutil
                        shutil.rmtree(job_dir)
                        logger.info(f"🧹 Cleaned up old failed job: {job_dir.name}")
        except Exception as e:
            # Don't fail job submission due to cleanup issues
            logger.warning(f"Cleanup warning: {e}")
    
    def monitor_jobs(self, job_ids: List[str] = None) -> Dict[str, str]:
        """Monitor job status."""
        
        if job_ids:
            cmd = ["squeue", "-j", ",".join(job_ids), "--format=%i,%T,%N,%R"]
        else:
            cmd = ["squeue", "-u", "$USER", "--format=%i,%T,%N,%R"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                job_status = {}
                for line in lines:
                    if line.strip():
                        parts = line.split(',')
                        job_id = parts[0]
                        status = parts[1] 
                        job_status[job_id] = status
                return job_status
            else:
                logger.error(f"squeue failed: {result.stderr}")
                return {}
        except Exception as e:
            logger.error(f"Error monitoring jobs: {e}")
            return {}
    
    def generate_job_summary(self) -> str:
        """Generate summary of submitted jobs."""
        
        summary = []
        summary.append("# OSC Job Management Summary")
        summary.append(f"Generated: {datetime.now()}")
        summary.append("")
        
        # List available commands
        summary.append("## Quick Commands")
        summary.append("")
        summary.append("```bash")
        summary.append("# Submit individual training for all datasets")
        summary.append("python scripts/osc_job_manager.py --submit-individual")
        summary.append("")
        summary.append("# Submit fusion training only") 
        summary.append("python scripts/osc_job_manager.py --submit-fusion --datasets hcrl_sa,set_04")
        summary.append("")
        summary.append("# Submit complete pipeline")
        summary.append("python scripts/osc_job_manager.py --submit-pipeline --datasets hcrl_sa")
        summary.append("")
        summary.append("# Parameter sweep for fusion")
        summary.append("python scripts/osc_job_manager.py --parameter-sweep --training fusion --dataset hcrl_sa")
        summary.append("")
        summary.append("# Monitor all jobs")
        summary.append("python scripts/osc_job_manager.py --monitor")
        summary.append("```")
        
        return "\\n".join(summary)
    
    def cleanup_outputs(self):
        """Clean up old job outputs and failed runs."""
        logger.info("🧹 Cleaning up old job outputs...")
        
        cleaned_count = 0
        
        # Remove failed job directories
        for job_dir in self.jobs_dir.glob("*/"):
            if job_dir.is_dir():
                # Check if job was successful
                out_files = list(job_dir.glob("slurm_*.out"))
                failed = True
                
                for out_file in out_files:
                    try:
                        with open(out_file, 'r') as f:
                            content = f.read()
                            if "✅" in content and "completed successfully" in content:
                                failed = False
                                break
                    except:
                        pass
                
                if failed:
                    import shutil
                    shutil.rmtree(job_dir)
                    logger.info(f"   Removed failed job: {job_dir.name}")
                    cleaned_count += 1
        
        # Clean Lightning checkpoints
        checkpoint_dir = self.project_root / "saved_models" / "lightning_checkpoints"
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)
            logger.info("   Removed Lightning checkpoints")
        
        # Clean nested output directories
        nested_dirs = list(self.jobs_dir.glob("*/*_hcrl_sa_*"))
        for nested_dir in nested_dirs:
            if nested_dir.is_dir():
                import shutil
                shutil.rmtree(nested_dir)
        
        if len(nested_dirs) > 0:
            logger.info(f"   Removed {len(nested_dirs)} nested output directories")
        
        logger.info(f"✅ Cleanup complete! Removed {cleaned_count} failed job directories")


def main():
    parser = argparse.ArgumentParser(description="OSC Job Manager for CAN-Graph")
    parser.add_argument("--submit-individual", action="store_true", 
                       help="Submit individual training jobs")
    parser.add_argument("--submit-complex", action="store_true",
                       help="Submit jobs optimized for complex datasets (set_01-set_04)")
    parser.add_argument("--submit-fusion", action="store_true",
                       help="Submit fusion training jobs") 
    parser.add_argument("--submit-pipeline", action="store_true",
                       help="Submit complete pipeline jobs")
    parser.add_argument("--parameter-sweep", action="store_true",
                       help="Submit parameter sweep jobs")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor job status")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up old failed jobs and outputs")
    parser.add_argument("--datasets", type=str, 
                       help="Comma-separated list of datasets")
    parser.add_argument("--training", type=str, choices=["individual_gat", "individual_vgae", 
                       "knowledge_distillation", "fusion"],
                       help="Training type for parameter sweep")
    parser.add_argument("--class-balance", type=str, 
                       choices=["focal", "weighted", "undersample", "oversample", "smote"],
                       help="Class imbalance handling method for complex datasets")
    parser.add_argument("--extra-args", type=str,
                       help="Extra arguments to pass to training command (format: 'key1=value1,key2=value2')")
    parser.add_argument("--early-stopping-patience", type=int,
                       help="Early stopping patience (epochs to wait without improvement)")
    parser.add_argument("--force-rebuild-cache", action="store_true",
                       help="Force rebuild of dataset cache (use if graphs count is wrong)")
    parser.add_argument("--generate-summary", action="store_true",
                       help="Generate job management summary")
    
    args = parser.parse_args()
    
    manager = OSCJobManager()
    
    if args.cleanup:
        manager.cleanup_outputs()
        return
    
    # Parse datasets
    datasets = None
    if args.datasets:
        if args.datasets.lower() == "all":
            datasets = manager.datasets
        else:
            datasets = args.datasets.split(",")
    
    if args.submit_individual:
        logger.info("🚀 Submitting individual training jobs...")
        extra_args = manager.parse_extra_args(args.extra_args) if args.extra_args else {}
        
        # Add force rebuild cache option if requested
        if args.force_rebuild_cache:
            extra_args['force-rebuild-cache'] = True
            
        # Add early stopping patience if specified
        if args.early_stopping_patience:
            extra_args['early-stopping-patience'] = str(args.early_stopping_patience)
        training_types = [args.training] if args.training else None
        jobs = manager.submit_individual_jobs(datasets, training_types=training_types, extra_args=extra_args)
        logger.info(f"✅ Submitted {len(jobs)} individual jobs")
        
    elif args.submit_complex:
        logger.info("🚀 Submitting complex dataset training jobs...")
        complex_datasets = [d for d in (datasets or manager.complex_datasets) 
                          if d in manager.complex_datasets]
        
        if not complex_datasets:
            logger.error("❌ No complex datasets specified. Available: set_01, set_02, set_03, set_04")
            return 1
            
        jobs = []
        for dataset in complex_datasets:
            logger.info(f"📊 Submitting jobs for complex dataset: {dataset}")
            
            # Submit both GAT and VGAE for each complex dataset
            gat_job = manager._submit_single_job("individual_gat", dataset)
            vgae_job = manager._submit_single_job("individual_vgae", dataset)
            
            if gat_job:
                jobs.append((f"individual_gat_{dataset}", gat_job))
                logger.info(f"  ✅ GAT job: {gat_job}")
            if vgae_job:
                jobs.append((f"individual_vgae_{dataset}", vgae_job))
                logger.info(f"  ✅ VGAE job: {vgae_job}")
                
        logger.info(f"✅ Submitted {len(jobs)} complex dataset jobs")
        logger.info(f"⏱️  Extended walltime (8 hours) and memory (64G) for complex datasets")
        
    elif args.submit_fusion:
        logger.info("🚀 Submitting fusion training jobs...")
        datasets = datasets or manager.datasets
        jobs = []
        for dataset in datasets:
            job_id = manager._submit_single_job("fusion", dataset)
            if job_id:
                jobs.append((f"fusion_{dataset}", job_id))
        logger.info(f"✅ Submitted {len(jobs)} fusion jobs")
        
    elif args.submit_pipeline:
        logger.info("🚀 Submitting pipeline jobs...")
        jobs = manager.submit_pipeline_jobs(datasets)
        logger.info(f"✅ Submitted {len(jobs)} pipeline jobs")
        
    elif args.parameter_sweep:
        if not args.training or not args.datasets:
            logger.error("❌ Parameter sweep requires --training and --datasets")
            return 1
            
        # Example parameter grid for fusion
        param_grids = {
            "fusion": {
                "fusion_lr": [0.001, 0.0005, 0.002],
                "fusion_episodes": [100, 200, 500], 
                "fusion_batch_size": [256, 512]
            },
            "knowledge_distillation": {
                "student_scale": [0.3, 0.5, 0.7],
                "distillation_alpha": [0.5, 0.7, 0.9],
                "temperature": [3.0, 4.0, 5.0]
            }
        }
        
        if args.training not in param_grids:
            logger.error(f"❌ No parameter grid defined for {args.training}")
            return 1
            
        dataset = datasets[0] if datasets else "hcrl_sa"
        param_grid = param_grids[args.training]
        
        logger.info(f"🚀 Submitting parameter sweep for {args.training} on {dataset}...")
        jobs = manager.submit_parameter_sweep(args.training, dataset, param_grid)
        logger.info(f"✅ Submitted {len(jobs)} sweep jobs")
        
    elif args.monitor:
        logger.info("📊 Monitoring job status...")
        status = manager.monitor_jobs()
        if status:
            logger.info("Current jobs:")
            for job_id, job_status in status.items():
                logger.info(f"  Job {job_id}: {job_status}")
        else:
            logger.info("No jobs found")
            
    elif args.generate_summary:
        summary = manager.generate_job_summary()
        summary_path = manager.jobs_dir / "OSC_JOB_SUMMARY.md"
        with open(summary_path, 'w') as f:
            f.write(summary)
        logger.info(f"✅ Generated job summary: {summary_path}")
        
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())