#!/usr/bin/env python3
"""
Submit sequential training pipelines with SLURM dependency chaining.

Combines the configuration power of oscjobmanager.py with automatic job dependencies.

Examples:
    # Teacher pipeline (VGAE → Curriculum GAT → Fusion DQN)
    python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa
    
    # Student pipeline (VGAE Student → Distilled GAT Student)
    python scripts/submit_pipeline.py --pipeline student --dataset hcrl_sa --teacher-path path/to/teacher.pth
    
    # Custom pipeline
    python scripts/submit_pipeline.py --pipeline custom \
        --presets autoencoder_hcrl_sa,curriculum_hcrl_sa,fusion_hcrl_sa \
        --dataset hcrl_sa --walltime 08:00:00 --memory 96G
    
    # Dry run (preview only)
    python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa --dry-run
"""

import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Pipeline Definitions
# ============================================================================

PIPELINES = {
    "teacher": {
        "name": "Teacher Pipeline",
        "description": "Train teacher-sized models: VGAE → Curriculum GAT → Fusion DQN",
        "stages": [
            {"name": "VGAE Autoencoder", "preset": "autoencoder_{dataset}", 
             "walltime": "06:00:00", "memory": "48G"},
            {"name": "Curriculum GAT", "preset": "curriculum_{dataset}",
             "walltime": "6:00:00", "memory": "48G"},
            {"name": "Fusion DQN", "preset": "fusion_{dataset}",
             "walltime": "6:00:00", "memory": "48G"}
        ]
    },
    "student": {
        "name": "Student Pipeline",
        "description": "Train student models with distillation",
        "stages": [
            {"name": "VGAE Student Autoencoder", "preset": "autoencoder_{dataset}",
             "walltime": "06:00:00", "memory": "64G"},
            {"name": "Distilled GAT Student", "preset": "distillation_{dataset}_scale_0.5",
             "walltime": "08:00:00", "memory": "96G"}
        ]
    },
    "supervised_only": {
        "name": "Supervised Pipeline",
        "description": "Train only supervised models: GAT Normal → Fusion",
        "stages": [
            {"name": "VGAE Autoencoder", "preset": "autoencoder_{dataset}",
             "walltime": "08:00:00", "memory": "96G"},
            {"name": "GAT Normal", "preset": "gat_normal_{dataset}",
             "walltime": "10:00:00", "memory": "128G"},
            {"name": "Fusion DQN", "preset": "fusion_{dataset}",
             "walltime": "10:00:00", "memory": "96G"}
        ]
    },
    "curriculum_only": {
        "name": "Curriculum Only",
        "description": "VGAE → Curriculum GAT (no fusion)",
        "stages": [
            {"name": "VGAE Autoencoder", "preset": "autoencoder_{dataset}",
             "walltime": "08:00:00", "memory": "96G"},
            {"name": "Curriculum GAT", "preset": "curriculum_{dataset}",
             "walltime": "12:00:00", "memory": "128G"}
        ]
    }
}


class PipelineSubmitter:
    """Submits training pipelines with SLURM dependency chaining."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.project_root = PROJECT_ROOT
        
    def create_slurm_script(self, preset: str, dataset: str, 
                           walltime: Optional[str] = None,
                           memory: Optional[str] = None,
                           cpus: Optional[int] = None,
                           gpus: Optional[int] = None) -> Path:
        """Create a SLURM script using oscjobmanager."""
        cmd = ["python", "oscjobmanager.py", "submit", preset, "--dry-run"]
        
        if walltime:
            cmd.extend(["--walltime", walltime])
        if memory:
            cmd.extend(["--memory", memory])
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=self.project_root, timeout=30)
            
            # Parse output to find script path
            for line in result.stdout.split('\n'):
                if 'Script:' in line or 'script' in line.lower():
                    # Extract path
                    parts = line.split()
                    for part in parts:
                        if '.sh' in part:
                            return Path(part)
            
            # Fallback: look for the script in slurm_runs
            slurm_dir = self.project_root / "experimentruns" / "slurm_runs"
            if slurm_dir.exists():
                scripts = sorted(slurm_dir.glob(f"*{preset}*.sh"), key=lambda p: p.stat().st_mtime)
                if scripts:
                    return scripts[-1]  # Most recent
                    
            raise RuntimeError(f"Could not find generated script for {preset}")
            
        except Exception as e:
            logger.error(f"Failed to create script for {preset}: {e}")
            raise
    
    def submit_job(self, script_path: Path, dependency_job_id: Optional[str] = None) -> str:
        """Submit a job to SLURM with optional dependency."""
        cmd = ["sbatch"]
        
        if dependency_job_id:
            cmd.append(f"--dependency=afterok:{dependency_job_id}")
            
        cmd.append(str(script_path))
        
        if self.dry_run:
            logger.info(f"   [DRY RUN] Would submit: {' '.join(cmd)}")
            # Return fake job ID for dry run
            return f"DRYRUN_{datetime.now().strftime('%H%M%S')}"
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Extract job ID from "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]
                return job_id
            else:
                raise RuntimeError(f"sbatch failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            raise
    
    def submit_pipeline(self, pipeline_name: str, dataset: str,
                       walltime: Optional[str] = None,
                       memory: Optional[str] = None,
                       cpus: Optional[int] = None,
                       gpus: Optional[int] = None,
                       custom_presets: Optional[List[str]] = None) -> Dict:
        """Submit a complete pipeline with dependency chaining."""
        
        if custom_presets:
            stages = [{"name": preset, "preset": preset} for preset in custom_presets]
            pipeline_desc = "Custom Pipeline"
        elif pipeline_name not in PIPELINES:
            raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {list(PIPELINES.keys())}")
        else:
            pipeline_config = PIPELINES[pipeline_name]
            stages = pipeline_config["stages"]
            pipeline_desc = pipeline_config["description"]
        
        logger.info("=" * 80)
        logger.info(f"🚀 Submitting Pipeline: {pipeline_desc}")
        logger.info("=" * 80)
        logger.info(f"Dataset: {dataset}")
        if walltime:
            logger.info(f"Walltime: {walltime}")
        if memory:
            logger.info(f"Memory: {memory}")
        logger.info(f"Stages: {len(stages)}")
        logger.info("")
        
        submitted_jobs = []
        previous_job_id = None
        
        for i, stage in enumerate(stages, 1):
            stage_name = stage["name"]
            preset_template = stage["preset"]
            preset = preset_template.format(dataset=dataset)
            
            # Use stage-specific resources if defined, otherwise use global defaults
            stage_walltime = walltime or stage.get("walltime")
            stage_memory = memory or stage.get("memory")
            stage_cpus = cpus or stage.get("cpus")
            stage_gpus = gpus or stage.get("gpus")
            
            logger.info(f"📦 Stage {i}/{len(stages)}: {stage_name}")
            logger.info(f"   Preset: {preset}")
            if stage_walltime:
                logger.info(f"   Walltime: {stage_walltime}")
            if stage_memory:
                logger.info(f"   Memory: {stage_memory}")
            
            # Create SLURM script
            try:
                script_path = self.create_slurm_script(
                    preset=preset,
                    dataset=dataset,
                    walltime=stage_walltime,
                    memory=stage_memory,
                    cpus=stage_cpus,
                    gpus=stage_gpus
                )
                logger.info(f"   Script: {script_path}")
            except Exception as e:
                logger.error(f"   ❌ Failed to create script: {e}")
                return {"status": "failed", "error": str(e), "stage": i}
            
            # Submit job
            try:
                job_id = self.submit_job(script_path, dependency_job_id=previous_job_id)
                
                if previous_job_id:
                    logger.info(f"   ✅ Job ID: {job_id} (depends on {previous_job_id})")
                else:
                    logger.info(f"   ✅ Job ID: {job_id}")
                
                submitted_jobs.append({
                    "stage": i,
                    "name": stage_name,
                    "preset": preset,
                    "job_id": job_id,
                    "depends_on": previous_job_id,
                    "script": str(script_path)
                })
                
                previous_job_id = job_id
                
            except Exception as e:
                logger.error(f"   ❌ Failed to submit job: {e}")
                return {"status": "failed", "error": str(e), "stage": i}
            
            logger.info("")
        
        # Summary
        logger.info("=" * 80)
        logger.info("✅ Pipeline Submitted Successfully!")
        logger.info("=" * 80)
        logger.info("Job Chain:")
        for job in submitted_jobs:
            dep_str = f" → (after {job['depends_on']})" if job['depends_on'] else ""
            logger.info(f"  {job['stage']}. {job['name']}: {job['job_id']}{dep_str}")
        logger.info("")
        
        if not self.dry_run:
            logger.info("Monitor with: squeue -u $USER")
            logger.info("Cancel all: scancel " + " ".join([j["job_id"] for j in submitted_jobs]))
        
        logger.info("=" * 80)
        
        return {
            "status": "success",
            "pipeline": pipeline_name if not custom_presets else "custom",
            "dataset": dataset,
            "jobs": submitted_jobs
        }


def main():
    parser = argparse.ArgumentParser(
        description="Submit training pipelines with automatic SLURM dependency chaining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Pipelines:
  teacher          - VGAE → Curriculum GAT → Fusion DQN (teacher-sized models)
  student          - VGAE Student → Distilled GAT Student
  supervised_only  - VGAE → GAT Normal → Fusion (no curriculum)
  curriculum_only  - VGAE → Curriculum GAT (no fusion)
  custom           - Specify custom preset sequence with --presets

Examples:
  # Teacher pipeline
  python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa
  
  # With custom resources
  python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa \\
      --walltime 12:00:00 --memory 128G
  
  # Custom pipeline
  python scripts/submit_pipeline.py --pipeline custom --dataset hcrl_sa \\
      --presets autoencoder_hcrl_sa,gat_normal_hcrl_sa,fusion_hcrl_sa
  
  # Dry run
  python scripts/submit_pipeline.py --pipeline teacher --dataset hcrl_sa --dry-run
        """
    )
    
    parser.add_argument("--pipeline", required=True,
                       choices=list(PIPELINES.keys()) + ["custom"],
                       help="Pipeline type to submit")
    
    parser.add_argument("--dataset", required=True,
                       help="Dataset name (e.g., hcrl_sa, hcrl_ch)")
    
    parser.add_argument("--presets",
                       help="Comma-separated preset names (for custom pipeline)")
    
    parser.add_argument("--walltime",
                       help="Job walltime (e.g., 08:00:00)")
    
    parser.add_argument("--memory",
                       help="Memory allocation (e.g., 96G)")
    
    parser.add_argument("--cpus", type=int,
                       help="CPUs per task")
    
    parser.add_argument("--gpus", type=int,
                       help="GPUs per node")
    
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview pipeline without submitting")
    
    parser.add_argument("--list-pipelines", action="store_true",
                       help="List available pipelines and exit")
    
    args = parser.parse_args()
    
    # List pipelines
    if args.list_pipelines:
        print("\nAvailable Pipelines:")
        print("=" * 80)
        for name, config in PIPELINES.items():
            print(f"\n{name}:")
            print(f"  {config['description']}")
            print(f"  Stages:")
            for i, stage in enumerate(config['stages'], 1):
                print(f"    {i}. {stage['name']}")
        print("\n" + "=" * 80)
        return
    
    # Parse custom presets
    custom_presets = None
    if args.pipeline == "custom":
        if not args.presets:
            parser.error("--presets required for custom pipeline")
        custom_presets = [p.strip() for p in args.presets.split(",")]
    
    # Submit pipeline
    submitter = PipelineSubmitter(dry_run=args.dry_run)
    
    try:
        result = submitter.submit_pipeline(
            pipeline_name=args.pipeline,
            dataset=args.dataset,
            walltime=args.walltime,
            memory=args.memory,
            cpus=args.cpus,
            gpus=args.gpus,
            custom_presets=custom_presets
        )
        
        if result["status"] == "success":
            sys.exit(0)
        else:
            logger.error(f"Pipeline submission failed: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
