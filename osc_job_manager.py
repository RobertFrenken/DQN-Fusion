#!/usr/bin/env python3
"""
I had to delete all of the old runs for a couple of reasons:
- First is that I was getting loading errors with the pickle files
- Second the path creation and saving is all messed up, fusion was under gat folders for some reason and this was causing the dqn models to freak out and the dqn models were trying to backup find paths that I deleted as they were old. What needs to happen is to seriously restructure the way models are saved and be strict on the appropiate paths for a given run. No fallbacks, if it isn't there give an error when you save a model it needs to be in a consisent place and it should be printed out. 
On Slurm outputs:
- The slurm job output needs to have contextual knowledge on the success of the job. Saying job completion without checking for errors doesn't help.

On file paths:
- Every single python file needs to be compatible with pytorch lightning and its utility functions. No old dependencies, no old code, no fallbacks, it either works or it crashes with an informative error.
On refactoring issues:
- The trainer python file that train_with_hydra_zen is now broken up in different parts into src.training. This has been causing issues and needs to be updated to the new dependency format
On new folder path grammer:
- in osc_job_manager.py I have started creating a new configuration setup to properly document each possible combination of model run.
- There needs to be a new pathing system to document each experiment which will take the following levels:
level 1: experiment_runs - This is the parent path that all models will be documented in
level 2: modality- lght now all automotive, but will expand to other types like physical systems and internet datasets
level 3: dataset (hcrl_ch, set_01, set_02, set_03, set_04): The specific dataset within that modality
level 4: learning type (unsupervised, classifier, fusion): The type of learning being conducted
level 5: model architecture (VGAE, GAT, DQN): this should have functionality for potential variations like generative GANs, LSTMS, other fusion mechancis
level 6: model size (Teacher, Student): this should have functionality for potential variations like an intermediate, a huge, or a tiny size 
level 7: Distillation (yes, no): This is above training type as a configuration could have both, which is why it is on a higher path level
level 8: training type (all samples, normal only , curriculum schedule classifier, curriculum schedule fusion, etc): This is the specific training strategy
level 9: Here the saved model should sit here, along with it's training metrics in a folder, the validation metrics, and when it is tested on the test set the evaluation results will be put into its own folder. Right now in the datasets it is split between train_ files and test_ files, with the test_ files in the set_xx datasets having unique tests for known/ unknown attacks and known/ unknown vehicles. I will need guidance here on the best way to orgainize these particular evaluations.
On model naming and saving:
- The models should be saved as a dictionary of the model weights in a file type that will not run into issues as the pickle files have
- the models need a descriptive name so that I can easily trace the path down and find the saved model. This was a big issue earlier

On MLFlow:
- I want MLflow to save the training metrics with that particular saved model, and I want the configuration of the GUI to be more comprehensive so when I launch an instance the UI will display strong organization of each type

On train_with_hyrda_zen:
- It looks like from around lines 192-272 there is a chunk of code that is present but dulled out by the linter implying that it will never be run. I want to make sure this is no longer needed, and if not I want to remove that section.

"""

"""
OSC Job Manager for CAN-Graph Training

Automates SLURM job submission on Ohio Supercomputer Center with:
- Hierarchical directory organization REWRITE THIS SECTION
- Parameterized job generation
- Batch job submission  
- Organized output management
- Job status monitoring
- Easy parameter sweeps
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import logging
from datetime import datetime
import shutil
import glob
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OSCJobManager:
    """Manage SLURM jobs on Ohio Supercomputer Center."""
    
    def __init__(self, project_root: Path = None):
        # Use repository root as canonical project root
        self.project_root = project_root or Path(__file__).parent.resolve()

        # Canonical experiments directory (new structure)
        self.experiments_dir = self.project_root / "experiment_runs"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # OSC-specific settings (customize for your account)
        self.osc_settings = {
            "account": "PAS3209",  # Your account
            "email": "frenken.2@osu.edu",  # Your email (used for SBATCH notifications if enabled)
            "project_path": str(self.project_root),  # Project path (keeps previous behaviour but derived)
            "conda_env": "gnn-gpu",  # Your conda environment
            "notify_webhook": "",  # Optional: Slack/Teams webhook URL for concise completion notifications
            "notify_email": "frenken.2@osu.edu"     # Optional: single email to receive job completion summaries (one per job)
        }

        self.osc_parameters = {
            "wall_time": "02:00:00",
            "memory": "32G",
            "cpus": 8,
            "gpus": 1
        }

        self.training_configurations = {
            "modalities": ["automotive", "internet", "water_treatment"],
            # will need to handle the pathing based on modalities in future
            # right now automotive is the only modality used
            "datasets": {"automotive": ["hcrl_sa", "hcrl_ch", "set_01", "set_02", "set_03", "set_04"],
                         "internet": [],
                         "water_treatment": []},
            "learning_types": ["unsupervised", "supervised", "rl_fusion"],
            "model_architectures": {"unsupervised": ["vgae"], "supervised": ["gat"], "rl_fusion": ["dqn"]},
            # right now small = student and teacher = large with options to expand
            "model_sizes": ["small", "medium", "large"],
            "distillation": ["no", "standard"],
            "training_modes": ["all_samples", "normals_only","curriculum_classifier", "curriculum_fusion"],
        }

    
    def generate_slurm_script(self, job_name: str, training_type: str, dataset: str, 
                            extra_args: Dict[str, Any] = None) -> str:
        """Generate an SBATCH script that runs `train_with_hydra_zen.py` for a given dataset and training.

        Returns the path to the generated script.
        """
        extra_args = extra_args or {}
        script_dir = self.project_root / 'slurm_jobs'
        script_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        script_path = script_dir / f"{job_name}_{dataset}_{timestamp}.sh"

        # Map training_type to model/training CLI args
        if training_type in ('autoencoder', 'vgae'):
            model = 'vgae'
            mode = 'autoencoder'
        elif training_type in ('curriculum', 'curriculum_classifier'):
            model = 'gat'
            mode = 'curriculum'
        elif training_type in ('fusion', 'rl_fusion'):
            model = 'gat'
            mode = 'fusion'
        elif training_type in ('normal', 'classifier'):
            model = 'gat'
            mode = 'normal'
        else:
            model = extra_args.get('model', 'gat')
            mode = training_type

        # Build the training command
        train_cmd = self._build_training_command(model, mode, dataset, extra_args)

        # SBATCH headers
        sbatch_headers = [
            '#!/usr/bin/env bash',
            f'#SBATCH --job-name={job_name}_{dataset}',
            f'#SBATCH --account={self.osc_settings.get("account")}',
            f'#SBATCH --time={self.osc_parameters.get("wall_time")}',
            f'#SBATCH --mem={self.osc_parameters.get("memory")}',
            f'#SBATCH --cpus-per-task={self.osc_parameters.get("cpus")}',
            f'#SBATCH --gres=gpu:{self.osc_parameters.get("gpus")}',
            f'#SBATCH --output={self.project_root}/slurm_logs/{job_name}_{dataset}_%j.out',
            f'#SBATCH --error={self.project_root}/slurm_logs/{job_name}_{dataset}_%j.err',
            f'#SBATCH --mail-user={self.osc_settings.get("notify_email")}',
            '#SBATCH --mail-type=END,FAIL'
        ]

        # Script body uses robust activation pattern
        body = [
            'set -euo pipefail',
            'echo "Starting job on $(hostname) at $(date)"',
            'echo "Project root: ' + str(self.project_root) + '"',
            'mkdir -p ' + str(self.project_root / 'slurm_logs'),
            'mkdir -p ' + str(self.project_root / 'slurm_jobs'),
            'source ~/.bashrc || true',
            f'echo "Activating conda env: {self.osc_settings.get("conda_env")}"',
            f'conda activate {self.osc_settings.get("conda_env")} || true',
            '# Export canonical experiment root for the job manager and training script',
            f'export CAN_EXPERIMENT_ROOT="{self.experiments_dir}"',
            '',
            'echo "Running training command:"',
            f'echo "{train_cmd}"',
            f'{train_cmd} 2>&1 | tee {self.project_root}/slurm_logs/{job_name}_{dataset}_$SLURM_JOB_ID.log',
            'echo "Job finished at $(date)"'
        ]

        with open(script_path, 'w') as f:
            f.write('\n'.join(sbatch_headers) + '\n\n' + '\n'.join(body) + '\n')

        # Make script executable
        script_path.chmod(0o755)
        logger.info(f"Generated SLURM script: {script_path}")
        return str(script_path)
    
    def parse_extra_args(self, extra_args_str: str) -> Dict[str, Any]:
        """Parse comma-separated key=value pairs into a dict.

        Example: 'max_epochs=50,batch_size=32'
        """
        if not extra_args_str:
            return {}
        parts = [p.strip() for p in extra_args_str.split(',') if p.strip()]
        out = {}
        for part in parts:
            if '=' in part:
                k, v = part.split('=', 1)
                out[k.strip()] = v.strip()
            else:
                out[part] = True
        return out
    
    def _build_training_command(self, model: str, mode: str, dataset: str, extra_args: Dict[str, Any]) -> str:
        """Construct the python command invoking train_with_hydra_zen.py with chosen options."""
        cmd = [
            'python', str(self.project_root / 'train_with_hydra_zen.py'),
            '--model', model,
            '--dataset', dataset,
            '--training', mode
        ]

        # Map known overrides
        if extra_args:
            for k, v in extra_args.items():
                if k in ('max_epochs', 'epochs'):
                    cmd += ['--epochs', str(v)]
                elif k == 'batch_size' or k == 'batch-size':
                    cmd += ['--batch_size', str(v)]
                elif k == 'teacher_path':
                    cmd += ['--teacher_path', str(v)]
                elif k == 'dependency_manifest':
                    cmd += ['--dependency-manifest', str(v)]
                elif isinstance(v, bool) and v:
                    cmd += [f'--{k}']
                else:
                    # Generic key-value passed as --key value
                    cmd += [f'--{k}', str(v)]

        # Join safely
        return ' '.join(cmd)
    
    def submit_individual_jobs(self, datasets: List[str] = None, 
                             training_types: List[str] = None,
                             extra_args: Dict[str, Any] = None, submit: bool = True) -> List[str]:
        """Generate and optionally submit jobs for each dataset x training_type pair.

        Returns list of submitted job IDs or script paths when submit=False.
        """
        datasets = datasets or self.training_configurations['datasets'].get('automotive', [])
        training_types = training_types or ['autoencoder', 'curriculum', 'fusion']

        submitted = []
        for t in training_types:
            for ds in datasets:
                job_name = f"can_{t}"
                script_path = self.generate_slurm_script(job_name, t, ds, extra_args=extra_args)
                if submit:
                    job_id = self._submit_slurm_job(Path(script_path))
                    submitted.append(job_id)
                else:
                    submitted.append(script_path)
        return submitted    
    def submit_pipeline_jobs(self, datasets: List[str] = None) -> List[str]:
        """Submit pipeline jobs (autoencoder -> curriculum -> fusion) for each dataset."""
        datasets = datasets or self.training_configurations['datasets'].get('automotive', [])
        submitted = []
        for ds in datasets:
            # Submit autoencoder
            j1 = self.submit_individual_jobs([ds], ['autoencoder'])
            # Submit curriculum (depends on autoencoder)
            j2 = self.submit_individual_jobs([ds], ['curriculum'])
            # Submit fusion (depends on classifier and autoencoder artifacts already present)
            j3 = self.submit_individual_jobs([ds], ['fusion'])
            submitted.extend(j1 + j2 + j3)
        return submitted    
    def submit_parameter_sweep(self, training_type: str, dataset: str,
                             param_grid: Dict[str, List[Any]]) -> List[str]:
        """Create multiple jobs for each parameter combination and submit them."""
        keys, values = zip(*param_grid.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
        job_ids = []
        for i, combo in enumerate(combos):
            job_name = f"sweep_{training_type}_{dataset}_{i}"
            script = self.generate_slurm_script(job_name, training_type, dataset, extra_args=combo)
            job_ids.append(self._submit_slurm_job(Path(script)))
        return job_ids
    
    def _submit_slurm_job(self, script_path: Path, dependency: str = None) -> str:
        """Submit SLURM job using sbatch and return the job ID string (or script path if sbatch not available)."""
        try:
            cmd = ["sbatch"]
            if dependency:
                cmd += ["--dependency", f"afterok:{dependency}"]
            cmd.append(str(script_path))
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
            # sbatch output: Submitted batch job 123456
            job_id = out.strip().split()[-1]
            logger.info(f"Submitted job {job_id} for script {script_path}")
            return job_id
        except Exception as e:
            logger.warning(f"sbatch failed or not available: {e}. Script written to {script_path}")
            return str(script_path)

    
    def _cleanup_old_jobs(self):
        """Remove old slurm job artifacts older than 90 days to prevent buildup."""
        log_dir = self.project_root / 'slurm_logs'
        if not log_dir.exists():
            return
        for p in log_dir.glob('*.log'):
            try:
                if (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days > 90:
                    p.unlink()
            except Exception:
                pass

    def monitor_jobs(self, job_ids: List[str] = None) -> Dict[str, str]:
        """Return a dict mapping job_id -> status (pending|running|completed|unknown).

        Uses `squeue` when available, otherwise returns 'unknown'.
        """
        statuses = {}
        try:
            out = subprocess.check_output(['squeue', '-j', ','.join(job_ids)]).decode('utf-8')
            for line in out.splitlines()[1:]:
                parts = line.split()
                if parts:
                    jid = parts[0]
                    state = parts[4]
                    statuses[jid] = state
            return statuses
        except Exception:
            for jid in job_ids or []:
                statuses[jid] = 'unknown'
            return statuses

    def generate_job_summary(self) -> str:
        """Return a short summary of experiments directory size and job counts."""
        total = sum(1 for _ in self.experiments_dir.rglob('*.pth'))
        return f"Experiments path: {self.experiments_dir} - model artifacts: {total} files"

    def cleanup_outputs(self):
        """Remove failed directories older than a month (heuristic)."""
        runs = list(self.experiments_dir.iterdir()) if self.experiments_dir.exists() else []
        for r in runs:
            try:
                if r.is_dir() and (datetime.now() - datetime.fromtimestamp(r.stat().st_mtime)).days > 30 and r.name.startswith('tmp'):
                    shutil.rmtree(r)
            except Exception:
                pass


def main():
      
    manager = OSCJobManager()


if __name__ == "__main__":
    sys.exit(main())