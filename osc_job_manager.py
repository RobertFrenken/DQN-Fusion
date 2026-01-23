#!/usr/bin/env python3
"""
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

        Returns the path to the generated script and writes a metadata sidecar file for traceability.
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

        # Write metadata sidecar
        import json as _json
        import subprocess as _subp
        meta = {
            'job_name': job_name,
            'dataset': dataset,
            'training_type': training_type,
            'extra_args': extra_args,
            'script_path': str(script_path),
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'experiment_root': str(self.experiments_dir)
        }
        # Try get git commit
        try:
            out = _subp.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=_subp.DEVNULL).decode().strip()
            meta['git_commit'] = out
        except Exception:
            meta['git_commit'] = None

        meta_path = script_path.with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            _json.dump(meta, f, indent=2)

        logger.info(f"Generated SLURM script: {script_path} and metadata: {meta_path}")
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
                             extra_args: Dict[str, Any] = None, submit: bool = True, dry_run: bool = False) -> List[str]:
        """Generate and optionally submit jobs for each dataset x training_type pair.

        When dry_run=True, do not write or submit scripts; instead return helpful suggestions
        including canonical artifact locations and dataset path suggestions.

        Returns list of submitted job IDs or script paths when submit=False.
        """
        datasets = datasets or self.training_configurations['datasets'].get('automotive', [])
        training_types = training_types or ['autoencoder', 'curriculum', 'fusion']

        submitted = []
        for t in training_types:
            for ds in datasets:
                job_name = f"can_{t}"

                # Validate job spec and gather any helpful messages
                ok, errors = self.validate_job_spec(t, ds, extra_args=extra_args)
                if dry_run:
                    # Build suggestion payload instead of writing scripts
                    suggestion = {
                        'job_name': job_name,
                        'dataset': ds,
                        'training_type': t,
                        'valid': ok,
                        'errors': errors,
                        'suggested_dataset_path': str(self.project_root / 'datasets' / 'can-train-and-test-v1.5' / ds),
                        'expected_artifacts': []
                    }

                    # If fusion/curriculum - include artifact canonical suggestions
                    if t in ('fusion', 'rl_fusion'):
                        ae = self.experiments_dir / 'automotive' / ds / 'unsupervised' / 'vgae' / 'teacher' / 'no_distillation' / 'autoencoder' / 'vgae_autoencoder.pth'
                        clf = self.experiments_dir / 'automotive' / ds / 'supervised' / 'gat' / 'teacher' / 'no_distillation' / 'normal' / f'gat_{ds}_normal.pth'
                        suggestion['expected_artifacts'].extend([str(ae), str(clf)])
                    if t in ('curriculum', 'curriculum_classifier'):
                        vgae = self.experiments_dir / 'automotive' / ds / 'unsupervised' / 'vgae' / 'teacher' / 'no_distillation' / 'autoencoder' / 'vgae_autoencoder.pth'
                        suggestion['expected_artifacts'].append(str(vgae))

                    submitted.append(suggestion)
                    continue

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
        """Submit SLURM job using sbatch and return the job ID string (or script path if sbatch not available).

        Updates the script metadata with submission details (job id, timestamp) for auditing.
        """
        meta_path = script_path.with_suffix('.meta.json')
        try:
            cmd = ["sbatch"]
            if dependency:
                cmd += ["--dependency", f"afterok:{dependency}"]
            cmd.append(str(script_path))
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
            # sbatch output: Submitted batch job 123456
            job_id = out.strip().split()[-1]
            logger.info(f"Submitted job {job_id} for script {script_path}")

            # Update metadata sidecar
            try:
                import json as _json
                meta = {}
                if meta_path.exists():
                    with open(meta_path, 'r') as mf:
                        meta = _json.load(mf)
                meta.update({'submitted_at': datetime.utcnow().isoformat() + 'Z', 'sbatch_job_id': str(job_id)})
                with open(meta_path, 'w') as mf:
                    _json.dump(meta, mf, indent=2)
            except Exception as e:
                logger.warning(f"Failed to update metadata for {script_path}: {e}")

            return job_id
        except Exception as e:
            logger.warning(f"sbatch failed or not available: {e}. Script written to {script_path}")
            # annotate metadata with failure
            try:
                import json as _json
                meta = {}
                if meta_path.exists():
                    with open(meta_path, 'r') as mf:
                        meta = _json.load(mf)
                meta.update({'submitted_at': None, 'submission_error': str(e)})
                with open(meta_path, 'w') as mf:
                    _json.dump(meta, mf, indent=2)
            except Exception:
                pass
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

    def validate_job_spec(self, training_type: str, dataset: str, extra_args: Dict[str, Any] = None) -> (bool, List[str]):
        """Validate that a job is valid for submission.

        Checks performed:
        - Basic type checks for common extra_args (epochs, batch_size)
        - For fusion: check that either a dependency_manifest exists and points to files or the canonical artifacts exist.
        - For curriculum: ensure VGAE artifact exists or vgae_model_path override provided.

        Returns (True, []) on success, else (False, errors)
        """
        errors: List[str] = []
        extra_args = extra_args or {}

        # Basic validation
        for int_key in ('epochs', 'batch_size', 'max_epochs'):
            if int_key in extra_args:
                try:
                    v = int(extra_args[int_key])
                    if v <= 0:
                        errors.append(f"{int_key} must be > 0")
                except Exception:
                    errors.append(f"{int_key} must be an integer")

        # Detect dependency manifest path
        dep_manifest = extra_args.get('dependency_manifest')

        # For fusion jobs, ensure artifacts are available
        if training_type in ('fusion', 'rl_fusion'):
            if dep_manifest:
                p = Path(dep_manifest)
                if not p.exists():
                    errors.append(f"Dependency manifest not found: {dep_manifest}")
            else:
                # Check canonical artifact paths
                ae = self.experiments_dir / 'automotive' / dataset / 'unsupervised' / 'vgae' / 'teacher' / 'no_distillation' / 'autoencoder' / 'vgae_autoencoder.pth'
                clf = self.experiments_dir / 'automotive' / dataset / 'supervised' / 'gat' / 'teacher' / 'no_distillation' / 'normal' / f'gat_{dataset}_normal.pth'
                if not ae.exists():
                    errors.append(f"Autoencoder artifact missing at canonical path: {ae}")
                if not clf.exists():
                    errors.append(f"Classifier artifact missing at canonical path: {clf}")

        # For curriculum, ensure VGAE exists or override provided
        if training_type in ('curriculum', 'curriculum_classifier'):
            vgae_override = extra_args.get('vgae_path') or extra_args.get('vgae_model_path')
            if vgae_override:
                if not Path(vgae_override).exists():
                    errors.append(f"Provided VGAE override path does not exist: {vgae_override}")
            else:
                vgae = self.experiments_dir / 'automotive' / dataset / 'unsupervised' / 'vgae' / 'teacher' / 'no_distillation' / 'autoencoder' / 'vgae_autoencoder.pth'
                if not vgae.exists():
                    errors.append(f"VGAE artifact required for curriculum is missing at {vgae}. Consider running VGAE first or provide --vgae_path.")

        # Return
        return (len(errors) == 0), errors

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