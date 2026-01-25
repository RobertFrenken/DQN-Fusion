import sys
from pathlib import Path
import subprocess
import json
import types

# Provide a lightweight fake for omegaconf if not available in this test environment
if 'omegaconf' not in sys.modules:
    fake_omegaconf = types.ModuleType('omegaconf')
    def _make_ns(d):
        # convert top-level dict to nested SimpleNamespace recursively one level deep
        out = {}
        for k,v in d.items():
            if isinstance(v, dict):
                out[k] = types.SimpleNamespace(**v)
            else:
                out[k] = v
        return types.SimpleNamespace(**out)
    fake_omegaconf.OmegaConf = types.SimpleNamespace(create=lambda x: _make_ns(x))
    sys.modules['omegaconf'] = fake_omegaconf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oscjobmanager import OSCJobManager


class FakeCompleted:
    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_submit_aborts_on_pre_submit_failure(monkeypatch, tmp_path):
    mgr = OSCJobManager()
    experiment_dir = tmp_path / 'runs'
    experiment_dir.mkdir()

    # Fake subprocess.run: return non-zero for pre_submit_check and ensure sbatch never called
    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        s = ' '.join(cmd)
        if 'pre_submit_check.py' in s:
            return FakeCompleted(returncode=2, stdout='pre-fail', stderr='failed')
        if 'sbatch' in s:
            raise AssertionError('sbatch should not be invoked when pre-submit fails')
        return FakeCompleted(returncode=1, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    job_id = mgr.submit_job('automotive_hcrlch_unsupervised_vgae_student_no_all_samples', experiment_dir, pre_submit=True)
    assert job_id is None


def test_submit_runs_when_pre_submit_passes(monkeypatch, tmp_path):
    mgr = OSCJobManager()
    experiment_dir = tmp_path / 'runs'
    experiment_dir.mkdir()

    # Fake subprocess.run: success for pre_submit_check and sbatch
    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        s = ' '.join(cmd)
        if 'pre_submit_check.py' in s:
            return FakeCompleted(returncode=0, stdout='pre-ok', stderr='')
        if 'sbatch' in s:
            return FakeCompleted(returncode=0, stdout='Submitted batch job 9999', stderr='')
        return FakeCompleted(returncode=1, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    job_id = mgr.submit_job('automotive_hcrlch_unsupervised_vgae_student_no_all_samples', experiment_dir, pre_submit=True)
    assert job_id == '9999'
