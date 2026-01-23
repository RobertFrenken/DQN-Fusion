import sys
import json
from pathlib import Path
# Ensure project root is on sys.path for test imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osc_job_manager import OSCJobManager

def test_dry_run_suggestions(tmp_path):
    mgr = OSCJobManager(project_root=tmp_path)
    # Use dataset name that is in training_configurations
    suggestions = mgr.submit_individual_jobs(datasets=['hcrl_ch'], training_types=['fusion'], submit=False, dry_run=True)
    assert isinstance(suggestions, list)
    assert len(suggestions) == 1
    s = suggestions[0]
    assert 'expected_artifacts' in s
    assert any('vgae_autoencoder.pth' in p for p in s['expected_artifacts'])
    assert 'suggested_dataset_path' in s
    # If dataset not present, valid should be False or errors non-empty
    assert (not s['valid']) or isinstance(s['errors'], list)
