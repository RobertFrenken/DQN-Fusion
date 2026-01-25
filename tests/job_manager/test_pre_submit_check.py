import sys
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.pre_submit_check as psc


def fake_run(cmd, capture_output=True, text=True, timeout=None):
    # Simple fake outputs depending on command
    s = ' '.join(cmd)
    if 'check_environment.py' in s:
        return SimpleNamespace(returncode=0, stdout='env ok', stderr='')
    if 'check_datasets.py' in s:
        return SimpleNamespace(returncode=0, stdout='dataset ok', stderr='')
    if 'local_smoke_experiment.py' in s:
        return SimpleNamespace(returncode=0, stdout='smoke ok', stderr='')
    if 'oscjobmanager.py' in s and 'preview' in s:
        return SimpleNamespace(returncode=0, stdout='[]', stderr='')
    return SimpleNamespace(returncode=1, stdout='', stderr='error')


@patch('subprocess.run', side_effect=fake_run)
def test_pre_submit_happy_path(mock_run, capsys):
    # Should exit 0 when all subtasks report success
    with patch.object(sys, 'argv', ['pre_submit_check.py', '--dataset', 'hcrl_ch', '--run-load', '--smoke', '--smoke-synthetic', '--preview-json']):
        try:
            psc.main()
        except SystemExit as e:
            assert e.code == 0

    # Check captured output contains READY
    captured = capsys.readouterr()
    assert 'READY' in captured.out
