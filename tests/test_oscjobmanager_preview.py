import sys
from pathlib import Path
import json

# Skip if omegaconf is not available (some dev envs may omit this dependency)
try:
    import omegaconf  # noqa: F401
except Exception:
    import pytest
    pytest.skip("omegaconf not available in test environment", allow_module_level=True)

# Ensure imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oscjobmanager as mgr_mod


def run_preview(args):
    import sys
    old = sys.argv
    try:
        sys.argv = ['oscjobmanager.py'] + args
        mgr_mod.main()
    finally:
        sys.argv = old


def test_preview_json_output(capsys):
    # Preview a small sweep and capture JSON output
    args = ['preview', '--dataset', 'hcrl_ch', '--model-sizes', 'student,teacher', '--distillations', 'no', '--training-modes', 'all_samples', '--json']
    run_preview(args)
    captured = capsys.readouterr()
    out = captured.out.strip()
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 2  # 2 model sizes -> 2 configs
    assert all('config_name' in d and 'run_dir' in d for d in data)
