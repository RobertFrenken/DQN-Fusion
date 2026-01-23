import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_datasets import check_path


def test_check_path(tmp_path):
    d = tmp_path / 'nonexistent'
    assert check_path(d) is False
    d.mkdir()
    assert check_path(d) is True
