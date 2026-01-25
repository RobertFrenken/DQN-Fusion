from pathlib import Path
import sys
# Make imports robust when running tests directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.local_smoke_experiment import create_synthetic_dataset


def test_create_synthetic_dataset(tmp_path):
    d = tmp_path / 'synth'
    create_synthetic_dataset(d, num_rows=150, seed=123)
    assert d.exists()
    csvs = list(d.glob('**/*.csv'))
    assert len(csvs) >= 1
    # Basic content check
    content = csvs[0].read_text()
    assert 'Timestamp' in content
    assert len(content.splitlines()) >= 151
