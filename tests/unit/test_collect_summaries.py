import json
from pathlib import Path
import sys
# Make imports robust when running tests directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_summaries import find_summaries


def test_find_summaries(tmp_path):
    root = tmp_path / 'experiment_runs'
    d1 = root / 'a' / 'run_000'
    d1.mkdir(parents=True)
    s1 = d1 / 'summary.json'
    s1.write_text(json.dumps({'model': 'vgae', 'dataset': 'hcrl_ch', 'training_mode': 'autoencoder'}))

    d2 = root / 'b' / 'run_000'
    d2.mkdir(parents=True)
    s2 = d2 / 'summary.json'
    s2.write_text(json.dumps({'model': 'gat', 'dataset': 'hcrl_sa', 'training_mode': 'normal'}))

    summaries = find_summaries(root)
    assert len(summaries) == 2
    models = {s['model'] for s in summaries}
    assert 'vgae' in models and 'gat' in models
