import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import tempfile
import pickle
import numpy as np
from pathlib import Path

from scripts.validate_artifact import validate_artifact


def test_validate_artifact_sanitizes(tmp_path):
    # Create a fake checkpoint containing a numpy array
    ck = {'q_network_state_dict': {'layer.weight': np.array([1.0, 2.0])}, 'some_meta': np.float64(0.5)}
    p = tmp_path / 'fake_ckpt.pkl'
    with open(p, 'wb') as f:
        pickle.dump(ck, f)

    # Validate and resave sanitized copy
    out = validate_artifact(p, resave_sanitized=True)
    assert out.exists()
    # sanitized file should have _sanitized in name
    assert '_sanitized' in out.name

    # Confirm sanitized content contains plain lists/scalars
    import pickle as _p
    with open(out, 'rb') as f:
        data = _p.load(f)
    assert isinstance(data['q_network_state_dict']['layer.weight'], list)
    assert not hasattr(data['some_meta'], 'item')  # should be native python float or scalar
