import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import numpy as np
from src.utils.seeding import set_global_seeds


def test_set_global_seeds_reproducible():
    s = 12345
    set_global_seeds(s)
    a = [random.random() for _ in range(5)]
    b = np.random.rand(5).tolist()

    # reset and re-seed
    set_global_seeds(s)
    a2 = [random.random() for _ in range(5)]
    b2 = np.random.rand(5).tolist()

    assert a == a2
    assert b == b2
