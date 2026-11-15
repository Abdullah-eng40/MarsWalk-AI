import numpy as np
import pytest
from marswalk.routing import find_path

def test_routes_connect_valid_endpoints():
    risk = np.zeros((8, 8), dtype=float)
    valid = np.ones((8, 8), dtype=bool)
    p, cost = find_path(risk, valid, (0, 0), (7, 7), mode="risk")
    assert p[0] == (0, 0) and p[-1] == (7, 7)
    assert all(valid[r, c] for r, c in p)
    assert cost > 0

def test_invalid_endpoint_rejected():
    risk = np.zeros((4, 4), dtype=float)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    with pytest.raises(ValueError):
        find_path(risk, valid, (0, 0), (3, 3))

def test_blocked_corridor_no_path():
    risk = np.zeros((5, 5), dtype=float)
    valid = np.ones((5, 5), dtype=bool)
    valid[:, 2] = False
    with pytest.raises(ValueError):
        find_path(risk, valid, (2, 0), (2, 4))
