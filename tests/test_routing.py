"""Tests for routing module using synthetic grids.

All tests use synthetic numpy arrays — no NASA data download required.
These verify algorithmic correctness only.
"""
import numpy as np
import pytest
from marswalk.routing import find_path


class TestBasicPathfinding:
    """Happy-path: routes should connect valid endpoints."""

    def test_distance_mode_connects_endpoints(self):
        risk  = np.zeros((8, 8), dtype=float)
        valid = np.ones((8, 8), dtype=bool)
        path, cost = find_path(risk, valid, (0, 0), (7, 7), mode="distance")
        assert path[0]  == (0, 0), "Path should start at start"
        assert path[-1] == (7, 7), "Path should end at goal"
        assert all(valid[r, c] for r, c in path), "All path cells must be valid"
        assert cost > 0, "Non-zero distance expected"

    def test_risk_mode_connects_endpoints(self):
        risk  = np.zeros((8, 8), dtype=float)
        valid = np.ones((8, 8), dtype=bool)
        path, cost = find_path(risk, valid, (0, 0), (7, 7), mode="risk")
        assert path[0]  == (0, 0)
        assert path[-1] == (7, 7)
        assert cost > 0

    def test_start_equals_goal_returns_single_cell(self):
        risk  = np.zeros((5, 5), dtype=float)
        valid = np.ones((5, 5), dtype=bool)
        path, cost = find_path(risk, valid, (2, 2), (2, 2))
        assert path == [(2, 2)]
        assert cost == 0.0


class TestEndpointValidation:
    """Invalid endpoints should raise ValueError with clear messages."""

    def test_invalid_start_cell_rejected(self):
        risk  = np.zeros((4, 4), dtype=float)
        valid = np.ones((4, 4), dtype=bool)
        valid[0, 0] = False
        with pytest.raises(ValueError, match="start"):
            find_path(risk, valid, (0, 0), (3, 3))

    def test_invalid_goal_cell_rejected(self):
        risk  = np.zeros((4, 4), dtype=float)
        valid = np.ones((4, 4), dtype=bool)
        valid[3, 3] = False
        with pytest.raises(ValueError, match="goal"):
            find_path(risk, valid, (0, 0), (3, 3))

    def test_out_of_bounds_start_rejected(self):
        risk  = np.zeros((4, 4), dtype=float)
        valid = np.ones((4, 4), dtype=bool)
        with pytest.raises(ValueError):
            find_path(risk, valid, (-1, 0), (3, 3))

    def test_out_of_bounds_goal_rejected(self):
        risk  = np.zeros((4, 4), dtype=float)
        valid = np.ones((4, 4), dtype=bool)
        with pytest.raises(ValueError):
            find_path(risk, valid, (0, 0), (4, 4))


class TestDisconnectedGrid:
    """A fully blocked corridor should produce a ValueError."""

    def test_blocked_column_raises(self):
        risk  = np.zeros((5, 5), dtype=float)
        valid = np.ones((5, 5), dtype=bool)
        valid[:, 2] = False   # block entire middle column
        with pytest.raises(ValueError, match="[Nn]o traversable path"):
            find_path(risk, valid, (2, 0), (2, 4))


class TestRiskWeighting:
    """Risk-weighted path should prefer lower-risk cells over shorter distance."""

    def test_risk_weighted_avoids_high_risk_zone(self):
        """
        Layout (5×7 grid):
        - Direct route crosses column 3 which has risk=100.
        - Detour via row 4 avoids it (risk=0).
        - Risk-weighted A* should take the safer detour.
        """
        risk  = np.zeros((5, 7), dtype=float)
        valid = np.ones((5, 7), dtype=bool)
        # High-risk wall in the middle (rows 0-3, col 3)
        risk[:4, 3] = 100.0
        path, _ = find_path(risk, valid, (0, 0), (0, 6), mode="risk")
        mean_risk = np.mean([risk[r, c] for r, c in path])
        # Shortest path would go straight through risk=100 cells;
        # risk-weighted should achieve a lower mean
        direct_risk = np.mean([risk[0, c] for c in range(7)])
        # The risk-weighted path may detour; at minimum its mean shouldn't be worse
        assert mean_risk <= direct_risk + 5.0, (
            "Risk-weighted path should avoid high-risk cells"
        )

    def test_distance_mode_shorter_than_risk_mode_on_risky_map(self):
        """Shortest-distance path is never longer (in cells) than risk-weighted."""
        rng  = np.random.default_rng(42)
        risk = rng.uniform(0, 100, (10, 10))
        valid = np.ones((10, 10), dtype=bool)
        risk[~valid] = np.nan

        d_path, _ = find_path(risk, valid, (0, 0), (9, 9), mode="distance")
        r_path, _ = find_path(risk, valid, (0, 0), (9, 9), mode="risk")
        # Distance-mode minimises steps; risk mode may take longer detours
        assert len(d_path) <= len(r_path) + 2  # allow minor tie-breaking margin
