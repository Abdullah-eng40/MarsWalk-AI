"""Grid-based Dijkstra (shortest-distance) and risk-weighted A* route planner.

Both algorithms traverse an 8-connected grid of valid MOLA terrain cells.
Step distances are computed in metres using Mars mean radius and the cosine
of the mean crop latitude.

Scientific note
---------------
Routes are algorithmic demonstrations on a coarse regional DEM (~3.7 km/pixel).
They do not represent verified traverses or account for vehicle dynamics,
energy, radiation, communications, or sub-grid obstacles.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

MARS_RADIUS_M: float = 3_389_500.0

# 8-connected neighbourhood offsets (Δrow, Δcol)
NEIGHBOURS: list[tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),           ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def _step_distance(dr: int, dc: int, lat_deg: float, ppd: int = 16) -> float:
    """Return approximate step distance in metres on a Mars equirectangular grid."""
    dy = math.radians(1.0 / ppd) * MARS_RADIUS_M
    dx = dy * math.cos(math.radians(lat_deg))
    return math.hypot(dr * dy, dc * dx)


def find_path(
    risk: np.ndarray,
    valid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    mode: str = "risk",
    mean_lat_deg: float = 18.5,
    risk_aversion: float = 4.0,
) -> tuple[list[tuple[int, int]], float]:
    """Find a path from *start* to *goal* through valid terrain cells.

    Parameters
    ----------
    risk : ndarray, shape (H, W)
        Risk index (0–100 or nan for invalid cells).
    valid : bool ndarray, shape (H, W)
        True where cells are traversable.
    start, goal : (row, col)
        Grid coordinates of the endpoints.  Both must lie in valid cells.
    mode : ``'distance'`` | ``'risk'``
        ``'distance'``  — Dijkstra shortest-distance path.
        ``'risk'``      — A* with edge cost scaled by cell risk.
    mean_lat_deg : float
        Mean latitude of the crop in decimal degrees.
    risk_aversion : float
        Scale factor for risk penalty in ``'risk'`` mode.
        Edge cost = distance × (1 + risk_aversion × mean_cell_risk / 100).

    Returns
    -------
    path : list of (row, col)
        Ordered list of grid cells from start to goal (inclusive).
    cost : float
        Planner objective cost (metres in distance mode; scaled in risk mode).

    Raises
    ------
    ValueError
        If an endpoint is outside grid bounds, in an invalid cell,
        or no traversable path exists.
    """
    risk  = np.asarray(risk,  dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(risk)
    h, w  = risk.shape

    # --- Validate endpoints ---
    for label, point in (("start", start), ("goal", goal)):
        r, c = point
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError(
                f"Endpoint {label} {point} is outside the grid ({h}×{w})."
            )
        if not valid[r, c]:
            raise ValueError(
                f"Endpoint {label} {point} is not in a valid (traversable) cell. "
                "Choose a cell where both counts > 0 and risk is finite."
            )

    # --- Trivial case ---
    if start == goal:
        return [start], 0.0

    # --- Precompute step distances for all 8 directions ---
    step_dist: dict[tuple[int, int], float] = {
        (dr, dc): _step_distance(dr, dc, mean_lat_deg)
        for dr, dc in NEIGHBOURS
    }

    def heuristic(a: tuple[int, int]) -> float:
        return _step_distance(goal[0] - a[0], goal[1] - a[1], mean_lat_deg)

    # Priority queue entries: (priority, cost_so_far, node)
    # Separate priority and cost avoids ties in the comparison on (cost, node).
    h0 = heuristic(start) if mode == "risk" else 0.0
    frontier: list[tuple[float, float, tuple[int, int]]] = [
        (h0, 0.0, start)
    ]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best: dict[tuple[int, int], float] = {start: 0.0}

    while frontier:
        _, cost, current = heapq.heappop(frontier)

        # Stale entry guard (lazy deletion)
        if cost > best.get(current, float("inf")) + 1e-9:
            continue

        if current == goal:
            # Reconstruct path
            path: list[tuple[int, int]] = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, cost

        cr, cc = current
        for dr, dc in NEIGHBOURS:
            nxt = (cr + dr, cc + dc)
            nr, nc = nxt
            if not (0 <= nr < h and 0 <= nc < w and valid[nr, nc]):
                continue

            dist = step_dist[(dr, dc)]
            if mode == "distance":
                edge = dist
            elif mode == "risk":
                mean_risk = (risk[cr, cc] + risk[nr, nc]) / 200.0  # 0–1 scale
                edge = dist * (1.0 + risk_aversion * mean_risk)
            else:
                raise ValueError(f"mode must be 'distance' or 'risk', got {mode!r}")

            new_cost = cost + edge
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                came_from[nxt] = current
                priority = new_cost + (heuristic(nxt) if mode == "risk" else 0.0)
                heapq.heappush(frontier, (priority, new_cost, nxt))

    raise ValueError(
        f"No traversable path connects {start} to {goal}. "
        "The two endpoints may be in disconnected valid-cell regions. "
        "Try different start/destination cells."
    )
