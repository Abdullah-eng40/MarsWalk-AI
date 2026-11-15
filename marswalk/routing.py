"""Grid-based Dijkstra and A* route comparison."""
import heapq
import math
import numpy as np

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1)]

def _step_distance(dr, dc, lat_deg, ppd=16):
    dy = math.radians(1 / ppd) * 3_389_500.0
    dx = dy * math.cos(math.radians(lat_deg))
    return math.hypot(dr * dy, dc * dx)

def find_path(risk, valid, start, goal, mode="risk", mean_lat_deg=18.5, risk_aversion=4.0):
    """Return (path[(row,col)], cost); mode is 'distance' or 'risk'."""
    risk = np.asarray(risk, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(risk)
    h, w = risk.shape
    for point in (start, goal):
        r, c = point
        if not (0 <= r < h and 0 <= c < w) or not valid[r, c]:
            raise ValueError(f"Endpoint {point} is outside valid terrain.")
    def heuristic(a):
        return _step_distance(goal[0]-a[0], goal[1]-a[1], mean_lat_deg)
    frontier = [(heuristic(start) if mode == "risk" else 0.0, 0.0, start)]
    came_from = {}
    best = {start: 0.0}
    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if cost != best.get(current):
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, cost
        for dr, dc in NEIGHBOURS:
            nxt = (current[0] + dr, current[1] + dc)
            rr, cc = nxt
            if not (0 <= rr < h and 0 <= cc < w and valid[rr, cc]):
                continue
            distance = _step_distance(dr, dc, mean_lat_deg)
            if mode == "distance":
                edge = distance
            elif mode == "risk":
                edge = distance * (1.0 + risk_aversion * ((risk[current] + risk[nxt]) / 200.0))
            else:
                raise ValueError("mode must be 'distance' or 'risk'")
            new_cost = cost + edge
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                came_from[nxt] = current
                priority = new_cost + (heuristic(nxt) if mode == "risk" else 0.0)
                heapq.heappush(frontier, (priority, new_cost, nxt))
    raise ValueError("No traversable path connects the selected endpoints.")
