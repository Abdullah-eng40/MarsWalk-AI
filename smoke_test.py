from pathlib import Path
from marswalk.terrain import load_jezero_crop, terrain_features, risk_index
from marswalk.routing import find_path
import numpy as np

RAW = Path("data/raw")
print("Loading Jezero crop...")
z, valid, meta = load_jezero_crop(RAW)
print(f"  Crop shape: {z.shape}  Valid cells: {valid.sum()}/{valid.size}")
print(f"  Elevation range: {np.nanmin(z):.1f} to {np.nanmax(z):.1f} m")

bounds = meta["crop_bounds_lon_east_lat"]
mean_lat = (bounds[1] + bounds[3]) / 2
print(f"  Mean latitude: {mean_lat:.2f} deg")

print("Computing terrain features...")
slope, rough, relief = terrain_features(z, valid, mean_lat)
risk = risk_index(slope, rough, relief, valid)
print(f"  Slope range: {np.nanmin(slope):.3f} - {np.nanmax(slope):.3f} deg")
print(f"  Roughness range: {np.nanmin(rough):.1f} - {np.nanmax(rough):.1f} m")
print(f"  Relief range: {np.nanmin(relief):.1f} - {np.nanmax(relief):.1f} m")
print(f"  Risk range: {np.nanmin(risk):.1f} - {np.nanmax(risk):.1f}")
print(f"  Mean risk: {np.nanmean(risk):.2f}/100")

# Assertions
assert np.nanmin(slope) >= 0 and np.nanmax(slope) <= 90, "Slope out of [0,90] deg"
assert np.nanmin(risk) >= 0 and np.nanmax(risk) <= 100, "Risk out of [0,100]"
assert valid.sum() >= 25, "Too few valid cells"

print("Finding routes...")
pts = [(int(r), int(c)) for r, c in zip(*np.where(valid))]
start, goal = pts[0], pts[-1]
shortest, dc = find_path(risk, valid, start, goal, mode="distance", mean_lat_deg=mean_lat)
safer,    rc = find_path(risk, valid, start, goal, mode="risk",     mean_lat_deg=mean_lat)
print(f"  Start: {start}  Goal: {goal}")
print(f"  Shortest path: {len(shortest)} cells  cost={dc/1000:.2f} km-eq")
print(f"  Risk-weighted: {len(safer)} cells  cost={rc/1000:.2f} km-eq")
print(f"  Shortest mean risk : {np.mean([risk[r,c] for r,c in shortest]):.1f}")
print(f"  Risk-wt  mean risk : {np.mean([risk[r,c] for r,c in safer]):.1f}")

assert shortest[0] == start and shortest[-1] == goal, "Shortest path endpoints wrong"
assert safer[0] == start and safer[-1] == goal, "Risk-wt path endpoints wrong"
assert all(valid[r, c] for r, c in shortest), "Invalid cell in shortest path"
assert all(valid[r, c] for r, c in safer), "Invalid cell in risk-wt path"

print("\nAll real-data smoke tests PASSED.")
