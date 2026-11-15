"""MOLA crop, terrain derivatives, and illustrative risk index."""
from pathlib import Path
import json
import numpy as np
import rasterio
from scipy.ndimage import generic_filter

MARS_RADIUS_M = 3_389_500.0
PPD = 16
# Global MEGDR convention: top row starts at 90N; columns start at 0E.
EXPECTED_WIDTH, EXPECTED_HEIGHT = 360 * PPD, 180 * PPD

def load_jezero_crop(raw_dir: Path, west=77.0, east=78.0, south=17.5, north=19.5):
    topo_path = raw_dir / "megt90n000eb.img"
    count_path = raw_dir / "megc90n000eb.img"
    with rasterio.open(topo_path) as topo_ds, rasterio.open(count_path) as count_ds:
        if (topo_ds.width, topo_ds.height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
            raise ValueError("Unexpected topography grid dimensions.")
        if (count_ds.width, count_ds.height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
            raise ValueError("Unexpected counts grid dimensions.")
        if topo_ds.width != count_ds.width or topo_ds.height != count_ds.height:
            raise ValueError("Topography and counts grids are not aligned.")
        # The selected MEGDR global file covers 0..360E, 90N..90S at 16 px/degree.
        # Bounds are snapped outward to whole source cells; no resampling occurs.
        c0 = max(0, int(np.floor(west * PPD)))
        c1 = min(EXPECTED_WIDTH, int(np.ceil(east * PPD)))
        r0 = max(0, int(np.floor((90.0 - north) * PPD)))
        r1 = min(EXPECTED_HEIGHT, int(np.ceil((90.0 - south) * PPD)))
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        z = topo_ds.read(1, window=window).astype("float32")
        counts = count_ds.read(1, window=window)
        transform = topo_ds.window_transform(window)
        nodata = topo_ds.nodata
        profile = topo_ds.profile.copy()

    valid = np.isfinite(z) & (counts > 0)
    if nodata is not None:
        valid &= z != nodata
    z[~valid] = np.nan
    if valid.sum() < 25:
        raise ValueError("Too few observed cells in Jezero crop; inspect source labels and counts.")
    metadata = {
        "source": "MGS MOLA MEGDR 16 px/degree, global topography + counts",
        "topography_file": topo_path.name,
        "counts_file": count_path.name,
        "source_url": "https://pds-geosciences.wustl.edu/missions/mgs/megdr.html",
        "crop_bounds_lon_east_lat": [west, south, east, north],
        "pixels_per_degree": PPD,
        "native_grid_shape": [EXPECTED_HEIGHT, EXPECTED_WIDTH],
        "crop_shape": list(z.shape),
        "valid_cells": int(valid.sum()),
        "total_cells": int(valid.size),
        "units": "metres (MOLA topography product; verify product label)",
        "nodata_source": "counts grid == 0, plus dataset nodata if supplied",
        "processing": "native grid crop; no resampling",
    }
    return z, valid, metadata

def _nanstd(values):
    v = values[np.isfinite(values)]
    return float(np.std(v)) if v.size else np.nan

def _nanrange(values):
    v = values[np.isfinite(values)]
    return float(np.max(v) - np.min(v)) if v.size else np.nan

def terrain_features(z: np.ndarray, valid: np.ndarray, mean_lat_deg: float):
    """Calculate derivatives on local equirectangular spacing; z is metres."""
    if z.shape != valid.shape:
        raise ValueError("DEM and valid mask shapes differ.")
    lat_step = np.deg2rad(1.0 / PPD) * MARS_RADIUS_M
    lon_step = np.deg2rad(1.0 / PPD) * MARS_RADIUS_M * np.cos(np.deg2rad(mean_lat_deg))
    filled = np.where(valid, z, np.nan)
    # Fill invalid cells only for stable derivative/filter calculation; mask outputs
    # back to the original valid cells. The interpolation is local and not data.
    if not np.isfinite(filled).any():
        raise ValueError("No valid DEM cells.")
    median = float(np.nanmedian(filled))
    work = np.where(np.isfinite(filled), filled, median)
    dz_dy, dz_dx = np.gradient(work, lat_step, lon_step)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    roughness = generic_filter(work, np.std, size=3, mode="nearest")
    relief = generic_filter(work, lambda a: np.max(a) - np.min(a), size=3, mode="nearest")
    for arr in (slope_deg, roughness, relief):
        arr[~valid] = np.nan
    return slope_deg.astype("float32"), roughness.astype("float32"), relief.astype("float32")

def risk_index(slope_deg, roughness_m, local_relief_m, valid):
    """Illustrative, unvalidated heuristic. All components are clipped 0..100."""
    slope_score = np.clip(slope_deg / 20.0 * 100.0, 0, 100)
    rough_score = np.clip(roughness_m / 75.0 * 100.0, 0, 100)
    relief_score = np.clip(local_relief_m / 150.0 * 100.0, 0, 100)
    risk = 0.50 * slope_score + 0.30 * rough_score + 0.20 * relief_score
    risk = np.asarray(risk, dtype="float32")
    risk[~valid] = np.nan
    return risk

def save_processed(out_dir: Path, arrays: dict, metadata: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "jezero_terrain.npz", **arrays)
    (out_dir / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
