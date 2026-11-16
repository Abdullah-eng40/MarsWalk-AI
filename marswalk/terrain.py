"""MOLA crop, terrain derivatives, and illustrative risk index.

Scientific notes
----------------
- Source: MGS MOLA MEGDR 16 px/degree (global topography + observation counts).
- MOLA altimetry acquisition ended 30 June 2001.  This is historical data, NOT
  current Mars environmental conditions.
- The MEGDR global file covers 0–360° E, 90° N to 90° S, 16 pixels per degree.
- Crop uses the documented grid convention; no resampling is performed.
- Risk weights and normalization caps are ILLUSTRATIVE demo values only.  They
  are not NASA-validated astronaut or rover safety thresholds.
- The ~3.7 km/pixel grid cannot detect individual boulders or certify a human-
  or rover-scale traverse.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

MARS_RADIUS_M: float = 3_389_500.0
PPD: int = 16  # pixels per degree for the selected MEGDR product
EXPECTED_WIDTH: int = 360 * PPD   # 5760
EXPECTED_HEIGHT: int = 180 * PPD  # 2880

# Illustrative normalisation caps — NOT validated safety thresholds.
SLOPE_CAP_DEG: float = 20.0
ROUGHNESS_CAP_M: float = 75.0
RELIEF_CAP_M: float = 150.0


def load_jezero_crop(
    raw_dir: Path,
    west: float = 77.0,
    east: float = 78.0,
    south: float = 17.5,
    north: float = 19.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Crop the MOLA MEGDR global grids to the Jezero study window.

    Parameters
    ----------
    raw_dir:
        Directory containing ``megt90n000eb.img`` and ``megc90n000eb.img``
        together with their ENVI ``.hdr`` sidecars.
    west, east, south, north:
        Bounding box in decimal degrees (longitude 0–360° E; latitude –90–90° N).

    Returns
    -------
    z : float32 ndarray, shape (rows, cols)
        Elevation in metres.  Invalid cells are ``nan``.
    valid : bool ndarray, shape (rows, cols)
        True where counts > 0 and elevation is not nodata.
    metadata : dict
        Provenance record for display and archival.

    Raises
    ------
    FileNotFoundError
        If the IMG or HDR files are missing.  Run ``python -m ingestion.fetch_mola``.
    ValueError
        If grid dimensions, alignment, or observed-cell count is unexpected.
    """
    topo_path = raw_dir / "megt90n000eb.img"
    count_path = raw_dir / "megc90n000eb.img"

    for p in (topo_path, count_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required MOLA file not found: {p}. "
                "Run:  python -m ingestion.fetch_mola"
            )

    with rasterio.open(topo_path) as topo_ds, rasterio.open(count_path) as count_ds:
        # --- Grid dimension checks ---
        if (topo_ds.width, topo_ds.height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
            raise ValueError(
                f"Topography grid is {topo_ds.width}×{topo_ds.height}; "
                f"expected {EXPECTED_WIDTH}×{EXPECTED_HEIGHT} for 16 px/degree. "
                "Inspect the PDS ENVI header; do not guess dimensions."
            )
        if (count_ds.width, count_ds.height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
            raise ValueError(
                f"Counts grid is {count_ds.width}×{count_ds.height}; "
                f"expected {EXPECTED_WIDTH}×{EXPECTED_HEIGHT}."
            )
        if topo_ds.width != count_ds.width or topo_ds.height != count_ds.height:
            raise ValueError("Topography and counts grids have different dimensions.")

        nodata = topo_ds.nodata          # 0.0 per ENVI header; may be None
        envi_dtype = str(topo_ds.dtypes[0])
        crs_wkt = topo_ds.crs.to_wkt() if topo_ds.crs else "unknown"

        # --- Crop window (0–360° E, 90° N–90° S convention) ---
        # Column index: 0 = 0° E; row index: 0 = 90° N
        c0 = max(0, int(np.floor(west  * PPD)))
        c1 = min(EXPECTED_WIDTH,  int(np.ceil(east  * PPD)))
        r0 = max(0, int(np.floor((90.0 - north) * PPD)))
        r1 = min(EXPECTED_HEIGHT, int(np.ceil((90.0 - south) * PPD)))

        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        z      = topo_ds.read(1, window=window).astype("float32")
        counts = count_ds.read(1, window=window)
        transform = topo_ds.window_transform(window)

    # --- Validity mask: counts > 0 AND not nodata ---
    valid: np.ndarray = np.isfinite(z) & (counts > 0)
    if nodata is not None:
        valid &= (z != nodata)
    z[~valid] = np.nan

    n_valid = int(valid.sum())
    if n_valid < 25:
        raise ValueError(
            f"Only {n_valid} observed cells in the Jezero crop. "
            "Inspect source labels and counts grid."
        )

    # --- Provenance metadata ---
    metadata: dict = {
        "source": "MGS MOLA MEGDR 16 px/degree — global topography and observation counts",
        "mola_data_note": (
            "MOLA altimetry ended 30 June 2001. Historical data only; "
            "not current Mars environmental conditions."
        ),
        "source_url": "https://pds-geosciences.wustl.edu/missions/mgs/megdr.html",
        "topography_file": "megt90n000eb.img",
        "counts_file": "megc90n000eb.img",
        "envi_dtype": envi_dtype,
        "rasterio_nodata": float(nodata) if nodata is not None else None,
        "nodata_note": (
            "Cells masked where counts == 0 (no altimetry observation) "
            "or where elevation equals the nodata fill value."
        ),
        "crop_bounds_lon_east_lat": [west, south, east, north],
        "pixels_per_degree": PPD,
        "native_grid_shape": [EXPECTED_HEIGHT, EXPECTED_WIDTH],
        "crop_shape": list(z.shape),
        "valid_cells": n_valid,
        "total_cells": int(valid.size),
        "units": "metres (MOLA topography product; verify product label)",
        "processing": "native grid crop; no resampling performed",
        "risk_model_note": (
            "Risk weights and normalisation caps are illustrative engineering "
            "choices, not NASA safety standards."
        ),
        "crs": crs_wkt,
    }
    return z, valid, metadata


def _nanstd(values: np.ndarray) -> float:
    """Return std of finite values, or nan if none."""
    v = values[np.isfinite(values)]
    return float(np.std(v)) if v.size else np.nan


def _nanrange(values: np.ndarray) -> float:
    """Return max−min of finite values, or nan if none."""
    v = values[np.isfinite(values)]
    return float(np.max(v) - np.min(v)) if v.size else np.nan


def terrain_features(
    z: np.ndarray,
    valid: np.ndarray,
    mean_lat_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute slope, roughness, and local relief from the DEM.

    Parameters
    ----------
    z : float32 ndarray
        Elevation in metres; invalid cells should be ``nan``.
    valid : bool ndarray
        Validity mask (same shape as *z*).
    mean_lat_deg : float
        Mean latitude of the crop in degrees (used for E–W pixel spacing).

    Returns
    -------
    slope_deg : float32 ndarray
        Gradient magnitude in degrees (0–90°).
    roughness_m : float32 ndarray
        3×3 neighbourhood standard deviation of elevation in metres.
    local_relief_m : float32 ndarray
        3×3 neighbourhood max − min elevation in metres.

    Notes
    -----
    Invalid cells are filled with the local median before computing
    derivatives (to avoid NaN propagation), then masked back to ``nan``
    in the output.  This local fill is NOT added to the DEM — it is a
    computational artefact only.
    """
    if z.shape != valid.shape:
        raise ValueError(
            f"DEM shape {z.shape} and valid mask shape {valid.shape} differ."
        )

    # Approximate pixel spacing in metres (equirectangular projection)
    lat_step_m = np.deg2rad(1.0 / PPD) * MARS_RADIUS_M
    lon_step_m = lat_step_m * np.cos(np.deg2rad(mean_lat_deg))

    if not np.isfinite(z[valid]).any():
        raise ValueError("No finite elevation values in valid DEM cells.")

    # Fill invalid cells with local median to stabilise derivatives.
    # Outputs are masked back to nan for invalid cells below.
    median_fill = float(np.nanmedian(z[valid]))
    work = np.where(valid, z, median_fill)

    # Slope (gradient magnitude → degrees)
    dz_dy, dz_dx = np.gradient(work, lat_step_m, lon_step_m)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    # Roughness: 3×3 std via fast uniform_filter approach (10× faster than
    # generic_filter with np.std on typical crop sizes).
    work_f64 = work.astype("float64")
    sq_mean = uniform_filter(work_f64 ** 2, size=3, mode="nearest")
    mean_sq = uniform_filter(work_f64,      size=3, mode="nearest") ** 2
    roughness_m = np.sqrt(np.maximum(sq_mean - mean_sq, 0.0))

    # Local relief: 3×3 max − min  (generic_filter with lambda is correct
    # but slow; use max_filter − min_filter from scipy for speed)
    from scipy.ndimage import maximum_filter, minimum_filter  # noqa: PLC0415
    local_relief_m = (
        maximum_filter(work, size=3, mode="nearest")
        - minimum_filter(work, size=3, mode="nearest")
    )

    # Mask invalid cells
    for arr in (slope_deg, roughness_m, local_relief_m):
        arr[~valid] = np.nan

    return (
        slope_deg.astype("float32"),
        roughness_m.astype("float32"),
        local_relief_m.astype("float32"),
    )


def risk_index(
    slope_deg: np.ndarray,
    roughness_m: np.ndarray,
    local_relief_m: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Compute an illustrative, unvalidated terrain-risk heuristic (0–100).

    Formula
    -------
    risk = 0.50 × slope_score + 0.30 × roughness_score + 0.20 × relief_score

    where each sub-score is clipped to [0, 100] after linear normalisation
    against the caps defined in this module.  Invalid cells → nan.

    Warning
    -------
    This risk model is a **demo heuristic** only.  The weights, caps, and
    formula have NOT been validated against operational astronaut or rover
    safety standards.  MOLA's ~3.7 km/pixel grid cannot certify traversability.
    """
    slope_score   = np.clip(slope_deg    / SLOPE_CAP_DEG   * 100.0, 0.0, 100.0)
    rough_score   = np.clip(roughness_m  / ROUGHNESS_CAP_M * 100.0, 0.0, 100.0)
    relief_score  = np.clip(local_relief_m / RELIEF_CAP_M  * 100.0, 0.0, 100.0)

    risk = (
        0.50 * slope_score
        + 0.30 * rough_score
        + 0.20 * relief_score
    )
    risk = np.asarray(risk, dtype="float32")
    risk[~valid] = np.nan
    return risk


def save_processed(out_dir: Path, arrays: dict, metadata: dict) -> None:
    """Persist terrain arrays and provenance metadata to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "jezero_terrain.npz", **arrays)
    (out_dir / "provenance.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
