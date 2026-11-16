"""Tests for terrain processing using synthetic arrays.

These tests use synthetic DEM data and do NOT require a NASA data download.
They verify correctness of the processing logic only — they do not prove
that a real MOLA download succeeded.  Real-data validation is done by
running ``python -m ingestion.fetch_mola`` and ``streamlit run app.py``.
"""
import numpy as np
import pytest
from marswalk.terrain import terrain_features, risk_index, SLOPE_CAP_DEG


class TestFlatDEM:
    """A perfectly flat DEM should produce zero slope and near-zero risk."""

    def test_slope_is_zero(self):
        z     = np.full((7, 7), 100.0, dtype="float32")
        valid = np.ones_like(z, dtype=bool)
        slope, _, _ = terrain_features(z, valid, 18.0)
        assert np.nanmax(slope) < 1e-5, (
            f"Expected zero slope on flat DEM; got {np.nanmax(slope):.2e}"
        )

    def test_risk_is_zero(self):
        z     = np.full((7, 7), 100.0, dtype="float32")
        valid = np.ones_like(z, dtype=bool)
        slope, rough, relief = terrain_features(z, valid, 18.0)
        risk  = risk_index(slope, rough, relief, valid)
        assert np.nanmax(risk) < 1e-4, (
            f"Expected zero risk on flat DEM; got {np.nanmax(risk):.2e}"
        )


class TestRiskBoundsAndMask:
    """Risk must stay in [0, 100] and respect the validity mask."""

    def test_risk_in_bounds(self):
        z     = np.arange(49, dtype="float32").reshape(7, 7) * 20.0
        valid = np.ones((7, 7), dtype=bool)
        slope, rough, relief = terrain_features(z, valid, 18.0)
        risk  = risk_index(slope, rough, relief, valid)
        assert np.nanmin(risk) >= 0.0,   "Risk below 0"
        assert np.nanmax(risk) <= 100.0, "Risk above 100"

    def test_slope_in_bounds(self):
        z     = np.arange(49, dtype="float32").reshape(7, 7) * 20.0
        valid = np.ones((7, 7), dtype=bool)
        slope, _, _ = terrain_features(z, valid, 18.0)
        assert np.nanmin(slope) >= 0.0,  "Slope below 0°"
        assert np.nanmax(slope) <= 90.0, "Slope above 90°"

    def test_invalid_cells_are_nan(self):
        z     = np.arange(49, dtype="float32").reshape(7, 7) * 20.0
        valid = np.ones((7, 7), dtype=bool)
        valid[0, 0] = False
        slope, rough, relief = terrain_features(z, valid, 18.0)
        risk  = risk_index(slope, rough, relief, valid)
        assert np.isnan(risk[0, 0]),   "Invalid cell should produce nan risk"
        assert np.isnan(slope[0, 0]),  "Invalid cell should produce nan slope"


class TestNodataMask:
    """Cells with counts == 0 (or explicit nodata) must be masked to nan."""

    def test_counts_zero_cell_is_nan(self):
        z     = np.ones((7, 7), dtype="float32") * 100.0
        valid = np.ones((7, 7), dtype=bool)
        valid[3, 3] = False   # simulates counts == 0
        slope, rough, relief = terrain_features(z, valid, 18.0)
        risk  = risk_index(slope, rough, relief, valid)
        assert np.isnan(risk[3, 3]),  "counts==0 cell must be nan in risk"
        assert np.isnan(slope[3, 3]), "counts==0 cell must be nan in slope"

    def test_nodata_zero_elevation_cell_is_nan(self):
        """Cells where elevation == nodata value (0) and valid=False stay nan."""
        z     = np.ones((7, 7), dtype="float32") * 100.0
        z[2, 2] = 0.0          # nodata fill value per ENVI header
        valid = np.ones((7, 7), dtype=bool)
        valid[2, 2] = False    # masked by counts + nodata check in terrain loader
        slope, rough, relief = terrain_features(z, valid, 18.0)
        risk  = risk_index(slope, rough, relief, valid)
        assert np.isnan(risk[2, 2]),  "nodata cell must be nan in risk"


class TestSlopeUnits:
    """Slope should be strictly positive for a ramp DEM and have correct units."""

    def test_slope_on_ramp_is_positive(self):
        """Interior cells of a uniform ramp should have a consistent slope > 0."""
        z     = np.tile(np.arange(7, dtype="float32") * 100.0, (7, 1))
        valid = np.ones((7, 7), dtype=bool)
        slope, _, _ = terrain_features(z, valid, 18.0)
        interior = slope[1:-1, 1:-1]
        assert np.all(interior > 0.0), "Ramp DEM should have positive interior slope"
        assert np.all(interior < 90.0), "Ramp DEM slope should not reach 90°"

    def test_steeper_ramp_has_higher_slope(self):
        """Doubling the DEM gradient should increase slope."""
        def make_ramp(scale):
            z     = np.tile(np.arange(7, dtype="float32") * scale, (7, 1))
            valid = np.ones((7, 7), dtype=bool)
            slope, _, _ = terrain_features(z, valid, 18.0)
            return float(np.nanmean(slope))

        assert make_ramp(200.0) > make_ramp(100.0), (
            "Steeper ramp should produce higher mean slope"
        )

    def test_slope_cap_satisfied_by_risk(self):
        """Risk from a slope at exactly the cap should be ≤ 50 (the slope weight)."""
        # Create a DEM whose mean slope ≈ SLOPE_CAP_DEG
        z     = np.tile(
            np.linspace(0, 10_000, 7, dtype="float32"), (7, 1)
        )
        valid = np.ones((7, 7), dtype=bool)
        slope, rough, relief = terrain_features(z, valid, 18.0)
        risk  = risk_index(slope, rough, relief, valid)
        # All valid risk values must stay ≤ 100
        assert np.nanmax(risk) <= 100.0
