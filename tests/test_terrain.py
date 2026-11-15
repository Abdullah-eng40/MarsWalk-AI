import numpy as np
from marswalk.terrain import terrain_features, risk_index

def test_flat_dem_has_zero_slope_and_risk():
    z = np.full((7, 7), 100.0, dtype=np.float32)
    valid = np.ones_like(z, dtype=bool)
    slope, rough, relief = terrain_features(z, valid, 18.0)
    risk = risk_index(slope, rough, relief, valid)
    assert np.nanmax(slope) < 1e-5
    assert np.nanmax(risk) < 1e-4

def test_risk_bounds_and_invalid_mask():
    z = np.arange(49, dtype=np.float32).reshape(7, 7) * 20
    valid = np.ones_like(z, dtype=bool)
    valid[0, 0] = False
    slope, rough, relief = terrain_features(z, valid, 18.0)
    risk = risk_index(slope, rough, relief, valid)
    assert np.isnan(risk[0, 0])
    assert np.nanmin(risk) >= 0
    assert np.nanmax(risk) <= 100
    assert np.nanmin(slope) >= 0
    assert np.nanmax(slope) <= 90
