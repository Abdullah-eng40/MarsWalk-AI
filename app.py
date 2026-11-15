from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from marswalk.terrain import load_jezero_crop, terrain_features, risk_index, save_processed
from marswalk.routing import find_path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
HIRISE_BROWSE = "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/ESP/ORB_094200_094299/ESP_094258_1985/ESP_094258_1985_RED.abrowse.jpg"

st.set_page_config(page_title="MarsWalk AI | AstroNox", layout="wide")
st.title("🚀 MarsWalk AI")
st.caption("NASA-powered regional terrain intelligence prototype · Jezero Crater, Mars")

if not (RAW / "megt90n000eb.img").exists():
    st.warning("MOLA numerical terrain data not found. Download the NASA/PDS products first.")
    st.code("python -m ingestion.fetch_mola")
    st.markdown("[Open the PDS MOLA archive](https://pds-geosciences.wustl.edu/missions/mgs/megdr.html)")
    st.stop()

@st.cache_data(show_spinner="Loading and analyzing MOLA terrain…")
def compute():
    z, valid, meta = load_jezero_crop(RAW)
    mean_lat = np.mean(meta["crop_bounds_lon_east_lat"][1::2])
    # bounds order [west, south, east, north]; mean latitude:
    mean_lat = (meta["crop_bounds_lon_east_lat"][1] + meta["crop_bounds_lon_east_lat"][3]) / 2
    slope, rough, relief = terrain_features(z, valid, mean_lat)
    risk = risk_index(slope, rough, relief, valid)
    save_processed(PROCESSED, {"elevation_m": z, "valid": valid, "slope_deg": slope,
                               "roughness_m": rough, "local_relief_m": relief, "risk": risk}, meta)
    return z, valid, slope, rough, relief, risk, meta, mean_lat

try:
    z, valid, slope, rough, relief, risk, meta, mean_lat = compute()
except Exception as exc:
    st.error(f"Could not load/process the PDS MOLA product: {exc}")
    st.info("Check that the IMG, HDR and XML sidecars downloaded together and that the source product dimensions match the expected 16 px/degree grid. Do not guess the binary format.")
    st.stop()

valid_risk = risk[np.isfinite(risk)]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Valid MOLA cells", f"{valid.sum():,}")
c2.metric("Mean prototype risk", f"{np.nanmean(risk):.1f}/100")
c3.metric("Mean slope", f"{np.nanmean(slope):.2f}°")
c4.metric("Grid", f"{z.shape[1]} × {z.shape[0]} cells")

st.warning("MOLA is a coarse regional DEM (16 pixels/degree, ~3.7 km at the equator). This map cannot detect individual rocks or certify a human/rover traverse. Risk weights and thresholds are illustrative, not NASA safety standards.")

layer = st.selectbox("Terrain layer", ["Combined risk (0–100)", "Elevation (m)", "Slope (degrees)", "Roughness (m)", "Local relief (m)"])
layer_data = {
    "Combined risk (0–100)": (risk, "Risk score", "RdYlGn_r"),
    "Elevation (m)": (z, "Elevation (m)", "Earth"),
    "Slope (degrees)": (slope, "Slope (degrees)", "YlOrRd"),
    "Roughness (m)": (rough, "3×3 local std (m)", "Viridis"),
    "Local relief (m)": (relief, "3×3 local relief (m)", "Magma"),
}
arr, color_title, colorscale = layer_data[layer]
fig = go.Figure(go.Heatmap(z=arr, colorscale=colorscale, colorbar={"title": color_title},
                           hoverongaps=False))
fig.update_layout(title=f"Jezero regional terrain — {layer}", height=520,
                  xaxis_title="Grid column (eastward)", yaxis_title="Grid row (southward)",
                  yaxis_autorange="reversed", margin=dict(l=20, r=20, t=55, b=20))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Risk-aware route planner")
st.caption("Choose grid cells, not latitude/longitude. One grid step is approximately 3–4 km at this latitude.")
rows, cols = z.shape
valid_points = np.argwhere(valid)
if len(valid_points) < 2:
    st.error("Not enough valid terrain cells for route planning.")
    st.stop()
default_start = tuple(valid_points[0])
default_goal = tuple(valid_points[-1])
left, right = st.columns(2)
with left:
    start_r = st.number_input("Start row", 0, rows-1, int(default_start[0]))
    start_c = st.number_input("Start column", 0, cols-1, int(default_start[1]))
with right:
    goal_r = st.number_input("Destination row", 0, rows-1, int(default_goal[0]))
    goal_c = st.number_input("Destination column", 0, cols-1, int(default_goal[1]))

if st.button("Calculate route alternatives", type="primary"):
    start, goal = (int(start_r), int(start_c)), (int(goal_r), int(goal_c))
    try:
        shortest, dist_cost = find_path(risk, valid, start, goal, mode="distance", mean_lat_deg=mean_lat)
        safer, risk_cost = find_path(risk, valid, start, goal, mode="risk", mean_lat_deg=mean_lat)
        overlay = go.Figure(go.Heatmap(z=risk, colorscale="RdYlGn_r", zmin=0, zmax=100,
                                      colorbar={"title": "Risk"}))
        for path, name, color in [(shortest, "Shortest-distance", "white"), (safer, "Risk-weighted A*", "cyan")]:
            rr = [p[0] for p in path]; cc = [p[1] for p in path]
            overlay.add_trace(go.Scatter(x=cc, y=rr, mode="lines+markers", name=name,
                                        line=dict(color=color, width=3), marker=dict(size=5)))
        overlay.update_layout(height=520, yaxis_autorange="reversed",
                              xaxis_title="Grid column", yaxis_title="Grid row",
                              title="Route comparison over prototype risk surface")
        st.plotly_chart(overlay, use_container_width=True)
        def path_stats(path):
            vals = np.array([risk[r, c] for r, c in path])
            return {"cells": len(path), "mean risk": float(vals.mean()), "max risk": float(vals.max())}
        a, b = path_stats(shortest), path_stats(safer)
        st.dataframe(pd.DataFrame([
            {"Route": "Shortest-distance", **a, "planner objective cost (km-equivalent)": dist_cost/1000},
            {"Route": "Risk-weighted A*", **b, "planner objective cost (km-equivalent)": risk_cost/1000},
        ]), use_container_width=True, hide_index=True)
        st.markdown("**Mission note:** The risk-weighted route is an algorithmic alternative on the coarse MOLA grid. A lower score does not prove real-world safety.")
        st.success(f"Both paths connect {start} to {goal} through valid cells. Shortest path: {len(shortest)} cells; risk-weighted path: {len(safer)} cells.")
    except Exception as exc:
        st.error(f"Route could not be calculated: {exc}")

with st.expander("Data provenance and assumptions"):
    st.json(meta)
    st.markdown("""
    **NASA source:** [PDS Geosciences Node — MOLA MEGDR](https://pds-geosciences.wustl.edu/missions/mgs/megdr.html)  
    **Optical context:** [HiRISE ESP_094258_1985 product page](https://hirise.lpl.arizona.edu/ESP_094258_1985), acquired 2026-09-04; image credit NASA/JPL-Caltech/UArizona. It is not co-registered to the MOLA grid.

    Risk heuristic: 50% slope (0–20° normalization cap), 30% local roughness (0–75 m), 20% local relief (0–150 m). These are demo assumptions and need domain validation.
    """)
    if st.button("Show HiRISE browse image (visual context only)"):
        st.image(HIRISE_BROWSE, caption="HiRISE ESP_094258_1985 — NASA/JPL-Caltech/UArizona. Separate high-resolution orbital observation; not used in MOLA calculations.")
