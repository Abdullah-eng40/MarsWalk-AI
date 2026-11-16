"""MarsWalk AI — Streamlit application.

Displays regional terrain intelligence and risk-aware route comparison
for Jezero Crater, Mars, using NASA/PDS MOLA MEGDR 16 px/degree data.

Scientific constraints
----------------------
- MOLA is a coarse global DEM (~3.7 km/pixel). It cannot resolve individual
  boulders, small craters, or certify a human/rover traverse.
- Risk weights are illustrative; not NASA safety standards.
- MOLA altimetry ended 2001-06-30; this is historical, not live, data.
- HiRISE image is visual context only; it is NOT co-registered to MOLA.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from marswalk.terrain import (
    load_jezero_crop,
    terrain_features,
    risk_index,
    save_processed,
    MARS_RADIUS_M,
    PPD,
)
from marswalk.routing import find_path

ROOT      = Path(__file__).resolve().parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

HIRISE_BROWSE_URL = (
    "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/"
    "ESP/ORB_094200_094299/ESP_094258_1985/"
    "ESP_094258_1985_RED.abrowse.jpg"
)
HIRISE_LOCAL = ROOT / "data" / "hirise" / "ESP_094258_1985_RED.abrowse.jpg"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MarsWalk AI | AstroNox",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 MarsWalk AI")
st.caption(
    "NASA-powered regional terrain intelligence prototype · "
    "Jezero Crater, Mars · "
    "Built on MGS MOLA MEGDR 16 px/degree data"
)

# ---------------------------------------------------------------------------
# Sidebar — visual context and disclaimer
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Visual Context (HiRISE)")
    st.caption(
        "HiRISE ESP_094258_1985 — acquired 2026-09-04, centre 18.478°N 77.540°E, "
        "50 cm/pixel. **Not co-registered to MOLA.** Credit: NASA/JPL-Caltech/UArizona."
    )
    if st.button("Load HiRISE browse image", key="hirise_btn"):
        hirise_src = str(HIRISE_LOCAL) if HIRISE_LOCAL.exists() else HIRISE_BROWSE_URL
        st.image(
            hirise_src,
            caption=(
                "HiRISE ESP_094258_1985 — NASA/JPL-Caltech/UArizona\n"
                "High-resolution visual context only. NOT used in MOLA calculations."
            ),
            use_container_width=True,
        )

    st.markdown("---")
    st.header("About")
    st.markdown(
        "**MarsWalk AI** turns historical NASA planetary data into regional "
        "terrain-risk context and illustrative route alternatives.\n\n"
        "This is a 4-hour hackathon MVP — not an operational navigation system."
    )

# ---------------------------------------------------------------------------
# Data gate: prompt download if files are absent
# ---------------------------------------------------------------------------
topo_file   = RAW / "megt90n000eb.img"
counts_file = RAW / "megc90n000eb.img"

if not topo_file.exists() or not counts_file.exists():
    st.warning(
        "⚠️ MOLA numerical terrain data not found. "
        "Download the NASA/PDS products first, then restart the app."
    )
    st.info(
        "**Demo startup guide**\n\n"
        "1. Open a terminal in the project directory.\n"
        "2. Activate the virtual environment:  `.venv\\Scripts\\Activate.ps1`\n"
        "3. Download MOLA data (~66 MB):  `python -m ingestion.fetch_mola`\n"
        "4. Restart the app:  `streamlit run app.py`\n\n"
        "Optionally fetch the HiRISE browse image:  "
        "`python -m ingestion.fetch_hirise`"
    )
    st.markdown(
        "[Open the PDS MOLA archive]"
        "(https://pds-geosciences.wustl.edu/missions/mgs/megdr.html)"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Data loading and terrain processing (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and analysing MOLA terrain…")
def compute() -> tuple:
    z, valid, meta = load_jezero_crop(RAW)
    west, south, east, north = meta["crop_bounds_lon_east_lat"]
    mean_lat = (south + north) / 2.0
    slope, rough, relief = terrain_features(z, valid, mean_lat)
    risk = risk_index(slope, rough, relief, valid)
    save_processed(
        PROCESSED,
        {
            "elevation_m":   z,
            "valid":         valid,
            "slope_deg":     slope,
            "roughness_m":   rough,
            "local_relief_m": relief,
            "risk":          risk,
        },
        meta,
    )
    return z, valid, slope, rough, relief, risk, meta, mean_lat


try:
    z, valid, slope, rough, relief, risk, meta, mean_lat = compute()
except FileNotFoundError as exc:
    st.error(f"Required data file not found: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Could not load/process the PDS MOLA product: {exc}")
    st.info(
        "Verify that the IMG, HDR, and XML sidecars downloaded together "
        "and that the source product dimensions match the expected 16 px/degree grid. "
        "Do NOT guess the binary format."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
# Approximate cell spacing at mean latitude
cell_lon_km = math.radians(1.0 / PPD) * MARS_RADIUS_M / 1000.0
cell_lat_km = cell_lon_km  # equirectangular; same N–S
cell_ew_km  = cell_lon_km * math.cos(math.radians(mean_lat))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Valid MOLA cells",           f"{valid.sum():,}")
c2.metric("Mean prototype risk",        f"{np.nanmean(risk):.1f} / 100")
c3.metric("Mean slope",                 f"{np.nanmean(slope):.2f}°")
c4.metric("Cell spacing (N–S / E–W)",
          f"~{cell_lat_km:.1f} km / ~{cell_ew_km:.1f} km")

st.warning(
    "⚠️ **Scientific limitations:** MOLA is a coarse regional DEM "
    f"(16 pixels/degree; ~{cell_lat_km:.1f} km N–S per cell at this latitude). "
    "It **cannot** detect individual rocks or certify a human/rover traverse. "
    "Risk weights and thresholds are illustrative demo assumptions — "
    "not NASA-validated safety standards. "
    "MOLA altimetry ended 2001-06-30; this is historical data, not live conditions."
)

# ---------------------------------------------------------------------------
# Terrain layer visualisation
# ---------------------------------------------------------------------------
st.subheader("Terrain layer viewer")

LAYER_OPTIONS: dict[str, tuple[np.ndarray, str, str]] = {
    "Combined risk (0–100)":  (risk,   "Risk score",         "RdYlGn_r"),
    "Elevation (m)":           (z,      "Elevation (m)",      "Earth"),
    "Slope (degrees)":         (slope,  "Slope (°)",          "YlOrRd"),
    "Roughness (m)":           (rough,  "3×3 local std (m)",  "Viridis"),
    "Local relief (m)":        (relief, "3×3 relief (m)",     "Magma"),
}

layer_name = st.selectbox("Select terrain layer", list(LAYER_OPTIONS.keys()))
arr, color_title, colorscale = LAYER_OPTIONS[layer_name]

fig = go.Figure(
    go.Heatmap(
        z=arr,
        colorscale=colorscale,
        colorbar={"title": color_title},
        hoverongaps=False,
        hovertemplate=(
            "Row: %{y}<br>Col: %{x}<br>" + color_title + ": %{z:.2f}<extra></extra>"
        ),
    )
)
fig.update_layout(
    title=f"Jezero regional terrain — {layer_name}",
    height=520,
    xaxis_title="Grid column (east →)",
    yaxis_title="Grid row (south ↓)",
    yaxis_autorange="reversed",
    margin=dict(l=20, r=20, t=55, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Risk-aware route planner
# ---------------------------------------------------------------------------
st.subheader("Risk-aware route planner")
st.caption(
    f"Select grid cells (not lat/lon). "
    f"One step ≈ {cell_lat_km:.1f} km N–S / {cell_ew_km:.1f} km E–W at this latitude."
)

rows, cols = z.shape
valid_points = np.argwhere(valid)
if len(valid_points) < 2:
    st.error("Not enough valid terrain cells for route planning.")
    st.stop()

default_start = tuple(map(int, valid_points[len(valid_points) // 4]))
default_goal  = tuple(map(int, valid_points[len(valid_points) * 3 // 4]))

left, right = st.columns(2)
with left:
    st.markdown("**Start**")
    start_r = st.number_input("Start row",    0, rows - 1, default_start[0], key="sr")
    start_c = st.number_input("Start column", 0, cols - 1, default_start[1], key="sc")
with right:
    st.markdown("**Destination**")
    goal_r = st.number_input("Destination row",    0, rows - 1, default_goal[0], key="gr")
    goal_c = st.number_input("Destination column", 0, cols - 1, default_goal[1], key="gc")

if st.button("Calculate route alternatives", type="primary"):
    start = (int(start_r), int(start_c))
    goal  = (int(goal_r),  int(goal_c))

    try:
        with st.spinner("Computing shortest-distance path (Dijkstra)…"):
            shortest, dist_cost = find_path(
                risk, valid, start, goal, mode="distance", mean_lat_deg=mean_lat
            )
        with st.spinner("Computing risk-weighted path (A*)…"):
            safer, risk_cost = find_path(
                risk, valid, start, goal, mode="risk", mean_lat_deg=mean_lat
            )

        # --- Route overlay chart ---
        overlay = go.Figure(
            go.Heatmap(
                z=risk,
                colorscale="RdYlGn_r",
                zmin=0,
                zmax=100,
                colorbar={"title": "Risk"},
                hoverongaps=False,
            )
        )
        for path, name, color, dash in [
            (shortest, "Shortest-distance (Dijkstra)", "white",  "solid"),
            (safer,    "Risk-weighted A*",             "cyan",   "dot"),
        ]:
            rr = [p[0] for p in path]
            cc = [p[1] for p in path]
            overlay.add_trace(
                go.Scatter(
                    x=cc, y=rr,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=3, dash=dash),
                    marker=dict(size=5),
                )
            )
        # Mark start and goal
        overlay.add_trace(go.Scatter(
            x=[start[1], goal[1]], y=[start[0], goal[0]],
            mode="markers+text",
            text=["S", "G"],
            textposition="top center",
            marker=dict(color="yellow", size=12, symbol="star"),
            name="Waypoints",
        ))
        overlay.update_layout(
            height=540,
            yaxis_autorange="reversed",
            xaxis_title="Grid column",
            yaxis_title="Grid row",
            title="Route comparison over prototype risk surface",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(overlay, use_container_width=True)

        # --- Statistics table ---
        def path_stats(path: list, label: str, obj_cost: float) -> dict:
            vals = np.array([risk[r, c] for r, c in path])
            # True ground-distance estimate (metres → km)
            ground_km = sum(
                math.hypot(
                    (path[i+1][0] - path[i][0]) * math.radians(1/PPD) * MARS_RADIUS_M,
                    (path[i+1][1] - path[i][1]) * math.radians(1/PPD) * MARS_RADIUS_M
                    * math.cos(math.radians(mean_lat)),
                )
                for i in range(len(path) - 1)
            ) / 1000.0
            return {
                "Route":               label,
                "Cells":               len(path),
                "Est. distance (km)":  round(ground_km, 2),
                "Mean risk":           round(float(vals.mean()), 1),
                "Max risk":            round(float(vals.max()), 1),
                "Planner cost":        round(obj_cost / 1000.0, 2),
            }

        stats_df = pd.DataFrame([
            path_stats(shortest, "Shortest-distance (Dijkstra)", dist_cost),
            path_stats(safer,    "Risk-weighted A*",             risk_cost),
        ])

        # Difference row
        diff = {
            "Route":              "Difference (A* − Shortest)",
            "Cells":              stats_df.loc[1, "Cells"] - stats_df.loc[0, "Cells"],
            "Est. distance (km)": round(
                stats_df.loc[1, "Est. distance (km)"]
                - stats_df.loc[0, "Est. distance (km)"], 2
            ),
            "Mean risk":          round(
                stats_df.loc[1, "Mean risk"] - stats_df.loc[0, "Mean risk"], 1
            ),
            "Max risk":           round(
                stats_df.loc[1, "Max risk"] - stats_df.loc[0, "Max risk"], 1
            ),
            "Planner cost":       round(
                stats_df.loc[1, "Planner cost"] - stats_df.loc[0, "Planner cost"], 2
            ),
        }
        stats_df = pd.concat(
            [stats_df, pd.DataFrame([diff])], ignore_index=True
        )

        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        st.markdown(
            "**Mission note:** The risk-weighted route is an algorithmic alternative "
            "on the coarse MOLA regional grid. A lower prototype risk score does **not** "
            "prove real-world traversability or safety."
        )
        st.success(
            f"✅ Both paths connect {start} → {goal} through valid cells. "
            f"Shortest: {len(shortest)} cells; "
            f"Risk-weighted: {len(safer)} cells."
        )

    except ValueError as exc:
        st.error(f"Route could not be calculated: {exc}")
        st.info(
            "Ensure both endpoints are in **valid** cells (counts > 0 and finite risk). "
            "Check the terrain layer viewer — invalid cells appear as gaps."
        )
    except Exception as exc:
        st.error(f"Unexpected error during route calculation: {exc}")

# ---------------------------------------------------------------------------
# Data provenance and limitations expander
# ---------------------------------------------------------------------------
with st.expander("📋 Data provenance, metadata and limitations"):
    st.json(meta)
    st.markdown(
        """
**NASA source:** [PDS Geosciences Node — MOLA MEGDR]
(https://pds-geosciences.wustl.edu/missions/mgs/megdr.html)

**Optical context:** [HiRISE ESP_094258_1985 product page]
(https://hirise.lpl.arizona.edu/ESP_094258_1985) —
acquired 2026-09-04; image credit **NASA/JPL-Caltech/UArizona**.
This image is **not co-registered** to the MOLA grid and is displayed
as visual context only. Do not infer DEM-derived slope from HiRISE.

**Risk heuristic:**
- 50% slope (normalisation cap 20°)
- 30% local roughness / 3×3 std (cap 75 m)
- 20% local relief / 3×3 max−min (cap 150 m)

These are demo assumptions requiring domain validation before any
operational use.  PDS citation guidance: https://pds.nasa.gov/datastandards/citing/
        """
    )
