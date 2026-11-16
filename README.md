# MarsWalk AI — Ares Forge

A 2-days hackathon MVP for exploring terrain risk and route alternatives in Jezero Crater using NASA/PDS Mars elevation data.

> **Scientific scope:** This is a research/demo prototype, not an operational astronaut navigation system. The included MOLA MEGDR product is a coarse global elevation grid (16 pixels/degree; roughly 3.7 km per pixel at the equator, somewhat less east-west at Jezero). It cannot resolve boulders, small craters, slopes at human/rover scale, or suit traversability. The risk score and thresholds are illustrative engineering choices, not NASA standards or certified safety guidance.

## What it does

1. Downloads the MOLA 16-pixel/degree global topography and observation-count grids from the PDS Geosciences Node.
2. Crops a small Jezero-area window from the global grids.
3. Uses the count grid to mask cells without altimetry observations.
4. Calculates slope, 3x3 local elevation standard deviation, and 3x3 local relief.
5. Creates an explainable terrain-risk index (0–100).
6. Compares a shortest-distance path with an A* risk-weighted path.
7. Displays the results in Streamlit/Plotly and reports data provenance.

## Quick start (Windows PowerShell)

Use Python 3.11 or 3.12 (64-bit). Rasterio wheels are available for common Windows/Python combinations.

```powershell
cd marswalk-ai
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ingestion.fetch_hirise
python -m ingestion.fetch_mola
streamlit run app.py
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `py -3.11` is unavailable, install Python 3.11 x64 from https://www.python.org/downloads/ and enable the Python Launcher / PATH options.

## NASA data used

### Numerical terrain source

**MGS MOLA MEGDR, 16 pixels/degree, global topography and counts.**

PDS archive landing page and product description:
- https://pds-geosciences.wustl.edu/missions/mgs/megdr.html
- Topography IMG: https://pds-geosciences.wustl.edu/mgs/urn-nasa-pds-mgs_mola_topography_derived/meg016/megt90n000eb.img
- Topography ENVI header: same URL with `.hdr`
- Topography PDS label: same URL with `.xml`
- Counts IMG: https://pds-geosciences.wustl.edu/mgs/urn-nasa-pds-mgs_mola_topography_derived/meg016/megc90n000eb.img
- Counts ENVI header and PDS label: same URLs with `.hdr` and `.xml`

The PDS page documents that MEGDR products are global topographic maps binned from MOLA altimetry, lists resolutions of 4/16/32/64/128 pixels per degree, and states that the image, XML/PDS label and ENVI header are supplied together. The code downloads the 16 px/degree product plus its header and XML label. The data are historical, based on the Mars Global Surveyor MOLA mission; MOLA altimetry acquisition ended 30 June 2001. They are not current environmental conditions.

**Data caveat:** The project expects the PDS ENVI header to be available alongside the `.img` file and lets Rasterio interpret the binary grid. The code verifies raster dimensions and uses the companion counts grid for validity. If the PDS server changes the files or the header cannot be read, stop rather than guessing the binary byte order or fill value.

### Optical context (visual only; not used for numerical risk)

A Jezero HiRISE browse image:
- Product page: https://hirise.lpl.arizona.edu/ESP_094258_1985
- Browse JPEG: https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/ESP/ORB_094200_094299/ESP_094258_1985/ESP_094258_1985_RED.abrowse.jpg

The product page records acquisition on 4 September 2026, center 18.478°N, 77.540°E, map-projected scale 50 cm/pixel, and a public-domain usage statement with requested credit `NASA/JPL-Caltech/UArizona`. This image is an orbital observation and is displayed as context only. It is **not automatically co-registered** to the coarse MOLA grid and must not be used to infer DEM slope.

### Licensing and attribution

NASA science-data licensing depends on the specific collection and any notices in its metadata. Check the product label and repository terms for the exact files used. NASA's Science Data Portal explains that NASA-led mission data are generally CC0 unless a restrictive notice applies, and recommends citing the original source. PDS citation guidance is at https://pds.nasa.gov/datastandards/citing/. For the HiRISE browse image, retain the requested credit `NASA/JPL-Caltech/UArizona`.

## Coordinate and resolution assumptions

- The selected MOLA file is the global 16 px/degree product `megt90n000eb.img`.
- The ingestion code crops by the documented global grid convention: longitude 0–360° east and latitude 90°N to 90°S, 16 samples per degree.
- The crop is 77.0–78.0°E and 17.5–19.5°N by default. These are approximate regional bounds around the cited Jezero HiRISE scene, not a rover traverse polygon.
- For local derivatives, latitude/longitude cell spacing is converted to approximate metres using Mars mean radius 3,389,500 m and the cosine of the crop's mean latitude.
- No resampling is performed. The crop retains the source grid's native spacing.
- The PDS label and ENVI header are saved beside the binary image. Inspect them if the product version or interpretation changes.

## Risk model

The prototype calculates:
- `slope_deg`: gradient magnitude from the DEM, in degrees.
- `roughness_m`: 3×3 neighbourhood standard deviation of elevation, in metres.
- `local_relief_m`: 3×3 neighbourhood max-minus-min elevation, in metres.

Illustrative score:
`risk = 0.50*slope_score + 0.30*roughness_score + 0.20*relief_score`

Normalization caps are currently 20° slope, 75 m roughness and 150 m local relief. These are tunable demo assumptions, **not validated astronaut or rover safety thresholds**. MOLA's kilometre-scale grid makes these broad regional terrain indicators only.

## Route planner

- 8-connected grid, with diagonal step distance adjusted for latitude.
- Baseline: shortest-distance Dijkstra path.
- Alternative: A* with step cost `distance * (1 + 4 * mean_cell_risk/100)`.
- Missing-data cells are blocked.
- The A* heuristic is straight-line distance and is admissible because the risk multiplier is at least 1.
- Grid routes are cell-to-cell demonstrations. They do not represent verified traverses or account for vehicle dynamics, energy, radiation, weather, communications, or sub-grid obstacles.

## Run tests and verify

```powershell
pytest -q
python -m ingestion.fetch_mola
streamlit run app.py
```

Sanity checks include:
- expected dimensions for the selected 16 px/degree global MEGDR grid;
- topography/count raster alignment;
- finite-value and count-mask checks;
- slope in [0, 90] degrees;
- risk in [0, 100];
- route endpoints and valid traversable cells.

The automated tests use synthetic arrays for deterministic checks. They do not prove that a live NASA download succeeded. A real-data run is verified only after the downloader completes and the app reports loaded source metadata.

## 90-second judging demo

1. “MarsWalk AI turns NASA planetary data into explainable route-planning context.”
2. Show the source provenance: MOLA global topography plus its counts/validity layer.
3. Show the Jezero crop and explain that MOLA is kilometre-scale.
4. Toggle slope, roughness, relief and the combined risk map.
5. Select start and destination; compare shortest and risk-weighted paths.
6. Explain why the alternative path incurs a distance trade-off to reduce the prototype risk score.
7. Show the HiRISE orbital image as separate high-resolution visual context, not as the source of MOLA-derived calculations.
8. Close: “This is a regional planning demonstrator. Operational Marswalk safety would require higher-resolution, co-registered terrain, validated mobility models, environmental forecasts and mission-specific constraints.”

## Credits

- MOLA: NASA Mars Global Surveyor / MOLA Science Team; archived by PDS Geosciences Node, Washington University in St. Louis.
- HiRISE browse image: NASA/JPL-Caltech/UArizona.
- Always preserve source labels, product identifiers, acquisition dates and citations in any derivative presentation.
