"""Download the MOLA 16 px/degree MEGDR topography + counts and metadata."""
from pathlib import Path
import requests
import rasterio

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://pds-geosciences.wustl.edu/mgs/urn-nasa-pds-mgs_mola_topography_derived/meg016"
PRODUCTS = {
    "topography": "megt90n000eb",
    "counts": "megc90n000eb",
}

def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using cached {destination.name}")
        return
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            raise RuntimeError(f"Expected binary/metadata file, received HTML for {url}")
        tmp = destination.with_suffix(destination.suffix + ".part")
        with tmp.open("wb") as out:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
        if tmp.stat().st_size < 100:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file suspiciously small: {url}")
        tmp.replace(destination)

def main():
    for name, stem in PRODUCTS.items():
        for ext in (".img", ".hdr", ".xml"):
            download(f"{BASE}/{stem}{ext}", RAW / f"{stem}{ext}")

    topo = RAW / f"{PRODUCTS['topography']}.img"
    counts = RAW / f"{PRODUCTS['counts']}.img"
    # Rasterio/GDAL should discover the ENVI .hdr sidecars. This check catches
    # incorrect downloads or missing/unsupported headers before analysis.
    with rasterio.open(topo) as ds:
        print("Topography:", ds.width, "x", ds.height, "dtype:", ds.dtypes[0],
              "transform:", ds.transform, "crs:", ds.crs, "nodata:", ds.nodata)
        if (ds.width, ds.height) != (5760, 2880):
            raise RuntimeError(
                "Unexpected MOLA 16 px/degree dimensions. Inspect the downloaded "
                "PDS ENVI header/XML; do not guess dimensions or byte order."
            )
    with rasterio.open(counts) as ds:
        print("Counts:", ds.width, "x", ds.height, "dtype:", ds.dtypes[0])
        if (ds.width, ds.height) != (5760, 2880):
            raise RuntimeError("Counts grid dimensions do not match expected MEGDR grid.")
    print("MOLA files downloaded and basic grid checks passed.")
    print("Source: https://pds-geosciences.wustl.edu/missions/mgs/megdr.html")

if __name__ == "__main__":
    main()
