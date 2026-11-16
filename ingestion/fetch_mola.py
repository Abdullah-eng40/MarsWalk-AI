"""Download the MOLA 16 px/degree MEGDR topography + counts and metadata.

Source: MGS MOLA MEGDR 16 px/degree, NASA PDS Geosciences Node.
Reference: https://pds-geosciences.wustl.edu/missions/mgs/megdr.html

This module validates that all downloaded files match the documented MEGDR
grid dimensions (5760 × 2880 for 16 px/degree global) and expected byte sizes.
It never guesses byte order, fill values, or dimensions.
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

import requests
import rasterio

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

BASE = (
    "https://pds-geosciences.wustl.edu/mgs/"
    "urn-nasa-pds-mgs_mola_topography_derived/meg016"
)

PRODUCTS = {
    "topography": "megt90n000eb",
    "counts": "megc90n000eb",
}

# Expected byte sizes verified from actual PDS downloads (2026-11-16).
# If the PDS server updates these files, stop and inspect the new metadata.
EXPECTED_SIZES: dict[str, int] = {
    "megt90n000eb.img": 33_177_600,   # 5760 × 2880 × 2 bytes (int16)
    "megt90n000eb.hdr": 0,            # 0 = skip size check for small text files
    "megt90n000eb.xml": 0,
    "megc90n000eb.img": 33_177_600,
    "megc90n000eb.hdr": 0,
    "megc90n000eb.xml": 0,
}

# Minimum acceptable byte count for any file (guards against HTML error pages).
_MIN_BINARY_BYTES = 1_024
_MIN_TEXT_BYTES = 100


def download(
    url: str,
    destination: Path,
    retries: int = 3,
    backoff_base: float = 10.0,
    timeout: tuple[int, int] = (30, 300),
) -> None:
    """Stream-download *url* to *destination* with retries and size validation.

    - Skips if *destination* already exists with the expected (or any non-zero) size.
    - Resumes a partial ``.part`` file if the server supports Range requests.
    - Raises ``RuntimeError`` on HTML responses (usually a PDS error page).
    - Raises ``RuntimeError`` if the final file is suspiciously small.
    - Retries up to *retries* times with exponential back-off.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    # --- Cache hit: skip if fully downloaded and size looks right ---
    expected = EXPECTED_SIZES.get(destination.name, 0)
    if destination.exists() and destination.stat().st_size > 0:
        if expected == 0 or destination.stat().st_size == expected:
            print(f"  Using cached {destination.name} "
                  f"({destination.stat().st_size:,} bytes)")
            return
        # Size mismatch: re-download
        print(f"  Cached {destination.name} has unexpected size "
              f"({destination.stat().st_size:,} vs {expected:,}) — re-downloading")
        destination.unlink()

    tmp = destination.with_suffix(destination.suffix + ".part")
    resume_at = tmp.stat().st_size if tmp.exists() else 0

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            headers: dict[str, str] = {}
            if resume_at > 0:
                headers["Range"] = f"bytes={resume_at}-"
                print(f"  Resuming {destination.name} from byte {resume_at:,} "
                      f"(attempt {attempt}/{retries})")
            else:
                print(f"  Downloading {url} (attempt {attempt}/{retries})")

            with requests.get(
                url, stream=True, timeout=timeout, headers=headers
            ) as resp:
                # Detect HTML error pages returned with a 200 status
                ct = resp.headers.get("Content-Type", "")
                if "text/html" in ct.lower():
                    raise RuntimeError(
                        f"Server returned HTML instead of binary data for {url}. "
                        "This is likely a PDS error page. Check the URL."
                    )

                # 206 Partial Content → append; anything else → overwrite
                if resp.status_code == 206 and resume_at > 0:
                    mode = "ab"
                elif resp.status_code == 200:
                    mode = "wb"
                    resume_at = 0  # server didn't honour Range → restart
                else:
                    resp.raise_for_status()
                    mode = "wb"

                with tmp.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)

            # Basic size sanity check
            final_size = tmp.stat().st_size
            min_ok = expected if expected > 0 else _MIN_BINARY_BYTES
            if final_size < min_ok:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Downloaded file is too small ({final_size:,} bytes, "
                    f"expected ≥ {min_ok:,}): {url}"
                )
            if expected > 0 and final_size != expected:
                raise RuntimeError(
                    f"File size mismatch for {destination.name}: "
                    f"got {final_size:,} bytes, expected {expected:,}. "
                    "Do not proceed — inspect the PDS product label."
                )

            tmp.replace(destination)
            print(f"  ✓ {destination.name} ({final_size:,} bytes)")
            return

        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            wait = backoff_base * (2 ** (attempt - 1))
            print(f"  ✗ Attempt {attempt} failed: {exc}")
            if attempt < retries:
                print(f"    Retrying in {wait:.0f}s …")
                time.sleep(wait)
                # Update resume position for next attempt
                resume_at = tmp.stat().st_size if tmp.exists() else 0

    raise RuntimeError(
        f"Failed to download {url} after {retries} attempts. "
        f"Last error: {last_exc}"
    )


def validate_grid(path: Path, label: str, expected_wh: tuple[int, int] = (5760, 2880)) -> None:
    """Open *path* with rasterio and assert expected grid dimensions.

    Raises ``RuntimeError`` with an actionable message if anything is wrong.
    Never guesses byte order or fill values.
    """
    with rasterio.open(path) as ds:
        w, h = ds.width, ds.height
        print(
            f"  {label}: {w} × {h}  dtype={ds.dtypes[0]}  "
            f"nodata={ds.nodata}  crs={'yes' if ds.crs else 'none'}"
        )
        if (w, h) != expected_wh:
            raise RuntimeError(
                f"Unexpected MOLA grid dimensions for {path.name}: "
                f"got {w} × {h}, expected {expected_wh[0]} × {expected_wh[1]}. "
                "Inspect the downloaded PDS ENVI header / XML label before continuing. "
                "Do not guess dimensions or byte order."
            )


def main() -> None:
    print("=" * 60)
    print("MarsWalk AI — MOLA MEGDR data fetcher")
    print("Source: https://pds-geosciences.wustl.edu/missions/mgs/megdr.html")
    print("=" * 60)

    for name, stem in PRODUCTS.items():
        print(f"\n[{name.upper()}]")
        for ext in (".img", ".hdr", ".xml"):
            url = f"{BASE}/{stem}{ext}"
            dest = RAW / f"{stem}{ext}"
            download(url, dest)

    print("\n[VALIDATION]")
    topo_path = RAW / f"{PRODUCTS['topography']}.img"
    counts_path = RAW / f"{PRODUCTS['counts']}.img"

    validate_grid(topo_path, "Topography")
    validate_grid(counts_path, "Counts")

    print(
        "\n✅  All MOLA files present and grid checks passed.\n"
        "    Data are historical MGS altimetry (ended 2001-06-30).\n"
        "    Not current Mars environmental conditions.\n"
        f"    Raw files at: {RAW}"
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\n❌  {exc}", file=sys.stderr)
        sys.exit(1)
