"""Download the HiRISE ESP_094258_1985 browse JPEG for Jezero Crater.

This image is used as VISUAL CONTEXT ONLY. It is NOT co-registered to the
MOLA DEM grid and must NOT be used for any numerical terrain calculation.

Product page : https://hirise.lpl.arizona.edu/ESP_094258_1985
Acquisition  : 2026-09-04; centre 18.478°N, 77.540°E
Scale        : 50 cm/pixel (map-projected)
Credit       : NASA/JPL-Caltech/UArizona (public domain; credit requested)

Source: HiRISE PDS EXTRAS browse image.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

URL = (
    "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR/"
    "ESP/ORB_094200_094299/ESP_094258_1985/"
    "ESP_094258_1985_RED.abrowse.jpg"
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "hirise"
DEST_NAME = "ESP_094258_1985_RED.abrowse.jpg"

# A JPEG browse image this small indicates an error page.
_MIN_SIZE_BYTES = 50_000


def download_hirise(
    output_dir: Path | None = None,
    url: str = URL,
    timeout: int = 120,
    retries: int = 3,
) -> Path:
    """Download the HiRISE browse JPEG if not already cached.

    Returns
    -------
    Path
        Local path to the downloaded file.

    Raises
    ------
    RuntimeError
        If the download fails after all retries or the file looks like an
        HTML error page.
    """
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    dest = out / DEST_NAME

    if dest.exists() and dest.stat().st_size >= _MIN_SIZE_BYTES:
        print(
            f"  Using cached HiRISE browse image: {dest.name} "
            f"({dest.stat().st_size / 1024:.1f} KB)"
        )
        return dest

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        print(f"  Downloading HiRISE browse image (attempt {attempt}/{retries}) …")
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()

            ct = resp.headers.get("Content-Type", "")
            if "text/html" in ct.lower():
                raise RuntimeError(
                    f"Server returned HTML instead of a JPEG for {url}. "
                    "Check that the HiRISE product URL is still valid."
                )

            if len(resp.content) < _MIN_SIZE_BYTES:
                raise RuntimeError(
                    f"Response too small ({len(resp.content):,} bytes); "
                    "likely an error page, not the actual browse JPEG."
                )

            dest.write_bytes(resp.content)
            print(
                f"  ✓ {dest.name} ({dest.stat().st_size / 1024:.1f} KB)\n"
                "  Note: HiRISE image is visual context only — "
                "not co-registered to MOLA grid."
            )
            return dest

        except (requests.RequestException, RuntimeError, OSError) as exc:
            last_exc = exc
            print(f"  ✗ Attempt {attempt} failed: {exc}")

    raise RuntimeError(
        f"Failed to download HiRISE browse image after {retries} attempts. "
        f"Last error: {last_exc}"
    )


def main() -> None:
    print("=" * 60)
    print("MarsWalk AI — HiRISE browse image fetcher")
    print("Product: ESP_094258_1985 | Jezero Crater")
    print("Credit: NASA/JPL-Caltech/UArizona")
    print("=" * 60)
    try:
        path = download_hirise()
        print(f"\n✅  HiRISE browse image available at: {path}")
        print(
            "    This image is visual context only.\n"
            "    It is NOT co-registered to the MOLA DEM.\n"
            "    Do not use it to infer DEM-derived slope or risk."
        )
    except RuntimeError as exc:
        print(f"\n❌  {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
