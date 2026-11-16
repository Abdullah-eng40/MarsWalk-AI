"""Tests for ingestion modules using mocks (no real NASA network calls).

These tests verify error-handling and caching logic in fetch_mola and
fetch_hirise without downloading any actual data.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.fetch_mola import download as mola_download, EXPECTED_SIZES
from ingestion.fetch_hirise import download_hirise


# ---------------------------------------------------------------------------
# fetch_mola tests
# ---------------------------------------------------------------------------
class TestMolaDownload:

    def test_skips_cached_file_correct_size(self, tmp_path: Path):
        """If the file already exists with the right size, download is skipped."""
        fname = "megt90n000eb.img"
        dest  = tmp_path / fname
        dest.write_bytes(b"x" * EXPECTED_SIZES[fname])   # correct size
        # requests.get should never be called
        with patch("ingestion.fetch_mola.requests.get") as mock_get:
            mola_download("http://never-called", dest)
            mock_get.assert_not_called()

    def test_skips_cached_text_file(self, tmp_path: Path):
        """Text sidecars (hdr/xml) with size 0 expected → any non-zero size accepted."""
        dest = tmp_path / "megt90n000eb.hdr"
        dest.write_bytes(b"ENVI\n" * 20)   # small but non-zero
        with patch("ingestion.fetch_mola.requests.get") as mock_get:
            mola_download("http://never-called", dest)
            mock_get.assert_not_called()

    def test_raises_on_html_response(self, tmp_path: Path):
        """An HTML Content-Type response should raise RuntimeError."""
        dest = tmp_path / "megt90n000eb.hdr"

        mock_resp = MagicMock()
        mock_resp.headers       = {"Content-Type": "text/html; charset=utf-8"}
        mock_resp.status_code   = 200
        mock_resp.__enter__     = lambda s: s
        mock_resp.__exit__      = MagicMock(return_value=False)
        mock_resp.iter_content  = MagicMock(return_value=iter([]))

        with patch("ingestion.fetch_mola.requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="[Hh][Tt][Mm][Ll]"):
                mola_download("http://example.com/fake.hdr", dest, retries=1)

    def test_raises_after_all_retries(self, tmp_path: Path):
        """Network failures exhaust retries and raise RuntimeError."""
        import requests as req_lib

        dest = tmp_path / "megt90n000eb.hdr"

        with patch("ingestion.fetch_mola.requests.get",
                   side_effect=req_lib.ConnectionError("network down")):
            with patch("ingestion.fetch_mola.time.sleep"):  # skip backoff waits
                with pytest.raises(RuntimeError, match="[Ff]ailed"):
                    mola_download("http://unreachable/file.hdr", dest, retries=2)


# ---------------------------------------------------------------------------
# fetch_hirise tests
# ---------------------------------------------------------------------------
class TestHiRiseDownload:

    def test_skips_cached_image(self, tmp_path: Path):
        """Cached JPEG larger than min size is not re-downloaded."""
        dest = tmp_path / "ESP_094258_1985_RED.abrowse.jpg"
        dest.write_bytes(b"\xff\xd8" + b"x" * 100_000)   # fake JPEG, right size
        with patch("ingestion.fetch_hirise.requests.get") as mock_get:
            result = download_hirise(output_dir=tmp_path)
            mock_get.assert_not_called()
        assert result == dest

    def test_raises_on_html_response(self, tmp_path: Path):
        """HTML Content-Type instead of JPEG should raise RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers     = {"Content-Type": "text/html"}
        mock_resp.content     = b"<html>Error</html>"
        mock_resp.raise_for_status = MagicMock()

        with patch("ingestion.fetch_hirise.requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="[Hh][Tt][Mm][Ll]"):
                download_hirise(output_dir=tmp_path, retries=1)

    def test_raises_on_too_small_response(self, tmp_path: Path):
        """A response below the minimum size threshold should raise RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers     = {"Content-Type": "image/jpeg"}
        mock_resp.content     = b"\xff\xd8" + b"x" * 100   # too small
        mock_resp.raise_for_status = MagicMock()

        with patch("ingestion.fetch_hirise.requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="[Ss]mall|[Ee]rror"):
                download_hirise(output_dir=tmp_path, retries=1)
