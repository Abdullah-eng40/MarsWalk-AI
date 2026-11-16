# conftest.py — project-root pytest configuration
# Ensures the marswalk and ingestion packages are on sys.path
# regardless of working directory when pytest is invoked.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
