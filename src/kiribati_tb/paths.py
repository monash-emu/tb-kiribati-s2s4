"""Repository paths (``tbh/paths.py``)."""

from pathlib import Path

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
DATA_FOLDER = REPO_ROOT_PATH / "data"
GOLDEN_FOLDER = REPO_ROOT_PATH / "tests" / "golden"
OUTPUT_FOLDER = REPO_ROOT_PATH / "outputs"
