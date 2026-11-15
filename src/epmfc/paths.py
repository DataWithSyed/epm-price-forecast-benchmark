from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REFERENCE = DATA / "reference"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TZ = "Europe/Berlin"
TRAIN_START = "2025-01-01"
TEST_START = "2026-01-01"
TEST_END = "2026-10-31"
