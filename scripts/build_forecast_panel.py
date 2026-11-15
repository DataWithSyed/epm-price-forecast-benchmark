"""Fetch (optional) and build data/processed/forecast_panel.csv.gz from Energy-Charts day-ahead forecasts."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import requests  # noqa: E402

from epmfc.data import FC_TYPES, NEIGHBOURS, build_forecast_panel  # noqa: E402
from epmfc.paths import PROCESSED, RAW  # noqa: E402

API = "https://api.energy-charts.info/public_power_forecast"
DE_WINDOWS = [("2024-12-30", "2025-06-30"), ("2025-07-01", "2025-12-31"),
              ("2026-01-01", "2026-06-30"), ("2026-07-01", "2026-10-31")]


def _get(params: dict) -> requests.Response:
    for _ in range(12):
        r = requests.get(API, params=params, timeout=180)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "40")) + 3)
            continue
        return r
    raise RuntimeError(params)


def _save(r: requests.Response, path: Path) -> None:
    if r.status_code != 200:
        path.write_text(json.dumps({"status": r.status_code}))
        return
    d = r.json()
    keep = ("unix_seconds", "forecast_values", "production_type", "forecast_type", "license", "deprecated")
    path.write_text(json.dumps({k: d[k] for k in d if k in keep}))


def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for pt in FC_TYPES:
        for a, b in DE_WINDOWS:
            p = RAW / f"fc_{pt}_{a}_{b}.json"
            if not p.exists():
                _save(_get({"country": "de", "production_type": pt, "forecast_type": "day-ahead",
                            "start": a, "end": b}), p)
                time.sleep(3)
    for c in NEIGHBOURS:
        for pt in FC_TYPES:
            p = RAW / f"nb_{c}_{pt}.json"
            if not p.exists():
                _save(_get({"country": c, "production_type": pt, "forecast_type": "day-ahead",
                            "start": "2024-12-30", "end": "2026-10-31"}), p)
                time.sleep(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download raw JSON first (git-ignored)")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    df = build_forecast_panel()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.round(1).to_csv(PROCESSED / "forecast_panel.csv.gz", compression="gzip")
    print(df.describe().T.round(0))
