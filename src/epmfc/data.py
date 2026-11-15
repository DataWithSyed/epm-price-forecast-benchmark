"""Load the lab panel (step 2) and build the ex-ante forecast panel.

Sources (CC BY 4.0 unless noted): Energy-Charts.info (Fraunhofer ISE) public net
generation, load, cross-border trade and *day-ahead* public power forecasts; DE-LU
day-ahead price from Bundesnetzagentur | SMARD.de; World Bank Pink Sheet; ECB;
DEHSt auction reports (official public figures).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from epmfc.paths import PROCESSED, RAW, REFERENCE, TZ

FC_TYPES = ("load", "solar", "wind_onshore", "wind_offshore")
NEIGHBOURS = ("fr", "nl", "be", "at", "pl", "cz", "dk", "ch")


def _series_from_json(path: Path) -> pd.Series:
    d = json.loads(path.read_text())
    if "unix_seconds" not in d:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
    s = pd.Series(d["forecast_values"], index=idx, dtype="float64")
    s = s[~s.index.duplicated()]
    return s.resample("1h").mean()


def _causal_fill(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Reindex and fill gaps with the value 7 days earlier (then 14 days), never with later data.

    A series that only starts part-way (e.g. a new offshore fleet) is 0 before its first value."""
    s = s.reindex(idx)
    first = s.first_valid_index()
    for lag in (168, 336):
        s = s.fillna(s.shift(lag))
    if first is not None:
        s[s.index < first] = 0.0
    return s.fillna(0.0)


def build_forecast_panel(raw: Path = RAW) -> pd.DataFrame:
    """Hourly (UTC) DE day-ahead forecasts + neighbour load / RES / residual-load forecast aggregates.

    Columns: load_fc_mw, solar_fc_mw, wind_onshore_fc_mw, wind_offshore_fc_mw (DE),
    nb_load_fc_mw, nb_res_fc_mw, nb_rl_fc_mw (sum over NEIGHBOURS with data), n_nb_gaps_filled.
    """
    cols = {}
    for pt in FC_TYPES:
        parts = [_series_from_json(p) for p in sorted(raw.glob(f"fc_{pt}_*.json"))]
        s = pd.concat(parts).sort_index()
        cols[f"{pt}_fc_mw"] = s[~s.index.duplicated()]
    df = pd.DataFrame(cols)
    idx = df.index
    nb_load = pd.Series(0.0, index=idx)
    nb_res = pd.Series(0.0, index=idx)
    gaps = pd.Series(0, index=idx)
    for c in NEIGHBOURS:
        for pt in ("load",) + FC_TYPES[1:]:
            p = raw / f"nb_{c}_{pt}.json"
            if not p.exists():
                continue
            s = _series_from_json(p)
            if s.empty:
                continue
            gaps += s.reindex(idx).isna() & (idx >= s.first_valid_index())
            s = _causal_fill(s, idx)
            if pt == "load":
                nb_load += s
            else:
                nb_res += s
    df["nb_load_fc_mw"] = nb_load
    df["nb_res_fc_mw"] = nb_res
    df["nb_rl_fc_mw"] = nb_load - nb_res
    df["n_nb_gaps_filled"] = gaps.astype(int)
    df.index.name = "timestamp"
    return df


def load_panel() -> pd.DataFrame:
    """Step-2 hourly panel (UTC index), realised values."""
    h = pd.read_csv(PROCESSED / "hourly_panel.csv", index_col=0)
    h.index = pd.to_datetime(h.index, utc=True)
    return h


def load_forecasts() -> pd.DataFrame:
    f = pd.read_csv(PROCESSED / "forecast_panel.csv.gz", index_col=0)
    f.index = pd.to_datetime(f.index, utc=True)
    return f


def load_step2_models() -> pd.DataFrame:
    m = pd.read_csv(PROCESSED / "step2_model_prices.csv.gz", index_col=0)
    m.index = pd.to_datetime(m.index, utc=True)
    return m


def load_monthly() -> pd.DataFrame:
    m = pd.read_csv(PROCESSED / "monthly_drivers.csv")
    m["month"] = pd.PeriodIndex(m["month"], freq="M")
    return m.set_index("month")


def load_installed() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "installed_gw.csv", index_col=0)


def eua_monthly_series() -> pd.Series:
    """Monthly EUA auction price (EUR/t): 2025 = annual DEHSt VWAP 73.86 (flat), 2026 = DEHSt monthly."""
    m = load_monthly()["eua_eur_t"].copy()
    ref = pd.read_csv(REFERENCE / "eua_dehst_monthly_2026.csv")
    for _, r in ref.iterrows():
        m.loc[pd.Period(r["month"], freq="M")] = float(r["eua_eur_t"])
    return m


def to_daily_matrix(s: pd.Series) -> pd.DataFrame:
    """UTC hourly series -> (local delivery day x 24 local hours) matrix.

    DST: the duplicated autumn hour is averaged; the missing spring hour (02:00) is
    linearly interpolated from 01:00 and 03:00. Applied identically to actuals and
    forecasts, so the scoring grid is 24 h/day.
    """
    loc = s.tz_convert(TZ)
    df = pd.DataFrame({"v": loc.to_numpy(), "day": loc.index.date, "hour": loc.index.hour})
    mat = df.groupby(["day", "hour"])["v"].mean().unstack("hour")
    mat = mat.reindex(columns=range(24))
    mat = mat.interpolate(axis=1, limit_area="inside")
    mat.index = pd.to_datetime(mat.index)
    mat.index.name = "day"
    return mat
