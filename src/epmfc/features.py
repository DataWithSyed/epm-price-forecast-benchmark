"""Strictly ex-ante feature construction at day-ahead gate closure (D-1 12:00 Europe/Berlin).

Information set for delivery day D (all ASSUMED publication times are labelled):
- DE-LU day-ahead prices up to and including D-1 (the D-1 auction cleared on D-2).
- TSO day-ahead forecasts of load, solar, wind on/offshore for D (DE and neighbours),
  as republished by Energy-Charts (published on D-1 before gate closure).
- Realised generation / cross-border trade only up to D-2 (complete days; conservative).
- Monthly Pink Sheet gas/coal (published ~2nd business day of the next month;
  ASSUMED usable from the 4th calendar day), and the monthly DEHSt EUA auction average
  for the previous month (reconstructible from public daily EEX auction results;
  ASSUMED usable from the 4th calendar day). 2025 EUA is the flat annual VWAP (proxy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from epmfc.data import eua_monthly_series, load_monthly, to_daily_matrix

MUSTRUN_RES_COLS = ["biomass_mw", "waste_mw", "geothermal_mw", "others_mw",
                    "hydro_ror_mw", "hydro_res_mw", "coal_gas_mw"]
PUBLICATION_DAY = 4  # ASSUMED: month m values usable from day 4 of month m+1


def latest_month_known(cutoff: pd.Timestamp) -> pd.Period:
    """Latest monthly-benchmark month whose value is published by ``cutoff`` (a date)."""
    m = pd.Period(cutoff, freq="M")
    return m - 1 if cutoff.day >= PUBLICATION_DAY else m - 2


def lagged_fuels(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Per delivery day D: gas, coal (EUR/MWh_th), oil and EUA known at D-1."""
    mon = load_monthly()
    eua = eua_monthly_series()
    rows = []
    for d in days:
        m = latest_month_known(d - pd.Timedelta(days=1))
        rows.append({"gas_eur_mwh_th": float(mon.loc[m, "gas_eur_mwh_th"]),
                     "coal_eur_mwh_th": float(mon.loc[m, "coal_eur_mwh_th"]),
                     "oil_eur_mwh_th": float(mon.loc[m, "oil_eur_mwh_th"]),
                     "eua_eur_t": float(eua.loc[m]), "fuel_month": str(m)})
    return pd.DataFrame(rows, index=days)


def contemporaneous_fuels(days: pd.DatetimeIndex) -> pd.DataFrame:
    """ORACLE: same-month fuel/EUA values (not known at D-1); used only for oracle references."""
    mon = load_monthly()
    eua = eua_monthly_series()
    ms = [pd.Period(d, freq="M") for d in days]
    return pd.DataFrame({"gas_eur_mwh_th": [mon.loc[m, "gas_eur_mwh_th"] for m in ms],
                         "coal_eur_mwh_th": [mon.loc[m, "coal_eur_mwh_th"] for m in ms],
                         "oil_eur_mwh_th": [mon.loc[m, "oil_eur_mwh_th"] for m in ms],
                         "eua_eur_t": [eua.loc[m] for m in ms]}, index=days)


def shift_days(mat: pd.DataFrame, k: int) -> pd.DataFrame:
    """Value of day D-k aligned on day D (calendar shift; daily index is complete)."""
    full = mat.asfreq("D")
    return full.shift(k)


class Matrices:
    """All daily (day x 24h) matrices used by the models."""

    def __init__(self, h: pd.DataFrame, f: pd.DataFrame, step2: pd.DataFrame | None = None):
        f = f.reindex(h.index)
        self.price = to_daily_matrix(h["da_price_eur_mwh"]).asfreq("D")
        self.load_fc = to_daily_matrix(f["load_fc_mw"]).asfreq("D")
        self.res_fc = to_daily_matrix(f["solar_fc_mw"] + f["wind_onshore_fc_mw"] + f["wind_offshore_fc_mw"]).asfreq("D")
        self.rl_fc = self.load_fc - self.res_fc
        self.nb_rl_fc = to_daily_matrix(f["nb_rl_fc_mw"]).asfreq("D")
        self.nb_res_fc = to_daily_matrix(f["nb_res_fc_mw"]).asfreq("D")
        # realised (oracle for day D; lagged versions are ex-ante)
        self.rl_real = to_daily_matrix(h["residual_load_mw"]).asfreq("D")
        self.load_real = to_daily_matrix(h["load_mw"]).asfreq("D")
        self.res_real = self.load_real - self.rl_real
        self.trade_real = to_daily_matrix(h["net_trade_mw"]).asfreq("D")
        self.mustrun_res_real = to_daily_matrix(h[MUSTRUN_RES_COLS].sum(axis=1)).asfreq("D")
        if step2 is not None:
            self.m0_oracle = to_daily_matrix(step2["M0_stack"].reindex(h.index)).asfreq("D")
            self.c3_oracle = to_daily_matrix(step2["C3_rolling90_linear"].reindex(h.index)).asfreq("D")
        self.days = self.price.index

    def mustrun_res_exante(self) -> pd.DataFrame:
        """Must-run RES/hydro/waste supply proxy for D: mean of the same hour over D-8..D-2 (realised)."""
        stack = [shift_days(self.mustrun_res_real, k) for k in range(2, 9)]
        return sum(stack) / len(stack)


def dow_dummies(days: pd.DatetimeIndex) -> np.ndarray:
    return np.eye(7)[days.dayofweek]
