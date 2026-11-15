"""Isolated-zone DE thermal merit order (vendored from epm-merit-order-model, step 2, MIT).

Same fleet, ASSUMED efficiency bands, availabilities, VOM and IPCC 2006 CO2 factors
as step 2; only the *inputs* change here (ex-ante demand and lagged fuels).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EF_T_PER_MWH_TH = {"lignite": 101_000 * 3.6e-6, "hard_coal": 94_600 * 3.6e-6,
                   "gas": 56_100 * 3.6e-6, "oil": 74_100 * 3.6e-6}
LIGNITE_FUEL = 4.0  # ASSUMED mine-mouth marginal fuel cost, EUR/MWh_th (step 2)
N_BANDS = 10
THERMAL_OBS = {"lignite": "lignite_mw", "hard_coal": "hard_coal_mw", "gas": "gas_mw", "oil": "oil_mw"}


@dataclass(frozen=True)
class Tech:
    name: str
    fuel: str
    key: str
    share: float
    eff_low: float
    eff_high: float
    avail: float
    vom: float
    adder: float = 0.0


TECHS = (
    Tech("lignite", "lignite", "fossil_brown_coal_lignite", 1.0, 0.31, 0.43, 0.85, 4.0),
    Tech("hard_coal", "hard_coal", "fossil_hard_coal", 1.0, 0.34, 0.46, 0.65, 4.0, 2.0),
    Tech("gas_ccgt", "gas", "fossil_gas", 0.60, 0.46, 0.60, 0.85, 3.0, 2.0),
    Tech("gas_peaker", "gas", "fossil_gas", 0.40, 0.30, 0.42, 0.85, 5.0, 2.0),
    Tech("oil", "oil", "fossil_oil", 1.0, 0.28, 0.38, 0.50, 5.0, 10.0),
)


def thermal_mustrun(h_train: pd.DataFrame, q: float = 0.02) -> dict[str, float]:
    return {f: float(h_train[c].quantile(q)) for f, c in THERMAL_OBS.items()}


def build_stack(fuels: dict, installed: pd.Series, mustrun: dict[str, float]) -> pd.DataFrame:
    rows = []
    price = {"lignite": LIGNITE_FUEL, "hard_coal": fuels["coal_eur_mwh_th"],
             "gas": fuels["gas_eur_mwh_th"], "oil": fuels["oil_eur_mwh_th"]}
    for t in TECHS:
        cap = float(installed[t.key]) * 1000 * t.share
        avail = cap * min(1.0, t.avail)
        fuel_cap = sum(float(installed[x.key]) * 1000 * x.share for x in TECHS if x.fuel == t.fuel)
        mr = min(mustrun.get(t.fuel, 0.0) * (cap / fuel_cap if fuel_cap else 0.0), avail)
        if mr > 0:
            rows.append((t.name + "_mustrun", np.nan, 0.0, mr))
        effs = np.clip(np.linspace(t.eff_high, t.eff_low, N_BANDS), 0.15, 0.65)
        fp = price[t.fuel] + t.adder
        ef = EF_T_PER_MWH_TH[t.fuel]
        for e in effs:
            rows.append((t.name, e, fp / e + ef / e * fuels["eua_eur_t"] + t.vom, (avail - mr) / N_BANDS))
    st = pd.DataFrame(rows, columns=["tech", "eff", "srmc", "cap_mw"])
    st = st.sort_values("srmc", kind="stable").reset_index(drop=True)
    st["cum_mw"] = st["cap_mw"].cumsum()
    return st


def clear(demand: np.ndarray, st: pd.DataFrame) -> np.ndarray:
    cum = st["cum_mw"].to_numpy()
    srmc = st["srmc"].to_numpy()
    d = np.asarray(demand, dtype=float)
    idx = np.clip(np.searchsorted(cum, np.clip(d, 0, None), side="left"), 0, len(cum) - 1)
    lo_cum = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)
    lo_p = np.where(idx > 0, srmc[np.maximum(idx - 1, 0)], srmc[0])
    frac = np.clip((d - lo_cum) / np.maximum(cum[idx] - lo_cum, 1e-9), 0, 1)
    p = lo_p + frac * (srmc[idx] - lo_p)
    return np.where(d <= 0, 0.0, p)


def stack_prices(demand: pd.DataFrame, fuels: pd.DataFrame, installed: pd.DataFrame,
                 mustrun: dict[str, float]) -> pd.DataFrame:
    """Day x 24h stack prices; ``fuels`` is indexed by day with gas/coal/oil/eua columns."""
    out = pd.DataFrame(np.nan, index=demand.index, columns=demand.columns)
    cache: dict = {}
    years = installed.dropna(how="all").index
    for d in demand.index:
        if d not in fuels.index or demand.loc[d].isna().any():
            continue
        fr = fuels.loc[d]
        yr = d.year if d.year in installed.index else max(years)
        key = (yr, round(fr["gas_eur_mwh_th"], 6), round(fr["coal_eur_mwh_th"], 6),
               round(fr["oil_eur_mwh_th"], 6), round(fr["eua_eur_t"], 6))
        if key not in cache:
            cache[key] = build_stack(fr, installed.loc[yr], mustrun)
        out.loc[d] = clear(demand.loc[d].to_numpy(), cache[key])
    return out
