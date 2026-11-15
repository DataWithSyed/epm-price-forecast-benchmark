"""Rolling-origin DE-LU day-ahead price forecast benchmark (step 5 of the European Power Market Lab).

Test: every delivery day 2026-01-01 .. 2026-10-31 (304 days x 24 h), information set at D-1 12:00.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # avoid BLAS over-subscription (much faster for 24 small Lasso fits)

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from epmfc.data import load_forecasts, load_installed, load_panel, load_step2_models  # noqa: E402
from epmfc.features import Matrices, contemporaneous_fuels, lagged_fuels  # noqa: E402
from epmfc.metrics import dm_matrix, dm_test, point_metrics  # noqa: E402
from epmfc.models import (gbm_forecast, lear_forecast, naive_d1, naive_lago, naive_w1,  # noqa: E402
                          ols_drivers, ols_drivers_rolling, rolling_linear_recal)
from epmfc.paths import REFERENCE, RESULTS, TEST_END, TEST_START, TRAIN_START, TZ  # noqa: E402
from epmfc.stack import stack_prices, thermal_mustrun  # noqa: E402

CACHE = RESULTS / "cache"
LEAR_REFIT = 1  # recalibrate LEAR every test day (Lago et al. style)
EXANTE = ["naive_d1", "naive_w1", "naive_lago", "stack_M0_exante", "stack_C3_exante", "ols_drivers_exante",
          "ols_drivers_rolling", "lear", "lear_nb", "gbm", "gbm_nb", "ens", "ens_nb"]
ORACLE = ["stack_M0_oracle_step2", "stack_C3_oracle_step2", "ols_drivers_oracle", "lear_oracle"]
PERIODS = {"test_2026_01_10": ("2026-01-01", "2026-10-31"),
           "H1_2026_01_05": ("2026-01-01", "2026-05-31"),
           "H2_2026_06_10": ("2026-06-01", "2026-10-31")}


def eua_quarterly_contemporaneous(days: pd.DatetimeIndex) -> pd.Series:
    """ORACLE step-3 EUA input: 2025 annual VWAP, 2026 DEHSt quarterly VWAP (Oct = Q3 carried)."""
    q = pd.read_csv(REFERENCE / "eua_dehst_quarterly.csv", parse_dates=["start", "end"])
    out = pd.Series(np.nan, index=days)
    for _, r in q.iterrows():
        out[(days >= r["start"]) & (days <= r["end"])] = r["eua_eur_t"]
    return out


def cached(name: str, fn, refresh: bool):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}.csv"
    if p.exists() and not refresh:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.columns = df.columns.astype(int)
        return df
    t = time.time()
    res = fn()
    df = res[0] if isinstance(res, tuple) else res
    df.to_csv(p)
    if isinstance(res, tuple):
        res[1].to_csv(CACHE / f"{name}_aux.csv")
    print(f"  {name}: {time.time() - t:.0f}s", flush=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute cached LEAR/GBM forecasts")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    h, f, s2 = load_panel(), load_forecasts(), load_step2_models()
    M = Matrices(h, f, s2)
    days = M.days
    test_days = pd.date_range(TEST_START, TEST_END, freq="D")
    train_days = days[(days >= TRAIN_START) & (days < TEST_START)]
    y = M.price.loc[test_days]
    fu = lagged_fuels(days)
    fu_or = contemporaneous_fuels(days)
    fu_or_q = fu_or.copy()
    fu_or_q["eua_eur_t"] = eua_quarterly_contemporaneous(days)

    F: dict[str, pd.DataFrame] = {}
    F["naive_d1"] = naive_d1(M).loc[test_days]
    F["naive_w1"] = naive_w1(M).loc[test_days]
    F["naive_lago"] = naive_lago(M).loc[test_days]

    # fundamentals: step-2 stack with ex-ante inputs
    loc = h.index.tz_convert(TZ)
    mr = thermal_mustrun(h[(loc >= pd.Timestamp(TRAIN_START, tz=TZ)) & (loc < pd.Timestamp(TEST_START, tz=TZ))])
    inst = load_installed()
    demand_ex = M.rl_fc - M.mustrun_res_exante()
    m0_ex = stack_prices(demand_ex.loc[days[days >= "2025-01-09"]], fu, inst, mr)
    F["stack_M0_exante"] = m0_ex.loc[test_days]
    F["stack_C3_exante"] = rolling_linear_recal(M, m0_ex, test_days, 90)
    F["ols_drivers_exante"], ols_ex_coef = ols_drivers(M, M.rl_fc, fu, train_days, test_days)
    F["ols_drivers_rolling"] = ols_drivers_rolling(M, M.rl_fc, fu, test_days, 364)

    F["lear"] = cached("lear", lambda: lear_forecast(M, fu, test_days, "base", refit_every=LEAR_REFIT), a.refresh)
    F["lear_nb"] = cached("lear_nb", lambda: lear_forecast(M, fu, test_days, "nb", refit_every=LEAR_REFIT), a.refresh)
    F["gbm"] = cached("gbm", lambda: gbm_forecast(M, fu, test_days, "base"), a.refresh)
    F["gbm_nb"] = cached("gbm_nb", lambda: gbm_forecast(M, fu, test_days, "nb"), a.refresh)
    F["ens"] = (F["lear"] + F["gbm"]) / 2
    F["ens_nb"] = (F["lear_nb"] + F["gbm_nb"]) / 2

    # oracle references (use information NOT available at D-1)
    F["stack_M0_oracle_step2"] = M.m0_oracle.loc[test_days]
    F["stack_C3_oracle_step2"] = M.c3_oracle.loc[test_days]
    F["ols_drivers_oracle"], ols_or_coef = ols_drivers(M, M.rl_real, fu_or_q, train_days, test_days)
    F["lear_oracle"] = cached("lear_oracle", lambda: lear_forecast(M, fu, test_days, "oracle", refit_every=LEAR_REFIT), a.refresh)

    for k, v in F.items():
        v.columns = range(24)
        if v.isna().any().any():
            raise RuntimeError(f"{k} has missing forecasts")

    # ---------- metrics ----------
    rows = []
    for pname, (s, e) in PERIODS.items():
        yy = y.loc[s:e]
        for k in EXANTE + ORACLE:
            rows.append({"period": pname, "model": k, "track": "ex-ante" if k in EXANTE else "ORACLE",
                         **point_metrics(yy, F[k].loc[s:e], F["naive_lago"].loc[s:e])})
    met = pd.DataFrame(rows)
    met.round(4).to_csv(RESULTS / "metrics.csv", index=False)

    months = y.index.to_period("M")
    bym = []
    for m in months.unique():
        sel = months == m
        for k in EXANTE + ORACLE:
            pm = point_metrics(y[sel], F[k][sel])
            bym.append({"month": str(m), "model": k, "mae": pm["mae"], "bias": pm["bias"]})
    bym = pd.DataFrame(bym)
    bym.pivot(index="month", columns="model", values="mae").round(2).to_csv(RESULTS / "mae_by_month.csv")
    bym.pivot(index="month", columns="model", values="bias").round(2).to_csv(RESULTS / "bias_by_month.csv")
    byh = pd.DataFrame({k: (F[k] - y).abs().mean() for k in EXANTE + ORACLE}).round(2)
    byh.index.name = "hour_local"
    byh.to_csv(RESULTS / "mae_by_hour.csv")

    dm_models = ["naive_d1", "naive_lago", "stack_C3_exante", "ols_drivers_rolling", "lear", "lear_nb", "gbm", "gbm_nb", "ens", "ens_nb"]
    for pname, (s, e) in PERIODS.items():
        dm_matrix(y.loc[s:e], {k: F[k].loc[s:e] for k in dm_models}).round(5).to_csv(RESULTS / f"dm_pvalues_{pname}.csv")
    # DM of every ex-ante model against the best baseline (lowest full-test MAE among naive/stack/OLS)
    baselines = ["naive_d1", "naive_w1", "naive_lago", "stack_M0_exante", "stack_C3_exante",
                 "ols_drivers_exante", "ols_drivers_rolling"]
    full_mae = met[met.period == "test_2026_01_10"].set_index("model")["mae"]
    best_base = str(full_mae.loc[baselines].idxmin())
    dmb = []
    for k in EXANTE:
        if k == best_base:
            continue
        for pname, (s, e) in PERIODS.items():
            r = dm_test(y.loc[s:e], F[best_base].loc[s:e], F[k].loc[s:e])
            dmb.append({"model": k, "vs": best_base, "period": pname, "dm_stat": r["stat"],
                        "p_model_better": r["p_b_better"], "mae_diff_eur_mwh": -r["mean_daily_loss_diff"]})
    pd.DataFrame(dmb).round(5).to_csv(RESULTS / "dm_vs_best_baseline.csv", index=False)
    nb_rows = []
    for base, nb in (("lear", "lear_nb"), ("gbm", "gbm_nb"), ("ens", "ens_nb")):
        for pname, (s, e) in PERIODS.items():
            r1 = dm_test(y.loc[s:e], F[base].loc[s:e], F[nb].loc[s:e])
            r2 = dm_test(y.loc[s:e], F[nb].loc[s:e], F[base].loc[s:e])
            nb_rows.append({"pair": f"{nb} vs {base}", "period": pname,
                            "mae_base": point_metrics(y.loc[s:e], F[base].loc[s:e])["mae"],
                            "mae_nb": point_metrics(y.loc[s:e], F[nb].loc[s:e])["mae"],
                            "p_nb_better": r1["p_b_better"], "p_base_better": r2["p_b_better"]})
    pd.DataFrame(nb_rows).round(5).to_csv(RESULTS / "neighbour_coupling_dm.csv", index=False)

    # LEAR sparsity by block
    nz = {}
    for k in ("lear", "lear_nb", "lear_oracle"):
        p = CACHE / f"{k}_aux.csv"
        if p.exists():
            nz[k] = pd.read_csv(p, index_col=0).mean().round(4).to_dict()
    pd.DataFrame(nz).to_csv(RESULTS / "lear_nonzero_share_by_block.csv")

    # input-forecast error (TSO day-ahead vs realised), test window
    rl_err = (M.rl_fc - M.rl_real).loc[test_days]
    fc_diag = {pn: {"rl_fc_bias_mw": float(rl_err.loc[s:e].stack().mean()),
                    "rl_fc_mae_mw": float(rl_err.loc[s:e].abs().stack().mean())} for pn, (s, e) in PERIODS.items()}

    long = []
    for k in EXANTE + ORACLE:
        t = F[k].stack().rename("forecast").reset_index()
        t.columns = ["day", "hour_local", "forecast"]
        t["model"] = k
        long.append(t)
    act = y.stack().rename("actual").reset_index()
    act.columns = ["day", "hour_local", "actual"]
    lf = pd.concat(long).pivot_table(index=["day", "hour_local"], columns="model", values="forecast").reset_index()
    lf = act.merge(lf, on=["day", "hour_local"])
    lf.round(2).to_csv(RESULTS / "forecasts_test.csv.gz", index=False, compression="gzip")

    full = met[met.period == "test_2026_01_10"].set_index("model")
    summary = {
        "test_days": len(test_days), "test_cells": int(y.size),
        "realised": {pn: {"mean": float(y.loc[s:e].stack().mean()), "std": float(y.loc[s:e].stack().std()),
                          "negative_share": float((y.loc[s:e].stack() < 0).mean())} for pn, (s, e) in PERIODS.items()},
        "thermal_mustrun_mw_2025_q02": {k: round(v, 1) for k, v in mr.items()},
        "ols_drivers_exante_coefs": ols_ex_coef, "ols_drivers_oracle_coefs": ols_or_coef,
        "input_forecast_error": fc_diag,
        "metrics": {pn: met[met.period == pn].set_index("model").drop(columns=["period"]).round(3).to_dict(orient="index")
                    for pn in PERIODS},
        "best_exante_full": str(full.loc[EXANTE, "mae"].idxmin()), "best_baseline_full": best_base,
        "settings": {"lear": "24 LassoLarsIC(AIC), asinh-median/MAD, 364-day window, refit every test day",
                     "gbm": "HistGradientBoosting max_iter=400 lr=0.05 leaves=31 min_leaf=40 l2=1, 364-day window, refit every 7 test days",
                     "stack_C3_exante": "a+b*M0_exante refit daily on trailing 90 days",
                     "ols_drivers_rolling": "step-3 OLS refit daily on trailing 364 days (ex-ante inputs)",
                     "fuel_publication_day": 4},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(met[met.period == "test_2026_01_10"][["model", "track", "mae", "rmse", "smape_pct", "rmae", "bias"]].round(2).to_string(index=False))
    print(met[met.period == "H2_2026_06_10"][["model", "mae", "rmae", "bias"]].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
