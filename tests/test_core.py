import numpy as np
import pandas as pd
import pytest

from epmfc.data import load_forecasts, load_panel, to_daily_matrix
from epmfc.features import Matrices, latest_month_known, lagged_fuels, shift_days
from epmfc.metrics import dm_test, point_metrics
from epmfc.models import AsinhScaler, lear_forecast, naive_lago, rolling_linear_recal
from epmfc.stack import build_stack, clear


def _toy_matrices(n_days=420, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=n_days, freq="D")
    hours = np.arange(24)
    load = 50000 + 8000 * np.sin(2 * np.pi * hours / 24)[None, :] + rng.normal(0, 1500, (n_days, 24))
    res = np.clip(rng.normal(15000, 6000, (n_days, 24)), 0, None)
    price = 20 + 0.004 * (load - res) + rng.normal(0, 5, (n_days, 24))
    M = Matrices.__new__(Matrices)
    mk = lambda a: pd.DataFrame(a, index=days, columns=range(24))  # noqa: E731
    M.price, M.load_fc, M.res_fc = mk(price), mk(load), mk(res)
    M.rl_fc = M.load_fc - M.res_fc
    M.load_real, M.res_real = M.load_fc, M.res_fc
    M.nb_rl_fc, M.nb_res_fc, M.trade_real = mk(load * 2), mk(res), mk(np.zeros_like(load))
    M.days = days
    fuels = pd.DataFrame({"gas_eur_mwh_th": 35.0, "coal_eur_mwh_th": 12.0, "eua_eur_t": 75.0,
                          "oil_eur_mwh_th": 40.0}, index=days)
    return M, fuels


def test_dst_matrix_is_24h_and_preserves_values():
    idx = pd.date_range("2026-03-28 23:00", "2026-03-30 21:00", freq="1h", tz="UTC")  # around spring DST
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    m = to_daily_matrix(s)
    assert m.shape[1] == 24 and not m.isna().any().any()
    spring = m.loc["2026-03-29"]
    assert spring[2] == pytest.approx((spring[1] + spring[3]) / 2)  # interpolated missing hour
    idx2 = pd.date_range("2026-10-24 22:00", "2026-10-25 22:00", freq="1h", tz="UTC")  # 25-h autumn day
    m2 = to_daily_matrix(pd.Series(1.0, index=idx2))
    assert (m2.loc["2026-10-25"] == 1.0).all()


def test_publication_lag_for_monthly_benchmarks():
    assert str(latest_month_known(pd.Timestamp("2026-07-03"))) == "2026-05"
    assert str(latest_month_known(pd.Timestamp("2026-07-04"))) == "2026-06"
    f = lagged_fuels(pd.DatetimeIndex(["2026-01-01", "2026-01-05", "2026-10-31"]))
    assert list(f["fuel_month"]) == ["2025-11", "2025-12", "2026-09"]


def test_naive_lago_uses_week_lag_on_mon_sat_sun():
    M, _ = _toy_matrices(30)
    nv = naive_lago(M)
    for d in M.days[8:]:
        lag = 7 if d.dayofweek in (0, 5, 6) else 1
        np.testing.assert_allclose(nv.loc[d], M.price.loc[d - pd.Timedelta(days=lag)])


def test_shift_days_never_looks_ahead():
    M, _ = _toy_matrices(20)
    s = shift_days(M.price, 1)
    assert s.iloc[0].isna().all()
    np.testing.assert_allclose(s.iloc[5], M.price.iloc[4])


def test_lear_forecast_is_ex_ante():
    """Changing prices on/after the test day must not change the forecast for that day."""
    M, fu = _toy_matrices(400)
    d = pd.DatetimeIndex([M.days[380]])
    f1, _ = lear_forecast(M, fu, d, "base", window=364, refit_every=1)
    M.price.iloc[380:] += 1000.0
    f2, _ = lear_forecast(M, fu, d, "base", window=364, refit_every=1)
    np.testing.assert_allclose(f1.to_numpy(), f2.to_numpy())


def test_lear_beats_naive_on_fundamental_toy():
    M, fu = _toy_matrices(420)
    td = M.days[380:400]
    f, nz = lear_forecast(M, fu, td, "base", window=364, refit_every=7)
    y = M.price.loc[td]
    assert point_metrics(y, f)["mae"] < 0.8 * point_metrics(y, naive_lago(M).loc[td])["mae"]
    assert nz["load_d0"].mean() > 0


def test_rolling_recal_is_out_of_sample():
    M, _ = _toy_matrices(200)
    base = M.price * 0.5 + 3.0
    td = M.days[150:155]
    r1 = rolling_linear_recal(M, base, td, 90)
    M.price.iloc[152:] += 500.0
    r2 = rolling_linear_recal(M, base, td, 90)
    np.testing.assert_allclose(r1.iloc[:3].to_numpy(), r2.iloc[:3].to_numpy())


def test_asinh_scaler_roundtrip():
    x = np.array([-50.0, 0.0, 10.0, 100.0, 600.0])
    sc = AsinhScaler().fit(x)
    np.testing.assert_allclose(sc.inverse(sc.transform(x)), x)


def test_metrics_and_dm_direction():
    idx = pd.date_range("2026-01-01", periods=60, freq="D")
    rng = np.random.default_rng(1)
    y = pd.DataFrame(rng.normal(100, 20, (60, 24)), index=idx)
    good = y + rng.normal(0, 2, y.shape)
    bad = y + rng.normal(0, 20, y.shape)
    m = point_metrics(y, good, naive=bad)
    assert m["rmae"] < 0.2 and m["mae"] < 2
    assert dm_test(y, bad, good)["p_b_better"] < 1e-6
    assert dm_test(y, good, bad)["p_b_better"] > 0.99


def test_stack_monotone_and_surplus():
    inst = pd.Series({"fossil_brown_coal_lignite": 15.0, "fossil_hard_coal": 15.0, "fossil_gas": 35.0, "fossil_oil": 3.5})
    st = build_stack({"gas_eur_mwh_th": 35.0, "coal_eur_mwh_th": 12.0, "oil_eur_mwh_th": 45.0, "eua_eur_t": 75.0},
                     inst, {"lignite": 2000.0})
    p = clear(np.array([-1000.0, 1000.0, 10000.0, 30000.0, 45000.0]), st)
    assert p[0] == 0.0 and np.all(np.diff(p) >= 0)


def test_processed_panels_align():
    h, f = load_panel(), load_forecasts()
    assert len(h) == 16056
    assert h.index.isin(f.index).all()
    M = Matrices(h, f)
    assert len(M.days) == 669 and not M.rl_fc.loc["2025-01-01":].isna().any().any()


def test_rolling_driver_ols_is_ex_ante():
    from epmfc.models import ols_drivers_rolling
    M, fu = _toy_matrices(400)
    td = M.days[380:383]
    f1 = ols_drivers_rolling(M, M.rl_fc, fu, td, 364)
    M.price.iloc[381:] += 1000.0
    f2 = ols_drivers_rolling(M, M.rl_fc, fu, td, 364)
    np.testing.assert_allclose(f1.iloc[:2].to_numpy(), f2.iloc[:2].to_numpy())
    assert not np.allclose(f1.iloc[2].to_numpy(), f2.iloc[2].to_numpy())
