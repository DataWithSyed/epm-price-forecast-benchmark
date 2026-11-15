"""Forecasting models for the rolling-origin benchmark. All are ex-ante unless named *oracle*."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoLarsIC, LinearRegression

from epmfc.features import Matrices, dow_dummies, shift_days


# ---------------- naive benchmarks ----------------
def naive_d1(M: Matrices) -> pd.DataFrame:
    return shift_days(M.price, 1)


def naive_w1(M: Matrices) -> pd.DataFrame:
    return shift_days(M.price, 7)


def naive_lago(M: Matrices) -> pd.DataFrame:
    """D-7 for Mondays, Saturdays and Sundays, D-1 otherwise (Lago et al. 2021 naive)."""
    d1, w1 = naive_d1(M), naive_w1(M)
    use_w = np.isin(M.price.index.dayofweek, [0, 5, 6])
    out = d1.copy()
    out.loc[use_w] = w1.loc[use_w]
    return out


# ---------------- variance-stabilising transform (Uniejewski et al. 2018) ----------------
class AsinhScaler:
    def fit(self, x: np.ndarray) -> "AsinhScaler":
        self.med = float(np.median(x))
        mad = float(np.median(np.abs(x - self.med))) / 0.6745
        self.mad = mad if mad > 1e-6 else 1.0
        return self

    def transform(self, x):
        return np.arcsinh((np.asarray(x) - self.med) / self.mad)

    def inverse(self, z):
        return np.sinh(np.asarray(z)) * self.mad + self.med


# ---------------- LEAR-style design ----------------
LEAR_PRICE_LAGS = (1, 2, 3, 7)


def lear_blocks(M: Matrices, variant: str) -> dict[str, pd.DataFrame]:
    """Feature blocks (each day x 24) for LEAR. variant: base | nb | oracle."""
    if variant == "oracle":
        load, res = M.load_real, M.res_real   # ORACLE: realised day-D load and wind+solar
    else:
        load, res = M.load_fc, M.res_fc
    blocks = {f"price_d{k}": shift_days(M.price, k) for k in LEAR_PRICE_LAGS}
    blocks["load_d0"] = load
    blocks["res_d0"] = res
    blocks["load_d1"] = shift_days(load, 1)
    blocks["res_d1"] = shift_days(res, 1)
    if variant == "nb":
        blocks["nb_rl_d0"] = M.nb_rl_fc
        blocks["nb_res_d0"] = M.nb_res_fc
        blocks["trade_d2"] = shift_days(M.trade_real, 2)
    return blocks


def _design(blocks, fuels, days):
    parts = [blocks[k].reindex(days).to_numpy(float) for k in blocks]
    parts.append(fuels.reindex(days)[["gas_eur_mwh_th", "coal_eur_mwh_th", "eua_eur_t"]].to_numpy(float))
    parts.append(dow_dummies(days))
    names = [f"{k}_h{h}" for k in blocks for h in range(24)] + ["gas", "coal", "eua"] + [f"dow{i}" for i in range(7)]
    return np.hstack(parts), names


def lear_forecast(M: Matrices, fuels: pd.DataFrame, test_days: pd.DatetimeIndex, variant: str = "base",
                  window: int = 364, refit_every: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """24 hourly LassoLarsIC(AIC) models on asinh-scaled prices.

    Recalibrated every ``refit_every`` test days on the trailing ``window`` days ending D-1;
    every test day is still forecast from its own D-1 information set (lags, D forecasts).
    Returns forecasts and, per refit, the share of non-zero coefficients by feature block."""
    blocks = lear_blocks(M, variant)
    price_keys = [k for k in blocks if k.startswith("price")]
    all_days = M.price.index
    X_all, names = _design(blocks, fuels, all_days)
    Y_all = M.price.to_numpy(float)
    block_of = np.array([n.rsplit("_h", 1)[0] if "_h" in n else ("fuel" if n in ("gas", "coal", "eua") else "dow")
                         for n in names])
    is_price = np.isin(block_of, price_keys)
    out = pd.DataFrame(np.nan, index=test_days, columns=range(24))
    nz_rows = []
    pos = {d: i for i, d in enumerate(all_days)}
    models, sc, mu, sd = None, None, None, None
    for k, d in enumerate(test_days):
        i = pos[d]
        xte = X_all[i:i + 1].copy()
        if k % refit_every == 0:
            models, sc, mu, sd, nz = _fit_lear(X_all, Y_all, i, window, is_price, block_of)
            nz_rows.append({"day": d, **nz})
        xte[:, is_price] = sc.transform(xte[:, is_price])
        xte = (xte - mu) / sd
        out.loc[d] = [float(sc.inverse(m.predict(xte))[0]) for m in models]
    return out, pd.DataFrame(nz_rows).set_index("day")


def _fit_lear(X_all, Y_all, i, window, is_price, block_of):
    """Fit 24 hourly LassoLarsIC models on days [i-window, i-1] (all known at D-1)."""
    tr = np.arange(max(0, i - window), i)
    Xtr, Ytr = X_all[tr], Y_all[tr]
    ok = ~np.isnan(Xtr).any(axis=1) & ~np.isnan(Ytr).any(axis=1)
    Xtr, Ytr = Xtr[ok], Ytr[ok]
    sc = AsinhScaler().fit(Ytr.ravel())
    Xs = Xtr.copy()
    Xs[:, is_price] = sc.transform(Xs[:, is_price])
    mu, sd = Xs.mean(0), Xs.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = (Xs - mu) / sd
    nz = {}
    models = []
    for hh in range(24):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LassoLarsIC(criterion="aic", max_iter=500).fit(Xs, sc.transform(Ytr[:, hh]))
        models.append(m)
        for b in np.unique(block_of):
            nz[b] = nz.get(b, 0.0) + float((m.coef_[block_of == b] != 0).mean()) / 24
    return models, sc, mu, sd, nz


# ---------------- GBM (long format) ----------------
def gbm_long(M: Matrices, fuels: pd.DataFrame, variant: str = "base") -> tuple[pd.DataFrame, list[str]]:
    days = M.price.index
    rows = {}
    hours = np.tile(np.arange(24), len(days))
    rep = lambda mat: mat.reindex(days).to_numpy(float).ravel()  # noqa: E731
    p1 = shift_days(M.price, 1)
    rows["hour"] = hours
    rows["dow"] = np.repeat(days.dayofweek, 24)
    rows["price_d1"] = rep(p1)
    rows["price_d2"] = rep(shift_days(M.price, 2))
    rows["price_d7"] = rep(shift_days(M.price, 7))
    rows["price_d1_mean"] = np.repeat(p1.reindex(days).mean(axis=1).to_numpy(), 24)
    rows["price_d1_max"] = np.repeat(p1.reindex(days).max(axis=1).to_numpy(), 24)
    rows["price_d1_min"] = np.repeat(p1.reindex(days).min(axis=1).to_numpy(), 24)
    rows["load_fc"] = rep(M.load_fc)
    rows["res_fc"] = rep(M.res_fc)
    rows["rl_fc"] = rep(M.rl_fc)
    rows["rl_fc_change_d1"] = rep(M.rl_fc - shift_days(M.rl_fc, 1))
    rows["rl_fc_day_mean"] = np.repeat(M.rl_fc.reindex(days).mean(axis=1).to_numpy(), 24)
    for c in ("gas_eur_mwh_th", "coal_eur_mwh_th", "eua_eur_t"):
        rows[c] = np.repeat(fuels.reindex(days)[c].to_numpy(float), 24)
    if variant == "nb":
        rows["nb_rl_fc"] = rep(M.nb_rl_fc)
        rows["nb_res_fc"] = rep(M.nb_res_fc)
        rows["nb_rl_fc_day_mean"] = np.repeat(M.nb_rl_fc.reindex(days).mean(axis=1).to_numpy(), 24)
        rows["trade_d2"] = rep(shift_days(M.trade_real, 2))
        rows["trade_d2_day_mean"] = np.repeat(shift_days(M.trade_real, 2).reindex(days).mean(axis=1).to_numpy(), 24)
    X = pd.DataFrame(rows)
    X["y"] = M.price.reindex(days).to_numpy(float).ravel()
    X["day"] = np.repeat(days, 24)
    return X, [c for c in rows]


def gbm_forecast(M: Matrices, fuels: pd.DataFrame, test_days: pd.DatetimeIndex, variant: str = "base",
                 window: int = 364, refit_every: int = 7, seed: int = 0) -> pd.DataFrame:
    """HistGradientBoosting (fixed, untuned hyper-parameters) refit every ``refit_every`` test days
    on the trailing ``window`` days ending D-1."""
    X, feats = gbm_long(M, fuels, variant)
    out = pd.DataFrame(np.nan, index=test_days, columns=range(24))
    model = None
    for k, d in enumerate(test_days):
        if k % refit_every == 0:
            tr = X[(X["day"] < d) & (X["day"] >= d - pd.Timedelta(days=window))].dropna()
            model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                  min_samples_leaf=40, l2_regularization=1.0,
                                                  categorical_features=[0, 1], random_state=seed)
            model.fit(tr[feats].to_numpy(float), tr["y"].to_numpy(float))
        te = X[X["day"] == d]
        out.loc[d] = model.predict(te[feats].to_numpy(float))
    return out


# ---------------- fundamentals-based ----------------
def rolling_linear_recal(M: Matrices, base: pd.DataFrame, test_days: pd.DatetimeIndex,
                         window: int = 90) -> pd.DataFrame:
    """C3-style a + b * stack, refit every test day on the trailing ``window`` days ending D-1."""
    out = pd.DataFrame(np.nan, index=test_days, columns=range(24))
    for d in test_days:
        tr_days = base.index[(base.index < d) & (base.index >= d - pd.Timedelta(days=window))]
        x = base.loc[tr_days].to_numpy(float).ravel()
        y = M.price.loc[tr_days].to_numpy(float).ravel()
        ok = ~np.isnan(x) & ~np.isnan(y)
        b, a = np.polyfit(x[ok], y[ok], 1)
        out.loc[d] = a + b * base.loc[d].to_numpy(float)
    return out


def _ols_design(rl: pd.DataFrame, fuels: pd.DataFrame, days) -> np.ndarray:
    f = fuels.reindex(days)
    return np.column_stack([rl.reindex(days).to_numpy(float).ravel(),
                            np.repeat(f["gas_eur_mwh_th"].to_numpy(float), 24),
                            np.repeat(f["coal_eur_mwh_th"].to_numpy(float), 24),
                            np.repeat(f["eua_eur_t"].to_numpy(float), 24)])


def ols_drivers_rolling(M: Matrices, rl: pd.DataFrame, fuels: pd.DataFrame, test_days: pd.DatetimeIndex,
                        window: int = 364) -> pd.DataFrame:
    """Step-3 driver OLS refit every test day on the trailing ``window`` days ending D-1 (ex-ante)."""
    out = pd.DataFrame(np.nan, index=test_days, columns=range(24))
    for d in test_days:
        tr = M.price.index[(M.price.index < d) & (M.price.index >= d - pd.Timedelta(days=window))]
        X, y = _ols_design(rl, fuels, tr), M.price.reindex(tr).to_numpy(float).ravel()
        ok = ~np.isnan(X).any(1) & ~np.isnan(y)
        lr = LinearRegression().fit(X[ok], y[ok])
        out.loc[d] = lr.predict(_ols_design(rl, fuels, pd.DatetimeIndex([d])))
    return out


def ols_drivers(M: Matrices, rl: pd.DataFrame, fuels: pd.DataFrame, train_days, test_days) -> tuple[pd.DataFrame, dict]:
    """Step-3 driver OLS: price ~ residual load + gas + coal + EUA (hourly, static fit on train days)."""
    def design(days):
        n = len(days)
        f = fuels.reindex(days)
        return np.column_stack([rl.reindex(days).to_numpy(float).ravel(),
                                np.repeat(f["gas_eur_mwh_th"].to_numpy(float), 24),
                                np.repeat(f["coal_eur_mwh_th"].to_numpy(float), 24),
                                np.repeat(f["eua_eur_t"].to_numpy(float), 24)]), n
    Xtr, _ = design(train_days)
    ytr = M.price.reindex(train_days).to_numpy(float).ravel()
    ok = ~np.isnan(Xtr).any(1) & ~np.isnan(ytr)
    lr = LinearRegression().fit(Xtr[ok], ytr[ok])
    Xte, n = design(test_days)
    pred = lr.predict(np.nan_to_num(Xte)).reshape(n, 24)
    coefs = dict(zip(["intercept", "rl_mw", "gas", "coal", "eua"], [float(lr.intercept_), *map(float, lr.coef_)]))
    return pd.DataFrame(pred, index=test_days, columns=range(24)), coefs
