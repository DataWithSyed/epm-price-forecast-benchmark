"""Point-forecast metrics and Diebold-Mariano tests on (day x 24h) matrices."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _aligned(y: pd.DataFrame, f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    idx = y.index.intersection(f.index)
    return y.loc[idx].to_numpy(float), f.loc[idx].to_numpy(float)


def point_metrics(y: pd.DataFrame, f: pd.DataFrame, naive: pd.DataFrame | None = None) -> dict:
    a, b = _aligned(y, f)
    e = b - a
    out = {"mae": float(np.nanmean(np.abs(e))),
           "rmse": float(np.sqrt(np.nanmean(e ** 2))),
           "smape_pct": float(100 * np.nanmean(2 * np.abs(e) / np.maximum(np.abs(a) + np.abs(b), 1e-9))),
           "bias": float(np.nanmean(e)),
           "n_days": int(len(a))}
    if naive is not None:
        a2, n = _aligned(y, naive.loc[y.index.intersection(f.index)])
        out["rmae"] = out["mae"] / float(np.nanmean(np.abs(n - a2)))
    return out


def dm_test(y: pd.DataFrame, fa: pd.DataFrame, fb: pd.DataFrame, norm: int = 1) -> dict:
    """Multivariate DM test (Lago et al. 2021, Appl. Energy 293:116983).

    Daily loss L_d = ||e_d||_norm over the 24 hours. d_t = L_A - L_B.
    One-sided p-value for H1: forecast B is more accurate than A (mean d > 0).
    """
    idx = y.index.intersection(fa.index).intersection(fb.index)
    ea = np.abs(fa.loc[idx].to_numpy(float) - y.loc[idx].to_numpy(float))
    eb = np.abs(fb.loc[idx].to_numpy(float) - y.loc[idx].to_numpy(float))
    if norm == 1:
        la, lb = ea.sum(axis=1), eb.sum(axis=1)
    else:
        la, lb = (ea ** 2).sum(axis=1), (eb ** 2).sum(axis=1)
    d = la - lb
    n = len(d)
    sd = d.std(ddof=1)
    stat = d.mean() / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return {"stat": float(stat), "p_b_better": float(1 - stats.norm.cdf(stat)), "n_days": int(n),
            "mean_daily_loss_diff": float(d.mean() / 24)}


def dm_matrix(y: pd.DataFrame, forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """p[i, j] = one-sided p-value that column model j beats row model i."""
    names = list(forecasts)
    p = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            if a != b:
                p.loc[a, b] = dm_test(y, forecasts[a], forecasts[b])["p_b_better"]
    return p
