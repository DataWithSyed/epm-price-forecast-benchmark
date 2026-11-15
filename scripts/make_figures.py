"""Figures from results/ (run scripts/run_benchmark.py first)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from epmfc.paths import FIGURES, RESULTS  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
LABEL = {"naive_d1": "Naive D-1", "naive_w1": "Naive D-7", "naive_lago": "Naive (D-1/D-7)",
         "stack_M0_exante": "Stack M0 (ex-ante)", "stack_C3_exante": "Stack C3 rolling (ex-ante)",
         "ols_drivers_exante": "Driver OLS static 2025 (ex-ante)", "ols_drivers_rolling": "Driver OLS rolling (ex-ante)", "lear": "LEAR", "lear_nb": "LEAR + neighbours",
         "gbm": "GBM", "gbm_nb": "GBM + neighbours", "ens": "Ensemble", "ens_nb": "Ensemble + neighbours",
         "stack_M0_oracle_step2": "Stack M0 (ORACLE, step 2)", "stack_C3_oracle_step2": "Stack C3 (ORACLE, step 2)",
         "ols_drivers_oracle": "Driver OLS (ORACLE, step 3)", "lear_oracle": "LEAR (ORACLE load/RES)"}


def main() -> int:
    FIGURES.mkdir(exist_ok=True)
    met = pd.read_csv(RESULTS / "metrics.csv")
    # 01 MAE by model, H1 vs H2
    piv = met.pivot(index="model", columns="period", values="mae")
    order = met[met.period == "test_2026_01_10"].sort_values("mae")["model"].tolist()
    piv = piv.loc[order]
    track = met.drop_duplicates("model").set_index("model")["track"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    yy = np.arange(len(piv))
    ax.barh(yy - 0.2, piv["H1_2026_01_05"], 0.4, color="#1f5f8b", label="Jan-May 2026 (H1)")
    ax.barh(yy + 0.2, piv["H2_2026_06_10"], 0.4, color="#e07b39", label="Jun-Oct 2026 (H2)")
    for i, m in enumerate(piv.index):
        if track[m] == "ORACLE":
            ax.barh(i, piv.loc[m, "test_2026_01_10"], 0.9, fill=False, hatch="//", edgecolor="grey", lw=0)
    ax.set_yticks(yy, [LABEL[m] for m in piv.index])
    ax.invert_yaxis()
    ax.set_xlabel("MAE (EUR/MWh), DE-LU day-ahead, rolling-origin test 2026-01-01..10-31")
    ax.set_title("DE-LU day-ahead forecast benchmark, ex-ante models vs ORACLE references\n(hatched = uses information not known at D-1)")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_mae_by_model_period.png")
    plt.close(fig)

    # 02 monthly MAE and bias
    mae = pd.read_csv(RESULTS / "mae_by_month.csv", index_col=0)
    bias = pd.read_csv(RESULTS / "bias_by_month.csv", index_col=0)
    keys = ["naive_d1", "stack_C3_exante", "ols_drivers_rolling", "lear", "lear_nb", "gbm_nb", "ens_nb", "stack_M0_oracle_step2"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k in keys:
        ls = "--" if "oracle" in k else "-"
        ax[0].plot(mae.index, mae[k], ls, marker="o", ms=3, label=LABEL[k])
        ax[1].plot(bias.index, bias[k], ls, marker="o", ms=3, label=LABEL[k])
    ax[0].set_ylabel("MAE (EUR/MWh)")
    ax[1].set_ylabel("bias = forecast - actual (EUR/MWh)")
    ax[1].axhline(0, color="k", lw=.8)
    for a in ax:
        a.axvspan(4.5, 9.5, color="#e07b39", alpha=.08)
        a.tick_params(axis="x", rotation=45)
    ax[0].set_title("Monthly MAE (shaded: H2 regime, Jun-Oct)")
    ax[1].set_title("Monthly bias")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_monthly_mae_bias.png")
    plt.close(fig)

    # 03 DM heatmap (full test)
    dm = pd.read_csv(RESULTS / "dm_pvalues_test_2026_01_10.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(dm.to_numpy(float), cmap="RdYlGn_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(len(dm)), [LABEL[c] for c in dm.columns], rotation=60, ha="right")
    ax.set_yticks(range(len(dm)), [LABEL[c] for c in dm.index])
    for i in range(len(dm)):
        for jj in range(len(dm)):
            v = dm.iloc[i, jj]
            if not np.isnan(v):
                ax.text(jj, i, f"{v:.2f}" if v >= 0.01 else "<.01", ha="center", va="center", fontsize=6)
    ax.set_title("Multivariate DM test p-values\n(column model more accurate than row model; green = significant)")
    fig.colorbar(im, ax=ax, shrink=.7)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_dm_pvalues.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
