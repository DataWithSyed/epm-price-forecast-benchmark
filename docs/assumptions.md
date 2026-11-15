# Assumptions and information-set register

Everything that is not directly observed is labelled **ASSUMED**. Anything that uses information not available at day-ahead gate closure (D-1 12:00 Europe/Berlin) is labelled **ORACLE** and is kept out of the ex-ante ranking and DM tests.

| ID | Item | Treatment | Status |
|---|---|---|---|
| A1 | Gate closure | Forecast for delivery day D (00–23 local) is issued at D-1 12:00 Europe/Berlin | standard SDAC timing |
| A2 | Lagged prices | DE-LU DA prices up to and including D-1 are known (D-1 auction cleared on D-2) | observed |
| A3 | Load / solar / wind forecasts | Energy-Charts `public_power_forecast`, `forecast_type=day-ahead`, DE and FR, NL, BE, AT, PL, CZ, DK, CH. Taken as the TSO forecast published on D-1 before gate closure. Energy-Charts does not report the publication timestamp, so the vintage is **ASSUMED** | ASSUMED vintage |
| A4 | Neighbour gaps | Missing neighbour forecast hours are filled with the value 7 (then 14) days earlier. A series that starts part-way (PL offshore wind from mid-2026) is 0 before its first value. AT/CH offshore and CZ wind are not published and are treated as 0 | ASSUMED, causal |
| A5 | Realised fundamentals | Realised generation and cross-border trade are used only up to D-2 (complete days). Realised D-1 morning data would also be legal, but are not used | conservative |
| A6 | Must-run RES/hydro/waste for the stack | Mean of the same hour over D-8..D-2 (realised) | ASSUMED proxy |
| A7 | Monthly fuels | World Bank Pink Sheet month *m* (gas Europe, coal South Africa, Brent; ECB USD/EUR) is **ASSUMED** usable from the 4th calendar day of month *m+1*. So a forecast on day D uses month m-1 or m-2 | ASSUMED publication lag |
| A8 | EUA | 2025: DEHSt annual German auction VWAP 73.86 €/t (flat, a proxy: the annual figure is only known after the year). 2026: DEHSt monthly average auction prices (Jan–Sep), lagged like A7. The monthly average can be rebuilt from public daily EEX auction results. October 2026 = Q3 VWAP carried | ASSUMED / proxy |
| A9 | Thermal stack | Same five technologies, efficiency bands, availabilities, VOM, fuel adders and IPCC 2006 CO₂ factors as step 2 (`epm-merit-order-model/docs/assumptions.md`). Thermal must-run = 2 % quantile of 2025 output | ASSUMED (step 2) |
| A10 | Installed capacity | Energy-Charts yearly installed stock (2026 value used for 2026). This is treated as known ex-ante | ASSUMED known |
| A11 | DST | 23-h day: the missing 02:00 is linearly interpolated. 25-h day: the duplicated hour is averaged. Applied identically to actuals and forecasts, so the scoring grid is 304 days × 24 h | convention |
| O1 | ORACLE: step-2 M0 / C3 | Shipped step-2 hourly prices use **realised** residual load and same-month fuels | ORACLE |
| O2 | ORACLE: driver OLS | Realised residual load, same-month fuels, DEHSt quarterly EUA (step-3 specification) | ORACLE |
| O3 | ORACLE: LEAR | Same as LEAR, but with realised day-D load and wind+solar in place of the forecasts | ORACLE |

## Model settings (fixed before looking at test results, no tuning on the test window)

- **LEAR**: 24 hour-specific LassoLarsIC (AIC) models on asinh(median/MAD)-scaled prices (Uniejewski, Weron & Ziel 2018). Inputs: prices D-1, D-2, D-3, D-7 (24 each), load and wind+solar forecasts for D and D-1 (24 each), lagged gas, coal and EUA, and day-of-week dummies. Uses a 364-day window and is recalibrated every test day. `lear_nb` adds the neighbour residual-load and RES forecasts for D and the realised DE net trade at D-2 (24 each).
- **GBM**: one sklearn `HistGradientBoostingRegressor` over all hours (max_iter 400, learning rate 0.05, 31 leaves, min leaf 40, L2 1.0). Inputs: hour, day of week, price lags and D-1 mean/max/min, forecast load / RES / residual load, D-1 change and daily mean, and fuels. `gbm_nb` adds the neighbour forecasts and D-2 trade. Uses a 364-day window and is refit every 7 test days.
- **Ensemble**: simple mean of LEAR and GBM (same feature family).
- **Stack C3 (ex-ante)**: `a + b·M0_exante`, refit every test day on the trailing 90 days. Step 2 refit before each month.
- **Driver OLS (static)**: hourly OLS of price on forecast residual load, lagged gas, coal and EUA, fitted once on 2025 (step-3 specification, ex-ante inputs).
- **Driver OLS (rolling)**: same four regressors, refit every test day on the trailing 364 days ending D-1. Added because the static 2025 fit extrapolates badly once 2026 fuel prices leave the 2025 range (see README).
