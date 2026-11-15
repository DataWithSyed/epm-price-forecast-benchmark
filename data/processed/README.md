# Processed data (provenance)

| File | Source | Notes |
|---|---|---|
| `hourly_panel.csv` | Copied unchanged from [`epm-merit-order-model`](https://github.com/DataWithSyed/epm-merit-order-model) `data/processed/` (byte-identical) | Energy-Charts / SMARD, CC BY 4.0. 16,056 h, UTC |
| `installed_gw.csv` | Same as step 2 | Energy-Charts installed capacity, CC BY 4.0 |
| `monthly_drivers.csv` | Step-2 file plus Nov–Dec 2024 rows (needed for the publication lag of Jan 2025) | World Bank Pink Sheet (CC BY 4.0) gas Europe / coal South Africa / Brent, ECB USD/EUR; EUA column = DEHSt annual VWAP (2024 65.01, 2025 73.86) |
| `step2_model_prices.csv.gz` | `M0_stack` and `C3_rolling90_linear` columns of step-2 `results/hourly_model_prices.csv.gz` (identical values) | ORACLE references only |
| `forecast_panel.csv.gz` | Built by `scripts/build_forecast_panel.py` from Energy-Charts `public_power_forecast` (`forecast_type=day-ahead`) for DE and FR, NL, BE, AT, PL, CZ, DK, CH | CC BY 4.0. Raw JSON is git-ignored and re-downloadable with `--fetch` |
