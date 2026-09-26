# Hourly forecast evidence

Status: research workbench, not a deployed traffic product. No signal control.

These are retrospective reruns on historically inspected data. Model selection uses validation MAE, including baselines, before test scoring. This does not make the old test period untouched.

One seed per run. MAE intervals resample observed calendar-day blocks (500 replicates). They do not measure training-seed variability or provide prediction intervals. Peak hours are weekdays 07–09 and 16–18 inclusive in the dataset clock. MAPE excludes actuals <= 1; its sample count is shown. Bias is prediction minus actual.

## bikes_v2_20260925

Source: [bikes_v2_20260925 manifest](../models/v2/bikes_v2_20260925/manifest.json).
Validation-selected candidate: **xgboost**. Units: {'rentals': 'rentals/hour'}.
LSTM epochs: 50/50 maximum. Fixed-budget candidates; no claim of exhaustive tuning or the best attainable LSTM.
XGBoost layout: `direct_lags`. The direct tree uses longer seasonal lags and the target clock; this is a practical candidate comparison, not an architecture-only ablation.
Eligible windows: `{'fit': 9518, 'validation': 2780, 'test': 3359}`. Excluded: `{'non_cadence_step': 1698, 'missing_input': 0, 'missing_target': 0, 'boundary_purge': 0}`.

Test h1:

| Candidate | MAE [95% CI] | RMSE | MAPE | MAPE n | Bias | Peak MAE | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | 36.83 [34.28, 39.11] | 60.20 | 24.13% | 3352 | -8.89 | 77.80 | 3359 |
| lstm | 38.35 [36.24, 40.75] | 59.65 | 28.56% | 3352 | -9.09 | 73.04 | 3359 |
| hour_of_week_mean | 118.46 [111.50, 126.29] | 166.98 | 52.76% | 3352 | -102.32 | 246.09 | 3359 |
| naive_seasonal | 79.86 [71.00, 89.86] | 133.94 | 66.54% | 3352 | 1.04 | 122.29 | 3359 |
| naive_persistence | 86.21 [80.38, 92.23] | 130.52 | 53.04% | 3352 | 0.12 | 219.94 | 3359 |

Paired XGBoost minus LSTM absolute-error difference: `{"mean": -1.5251621206674446, "ci": {"low": -3.8337697448001435, "high": 0.7628505984406215, "confidence": 0.95, "blocks": 145, "block_hours": 24, "replicates": 500, "seed": 42}}`. Negative favors XGBoost.

## motorway_v2_20260925

Source: [motorway_v2_20260925 manifest](../models/v2/motorway_v2_20260925/manifest.json).
Validation-selected candidate: **xgboost**. Units: {'traffic_volume': 'vehicles/hour'}.
LSTM epochs: 50/50 maximum. Fixed-budget candidates; no claim of exhaustive tuning or the best attainable LSTM.
XGBoost layout: `direct_lags`. The direct tree uses longer seasonal lags and the target clock; this is a practical candidate comparison, not an architecture-only ablation.
Eligible windows: `{'fit': 15027, 'validation': 6158, 'test': 7686}`. Excluded: `{'non_cadence_step': 11680, 'missing_input': 0, 'missing_target': 0, 'boundary_purge': 0}`.

Test h1:

| Candidate | MAE [95% CI] | RMSE | MAPE | MAPE n | Bias | Peak MAE | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | 153.84 [147.78, 159.74] | 241.44 | 6.36% | 7686 | 0.39 | 229.37 | 7686 |
| lstm | 277.98 [264.61, 292.32] | 424.83 | 11.47% | 7686 | -134.52 | 428.97 | 7686 |
| hour_of_week_mean | 288.76 [260.46, 317.74] | 508.42 | 12.21% | 7686 | -34.83 | 472.19 | 7686 |
| naive_seasonal | 571.58 [524.66, 622.29] | 1030.83 | 25.47% | 7686 | -11.21 | 842.46 | 7686 |
| naive_persistence | 585.33 [568.17, 602.59] | 811.83 | 26.89% | 7686 | 0.63 | 744.14 | 7686 |

Paired XGBoost minus LSTM absolute-error difference: `{"mean": -124.13906245117255, "ci": {"low": -136.325030445357, "high": -113.30816657463171, "confidence": 0.95, "blocks": 337, "block_hours": 24, "replicates": 500, "seed": 42}}`. Negative favors XGBoost.

## Delivery status

Rolling-origin, three-seed evaluation and separate interval calibration are now recorded in [BACKTEST_REPORT.md](BACKTEST_REPORT.md). A persistent local workflow with access tokens, isolated workers, forecasts/actuals, model activation, monitoring gates and backup/restore is implemented. See [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) and [operations](../docs/OPERATIONS.md).

Remaining gates:

- Untouched future/site holdouts and field interval calibration; historical reanalysis is not confirmation.
- Resolve observed bike interval undercoverage under shift before relying on those intervals.
- Authorized local observations, interval semantics, sensor provenance and operator utility validation.
- Managed identity/TLS, deployment isolation, encrypted secrets/backups and alert delivery before shared hosting.
- Legacy educational UI paths remain v1 and cannot substantiate product accuracy.

Historical v1 evidence is in [LEGACY_REPORT.md](LEGACY_REPORT.md). The aborted two-epoch window sweep is not model-selection evidence.
