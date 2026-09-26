# Rolling-origin and calibration evidence

Retrospective evaluation on previously inspected historical datasets. No untouched future/site confirmation is claimed.

Three expanding-origin folds × three seeds per dataset. Fixed 20-epoch maximum for the 32/16-unit LSTM, validation early stopping, direct-lag XGBoost and seasonal baselines. Fit, validation, calibration and evaluation timestamps are separated. Calibration uses the next block after validation; model selection uses only validation. All candidates share eligible windows. Counts and calendar only; no weather ablation.

The table reports mean MAE across seeds ± sample standard deviation within a fold. These are not confidence intervals. Daily/3-day/weekly paired bootstrap sensitivity and per-seed metrics are in each results.json. Identical baseline results repeated across seeds are not independent evidence. Fold boundaries use sorted observed-row fractions, so wall-clock durations differ. No equal-duration comparison is implied.

## bikes

[Protocol and all runs](backtests/bikes_20260925/results.json)

| Fold | Candidate | MAE mean ± seed SD | MAPE mean | Bias mean | Peak MAE mean | 90% interval coverage range | Mean interval width |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | xgboost | 37.15 ± 0.53 | 25.11% | -15.46 | 83.04 | 76.9%–79.0% | 88.14 |
| 1 | lstm | 68.96 ± 8.14 | 58.80% | -32.53 | 188.90 | 80.3%–83.9% | 202.03 |
| 1 | hour_of_week_mean | 82.63 ± 0.00 | 42.97% | -67.41 | 175.73 | 69.5%–69.5% | 180.88 |
| 1 | yesterday | 71.03 ± 0.00 | 67.48% | -2.81 | 108.60 | 83.0%–83.0% | 231.17 |
| 2 | xgboost | 58.77 ± 0.99 | 26.43% | -33.91 | 144.28 | 88.0%–88.4% | 264.59 |
| 2 | lstm | 123.78 ± 5.11 | 49.31% | -120.46 | 275.72 | 85.4%–86.2% | 370.71 |
| 2 | hour_of_week_mean | 138.91 ± 0.00 | 48.57% | -136.02 | 269.92 | 85.4%–85.4% | 414.88 |
| 2 | yesterday | 83.04 ± 0.00 | 60.40% | -0.45 | 121.64 | 91.3%–91.3% | 437.36 |
| 3 | xgboost | 36.60 ± 0.33 | 24.04% | -7.87 | 75.55 | 89.1%–90.2% | 160.37 |
| 3 | lstm | 56.52 ± 0.90 | 48.69% | 11.27 | 117.08 | 86.0%–89.8% | 227.10 |
| 3 | hour_of_week_mean | 121.60 ± 0.00 | 45.90% | -110.27 | 250.61 | 90.5%–90.5% | 462.72 |
| 3 | yesterday | 88.30 ± 0.00 | 65.89% | 2.98 | 135.32 | 88.6%–88.6% | 390.91 |

Validation selections: fold 1 / seed 7: xgboost; fold 1 / seed 42: xgboost; fold 1 / seed 123: xgboost; fold 2 / seed 7: xgboost; fold 2 / seed 42: xgboost; fold 2 / seed 123: xgboost; fold 3 / seed 7: xgboost; fold 3 / seed 42: xgboost; fold 3 / seed 123: xgboost

Selected-candidate observed coverage spans **76.9%–90.2%** against nominal 90%.
Calibration does not establish a coverage guarantee under shift. Review undercoverage and width before any pilot.

## motorway

[Protocol and all runs](backtests/motorway_20260925/results.json)

| Fold | Candidate | MAE mean ± seed SD | MAPE mean | Bias mean | Peak MAE mean | 90% interval coverage range | Mean interval width |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | xgboost | 177.54 ± 0.55 | 25.42% | 24.29 | 256.26 | 91.6%–91.9% | 849.07 |
| 1 | lstm | 334.13 ± 4.46 | 34.42% | 7.89 | 547.19 | 89.0%–89.9% | 1425.03 |
| 1 | hour_of_week_mean | 319.58 ± 0.00 | 36.13% | 158.93 | 436.45 | 91.0%–91.0% | 1383.64 |
| 1 | yesterday | 528.22 ± 0.00 | 43.04% | -12.63 | 749.82 | 91.0%–91.0% | 2598.84 |
| 2 | xgboost | 150.30 ± 0.39 | 6.15% | -7.45 | 198.06 | 89.7%–90.4% | 684.59 |
| 2 | lstm | 399.95 ± 61.46 | 13.65% | -324.19 | 728.85 | 92.2%–93.1% | 1833.20 |
| 2 | hour_of_week_mean | 252.27 ± 0.00 | 9.76% | -134.16 | 370.77 | 91.6%–91.6% | 1172.40 |
| 2 | yesterday | 532.88 ± 0.00 | 23.40% | 1.67 | 750.45 | 91.3%–91.3% | 2643.27 |
| 3 | xgboost | 152.93 ± 0.54 | 6.42% | 1.81 | 219.14 | 92.3%–93.1% | 801.03 |
| 3 | lstm | 315.01 ± 28.67 | 14.48% | -47.45 | 500.80 | 90.8%–93.0% | 1524.15 |
| 3 | hour_of_week_mean | 292.04 ± 0.00 | 12.60% | -35.07 | 461.59 | 93.8%–93.8% | 1536.29 |
| 3 | yesterday | 596.35 ± 0.00 | 26.61% | -21.38 | 868.31 | 89.7%–89.7% | 2894.04 |

Validation selections: fold 1 / seed 7: xgboost; fold 1 / seed 42: xgboost; fold 1 / seed 123: xgboost; fold 2 / seed 7: xgboost; fold 2 / seed 42: xgboost; fold 2 / seed 123: xgboost; fold 3 / seed 7: xgboost; fold 3 / seed 42: xgboost; fold 3 / seed 123: xgboost

Selected-candidate observed coverage spans **89.7%–93.1%** against nominal 90%.
Calibration does not establish a coverage guarantee under shift. Review undercoverage and width before any pilot.

## Remaining external gates

Authorized local observations, certified interval/timezone semantics, untouched future/site tests, operator acceptance and field interval coverage remain unverified. These artifacts support research review, not street readiness.
