# Delivery review — 2026-09-25

**Updated decision: tested local observatory delivered; field/customer/shared-hosting release remains blocked.**

The later implementation is documented in [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md), [BACKTEST_REPORT.md](BACKTEST_REPORT.md) and [the runbook](../docs/OPERATIONS.md). It adds durable jobs, auth, calibration, monitoring and backup/restore. The review below records the earlier foundation-only stage; its statements that those local capabilities are absent are superseded by that implementation record.

The prior project combined useful educational introspection with overstated readiness. Fixing preprocessing alone cannot establish product-market fit, traffic-engineering validity or operational reliability. The changes in this checkout address the first scientific and reproducibility gate, not the entire product roadmap.

## Evidence obtained

- Recovered the original causal implementation and tests from `origin/observatory/pipeline-v2` at `0c0e62d`. The old missing-code blocker is resolved; unrelated branch deletions were not imported.
- Ran actual motorway and bike training through v2 with fixed recorded protocols, one seed, 24-hour windows and 50 maximum epochs. Validation selected XGBoost on both. Models and manifests are retained in `models/v2/`.
- Motorway test MAE: XGBoost 153.84, LSTM 277.98, weekly mean 288.76; 7,686 matched windows. Paired XGBoost-minus-LSTM MAE interval: [-136.33, -113.31]. The tree advantage is substantial in this experiment.
- Bike test MAE: XGBoost 36.83, LSTM 38.35; 3,359 matched windows. Paired difference interval: [-3.83, 0.76]. This does **not** establish a reliable MAE ordering. XGBoost MAPE is lower, but LSTM RMSE and peak-hour MAE are lower: the choice depends on operational error costs.
- The direct tree and LSTM use different information sets; this evaluates practical candidates, not recurrence alone. Both reruns use counts and calendar; neither is a weather ablation.
- LSTM reached the 50-epoch budget on both runs. These are fixed-budget results, not evidence of the best possible tuned recurrent model. No test-informed retraining was performed.

The generated [REPORT.md](REPORT.md) and linked manifests are authoritative for precise numbers. Test periods are historically inspected, so these are retrospective reanalyses. Daily-block bootstrap intervals quantify sampling uncertainty conditional on these fitted models; they do not prove robustness under future shift or across seeds.

## Verification

- `./test.ps1`: 16 legacy checks passed; pytest 9/9 passed. The pytest temporal aggregate includes 18 recovered temporal checks, plus seven new evaluation/package tests and one Streamlit application test.
- Tests cover future-value mutation, chronological splits and horizon purging, gap exclusions, schema/units, undefined metrics, Keras/XGBoost save/load parity, corruption refusal, timestamp-true baselines, direct lag availability, matched evaluation counts and the default evidence-viewer flow.
- `uv pip sync requirements.lock --dry-run`: 145 installed packages match the lock; no changes required. Python 3.12.10 on Windows.
- `git diff --check`: no whitespace errors. Line-ending normalization notices are unrelated to correctness.
- Windows CI is configured but has not run remotely. No push, merge or deployment was performed. Streamlit was exercised through AppTest, not an external-browser visual audit.
- Keras emits NumPy `__array__(copy=...)` deprecation warnings; tests pass. These are dependency compatibility debt to track, not hidden test failures.

## Remaining release blockers, in order

1. **No validated customer data or workflow.** Define the operator decision, obtain an authorized feed, certify units/cadence/timezone/source identity and assess coverage. Historical motorway volume does not validate street queues or signal decisions.
2. **Insufficient generalization evidence.** Add rolling-origin folds, untouched future/site evaluation, seeds and block-length sensitivity. Account for shift, missingness and rare peaks. Freeze tuning before holdout inspection.
3. **No forecast uncertainty contract.** Calibrate prediction intervals separately and monitor coverage/width. MAE confidence intervals are not uncertainty bounds around individual forecasts.
4. **No durable product service.** Imports, job execution, forecasts and actual/error joins need persistence, isolation, cancellation, retry, recovery and an operator acceptance test.
5. **No shared-hosting controls.** Authentication, authorization, quotas, secrets, audit logging, backup/restore and incident ownership remain open. The existing UI still contains legacy synchronous educational training and shared report/vault exports.
6. **No model operations.** Add approval state, provenance validation, immutable serving selection, staleness/drift alerts and rollback. File checksums detect corruption; they do not authenticate a maliciously replaced manifest. Do not load untrusted model packages.

Build a read-only observatory first. Do not spend the next iteration on animation or a larger network. Prove an observation-to-forecast-to-error workflow on authorized local data, then decide whether forecasting creates enough operational value to justify a pilot.
