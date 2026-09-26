# Traffic Observatory: delivery review and acceptance gates

Review date: 2026-09-25. Scope: local hourly-count observatory and research workbench. No street deployment or signal control.

## Objective assessment

This repository was an educational LSTM demonstration presented more confidently than its evidence justified. Its useful assets are inspectable models, real historical data, and the recovered causal pipeline. The primary weaknesses were leakage in published preprocessing, model selection informed by test results, weak baselines, inconsistent documentation and unreproducible local setup. A sophisticated visualization does not resolve any of those defects.

The original product plan said street preflight and several reliability fixes were implemented, but those files are absent from this checkout. Those claims are withdrawn here. Branch names and old session notes are not verification evidence.

## Implemented in this checkout

- Recovered `pipeline_v2.py` and temporal tests from `origin/observatory/pipeline-v2`, commit `0c0e62d`; no wholesale merge of unrelated branch changes.
- Added fit-only weekly climatology, causal timestamp baselines, direct lag/calendar XGBoost, bias and peak diagnostics, paired daily-block bootstrap intervals, and validation-based candidate selection including baselines.
- Aligned LSTM optimization and stopping with MAE. Preserved v1 semantics for archived artifacts.
- Added recorded experiment contracts, immutable package names, source fingerprints and manifest-driven reports. New evidence remains retrospective because historical test data were already inspected.
- Added project-local launchers, a hashed dependency lock and Windows CI configuration. CI has to run on the remote before it can be called passing there.
- Added a read-only v2 evidence page. Remaining app pages are legacy education, not the product workflow. Removed the process-wide callback monkeypatch and isolated educational run names.
- Marked legacy publications exploratory and the aborted two-epoch sweep unusable for selection.

See `reports/REPORT.md` for actual run evidence. Test execution evidence and remaining limitations are recorded in `reports/DELIVERY_REVIEW.md`.

## Current engineering status

The local observation-to-error workflow is now implemented: strict imports, immutable SQLite snapshots, authenticated roles/tokens, bounded background jobs, activation/disable/rollback, calibrated intervals, unavailable states, forecast/actual joins, monitoring gates, audit logs and tested backup/restore. The new app entry point is `app/observatory_app.py`.

Executed three rolling-origin folds ? three seeds on both datasets. Prediction records and separate calibration/evaluation diagnostics are in `reports/BACKTEST_REPORT.md`. Motorway interval coverage is near nominal in those folds; bike undercoverage remains material. Do not claim general calibration has been solved.

All local verification and the bundled-data acceptance workflow passed. Read `reports/IMPLEMENTATION_LOG.md` for the steps, failures found and corrected, commands, evidence and limits. Read `docs/OPERATIONS.md` to run the product. Remote CI and external-browser visual review have not been completed.

## What a credible first product should do

For one authorized corridor, an engineer can inspect the latest completed observations, identify missing or stale intervals, review historical demand, compare forecasts with actuals and understand when forecasts are unavailable. The product must distinguish observed counts, predictions and simulations. Volume alone cannot establish congestion or queues.

A motorway count model is not validated for intersection turning movements or another city's five-minute counts. Renaming its output or units is not adaptation. Do not offer traffic-light optimization or delay-reduction claims based on this repository.

## Prioritized delivery gates

| Priority | Work | Acceptance evidence |
| --- | --- | --- |
| P0: scientific credibility | Freeze validation-only tuning budgets; rolling-origin evaluation across seasons; reserve a genuinely untouched future/site holdout | Every candidate uses the same eligible target timestamps; transformations fit within folds; all folds and exclusions reported; no test-driven retuning |
| P0: useful uncertainty | Multiple seeds and block-length sensitivity; independently calibrated forecast intervals | MAE/RMSE/MAPE/bias/peak slices, paired differences, interval coverage and width by regime; low-count and missingness analysis |
| P0: input contract | Authorized local feed, source identity, aggregation semantics, UTC timestamps, unit schema, deduplication policy | DST/outage/duplicate/staleness tests; missing observations never become zeros; no mixtures of sensors or sites |
| P1: complete workflow | Durable imports, isolated background jobs, versioned datasets and model registry, immutable forecast records | Import to restart to forecast to actual/error review works; cancellation, failure and retry paths tested |
| P1: shared-service reliability | Authentication, authorization, upload/job limits, secrets handling, audit trail, backup and recovery | Unauthorized access rejected; bounded workload; restore and rollback drills pass; training kept out of request handlers |
| P1: operational quality | Drift/coverage/error monitoring, staleness thresholds, unavailable states and alert ownership | Simulated outage and model failure produce explicit unavailable state; operator can identify source/version/time and roll back |
| P2: pilot value | Read-only shadow pilot with traffic engineers | Agreed forecast utility and operator time-saving criteria measured over an agreed period; no claim of road-delay reduction |

Local implementation and automated evidence now satisfy the basic persistence/job/auth/recovery workflow portions of these gates. Field-data, operator-utility, untouched future/site evidence and shared-hosting requirements remain open. Completing infrastructure does not validate the model, and a lower MAE does not establish customer value.

## Model policy

XGBoost is the historical champion and the first production candidate to investigate. LSTM remains an educational and research challenger. The new validation result may select a baseline instead; that is a valid outcome. Direct-tree lag features and sequence features have different information sets, so do not present that comparison as an isolated architecture ablation. Bike weather and other ablations require matched budgets and a separate validation protocol.

Train/validation/test chronology is mandatory. Calendar-only reruns are not evidence that weather hurts. Reusing an old test period gives a reanalysis, not confirmation. Bootstrap intervals conditional on one fit do not replace seed stability checks. One split is insufficient under temporal shift.

## Architecture policy

Keep the local Python workbench while proving data access and operator value. A pilot can use one application, a durable database and one bounded job worker. Introduce a model registry, immutable observations and forecasts, quality checks, structured logs and explicit rollback. Microservices and Kubernetes are unjustified until measured operational constraints require them.

Avoid further animation, larger recurrent networks, new feature ablations under v1, and deployment claims. The next release should earn trust through an end-to-end observation-to-error workflow and an honest evidence report.
