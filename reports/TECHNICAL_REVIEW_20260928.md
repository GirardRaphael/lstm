# Technical review and improvement plan

Review date: 2026-09-28  
Reviewed branch: `main`  
Scope: supervised learning, evaluation, packaging and local observatory behavior

## Verdict

This is now a legitimate retrospective forecasting workbench. It is not a
traffic-control product. The strongest part is the causal v2 pipeline and the
operational refusal to serve gapped or schema-incompatible observations. The
champion is XGBoost, not the LSTM.

The single-split v2 evidence is clear on I-94: XGBoost MAE 153.84
vehicles/hour versus LSTM 277.98. The paired daily-block interval for
XGBoost-minus-LSTM absolute error is entirely below zero
(`[-136.33, -113.31]`), so this is not a cosmetic win. On bikes, XGBoost MAE
36.83 versus LSTM 38.35, but the paired interval crosses zero
(`[-3.83, 0.76]`). There is no defensible bike LSTM win.

The rolling-origin evidence is stronger than the single split: validation
selected XGBoost in all 18 fold/seed runs. The weak point is uncertainty under
shift. Nominal 90% bike interval coverage falls as low as 76.9%. Those
intervals are not ready for an operational claim.

The app is useful for local historical replay, model registration, activation,
forecast storage and later error joins. It is not ready for a shared server or
street deployment. It has no certified live feed, no certified timestamp
semantics, no untouched future/site holdout, no managed identity/TLS, and no
field utility study.

## Findings ranked by cost

### P0 — do not claim completion without these

1. **Untouched evidence is absent.** All current datasets and historical test
   periods have already influenced project decisions. A new future period or
   site must be frozen before collection and evaluated once.
2. **Bike intervals under-cover.** Coverage of 76.9% against nominal 90% is a
   failed calibration result, not acceptable noise.
3. **Source semantics are uncertified.** The project does not know whether
   every historical timestamp is interval-start or interval-end, or its
   authoritative timezone. This blocks field use.
4. **The model predicts hourly aggregate counts, not queues, delay or signal
   demand.** It cannot justify intersection phase timing.

### P1 — needed for a defensible forecasting system

1. Monitor WAPE, sMAPE, median absolute error and P90 absolute error in
   addition to MAE/RMSE/MAPE. Mean error alone hides tail failures.
2. Preserve serving parity for direct trees. A model trained with weekly lags
   must not silently receive only 24 hours of history.
3. Add short-term momentum and rolling summaries to the direct tree without
   introducing imputation or future information.
4. Evaluate calibration by season and demand regime. Do not widen one global
   interval until aggregate coverage looks acceptable; that hides conditional
   failure.
5. Keep model selection on validation only. Test is scored once after the
   candidate and feature contract are frozen.

### P2 — engineering before shared use

1. Managed identity, TLS, encrypted secrets and encrypted backups.
2. A scheduled data-quality monitor for missing intervals, clock shifts,
   duplicate timestamps and range drift.
3. Alert routing for unavailable forecasts, interval under-coverage and error
   threshold breaches.
4. A documented rollback exercise using an actual prior model package.

## Algorithm critique

### What is correct

- Chronological split occurs before fit-dependent transforms.
- Scalers see fit rows only.
- Incomplete and non-cadence windows are excluded rather than filled.
- Timestamp-based persistence and seasonal baselines avoid row-offset errors.
- XGBoost and LSTM are selected using validation MAE.
- Fit-only hour-of-week climatology is included.
- Rolling-origin folds, multiple LSTM seeds and paired block resampling are
  recorded.

### What remains weak

- The LSTM architecture has no empirical advantage. On motorway it is slower
  and materially less accurate; on bikes its apparent single-split difference
  is inconclusive and its MAPE is worse.
- A 24-hour recurrent window is branding, not a demonstrated optimum. The
  historical 12-hour LSTM artifact was better than the historical 24-hour
  artifact, and neither replaces a controlled validation-only sweep.
- One global absolute-residual interval is too crude under bike demand shift.
- The direct tree previously had only `t-1`, `t-24`, `t-168` and clock
  features. That misses immediate acceleration/deceleration and local level.
- MAPE is unstable at low demand. It must never be the only percentage metric.
- Weather is not a priority on I-94. Calendar and count history carry the
  result; sparse snow and faulty rain values do not justify more model
  complexity.

## Implemented in this review

### Direct-tree feature contract `seasonal_v2`

Newly trained direct trees receive only causal information:

- count lags: `t-1`, `t-2`, `t-3`, `t-24`, `t-25`, `t-168`;
- complete-history rolling mean over 3 and 24 hours;
- complete-history 24-hour standard deviation;
- target-hour sine/cosine;
- target-weekday sine/cosine;
- target-weekend flag.

Rolling summaries become missing if any timestamp in their source window is
absent. Nothing is interpolated. Existing packages remain loadable:
`seasonal_v1` preserves their original seven features and is inferred when an
old manifest has no feature-set field.

### Serving diagnostics

A direct-tree forecast now reports every unavailable input. Supplying only a
24-hour window to a model that expects `t-168` no longer degrades silently.
The estimator may still score a missing feature because XGBoost learned
missing-value branches, but the response states that serving parity is not
met.

### Error diagnostics

Every v2 candidate now reports:

- MAE;
- median absolute error;
- P90 absolute error;
- RMSE;
- MAPE and its eligible sample count;
- sMAPE;
- WAPE;
- signed bias;
- peak-hour MAE;
- daily-block MAE interval.

The report generator retains compatibility with older manifests; new columns
show `undefined` until a model is retrained.

### Baselines and serving selection

Commit `105734e` made direct-lag XGBoost the default comparator, added a
timestamp-true last-week baseline, and made packages serve the
validation-selected candidate by default. This review keeps that behavior and
version-controls the expanded direct feature contract.

## Acceptance checks

The implementation is acceptable only when all of the following are true:

1. Mutation of validation/test values cannot change fit tensors or scalers.
2. Every candidate in a partition has the same `n`.
3. A missing timestamp makes the affected rolling features missing; it is not
   averaged around.
4. Old direct-tree packages use `seasonal_v1`.
5. New packages record `seasonal_v2` and the exact ordered feature names.
6. Save/reload predictions match.
7. Default package forecast names the validation-selected candidate.
8. Missing weekly history appears in `input_warnings`.
9. Metric functions reject broadcasting and non-finite values.
10. CI and local test commands pass on the committed revision.

## Experiment plan

### Next run

1. Freeze a new immutable run name and protocol before training.
2. Retrain motorway and bikes using `seasonal_v2`, with the existing split and
   seed only for comparability—not as new confirmatory evidence.
3. Compare validation MAE of `seasonal_v1` and `seasonal_v2`; choose there.
4. Score test once after selection.
5. Generate `reports/REPORT.md` from manifests.

### Robustness run

1. Repeat three expanding-origin folds and seeds 7/42/123.
2. Retain every prediction.
3. Report fold/seed MAE, WAPE, P90 error, peak MAE and bias.
4. Recalibrate intervals per fold using a disjoint calibration block.
5. Stratify interval coverage by hour-of-week and demand quartile.

### Confirmation

1. Register an untouched future period or different sensor/site.
2. Freeze the champion, preprocessing, thresholds and report template first.
3. Run once.
4. Accept only if it beats hour-of-week climatology and timestamp-true last
   week, meets the declared MAE/peak-error threshold, and achieves acceptable
   interval coverage by regime.

## Stop doing

- Do not make the LSTM larger until it wins a controlled validation experiment.
- Do not tune against the test period.
- Do not describe the intersection demo as model control.
- Do not compare bike MAE numerically with motorway MAE; the units differ.
- Do not claim nominal interval coverage when a recorded fold under-covers.
- Do not spend engineering time on neuron animation before collecting
  untouched evidence.
