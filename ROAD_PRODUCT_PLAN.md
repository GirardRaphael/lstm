# From the LSTM demonstration to a street analytics product

Status: read-only prototype; NOT approved for street deployment or signal control.
Review date: 2026-09-12. This plan distinguishes implemented changes from proposed work.

## Product decision

Start with a **Traffic Observatory** for a traffic engineering team: one corridor,
three intersections, visibility into observation quality, historical patterns and
eventually locally validated forecasts. The operator should answer: which feeds
are trustworthy, where demand is changing, and how accurate were our forecasts?
Do not sell this as autonomous traffic-light control.

The existing UCI motorway model predicts hourly counts. It has not learned
Montréal intersection queues, turning movements or five-minute demand. Changing
the interface label from hours to minutes does not change that. Vehicle volume
alone does not establish congestion: low measured flow can also occur when
traffic is blocked. Speed, occupancy, queue and capacity evidence are separate.

## Implemented in this iteration

- `street_data.py`: read-only observation preflight, CLI and synthetic demo.
- Streamlit page **7 · Street data checks**: inspect a CSV and download a JSON
  quality report; valid demo and cadence-failure states are tested.
- Reject mixed stream IDs, absent columns, bad interval units, invalid counts,
  naive/missing timestamps, duplicates, disorder, missing intervals, future
  observations, stale windows and insufficient history. UTC offsets are required
  and compared in UTC, including across daylight-saving transitions.
- Passing returns `valid_for_analysis`, never deployment approval. It does not
  certify sensor accuracy, geographic identity or model suitability.
- Fix the live training callback by passing callbacks explicitly, without global
  monkeypatching. Use a unique run name to avoid overwriting another UI run.
- Save input-feature scaler metadata for NEW runs. Older multivariate models
  still lack a complete portable inference contract; this is not repaired by
  inventing a scaler at serving time.
- Reject non-finite or incorrectly shaped inference inputs and bad model outputs.
- Forecast uses the trained bundle's series, not an unrelated newly selected CSV.
- Fix forecast chart origin and selected-horizon timestamps; fix introspection's
  last-input timestamp label.
- Long-horizon seasonal baseline now repeats the last observed daily cycle
  instead of looking into future rows when the horizon exceeds 24.
- Replace volume-only congestion/free-flow assertions with explicitly limited
  volume bands.

Try it from the repository root:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m traffic_lstm.street_data --demo
.\run_app.ps1
```

For a local, authorized feed exported to the schema below:

```powershell
.\.venv\Scripts\python.exe -m traffic_lstm.street_data --csv observations.csv --stream-id intersection-001/north/through/sensor-01 --interval-seconds 300 --max-age-seconds 600
```

No network connection to a city sensor or controller is made by either command.
Exit code 0 means structurally valid for analysis; 2 means blocked/invalid.
Demo timestamps are generated at runtime and always labelled synthetic.

| Required field | Meaning |
| --- | --- |
| `stream_id` | One site + approach + movement + sensor/aggregation definition |
| `timestamp` | End of a COMPLETED observation interval, with explicit offset |
| `interval_seconds` | Count aggregation duration, matching the declared cadence |
| `vehicle_count` | Finite, nonnegative integer observed count, not a rate |

Missing intervals are not zeros. Do not average duplicate count records without
knowing whether they are retransmissions or different observations. Do not mix
locations in one model sequence. The adapter must separately map lane geometry,
sensor version, source provenance, observed/received times and coverage status.
This preflight does not yet authenticate source identity or handle estimated,
fractional counts; a different schema would be needed for those.

## Critical findings STILL OPEN

These qualify the earlier audit: passing its smoke tests was not proof of a
leakage-free production pipeline.

1. **Weather look-ahead:** `_clip_to_physical` imputes from the full-frame median;
   feature construction also uses backward fill. Reproduction: a missing early
   temperature changes from 290 K to 310 K when only later observations change.
   Implement a versioned causal preprocessing pipeline with bounded past-only
   fill and explicit missingness. Fit any learned imputer/scaler within each
   fitting fold only. Retrain and rescore rather than silently replacing the
   preprocessing of archived models. Validate weather units per dataset: a
   column named `temp` is not necessarily kelvin.
2. **Validation contamination:** the scaler is fit to the full pre-test slice,
   which includes Keras's later validation segment. Split raw timestamps into
   fit/validation/test BEFORE fitting transforms. For multiple horizons, purge
   training examples whose targets overlap the validation targets.
3. **Time gaps:** legacy defaults keep windows across missing hours. Make a
   versioned contiguous-window policy the default for new runs. Use actual
   timestamps for seasonal baselines; row offsets are not wall-clock yesterday
   on an irregular series. Compare all candidates on the same eligible windows.
4. **Serving contract:** persist schema, feature order, cadence, site scope,
   timestamp convention, dataset checksum, dependency versions, model hash and
   training code revision together. Never silently substitute a same-named
   dataset from another directory in a production run. Separate archived runs
   from newly validated candidates.
5. **Application boundaries:** this is a local workbench, not a multi-tenant
   service. Training runs inside the UI, exports share output directories,
   authentication and resource limits are absent, and not every upload/error
   path has integration coverage. Do not publicly expose it as a city service.
6. **Evaluation edge cases:** percentage-error metrics and improvement over a
   zero-error baseline need an explicit undefined-value policy. Empty or nearly
   constant series must not crash report generation or imply certainty.

## Local data and research checked

Montréal publishes a count dataset covering vehicles, cyclists and pedestrians.
Its indexed description mentions 15-minute surveys during selected periods.
That is useful for historical demand studies; it does not establish continuous
live-feed availability. The catalog page returned HTTP 403 during direct access,
so no resource CSV, exact schema, licensing terms or live coverage was ingested
or independently validated in this iteration. A city-approved source adapter and
coverage assessment remain prerequisites.

- [Montréal count catalog](https://donnees.montreal.ca/dataset/comptage-vehicules-pietons)
- [SUMO signal simulation documentation](https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html): supports existing-plan modelling and research comparisons inside simulation.
- [SUMO project overview](https://eclipse.dev/sumo/about/): models vehicles, public transport and pedestrians.
- [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): chronological splits and a gap parameter; comparable fold durations require equally spaced observations.

## Implementation roadmap with acceptance gates

Durations below are engineering planning estimates, not delivery promises;
data access and partner approval are external dependencies.

| Stage | Deliverable | Exit gate |
| --- | --- | --- |
| 1: data/ML repair, roughly 1–2 weeks | Versioned causal transforms; raw timestamp splits; frozen model bundle; regression tests | Future-value mutation cannot change earlier features; no overlapping target leakage; save/load reproduces predictions |
| 2: historical corridor replay, roughly 2–3 weeks after data access | Source adapter, per-movement counts, timestamp/coverage reports, operator playback | Data provenance and interval semantics reviewed; no invented observations; site held out during model evaluation |
| 3: forecasting experiment, roughly 2–3 weeks | Persistence/seasonal, XGBoost, small GRU/LSTM; rolling-origin folds; interval calibration | Models scored on identical windows; per-site/per-horizon MAE and bias reported, including peak periods and outages |
| 4: simulation study, roughly 2–4 weeks | Three-intersection SUMO scenario; existing/fixed or actuated baseline and research alternatives | Common demand seeds; delay, queues and spillback plus pedestrian/cyclist/transit service assessed; no road-controller connectivity |
| 5: authorized read-only field evaluation | Partner feed, shadow predictions, drift/uptime dashboard, incident runbook | Operator reviews results against actuals over an agreed period; stale/missing data yields unavailable status; no signal actuation |

Any later road-control deployment is a separate professionally supervised program
requiring the road authority, qualified traffic/safety engineers and applicable
approvals. Do not access signal cabinets, install roadside hardware yourself or
connect this prototype to public-road controls. No such work is implemented here.

## Architecture and optimization choices

Keep the current Python workbench while validating demand. Proposed service split:
an authorized source adapter normalizes immutable observations; a quality service
validates them; PostgreSQL/PostGIS stores site geometry and time-indexed data;
a background worker owns training/backtests; a read-only inference service loads
one validated model bundle per worker; an operator UI shows timestamps, uncertainty,
quality flags and comparison to actuals. Keep training out of request handlers.

Model-selection strategy: start with seasonal/persistence and XGBoost baselines.
Keep LSTM/GRU as challengers, not a predetermined winner. The prior stored scores
favor XGBoost on the motorway data, but the preprocessing findings mean none is
yet a validated Montréal street model. Re-run candidates with equal folds, data,
hardware, seeds and tuning budgets. Report distributions, not just one lucky seed.
Do not compare archived training times on one machine with reruns on another as
if they were a controlled latency benchmark.

Optimize only measured bottlenecks: cache validated model loading; batch forecasts
across streams with compatible schemas; incrementally maintain lag windows; profile
preprocessing, model execution and serialization separately; avoid repeated Keras
graph creation on each UI rerun. Add queued jobs with limits before scaling users.
Avoid microservices and Kubernetes until operational requirements justify them.

Suggested pilot targets to negotiate, not achieved results: p95 inference under
250 ms on declared hardware, >99% expected-interval coverage, and a pre-agreed
improvement over the strongest baseline across multiple temporal folds. Measure
forecast interval coverage AND width; sequential drift can invalidate simple
conformal guarantees. Split calibration and final test data, and track coverage
by site/horizon rather than assuming nominal 90% coverage is achieved.

## Product evidence before selling

- An engineer can distinguish actuals, forecast, simulation and unavailable data.
- Measure operator time saved and forecast utility in a shadow pilot; do not
  claim reduced road delays without a controlled study.
- Use aggregate counts by default; avoid faces, plates and individual tracking.
  Any retained video needs partner-approved purpose, access, retention and review.
- Add role-based access, tenant isolation, audit logs, request/upload limits,
  explicit resource budgets, monitored failures and tested backup/restore.
- Maintain per-site approved model versions, manual rollback and alert routing.
  Model quality problems should disable forecasts, never invent safe traffic states.

The next most valuable task is Stage 1: repair and version the temporal pipeline
before new accuracy claims, then secure an authorized corridor dataset for Stage 2.
