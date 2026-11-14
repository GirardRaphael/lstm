# Traffic Intersection, Observatory and Forecast Workbench

A local, authenticated observatory for importing vehicle counts, running durable training jobs, storing forecasts and reviewing later actuals. The research workbench compares hourly I-94 counts and Washington DC bike rentals. Compare a seasonal baseline, XGBoost and an inspectable LSTM. The forecast workbench does not control traffic. A separate interactive intersection simulation now includes an automatic scene-based light controller; no real signals or live sensors are connected.

**Delivery status:** the local observation-to-error workflow is implemented and tested, including backup/restore. Field validation and shared-hosting readiness remain open. The legacy scores are exploratory; clean reanalysis of previously inspected data is still not an untouched confirmatory test.

## Live intersection and image gallery

The live intersection previously on `observatory/phase2-v2-wiring` is now included here, with automatic decisions based on the vehicles and pedestrians visible in the simulation.

```powershell
# After ./setup.ps1; also install Node.js 22.12+ (tested with Node 24.14)
./run_app.ps1                 # Live intersection at http://127.0.0.1:43125/
./run_app.ps1 -NoBrowser      # Start without opening a browser
./run_app.ps1 -Observatory    # Authenticated operational observatory
./run_app.ps1 -Workbench      # Historical forecasting and neuron inspection
./run_app.ps1 -Images         # Separate local aerial-image gallery
```

Automatic control is enabled by default. At each cleared junction, the controller uses visible queue sizes, stopped vehicles, waiting time and pedestrians to choose which direction receives green and for how long. Yellow, clearance and walk phases remain in the simulation; waiting-time priority helps prevent a smaller queue from being ignored. The panel explains applied decisions and records them. Operator requests are queued for a decision boundary.

This is a rule-based controller reading scene objects, not a camera-image detector or a trained neural controller. The optional historical LSTM panel replays stored traffic-count forecasts and does not choose the lights. See [the demo guide](presentation/README.md) for the decision rules, tests and source revision. Press Ctrl+C in the launcher terminal to stop its services. The first launch installs the locked JavaScript dependencies.

The dataset tools support a local **1,500-image UIT-ADrone subset: 1,200 normal and 300 anomalous aerial frames**. Images, labels and retained source metadata live in the ignored `data/intersection_anomalies/` directory and are **not bundled in GitHub**. A fresh clone's gallery reports that the dataset is missing until that directory is restored or prepared. The photographs show real roundabouts, and the labels are publisher-provided binary anomalies; verified violation subtypes and object boxes are not included. See [the dataset specification](reports/INTERSECTION_DATASET_SPEC.md) for provenance, split rules and verification.

## Evidence and model choice

Read [the generated current report](reports/REPORT.md) and its linked `models/v2/<run>/manifest.json` packages. Those manifests are the source of truth for new results. Select candidates on validation MAE, including baselines, before reporting test metrics. Do not assume the LSTM wins.

The v1 motorway results favored XGBoost plus calendar. The tiny historical bike MAE difference does not establish an LSTM advantage; its MAPE was worse. [Legacy results](reports/LEGACY_REPORT.md) retain the original numbers and limitations. The two-epoch `window_sweep.json` is explicitly invalid for selection. Archived 12-hour/24-hour records are metrics-only, not loadable models.

The current senior review, prioritized upgrade plan, acceptance checks and
algorithm critique are in
[reports/TECHNICAL_REVIEW_20260928.md](reports/TECHNICAL_REVIEW_20260928.md).

## Reproducible setup

Tested runtime: Python 3.12 on Windows. Install [uv](https://docs.astral.sh/uv/), then from this directory:

```powershell
./setup.ps1
./test.ps1
./observatory.ps1 init
./run_app.ps1 -Observatory
```

Initialization prints an administrator token once; save it securely and enter it in the app. The `-Observatory` option opens the authenticated operational workflow. Follow [the operations runbook](docs/OPERATIONS.md) for bundled replay, worker startup, activation, tokens and backup/restore. The older evidence viewer and neuron tools remain accessible with `./run_app.ps1 -Workbench`; they are educational v1 paths. Streamlit launch modes use the project `.venv`. The intersection launcher also supports the existing shared `~/.venvs/traffic_lstm` environment when no project environment is present.

The landing page and signed-in Overview include a 12-slide project briefing: purpose, workflow, dataset coverage, candidate models, evaluation protocols, source-backed results, uncertainty, safeguards and delivery limits. Use Previous/Next or the section selector; download the complete briefing as Markdown. Only committed project evidence is shown before sign-in—operational workspace records remain protected.

## Local workflow

```powershell
# In a terminal with OBSERVATORY_TOKEN set (see the runbook):
./observatory.ps1 bundled --dataset motorway --hold-back 24
./observatory.ps1 enqueue SNAPSHOT_ID
./observatory.ps1 worker --once
./observatory.ps1 list models
./observatory.ps1 activate MODEL_ID
./observatory.ps1 forecast SNAPSHOT_ID --replay
./observatory.ps1 bundled --dataset motorway --hold-back 0
./observatory.ps1 errors
```

This is historical replay. UTC is assigned to bundled clock labels only for the replay adapter; source timezone and interval-end semantics are not certified. The worker's simple tree/weekly-mean candidates use separate validation, calibration and evaluation periods. Activation and serving enforce quality gates. The operational schema is hourly vehicle counts, not bike rentals.

## Research training

```powershell
./train.ps1 --dataset motorway --run-name motorway_new
./train.ps1 --dataset bikes --run-name bikes_new
.venv/Scripts/python.exe scripts/report_v2.py
```

Every name must be new. An experiment contract is written to `reports/protocols/` before training. Defaults: 64/16/20 chronological fit/validation/test, 24-hour complete windows, one-hour horizon, seed 42, 50 maximum epochs, patience 8. Hyperparameter exploration must use validation; do not rerun choices to improve the displayed test score.

- Split timestamps before fitting scalers; never fill missing targets or bridge outages.
- Compare on identical eligible timestamps: fit-only hour-of-week mean, timestamp-true yesterday, timestamp-true last week, persistence, direct XGBoost and LSTM. A saved package forecasts the validation-selected candidate unless a model is named.
- New direct XGBoost runs use t-1/t-2/t-3, t-24/t-25 and t-168 count lags, causal 3h/24h history summaries, and the target hour/weekday clock. Serving refuses a history span shorter than the weekly contract; isolated outages stay missing and are reported. Existing `seasonal_v1` packages retain their original seven-feature contract. `--tree-layout flattened` retains the original tensor comparator.
- Direct-tree and LSTM information sets differ: this compares practical candidates, not recurrence alone. No weather ablation is claimed.
- LSTM optimizes scaled MAE, proportional to physical-unit MAE. The chronological tail of fit is reserved for early stopping; the separate validation partition selects the candidate. XGBoost uses squared-error fitting and the same fit-tail role for stopping.
- Report MAE, RMSE, MAPE and its denominator count, signed bias, peak-hour MAE, and paired daily-block bootstrap uncertainty. A one-seed bootstrap does not measure training variability or forecast interval coverage.
- Packages retain model checksums, scalers, schema, units, site scope, split/coverage records, dependency versions, revision and source hashes. Invalid/gapped recent windows are refused.

The source datasets use historical clock labels. Their timezone and interval start/end semantics have not been independently certified for live ingestion. Previous observations in validation/test may become input history for later forecasts: this is rolling one-step evaluation, not a forecast of the entire test period from a single origin.

## Rolling-origin evaluation

[BACKTEST_REPORT.md](reports/BACKTEST_REPORT.md) records three folds and three seeds on both datasets, separate interval calibration, retained predictions, and bootstrap block-length sensitivity. Nominal 90% tree intervals covered 89.7?93.1% on motorway and 76.9?90.2% on bikes. The bike undercoverage remains a limitation, not an implementation claim to hide.

```powershell
$env:PYTHONPATH = "src"
.venv/Scripts/python.exe -m traffic_lstm.backtest --dataset motorway --output reports/backtests/NEW_NAME
.venv/Scripts/python.exe scripts/report_backtests.py
```

## Project layout

| Path | Purpose |
| --- | --- |
| `presentation/` | Interactive intersection, scene-based signal controller and historical LSTM replay |
| `app/intersection_images.py` | Standalone gallery for locally prepared aerial images |
| `scripts/*intersection*.py` | Select, download and verify the image subset |
| `src/traffic_lstm/observatory.py` | Durable workspace, access, jobs, models, forecasts, actuals and recovery |
| `src/traffic_lstm/operations.py` | CLI and bounded worker process |
| `src/traffic_lstm/observations.py` | Strict UTC interval-end observation contract |
| `src/traffic_lstm/candidate.py` | Portable operational tree/baseline and calibrated intervals |
| `src/traffic_lstm/pipeline_v2.py` | Causal preparation, training, packaging and inference |
| `src/traffic_lstm/evaluation_v2.py` | Seasonal climatology, direct lags and uncertainty |
| `src/traffic_lstm/research.py` | Recorded comparison protocol and CLI |
| `tests/` | Leakage mutation, timestamps, metrics, artifact and inference checks |
| `models/v2/` | Immutable run packages |
| `scripts/report_v2.py` | Report generated from verified packages |
| `obsidian_vault/Traffic_LSTM_Brain/` | Archived educational neuron/gate explanations |

V1 modules remain for reproducibility and education. Their preprocessing is not repaired retroactively. Rebuilding a legacy presentation does not produce v2 evidence.

## Product delivery

See [ROAD_PRODUCT_PLAN.md](ROAD_PRODUCT_PLAN.md) for current blockers and acceptance gates. The immediate product hypothesis is a read-only observatory that lets an engineer inspect observation quality and forecast errors. Local data access, untouched future/site holdouts and field interval coverage still require external validation. Local persistence, access controls, job isolation, calibration and recovery tests are implemented; managed identity/TLS, OS isolation and encrypted secret/backup handling are still required before shared hosting. More neurons or vehicle animation will not resolve these gaps.

The step-by-step implementation and verification record is [IMPLEMENTATION_LOG.md](reports/IMPLEMENTATION_LOG.md).
