# Implementation log: local Traffic Observatory

## Continuation verification - 2026-09-25

Resumed from the saved implementation record and verified the current checkout after the project briefing changes. `./test.ps1` passed all 16 legacy checks and 25 pytest tests (32 existing Keras/NumPy deprecation warnings). `git diff --check` passed with line-ending notices only. Updated the root handoff to supersede the obsolete missing-v2 recovery instructions. No further unfinished local implementation task was identified in the latest delivery record. External acceptance, remote CI and external-browser visual review remain open as documented below. No push, merge or deployment was performed.

## Scope and acceptance plan — 2026-09-25

User requested completing the remaining issues step by step and documenting the process.
Prior changes are retained. No push, merge or deployment is included.

1. **Input contract and persistence:** explicit UTC interval-end observations, one stream/cadence, immutable imports, quality reports, bounded storage and authenticated access.
2. **Durable execution:** persisted jobs with atomic claiming, cancellation, retry and abandoned-job recovery; training in a separate worker process.
3. **Model operations:** locally produced model registry, explicit activation/rollback, schema/provenance checks, stored forecasts, actual/error joins, unavailable states and audit events.
4. **Evaluation:** recorded rolling-origin, multi-seed comparisons with disjoint calibration/evaluation periods; calibrated prediction intervals and interval-coverage/width diagnostics, including bootstrap block-length sensitivity.
5. **Operator workflow:** authenticated Streamlit import/job/model/forecast/error views, CLI and documented worker/backup/restore commands.
6. **Verification:** adversarial contract tests, restart/concurrency/recovery tests, end-to-end model workflow, application tests and browser inspection where available.

External acceptance remains separate: authorized local feed, certified source identity and interval semantics, untouched future/site data, traffic-engineer review and measured operator value. Synthetic and bundled-data tests cannot satisfy those conditions.

## Initial inventory

- Prior v2 packages and tests are present in the working tree; legacy educational paths remain.
- No AGENTS.md found in the repository. The implementation brief calls for read-only analytics, not signal control.
- Runtime: project Python 3.12 environment and hashed requirements lock already available.
- Shared hosting is not being enabled. The new workflow will bind to loopback by default and require a local access token for operational actions.

Further entries will record implementation decisions, commands, results and remaining limitations as each step is verified.

## Step 1 — Data and workspace contracts

Implemented `observations.py` and the SQLite workspace in `observatory.py`.
Imports require one stream, one cadence, offset-aware completed-interval timestamps and finite nonnegative integer counts. Gaps are measured, not filled. Duplicate/disordered/future/off-grid rows and conflicting actuals are rejected. A stream's cadence and time grid cannot silently change. Immutable canonical snapshots have SHA256 checksums and explicit historical/live/synthetic provenance.

The first contract tests exposed a pandas 3 timestamp-resolution assumption: integer timestamp storage is not always nanoseconds. Replaced raw integer division with timedeltas expressed in seconds. DST fall-back and grid tests now pass. A later test initially depended on wall-clock minutes; made its explicit `now` parameter deterministic.

Workspace credentials are random high-entropy tokens stored only as hashes, with roles, revocation, expiration and rotation. No passwords or tokens are logged. This is one local workspace, not tenant isolation. Application authorization does not secure a compromised OS account or database file.

## Step 2 — Durable jobs and recovery

Implemented a bounded queue with transactional claiming, one running worker per workspace, separate training subprocesses, heartbeats, timeouts, cancellation, immutable failed-job records and explicit retries. An expired lease cannot register a model. The initial timeout test showed real training could finish in less than a second; replaced timing assumptions with an injected sleeping subprocess to test actual termination deterministically.

Training uses a fixed portable direct-tree/weekly-mean comparison, validation-only selection, separate calibration and final evaluation. Native model bytes and metadata are stored transactionally in SQLite; no Python pickle loading or uploaded model execution is supported. The LSTM remains in the separate research evaluation, not the operator worker.

## Step 3 — Forecast operations

Added administrator activation/deactivation/rollback, model and metadata integrity checks, origin-after-evidence checks, source-mode/cadence matching, explicit unavailable states, immutable forecasts and later-actual joins. Predictions for the same stream/model/origin/mode are idempotent, preventing repeated clicks from inflating monitoring statistics.

Intervals use a separate calibration block. Activation requires ≥85% measured interval coverage on at least 30 evaluation windows, against a nominal 90% target. This is an engineering gate, not field validation. Serving blocks stale/gapped/mismatched data, severe training-range shift, and—after at least 30 scored forecasts—coverage below 85% or MAE above twice the recorded evaluation error. The UI and audit log expose reasons. External notification routing is not configured.

## Step 4 — Broader science and a material finding

Executed `traffic_lstm.backtest` for motorway and bikes: three expanding-origin folds × seeds 7/42/123 × two trained model families, plus baselines. Eighteen fold/seed comparisons completed. The maximum LSTM budget was fixed at 20 epochs with 32/16 units; these are not tuned-best LSTM claims. Calibration/evaluation timestamps are disjoint, and evaluation periods across folds do not overlap. All predictions are retained as CSV evidence.

Generated `reports/BACKTEST_REPORT.md`. Validation selected XGBoost in all recorded fold/seed runs. Its observed nominal-90% interval coverage ranged 89.7–93.1% on motorway but 76.9–90.2% on bikes. This is evidence of undercoverage under shift, not a problem that can honestly be erased by changing a label. Daily/3-day/weekly paired block sensitivity, seed variation, bias, peak MAE and interval widths are recorded per run.

## Step 5 — Operator interface and runbook

Added authenticated `app/observatory_app.py`, the new `run_app.ps1` entry point, `observatory.ps1`, `setup.ps1`, loopback/XSRF/upload configuration and `docs/OPERATIONS.md`. The UI covers import, persistent jobs, model versions, activation/disable, historical replay, actual/error review and audit/recovery. Existing educational pages remain available separately.

CLI supports bundled historical import, synthetic fixtures, worker management, token/user operations and backup/restore. The bundled importer explicitly assigns UTC to the original clock labels for replay; it does not claim those labels are certified UTC interval ends. Bike rentals are not mislabeled as vehicles in the operational schema.

## Step 6 — Verification evidence

- `./test.ps1`: **16 legacy checks passed; 22 pytest tests passed**, including the aggregate of 18 original temporal checks, the original seven v2 evaluation tests, nine observation/operations tests, two authenticated application flows, two artifact-separation checks and the legacy evidence-viewer test.
- `scripts/verify_observatory.py`: **passed on actual bundled I-94 data**. Imported a snapshot withholding 24 observations; queued and executed a separate worker; activated the selected model; forecast; imported the withheld actuals; joined error; disabled/reactivated; backed up and restored with identical forecast output. Evidence: `reports/observatory_acceptance.json`.
- That operational candidate scored MAE 139.31 on its distinct 3,960-window final evaluation block, with 94.67% interval coverage and mean interval width 808.55 counts. It uses a different split from the earlier 153.84 result; they must not be ranked as a controlled improvement.
- Streamlit launched on `127.0.0.1:18501`; root and `/_stcore/health` returned HTTP 200, health body `ok`.
- Browser runtime discovery returned no available browsers. **No external-browser visual verification is claimed.** AppTest exercised login/invalid/revoked/viewer states and the full queue/activate/forecast/actual workflow.
- `git diff --check` passed. Existing Keras/NumPy deprecation warnings remain visible; all checks pass. CI is configured but not yet run remotely.

## What is still not completed

Authorized field data, certified source semantics, untouched future/site evidence, operator utility validation and field interval performance require external input. Shared hosting still requires managed identity/TLS, OS/container isolation, encrypted secrets/backups and alert delivery. These are not represented as completed by a local token system. All changes remain local; no push, merge or deployment occurred.

## Landing-page project briefing

Added a 12-slide briefing before initialization/sign-in and on the authenticated Overview. It provides Previous/Next navigation, a section selector, progress, source details and a complete Markdown download. Quantitative result tables, dataset coverage and rolling-origin interval ranges are loaded from committed manifests/results, not copied into presentation prose. Missing/incomplete evidence shows unavailable state instead of substitute numbers. No operational workspace records are read by the public briefing.

Content distinguishes the single-split v2 comparison, operational 64/16/10/10 candidate protocol and three-fold/three-seed research. It explains metric units, the inconclusive bike MAE gap, interval undercoverage, different model information sets, historical replay semantics and remaining field/shared-hosting gates.

Verification: five targeted briefing/authenticated-app tests passed, including navigation through every slide, end controls, artifact-to-table agreement, missing-evidence behavior and continued login/role/workflow operation. This is automated application verification, not an external-browser visual review.
