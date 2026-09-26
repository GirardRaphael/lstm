# Local observatory operations

This is one trusted local workspace. It is not a multi-tenant service. Access tokens authorize application actions; they do not protect against someone who already controls the database files or your operating-system account. Keep the app on loopback. No traffic-controller connections exist.

## 1. Install and initialize

From a PowerShell terminal in the repository:

```powershell
./setup.ps1
./test.ps1
./observatory.ps1 init
```

Initialization prints a high-entropy administrator token once. Save it in your password manager. Do not put it in source control, screenshots, shell scripts or support logs. The local database is `var/observatory/observatory.sqlite3`; `var/` is ignored by Git. Initialization refuses to overwrite an existing database.

Enter the token into the app's password field. For CLI/worker commands, set it in the current terminal without echoing it:

```powershell
$secureToken = Read-Host 'Observatory token' -AsSecureString
$env:OBSERVATORY_TOKEN = [System.Net.NetworkCredential]::new('', $secureToken).Password
./run_app.ps1
```

Open `http://127.0.0.1:8501`. The application binds to loopback and uses Streamlit's XSRF protection. The token is held in that browser session, not written into the URL. Sign out when finished.

## 2. Run the bundled historical workflow

In a terminal that has `OBSERVATORY_TOKEN` set:

```powershell
./observatory.ps1 bundled --dataset motorway --hold-back 24
./observatory.ps1 list datasets
```

This preserves I-94 clock labels and assigns UTC **only for replay**. It does not certify the source timezone or interval-end semantics. Duplicate rows are collapsed only after verifying identical counts; conflicting counts are refused. The stream is explicitly named `historical-replay/i94-clock-as-utc`. Do not relabel it as live.

Copy the returned snapshot ID, then:

```powershell
./observatory.ps1 enqueue SNAPSHOT_ID
./observatory.ps1 worker --once
./observatory.ps1 list jobs
./observatory.ps1 list models
./observatory.ps1 activate MODEL_ID
./observatory.ps1 forecast SNAPSHOT_ID --replay
./observatory.ps1 bundled --dataset motorway --hold-back 0
./observatory.ps1 errors
```

The last import supplies actuals that were withheld when forecasting. The app exposes the same queue, activation, forecast and error-review actions. For a continuously polling worker, run `./observatory.ps1 worker` in a separate terminal. Training is never executed in a web request.

For a quick synthetic fixture use `./observatory.ps1 demo`; it is labeled synthetic and provides no field-validation evidence. Bike data are used in the research backtests, not mislabeled as vehicle observations in the operational schema.

## 3. Import your own observations

Required CSV columns:

| Column | Contract |
| --- | --- |
| `stream_id` | One site/approach/movement/sensor/aggregation version; nonempty |
| `timestamp` | End of a completed interval, explicit offset; converted to UTC |
| `interval_seconds` | One positive integer cadence per import and stream version |
| `vehicle_count` | Finite nonnegative integer count, not a rate or estimated fractional count |

```powershell
./observatory.ps1 import observations.csv --source 'Owner authorization and source reference' --mode historical
```

Imports reject naive/future timestamps, duplicates, disorder, off-grid intervals, mixed streams/cadences, nonfinite/negative/fractional counts and conflicting actuals. Missing intervals remain missing and are reported; they are never filled with zeros. Dataset snapshots are immutable and checksummed. Changing cadence/grid/count semantics requires a new stream version. Source identity is declared, not independently authenticated by this importer.

The portable candidate currently supports **hourly vehicle counts only**, with 24 complete recent observations and timestamp lags at 1/24/168 hours. Other cadences can be inspected/imported but cannot train this model. Units are vehicles per completed hour. A missing weekly lag stays missing for XGBoost; the recent 24-hour window must be complete.

## 4. Model and interval policy

The worker uses 64% fit, 16% validation, 10% calibration and 10% evaluation, split chronologically by raw timestamps. The tree and hour-of-week mean are selected on validation MAE. Calibration sizes nominal 90% residual intervals, then evaluation measures coverage and width. The product worker deliberately uses these two simple candidates; LSTM remains in the separately recorded research comparisons.

Activation is administrator-only and requires measured interval coverage ≥85% on at least 30 evaluation windows. The nominal target remains 90%; this 85% floor is an engineering release gate, not proof of calibrated field coverage. Activation never certifies street suitability.

Serving checks model/metadata checksums, stream and cadence, source mode, origin after model evidence, complete recent history and staleness. Live observations cannot bypass freshness through replay mode. Inputs with more than half the latest counts outside the training 1st–99th percentile range are blocked for review. This is a simple shift screen, not a comprehensive drift detector.

After at least 30 scored predictions for the active model, the latest 100 are checked. Coverage below 85% or MAE above twice the model's evaluation MAE (with a one-count floor) disables new forecasts with an explicit reason. Repeated forecasts for the same model/origin/mode are idempotent and do not inflate these statistics. Review source quality and model suitability; activate a reviewed version or disable the stream. Do not tune thresholds to hide failures.

## 5. Jobs, access and recovery

- One running job per workspace; at most five queued/running jobs.
- Worker subprocess timeout defaults to 600 seconds; permitted range is 1–3600 seconds. Tree execution is single-threaded.
- Imports: 10 MiB per file, 200,000 rows, 100 snapshots and 500 MiB canonical CSV content per workspace. These are application quotas, not an OS memory/container sandbox.
- At most 1,000 forecast attempts per day. Audit records track state changes without access tokens. Job logs are under `var/observatory/logs/`.
- Administrators manage users, workers, activation, audit and backups. Operators import, queue their own jobs, cancel/retry their jobs and forecast. Viewers can inspect/download workspace data. All roles share this one workspace; there is no tenant isolation.

```powershell
./observatory.ps1 cancel JOB_ID
./observatory.ps1 recover
./observatory.ps1 retry JOB_ID
./observatory.ps1 activate PREVIOUS_MODEL_ID
./observatory.ps1 deactivate STREAM_ID
./observatory.ps1 add-user analyst --role operator
./observatory.ps1 revoke analyst
./observatory.ps1 rotate-token
```

Cancellation terminates the subprocess and prevents model registration. Recovery marks jobs with a heartbeat older than 60 seconds failed; retry creates a new job and preserves the failure. An expired lease cannot commit a model after recovery. A crashed worker does not automatically retry expensive work.

Tokens expire after 30 days. Rotate before expiry and update the terminal/app token. Rotation immediately invalidates the previous token. Maintain a second administrator for recovery; if all administrator tokens expire, an authorized owner with filesystem access must perform offline account recovery. The app intentionally offers no unauthenticated reset endpoint. Revocation takes effect on the next action/rerun.

## 6. Backup and restore

```powershell
./observatory.ps1 backup backup-20260925.sqlite3
./observatory.ps1 --root var/restored restore backup-20260925.sqlite3
```

SQLite's backup API provides a consistent snapshot containing imports, native model bytes, metadata, jobs, users/token hashes, forecasts, actuals and audit records. Job log files are operational diagnostics and are not included. The backup is integrity-checked and never overwrites an existing destination. Restore requires a valid administrator token from that backup and a new destination directory.

After restore, run `list`, `errors`, and a known historical replay; recover abandoned jobs and review active versions before switching. Set `OBSERVATORY_ROOT` for the app and pass `--root` for CLI/worker commands to use the restored workspace. Keep backups in access-controlled storage; encryption at rest and managed secret storage are not implemented by this local application.

## 7. Verification and limits

```powershell
./test.ps1
.venv/Scripts/python.exe scripts/verify_observatory.py
```

The second command runs a real bundled-data workflow in an isolated ignored workspace and writes `reports/observatory_acceptance.json`. It never exports its test credential. It checks import, separate worker, activation, forecast, later actuals, deactivation/reactivation, backup/restore and prediction parity.

Before shared hosting or a pilot: add managed identity/TLS, deployment-specific OS isolation, encrypted backup/secret handling, monitoring/alert delivery and owner response procedures. Validate an authorized local feed and obtain operator acceptance. The UI, token system and historical tests do not substitute for those requirements.
