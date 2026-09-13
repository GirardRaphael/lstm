---
id: toc-04-contracts
type: contracts
status: draft
owner: principal
updated_utc: 2026-09-13T14:05:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# Data and ML Contracts

## Present on main (v1 — implemented)

- Chronological train/test split; scaler fit on pre-test rows (includes Keras validation segment) — see ROAD plan critical findings.
- Optional `drop_gapped_windows`.
- Artifacts: config, scaler (target; feature scaler incomplete on older multivariate), metrics, diagnosis.
- Metrics-only window records without weights must remain non-loadable.

## Required for M1 (planned / blocked on missing v2)

Versioned entities with stable IDs, timestamps, status, error semantics:

| Entity | Must record |
| --- | --- |
| Dataset version | source, hash, units, cadence, site scope, quality |
| Stream | identity, timezone policy, missing≠zero |
| Training job | config version, seed, folds, status |
| Model package | weights, both scalers, feature order, preprocess version, data hash, code revision, deps |
| Forecast | issue time, model version, horizon timestamps, unavailable reason |
| Evaluation | aligned windows, coverage, MAE/RMSE in physical units, null MAPE when undefined |

## Compatibility rule

`TrainingConfig` without `pipeline_version` must keep **v1** meaning for archived JSON. New callers must opt into v2 explicitly. Never silently reinterpret archived models.

## Street observation preflight (on readiness branch only)

Schema: `stream_id`, `timestamp` (offset required), `interval_seconds`, `vehicle_count`. See ROAD_PRODUCT_PLAN.md. **Not on main tip.**
