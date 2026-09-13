---
id: toc-01-current-state
type: current-state
status: active
owner: principal
updated_utc: 2026-09-13T14:05:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# Current State

## Checkout (verified now)

| Item | Value |
| --- | --- |
| Workspace (after move) | `C:\Users\azulr\OneDrive\Desktop\Traffic_LSTM_Project` |
| Wrong empty Cursor folder | `...\Documents\GitHub\LSTM` → remote `LSTM_TrafficLightControl`, only `.gitattributes` |
| Remote | `https://github.com/GirardRaphael/lstm.git` |
| Branch | `observatory/m0-recover` @ `4ce771d` (M0 checkpoint, local only — not pushed) |
| Base | `main` = `origin/main` = `a652a04` |
| Working tree before M0 edits | clean, up to date with `origin/main` |
| Python | 3.12.10 via `C:\Users\azulr\.venvs\traffic_lstm` |
| TF / Keras / NumPy / pandas / sklearn / Streamlit / XGB | 2.21.0 / 3.15.1 / 2.5.3 / 3.0.5 / 1.9.1 / 1.63.0 / 3.4.1 |

## Remote branches (verified now)

| Branch | Tip | Relation to main |
| --- | --- | --- |
| `main` | `a652a04` | current published product baseline |
| `codex/lstm-reliability-fixes` | `f1ca0a0` | behind main; merge-base `b700b69` |
| `codex/street-shadow-readiness` | `c3913ff` | stacks on reliability; adds street preflight, AUDIT, ROAD plan, readiness tests |
| `codex/temporal-pipeline-v2` | **absent** | **blocked** — not local, not on GitHub, not in agent stores/transcripts |

## What is implemented on `main` @ `a652a04` (code present)

Educational LSTM workbench: train/evaluate/introspect, Streamlit multi-page app, Obsidian brain vault, XGBoost benchmarks, ablation/window-sweep/multisite experiment scripts, nine `.keras` packages + two metrics-only window sweep JSON records (`window_012`, `window_024`).

Pipeline behaviour is the **legacy v1** contract: scaler fit on full pre-test slice, optional gap dropping, weather/calendar feature path with documented look-ahead risks in ROAD plan. **No** `pipeline_version` field in `TrainingConfig`.

## Labels for claimed v2 work (historical / blocked)

The 2026-09-13 continuity note claimed local branch `codex/temporal-pipeline-v2` with continuity commit `646d3a7`, 12 temporal checks, fit-only scaling, explicit chronological validation, etc. **None of that code is recoverable from this machine or from GitHub today.** Treat those claims as **historical evidence**, not **verified now**. Do not silently recreate them.

## Verification this session

| Check | Result | Label |
| --- | --- | --- |
| `tests/test_pipeline.py` on main | **16/16 pass** on `a652a04` | verified now |
| Readiness tip `c3913ff` in isolated worktree | pipeline **17/18** (1 fail: `to_dict` path separators on Windows — [[Tasks/FINDING-readiness-path-separator]]), saved-models **6/6**, readiness **21/21** | verified now |
| `tests/test_temporal.py` (12 checks) | file exists nowhere reachable | historical evidence — **blocked** |
| Prior session: nine models, seven app pages | not re-run here | historical evidence |

## Gaps vs product mission

| Capability | Status |
| --- | --- |
| Causal pipeline v2 | **blocked** (missing branch) |
| Durable jobs / auth / registry / monitoring product UI | **planned** |
| Multivariate portable inference | incomplete (known on readiness notes) |
| Dependency lock / CI on main | CI exists on readiness branch only |
| Beijing observed (non-interpolated) dataset experiment | script intent on main; full experiment not verified |

## Publication

No push of M0 work yet. Historical HTML report upload rejection remains unresolved; do not republish without explicit authorization.
