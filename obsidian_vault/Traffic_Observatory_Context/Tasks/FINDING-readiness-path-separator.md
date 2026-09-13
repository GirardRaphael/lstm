---
id: finding-readiness-path-separator
type: finding
status: open
severity: low
owner: data-ml
updated_utc: 2026-09-13T14:25:00Z
verified_commit: c3913fff57d58385644e7d0835d25ce1f5a83771
---

# FINDING: readiness branch fails its own path-portability test on Windows

## Reproduction

```powershell
git worktree add <wt> origin/codex/street-shadow-readiness
cd <wt>; $env:PYTHONPATH = "src"
python tests/test_pipeline.py
# -> 17 passed, 1 failed: config: saved Windows dataset paths remain portable
```

## Cause

`src/traffic_lstm/config.py::TrainingConfig.to_dict()` serializes the project-relative
dataset path using OS-native separators. On Windows that yields
`data\raw\Metro_Interstate_Traffic_Volume.csv`; the test (and cross-platform artifact
portability) expects `data/raw/Metro_Interstate_Traffic_Volume.csv`.

Path **resolution** (`resolve_data_path`) works correctly; only serialization differs.

## Proposed fix (when integration is authorized)

Normalize with `.as_posix()` (or equivalent) in `to_dict()` for `data_path`, and keep
`resolve_data_path` accepting both separators. Add a round-trip assert.

## Impact

Low for local runs; breaks the readiness branch's own CI gate on Windows and would
produce non-portable artifact JSON on Windows machines. Not release-blocking for M0.

## Status

Open. Deferred until readiness/v2 integration is unblocked (see [[../07 Risks and Blockers]] BLOCKER-01).
