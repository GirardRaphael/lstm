---
id: ev-2026-09-13-readiness-worktree
type: evidence
status: verified
owner: principal
updated_utc: 2026-09-13T14:25:00Z
verified_commit: c3913fff57d58385644e7d0835d25ce1f5a83771
---

# EV-2026-09-13-readiness-worktree

Suites run against `origin/codex/street-shadow-readiness` tip `c3913ff` in an isolated
git worktree (`Traffic_LSTM_readiness_wt`), not merged into main.

| Suite | Command | Outcome |
| --- | --- | --- |
| pipeline | `PYTHONPATH=src python tests/test_pipeline.py` | **17 passed, 1 failed** |
| saved models | `python tests/test_saved_models.py` | **6 passed** |
| readiness | `python tests/test_readiness.py` | **21 passed** (unittest `OK`) |

Environment: Windows 10.0.26200, Python 3.12.10, TF CPU 2.21.0, NumPy 2.5.3.

## Failing check

`config: saved Windows dataset paths remain portable` — see
[[../Tasks/FINDING-readiness-path-separator]].

Root cause: `TrainingConfig.to_dict()` emits the relative dataset path with Windows
backslashes (`data\raw\...`) while the test expects POSIX separators (`data/raw/...`).
Direct reproduction confirms resolution itself is correct; only separator
normalization in serialization fails on Windows.

## Notes

- Historical claim "21 readiness checks pass" is **confirmed** on `c3913ff`.
- Historical claim "18 pipeline checks" matches this branch's 18 checks (17 pass here).
- The 12 temporal checks belong to the missing v2 branch and remain unverifiable.
