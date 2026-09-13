---
id: toc-06-verification
type: verification-index
status: active
owner: principal
updated_utc: 2026-09-13T14:10:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# Verification Index

Only checks with an evidence note may be marked passed.

| Evidence ID | Command / workflow | Commit | Outcome | Label |
| --- | --- | --- | --- | --- |
| [[Evidence/EV-2026-09-13-pipeline]] | `PYTHONPATH=src python tests/test_pipeline.py` | `a652a04` (main) | 16/16 pass | verified now |
| [[Evidence/EV-2026-09-13-readiness-worktree]] | pipeline + saved-models + readiness suites in isolated worktree | `c3913ff` (readiness tip) | pipeline 17/18 (**1 fail: path separator**), saved-models 6/6, readiness 21/21 | verified now |
| EV-hist-temporal | temporal suite (12 checks) | prior session / `646d3a7` claim | claimed pass | historical evidence — **branch missing** |
| EV-hist-models | nine packages load+predict | prior session | claimed pass | historical evidence (6/6 re-verified on readiness tip) |
| EV-hist-app | seven Streamlit pages | prior session | claimed pass | historical evidence |

## Not run this session

- `tests/test_app.py` (present on readiness branch; Streamlit UI flow)
- `tests/test_temporal.py` (nowhere recoverable)
- Fresh-environment install from lockfile (no lockfile on main)
- Browser workflow automation
