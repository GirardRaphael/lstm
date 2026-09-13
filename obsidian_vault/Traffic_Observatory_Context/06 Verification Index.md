---
id: toc-06-verification
type: verification-index
status: active
owner: principal
updated_utc: 2026-09-13T14:10:00Z
verified_commit: e50455d1c749e425fe984278e9c2153080a61120
---

# Verification Index

Only checks with an evidence note may be marked passed.

| Evidence ID | Command / workflow | Commit | Outcome | Label |
| --- | --- | --- | --- | --- |
| [[Evidence/EV-2026-09-13-pipeline]] | `PYTHONPATH=src python tests/test_pipeline.py` | `a652a04` (main) | 16/16 pass | verified now |
| [[Evidence/EV-2026-09-13-readiness-worktree]] | pipeline + saved-models + readiness suites in isolated worktree | `c3913ff` (readiness tip) | pipeline 17/18 (**1 fail: path separator**), saved-models 6/6, readiness 21/21 | verified now |
| [[Evidence/EV-2026-09-13-push-m0]] | `git push origin observatory/m0-recover`; `git rev-parse`; `git ls-remote` | `e50455d` (`observatory/m0-recover`) | local and remote SHAs match | verified now |
| EV-hist-temporal | temporal suite (12 checks) | prior session / `646d3a7` claim | claimed pass | historical evidence — **branch missing** |
| EV-hist-models | nine packages load+predict | prior session | claimed pass | historical evidence (6/6 re-verified on readiness tip) |
| EV-hist-app | seven Streamlit pages | prior session | claimed pass | historical evidence |

## Not run this session

- `tests/test_app.py` (present on readiness branch; Streamlit UI flow)
- `tests/test_temporal.py` on `observatory/pipeline-v2` (planned fresh v2 suite; not merged, not yet verified here)
- Fresh-environment install from lockfile (no lockfile on main)
- Browser workflow automation
