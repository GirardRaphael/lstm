---
id: toc-09-next-session
type: next-session
status: active
owner: principal
updated_utc: 2026-09-13T14:10:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# Next Session

## Read first

1. [[00 Start Here]]
2. [[01 Current State]]
3. [[07 Risks and Blockers]]

## Checkout / setup

```powershell
cd C:\Users\azulr\OneDrive\Desktop\Traffic_LSTM_Project
git fetch --all --prune
git checkout observatory/m0-recover   # M0 checkpoint: 4ce771d (local only)
git status
$env:PYTHONPATH = "src"
C:\Users\azulr\.venvs\traffic_lstm\Scripts\python.exe tests/test_pipeline.py
```

A detached-head worktree of the readiness tip exists at
`C:\Users\azulr\OneDrive\Desktop\Traffic_LSTM_readiness_wt` (`c3913ff`) — reuse it for
readiness verification or remove with `git worktree remove` once integration lands.

## Unfinished task IDs

- **M0-05** — obtain unpublished `codex/temporal-pipeline-v2` (BLOCKED)
- **M0-06** — resume-check by a second reader
- **M1-01** — integrate v2 once recovered (do not invent from prose)

## Current failure / gap reproduction

```text
git branch -a | findstr temporal
# expected today: no matches
git cat-file -t 646d3a7
# expected today: fatal / missing
```

## Next bounded task (when unblocked)

If user supplies the v2 branch/bundle: verify its tip, run its temporal + pipeline + readiness suites, compare to `a652a04`, then open an integration branch. If user authorizes reconstruction instead: write Decision note with missing evidence, implement v2 behind explicit `pipeline_version="v2"`, keep v1 default for archived JSON, add `tests/test_temporal.py` before any product UI work.

## Do not do next

- Another architecture / dropout / calendar ablation under v1
- Push/PR/publish without explicit authorization
- Merge readiness PRs blindly onto main without resolving window-sweep divergence
