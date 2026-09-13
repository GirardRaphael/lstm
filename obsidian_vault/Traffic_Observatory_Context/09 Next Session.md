---
id: toc-09-next-session
type: next-session
status: active
owner: principal
updated_utc: 2026-09-13T14:10:00Z
verified_commit: e50455d1c749e425fe984278e9c2153080a61120
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
git checkout observatory/m0-recover   # M0 checkpoint: e50455d (pushed)
git status
$env:PYTHONPATH = "src"
C:\Users\azulr\.venvs\traffic_lstm\Scripts\python.exe tests/test_pipeline.py
```

A detached-head worktree of the readiness tip exists at
`C:\Users\azulr\OneDrive\Desktop\Traffic_LSTM_readiness_wt` (`c3913ff`) — reuse it for
readiness verification or remove with `git worktree remove` once integration lands.

## Unfinished task IDs

- **M1-00** — integrate readiness branch on `observatory/readiness-integration`; include the Windows path fix for `TrainingConfig.to_dict()` (`data_path` repo-relative with `.as_posix()`)
- **M1-01** — continue fresh pipeline v2 build on `observatory/pipeline-v2`; keep it explicitly clean-room and not a recovery of `646d3a7`
- **WEB-01** — browser-verify the `observatory/web-demo` page: LSTM neuron/gate visualization, CSV import/explorer, Three.js read-only intersection, 4+ scenario presets
- **M1 phase 2** — wire `pipeline_version` into `TrainingConfig` and CLI after the fresh module/tests phase is ready

## Current failure / gap reproduction

```text
git branch -a | findstr temporal
# expected today: no matches
git cat-file -t 646d3a7
# expected today: fatal / missing
```

## Next bounded task (when unblocked)

Integrate the active branches in a bounded order: readiness integration first, then the fresh pipeline-v2 branch, then web-demo verification. Keep `pipeline_version` wiring into `TrainingConfig`/CLI as phase 2 after the module and temporal tests are stable.

If the original v2 bundle ever surfaces, verify its tip, run its temporal + pipeline + readiness suites, and diff it against the fresh implementation before changing provenance claims.

## Do not do next

- Another architecture / dropout / calendar ablation under v1
- Push/PR/publish without explicit authorization
- Merge readiness PRs blindly onto main without resolving window-sweep divergence
- Describe the fresh v2 implementation as recovery of commit `646d3a7`
