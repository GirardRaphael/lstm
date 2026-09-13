---
id: toc-07-risks
type: risks
status: active
owner: principal
updated_utc: 2026-09-13T14:10:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# Risks and Blockers

## BLOCKER-01 — Missing `codex/temporal-pipeline-v2`

**Impact:** M1 cannot honestly claim continuity with the 2026-09-13 temporal-pipeline work.

**Searched:** local `Traffic_LSTM_Project` branches, `origin` heads, agent store, session transcripts, `/workspace/scratch/...` path (environment-specific; not this Windows host).

**Missing artifacts:** commit `646d3a7`, `tests/test_temporal.py`, pipeline_version=v2 training path, beijing observed CSV sidecar experiment results.

**Ask of user:** provide one of: git bundle, patch series, zip of the worktree, or another machine path containing that branch. Do not authorize silent recreation from the handoff prose alone.

## RISK-02 — Main advanced past readiness PRs

`main` @ `a652a04` includes ablation + window/multisite scripts that readiness PRs do not contain. Integrating PR #1/#2 requires a careful merge/rebase onto current main, not a fast-forward of main onto the PR tips.

## RISK-03 — Known v1 leakage/preprocessing issues

Documented in ROAD_PRODUCT_PLAN: weather look-ahead via full-frame median / bfill; scaler includes validation segment; gap windows kept by default; incomplete multivariate serving contract. These remain open on main.

## RISK-03b — Readiness tip has one failing check on Windows

`c3913ff` pipeline suite: 17/18 — `to_dict()` writes `data_path` with backslashes; test expects POSIX separators. Tracked as [[Tasks/FINDING-readiness-path-separator]] (severity low, fix deferred to integration).

## RISK-04 — Wrong Cursor workspace trap

`Documents\GitHub\LSTM` is an unrelated nearly empty repo. Always work in `Traffic_LSTM_Project` (or a clone of `GirardRaphael/lstm`).

## RISK-05 — Publication authorization

Prior HTML report MCP upload was user-rejected. Do not push/publish without fresh explicit approval.
