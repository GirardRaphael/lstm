# Repository map

GitHub: https://github.com/GirardRaphael/lstm

Checked 2026-09-28. Every branch below is already on that remote. This checkout’s `main` matches GitHub `main`. Nothing else was sitting only on this machine.

The empty commit `af566aa` (“Initialize project”) exists only on the Cursor temporary remote. It is not part of this repository and was not pushed.

## What to open

Use **`main`** when you are explaining the training work, the observatory, or the numbers.

Use **`observatory/phase2-v2-wiring`** when you are showing the live intersection. That branch is a presentation. It does not replace `main`, and it was not merged.

## `main`

Tip: [`105734e`](https://github.com/GirardRaphael/lstm/commit/105734e) — 2026-09-28 — “Score a last-week baseline and serve the validation winner”.

This is the line to defend.

| Commit | Date | What it actually did |
| --- | --- | --- |
| `105734e` | 2026-09-28 | Direct-lag XGBoost (`t-1`, `t-24`, `t-168` plus the clock at the target hour) is the default tree. Every v2 run also scores a timestamp-true last-week baseline. A saved package forecasts the candidate that won on **validation** MAE, not the LSTM by default. |
| `8900c36` | 2026-09-26 | Local observatory: import counts, train in a worker, store forecasts, join later actuals, activate or roll back a model. Causal pipeline v2, hour-of-week climatology, timestamp-true yesterday, peak-hour MAE, block-bootstrap intervals, and a three-fold three-seed backtest. |
| `4d7e284` | 2026-09-13 | Merge of pull request #3, `observatory/m0-recover`, through `e50455d` only. |
| `e50455d` | 2026-09-13 | Recovery notes and the project-context vault. Records that the unpublished v2 branch was missing at that moment. |
| `a652a04` | earlier | Two experiments on where an LSTM is supposed to win. This is the last commit shared with the presentation branch. |

Published motorway comparison, still the exploratory v1 table in `reports/model_comparison.json`:

| Candidate | MAE (vehicles/hour) |
| --- | ---: |
| XGBoost + calendar | 154.3 |
| LSTM + calendar | 201.4 |
| LSTM univariate | 228.8 |
| Last hour | 585.6 |

On bikes the MAE gap is about 0.1 rental/hour and MAPE favors XGBoost. Do not say the verdict flips. Do not say the LSTM drives the lights.

Read next, in this order:

1. [ROAD_PRODUCT_PLAN.md](../ROAD_PRODUCT_PLAN.md) — what is implemented and what is still only a claim.
2. [reports/REPORT.md](../reports/REPORT.md) — the comparison and its limits.
3. [reports/BACKTEST_REPORT.md](../reports/BACKTEST_REPORT.md) — folds, seeds, interval coverage.
4. [docs/OPERATIONS.md](OPERATIONS.md) — how to run the local observatory.
5. [reports/IMPLEMENTATION_LOG.md](../reports/IMPLEMENTATION_LOG.md) — what was verified on 2026-09-25.

## Branches that are not `main`

All of these are already on GitHub. None of them was merged on 2026-09-28. Reasons are below. “Ready to push” is not the same as “ready to merge into `main`”.

### `observatory/phase2-v2-wiring`

https://github.com/GirardRaphael/lstm/tree/observatory/phase2-v2-wiring

Tip: `0b21861` — 2026-09-25 — “Keep roll-down only on the decision ledger”.

34 commits are not in `main`. The histories split at `a652a04`. This branch is the slideshow and the 2D intersection (`presentation/`). Later commits on it:

- Cars stop on red. Opposite approaches are green together. West-right and east-left stay out of the oncoming lane.
- Pedestrians wait in corner groups and cross together on an all-red walk. New arrivals wait for the next walk.
- The neural graph lights every layer. That lighting is scaled so the tree is visible. It is not a weight-accurate replay of a closed-loop controller.
- Slides state the real comparison: XGBoost won on the motorway, the bike MAE “win” is not a finding.
- The live forecast is a NumPy replay of `baseline_univariate.keras` on the closest stored Metro Interstate hour. Green time is `3.5 + 0.55 × queue`. The bottom ledger separates forecast error from that heuristic, and a correction is your override, not training.
- Only the decision ledger expands and collapses. The intersection, brain, and controls stay put.

Leave it as its own branch. Merging it into `main` would drop a demo application and an older copy of the pipeline onto the observatory history.

### `observatory/pipeline-v2`

Tip: `0c0e62d` — 2026-09-13 — “Add causal training pipeline v2 with versioned artifact packages”.

This is the historical source of `pipeline_v2.py`. `main` already contains that pipeline, recovered without merging the rest of the old branch. The branch stays as the record of that commit. There is nothing new on it to push.

### `observatory/readiness-integration`

Tip: `c559e33` — 2026-09-13 — “Session checkpoint 2026-09-13: branch/worktree map, verified evidence, resume plan”.

A session note plus the two Codex commits below. Superseded as a working line by `main` after `8900c36`. Not a release branch.

### `codex/street-shadow-readiness`

Tip: `c3913ff` — 2026-09-12 — “Add read-only street data preflight and harden forecast workflow”.

Adds `src/traffic_lstm/street_data.py`, an audit note, and an older product plan. Also rewrites README, comparison figures, and `reports/model_comparison.json` relative to the 2026-09-12 tree.

Not merged into current `main`. [ROAD_PRODUCT_PLAN.md](../ROAD_PRODUCT_PLAN.md) withdraws the claim that street preflight is in this checkout, because those files are not here. Replaying this branch onto today’s `main` would also bring back that older comparison write-up. It is already pushed. It is not a clean update.

### `codex/lstm-reliability-fixes`

Tip: `f1ca0a0` — 2026-09-12 — “Fix portability and add end-to-end validation”.

Parent of the street-shadow branch. Already pushed. Already included in `codex/street-shadow-readiness`. Same reason not to merge it by itself.

### `observatory/m0-recover`

Tip on the remote: `9e44f34` — 2026-09-13 — “Checkpoint: aborted steward vault update”.

The useful part of this branch was merged in pull request #3 (`4d7e284` contains `e50455d`). The tip after that merge is an aborted vault edit. Do not merge `9e44f34`.

## What you can say out loud

- The task is next-hour counts, not signal control.
- On the motorway data, XGBoost beat the LSTM. A tree on the clock and recent lags is the default candidate.
- The LSTM remains in the repo so its gates can be inspected. Losing on the test set is the result, not a temporary slide.
- v1 scores were fit with leaks (scaler saw validation, gaps could be crossed, seasonal naive used row offsets). v2 is the pipeline that blocks those. New packages live under `models/v2/`.
- The intersection is a simulation. The forecast shown there is a stored hour from the univariate LSTM, which is not the winning model. The light timing is a queue rule.
- No branch in this map is a street deployment.

## What was checked before writing this

```text
git fetch github
git rev-parse github/main                         -> 105734e
git rev-parse github/observatory/phase2-v2-wiring -> 0b21861
git rev-list --count github/main..github/observatory/phase2-v2-wiring -> 34
```

Local `main` was even with `github/main`. No other local branch existed. No side-branch commit was missing from GitHub, so no extra push was required.
