# Repository map

GitHub: https://github.com/GirardRaphael/lstm

## Current integration — 2026-11-14

Use **`main`** for the live intersection, the authenticated observatory and the
research workbench. The intersection has been selectively added on top of
[`5156433`](https://github.com/GirardRaphael/lstm/commit/5156433), preserving
main's modern operational application, training pipeline and model packages.
The older presentation branch was not merged wholesale.

`presentation/` originates from `observatory/phase2-v2-wiring` at `0b21861` and
now includes a controller that chooses the simulated green direction and
duration from visible vehicles, stopped queues, pedestrian requests and waiting
time. It applies decisions after the required clearance and records their
reasons. The historical LSTM remains separately inspectable: its displayed
forecast comes from a stored Metro Interstate example and does not choose the
lights. The controller reads simulation objects, not pixels from a camera.

| Command from the repository root | What opens |
| --- | --- |
| `./run_app.ps1` | Live intersection and its local forecast service |
| `./run_app.ps1 -Observatory` | Authenticated observation, training-job and forecast workflow |
| `./run_app.ps1 -Workbench` | Research evidence viewer and educational neuron tools |
| `./run_app.ps1 -Images` | Local aerial-image dataset gallery |

See [the main README](../README.md) for setup,
[presentation/README.md](../presentation/README.md) for the simulation and
controller, [OPERATIONS.md](OPERATIONS.md) for the observatory, and
[the dataset specification](../reports/INTERSECTION_DATASET_SPEC.md) for the
local image subset. Downloaded aerial images are not included in Git.

This remains a local research project. The simulated signal controller is
separate from the read-only forecasting workflow; neither is connected to real
traffic signals. Research model comparisons and their limitations remain in
[reports/REPORT.md](../reports/REPORT.md) and the linked versioned packages.

## Historical branch comparison — checked 2026-09-28

The remainder records the branch comparison made on **2026-09-28**. Commit
counts, remote state and recommendations in this section describe that date,
not the current integration. At that time every branch below was already on
GitHub, local `main` matched GitHub `main`, and the presentation was separate.
The current launch instructions above supersede the old instruction to switch
branches for the intersection.

The empty commit `af566aa` (“Initialize project”) existed only on the Cursor
temporary remote. It was not part of this repository and was not pushed.

### Historical `main`

Tip: [`105734e`](https://github.com/GirardRaphael/lstm/commit/105734e) — 2026-09-28 — “Score a last-week baseline and serve the validation winner”.

This was the research and observatory line at the time of the comparison.

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

### Historical side-branch status

All of these were already on GitHub. None of them was merged on 2026-09-28.
The notes below preserve the reasons recorded then.

#### `observatory/phase2-v2-wiring`

https://github.com/GirardRaphael/lstm/tree/observatory/phase2-v2-wiring

Tip: `0b21861` — 2026-09-25 — “Keep roll-down only on the decision ledger”.

At the time, 34 commits were not in `main`. The histories split at `a652a04`.
This branch held the slideshow and the 2D intersection (`presentation/`).
Later commits on it:

- Cars stop on red. Opposite approaches are green together. West-right and east-left stay out of the oncoming lane.
- Pedestrians wait in corner groups and cross together on an all-red walk. New arrivals wait for the next walk.
- The neural graph lights every layer. That lighting is scaled so the tree is visible. It is not a weight-accurate replay of a closed-loop controller.
- Slides state the real comparison: XGBoost won on the motorway, the bike MAE “win” is not a finding.
- The live forecast is a NumPy replay of `baseline_univariate.keras` on the closest stored Metro Interstate hour. Green time is `3.5 + 0.55 × queue`. The bottom ledger separates forecast error from that heuristic, and a correction is your override, not training.
- Only the decision ledger expands and collapses. The intersection, brain, and controls stay put.

The recommendation then was to keep this branch separate because merging it
wholesale would also bring an older pipeline into the observatory history.
The current integration resolves this by copying the presentation selectively
and adding the scene-based controller on top of newer main code.

#### `observatory/pipeline-v2`

Tip: `0c0e62d` — 2026-09-13 — “Add causal training pipeline v2 with versioned artifact packages”.

This is the historical source of `pipeline_v2.py`. `main` already contains that pipeline, recovered without merging the rest of the old branch. The branch stays as the record of that commit. There is nothing new on it to push.

#### `observatory/readiness-integration`

Tip: `c559e33` — 2026-09-13 — “Session checkpoint 2026-09-13: branch/worktree map, verified evidence, resume plan”.

A session note plus the two Codex commits below. Superseded as a working line by `main` after `8900c36`. Not a release branch.

#### `codex/street-shadow-readiness`

Tip: `c3913ff` — 2026-09-12 — “Add read-only street data preflight and harden forecast workflow”.

Adds `src/traffic_lstm/street_data.py`, an audit note, and an older product plan. Also rewrites README, comparison figures, and `reports/model_comparison.json` relative to the 2026-09-12 tree.

Not merged into current `main`. [ROAD_PRODUCT_PLAN.md](../ROAD_PRODUCT_PLAN.md) withdraws the claim that street preflight is in this checkout, because those files are not here. Replaying this branch onto today’s `main` would also bring back that older comparison write-up. It is already pushed. It is not a clean update.

#### `codex/lstm-reliability-fixes`

Tip: `f1ca0a0` — 2026-09-12 — “Fix portability and add end-to-end validation”.

Parent of the street-shadow branch. Already pushed. Already included in `codex/street-shadow-readiness`. Same reason not to merge it by itself.

#### `observatory/m0-recover`

Tip on the remote: `9e44f34` — 2026-09-13 — “Checkpoint: aborted steward vault update”.

The useful part of this branch was merged in pull request #3 (`4d7e284` contains `e50455d`). The tip after that merge is an aborted vault edit. Do not merge `9e44f34`.

### Interpretation recorded on 2026-09-28

- The task is next-hour counts, not signal control.
- On the motorway data, XGBoost beat the LSTM. A tree on the clock and recent lags is the default candidate.
- The LSTM remains in the repo so its gates can be inspected. Losing on the test set is the result, not a temporary slide.
- v1 scores were fit with leaks (scaler saw validation, gaps could be crossed, seasonal naive used row offsets). v2 is the pipeline that blocks those. New packages live under `models/v2/`.
- The intersection is a simulation. The forecast shown there is a stored hour from the univariate LSTM, which is not the winning model. The light timing is a queue rule.
- No branch in this map is a street deployment.

### Commands recorded on 2026-09-28

```text
git fetch github
git rev-parse github/main                         -> 105734e
git rev-parse github/observatory/phase2-v2-wiring -> 0b21861
git rev-list --count github/main..github/observatory/phase2-v2-wiring -> 34
```

Local `main` was even with `github/main`. No other local branch existed. No side-branch commit was missing from GitHub, so no extra push was required.
