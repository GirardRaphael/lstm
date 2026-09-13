# Master build prompt — Traffic Observatory

This file is the **implementation brief** for agent-assisted product work.
It is not a claim that future product capabilities already exist.

For day-to-day resume state, open `obsidian_vault/Traffic_Observatory_Context`
(`00 Start Here` → `09 Next Session`). Keep this brief stable; put live status in the vault.

## Mission

Build a usable **read-only** Traffic Observatory for observation streams, data quality,
forecast training/comparison, forecast review and error measurement. Keep the LSTM
educational workspace. Do not control signals or roadside hardware.

## Instruction hierarchy

1. User authorization and environment permissions
2. This brief + vault Current State / blockers
3. `ROAD_PRODUCT_PLAN.md` acceptance intent
4. Existing code and tests on the recovered commit

Investigate before changing code. Support completion claims with evidence.
Never relabel **planned** work as **verified**.

## Milestones

| ID | Deliverable | Exit gate |
| --- | --- | --- |
| M0 | Inventory, baseline, context vault, task graph | Local/remote state identified; missing branch recorded |
| M1 | Causal ML contracts, matched eval, portable packages | Future-mutation/split tests; uni/multi save-load; legacy readable |
| M2 | Import → persist → job → forecast slice | Survives restart; clear invalid input; isolated outputs |
| M3 | Operational UI flows + monitoring + education access | Real browser/API/storage verification |
| M4 | Auth, limits, lockfile, CI, recovery | Isolation, bounded jobs, fresh setup |
| M5 | Release candidate + accurate docs | No release-blocking defect; publication state explicit |

## Agent roles (cap: 4 active)

Data/ML, Backend, Frontend/product, QA/reliability, Platform, Context steward.
Principal owns architecture, gates and integration. Assign exclusive file ownership.
Do not force-push, merge main, deploy or change permissions without principal + user auth.

## Hard ML rules

- Chronological splits before fitting transforms; purge horizon crossover
- No future-value filling of historical inputs; no invented targets for scoring
- Version configs: archived JSON without version remains **v1**
- Missing weights on metrics-only records must not load as models
- Report MAE/RMSE in physical units; undefined MAPE as JSON null

## Publication

A past push instruction does not override a later rejection. Report exactly whether a
branch was pushed, a PR opened, a merge landed, or a deploy occurred.

## First action for any new session

Recover repository status, read the context vault, run documented checks, then
implement the earliest unmet milestone that is not blocked.
