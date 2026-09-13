---
id: toc-00-start-here
type: index
status: active
owner: principal
updated_utc: 2026-09-13T14:00:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# Start Here — Traffic Observatory Context

This vault is the **authoritative development context** for the Traffic Observatory product build.
It is separate from the generated educational vault [[../Traffic_LSTM_Brain/00 Start Here|Traffic_LSTM_Brain]].

## Read in this order

1. [[01 Current State]] — what exists on disk right now
2. [[09 Next Session]] — checkout, tests, next bounded task
3. [[05 Roadmap and Task Board]] — milestone gates and task IDs
4. [[06 Verification Index]] — what was actually run
5. [[07 Risks and Blockers]] — especially the missing v2 branch

## Where truth lives

| Category | Owner file |
| --- | --- |
| Current branch/commit & capability labels | [[01 Current State]] |
| Product boundaries | [[02 Product Scope]] |
| Architecture decisions | [[03 Architecture]] + `Decisions/` |
| Dataset/model/job contracts | [[04 Data and ML Contracts]] |
| Task graph & owners | [[05 Roadmap and Task Board]] + `Tasks/` |
| Evidence of checks | [[06 Verification Index]] + `Evidence/` |
| Blockers | [[07 Risks and Blockers]] |
| How to run/recover | [[08 Runbook]] |
| Resume instructions | [[09 Next Session]] |
| Short human entry at repo root | [HANDOFF.md](../../HANDOFF.md) |
| Product acceptance criteria (from readiness PR) | [ROAD_PRODUCT_PLAN.md](../../ROAD_PRODUCT_PLAN.md) |
| Earlier audit | [AUDIT.md](../../AUDIT.md) |
| Implementation brief | [MASTER_BUILD_PROMPT.md](../../MASTER_BUILD_PROMPT.md) |

## Status vocabulary (do not blur)

- **verified now** — re-checked in this session on a named commit
- **historical evidence** — prior session claim; not re-run here
- **implemented but unverified** — code present; checks not yet run
- **planned** — described, not built
- **blocked** — cannot proceed without an external input
