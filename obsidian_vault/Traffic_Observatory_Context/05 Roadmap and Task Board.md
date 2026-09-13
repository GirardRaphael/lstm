---
id: toc-05-roadmap
type: roadmap
status: active
owner: principal
updated_utc: 2026-09-13T14:05:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# Roadmap and Task Board

## Milestone gates

| ID | Milestone | Exit gate | Status |
| --- | --- | --- | --- |
| M0 | Recover | Inventory, baseline, context vault, task graph | **in progress** |
| M1 | ML foundation | Causal contracts, matched eval, portable packages | **blocked** on missing v2 |
| M2 | Application slice | Import → job → forecast with persistence | planned |
| M3 | Complete product flows | Operational UI + monitoring + education | planned |
| M4 | Reliability | Auth, limits, lockfile, CI, recovery | planned |
| M5 | Delivery | RC + accurate docs + resumable context | planned |

## Active tasks

| Task ID | Title | Owner | Deps | Status | Evidence |
| --- | --- | --- | --- | --- | --- |
| M0-01 | Recover local/remote state | principal | — | **done** | [[01 Current State]] |
| M0-02 | Baseline `test_pipeline` on `a652a04` | principal | M0-01 | **done** | [[Evidence/EV-2026-09-13-pipeline]] |
| M0-03 | Create project-context vault | context steward | M0-01 | **in progress** | this vault |
| M0-04 | Import ROAD/AUDIT from readiness tip into tree | principal | M0-01 | **done** (files present) | root docs |
| M0-05 | Obtain `codex/temporal-pipeline-v2` | user + principal | M0-01 | **blocked** | [[07 Risks and Blockers]] |
| M0-06 | Resume-check vault | QA | M0-03 | pending | — |
| M1-01 | Integrate or reconstruct pipeline v2 | data/ML | M0-05 | blocked | — |
| M1-02 | Save/load parity univariate+multivariate | data/ML | M1-01 | planned | — |
| M1-03 | Legacy artifact readability | data/ML | M1-01 | planned | — |

## File ownership (initial)

| Area | Owner |
| --- | --- |
| `obsidian_vault/Traffic_Observatory_Context/**`, `HANDOFF.md` | context steward / principal |
| `src/traffic_lstm/**`, `tests/test_*pipeline*`, `tests/test_temporal*` | data/ML |
| `app/**`, future web UI | frontend |
| jobs/API/persistence (not yet created) | backend |
| CI, lockfiles, launchers | platform |
| Independent review of M1+ | QA |

Only principal integrates shared contracts and release changes. No force-push / merge main / deploy without principal + user authorization.
