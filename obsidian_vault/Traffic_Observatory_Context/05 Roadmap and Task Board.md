---
id: toc-05-roadmap
type: roadmap
status: active
owner: principal
updated_utc: 2026-09-13T14:10:00Z
verified_commit: e50455d1c749e425fe984278e9c2153080a61120
---

# Roadmap and Task Board

## Milestone gates

| ID | Milestone | Exit gate | Status |
| --- | --- | --- | --- |
| M0 | Recover | Inventory, baseline, context vault, task graph, pushed checkpoint | **done** |
| M1 | ML foundation | Causal contracts, matched eval, portable packages | **in progress** |
| M2 | Application slice | Import → job → forecast with persistence | planned; demo slice in progress |
| M3 | Complete product flows | Operational UI + monitoring + education | planned |
| M4 | Reliability | Auth, limits, lockfile, CI, recovery | planned |
| M5 | Delivery | RC + accurate docs + resumable context | planned |

## Active tasks

| Task ID | Title | Owner | Deps | Status | Evidence |
| --- | --- | --- | --- | --- | --- |
| M0-01 | Recover local/remote state | principal | — | **done** | [[01 Current State]] |
| M0-02 | Baseline `test_pipeline` on `a652a04` | principal | M0-01 | **done** | [[Evidence/EV-2026-09-13-pipeline]] |
| M0-03 | Create project-context vault | context steward | M0-01 | **done** | this vault |
| M0-04 | Import ROAD/AUDIT from readiness tip into tree | principal | M0-01 | **done** (files present) | root docs |
| M0-05 | Decide path for missing `codex/temporal-pipeline-v2` | user + principal | M0-01 | **done** (resolved by clean-room rebuild decision; original branch still lost) | [[Decisions/DEC-2026-09-13-v2-clean-room-rebuild]] |
| M0-06 | Resume-check vault | QA | M0-03 | **done** | [[Sessions/2026-09-13-m1-kickoff]] |
| PUSH-01 | Push M0 recovery branch | principal | M0-03 | **done** | [[Evidence/EV-2026-09-13-push-m0]] |
| M1-00 | Readiness integration | principal | M0 | **in progress** | `observatory/readiness-integration`; fix `TrainingConfig.to_dict()` path serialization |
| M1-01 | Pipeline v2 fresh build | data/ML | [[Decisions/DEC-2026-09-13-v2-clean-room-rebuild]] | **in progress** | `observatory/pipeline-v2`; planned temporal suite |
| M1-02 | Save/load parity univariate+multivariate | data/ML | M1-01 | planned | — |
| M1-03 | Legacy artifact readability | data/ML | M1-01 | planned | — |
| WEB-01 | Browser demo page | frontend | M1 context | **in progress** | `observatory/web-demo` |

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
