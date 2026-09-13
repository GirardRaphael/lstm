---
id: task-m0-05
type: task
status: resolved-by-decision
owner: principal
updated_utc: 2026-09-13T14:10:00Z
verified_commit: e50455d1c749e425fe984278e9c2153080a61120
---

# M0-05 Decide path for unpublished temporal-pipeline-v2

## Acceptance

- Missing branch status recorded
- User decision recorded
- No claim of identity with missing commit `646d3a7`
- Fresh v2 work tracked separately from historical claims

## Dependencies

M0-01 done.

## File ownership

Fresh implementation proceeds on `observatory/pipeline-v2`; original branch remains read-only because it is absent.

## Checks

Run temporal + pipeline suites on the fresh branch before merge; record Evidence notes.

## Resolution

No local or GitHub copy was found. User authorized clean-room rebuild; see [[../Decisions/DEC-2026-09-13-v2-clean-room-rebuild]].

If the original bundle ever appears, diff it against the fresh implementation before changing provenance claims.
