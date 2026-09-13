---
id: dec-2026-09-13-v2-clean-room-rebuild
type: decision
status: accepted
owner: principal
updated_utc: 2026-09-13T14:10:00Z
verified_commit: e50455d1c749e425fe984278e9c2153080a61120
---

# Decision: Clean-room pipeline v2 rebuild

## Problem

The previously claimed `codex/temporal-pipeline-v2` branch is lost. The named continuity commit `646d3a7` is not present locally, on GitHub, in the searched agent stores, or in available session transcript evidence.

## Evidence of loss

- `codex/temporal-pipeline-v2` is absent from local and remote branch lists.
- `646d3a7` cannot be resolved from the current repository.
- `tests/test_temporal.py` and the claimed v2 training path are not recoverable from this machine.
- Earlier v2 claims remain [[06 Verification Index|historical evidence]], not verified current code.

## Authorization

The user explicitly authorized a rebuild with the quote **"rebuild it now"** on 2026-09-13 around 14:06Z.

## Choice

Proceed with a **fresh implementation** on `observatory/pipeline-v2`, using documented requirements from `ROAD_PRODUCT_PLAN.md` critical findings and `MASTER_BUILD_PROMPT.md` section 7.

This implementation must never be described as recovery of the lost commit `646d3a7`, nor as identical to the missing branch. It is a new clean-room v2.

## Consequences

- M1 is unblocked without falsifying provenance.
- `BLOCKER-01` is resolved-by-decision, while the original branch remains recorded as lost.
- The v2 suite is planned against the new `tests/test_temporal.py` on `observatory/pipeline-v2`.
- If the original bundle ever surfaces, diff it against the fresh implementation before merging or changing historical claims.

## Rollback implications

Rollback does not restore the missing branch. If the fresh v2 path proves wrong, remove or revert the new `observatory/pipeline-v2` work and return to the documented legacy v1 baseline on `main` @ `a652a04`.
